// Nemotron-3.5 Bolt decode ops for ONNX Runtime (CUDA EP, custom-op ABI), domain "bolt".
//
//   MambaStep   : Mamba2 mixer core of R <= 16 rows (conv1d+SiLU, dt/decay, SSM state update, D skip, gated grouped RMSNorm).
//                 Replaces the ~35-node ONNX chain between in_proj and out_proj; states are updated in place friendly.
//   MoeRoute    : fp32 router (sigmoid, bias-corrected top-k, normalisation, scaling) emitting the dense top-k selection /
//                 weight matrices consumed by QMoE (router_probs / router_weights).
//   Relu2       : y = bf16(relu(x)^2)
//   Nvfp4Gemv   : y[R,N] = x[R,K] @ W^T with a modelopt NVFP4 weight (u8 [N,K/2], fp8 e4m3 block-16 scales, fp32 global scale).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "iq2xxs_grid.h"

#ifndef BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
#define BOLT_MAMBA_ROUND_CONV_BEFORE_SILU 0
#endif
#ifndef BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM
#define BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM 0
#endif

namespace {

using bf16 = __nv_bfloat16;
constexpr int kMaxRows = 16;

__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float Silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float MambaNormReadout(float v) {
#if BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM
  // Preserve the BF16 scan-output boundary before FP32 gate/variance math.
  return B2F(__float2bfloat16(v));
#else
  return v;
#endif
}

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

template <typename T>
const T* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const T*>(c.GetInput(i).GetTensorRawData());
}
template <typename T>
T* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<T*>(v.GetTensorMutableRawData());
}
int64_t Dim(const Ort::ConstValue& v, size_t i) { return v.GetTensorTypeAndShapeInfo().GetShape().at(i); }
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

// ------------------------------------------------------------------------------------------------------------------
// MambaStep
// ------------------------------------------------------------------------------------------------------------------
// proj row layout: [gate (di) | xBC (conv_dim = di + 2*ng*ds) | dt (heads)]
template <int R_FIXED = 0, bool ALL_ROWS = false>
__global__ void MambaConvKernel(const bf16* __restrict__ proj, int R_rt, int proj_stride, int di, int conv_dim,
                                const bf16* __restrict__ conv_state_in, const bf16* __restrict__ conv_w,
                                const bf16* __restrict__ conv_b, bf16* __restrict__ conv_state_out,
                                bf16* __restrict__ xbc_act, const int* __restrict__ sel, bool all_rows_rt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  const bool all_rows = R_FIXED > 0 ? ALL_ROWS : all_rows_rt;
  // Verify mode: the input state is row `sel` of a [rows, conv_dim, 3] buffer and the state after token r goes to row r.
  if (sel) conv_state_in += static_cast<int64_t>(*sel) * conv_dim * 3;
  float inp[3 + (R_FIXED > 0 ? R_FIXED : kMaxRows)];
  inp[0] = B2F(conv_state_in[c * 3 + 0]);
  inp[1] = B2F(conv_state_in[c * 3 + 1]);
  inp[2] = B2F(conv_state_in[c * 3 + 2]);
  const uint2 wv = __ldg(reinterpret_cast<const uint2*>(conv_w + c * 4));
  const bf16* wp = reinterpret_cast<const bf16*>(&wv);
  const float w0 = B2F(wp[0]), w1 = B2F(wp[1]), w2 = B2F(wp[2]), w3 = B2F(wp[3]);
  const float bias = B2F(conv_b[c]);
  if constexpr (R_FIXED > 0) {
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) {
      float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
#if BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
      acc = B2F(__float2bfloat16(acc));
#endif
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
    }
    if constexpr (ALL_ROWS) {
#pragma unroll
      for (int r = 0; r < R_FIXED; ++r) {
        bf16* o = conv_state_out + (static_cast<int64_t>(r) * conv_dim + c) * 3;
        o[0] = __float2bfloat16(inp[r + 1]);
        o[1] = __float2bfloat16(inp[r + 2]);
        o[2] = __float2bfloat16(inp[r + 3]);
      }
    } else {
      conv_state_out[c * 3 + 0] = __float2bfloat16(inp[R_FIXED]);
      conv_state_out[c * 3 + 1] = __float2bfloat16(inp[R_FIXED + 1]);
      conv_state_out[c * 3 + 2] = __float2bfloat16(inp[R_FIXED + 2]);
    }
  } else {
    for (int r = 0; r < R; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
    for (int r = 0; r < R; ++r) {
      float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
#if BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
      acc = B2F(__float2bfloat16(acc));
#endif
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
    }
    if (all_rows) {
      for (int r = 0; r < R; ++r) {
        bf16* o = conv_state_out + (static_cast<int64_t>(r) * conv_dim + c) * 3;
        o[0] = __float2bfloat16(inp[r + 1]);
        o[1] = __float2bfloat16(inp[r + 2]);
        o[2] = __float2bfloat16(inp[r + 3]);
      }
    } else {
      conv_state_out[c * 3 + 0] = __float2bfloat16(inp[R]);
      conv_state_out[c * 3 + 1] = __float2bfloat16(inp[R + 1]);
      conv_state_out[c * 3 + 2] = __float2bfloat16(inp[R + 2]);
    }
  }
}

// Block (h, pb) owns head h and p-tile pb of width PT (HD = PT * gridDim.y). State layout [heads, ds, hd] (hd contiguous), fp32.
// Thread (p_local, part) owns state[n in part][p_base + p_local]; the QP parts split d_state.
template <int DS, int HD, int QP, int PT = HD, int R_FIXED = 0, bool ALL_ROWS = false>
__global__ void MambaSsmKernel(const bf16* __restrict__ proj, int R_rt, int proj_stride, int di, int conv_dim, int heads,
                               int heads_per_group, const bf16* __restrict__ xbc_act,
                               const float* __restrict__ ssm_in, const float* __restrict__ dt_bias,
                               const float* __restrict__ a_neg, const float* __restrict__ d_skip, float tmin,
                               float* __restrict__ ssm_out, float* __restrict__ ypre, const int* __restrict__ sel,
                               bool all_rows_rt) {
  const int h = blockIdx.x;
  const int p_base = blockIdx.y * PT;
  const int g = h / heads_per_group;
  const int tid = threadIdx.x;
  const int pl = tid % PT;
  const int p = p_base + pl;
  const int half = tid / PT;  // 0 .. QP-1
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  const bool all_rows = R_FIXED > 0 ? ALL_ROWS : all_rows_rt;
  constexpr int NH = DS / QP;
  constexpr int MAX_R = R_FIXED > 0 ? R_FIXED : kMaxRows;
  __shared__ float sB[MAX_R][DS];
  __shared__ float sC[MAX_R][DS];
  __shared__ float sPart[MAX_R][QP][PT];
  __shared__ float sDt[MAX_R];
  const int b_off = di + g * DS;
  const int c_off = di + (heads / heads_per_group) * DS + g * DS;
  for (int i = tid; i < R * DS; i += blockDim.x) {
    const int r = i / DS, n = i % DS;
    sB[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + b_off + n]);
    sC[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + c_off + n]);
  }
  if (tid < R) {
    const float raw = B2F(proj[static_cast<int64_t>(tid) * proj_stride + di + conv_dim + h]) + dt_bias[h];
    float dt = raw > 20.0f ? raw : log1pf(expf(raw));
    sDt[tid] = fmaxf(dt, tmin);
  }
  __syncthreads();
  float st[NH];
  const int64_t row_stride = static_cast<int64_t>(heads) * DS * HD;
  const float* sp = ssm_in + (sel ? static_cast<int64_t>(*sel) * row_stride : 0) + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) st[n] = sp[static_cast<int64_t>(n) * HD];
  const float A = a_neg[h];
  const float D = d_skip[h];
  if constexpr (R_FIXED > 0) {
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) {
      const float dt = sDt[r];
      const float dA = expf(A * dt);
      const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
      const float dtx = dt * x;
      float acc = 0.f;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int nn = half * NH + n;
        st[n] = st[n] * dA + dtx * sB[r][nn];
        acc += sC[r][nn] * st[n];
      }
      sPart[r][half][pl] = acc;
      if constexpr (ALL_ROWS) {
        float* orow = ssm_out + r * row_stride + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
        for (int n = 0; n < NH; ++n) orow[static_cast<int64_t>(n) * HD] = st[n];
      }
    }
    if constexpr (!ALL_ROWS) {
      float* op = ssm_out + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
      for (int n = 0; n < NH; ++n) op[static_cast<int64_t>(n) * HD] = st[n];
    }
  } else {
    for (int r = 0; r < R; ++r) {
      const float dt = sDt[r];
      const float dA = expf(A * dt);
      const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
      const float dtx = dt * x;
      float acc = 0.f;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int nn = half * NH + n;
        st[n] = st[n] * dA + dtx * sB[r][nn];
        acc += sC[r][nn] * st[n];
      }
      sPart[r][half][pl] = acc;
      if (all_rows) {
        float* orow = ssm_out + r * row_stride + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
        for (int n = 0; n < NH; ++n) orow[static_cast<int64_t>(n) * HD] = st[n];
      }
    }
    if (!all_rows) {
      float* op = ssm_out + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
      for (int n = 0; n < NH; ++n) op[static_cast<int64_t>(n) * HD] = st[n];
    }
  }
  __syncthreads();
  for (int i = tid; i < R * PT; i += blockDim.x) {
    const int r = i / PT, pp = i % PT;
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p_base + pp]);
    float sum = D * x;
#pragma unroll
    for (int q = 0; q < QP; ++q) sum += sPart[r][q][pp];
    ypre[static_cast<int64_t>(r) * di + h * HD + p_base + pp] = sum;
  }
}

// grid (R, groups), block 256; gate is proj[r][0..di)
__global__ void MambaNormKernel(const bf16* __restrict__ proj, int proj_stride, int di, int group_size, float eps,
                                const float* __restrict__ ypre, const float* __restrict__ norm_w, bf16* __restrict__ y) {
  const int r = blockIdx.x, g = blockIdx.y;
  const int64_t base = static_cast<int64_t>(r) * di + static_cast<int64_t>(g) * group_size;
  const bf16* gate = proj + static_cast<int64_t>(r) * proj_stride + static_cast<int64_t>(g) * group_size;
  __shared__ float red[32];
  if (group_size == 1024 && blockDim.x == 256) {
    const int tid = threadIdx.x;
    float v[4];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      const float gt = B2F(__ldg(gate + i));
      v[k] = MambaNormReadout(__ldg(ypre + base + i)) * (gt / (1.0f + expf(-gt)));
      ss += v[k] * v[k];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
      float w = tid < 8 ? red[tid] : 0.f;
      for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffff, w, o);
      if (tid == 0) red[0] = w;
    }
    __syncthreads();
    const float rs = rsqrtf(red[0] / group_size + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      y[base + i] = __float2bfloat16(v[k] * rs * __ldg(norm_w + g * 1024 + i));
    }
    return;
  }
  float ss = 0.f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = MambaNormReadout(ypre[base + i]) * (gt / (1.0f + expf(-gt)));
    ss += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / group_size + eps);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = MambaNormReadout(ypre[base + i]) * (gt / (1.0f + expf(-gt)));
    y[base + i] = __float2bfloat16(v * rs * norm_w[g * group_size + i]);
  }
}

struct MambaStepState {
  const OrtApi& api;
  int heads, hd, ds, ng;
  float eps, tmin;
  bool verify;
  MambaStepState(const OrtApi& a, const OrtKernelInfo* info, bool verify_mode = false) : api(a), verify(verify_mode) {
    Ort::ConstKernelInfo k(info);
    heads = static_cast<int>(k.GetAttribute<int64_t>("heads"));
    hd = static_cast<int>(k.GetAttribute<int64_t>("head_dim"));
    ds = static_cast<int>(k.GetAttribute<int64_t>("d_state"));
    ng = static_cast<int>(k.GetAttribute<int64_t>("n_groups"));
    eps = k.GetAttribute<float>("epsilon");
    tmin = k.GetAttribute<float>("time_step_min");
    if (hd != 64 || ds != 128) throw std::invalid_argument("MambaStep: only head_dim=64, d_state=128 are implemented");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto proj = c.GetInput(0);
      const int R = static_cast<int>(Dim(proj, 0));
      if (R < 1 || R > kMaxRows) throw std::invalid_argument("MambaStep: rows must be in [1,16]");
      const int di = heads * hd;
      const int conv_dim = di + 2 * ng * ds;
      const int stride = di + conv_dim + heads;
      if (Dim(proj, 1) != stride) throw std::invalid_argument("MambaStep: bad proj width");
      auto y = c.GetOutput(0, std::vector<int64_t>{R, di});
      const int* sel = verify ? In<int>(c, 9) : nullptr;
      if (verify && Dim(c.GetInput(1), 0) < R) throw std::invalid_argument("MambaVerify: state buffer needs >= R rows");
      auto conv_out = c.GetOutput(1, c.GetInput(1).GetTensorTypeAndShapeInfo().GetShape());
      auto ssm_out = c.GetOutput(2, c.GetInput(2).GetTensorTypeAndShapeInfo().GetShape());
      auto xbc_act = c.GetOutput(3, std::vector<int64_t>{R, conv_dim});
      auto ypre = c.GetOutput(4, std::vector<int64_t>{R, di});
      cudaStream_t s = Stream(api, raw);
      if (R == 8 && verify) {
        MambaConvKernel<8, true><<<(conv_dim + 63) / 64, 64, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, In<bf16>(c, 1),
                                                                     In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv_out),
                                                                     Out<bf16>(xbc_act), sel, verify);
        MambaSsmKernel<128, 64, 4, 64, 8, true><<<heads, 256, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads,
                                                                      heads / ng, Out<bf16>(xbc_act), In<float>(c, 2),
                                                                      In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
                                                                      tmin, Out<float>(ssm_out), Out<float>(ypre), sel,
                                                                      verify);
      } else {
        MambaConvKernel<<<(conv_dim + 127) / 128, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, In<bf16>(c, 1),
                                                               In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv_out),
                                                               Out<bf16>(xbc_act), sel, verify);
        // QP = 4 state parts (256 threads/head) for the multi-row verify path, 2 parts (128 threads) for single-token decode
        if (R > 1) MambaSsmKernel<128, 64, 4><<<heads, 256, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads, heads / ng,
                                                       Out<bf16>(xbc_act), In<float>(c, 2), In<float>(c, 5),
                                                       In<float>(c, 6), In<float>(c, 7), tmin, Out<float>(ssm_out),
                                                       Out<float>(ypre), sel, verify);
        else MambaSsmKernel<128, 64, 2><<<heads, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads, heads / ng,
                                                       Out<bf16>(xbc_act), In<float>(c, 2), In<float>(c, 5),
                                                       In<float>(c, 6), In<float>(c, 7), tmin, Out<float>(ssm_out),
                                                       Out<float>(ypre), sel, verify);
      }
      MambaNormKernel<<<dim3(R, ng), 256, 0, s>>>(In<bf16>(c, 0), stride, di, di / ng, eps, Out<float>(ypre),
                                                  In<float>(c, 8), Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// MoeRoute
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float WarpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// grid (R, ceil(E/8)), block 256: one warp per expert logit.
__global__ void RouterLogitsKernel(const bf16* __restrict__ x, const bf16* __restrict__ w, int K, int E,
                                   float* __restrict__ scores) {
  const int r = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.y * 8 + warp;
  if (e >= E) return;
  const bf16* xr = x + static_cast<int64_t>(r) * K;
  const bf16* wr = w + static_cast<int64_t>(e) * K;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 xv = *reinterpret_cast<const uint4*>(xr + k);
    const uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += B2F(xp[i]) * B2F(wp[i]);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) scores[static_cast<int64_t>(r) * E + e] = 1.0f / (1.0f + expf(-acc));
}

// one warp per row (E <= 512): iterative warp arg-max over bias-corrected scores
__global__ void RouterTopkKernel(const float* __restrict__ scores, const float* __restrict__ bias, int E, int topk,
                                 float scale, int* __restrict__ idx_out, float* __restrict__ w_out) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x;
  float val[16], sc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + 32 * i;
    if (e < E) {
      sc[i] = scores[static_cast<int64_t>(r) * E + e];
      val[i] = sc[i] + bias[e];
    } else {
      sc[i] = 0.f;
      val[i] = -INFINITY;
    }
  }
  int my_idx = -1;
  float my_sc = 0.f;
  float sum = 0.f;
  for (int j = 0; j < topk; ++j) {
    float bv = -INFINITY, bsc = 0.f;
    int bi = -1;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (val[i] > bv) { bv = val[i]; bsc = sc[i]; bi = lane + 32 * i; }
    float wv = bv;
    int wi = bi;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, wv, o);
      const int oi = __shfl_xor_sync(0xffffffff, wi, o);
      if (ov > wv || (ov == wv && oi >= 0 && (wi < 0 || oi < wi))) { wv = ov; wi = oi; }
    }
    const int win_lane = wi & 31;
    const int win_slot = wi >> 5;
    if (wi >= 0 && win_lane == lane) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (i == win_slot) val[i] = -INFINITY;
    }
    const float s = wi >= 0 ? __shfl_sync(0xffffffff, bsc, win_lane) : 0.f;
    if (lane == j) {
      my_idx = wi;
      my_sc = s;
    }
    sum += s;
  }
  const float denom = sum + 1e-20f;
  if (lane < topk) {
    idx_out[r * topk + lane] = my_idx;
    w_out[r * topk + lane] = my_sc / denom * scale;
  }
}

struct MoeRouteState {
  const OrtApi& api;
  int topk;
  float scale;
  MoeRouteState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = static_cast<int>(k.GetAttribute<int64_t>("topk"));
    scale = k.GetAttribute<float>("scale");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      const int R = static_cast<int>(Dim(x, 0)), K = static_cast<int>(Dim(x, 1));
      const int E = static_cast<int>(Dim(c.GetInput(1), 0));
      if (E > 512 || K % 256 != 0 || topk > 32) throw std::invalid_argument("MoeRoute: unsupported shape");
      auto idx = c.GetOutput(0, std::vector<int64_t>{R, topk});
      auto wt = c.GetOutput(1, std::vector<int64_t>{R, topk});
      auto scores = c.GetOutput(2, std::vector<int64_t>{R, E});
      cudaStream_t s = Stream(api, raw);
      RouterLogitsKernel<<<dim3(R, (E + 7) / 8), 256, 0, s>>>(In<bf16>(c, 0), In<bf16>(c, 1), K, E, Out<float>(scores));
      RouterTopkKernel<<<R, 32, 0, s>>>(Out<float>(scores), In<float>(c, 2), E, topk, scale, Out<int>(idx),
                                        Out<float>(wt));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// MoeDecode: routed experts (int4 block-32, zero point 8, nibble-low-first) for R*topk (token, expert) pairs.
//   FC1: h = bf16(relu2(bf16(z @ W1[e]^T)));  FC2: p = bf16(h @ W2[e]^T);  out = bf16(sum_k w_k * p_k)
// ------------------------------------------------------------------------------------------------------------------

// dot of 32 int4 (one 16-byte chunk) with x[32] floats
__device__ __forceinline__ float DotInt4x32(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      const float lo = __uint_as_float(0x4B000000u | (byte & 0xFu)) - 8388608.0f - 8.0f;
      const float hi = __uint_as_float(0x4B000000u | (byte >> 4)) - 8388608.0f - 8.0f;
      acc += lo * x[q * 8 + b * 2] + hi * x[q * 8 + b * 2 + 1];
    }
  }
  return acc;
}

// grid (R*topk, I/8), block 256; K1 must be 1024 (one 16B chunk per lane)
__global__ void MoeFc1Kernel(const bf16* __restrict__ z, const int* __restrict__ idx, int topk, int I,
                             const uint8_t* __restrict__ w1, const bf16* __restrict__ s1, bf16* __restrict__ h) {
  const int pair = blockIdx.x;
  const int t = pair / topk;
  const int e = idx[pair];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.y * 8 + warp;
  if (j >= I) return;
  float x[32];
  const uint4* zp = reinterpret_cast<const uint4*>(z + static_cast<int64_t>(t) * 1024 + lane * 32);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 v = zp[q];
    const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[q * 8 + i] = B2F(b[i]);
  }
  const int64_t row = static_cast<int64_t>(e) * I + j;
  const uint4 wv = *reinterpret_cast<const uint4*>(w1 + row * 512 + lane * 16);
  const float sc = B2F(s1[row * 32 + lane]);
  const float tot = WarpSum(DotInt4x32(wv, x) * sc);
  if (lane == 0) {
    const float r = fmaxf(B2F(__float2bfloat16(tot)), 0.f);
    h[static_cast<int64_t>(pair) * I + j] = __float2bfloat16(r * r);
  }
}

// grid (R*topk, 1024/8), block 256; K2 = I (multiple of 32)
__global__ void MoeFc2Kernel(const bf16* __restrict__ h, const int* __restrict__ idx, int I,
                             const uint8_t* __restrict__ w2, const bf16* __restrict__ s2, float* __restrict__ part) {
  const int pair = blockIdx.x;
  const int e = idx[pair];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.y * 8 + warp;
  const int nchunks = I / 32;
  const int64_t row = static_cast<int64_t>(e) * 1024 + n;
  const bf16* hp = h + static_cast<int64_t>(pair) * I;
  float acc = 0.f;
  for (int c = lane; c < nchunks; c += 32) {
    float x[32];
    const uint4* xp = reinterpret_cast<const uint4*>(hp + c * 32);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 v = xp[q];
      const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[q * 8 + i] = B2F(b[i]);
    }
    const uint4 wv = *reinterpret_cast<const uint4*>(w2 + row * (I / 2) + c * 16);
    acc += DotInt4x32(wv, x) * B2F(s2[row * nchunks + c]);
  }
  acc = WarpSum(acc);
  if (lane == 0) part[static_cast<int64_t>(pair) * 1024 + n] = B2F(__float2bfloat16(acc));
}

__global__ void MoeFinalizeKernel(const float* __restrict__ part, const float* __restrict__ w, int topk,
                                  bf16* __restrict__ out) {
  const int t = blockIdx.x;
  const int vec_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (vec_idx >= 256) return;
  const float* wt = w + t * topk;
  const float4* pt = reinterpret_cast<const float4*>(part + static_cast<int64_t>(t) * topk * 1024) + vec_idx;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  if (topk == 9) {
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float wk = wt[k];
      const float4 pv = __ldg(pt + k * 256);
      a0 += wk * pv.x;
      a1 += wk * pv.y;
      a2 += wk * pv.z;
      a3 += wk * pv.w;
    }
  } else {
    for (int k = 0; k < topk; ++k) {
      const float wk = wt[k];
      const float4 pv = __ldg(pt + k * 256);
      a0 += wk * pv.x;
      a1 += wk * pv.y;
      a2 += wk * pv.z;
      a3 += wk * pv.w;
    }
  }
  uint2 out2;
  bf16* op = reinterpret_cast<bf16*>(&out2);
  op[0] = __float2bfloat16(a0);
  op[1] = __float2bfloat16(a1);
  op[2] = __float2bfloat16(a2);
  op[3] = __float2bfloat16(a3);
  reinterpret_cast<uint2*>(out + static_cast<int64_t>(t) * 1024)[vec_idx] = out2;
}

struct MoeDecodeState {
  const OrtApi& api;
  int topk;
  MoeDecodeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = static_cast<int>(k.GetAttribute<int64_t>("topk"));
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto z = c.GetInput(0);
      const int R = static_cast<int>(Dim(z, 0));
      if (Dim(z, 1) != 1024) throw std::invalid_argument("MoeDecode: latent size must be 1024");
      const int I = static_cast<int>(Dim(c.GetInput(3), 1));  // fc1_w [E, I, 512]
      if (I % 32 != 0 || I % 8 != 0) throw std::invalid_argument("MoeDecode: bad inter size");
      auto out = c.GetOutput(0, std::vector<int64_t>{R, 1024});
      auto h = c.GetOutput(1, std::vector<int64_t>{static_cast<int64_t>(R) * topk, I});
      auto part = c.GetOutput(2, std::vector<int64_t>{static_cast<int64_t>(R) * topk, 1024});
      cudaStream_t s = Stream(api, raw);
      MoeFc1Kernel<<<dim3(R * topk, I / 8), 256, 0, s>>>(In<bf16>(c, 0), In<int>(c, 1), topk, I, In<uint8_t>(c, 3),
                                                         In<bf16>(c, 4), Out<bf16>(h));
      MoeFc2Kernel<<<dim3(R * topk, 1024 / 8), 256, 0, s>>>(Out<bf16>(h), In<int>(c, 1), I, In<uint8_t>(c, 5),
                                                           In<bf16>(c, 6), Out<float>(part));
      MoeFinalizeKernel<<<dim3(R, 4), 64, 0, s>>>(Out<float>(part), In<float>(c, 2), topk, Out<bf16>(out));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// Relu2
// ------------------------------------------------------------------------------------------------------------------
__global__ void Relu2Kernel(const bf16* x, bf16* y, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float r = fmaxf(B2F(x[i]), 0.f);
  y[i] = __float2bfloat16(r * r);
}
struct Relu2State {
  const OrtApi& api;
  Relu2State(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto out = c.GetOutput(0, x.GetTensorTypeAndShapeInfo().GetShape());
      const int64_t n = Count(x);
      Relu2Kernel<<<(n + 255) / 256, 256, 0, Stream(api, raw)>>>(In<bf16>(c, 0), Out<bf16>(out), n);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// Nvfp4Gemv: a warp owns ROWS weight rows; lanes stride over 16-byte (32-value) chunks of K.
// ------------------------------------------------------------------------------------------------------------------
// e2m1 nibble pairs -> fp16x2 (value * 2^-14); the caller rescales by 16384 (exact).
__device__ __forceinline__ uint32_t Fp4PairPattern(uint32_t s) {
  return ((s & 0x00070007u) << 9) | ((s & 0x00080008u) << 12);
}
__device__ __forceinline__ float Fp8E4M3ToFloat(uint32_t byte) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3)));
}

// x is pre-scaled by 2^-6 so that fp16 partial sums of 16 products cannot overflow for |x| < ~4e4 (massive activations).
constexpr float kXScale = 1.0f / 64.0f;

// A warp owns ROWS weight rows; lane l handles 16-byte (32-value) chunks l + 32*it. All weight loads are issued up front.
template <int R, int ROWS, int IT>
__global__ void Nvfp4GemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w,
                                const uint8_t* __restrict__ ws, const float* __restrict__ gscale, int N, int K,
                                bf16* __restrict__ y) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int kbytes = K / 2;
  const int kscale = K / 16;
  uint4 wv[ROWS][IT];
  uint32_t sc[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int cb = it * 512 + lane * 16;
      wv[i][it] = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * kbytes + cb);
      sc[i][it] = *reinterpret_cast<const uint16_t*>(ws + static_cast<int64_t>(n) * kscale + (cb >> 3));
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int k0 = (it * 512 + lane * 16) * 2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      __half2 xp[16];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + k0);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint4 v = xq[g];
        const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int p = 0; p < 4; ++p) xp[g * 4 + p] = __floats2half2_rn(B2F(b[p]) * kXScale, B2F(b[p + 4]) * kXScale);
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        const uint32_t words[4] = {wv[i][it].x, wv[i][it].y, wv[i][it].z, wv[i][it].w};
        float tot = 0.f;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          __half2 a = __float2half2_rn(0.f);
#pragma unroll
          for (int wd = 0; wd < 2; ++wd) {
            const uint32_t word = words[blk * 2 + wd];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const uint32_t pb = Fp4PairPattern(word >> (4 * p));
              __half2 pat = *reinterpret_cast<const __half2*>(&pb);
              pat = __hmul2(pat, __float2half2_rn(16384.f));
              a = __hfma2(pat, xp[(blk * 2 + wd) * 4 + p], a);
            }
          }
          const float f = (__low2float(a) + __high2float(a)) * (1.0f / kXScale);
          tot += f * Fp8E4M3ToFloat((sc[i][it] >> (8 * blk)) & 0xFFu);
        }
        acc[i][r] += tot;
      }
    }
  }
  const float g = gscale[0];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v * g);
    }
}

template <int R, int IT>
void LaunchNvfp4Impl(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                     cudaStream_t s) {
  if (N <= 8192) {
    constexpr int ROWS = 1, WARPS = 4;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  }
}
template <int R>
void LaunchNvfp4(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                 cudaStream_t s) {
  if (K == 4096) LaunchNvfp4Impl<R, 4>(x, w, ws, g, N, K, y, s);
  else if (K == 8192) LaunchNvfp4Impl<R, 8>(x, w, ws, g, N, K, y, s);
  else throw std::invalid_argument("Nvfp4Gemv: K must be 4096 or 8192");
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor-core skinny GEMM (2 <= R <= 8 rows): y[R,N] = x[R,K] @ W^T with mma.m16n8k16 (bf16 in, fp32 accumulate).
// ------------------------------------------------------------------------------------------------------------------
// A = 16 weight rows, B = the (<= 8) activation rows. A warp walks 64-wide K chunks; thread (g = lane/4, t = lane%4) owns the
// 16 consecutive k of weight rows g and g+8 at k = 64c + 16t (one 16-byte load for int8, 8 bytes for int4 / fp4), so the
// weights need no repacking. mma sums over k, so any k permutation shared by A and B is valid: MMA step s uses
// k-slots (s, s+4) and (8+s, 12+s) of the thread's 16 values (adjacent nibbles/bytes need no shuffling, only x is paired up).
// kFmtIq2: exact IQ2_XXS (ggml) weights as 4-bit LUT codes: code = mag_idx + 4 * (negative), value = +-{8, 25, 43} * block scale.
// Same packing and scale layout as kFmtQ4, but the 16 values of a thread are consumed as 4 adjacent groups of 4 (one prmt LUT
// lookup per 4 weights) instead of the (s, s+4) k-slot pairing.
//
// kFmtIq2N: the ggml IQ2_XXS data, repacked 8-byte aligned: B = [N, K/32, 8] bytes (per 32 k: 4 grid-index bytes and one u32 with
// four 7-bit sign indices + a 4-bit scale, exactly the ggml records) and D = [N, K/256] fp16 super-block scales (the raw fp16 bits,
// carried in the op's bf16-typed scale input). 2.06 bits per weight. A thread's 16 k are two grid vectors; the decoded weights
// are bit-identical to the kFmtIq2 path (bf16(d * (0.5 + s) * 0.25) as the block scale).
enum { kFmtQ8 = 0, kFmtQ4 = 1, kFmtFp4 = 2, kFmtIq2 = 3, kFmtIq2N = 4 };

__device__ __forceinline__ void Mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t PackBf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ __nv_bfloat162 AsBf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
__device__ __forceinline__ uint32_t AsU32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

// (byte `s` of w) as the integer value in a float: 2^23 + byte
__device__ __forceinline__ float ByteToMagic(uint32_t w, int s) {
  return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440u | s));
}

// IQ2_XXS decode tables in shared memory: 256 x 8 bf16 magnitudes and 256 x 8 bf16 sign masks (index = 8-bit sign pattern).
struct Iq2Tables {
  uint4 grid[256];
  uint4 sign[256];
};
__device__ __forceinline__ Iq2Tables& Iq2Smem() {
  __shared__ Iq2Tables tables;
  return tables;
}
__device__ __forceinline__ void Iq2TablesInit() {
  Iq2Tables& t = Iq2Smem();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    t.grid[i] = make_uint4(kIq2xxsGridBf16[i][0], kIq2xxsGridBf16[i][1], kIq2xxsGridBf16[i][2], kIq2xxsGridBf16[i][3]);
    // bf16 sign-bit masks of the 8 elements selected by the 8-bit sign pattern i
    t.sign[i] = make_uint4(((i & 1u) << 15) | (((i >> 1) & 1u) << 31), (((i >> 2) & 1u) << 15) | (((i >> 3) & 1u) << 31),
                           (((i >> 4) & 1u) << 15) | (((i >> 5) & 1u) << 31), (((i >> 6) & 1u) << 15) | (((i >> 7) & 1u) << 31));
  }
  __syncthreads();
}

// WIDE (int4 / fp4 only): a thread loads 16 bytes = 32 k of a row per chunk (chunk = 128 k, 64 contiguous bytes per row for
// the 4 threads t) instead of 8 bytes; the two 16-k halves are consumed as two MMA groups.
template <int FMT, bool WIDE>
__device__ __forceinline__ void LoadWeightRow(const uint8_t* w, const void* sc, int64_t row, int K, int c, int t, uint4& raw,
                                              float* scale) {
  if constexpr (FMT == kFmtIq2N) {
    // chunk c = 64 k = two 32-k groups of 8-byte records; thread t owns half (t & 1) of group 2c + (t >> 1)
    const uint2 rec = __ldg(reinterpret_cast<const uint2*>(w + (row * (K / 32) + 2 * c + (t >> 1)) * 8));
    const int h = t & 1;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(sc) + row * (K / 256) + (c >> 2))));
    const uint32_t sg = rec.y;
    const uint32_t s0 = (sg >> (14 * h)) & 127u, s1 = (sg >> (14 * h + 7)) & 127u;
    const uint32_t sa = s0 | ((__popc(s0) & 1) << 7), sb = s1 | ((__popc(s1) & 1) << 7);
    raw = make_uint4((rec.x >> (16 * h)) & 0xFFu, (rec.x >> (16 * h + 8)) & 0xFFu, sa, sb);
    scale[0] = d * (0.5f + static_cast<float>(sg >> 28)) * 0.25f;
  } else if constexpr (FMT == kFmtQ8) {
    raw = __ldg(reinterpret_cast<const uint4*>(w + row * K + 64 * c + 16 * t));
    scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[row * (K / 32) + 2 * c + (t >> 1)]);
  } else if constexpr (!WIDE) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(w + row * (K / 2) + 32 * c + 8 * t));
    raw = make_uint4(v.x, v.y, 0u, 0u);
    if constexpr (FMT == kFmtQ4 || FMT == kFmtIq2) scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[row * (K / 32) + 2 * c + (t >> 1)]);
    else scale[0] = Fp8E4M3ToFloat(reinterpret_cast<const uint8_t*>(sc)[row * (K / 16) + 4 * c + t]) * 0x1p110f;
  } else {
    raw = __ldg(reinterpret_cast<const uint4*>(w + row * (K / 2) + 64 * c + 16 * t));
    if constexpr (FMT == kFmtQ4 || FMT == kFmtIq2) {
      scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[row * (K / 32) + 4 * c + t]);
    } else {
      const uint16_t two = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(sc) + row * (K / 16) + 8 * c + 2 * t);
      scale[0] = Fp8E4M3ToFloat(two & 0xFFu) * 0x1p110f;
      scale[1] = Fp8E4M3ToFloat(two >> 8) * 0x1p110f;
    }
  }
}

template <int FMT, bool WIDE, int MT>
struct StageT {
  static constexpr int NH = WIDE ? 2 : 1;  // 16-k halves per thread and chunk
  static constexpr int NSC = (FMT == kFmtFp4 && WIDE) ? 2 : 1;
  uint4 raw[MT][2];
  float sc[MT][2][NSC];
  uint4 x[2 * NH];
};

// Dequantised bf16x2 pairs for MMA step s: lo[s] = k-slots (s, s+4), hi[s] = (8+s, 12+s).
template <int FMT>
__device__ __forceinline__ void DequantRow(const uint4& raw, float scale, uint32_t* lo, uint32_t* hi) {
  if constexpr (FMT == kFmtIq2N) {
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
    const uint4 ga = __ldg(reinterpret_cast<const uint4*>(kIq2xxsGridBf16[raw.x])), gb = __ldg(reinterpret_cast<const uint4*>(kIq2xxsGridBf16[raw.y]));
    const uint32_t sa = raw.z, sb = raw.w;
    // bf16 sign-bit masks: element j of the 8 takes bit j of the sign pattern
    // (x * (2^(15-2k) + 2^(30-2k))) puts bit 2k of x at bit 15 and bit 2k+1 at bit 31 without a carry between the two products
    auto mask = [](uint32_t sgn, int pair) { return (sgn * ((1u << (15 - 2 * pair)) + (1u << (30 - 2 * pair)))) & 0x80008000u; };
    lo[0] = AsU32(__hmul2(AsBf162(ga.x ^ mask(sa, 0)), sc2));
    hi[0] = AsU32(__hmul2(AsBf162(ga.y ^ mask(sa, 1)), sc2));
    lo[1] = AsU32(__hmul2(AsBf162(ga.z ^ mask(sa, 2)), sc2));
    hi[1] = AsU32(__hmul2(AsBf162(ga.w ^ mask(sa, 3)), sc2));
    lo[2] = AsU32(__hmul2(AsBf162(gb.x ^ mask(sb, 0)), sc2));
    hi[2] = AsU32(__hmul2(AsBf162(gb.y ^ mask(sb, 1)), sc2));
    lo[3] = AsU32(__hmul2(AsBf162(gb.z ^ mask(sb, 2)), sc2));
    hi[3] = AsU32(__hmul2(AsBf162(gb.w ^ mask(sb, 3)), sc2));
  } else if constexpr (FMT == kFmtQ8) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      lo[s] = PackBf16((ByteToMagic(raw.x, s) - 8388736.0f) * scale, (ByteToMagic(raw.y, s) - 8388736.0f) * scale);
      hi[s] = PackBf16((ByteToMagic(raw.z, s) - 8388736.0f) * scale, (ByteToMagic(raw.w, s) - 8388736.0f) * scale);
    }
  } else if constexpr (FMT == kFmtIq2) {
    // LUT bytes indexed by code 0..7: bf16 of +8, +25, +43, -, -8, -25, -43, - (high / low byte planes)
    constexpr uint32_t kHiA = 0x00424141u, kHiB = 0x00C2C1C1u, kLoA = 0x002CC800u, kLoB = 0x002CC800u;
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w = (s < 2) ? raw.x : raw.y;
      const uint32_t sel = (s & 1) ? (w >> 16) : (w & 0xFFFFu);
      const uint32_t H = __byte_perm(kHiA, kHiB, sel);
      const uint32_t L = __byte_perm(kLoA, kLoB, sel);
      lo[s] = AsU32(__hmul2(AsBf162(__byte_perm(L, H, 0x5140)), sc2));
      hi[s] = AsU32(__hmul2(AsBf162(__byte_perm(L, H, 0x7362)), sc2));
    }
  } else if constexpr (FMT == kFmtQ4) {
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
    const __nv_bfloat162 zp = __floats2bfloat162_rn(136.f, 136.f);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      // nibble s and s+4 of a word -> bf16 128 + q (exact), then (q - 8) * scale
      const uint32_t a = ((raw.x >> (4 * s)) & 0x000F000Fu) | 0x43004300u;
      const uint32_t b = ((raw.y >> (4 * s)) & 0x000F000Fu) | 0x43004300u;
      lo[s] = AsU32(__hmul2(__hsub2(AsBf162(a), zp), sc2));
      hi[s] = AsU32(__hmul2(__hsub2(AsBf162(b), zp), sc2));
    }
  } else {
    // e2m1 -> bf16 pattern worth value * 2^-126 (subnormal for |v| = 0.5); scale is pre-multiplied by 2^110 and epilogue by 2^16.
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t pa = raw.x >> (4 * s), pb = raw.y >> (4 * s);
      const uint32_t a = ((pa << 6) & 0x01C001C0u) | ((pa << 12) & 0x80008000u);
      const uint32_t b = ((pb << 6) & 0x01C001C0u) | ((pb << 12) & 0x80008000u);
      lo[s] = AsU32(__hmul2(AsBf162(a), sc2));
      hi[s] = AsU32(__hmul2(AsBf162(b), sc2));
    }
  }
}

// Loads the x fragments of one chunk: 16 (or 32) contiguous k of activation row g for this thread.
template <int NH>
__device__ __forceinline__ void LoadX(const bf16* xr, bool valid, int c, int t, uint4* xo) {
  const uint4* xp = reinterpret_cast<const uint4*>(xr + 64 * NH * c + 16 * NH * t);
#pragma unroll
  for (int i = 0; i < 2 * NH; ++i) xo[i] = valid ? __ldg(xp + i) : make_uint4(0, 0, 0, 0);
}

template <int FMT, bool WIDE, int MT>
__device__ __forceinline__ void ComputeStage(const StageT<FMT, WIDE, MT>& st, float (*acc)[4]) {
#pragma unroll
  for (int h = 0; h < StageT<FMT, WIDE, MT>::NH; ++h) {
    const uint4 xa = st.x[2 * h], xb = st.x[2 * h + 1];
    uint32_t bl[4], bh[4];
    if constexpr (FMT == kFmtIq2 || FMT == kFmtIq2N) {  // adjacent k pairs: MMA s consumes x[4s .. 4s+3]
      bl[0] = xa.x; bh[0] = xa.y; bl[1] = xa.z; bh[1] = xa.w;
      bl[2] = xb.x; bh[2] = xb.y; bl[3] = xb.z; bh[3] = xb.w;
    } else {
      bl[0] = __byte_perm(xa.x, xa.z, 0x5410);
      bl[1] = __byte_perm(xa.x, xa.z, 0x7632);
      bl[2] = __byte_perm(xa.y, xa.w, 0x5410);
      bl[3] = __byte_perm(xa.y, xa.w, 0x7632);
      bh[0] = __byte_perm(xb.x, xb.z, 0x5410);
      bh[1] = __byte_perm(xb.x, xb.z, 0x7632);
      bh[2] = __byte_perm(xb.y, xb.w, 0x5410);
      bh[3] = __byte_perm(xb.y, xb.w, 0x7632);
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      uint32_t lo0[4], hi0[4], lo1[4], hi1[4];
      uint4 r0 = st.raw[m][0], r1 = st.raw[m][1];
      if constexpr (WIDE) {
        if (h == 1) {
          r0 = make_uint4(r0.z, r0.w, 0u, 0u);
          r1 = make_uint4(r1.z, r1.w, 0u, 0u);
        }
      }
      constexpr int sh = (FMT == kFmtFp4 && WIDE) ? 1 : 0;
      DequantRow<FMT>(r0, st.sc[m][0][h * sh], lo0, hi0);
      DequantRow<FMT>(r1, st.sc[m][1][h * sh], lo1, hi1);
#pragma unroll
      for (int s = 0; s < 4; ++s) Mma16816(acc[m], lo0[s], lo1[s], hi0[s], hi1[s], bl[s], bh[s]);
    }
  }
}

// Block-wide reduction of the per-warp K partials (red: [KW][MT*16][8] floats) and the epilogue callback f(row_local, col, value).
template <int MT, int KW, typename F>
__device__ __forceinline__ void ReduceEpilogue(float (*acc)[4], int warp, int g, int t, F&& f) {
  extern __shared__ float red[];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float* r0 = red + (warp * MT * 16 + m * 16 + g) * 8 + 2 * t;
    r0[0] = acc[m][0];
    r0[1] = acc[m][1];
    r0[8 * 8] = acc[m][2];
    r0[8 * 8 + 1] = acc[m][3];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < MT * 16 * 8; i += KW * 32) {
    const int rl = i >> 3, col = i & 7;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) v += red[(k * MT * 16 + rl) * 8 + col];
    f(rl, col, v);
  }
}

// Main loop shared by the dense and the MoE kernels: UNR chunks per warp in flight, next batch issued before the current one
// is consumed. load(c, stage) / the compute step are chunk-indexed.
template <int FMT, bool WIDE, int MT, int KW, int UNR, typename L>
__device__ __forceinline__ void MmaMainLoop(int warp, int nchunks, float (*acc)[4], L&& load) {
  using Stage = StageT<FMT, WIDE, MT>;
  Stage cur[UNR], nxt[UNR];
  int c0 = warp;
#pragma unroll
  for (int j = 0; j < UNR; ++j)
    if (c0 + j * KW < nchunks) load(c0 + j * KW, cur[j]);
  for (; c0 < nchunks; c0 += KW * UNR) {
#pragma unroll
    for (int j = 0; j < UNR; ++j)
      if (c0 + (UNR + j) * KW < nchunks) load(c0 + (UNR + j) * KW, nxt[j]);
#pragma unroll
    for (int j = 0; j < UNR; ++j)
      if (c0 + j * KW < nchunks) ComputeStage<FMT, WIDE, MT>(cur[j], acc);
#pragma unroll
    for (int j = 0; j < UNR; ++j) cur[j] = nxt[j];
  }
}

template <int FMT, bool WIDE, int MT, int KW, int UNR>
__global__ void __launch_bounds__(KW * 32)
    MmaGemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w, const void* __restrict__ sc,
                  const float* __restrict__ gscale, int R, int N, int K, bf16* __restrict__ y) {
  constexpr int NH = WIDE ? 2 : 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nchunks = K / (64 * NH);
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * (MT * 16);
  const bf16* xr = x + static_cast<int64_t>(min(g, R - 1)) * K;
  const bool xvalid = g < R;
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0.f;
  MmaMainLoop<FMT, WIDE, MT, KW, UNR>(warp, nchunks, acc, [&](int c, StageT<FMT, WIDE, MT>& st) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = min(n_base + m * 16 + g + 8 * h, static_cast<int64_t>(N - 1));
        LoadWeightRow<FMT, WIDE>(w, sc, row, K, c, t, st.raw[m][h], st.sc[m][h]);
      }
    LoadX<NH>(xr, xvalid, c, t, st.x);
  });
  const float gs = (gscale ? gscale[0] : 1.0f) * (FMT == kFmtFp4 ? 0x1p16f : 1.0f);
  ReduceEpilogue<MT, KW>(acc, warp, g, t, [&](int rl, int col, float v) {
    if (col < R && n_base + rl < N) y[static_cast<int64_t>(col) * N + n_base + rl] = __float2bfloat16(v * gs);
  });
}

template <int FMT, bool WIDE, int MT, int KW, int UNR = 1>
void LaunchMmaGemvImpl(const bf16* x, const uint8_t* w, const void* sc, const float* gs, int R, int N, int K, bf16* y,
                       cudaStream_t s) {
  const int blocks = (N + MT * 16 - 1) / (MT * 16);
  MmaGemvKernel<FMT, WIDE, MT, KW, UNR><<<blocks, KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(x, w, sc, gs, R, N, K, y);
}

template <int FMT>
constexpr bool kCanWide = FMT != kFmtQ8 && FMT != kFmtIq2N;

template <int FMT>
void LaunchMmaGemv(const bf16* x, const uint8_t* w, const void* sc, const float* gs, int R, int N, int K, bf16* y,
                   cudaStream_t s) {
  static const int forced = getenv("BOLT_GEMV_CFG") ? atoi(getenv("BOLT_GEMV_CFG")) : 0;  // wide*10000 + MT*1000 + KW*10 + UNR
  const bool can_wide = kCanWide<FMT> && K % 128 == 0;
  bool wide = can_wide && (forced == 0 ? true : forced >= 10000);
  switch (forced % 10000) {
#define BOLT_CFG(MT_, KW_, UNR_)                                                                                    \
  case MT_ * 1000 + KW_ * 10 + UNR_:                                                                              \
    if (wide) LaunchMmaGemvImpl<FMT, kCanWide<FMT>, MT_, KW_, UNR_>(x, w, sc, gs, R, N, K, y, s);                  \
    else LaunchMmaGemvImpl<FMT, false, MT_, KW_, UNR_>(x, w, sc, gs, R, N, K, y, s);                               \
    return;
    BOLT_CFG(1, 4, 1) BOLT_CFG(1, 8, 1) BOLT_CFG(1, 16, 1) BOLT_CFG(1, 8, 2) BOLT_CFG(2, 4, 1) BOLT_CFG(2, 8, 1) BOLT_CFG(2, 4, 2)
#undef BOLT_CFG
    default: break;
  }
  if (N >= 8192) {
    if (wide) LaunchMmaGemvImpl<FMT, kCanWide<FMT>, 2, 4>(x, w, sc, gs, R, N, K, y, s);
    else LaunchMmaGemvImpl<FMT, false, 2, 4>(x, w, sc, gs, R, N, K, y, s);
  } else {
    if (wide) LaunchMmaGemvImpl<FMT, kCanWide<FMT>, 1, 16>(x, w, sc, gs, R, N, K, y, s);
    else LaunchMmaGemvImpl<FMT, false, 1, 16>(x, w, sc, gs, R, N, K, y, s);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// MoeVerify: MoeDecode for R <= 8 rows with experts shared across rows. The (token, expert) pairs are grouped by expert; every
// unique expert's int4 weights are read once and applied to all rows with the tensor-core GEMM above (the 8 columns of the
// MMA are the rows, so rows that did not select the expert simply compute an unused column).
// ------------------------------------------------------------------------------------------------------------------
// meta (int32): [0] = #unique experts U', [1 .. P] = unique expert ids, [1+P + u*8 + row] = pair index (row * topk + k) or -1
__global__ void MoeGroupKernel(const int* __restrict__ idx, int P, int topk, int* __restrict__ meta) {
  __shared__ int sidx[128], srank[128], warp_tot[4];
  const int p = threadIdx.x, lane = p & 31, warp = p >> 5;
#pragma unroll 5
  for (int i = p; i < P * 8; i += blockDim.x) meta[1 + P + i] = -1;
  if (p < P) sidx[p] = idx[p];
  __syncthreads();
  int q = -1;
  if (p < P) {
    const int my = sidx[p];
    for (int j = 0; j <= p; ++j)
      if (sidx[j] == my) { q = j; break; }
  }
  const bool first = p < P && q == p;
  const unsigned b = __ballot_sync(0xffffffffu, first);
  if (lane == 0) warp_tot[warp] = __popc(b);
  __syncthreads();
  int off = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w)
    if (w < warp) off += warp_tot[w];
  if (first) {
    srank[p] = off + __popc(b & ((1u << lane) - 1u));
    meta[1 + srank[p]] = sidx[p];
  }
  if (p == 0) meta[0] = warp_tot[0] + warp_tot[1] + warp_tot[2] + warp_tot[3];
  __syncthreads();
  if (p < P) meta[1 + P + srank[q] * 8 + p / topk] = p;
}

// PH 1: h[pair, j] = relu2(bf16(sum_k z[row, k] * W1[e, j, k])); PH 2: part[pair, n] = bf16(sum_k h[pair, k] * W2[e, n, k]).
template <int FMT, int PH, bool WIDE, int MT, int KW>
__global__ void __launch_bounds__(KW * 32)
    MoeMmaKernel(const bf16* __restrict__ xin, const int* __restrict__ meta, int P, int R, int Nrows, int K,
                 const uint8_t* __restrict__ w, const bf16* __restrict__ sc, void* __restrict__ out) {
  constexpr int NH = WIDE ? 2 : 1;
  const int u = blockIdx.y;
  if (u >= meta[0]) return;
  const int64_t e = meta[1 + u];
  const int* pair_of = meta + 1 + P + u * 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nchunks = K / (64 * NH);
  const int n_base = blockIdx.x * (MT * 16);
  const int xpair = pair_of[g];
  const bf16* xr = xin;
  bool xvalid;
  if constexpr (PH == 1) {
    xvalid = g < R;
    xr += static_cast<int64_t>(min(g, R - 1)) * K;
  } else {
    xvalid = xpair >= 0;
    xr += static_cast<int64_t>(max(xpair, 0)) * K;
  }
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0.f;
  MmaMainLoop<FMT, WIDE, MT, KW, 1>(warp, nchunks, acc, [&](int c, StageT<FMT, WIDE, MT>& st) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = e * Nrows + min(n_base + m * 16 + g + 8 * h, Nrows - 1);
        LoadWeightRow<FMT, WIDE>(w, sc, row, K, c, t, st.raw[m][h], st.sc[m][h]);
      }
    LoadX<NH>(xr, xvalid, c, t, st.x);
  });
  constexpr float gs = FMT == kFmtFp4 ? 0x1p16f : 1.0f;
  ReduceEpilogue<MT, KW>(acc, warp, g, t, [&](int rl, int col, float v) {
    const int j = n_base + rl;
    if (j >= Nrows) return;
    const int pair = pair_of[col];
    if (pair < 0) return;
    v = B2F(__float2bfloat16(v * gs));
    if constexpr (PH == 1) {
      const float r = fmaxf(v, 0.f);
      reinterpret_cast<bf16*>(out)[static_cast<int64_t>(pair) * Nrows + j] = __float2bfloat16(r * r);
    } else {
      reinterpret_cast<float*>(out)[static_cast<int64_t>(pair) * Nrows + j] = v;
    }
  });
}

// Optional integer attribute (custom-op attributes are not declared, so absence has to be probed).
int OptAttr(const OrtKernelInfo* info, const char* name, int def) {
  try {
    return static_cast<int>(Ort::ConstKernelInfo(info).GetAttribute<int64_t>(name));
  } catch (...) {
    return def;
  }
}

struct MoeVerifyState {
  const OrtApi& api;
  int topk;
  int iq2;  // 1: exact IQ2_XXS as 4-bit LUT codes (kFmtIq2); 2: the native ggml IQ2_XXS blocks (kFmtIq2N); 0: symmetric int4
  MoeVerifyState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = static_cast<int>(k.GetAttribute<int64_t>("topk"));
    iq2 = OptAttr(info, "iq2", 0);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto z = c.GetInput(0);
      const int R = static_cast<int>(Dim(z, 0));
      const int P = R * topk;
      if (R < 1 || R > 8 || P > 128) throw std::invalid_argument("MoeVerify: rows must be in [1,8] and rows*topk <= 128");
      if (Dim(z, 1) != 1024) throw std::invalid_argument("MoeVerify: latent size must be 1024");
      const int I = static_cast<int>(Dim(c.GetInput(3), 1));  // fc1_w [E, I, 512]
      if (I % 64 != 0) throw std::invalid_argument("MoeVerify: inter size must be a multiple of 64");
      auto out = c.GetOutput(0, std::vector<int64_t>{R, 1024});
      auto h = c.GetOutput(1, std::vector<int64_t>{P, I});
      auto part = c.GetOutput(2, std::vector<int64_t>{P, 1024});
      auto meta = c.GetOutput(3, std::vector<int64_t>{1 + P + P * 8});
      cudaStream_t s = Stream(api, raw);
      int* mp = Out<int>(meta);
      MoeGroupKernel<<<1, 128, 0, s>>>(In<int>(c, 1), P, topk, mp);
      static const int cfg = getenv("BOLT_MOE_CFG") ? atoi(getenv("BOLT_MOE_CFG")) : 0;  // 1: narrow chunks
      const bool wide2 = cfg == 0 && I % 128 == 0 && iq2 != 2;
      using C1 = std::integral_constant<int, 1>;
      using C2 = std::integral_constant<int, 2>;
      using C4 = std::integral_constant<int, 4>;
      using C8 = std::integral_constant<int, 8>;
      using C16 = std::integral_constant<int, 16>;
      auto launch = [&](auto fmt) {
        constexpr int FMT = decltype(fmt)::value;
        auto fc1 = [&](auto mt, auto kw) {
          constexpr int MT = decltype(mt)::value, KW = decltype(kw)::value;
          MoeMmaKernel<FMT, 1, kCanWide<FMT>, MT, KW><<<dim3(I / (16 * MT), P), KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(
              In<bf16>(c, 0), mp, P, R, I, 1024, In<uint8_t>(c, 3), In<bf16>(c, 4), Out<bf16>(h));
        };
        auto fc2 = [&](auto mt, auto kw) {
          constexpr int MT = decltype(mt)::value, KW = decltype(kw)::value;
          constexpr bool WD = kCanWide<FMT>;
          if (WD && wide2) {
            MoeMmaKernel<FMT, 2, WD, MT, KW><<<dim3(1024 / (16 * MT), P), KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(
                Out<bf16>(h), mp, P, R, 1024, I, In<uint8_t>(c, 5), In<bf16>(c, 6), Out<float>(part));
          } else {
            MoeMmaKernel<FMT, 2, false, MT, KW><<<dim3(1024 / (16 * MT), P), KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(
                Out<bf16>(h), mp, P, R, 1024, I, In<uint8_t>(c, 5), In<bf16>(c, 6), Out<float>(part));
          }
        };
        // tuning knob (BOLT_MOE_CFG = 10 * fc1 + fc2 choice); the default is (2 MMA tiles, 4 warps) for fc1 and (2, 8) for fc2
        const int sel = (cfg == 0 || cfg == 1) ? 0 : cfg;
        switch (sel / 10) {
          case 1: fc1(C1{}, C8{}); break;
          case 2: fc1(C2{}, C8{}); break;
          case 3: fc1(C1{}, C4{}); break;
          case 4: fc1(C4{}, C4{}); break;
          default: fc1(C2{}, C4{}); break;
        }
        switch (sel % 10) {
          case 2: fc2(C1{}, C8{}); break;
          case 3: fc2(C2{}, C4{}); break;
          case 4: fc2(C4{}, C4{}); break;
          case 5: fc2(C1{}, C16{}); break;
          case 6: fc2(C2{}, C16{}); break;
          default: fc2(C2{}, C8{}); break;
        }
      };
      if (iq2 == 2) launch(std::integral_constant<int, kFmtIq2N>{});
      else if (iq2) launch(std::integral_constant<int, kFmtIq2>{});
      else launch(std::integral_constant<int, kFmtQ4>{});
      MoeFinalizeKernel<<<dim3(R, 4), 64, 0, s>>>(Out<float>(part), In<float>(c, 2), topk, Out<bf16>(out));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct Nvfp4GemvState {
  const OrtApi& api;
  Nvfp4GemvState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto w = c.GetInput(1);
      auto out_shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int K = static_cast<int>(out_shape.back());
      const int R = static_cast<int>(Count(x) / K);
      const int N = static_cast<int>(Dim(w, 0));
      if (Dim(w, 1) * 2 != K || K % 1024 != 0) throw std::invalid_argument("Nvfp4Gemv: K mismatch (K % 1024 required)");
      out_shape.back() = N;
      auto y = c.GetOutput(0, out_shape);
      cudaStream_t s = Stream(api, raw);
      const bf16* xp = In<bf16>(c, 0);
      const uint8_t* wp = In<uint8_t>(c, 1);
      const uint8_t* sp = In<uint8_t>(c, 2);
      const float* gp = In<float>(c, 3);
      bf16* yp = Out<bf16>(y);
      if (R >= 2 && R <= 8 && K % 64 == 0) {
        LaunchMmaGemv<kFmtFp4>(xp, wp, sp, gp, R, N, K, yp, s);
        CheckLaunch();
        return nullptr;
      }
      switch (R) {
        case 1: LaunchNvfp4<1>(xp, wp, sp, gp, N, K, yp, s); break;
        case 2: LaunchNvfp4<2>(xp, wp, sp, gp, N, K, yp, s); break;
        case 3: case 4: {
          // pad rows up to 4 is not possible without copying; run 4-row variant only when R == 4
          if (R == 4) LaunchNvfp4<4>(xp, wp, sp, gp, N, K, yp, s);
          else LaunchNvfp4<3>(xp, wp, sp, gp, N, K, yp, s);
          break;
        }
        case 5: LaunchNvfp4<5>(xp, wp, sp, gp, N, K, yp, s); break;
        case 6: LaunchNvfp4<6>(xp, wp, sp, gp, N, K, yp, s); break;
        case 7: LaunchNvfp4<7>(xp, wp, sp, gp, N, K, yp, s); break;
        case 8: LaunchNvfp4<8>(xp, wp, sp, gp, N, K, yp, s); break;
        default: throw std::invalid_argument("Nvfp4Gemv: rows must be in [1,8]");
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// op registration
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// QGemv: block-32 symmetric int8 (zero point 128) / int4 (zero point 8, nibble-low-first) GEMV, bf16 scales,
//        weights in the MatMulNBits layout [N, K/32, 32 or 16].  y[R,N] = x[R,K] @ W^T
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float DotU8x16(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      acc += (__uint_as_float(0x4B000000u | byte) - 8388608.0f - 128.0f) * x[q * 4 + b];
    }
  return acc;
}

template <int BITS, int R, int ROWS, int IT>
__global__ void QGemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w, const bf16* __restrict__ sc,
                            int N, int K, bf16* __restrict__ y) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int nb = K / 32;
  constexpr int CH = BITS == 8 ? 2 : 1;  // uint4 per 32-value block
  uint4 wv[ROWS][IT][CH];
  float sv[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int b = lane + 32 * it;
      if (b < nb) {
        const uint4* wp = reinterpret_cast<const uint4*>(w + (static_cast<int64_t>(n) * nb + b) * (BITS * 4));
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = wp[c];
        sv[i][it] = B2F(sc[static_cast<int64_t>(n) * nb + b]);
      } else {
        sv[i][it] = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int b = lane + 32 * it;
    if (b >= nb) continue;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float xf[32];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + b * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = xq[q];
        const bf16* bb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) xf[q * 8 + i] = B2F(bb[i]);
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float d;
        if constexpr (BITS == 4) d = DotInt4x32(wv[i][it][0], xf);
        else d = DotU8x16(wv[i][it][0], xf) + DotU8x16(wv[i][it][1], xf + 16);
        acc[i][r] += d * sv[i][it];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v);
    }
}

template <int BITS, int R, int IT>
void LaunchQGemvImpl(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  if (N <= 4096) {  // small N: one row per warp to fill the GPU
    constexpr int ROWS = 1, WARPS = 4;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  }
}
template <int BITS, int R>
void LaunchQGemvR(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  const int nb = K / 32;
  if (nb <= 32) LaunchQGemvImpl<BITS, R, 1>(x, w, sc, N, K, y, s);
  else if (nb <= 64) LaunchQGemvImpl<BITS, R, 2>(x, w, sc, N, K, y, s);
  else if (nb <= 128) LaunchQGemvImpl<BITS, R, 4>(x, w, sc, N, K, y, s);
  else if (nb <= 192) LaunchQGemvImpl<BITS, R, 6>(x, w, sc, N, K, y, s);
  else throw std::invalid_argument("QGemv: K too large");
}
template <int BITS>
void LaunchQGemv(int R, const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  switch (R) {
    case 1: LaunchQGemvR<BITS, 1>(x, w, sc, N, K, y, s); break;
    case 2: LaunchQGemvR<BITS, 2>(x, w, sc, N, K, y, s); break;
    case 3: LaunchQGemvR<BITS, 3>(x, w, sc, N, K, y, s); break;
    case 4: LaunchQGemvR<BITS, 4>(x, w, sc, N, K, y, s); break;
    case 5: LaunchQGemvR<BITS, 5>(x, w, sc, N, K, y, s); break;
    case 6: LaunchQGemvR<BITS, 6>(x, w, sc, N, K, y, s); break;
    case 7: LaunchQGemvR<BITS, 7>(x, w, sc, N, K, y, s); break;
    case 8: LaunchQGemvR<BITS, 8>(x, w, sc, N, K, y, s); break;
    default: throw std::invalid_argument("QGemv: rows must be in [1,8]");
  }
}

template <int BITS>
struct QGemvState {
  const OrtApi& api;
  int iq2;  // Q4Gemv only: 1 = exact IQ2_XXS LUT codes (2..8 rows), 2 = native IQ2_XXS blocks (1..8 rows)
  QGemvState(const OrtApi& a, const OrtKernelInfo* info) : api(a), iq2(BITS == 4 ? OptAttr(info, "iq2", 0) : 0) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto w = c.GetInput(1);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int K = static_cast<int>(shape.back());
      const int R = static_cast<int>(Count(x) / K);
      const int N = static_cast<int>(Dim(w, 0));
      if (iq2 == 2) {  // native IQ2_XXS: B [N, K/32, 8], scale input = fp16 super-block scales [N, K/256]
        if (K % 256 != 0 || Dim(w, 1) * 32 != K || Dim(w, 2) != 8 || R < 1 || R > 8)
          throw std::invalid_argument("Q4Gemv iq2=2: needs B [N, K/32, 8] and 1..8 rows");
      } else if (K % 32 != 0 || Dim(w, 1) * 32 != K) {
        throw std::invalid_argument("QGemv: K mismatch");
      }
      shape.back() = N;
      auto y = c.GetOutput(0, shape);
      if (iq2 == 1 && !(R >= 2 && R <= 8 && K % 128 == 0)) throw std::invalid_argument("Q4Gemv iq2: needs 2..8 rows and K % 128 == 0");
      if (iq2 == 2) {
        LaunchMmaGemv<kFmtIq2N>(In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), nullptr, R, N, K, Out<bf16>(y), Stream(api, raw));
      } else if (iq2) {
        LaunchMmaGemv<kFmtIq2>(In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), nullptr, R, N, K, Out<bf16>(y), Stream(api, raw));
      } else if (R >= 2 && R <= 8 && K % 64 == 0) {
        LaunchMmaGemv<BITS == 8 ? kFmtQ8 : kFmtQ4>(In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), nullptr, R, N, K,
                                                   Out<bf16>(y), Stream(api, raw));
      } else {
        LaunchQGemv<BITS>(R, In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), N, K, Out<bf16>(y), Stream(api, raw));
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};
template <typename Derived, typename State>
struct BoltOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
};

constexpr auto BF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto F32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto U8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
constexpr auto I32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

struct MambaStepOp : BoltOp<MambaStepOp, MambaStepState> {
  const char* GetName() const { return "MambaStep"; }
  size_t GetInputTypeCount() const { return 9; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    // proj, conv_state, ssm_state, conv_w, conv_b, dt_bias, A_neg, D, norm_w
    static const ONNXTensorElementDataType t[9] = {BF, BF, F32, BF, BF, F32, F32, F32, F32};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[5] = {BF, BF, F32, BF, F32};
    return t[i];
  }
};
// MambaVerify: MambaStep for speculative verification. The state buffers hold one state per row ([Rs, ...], Rs >= R): the
// op starts from row `state_row` (int32 [1], device) and writes the state after token r into row r.
struct MambaVerifyState : MambaStepState {
  MambaVerifyState(const OrtApi& a, const OrtKernelInfo* info) : MambaStepState(a, info, true) {}
};
struct MambaVerifyOp : BoltOp<MambaVerifyOp, MambaVerifyState> {
  const char* GetName() const { return "MambaVerify"; }
  size_t GetInputTypeCount() const { return 10; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[10] = {BF, BF, F32, BF, BF, F32, F32, F32, F32, I32};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[5] = {BF, BF, F32, BF, F32};
    return t[i];
  }
};
struct MoeRouteOp : BoltOp<MoeRouteOp, MoeRouteState> {
  const char* GetName() const { return "MoeRoute"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i < 2 ? BF : F32; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[3] = {I32, F32, F32};
    return t[i];
  }
};
struct MoeDecodeOp : BoltOp<MoeDecodeOp, MoeDecodeState> {
  const char* GetName() const { return "MoeDecode"; }
  size_t GetInputTypeCount() const { return 7; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    // z, idx, w, fc1_w, fc1_s, fc2_w, fc2_s
    static const ONNXTensorElementDataType t[7] = {BF, I32, F32, U8, BF, U8, BF};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[3] = {BF, BF, F32};
    return t[i];
  }
};
struct MoeVerifyOp : BoltOp<MoeVerifyOp, MoeVerifyState> {
  const char* GetName() const { return "MoeVerify"; }
  size_t GetInputTypeCount() const { return 7; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[7] = {BF, I32, F32, U8, BF, U8, BF};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[4] = {BF, BF, F32, I32};
    return t[i];
  }
};
struct Relu2Op : BoltOp<Relu2Op, Relu2State> {
  const char* GetName() const { return "Relu2"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct Nvfp4GemvOp : BoltOp<Nvfp4GemvOp, Nvfp4GemvState> {
  const char* GetName() const { return "Nvfp4Gemv"; }
  size_t GetInputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[4] = {BF, U8, U8, F32};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};

struct Q8GemvOp : BoltOp<Q8GemvOp, QGemvState<8>> {
  const char* GetName() const { return "Q8Gemv"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 1 ? U8 : BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct Q4GemvOp : BoltOp<Q4GemvOp, QGemvState<4>> {
  const char* GetName() const { return "Q4Gemv"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 1 ? U8 : BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static MambaStepOp mamba;
    static MambaVerifyOp mamba_verify;
    static MoeRouteOp route;
    static MoeDecodeOp moe_decode;
    static MoeVerifyOp moe_verify;
    static Relu2Op relu2;
    static Nvfp4GemvOp fp4;
    static Q8GemvOp q8;
    static Q4GemvOp q4;
    static Ort::CustomOpDomain domain("bolt");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&mamba);
      domain.Add(&mamba_verify);
      domain.Add(&route);
      domain.Add(&moe_decode);
      domain.Add(&moe_verify);
      domain.Add(&relu2);
      domain.Add(&fp4);
      domain.Add(&q8);
      domain.Add(&q4);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
