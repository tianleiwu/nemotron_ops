// Exercise the actual kernels without requiring a Python ORT/plugin installation.
#include "../src/bolt_ops.cu"

#include <iostream>

namespace {

void Cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

float Round(float value) { return __bfloat162float(__float2bfloat16(value)); }
float HostSilu(float value) { return value / (1.0f + std::exp(-value)); }

template <typename T>
struct Buffer {
  T* data = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { Cuda(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Put(const std::vector<T>& values) {
    Require(values.size() == size, "Wrong input size");
    Cuda(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> Get() {
    std::vector<T> values(size);
    Cuda(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

void TestConv(int rows, bool verify, bool fixed) {
  constexpr int di = 64, cd = 128, stride = di + cd + 1;
  const int state_rows = verify ? rows : 1;
  std::vector<bf16> projection(rows * stride, __float2bfloat16(0.f));
  std::vector<bf16> weights(cd * 4, __float2bfloat16(0.f));
  std::vector<bf16> bias(cd, __float2bfloat16(0.f));
  std::vector<bf16> state(state_rows * cd * 3, __float2bfloat16(0.f));
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cd; ++c)
      projection[r * stride + di + c] = __float2bfloat16(0.70703125f);
  for (int c = 0; c < cd; ++c) weights[c * 4 + 3] = __float2bfloat16(1.0078125f);
  Buffer<bf16> p(projection.size()), w(weights.size()), b(cd), si(state.size()), so(state.size()), activation(rows * cd);
  Buffer<int> selected(1);
  p.Put(projection); w.Put(weights); b.Put(bias); si.Put(state); selected.Put({state_rows - 1});
  const int* sel = verify ? selected.data : nullptr;
  if (fixed) {
    MambaConvKernel<8, true><<<2, 64>>>(p.data, rows, stride, di, cd, si.data, w.data, b.data,
                                       so.data, activation.data, sel, verify);
  } else {
    MambaConvKernel<<<1, 128>>>(p.data, rows, stride, di, cd, si.data, w.data, b.data,
                               so.data, activation.data, sel, verify);
  }
  Cuda(cudaGetLastError());
  const float accumulator = 0.70703125f * 1.0078125f;
  const float raw = Round(HostSilu(accumulator));
  const float rounded = Round(HostSilu(Round(accumulator)));
  Require(raw != rounded, "Convolution fixture does not distinguish rounding policies");
  const float expected = BOLT_MAMBA_ROUND_CONV_BEFORE_SILU ? rounded : raw;
  for (bf16 value : activation.Get())
    Require(__bfloat162float(value) == expected, "Convolution rounded at the wrong boundary");
  const auto states = so.Get();
  for (int r = 0; r < state_rows; ++r) {
    const int last = verify ? r : rows - 1;
    for (int c = 0; c < cd; ++c)
      for (int tap = 0; tap < 3; ++tap) {
        const float expected_state = last + tap >= 2 ? 0.70703125f : 0.f;
        Require(__bfloat162float(states[(r * cd + c) * 3 + tap]) == expected_state,
                "Convolution rounding changed raw state history");
      }
  }
}

std::vector<float> ReferenceNorm(const std::vector<float>& input, int group_size, bool round_readout) {
  std::vector<float> result(input.size());
  const float gate = HostSilu(1.f);
  for (size_t start = 0; start < input.size(); start += group_size) {
    float sum = 0.f;
    for (int i = 0; i < group_size; ++i) {
      const float value = (round_readout ? Round(input[start + i]) : input[start + i]) * gate;
      sum += value * value;
    }
    const float inv_std = 1.f / std::sqrt(sum / group_size + 1e-5f);
    for (int i = 0; i < group_size; ++i) {
      const float value = (round_readout ? Round(input[start + i]) : input[start + i]) * gate;
      result[start + i] = Round(value * inv_std);
    }
  }
  return result;
}

void TestNorm(int group_size) {
  constexpr int rows = 2, groups = 2;
  const int di = group_size * groups;
  const float pattern[] = {1.00390625f, 1.01171875f, 0.4990234375f, 2.0078125f};
  std::vector<float> input(rows * di);
  for (size_t i = 0; i < input.size(); ++i) input[i] = pattern[i % 4];
  std::vector<bf16> gate(rows * di, __float2bfloat16(1.f));
  std::vector<float> weight(di, 1.f);
  Buffer<bf16> p(gate.size()), output(rows * di);
  Buffer<float> readout(input.size()), gamma(di);
  p.Put(gate); readout.Put(input); gamma.Put(weight);
  MambaNormKernel<<<dim3(rows, groups), 256>>>(p.data, di, di, group_size, 1e-5f,
                                             readout.data, gamma.data, output.data);
  Cuda(cudaGetLastError());
  const auto raw = ReferenceNorm(input, group_size, false);
  const auto rounded = ReferenceNorm(input, group_size, true);
  Require(raw != rounded, "Normalization fixture does not distinguish rounding policies");
  const auto& expected = BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM ? rounded : raw;
  const auto actual = output.Get();
  for (size_t i = 0; i < actual.size(); ++i)
    Require(__bfloat162float(actual[i]) == expected[i], "Normalization rounded at the wrong boundary");
  Require(readout.Get() == input, "Normalization modified the FP32 readout buffer");
}

}  // namespace

int main() {
  try {
    Cuda(cudaSetDevice(0));
    size_t free_bytes, total_bytes;
    Cuda(cudaMemGetInfo(&free_bytes, &total_bytes));
    Require(free_bytes > 256 * 1024 * 1024, "Insufficient CUDA headroom");
    TestConv(1, false, false);
    TestConv(3, true, false);
    TestConv(8, true, true);
    TestConv(16, false, false);
    TestNorm(1024);
    TestNorm(512);
    Cuda(cudaDeviceSynchronize());
    std::cout << "PASS conv_round=" << BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
              << " readout_round=" << BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
