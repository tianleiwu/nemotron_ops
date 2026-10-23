# nemotron_ops

CUDA custom-op library for **NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB** (Mamba2 + attention + LatentMoE hybrid) decode on
ONNX Runtime's CUDA EP. Same layout as `qwen38-flash-ops`: plain ORT custom-op ABI, `sm_90`, ops run on the provider's compute
stream (CUDA-graph capturable), one shared library, domain `bolt`.

Build:

```bash
export PATH=$CUDA_HOME/bin:$PATH
cmake -S . -B build -DORT_INCLUDE_DIR=<ort>/include/onnxruntime/core/session -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build -j
# build/lib/libbolt_ops.so; load with SessionOptions.register_custom_ops_library
```

Test (every op against a torch reference, random data, real shapes of the model):

```bash
BOLT_PLUGIN=<libonnxruntime_providers_cuda.so of the CUDA plugin EP> BOLT_OPS=build/lib/libbolt_ops.so python tests/test_bolt_ops.py
```

| op | what it replaces | inputs -> outputs |
| --- | --- | --- |
| `MambaStep` | the ~35-node chain between `in_proj` and `out_proj` of a Mamba2 layer (Transpose, `CausalConvWithState`, Split, dt/softplus/clamp, `LinearAttention[gated]`, D skip, gated grouped RMSNorm, casts), R <= 16 rows | proj `[R,18560]`, conv_state `[1,10240,3]`, ssm_state `[1,128,128,64]` f32, conv_w, conv_b, dt_bias, A_neg, D, norm_w -> y `[R,8192]`, conv_state', ssm_state' (+2 scratch). State outputs may alias the state inputs. |
| `MambaVerify` | `MambaStep` for speculative verification: states are `[Rs>=R, ...]` buffers, input 9 `state_row` (int32 `[1]`, device) selects the starting state and the state after token r is written to row r (in-place safe) | as `MambaStep` + `state_row` |
| `MoeRoute` | fp32 router (`MatMul`, sigmoid, bias-corrected `TopK`, `GatherElements`, normalise, x5, scatter to the dense QMoE inputs) | x `[R,4096]`, gate_w `[E,4096]` bf16, bias `[E]` f32 -> idx `[R,k]` i32, w `[R,k]` f32 (+scores) |
| `MoeDecode` | `QMoE` for R <= 8 rows (relu2 experts, latent size 1024, int4 block-32 raw `[E,N,K/2]` weights + bf16 scales, zero point 8) | z `[R,1024]`, idx, w, fc1_w, fc1_s, fc2_w, fc2_s -> out `[R,1024]` (+2 scratch) |
| `Nvfp4Gemv` | modelopt NVFP4 linear (u8 `[N,K/2]`, fp8 e4m3 block-16 scales, fp32 global scale), R <= 8 rows, K in {4096, 8192}; fp16x2 FMA on rescaled e2m1 patterns | x, w, w_scale (u8 view), global scale -> y |
| `Q8Gemv` / `Q4Gemv` | `MatMulNBits` bits=8 / bits=4, block 32, symmetric (Q8_0 / IQ2_XXS-derived weights), bf16 scales, R <= 8 | x, B `[N,K/32,32 or 16]`, scales `[N,K/32]` -> y |
| `MoeVerify` | `MoeDecode` for R <= 8 rows (speculative verification): (token, expert) pairs are grouped by expert, each unique expert's int4 weights are read once and applied to all rows with tensor-core MMAs | as `MoeDecode` + an int32 scratch output (expert groups) |
| `Relu2` | `Relu` + `Mul` | x -> bf16(relu(x)^2) |

Match the bf16/fp32 tensor boundaries of the intended source graph/backend;
in particular, the Mamba boundaries below must be chosen explicitly.

## Explicit Mamba BF16 boundary controls

The default kernels preserve the existing FP32 convolution-through-SiLU and
FP32-readout-to-normalization behavior. For source-backend comparisons, two
independent build options restore intervening BF16 tensor boundaries:

```bash
cmake -S . -B build-bf16-boundaries \
  -DORT_INCLUDE_DIR=<ort>/include/onnxruntime/core/session \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DBOLT_MAMBA_ROUND_CONV_BEFORE_SILU=ON \
  -DBOLT_MAMBA_ROUND_READOUT_BEFORE_NORM=ON
cmake --build build-bf16-boundaries -j
```

Both options default to `OFF`; enable either independently to isolate its
effect. They apply to `MambaStep` and `MambaVerify`, including the specialized
eight-row path:

- `BOLT_MAMBA_ROUND_CONV_BEFORE_SILU`: cast the completed FP32 convolution to
  BF16 and widen back to FP32 before SiLU. Raw convolution state is unchanged.
- `BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM`: cast the readout to BF16 and widen
  back to FP32 before gating and variance accumulation. Recurrent state and
  the diagnostic FP32 readout output remain unchanged. Gate, normalization
  and gamma arithmetic retain their existing FP32 policy.

These are **explicit numerical variants**, not a claim of full-model parity.
Separate PyTorch BF16 convolution and SiLU impose the first boundary, whereas
a fused causal-convolution backend may not. BF16 scan/update output imposes
the second boundary even when recurrent state is FP32. Pin the intended
reference backend before choosing a variant. Do not overwrite a qualified
package's library or compare different packed weights when attributing drift.
Use ORT headers matching the runtime that will load the library: a custom
library requesting a newer ORT API cannot be loaded by an older runtime.

Focused CUDA tests build all four combinations and exercise both generic and
specialized convolution, state history, and group-size 1,024 and 512 norm
paths. They use small discriminating operands and no model weights, Python
ORT, downloads, or explicit protobuf APIs:

```bash
cmake -S . -B build-rounding-tests \
  -DORT_INCLUDE_DIR=<ort>/include/onnxruntime/core/session \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DBOLT_BUILD_MAMBA_ROUNDING_TESTS=ON
cmake --build build-rounding-tests -j
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-rounding-tests \
  --output-on-failure -R mamba_rounding
```

Passing these stage regressions does not establish full decoder, ordinary
generation, or speculative-integration correctness.

## Multi-row (speculative verify) path

`Nvfp4Gemv`, `Q8Gemv`, `Q4Gemv` (2 <= R <= 8, K % 64 == 0) and `MoeVerify` use `mma.m16n8k16` (bf16 in, fp32 accumulate): a warp owns 16 weight rows,
the (<= 8) activation rows are the MMA's N dimension, weights are dequantised in registers straight from the checkpoint layout (no repacking; mma sums over k, so
k is permuted consistently between weights and activations). Weights become bf16 (`(q - zp) * scale` rounded once, like ORT's `MatMulNBits` dequantisation; NVFP4
is exact), the fp32 accumulation replaces the old fp16 partial sums. R = 1 keeps the dedicated GEMV kernels. Debug knobs: `BOLT_GEMV_CFG`, `BOLT_MOE_CFG`.

`MambaVerify`: states are `[Rs, ...]` buffers; input 9 (`state_row`, device int32) selects the start state and row r receives the state after token r, so rejected
draft tokens need no rollback copy.

Benchmarks: `tools/bench_gemv.py`, `tools/bench_moe.py` (eager, so kernels below ~10 us are launch-bound there; use nsys for those).

## Exact IQ2_XXS (`iq2=1`)

`Q4Gemv` and `MoeVerify` take an optional integer attribute `iq2`. With `iq2=1` the 4-bit weights are **LUT codes of the exact ggml IQ2_XXS values** instead of symmetric int4:
code = magnitude index (0, 1, 2 <-> 8, 25, 43) + 4 * (negative), packed low nibble first, bf16 block-32 scale `db` (see `iq2xxs_to_lut4` in the exporter's `bolt_weights.py`).
The int4 re-quantisation used before maps {8, 25, 43} to {8.5, 25.5, 42.6} (up to 6 % error on the smallest magnitude, measurably worse NLL: +0.012 nats/token on 2k-token texts).
The kernel decodes 4 weights per `prmt` pair (high- and low-byte LUT planes of the bf16 values), multiplies by the scale and feeds adjacent k pairs to `mma.m16n8k16`
(the x fragments need no repacking). Same memory and bandwidth as int4. Requires 2..8 rows (the verify path); R = 1 is not implemented for `iq2=1`.

## Native IQ2_XXS (`iq2=2`)

`Q4Gemv` and `MoeVerify` with `iq2=2` read the IQ2_XXS data at its native 2.0625 bits per weight instead of the 4-bit LUT codes of `iq2=1`
(2.2x smaller than int4 + bf16 scales; the Bolt-74B engine drops from 39.6 GiB to 20.3 GiB of weights). Layout (`iq2xxs_to_native` in the exporter's `bolt_weights.py`):
`B` = `[N, K/32, 8]` uint8, the ggml 8-byte records (4 codebook-index bytes + one u32 with four 7-bit sign indices and a 4-bit scale);
`scale` input = the fp16 super-block scales `[N, K/256]` carried as raw bits in a bf16-typed tensor. K % 256 == 0, 1..8 rows.
Decoding: a thread owns 16 adjacent k = two codebook vectors, looked up in `kIq2xxsGridBf16` (pre-expanded bf16 magnitudes, global memory/L1), sign bits are xor-ed in
with one IMAD + LOP per bf16 pair, then scaled by bf16(d * (0.5 + s) * 0.25): the weights are bit-identical to `iq2=1`.
Cost: MoeVerify 111 us -> 145 us per layer (R=8, E=304, I=2304) and ~1 ms more per 8-token verify step (about 8 % lower speculative throughput) for half the weight memory.
A shared-memory table variant was slower (per-block table fill dominates the tiny per-block work); `BOLT_MOE_CFG` selects other tilings (default is the best).
Uniform 2-bit (ORT `QMoE` int2) cannot represent IQ2_XXS (6 magnitudes/signs from a 256-entry vector codebook): 37 % relative weight error and a destroyed model (teacher-forced NLL 12 vs 0.23).
