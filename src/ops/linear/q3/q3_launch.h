#pragma once

#include "core/tensor.h"
#include "ops/common/rowsplit_a8_quantize.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q3Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q3_gemv_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// One weight row per warp over cp.async staged code windows; the small-T decode route.
void launch_q3_gemv_r8_c8_staged(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q3_mma_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q3_mma_r32_c64_f32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// Pipelined tall prefill routes: 128 weight rows x 64 or 128 tokens per CTA.
void launch_q3_mma_tall_r128_c64(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q3_mma_tall_r128_c128(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
// Folded gate/up SwiGLU over the stacked Q3 parent, 64 output rows x 128 tokens per CTA.
void launch_q3_mma_tall_swiglu_r64_c128(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream);
// A8 prefill routes: the activation is quantized per token and 64-code group by
// a8_g64_quantize, and the weight codes multiply it with m16n8k32 s8 MMAs.
void launch_q3_mma_tall_a8_r128_c128(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void launch_q3_mma_tall_a8_swiglu_r64_c128(const A8G64Activation& act, const Weight& w,
                                           Tensor& out, cudaStream_t stream);
// Small-T tensor-core routes: 32 weight rows x up to 8 tokens per CTA over whole-group
// staged codes, several CTAs per SM and an eight-warp K-split.
void launch_q3_mma_small_t_r32_c8(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
void launch_q3_mma_small_t_swiglu_r16_c8(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
// Small-T A8 routes: 32 weight rows x up to 8 tokens per CTA over whole 512-code stages of the
// activation quantized by a8_g64_quantize, one 64-code group per warp and m16n8k32 s8 MMAs.
void launch_q3_mma_small_t_a8_r32_c8(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void launch_q3_mma_small_t_a8_swiglu_r16_c8(const A8G64Activation& act, const Weight& w,
                                            Tensor& out, cudaStream_t stream);
} // namespace ninfer::ops::detail
