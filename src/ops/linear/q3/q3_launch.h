#pragma once

#include "core/tensor.h"

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

} // namespace ninfer::ops::detail
