#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q3Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q3_gemv_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q3_mma_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q3_mma_r32_c64_f32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
