#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Applies the activation-free finish side of linear_dynamic_grouped_conv_add to an already
// materialized BF16 projection [5120, W*B]. The caller owns the projection workspace and the
// operand non-overlap rules of the public Op; this entry point only adds the convolution taps.
void dynamic_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                    const Tensor& finish_delta, Tensor& residual, int width,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
