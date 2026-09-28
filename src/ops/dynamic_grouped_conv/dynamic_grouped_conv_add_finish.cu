#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"

#include "core/device.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

__global__ void dynamic_conv_add_finish_kernel(const __nv_bfloat16* projected,
                                               const __nv_bfloat16* base,
                                               const __nv_bfloat16* delta, __nv_bfloat16* residual,
                                               int width) {
    constexpr int kRows = kDynamicConvAddRows;
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    dynamic_conv_add_finish_value(row, col, width, __bfloat162float(projected[index]),
                                  col % width ? __bfloat162float(projected[index - kRows]) : 0.0f,
                                  base, delta, residual);
}

} // namespace

void dynamic_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                    const Tensor& finish_delta, Tensor& residual, int width,
                                    cudaStream_t stream) {
    const int tokens = residual.ne[1] * residual.ne[2];
    const dim3 grid((kDynamicConvAddRows + 255) / 256, static_cast<unsigned>(tokens));
    dynamic_conv_add_finish_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
