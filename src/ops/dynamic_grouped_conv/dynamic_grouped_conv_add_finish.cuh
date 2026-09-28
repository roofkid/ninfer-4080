#pragma once

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// Finish side of the DFlash2 dynamic grouped convolution. The runtime base_kernel view is
// BF16 [5120,2,2] indexed as channel + 5120 * (tap + 2 * side); the finish side uses side 1.
constexpr int kDynamicConvAddRows   = 5120;
constexpr int kDynamicConvAddGroups = 320;

__device__ __forceinline__ void dynamic_conv_add_finish_value(int row, int col, int width,
                                                              float current, float previous,
                                                              const __nv_bfloat16* base,
                                                              const __nv_bfloat16* delta,
                                                              __nv_bfloat16* residual) {
    constexpr int kRows   = kDynamicConvAddRows;
    constexpr int kGroups = kDynamicConvAddGroups;
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

} // namespace ninfer::ops::detail
