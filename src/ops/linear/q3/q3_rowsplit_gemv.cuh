#pragma once

// Q3G128_F16S RowSplit x BF16 warp-per-row GEMV.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row. Each iteration consumes two consecutive 128-code groups: the low
// half-warp decodes the even group's eight-code windows, the high half-warp the odd group's, so
// all lanes stay busy and each code-byte triple is fetched once. Every lane accumulates all Cols
// tokens and one warp reduction publishes the row. There is no shared staging and no barrier, so
// weight traffic is read exactly once per token step and the token extent never changes it. This
// is the decode route for the MTP width (1..8 tokens).

#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// clang-format off
template <int MaxCols, bool FullK>
__global__ __launch_bounds__(256, 2) void q3_rowsplit_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    static_assert(MaxCols >= 1 && MaxCols <= 8);
    constexpr int kWarps      = 8;
    constexpr int kGroupSize  = Q3RowSplitStorage::kGroupK;
    constexpr int kGroupBytes = Q3RowSplitStorage::kCodeBytesPerGroup;

    const std::int32_t warp = static_cast<std::int32_t>(threadIdx.x) >> 5;
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x) & 31;
    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.x) * kWarps + warp;
    if (row >= rows) { return; }

    const std::int32_t groups_per_row = padded_k / kGroupSize;
    const std::uint8_t* row_codes =
        codes + static_cast<std::int64_t>(row) * groups_per_row * kGroupBytes;
    const std::uint8_t* row_scales = scales + static_cast<std::int64_t>(row) * groups_per_row * 2;

    const std::int32_t group_lane = lane >> 4; // 0 selects the even group, 1 the odd group
    const std::int32_t local      = lane & 15;
    const std::int32_t byte0      = 3 * local;
    const std::int32_t code_base  = 8 * local;

    float accum[MaxCols];
#pragma unroll
    for (int c = 0; c < MaxCols; ++c) { accum[c] = 0.0F; }

    for (std::int32_t group = 0; group < groups_per_row; group += 2) {
        const std::int32_t active_group = group + group_lane;
        float weights[8];
        std::int32_t kbase = 0;
        if (active_group < groups_per_row) {
            const std::uint8_t* group_ptr = row_codes + active_group * kGroupBytes;
            const std::uint32_t window =
                static_cast<std::uint32_t>(group_ptr[byte0]) |
                (static_cast<std::uint32_t>(group_ptr[byte0 + 1]) << 8) |
                (static_cast<std::uint32_t>(group_ptr[byte0 + 2]) << 16);
            const float scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(row_scales + active_group * 2)));
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                weights[c] = static_cast<float>(q3_signed_code((window >> (3 * c)) & 0x7u)) * scale;
            }
            kbase = active_group * kGroupSize + code_base;
        } else {
#pragma unroll
            for (int c = 0; c < 8; ++c) { weights[c] = 0.0F; }
        }
#pragma unroll
        for (int col = 0; col < MaxCols; ++col) {
            if (col >= cols) { continue; }
            const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(col) * k + kbase;
            if constexpr (FullK) {
                const uint4 packed = *reinterpret_cast<const uint4*>(xrow);
                const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
                const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&packed.z);
                const __nv_bfloat162 p3 = *reinterpret_cast<const __nv_bfloat162*>(&packed.w);
                const float2 f0 = __bfloat1622float2(p0);
                const float2 f1 = __bfloat1622float2(p1);
                const float2 f2 = __bfloat1622float2(p2);
                const float2 f3 = __bfloat1622float2(p3);
                accum[col] += weights[0] * f0.x + weights[1] * f0.y + weights[2] * f1.x +
                              weights[3] * f1.y + weights[4] * f2.x + weights[5] * f2.y +
                              weights[6] * f3.x + weights[7] * f3.y;
            } else {
                float partial = 0.0F;
                for (int c = 0; c < 8; ++c) {
                    if (kbase + c < k) { partial += weights[c] * __bfloat162float(xrow[c]); }
                }
                accum[col] += partial;
            }
        }
    }

#pragma unroll
    for (int col = 0; col < MaxCols; ++col) {
        float value = accum[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && col < cols) {
            out[static_cast<std::int64_t>(col) * rows + row] = __float2bfloat16(value);
        }
    }
}

} // namespace ninfer::ops::detail
