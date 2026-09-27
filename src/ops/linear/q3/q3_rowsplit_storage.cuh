#pragma once

// Q3G128_F16S RowSplit storage primitives.
//
// Each full logical group owns 128 signed three-bit codes packed little-endian into 48 bytes
// (code j occupies bits 3j..3j+2 of the group stream) plus one FP16 dequantization multiplier.
// The represented weight is code * scale. A row's code bytes lead its payload; the scale plane
// follows at the registered alignment. There is no high plane.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q3RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 48;
    static constexpr int kScaleBytesPerGroup = 2;
};

// Sign-extend one unsigned three-bit code word.
__device__ __forceinline__ int q3_signed_code(std::uint32_t unsigned_code) noexcept {
    return static_cast<int>(unsigned_code) - ((unsigned_code & 0x4u) != 0 ? 8 : 0);
}


} // namespace ninfer::ops::detail
