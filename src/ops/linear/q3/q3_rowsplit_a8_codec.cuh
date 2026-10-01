#pragma once

// A8 decode primitives shared by the Q3G128_F16S int8 routes: the tall prefill GEMM
// (q3_rowsplit_tall_a8_mma.cuh) and the small-T decode/verification GEMM
// (q3_rowsplit_small_t_a8_mma.cuh).
//
// The weight planes store signed three-bit codes in two's complement. The A8 routes multiply them
// as exact int8 operands, so the packed 3-bit window has to become one sign-extended byte per code.

namespace ninfer::ops::detail::q3_a8 {

// Four int8 lanes of 3-bit two's-complement codes: ((n ^ 4) + 0x7C) ^ 0x80 is n - 8 for n >= 4
// and n otherwise, with no carry between bytes.
__device__ __forceinline__ unsigned sign_extend(unsigned bytes) {
    return ((bytes ^ 0x04040404u) + 0x7C7C7C7Cu) ^ 0x80808080u;
}

// One 12-bit window of four 3-bit codes, placed one per byte and sign extended to int8.
__device__ __forceinline__ unsigned spread4(unsigned window) {
    const unsigned bytes = (window & 0x7u) | ((window & 0x38u) << 5) | ((window & 0x1C0u) << 10) |
                           ((window & 0xE00u) << 15);
    return sign_extend(bytes);
}

} // namespace ninfer::ops::detail::q3_a8
