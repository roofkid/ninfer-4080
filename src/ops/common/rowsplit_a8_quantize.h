#pragma once

// Int8 activation (A8) for the RowSplit prefill GEMMs, aligned with their 64-value activation
// groups. Per token t and 64-column group g of x [K, T]: amax = max |x|, scale = amax / 127 and
// q = rint(x * (127 / amax)) clamped to [-127, 127], all in FP32 with IEEE division and
// round-to-nearest-even (scale 0 and q 0 for an all-zero group). The weights' codes multiply q
// exactly; see q3_rowsplit_tall_a8_mma.cuh and docs/maintainer/op-development.md 6.4.

#include "core/dtype.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct A8G64Activation {
    Tensor q;     // int8 [K, T], token-major like x
    Tensor scale; // FP32 [T, K / 64]: the scales of group g are contiguous over tokens
};

template <class Allocator>
A8G64Activation allocate_a8_g64_activation(Allocator& allocator, std::int32_t k,
                                           std::int32_t tokens) {
    return {allocator.alloc(DType::I8, {k, tokens}), allocator.alloc(DType::FP32, {tokens, k / 64})};
}

// x: contiguous BF16 [K, T], K a multiple of 64; `out` allocated for the same K and T.
void a8_g64_quantize(const Tensor& x, A8G64Activation& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
