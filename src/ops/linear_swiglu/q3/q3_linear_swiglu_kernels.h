#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Fused Q3G128_F16S gate/up projection with the SwiGLU activation. The single-token width runs a
// warp-per-row GEMV, widths 2..16 the small-T tensor-core route, widths of at least 64 tokens the
// folded pipelined tall engine (the A8 int8 route from 129), and the narrow remainder the staged
// Q3 MMA schedule over a column-chunk FP32 plane. Only the chunked route and the A8 quantization
// need workspace; the single BF16 rounding is the Op's output storage rounding either way.
[[nodiscard]] std::size_t q3_linear_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                   std::int32_t max_tokens,
                                                                   LinearPolicy policy);

void q3_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                               LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
