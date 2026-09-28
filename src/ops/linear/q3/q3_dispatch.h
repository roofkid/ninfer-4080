#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// With an A8 permission, exact-K Q3 parents run the int8 tensor-core route from this width on;
// narrower extents (decode and speculative verification) keep the A16 routes.
inline constexpr std::int32_t kQ3A8MinTokens = 129;
// The tall engine's block shape; the dispatch layer cannot include the device header.
inline constexpr std::int32_t kQ3A8Rows = 128;
inline constexpr std::int32_t kQ3A8StepK = 64;
// The small-T tensor-core route's block shape; the dispatch layer cannot include the device
// header.
inline constexpr std::int32_t kQ3SmallTRows = 32;
inline constexpr std::int32_t kQ3SmallTStepK = 256;

Q3Launch select_q3_a16_launch(std::int32_t n, std::int32_t k, std::int32_t padded_k,
                             std::int32_t t);
// Resolves the A16 route. The A8 route needs caller workspace and is selected by q3_uses_a8()
// and q3_dispatch() instead, so a permissive policy resolves to A16 here.
Q3Launch select_q3_launch(std::int32_t n, std::int32_t k, std::int32_t padded_k, std::int32_t t,
                          LinearPolicy policy);

// True when the A8 route serves this exact problem: the policy admits it, N is a whole number of
// 128-row blocks, K is an exact multiple of 64 with no padding, and the width is at least
// kQ3A8MinTokens.
[[nodiscard]] bool q3_uses_a8(std::int32_t n, std::int32_t k, std::int32_t padded_k,
                              LinearPolicy policy, std::int32_t tokens) noexcept;

// Bytes the A8 activation quantization (int8 [K,T] + FP32 [T,K/64]) needs.
[[nodiscard]] std::size_t q3_a8_workspace_capacity_bytes(std::int32_t k, std::int32_t tokens);

void q3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
