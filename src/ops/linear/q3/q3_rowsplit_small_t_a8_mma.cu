#include "ops/linear/q3/q3_rowsplit_small_t_a8_mma.cuh"

#include "core/device.h"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/linear/q3/q3_dispatch.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
static_assert(q3_small_t_a8::kRows == kQ3SmallTRows, "small-T dispatch rows must match the kernel");
static_assert(q3_small_t_a8::kStageK == kQ3SmallTA8StepK,
              "small-T A8 dispatch stage must match the kernel");

namespace {

void require_small_t_a8_linear_shape(const A8G64Activation& act, const Weight& w,
                                     const Tensor& out) {
    if (w.n <= 0 || (w.n % q3_small_t::kRows) != 0 || w.k <= 0 ||
        static_cast<std::int32_t>(w.padded_shape[1]) != w.k ||
        (w.k % q3_small_t_a8::kStageK) != 0) {
        throw std::invalid_argument("q3 small-T A8 linear: unsupported weight shape");
    }
    if (out.ne[0] != w.n || out.ne[1] != act.q.ne[1] || act.q.ne[0] != w.k ||
        act.scale.ne[0] != act.q.ne[1] || act.scale.ne[1] != w.k / 64 || act.q.ne[1] < 1 ||
        act.q.ne[1] > q3_small_t::kMaxTokens) {
        throw std::invalid_argument("q3 small-T A8 linear: unsupported tensor shape");
    }
}

void require_small_t_a8_swiglu_shape(const A8G64Activation& act, const Weight& w,
                                     const Tensor& out) {
    const std::int32_t intermediate = out.ne[0];
    if (intermediate <= 0 || (intermediate % 16) != 0 || w.n != 2 * intermediate || w.k <= 0 ||
        static_cast<std::int32_t>(w.padded_shape[1]) != w.k ||
        (w.k % q3_small_t_a8::kStageK) != 0) {
        throw std::invalid_argument("q3 small-T A8 swiglu: unsupported weight shape");
    }
    if (out.ne[1] != act.q.ne[1] || act.q.ne[0] != w.k || act.scale.ne[0] != act.q.ne[1] ||
        act.scale.ne[1] != w.k / 64 || act.q.ne[1] < 1 ||
        act.q.ne[1] > q3_small_t::kMaxTokens) {
        throw std::invalid_argument("q3 small-T A8 swiglu: unsupported tensor shape");
    }
}

} // namespace

void launch_q3_mma_small_t_a8_r32_c8(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    require_small_t_a8_linear_shape(act, w, out);
    if (act.q.ne[1] > q3_small_t::kTokens) {
        throw std::invalid_argument("q3 small-T A8 linear: the 8-column tile owns 1..8 tokens");
    }
    const q3_small_t::Q3SmallTLinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], w.k / Q3RowSplitStorage::kGroupK};
    q3_small_t_a8::launch(problem, w.n / q3_small_t::kRows,
                          static_cast<const std::int8_t*>(act.q.data),
                          static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

void launch_q3_mma_small_t_a8_r32_c16(const A8G64Activation& act, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    require_small_t_a8_linear_shape(act, w, out);
    if (act.q.ne[1] <= q3_small_t::kTokens || act.q.ne[1] > q3_small_t::kWideTokens) {
        throw std::invalid_argument("q3 small-T A8 linear: the 16-column tile owns 9..16 tokens");
    }
    const q3_small_t::Q3SmallTLinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], w.k / Q3RowSplitStorage::kGroupK};
    q3_small_t_a8::launch(problem, w.n / q3_small_t::kRows,
                          static_cast<const std::int8_t*>(act.q.data),
                          static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

void launch_q3_mma_small_t_a8_swiglu_r16_c8(const A8G64Activation& act, const Weight& w,
                                            Tensor& out, cudaStream_t stream) {
    require_small_t_a8_swiglu_shape(act, w, out);
    if (act.q.ne[1] > q3_small_t::kTokens) {
        throw std::invalid_argument("q3 small-T A8 swiglu: the 8-column tile owns 1..8 tokens");
    }
    const q3_small_t::Q3SmallTSwiGluProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], w.k / Q3RowSplitStorage::kGroupK};
    q3_small_t_a8::launch(problem, out.ne[0] / 16, static_cast<const std::int8_t*>(act.q.data),
                          static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

void launch_q3_mma_small_t_a8_swiglu_r16_c16(const A8G64Activation& act, const Weight& w,
                                             Tensor& out, cudaStream_t stream) {
    require_small_t_a8_swiglu_shape(act, w, out);
    if (act.q.ne[1] <= q3_small_t::kTokens || act.q.ne[1] > q3_small_t::kWideTokens) {
        throw std::invalid_argument("q3 small-T A8 swiglu: the 16-column tile owns 9..16 tokens");
    }
    const q3_small_t::Q3SmallTSwiGluProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], w.k / Q3RowSplitStorage::kGroupK};
    q3_small_t_a8::launch(problem, out.ne[0] / 16, static_cast<const std::int8_t*>(act.q.data),
                          static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

} // namespace ninfer::ops::detail
