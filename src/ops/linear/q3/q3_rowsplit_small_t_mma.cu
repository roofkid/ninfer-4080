#include "ops/linear/q3/q3_rowsplit_small_t_mma.cuh"

#include "core/device.h"
#include "ops/linear/q3/q3_dispatch.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
static_assert(q3_small_t::kRows == kQ3SmallTRows, "small-T dispatch rows must match the kernel");
static_assert(q3_small_t::kGroupK == kQ3SmallTStepK, "small-T dispatch K step must match the kernel");

namespace {

void require_small_t_linear_shape(const Tensor& x, const Weight& w, const Tensor& out) {
    if (w.n <= 0 || (w.n % q3_small_t::kRows) != 0 || w.k <= 0 || (w.k % 8) != 0 ||
        w.padded_shape[1] < w.k || (w.padded_shape[1] % q3_small_t::kGroupK) != 0) {
        throw std::invalid_argument("q3 small-T MMA linear: unsupported weight shape");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1] || x.ne[2] != 1 ||
        x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1 || x.ne[1] < 1 ||
        x.ne[1] > q3_small_t::kMaxTokens) {
        throw std::invalid_argument("q3 small-T MMA linear: unsupported tensor shape");
    }
}

void require_small_t_swiglu_shape(const Tensor& x, const Weight& w, const Tensor& out) {
    const std::int32_t intermediate = out.ne[0];
    if (intermediate <= 0 || (intermediate % 16) != 0 || w.n != 2 * intermediate || w.k <= 0 ||
        (w.k % 8) != 0 || w.padded_shape[1] < w.k ||
        (w.padded_shape[1] % q3_small_t::kGroupK) != 0) {
        throw std::invalid_argument("q3 small-T MMA swiglu: unsupported weight shape");
    }
    if (x.ne[0] != w.k || out.ne[1] != x.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || x.ne[1] < 1 || x.ne[1] > q3_small_t::kMaxTokens) {
        throw std::invalid_argument("q3 small-T MMA swiglu: unsupported tensor shape");
    }
}

} // namespace

void launch_q3_mma_small_t_r32_c8(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    require_small_t_linear_shape(x, w, out);
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const q3_small_t::Q3SmallTLinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], padded_k / Q3RowSplitStorage::kGroupK};
    q3_small_t::launch(problem, w.n / q3_small_t::kRows,
                       static_cast<const __nv_bfloat16*>(x.data), w.k, padded_k, x.ne[1], stream);
}

void launch_q3_mma_small_t_swiglu_r16_c8(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    require_small_t_swiglu_shape(x, w, out);
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const q3_small_t::Q3SmallTSwiGluProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0], padded_k / Q3RowSplitStorage::kGroupK};
    q3_small_t::launch(problem, out.ne[0] / 16, static_cast<const __nv_bfloat16*>(x.data), w.k,
                       padded_k, x.ne[1], stream);
}

} // namespace ninfer::ops::detail
