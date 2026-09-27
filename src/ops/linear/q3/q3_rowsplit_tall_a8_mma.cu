#include "ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh"

#include "core/device.h"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/linear/q3/q3_launch.h"
#include "ops/linear/q3/q3_dispatch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
static_assert(q3_tall_a8::kRows == kQ3A8Rows, "A8 dispatch rows must match the tall engine");
static_assert(q3_tall_a8::kStepK == kQ3A8StepK, "A8 dispatch K step must match the tall engine");

namespace {

void require_tall_a8_linear_shape(const A8G64Activation& act, const Weight& w, const Tensor& out) {
    if (w.n <= 0 || w.n % q3_tall_a8::kRows != 0 || w.k <= 0 ||
        w.padded_shape[1] != w.k || w.k % q3_tall_a8::kStepK != 0) {
        throw std::invalid_argument("q3 tall A8 linear: unsupported weight shape");
    }
    if (out.ne[0] != w.n || out.ne[1] != act.q.ne[1] || act.q.ne[0] != w.k ||
        act.scale.ne[0] != act.q.ne[1] || act.scale.ne[1] != w.k / 64) {
        throw std::invalid_argument("q3 tall A8 linear: unsupported tensor shape");
    }
}

} // namespace

void launch_q3_mma_tall_a8_r128_c128(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    require_tall_a8_linear_shape(act, w, out);
    const q3_tall::Q3LinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    q3_tall_a8::launch<128>(problem, w.n / q3_tall_a8::kRows,
                            static_cast<const std::int8_t*>(act.q.data),
                            static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

void launch_q3_mma_tall_a8_swiglu_r64_c128(const A8G64Activation& act, const Weight& w,
                                           Tensor& out, cudaStream_t stream) {
    const std::int32_t intermediate = out.ne[0];
    if (w.n != 2 * intermediate || intermediate <= 0 || intermediate % 64 != 0 || w.k <= 0 ||
        w.padded_shape[1] != w.k || w.k % q3_tall_a8::kStepK != 0 ||
        out.ne[1] != act.q.ne[1] || act.q.ne[0] != w.k || act.scale.ne[0] != act.q.ne[1] ||
        act.scale.ne[1] != w.k / 64) {
        throw std::invalid_argument("q3 tall A8 swiglu: unsupported tensor shape");
    }
    const q3_tall::Q3SwiGluProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), intermediate};
    q3_tall_a8::launch<128>(problem, intermediate / 64,
                            static_cast<const std::int8_t*>(act.q.data),
                            static_cast<const float*>(act.scale.data), w.k, act.q.ne[1], stream);
}

} // namespace ninfer::ops::detail
