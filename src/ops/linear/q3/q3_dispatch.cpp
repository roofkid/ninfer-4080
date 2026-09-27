#include "ops/linear/q3/q3_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q3Launch select_q3_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (n <= 0 || k <= 0 || t <= 0) { throw std::invalid_argument("q3 linear: unsupported shape or T"); }
    // Decode and MTP widths use the warp-per-row GEMV; prefill uses the pipelined tall engine and
    // its 64-token tile through T=127; the staged 32x64 tile serves the narrow remainder and the
    // shapes the tall engine does not own.
    if (t <= 8) { return launch_q3_gemv_r8_c8; }
    if (n % 128 == 0 && t >= 64) {
        return t >= 128 ? launch_q3_mma_tall_r128_c128 : launch_q3_mma_tall_r128_c64;
    }
    return launch_q3_mma_r32_c64;
}

Q3Launch select_q3_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_q3_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q3 linear: unsupported policy");
}

void q3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    if (w.group_size != 128 || w.group != 128 || w.scale_dtype != DType::FP16 ||
        w.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("q3 linear: weight is not a Q3G128_F16S row-split tensor");
    }
    if (w.padded_shape[1] <= 0 || w.padded_shape[1] % 128 != 0 || w.padded_shape[1] < w.k ||
        w.qdata == nullptr || w.scales == nullptr || w.qhigh != nullptr) {
        throw std::invalid_argument("q3 linear: weight planes are not a padded Q3 row-split tensor");
    }
    const Q3Launch launch = select_q3_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
