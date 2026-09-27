#include "ops/linear/q3/q3_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRowsPerCta = 8;

constexpr std::int32_t div_up(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

template <int MaxCols>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerCta)), 1u, 1u);

    if (k == padded_k) {
        q3_rowsplit_gemv_kernel<MaxCols, true><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows,
            k, cols, padded_k);
    } else {
        q3_rowsplit_gemv_kernel<MaxCols, false><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows,
            k, cols, padded_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q3_gemv_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_schedule<8>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
