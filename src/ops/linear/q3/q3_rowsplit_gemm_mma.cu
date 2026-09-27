#include "ops/linear/q3/q3_rowsplit_gemm_mma.cuh"

#include "core/device.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// One qualified schedule for every registered Q3 shape: a 32-row x 64-token tile with four warps
// and a two-stage cp.async pipeline. The static shared-memory budget keeps two CTAs resident.
using Q3MmaR32C64Schedule =
    Q3RowSplitMmaGemmSchedule<32, 64, 128, 16, 32, 2, 2, Q3FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q3ScaleLoad::Pair32>;


template <class Schedule, class OutT = __nv_bfloat16>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);

    const dim3 grid(static_cast<unsigned>((rows + Schedule::kBlockRows - 1) / Schedule::kBlockRows),
                    static_cast<unsigned>((cols + Schedule::kBlockCols - 1) / Schedule::kBlockCols),
                    1u);
    const bool full = (rows % Schedule::kBlockRows) == 0 &&
                      (cols % Schedule::kBlockCols) == 0 && k == padded_k;
    if (full) {
        q3_rowsplit_gemm_mma_kernel<Schedule, true, OutT><<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<OutT*>(out.data), rows, k,
            cols, padded_k);
    } else {
        q3_rowsplit_gemm_mma_kernel<Schedule, false, OutT><<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<OutT*>(out.data), rows, k,
            cols, padded_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q3_mma_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_schedule<Q3MmaR32C64Schedule>(x, w, out, stream);
}

void launch_q3_mma_r32_c64_f32(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream) {
    launch_schedule<Q3MmaR32C64Schedule, float>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
