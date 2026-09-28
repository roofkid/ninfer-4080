#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120;

using Launch = W8Launch;

template <int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
#if defined(NINFER_SM86)
    // Eight K-warps overflow the 48 KB sm_86/sm_89 static shared-memory cap from a 40-token tile
    // on (w8_small_t_mma.cuh); keep the 5090's 8-warp T<=40 route for the 4096-row input only
    // where it fits.
    constexpr int Warps = TileColumns <= 32 ? 8 : 4;
#else
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
#endif
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry            = W8LinearGeometry<kRows, InputRows>;
    using Schedule            = W8SmallTMmaSchedule<Warps, TileColumns, Warps == 8 ? 2 : 3,
                                                    W8SmallTMmaScaleAccess::Shared, Activation>;
    constexpr int SharedBytes = TileColumns > 64 ? sizeof(W8SmallTMmaSharedStorage<Schedule>) : 0;
    if constexpr (SharedBytes > 0) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            w8_small_t_mma_kernel<Geometry, TileColumns, Schedule, W8ContiguousOutput,
                                  W8SmallTMmaStoreEpilogue, W8SmallTMmaIdentityRows, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
        CUDA_CHECK(attribute);
    }
    const int columns = x.ne[1];
    W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    w8_small_t_mma_kernel<Geometry, TileColumns, Schedule, W8ContiguousOutput,
                          W8SmallTMmaStoreEpilogue, W8SmallTMmaIdentityRows, false, true>
        <<<grid, Schedule::kThreads, SharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaStoreEpilogue{},
            W8SmallTMmaIdentityRows{}, columns);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&tiled_projection<C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto attention = make_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp       = make_launchers<17408>(std::make_index_sequence<11>{});


void materialized(W8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    switch (schedule) {
    case W8DynamicConvAddSchedule::TiledMma: {
        const auto& launchers = x.ne[0] == 4096 ? attention : mlp;
        launchers[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    }
    case W8DynamicConvAddSchedule::MmaK128:
        launch_w8_mma_r64x32_c64_k128_a1(flat, weight, result, stream);
        break;
    }
    dynamic_conv_add_finish_launch(projected, base, delta, residual, x.ne[1], stream);
}
} // namespace

void w8_dynamic_grouped_conv_add_materialized_launch(W8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
