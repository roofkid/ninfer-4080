#pragma once

// Cooperative-launch residency of the BF16 GDN gating projection's split-K specializations.
//
// A cooperative launch requires the entire grid to be simultaneously resident, so both the launch
// decomposition in `bf16_gdn_gating_proj_kernels.cu` and the compile-time route catalog in
// `bf16_gdn_gating_proj_plan.cpp` need the same per-SM occupancy facts. They live here once so the
// runtime capacity and the route bounds cannot drift apart.
//
// The per-SM counts below are facts of the sm_89 build, measured with `cuobjdump -res-usage` on
// the compiled cubins (RTX 4080, CUDA 13.1):
//
//   geometry     split  threads  registers  smem     CTAs/SM  limiting resource
//   48x5120        8      256        65     40 KiB      2      shared memory
//   48x5120        4      512        74     40 KiB      1      registers
//   48x5120        2      512        74     40 KiB      1      registers
//   32x2048       32      256     88 / 126  24 KiB      2      registers (126 with fused norm)
//   32x2048       16      256        56     24 KiB      4      shared memory and registers
//   32x2048      8/4/2    256        74     24 KiB      3      registers
//
// The sm_89 register file, thread, and shared-memory limits are shared by every sm_89 device, so
// the per-SM counts carry over unchanged; only the device's SM count scales the device-wide
// budget. The launcher partitions independent token tiles when a whole-route grid exceeds the
// budget, so a route stays usable as long as one token tile is resident.

#include <cstdint>

namespace ninfer::ops::detail {

// RTX 4090: the device the route catalog and its chunk guidance were tuned on.
inline constexpr std::int32_t kBf16GdnReferenceMultiprocessorCount = 128;

// RTX 4080: the smallest sm_89 device this fork supports. The route catalog must remain launchable
// here through token-tile partitioning.
inline constexpr std::int32_t kBf16GdnMinimumMultiprocessorCount = 76;

// Resident CTAs per SM of one cooperative split-K specialization; 0 marks a specialization that is
// not launched cooperatively.
constexpr std::int32_t bf16_gdn_resident_ctas_per_sm(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t split_k) noexcept {
    if (heads == 48 && input_rows == 5120) {
        switch (split_k) {
        case 8:
            return 2;
        case 4:
        case 2:
            return 1;
        default:
            return 0;
        }
    }
    if (heads == 32 && input_rows == 2048) {
        switch (split_k) {
        case 32:
            return 2;
        case 16:
            return 4;
        case 8:
        case 4:
        case 2:
            return 3;
        default:
            return 0;
        }
    }
    return 0;
}

// Token-column capacity of one cooperative launch on a device with `sm_count` SMs: the number of
// token tiles (each `tile_cols` columns wide, `row_tiles * split_k` CTAs) that fit simultaneously,
// times the tile width. The launcher partitions a longer token range into slices of at most this
// many columns. 0 means not even one token tile is resident, which selects the non-cooperative
// fallback.
constexpr std::int32_t bf16_gdn_single_launch_columns(std::int32_t heads, std::int32_t input_rows,
                                                      std::int32_t split_k,
                                                      std::int32_t tile_cols,
                                                      std::int32_t row_tiles,
                                                      std::int32_t sm_count) noexcept {
    const std::int32_t per_sm = bf16_gdn_resident_ctas_per_sm(heads, input_rows, split_k);
    const std::int32_t ctas_per_token_tile = row_tiles * split_k;
    if (per_sm == 0 || ctas_per_token_tile == 0 || sm_count <= 0) { return 0; }
    const std::int32_t token_tiles = sm_count * per_sm / ctas_per_token_tile;
    return token_tiles == 0 ? 0 : token_tiles * tile_cols;
}

} // namespace ninfer::ops::detail
