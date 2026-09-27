// Pure host-side qualification of the BF16 GDN gating projection's cooperative-launch bounds.
//
// The per-SM residency facts and the token-column ceilings they imply are the contract between the
// kernel launcher, the compile-time route catalog, and the documented prefill-chunk guidance. This
// test pins them for the reference 128-SM device and the 76-SM minimum supported device without
// touching a GPU.

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_residency.h"

#include <cstdint>
#include <iostream>
#include <string>

using namespace ninfer::ops::detail;

namespace {

int failures = 0;

void expect_eq(const std::string& label, std::int32_t actual, std::int32_t expected) {
    if (actual != expected) {
        std::cerr << label << ": expected " << expected << ", got " << actual << "\n";
        ++failures;
    }
}

std::int32_t columns(std::int32_t heads, std::int32_t input_rows, std::int32_t split_k,
                     std::int32_t tile_cols, std::int32_t row_tiles, std::int32_t sm_count) {
    return bf16_gdn_single_launch_columns(heads, input_rows, split_k, tile_cols, row_tiles, sm_count);
}

} // namespace

int main() {
    // Measured per-SM occupancy of the sm_89 build (cuobjdump -res-usage on the compiled cubins).
    // A toolchain change that invalidates these numbers must be caught here, before it becomes a
    // driver rejection on the first prefill wide enough to reach a whole-route cooperative grid.
    expect_eq("27B split8 CTAs/SM", bf16_gdn_resident_ctas_per_sm(48, 5120, 8), 2);
    expect_eq("27B split4 CTAs/SM", bf16_gdn_resident_ctas_per_sm(48, 5120, 4), 1);
    expect_eq("27B split2 CTAs/SM", bf16_gdn_resident_ctas_per_sm(48, 5120, 2), 1);
    expect_eq("35B split32 CTAs/SM", bf16_gdn_resident_ctas_per_sm(32, 2048, 32), 2);
    expect_eq("35B split16 CTAs/SM", bf16_gdn_resident_ctas_per_sm(32, 2048, 16), 4);
    expect_eq("35B split8 CTAs/SM", bf16_gdn_resident_ctas_per_sm(32, 2048, 8), 3);
    expect_eq("35B split4 CTAs/SM", bf16_gdn_resident_ctas_per_sm(32, 2048, 4), 3);
    expect_eq("35B split2 CTAs/SM", bf16_gdn_resident_ctas_per_sm(32, 2048, 2), 3);
    expect_eq("27B unsplit is not cooperative", bf16_gdn_resident_ctas_per_sm(48, 5120, 1), 0);
    expect_eq("unknown geometry is not cooperative", bf16_gdn_resident_ctas_per_sm(64, 7168, 8), 0);

    // Reference device (RTX 4090, 128 SMs): one whole-route cooperative launch. The 27B values are
    // the route endpoints in bf16_gdn_gating_proj_plan.cpp.
    expect_eq("27B split8 @128", columns(48, 5120, 8, 128, 3, 128), 1280);
    expect_eq("27B split2 @128", columns(48, 5120, 2, 128, 3, 128), 2688);
    expect_eq("35B split16 @128", columns(32, 2048, 16, 64, 2, 128), 1024);
    expect_eq("35B split8 @128", columns(32, 2048, 8, 64, 2, 128), 1536);
    expect_eq("35B split4 @128", columns(32, 2048, 4, 64, 2, 128), 3072);
    expect_eq("35B split2 @128", columns(32, 2048, 2, 64, 2, 128), 6144);
    expect_eq("35B split32 @128", columns(32, 2048, 32, 64, 2, 128), 256);

    // Minimum supported device (RTX 4080, 76 SMs): the launcher must partition a longer token range
    // into these slices instead of submitting one whole-route grid. The 27B split2 ceiling drops
    // from the reference 2688 columns to 1536, and split8 from 1280 to 768.
    expect_eq("27B split8 @76", columns(48, 5120, 8, 128, 3, 76), 768);
    expect_eq("27B split4 @76", columns(48, 5120, 4, 128, 3, 76), 768);
    expect_eq("27B split2 @76", columns(48, 5120, 2, 128, 3, 76), 1536);
    expect_eq("35B split32 @76", columns(32, 2048, 32, 64, 2, 76), 128);
    expect_eq("35B split16 @76", columns(32, 2048, 16, 64, 2, 76), 576);
    expect_eq("35B split8 @76", columns(32, 2048, 8, 64, 2, 76), 896);
    expect_eq("35B split4 @76", columns(32, 2048, 4, 64, 2, 76), 1792);
    expect_eq("35B split2 @76", columns(32, 2048, 2, 64, 2, 76), 3648);

    // Every cooperative route must seat at least one token tile on the minimum device; otherwise
    // the launcher falls back to the unsplit kernel, which is outside the registered accuracy
    // contract. These are the same bounds the compile-time catalog guard asserts.
    const auto seats_one_tile = [](std::int32_t heads, std::int32_t input_rows, std::int32_t split_k,
                                   std::int32_t tile_cols, std::int32_t row_tiles,
                                   std::int32_t sm_count) {
        return columns(heads, input_rows, split_k, tile_cols, row_tiles, sm_count) >= tile_cols;
    };
    expect_eq("27B split8 seats a tile @76", seats_one_tile(48, 5120, 8, 128, 3, 76), 1);
    expect_eq("27B split2 seats a tile @76", seats_one_tile(48, 5120, 2, 128, 3, 76), 1);
    expect_eq("35B split16 seats a tile @76", seats_one_tile(32, 2048, 16, 64, 2, 76), 1);
    expect_eq("35B split8 seats a tile @76", seats_one_tile(32, 2048, 8, 64, 2, 76), 1);
    expect_eq("35B split4 seats a tile @76", seats_one_tile(32, 2048, 4, 64, 2, 76), 1);
    expect_eq("35B split2 seats a tile @76", seats_one_tile(32, 2048, 2, 64, 2, 76), 1);
    expect_eq("35B split32 seats a tile @76", seats_one_tile(32, 2048, 32, 64, 2, 76), 1);
    expect_eq("non-cooperative has no ceiling", columns(48, 5120, 1, 128, 3, 128), 0);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_proj residency bounds\n";
    return failures == 0 ? 0 : 1;
}
