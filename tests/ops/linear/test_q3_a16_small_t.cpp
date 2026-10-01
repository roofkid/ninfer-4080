// The small-T Q3 tensor-core route: registered route boundaries, and the A16 oracle on
// normal-range multipliers.
//
// The shared Q3 fixture's "Small" scale pattern is a set of tiny powers of two whose
// code * scale products are all exactly representable in BF16. A route that decodes its weights
// to BF16 therefore passes the oracle there without ever exercising that rounding, so this suite
// repeats the small-width oracle with the fixture's "Unit" pattern (0.5 .. 1.125, normal-range
// exponents) and, at the same time, pins which width each route owns.

#include "ops/linear/linear_test_common.h"

#include "ops/linear/q3/q3_dispatch.h"
#include "ops/linear/q3/q3_launch.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>

namespace {

namespace detail = ninfer::ops::detail;

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

int check_route_boundaries() {
    using detail::select_q3_a16_launch;
    int failures = 0;

    struct Case {
        std::int32_t n;
        std::int32_t k;
        std::int32_t padded_k;
        std::int32_t t;
        ninfer::ops::detail::Q3Launch expected;
        const char* label;
    };
    const std::array<Case, 10> cases{{
        {34816, 5120, 5120, 1, detail::launch_q3_gemv_r8_c8_staged, "T=1 stays on the staged GEMV"},
        {34816, 5120, 5120, 2, detail::launch_q3_mma_small_t_r32_c8, "T=2 takes the small-T MMA"},
        {34816, 5120, 5120, 4, detail::launch_q3_mma_small_t_r32_c8, "T=4 takes the small-T MMA"},
        {34816, 5120, 5120, 8, detail::launch_q3_mma_small_t_r32_c8, "T=8 takes the small-T MMA"},
        {34816, 5120, 5120, 9, detail::launch_q3_mma_small_t_r32_c8,
         "T=9 keeps the small-T MMA with two token tiles"},
        {34816, 5120, 5120, 16, detail::launch_q3_mma_small_t_r32_c8, "T=16 keeps the small-T MMA"},
        {34816, 5120, 5120, 17, detail::launch_q3_mma_r32_c64, "T=17 falls back to the staged 32x64"},
        {34816, 4224, 4224, 12, detail::launch_q3_mma_r32_c64,
         "a padded K beyond whole 256-code stages keeps the staged 32x64"},
        {34816, 4304, 4352, 12, detail::launch_q3_mma_small_t_r32_c8,
         "a padded final group still fits whole 256-code stages at two tiles"},
        {34816, 4224, 4224, 4, detail::launch_q3_gemv_r8_c8_staged,
         "a 128-code-padded tail keeps the GEMV"},
    }};
    for (const Case& item : cases) {
        const ninfer::ops::detail::Q3Launch selected =
            select_q3_a16_launch(item.n, item.k, item.padded_k, item.t);
        if (selected != item.expected) {
            std::cerr << "Q3 small-T route boundary: " << item.label << " is not selected\n";
            ++failures;
        }
    }
    return failures;
}

int small_t_oracle() {
    int failures = 0;

    // The whole small-T domain on one registered shape, with normal-range multipliers.
    constexpr std::array kSmallShapeTokens{a16(1),  a16(2),  a16(3),  a16(4),  a16(5),  a16(6),
                                           a16(7),  a16(8),  a16(9),  a16(10), a16(11), a16(12),
                                           a16(13), a16(14), a16(15), a16(16)};
    failures += run_shape("Q3_A16 small-T unit scales", ActivationCompute::A16,
                          make_q3g128_f16s_unit_weight,
                          {1024, 5120, 2609U, Comparison::Full, true, kSmallShapeTokens});

    // The widest registered parent at the decode and MTP verify widths, sampled.
    constexpr std::array kGateUpTokens{a16(1), a16(2), a16(4), a16(6), a16(8), a16(9), a16(12),
                                       a16(16)};
    failures += run_shape("Q3_A16 small-T unit scales gate_up", ActivationCompute::A16,
                          make_q3g128_f16s_unit_weight,
                          {34816, 5120, 2617U, Comparison::Sampled, false, kGateUpTokens});

    // A logical K that only pads to a whole 128-code group, with the route's own 256-code stage.
    constexpr std::array kPaddedTokens{a16(2), a16(4), a16(8), a16(9), a16(16)};
    failures += run_shape("Q3_A16 small-T unit scales padded", ActivationCompute::A16,
                          make_q3g128_f16s_unit_weight,
                          {4096, 4304, 2621U, Comparison::Sampled, false, kPaddedTokens});

    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = check_route_boundaries() + small_t_oracle();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 small-T MMA\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q3_A16 small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
