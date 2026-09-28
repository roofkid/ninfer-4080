#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Token cases straddle the composed route's T=8/9 projection-schedule boundary and the
        // registered prefill tile boundaries without asserting any private selector.
        constexpr std::array<std::int32_t, 24> kTokenCases{
            1,  2,  3,   4,  5,   6,   7,  8,   9,   12,  16,
            17, 24, 25,  32, 33,  48,  64, 65,  96,  128, 129, 256, 513,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q3_A16",
            {QType::Q3G128_F16S, 34816, 5120, 17408, 1501U, ActivationCompute::A16},
            kTokenCases, std::array<std::int32_t, 4>{1, 9, 65, 128});
        // Normal-range multipliers exercise the BF16 weight rounding the fused small-T route
        // performs and the default fixture's tiny scales never reach.
        constexpr std::array<std::int32_t, 5> kUnitScaleTokens{1, 2, 4, 6, 8};
        const int unit_failures = run_profile(
            "LinearSwiGLU Q3_A16 unit scales",
            {QType::Q3G128_F16S, 34816, 5120, 17408, 1601U, ActivationCompute::A16,
             ninfer::test::quantized_weight::RowSplitScalePattern::Unit},
            kUnitScaleTokens);
        std::cout << (failures + unit_failures == 0 ? "OK" : "FAIL")
                  << " LinearSwiGLU Q3_A16 correctness\n";
        return failures + unit_failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q3_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
