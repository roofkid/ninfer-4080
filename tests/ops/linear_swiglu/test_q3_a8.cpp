#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Token cases straddle the A16 projection-schedule boundaries and the documented A8
        // threshold at 129 columns, and cover one 64-token tile, several tiles and a partial last
        // tile under AllowA8.
        constexpr std::array<std::int32_t, 20> kTokenCases{
            1,  2,  3,   4,   5,   6,   7,   8,   9,   12,
            16, 17, 64,  65,  128, 129, 192, 256, 385, 513,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q3_A8",
            {QType::Q3G128_F16S, 34816, 5120, 17408, 1601U, ActivationCompute::A8},
            kTokenCases, std::array<std::int32_t, 3>{129, 192, 385});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q3_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q3_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
