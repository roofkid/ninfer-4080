#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int q3_a16_conformance() {
    int failures = 0;

    // A small registered shape checks the complete output, every route boundary, and the
    // predicated tails of both tiles, including the Q3 group padding at K beyond the logical
    // extent.
    constexpr std::array kN1024K5120{
        convenience(1), a16(2),  a16(3),  a16(4),  a16(5),  a16(6),  a16(7),  a16(8),
        a16(15),        a16(16), a16(17), a16(63), a16(64), a16(65), a16(66), a16(128),
    };
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {1024, 5120, 211U, Comparison::Full, true, kN1024K5120});

    // Every Q3 parent shape of the Qwen3.8-27B body. Large-N shapes sample the verified rows.
    constexpr std::array kBodyTokens{a16(1), a16(4), a16(8), a16(17), a16(64), a16(128)};
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {14336, 5120, 223U, Comparison::Sampled, false, kBodyTokens});
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {16384, 5120, 227U, Comparison::Sampled, false, kBodyTokens});
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {5120, 6144, 229U, Comparison::Sampled, false, kBodyTokens});
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {34816, 5120, 233U, Comparison::Sampled, false, kBodyTokens});
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {5120, 17408, 239U, Comparison::Sampled, false, kBodyTokens});

    // K is padded to a whole 128-code group rather than a whole code byte, so 4304 exercises a
    // partially populated final group.
    constexpr std::array kPaddedTokens{convenience(1), a16(3), a16(65), a16(128)};
    failures += run_shape("Q3_A16", ActivationCompute::A16, make_q3g128_f16s_weight,
                          {4096, 4304, 241U, Comparison::Sampled, false, kPaddedTokens});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q3_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q3_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
