#include "ops/linear/linear_test_common.h"

#include "ops/linear/q3/q3_launch.h"
#include "ops/op_tester.h"

#include "ninfer/ops/linear.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

// The same deterministic BF16 operand pattern the shared oracle harness uses, so the two suites
// compare on equivalent inputs.
std::vector<std::uint16_t> make_activation_bits(std::int32_t k, std::int32_t t,
                                                std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(k) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t coordinate =
                seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            coordinate ^= coordinate >> 16;
            coordinate *= 0x7feb352dU;
            coordinate ^= coordinate >> 15;
            coordinate *= 0x846ca68bU;
            coordinate ^= coordinate >> 16;
            const int raw = static_cast<int>(coordinate & 0xffU);
            result[static_cast<std::size_t>(token) * k + column] =
                test::f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 256.0F));
        }
    }
    return result;
}

using RouteLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Writes `route` into a guarded buffer and byte-compares it with `reference`.
int compare_route(std::string_view label, const Tensor& x, const Weight& weight, std::int32_t n,
                  std::int32_t t, RouteLaunch route, const std::vector<std::uint16_t>& reference) {
    GuardedDeviceBuffer output(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
    Tensor destination(output.data(), DType::BF16, {n, t});
    route(x, weight, destination, nullptr);
    test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize q3 route");
    int failures = output.verify_guards(label);
    std::vector<std::uint16_t> actual(static_cast<std::size_t>(n) * t);
    output.copy_to_host(actual.data(), actual.size() * sizeof(std::uint16_t));
    if (actual != reference) {
        const auto mismatch =
            std::mismatch(actual.begin(), actual.end(), reference.begin(), reference.end());
        const std::size_t index = static_cast<std::size_t>(mismatch.first - actual.begin());
        std::cerr << label << ": byte mismatch at index " << index << " actual=0x" << std::hex
                  << actual[index] << " reference=0x" << reference[index] << std::dec << '\n';
        ++failures;
    }
    return failures;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed,
              std::span<const std::int32_t> tokens) {
    const quantized_weight::PackedWeight packed = make_q3g128_f16s_weight(n, k, seed);
    const std::int32_t max_t =
        *std::max_element(tokens.begin(), tokens.end());
    const std::vector<std::uint16_t> activation = make_activation_bits(k, max_t, seed + 1U);

    DeviceBuffer weight_device(packed.payload.size());
    weight_device.copy_from_host(packed.payload.data(), weight_device.bytes);
    const Weight weight = packed.device_weight(weight_device.p);
    DeviceBuffer activation_device(activation.size() * sizeof(std::uint16_t));
    activation_device.copy_from_host(activation.data(), activation_device.bytes);

    int failures = 0;
    for (const std::int32_t t : tokens) {
        const std::string label =
            "Q3 tall [n=" + std::to_string(n) + ",k=" + std::to_string(k) +
            ",padded=" + std::to_string(weight.padded_shape[1]) + "] T=" + std::to_string(t);
        const Tensor x(activation_device.p, DType::BF16, {k, t});

        GuardedDeviceBuffer reference_output(static_cast<std::size_t>(n) * t *
                                             sizeof(std::uint16_t));
        Tensor reference(reference_output.data(), DType::BF16, {n, t});
        ninfer::ops::detail::launch_q3_mma_r32_c64(x, weight, reference, nullptr);
        test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize staged reference");
        failures += reference_output.verify_guards(label + " staged reference");
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(n) * t);
        reference_output.copy_to_host(expected.data(), expected.size() * sizeof(std::uint16_t));

        failures += compare_route(label + " tall c64", x, weight, n, t,
                                  ninfer::ops::detail::launch_q3_mma_tall_r128_c64, expected);
        failures += compare_route(label + " tall c128", x, weight, n, t,
                                  ninfer::ops::detail::launch_q3_mma_tall_r128_c128, expected);

        // The dispatched route is the small-T tensor-core engine at T <= 16 (see
        // test_q3_a16_small_t.cpp for those widths), so the staged byte-equality check only
        // applies where the staged or tall engine owns the dispatch.
        if (t > 16) {
            GuardedDeviceBuffer dispatch_output(static_cast<std::size_t>(n) * t *
                                                sizeof(std::uint16_t));
            Tensor dispatch(dispatch_output.data(), DType::BF16, {n, t});
            ops::linear(x, weight, dispatch, nullptr);
            test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize dispatched route");
            failures += dispatch_output.verify_guards(label + " dispatched route");
            std::vector<std::uint16_t> dispatched(static_cast<std::size_t>(n) * t);
            dispatch_output.copy_to_host(dispatched.data(),
                                         dispatched.size() * sizeof(std::uint16_t));
            if (dispatched != expected) {
                const auto mismatch = std::mismatch(dispatched.begin(), dispatched.end(),
                                                    expected.begin(), expected.end());
                const std::size_t index =
                    static_cast<std::size_t>(mismatch.first - dispatched.begin());
                std::cerr << label << " dispatched: byte mismatch at index " << index << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!test::cuda_unavailable()) {
        try {
            int failures = 0;

            // The registered narrow body shape across the complete prefill boundary range.
            std::vector<std::int32_t> sweep;
            for (std::int32_t t = 9; t <= 513; t += 8) { sweep.push_back(t); }
            failures += run_shape(1024, 5120, 211U, sweep);

            // A wide body shape samples the 128-row block tiling and both token tiles.
            constexpr std::array<std::int32_t, 3> kWideTokens{64, 129, 512};
            failures += run_shape(14336, 5120, 223U, kWideTokens);

            // Logical K 4304 pads to 4352, so the final decoded 64-code step overlaps zero
            // activation columns and the K masking must match the staged route bit for bit.
            constexpr std::array<std::int32_t, 4> kPaddedTokens{9, 65, 129, 512};
            failures += run_shape(4096, 4304, 241U, kPaddedTokens);

            std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 tall GEMM\n";
            return failures == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "Q3_A16 tall GEMM: " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
}
