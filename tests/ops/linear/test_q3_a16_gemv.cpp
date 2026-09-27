#include "ops/linear/linear_test_common.h"

#include "ops/linear/q3/q3_launch.h"
#include "ops/op_tester.h"

#include "ninfer/ops/linear.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

// The same deterministic BF16 operand pattern the shared oracle harness uses, so this suite
// compares on equivalent inputs.
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

// Runs `route` into a guarded buffer and reports a byte mismatch against the reference.
int compare_route(std::string_view label, const Tensor& x, const Weight& weight, std::int32_t n,
                  std::int32_t t, RouteLaunch route, const std::vector<std::uint16_t>& reference) {
    GuardedDeviceBuffer output(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
    Tensor destination(output.data(), DType::BF16, {n, t});
    route(x, weight, destination, nullptr);
    test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize q3 gemv route");
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

// The staged small-T route must reproduce the direct warp-per-row GEMV bit for bit: it decodes
// the same window values and accumulates them in the same per-output order, so no tolerance
// applies to this comparison.
int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    const quantized_weight::PackedWeight packed = make_q3g128_f16s_weight(n, k, seed);
    constexpr std::int32_t kMaxTokens           = 8;
    const std::vector<std::uint16_t> activation = make_activation_bits(k, kMaxTokens, seed + 1U);

    DeviceBuffer weight_device(packed.payload.size());
    weight_device.copy_from_host(packed.payload.data(), weight_device.bytes);
    const Weight weight = packed.device_weight(weight_device.p);
    DeviceBuffer activation_device(activation.size() * sizeof(std::uint16_t));
    activation_device.copy_from_host(activation.data(), activation_device.bytes);

    int failures = 0;
    for (std::int32_t t = 1; t <= kMaxTokens; ++t) {
        const std::string label =
            "Q3 gemv [n=" + std::to_string(n) + ",k=" + std::to_string(k) +
            ",padded=" + std::to_string(weight.padded_shape[1]) + "] T=" + std::to_string(t);
        const Tensor x(activation_device.p, DType::BF16, {k, t});

        GuardedDeviceBuffer reference_output(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
        Tensor reference(reference_output.data(), DType::BF16, {n, t});
        ops::detail::launch_q3_gemv_r8_c8(x, weight, reference, nullptr);
        test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize direct gemv reference");
        failures += reference_output.verify_guards(label + " direct reference");
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(n) * t);
        reference_output.copy_to_host(expected.data(), expected.size() * sizeof(std::uint16_t));

        failures +=
            compare_route(label + " staged", x, weight, n, t,
                          ops::detail::launch_q3_gemv_r8_c8_staged, expected);

        GuardedDeviceBuffer dispatch_output(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
        Tensor dispatch(dispatch_output.data(), DType::BF16, {n, t});
        ops::linear(x, weight, dispatch, nullptr);
        test::cuda_check(cudaStreamSynchronize(nullptr), "synchronize dispatched route");
        failures += dispatch_output.verify_guards(label + " dispatched route");
        std::vector<std::uint16_t> dispatched(static_cast<std::size_t>(n) * t);
        dispatch_output.copy_to_host(dispatched.data(), dispatched.size() * sizeof(std::uint16_t));
        if (dispatched != expected) {
            const auto mismatch = std::mismatch(dispatched.begin(), dispatched.end(),
                                                expected.begin(), expected.end());
            const std::size_t index = static_cast<std::size_t>(mismatch.first - dispatched.begin());
            std::cerr << label << " dispatched: byte mismatch at index " << index << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!test::cuda_unavailable()) {
        try {
            int failures = 0;

            // Every registered Q3 parent shape across the whole small-T route, including the
            // narrow body shape whose single CTA row block and the padded shape whose final
            // group is partially represented.
            failures += run_shape(1024, 5120, 211U);
            failures += run_shape(14336, 5120, 223U);
            failures += run_shape(16384, 5120, 227U);
            failures += run_shape(5120, 6144, 229U);
            failures += run_shape(34816, 5120, 233U);
            failures += run_shape(5120, 17408, 239U);
            failures += run_shape(4096, 4304, 241U);

            std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 staged GEMV\n";
            return failures == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "Q3_A16 staged GEMV: " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
}
