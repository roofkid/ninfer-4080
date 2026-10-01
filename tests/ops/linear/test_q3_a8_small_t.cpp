// The small-T Q3 A8 route: the int8 tensor-core engine for decode and verification widths, the
// A16 fallback at the route edges, and the documented-quantization oracle.
//
// The activation fixture has one 16x outlier per 64-column group, so an A16 execution cannot pass
// the documented-quantization criterion.

#include "ops/linear/linear_test_common.h"

#include "ops/a8_g64_reference.h"
#include "ops/linear/q3/q3_dispatch.h"
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
namespace detail = ninfer::ops::detail;

std::vector<std::uint16_t> make_a8_activation_bits(std::int32_t k, std::int32_t t,
                                                   std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(k) * t);
    for (std::size_t index = 0; index < result.size(); ++index) {
        std::uint32_t coordinate =
            seed ^ (static_cast<std::uint32_t>(index) * 0x9e3779b9U);
        coordinate ^= coordinate >> 16;
        coordinate *= 0x7feb352dU;
        coordinate ^= coordinate >> 15;
        coordinate *= 0x846ca68bU;
        coordinate ^= coordinate >> 16;
        const int raw = static_cast<int>(coordinate & 0xffU);
        result[index] = f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 256.0F));
        if (((index * 2654435761U) >> 7) & 1U) { result[index] ^= 0x8000U; }
        if (index % 64 == (index / 64) % 61) {
            result[index] = f32_to_bf16(16.0F * bf16_to_f32(result[index]));
        }
    }
    return result;
}

std::vector<std::int32_t> oracle_rows(std::int32_t n) {
    std::vector<std::int32_t> rows;
    if (n <= 1024) {
        rows.resize(static_cast<std::size_t>(n));
        for (std::int32_t row = 0; row < n; ++row) { rows[static_cast<std::size_t>(row)] = row; }
        return rows;
    }
    constexpr int kSamples = 32;
    for (int sample = 0; sample < kSamples; ++sample) {
        const std::int32_t row =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n - 1) * sample) / (kSamples - 1));
        if (std::find(rows.begin(), rows.end(), row) == rows.end()) { rows.push_back(row); }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

int check_route_boundaries() {
    using detail::kQ3SmallTMaxTokens;
    int failures = 0;
    const auto expect = [&](std::int32_t n, std::int32_t k, std::int32_t padded_k,
                            std::int32_t t, bool a8, const char* label) {
        if (detail::q3_uses_a8(n, k, padded_k, ops::LinearPolicy::AllowA8, t) != a8) {
            std::cerr << "Q3 small-T A8 route boundary: " << label << " is not selected\n";
            ++failures;
        }
    };
    // The single-token decode stays on the A16 GEMV, 2..16 columns run the small-T A8 engine, the
    // 17..128 window keeps the A16 routes, and 129+ keeps the tall A8 engine.
    expect(34816, 5120, 5120, 1, false, "T=1 stays A16");
    expect(34816, 5120, 5120, 2, true, "T=2 takes the small-T A8 engine");
    expect(34816, 5120, 5120, 8, true, "T=8 takes the small-T A8 engine");
    expect(34816, 5120, 5120, kQ3SmallTMaxTokens, true, "T=16 takes the small-T A8 engine");
    expect(34816, 5120, 5120, kQ3SmallTMaxTokens + 1, false, "T=17 stays A16");
    expect(34816, 5120, 5120, 128, false, "T=128 stays A16");
    expect(34816, 5120, 5120, 129, true, "T=129 takes the tall A8 engine");
    // The small-T A8 stage is 512 codes, so a K that is not a whole number of stages stays A16.
    expect(34816, 4992, 4992, 4, false, "a K outside whole 512-code stages stays A16");
    // Padded weights never take an A8 route.
    expect(4096, 4304, 4352, 4, false, "a padded weight stays A16");
    // A16Only never admits A8.
    if (detail::q3_uses_a8(34816, 5120, 5120, ops::LinearPolicy::A16Only, 4)) {
        std::cerr << "Q3 small-T A8 route boundary: A16Only admitted A8\n";
        ++failures;
    }
    return failures;
}

int run_shape(std::string_view label, std::int32_t n, std::int32_t k, std::uint32_t seed,
              std::span<const std::int32_t> a8_tokens, std::span<const std::int32_t> a16_tokens) {
    std::int32_t max_t = 0;
    for (const std::int32_t t : a8_tokens) { max_t = std::max(max_t, t); }
    for (const std::int32_t t : a16_tokens) { max_t = std::max(max_t, t); }
    if (max_t <= 0) { throw std::invalid_argument("q3 small-t a8 test: no token cases"); }
    const auto rows              = oracle_rows(n);
    const auto packed            = make_q3g128_f16s_weight(n, k, seed);
    const std::vector<float> oracle_weight =
        quantized_weight::materialize_rows_fp32(packed, rows);
    const std::vector<std::uint16_t> activation = make_a8_activation_bits(k, max_t, seed + 1U);

    DeviceBuffer weight_device(packed.payload.size());
    weight_device.copy_from_host(packed.payload.data(), weight_device.bytes);
    const Weight weight = packed.device_weight(weight_device.p);
    DeviceBuffer activation_device(activation.size() * sizeof(std::uint16_t));
    activation_device.copy_from_host(activation.data(), activation_device.bytes);

    // The documented quantization is computed once over the full width and then sliced per case.
    const bool quantizable = !a8_tokens.empty() && (k % 64) == 0;
    const std::vector<double> quantized =
        quantizable ? a8_g64_dequantized(activation, k, max_t) : std::vector<double>{};
    int failures = 0;
    const auto run_case = [&](std::int32_t t, bool expect_a8) {
        const std::string case_label =
            std::string(label) + " [" + std::to_string(n) + "," + std::to_string(k) +
            "] T=" + std::to_string(t) + (expect_a8 ? " A8" : " A16");
        const std::size_t elements = static_cast<std::size_t>(n) * t;
        GuardedDeviceBuffer output(elements * sizeof(std::uint16_t));
        Tensor x(activation_device.p, DType::BF16, {k, t});
        Tensor destination(output.data(), DType::BF16, {n, t});

        std::vector<float> oracle_activation(static_cast<std::size_t>(k) * t);
        if (expect_a8) {
            for (std::size_t index = 0; index < oracle_activation.size(); ++index) {
                oracle_activation[index] = static_cast<float>(quantized[index]);
            }
        } else {
            for (std::size_t index = 0; index < oracle_activation.size(); ++index) {
                oracle_activation[index] = bf16_to_f32(activation[index]);
            }
        }
        std::vector<double> reference(static_cast<std::size_t>(rows.size()) * t);
        cpu_linear_gemm_fp64(oracle_weight.data(), oracle_activation.data(), reference.data(),
                             static_cast<std::int32_t>(rows.size()), k, t);

        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            QType::Q3G128_F16S, n, k, ops::LinearPolicy::AllowA8, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        try {
            ops::linear(x, weight, destination, ops::LinearPolicy::AllowA8, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize q3 small-t a8 route");
            failures += output.verify_guards(case_label.c_str());
            if (workspace.used() != 0 || workspace.peak_used() > capacity) {
                std::cerr << case_label << ": workspace query/execution mismatch\n";
                ++failures;
            }
            std::vector<std::uint16_t> bits(elements);
            output.copy_to_host(bits.data(), elements * sizeof(std::uint16_t));
            std::vector<double> actual(elements);
            for (std::size_t index = 0; index < elements; ++index) {
                actual[index] = static_cast<double>(bf16_to_f32(bits[index]));
            }
            // Sample the same rows for large N.
            std::vector<double> selected(static_cast<std::size_t>(rows.size()) * t);
            for (std::size_t column = 0; column < static_cast<std::size_t>(t); ++column) {
                for (std::size_t row = 0; row < rows.size(); ++row) {
                    selected[column * rows.size() + row] =
                        actual[column * static_cast<std::size_t>(n) +
                               static_cast<std::size_t>(rows[row])];
                }
            }
            failures += verify_reduction(case_label, selected, reference, kDocumentedA8Criterion);
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
    };

    for (const std::int32_t t : a8_tokens) { run_case(t, true); }
    for (const std::int32_t t : a16_tokens) { run_case(t, false); }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = check_route_boundaries();
        // The whole small-T A8 domain on one registered shape.
        constexpr std::array kSmallA8{2, 3, 4, 8, 9, 12, 16};
        constexpr std::array kSmallA16{1, 17, 128};
        failures += run_shape("Q3 small-T A8", 1024, 5120, 611U, kSmallA8, kSmallA16);

        // The registered body parents at the decode, MTP and DFlash2 windows.
        constexpr std::array kBodyA8{2, 4, 8, 16};
        constexpr std::array kBodyA16{1};
        failures += run_shape("Q3 small-T A8", 34816, 5120, 613U, kBodyA8, kBodyA16);
        failures += run_shape("Q3 small-T A8", 7168, 5120, 617U, kBodyA8, kBodyA16);
        failures += run_shape("Q3 small-T A8", 12288, 5120, 619U, kBodyA8, kBodyA16);
        failures += run_shape("Q3 small-T A8", 5120, 6144, 623U, kBodyA8, kBodyA16);
        failures += run_shape("Q3 small-T A8", 5120, 17408, 631U, kBodyA8, kBodyA16);

        // A padded K is not an exact-K problem, so every width stays A16 even under AllowA8.
        const std::array<std::int32_t, 0> kPaddedA8{};
        constexpr std::array kPaddedA16{1, 2, 4, 8, 16, 129};
        failures += run_shape("Q3 small-T A8", 4096, 4304, 641U, kPaddedA8, kPaddedA16);

        // 4864 is a whole number of 256-code stages but not of the small-T A8 route's 512.
        const std::array<std::int32_t, 0> kStageA8{};
        constexpr std::array kStageA16{2, 4, 8, 16};
        failures += run_shape("Q3 small-T A8", 1024, 4864, 643U, kStageA8, kStageA16);

        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3 small-T A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q3 small-T A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
