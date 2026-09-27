#pragma once

// The documented A8 activation of the RowSplit G64-group prefill GEMMs, emulated from its
// contract (src/ops/common/rowsplit_a8_quantize.h), not from a kernel: per token and 64-column
// group of a BF16 [K, T] activation, amax = max |x|, scale = amax / 127, q = rint(x * (127 /
// amax)) clamped to [-127, 127], in IEEE FP32 with round-to-nearest-even. The quantization
// boundary is an observable semantic boundary of those Ops, so their A8 oracles multiply the
// decoded weights by q * scale (exact in FP64) instead of by x.

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test {

// With an AllowA8 permission, the RowSplit Ops quantize their activation from this width on;
// narrower widths (decode and speculative verification) are A16.
inline constexpr std::int32_t kA8G64MinTokens = 129;

inline std::vector<double> a8_g64_dequantized(const std::vector<std::uint16_t>& activation,
                                              std::int32_t k, std::int32_t tokens) {
    if (k <= 0 || k % 64 != 0 || tokens <= 0 ||
        activation.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens)) {
        throw std::invalid_argument("a8_g64 reference: invalid activation extent");
    }
    std::vector<double> result(static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t group = 0; group < k / 64; ++group) {
            const std::size_t base = static_cast<std::size_t>(token) * k + group * 64;
            float values[64];
            float amax = 0.0F;
            for (int i = 0; i < 64; ++i) {
                values[i] = bf16_to_f32(activation[base + i]);
                amax      = std::max(amax, std::fabs(values[i]));
            }
            const volatile float inverse = amax > 0.0F ? 127.0F / amax : 0.0F;
            const volatile float scale   = amax / 127.0F;
            for (int i = 0; i < 64; ++i) {
                const volatile float product = values[i] * inverse;
                const float code = std::clamp(std::nearbyint(static_cast<float>(product)), -127.0F,
                                              127.0F);
                result[base + i] = static_cast<double>(code) * static_cast<double>(scale);
            }
        }
    }
    return result;
}

// The A8 route applies one FP32 (w_scale * activation_scale) product and fma per group; only the
// quantization is the named semantic boundary, so the remaining fp32 rounding keeps the A16
// criterion.
inline constexpr ReductionCriterion kDocumentedA8Criterion{1.0 / 256.0, 1.0 / 256.0,
                                                           2.0 / 256.0};

} // namespace ninfer::test
