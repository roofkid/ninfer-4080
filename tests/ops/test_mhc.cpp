#include "ninfer/ops/mhc.h"

#include "core/tensor.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Oracle {
    std::vector<double> branch;
    std::vector<double> post;
    std::vector<double> mix;
    std::vector<double> merged;
};

float bf16(float value) { return bf16_to_f32(f32_to_bf16(value)); }

double sinkhorn_sum(const double (&logits)[2][2], int row, int column, int iterations) {
    double block[2][2] = {{logits[0][0], logits[0][1]}, {logits[1][0], logits[1][1]}};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int r = 0; r < 2; ++r) {
            const double m = std::max(block[r][0], block[r][1]);
            const double lse = m + std::log(std::exp(block[r][0] - m) + std::exp(block[r][1] - m));
            block[r][0] -= lse;
            block[r][1] -= lse;
        }
        for (int c = 0; c < 2; ++c) {
            const double m = std::max(block[0][c], block[1][c]);
            const double lse = m + std::log(std::exp(block[0][c] - m) + std::exp(block[1][c] - m));
            block[0][c] -= lse;
            block[1][c] -= lse;
        }
    }
    return std::exp(block[row][column]);
}

Oracle oracle_router(const std::vector<float>& residual, const std::vector<float>& weight,
                     const std::vector<float>& bias, int hidden, int tokens, int iterations) {
    const int width = 2 * hidden;
    Oracle out;
    out.branch.resize(static_cast<std::size_t>(hidden) * tokens);
    out.post.resize(static_cast<std::size_t>(2) * tokens);
    out.mix.resize(static_cast<std::size_t>(4) * tokens);
    std::vector<double> normalized(width);
    double logits[8];
    for (int t = 0; t < tokens; ++t) {
        double sum = 0.0;
        for (int i = 0; i < width; ++i) {
            const double value = residual[static_cast<std::size_t>(t) * width + i];
            sum += value * value;
        }
        const double inv = 1.0 / std::sqrt(sum / width + 1e-6);
        for (int i = 0; i < width; ++i) {
            normalized[i] = residual[static_cast<std::size_t>(t) * width + i] * inv;
        }
        for (int o = 0; o < 8; ++o) {
            double dot = 0.0;
            for (int i = 0; i < width; ++i) {
                dot += normalized[i] * weight[static_cast<std::size_t>(o) * width + i];
            }
            logits[o] = dot + bias[o];
        }
        double pre[2];
        for (int s = 0; s < 2; ++s) {
            pre[s] = 1.0 / (1.0 + std::exp(-logits[s]));
            out.post[static_cast<std::size_t>(s) * tokens + t] =
                2.0 / (1.0 + std::exp(-logits[2 + s]));
        }
        double block[2][2] = {{logits[4], logits[5]}, {logits[6], logits[7]}};
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                out.mix[static_cast<std::size_t>(r * 2 + c) * tokens + t] =
                    sinkhorn_sum(block, r, c, iterations);
            }
        }
        for (int h = 0; h < hidden; ++h) {
            double value = 0.0;
            for (int s = 0; s < 2; ++s) {
                value += static_cast<double>(bf16(static_cast<float>(pre[s]))) *
                         residual[static_cast<std::size_t>(t) * width + s * hidden + h];
            }
            out.branch[static_cast<std::size_t>(t) * hidden + h] = bf16(static_cast<float>(value));
        }
    }
    return out;
}

std::vector<double> oracle_merge(const std::vector<float>& residual,
                                 const std::vector<double>& branch,
                                 const std::vector<double>& post, const std::vector<double>& mix,
                                 int hidden, int tokens) {
    std::vector<double> out(static_cast<std::size_t>(2 * hidden) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < 2; ++s) {
            for (int h = 0; h < hidden; ++h) {
                double mixed = 0.0;
                for (int j = 0; j < 2; ++j) {
                    mixed += static_cast<double>(bf16(static_cast<float>(
                                 mix[static_cast<std::size_t>(s * 2 + j) * tokens + t]))) *
                             residual[static_cast<std::size_t>(t) * 2 * hidden + j * hidden + h];
                }
                mixed = bf16(static_cast<float>(mixed));
                const double update =
                    static_cast<double>(bf16(static_cast<float>(
                        post[static_cast<std::size_t>(s) * tokens + t]))) *
                    branch[static_cast<std::size_t>(t) * hidden + h];
                out[static_cast<std::size_t>(t) * 2 * hidden + s * hidden + h] =
                    bf16(static_cast<float>(mixed + update));
            }
        }
    }
    return out;
}

int run(int hidden, int tokens, std::uint32_t seed) {
    const int width = 2 * hidden;
    std::vector<float> residual(static_cast<std::size_t>(width) * tokens);
    std::vector<float> weight(static_cast<std::size_t>(8) * width);
    std::vector<float> bias(8);
    fill_uniform(residual, seed, -3.0F, 3.0F);
    fill_uniform(weight, seed + 1, -0.2F, 0.2F);
    fill_uniform(bias, seed + 2, -0.5F, 0.5F);
    round_to_bf16(residual);
    // Weight/bias remain exact FP32 public inputs.

    DeviceBuffer d_residual = to_device_bf16(residual);
    DeviceBuffer d_weight = to_device_f32(weight);
    DeviceBuffer d_bias = to_device_f32(bias);
    DeviceBuffer d_branch(static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_post(static_cast<std::size_t>(2) * tokens * sizeof(float));
    DeviceBuffer d_mix(static_cast<std::size_t>(4) * tokens * sizeof(float));
    DeviceBuffer d_merged(static_cast<std::size_t>(width) * tokens * sizeof(std::uint16_t));

    Tensor residual_tensor(d_residual.p, DType::BF16, {width, tokens});
    Tensor weight_tensor(d_weight.p, DType::FP32, {8, width});
    Tensor bias_tensor(d_bias.p, DType::FP32, {8});
    Tensor branch_tensor(d_branch.p, DType::BF16, {hidden, tokens});
    Tensor post_tensor(d_post.p, DType::FP32, {2, tokens});
    Tensor mix_tensor(d_mix.p, DType::FP32, {4, tokens});
    Tensor merged_tensor(d_merged.p, DType::BF16, {width, tokens});

    ops::mhc_router(residual_tensor, weight_tensor, bias_tensor, 20, 1.0F, branch_tensor,
                    post_tensor, mix_tensor, nullptr);
    ops::mhc_merge(residual_tensor, branch_tensor, post_tensor, mix_tensor, merged_tensor,
                   nullptr);
    cuda_synchronize();

    const Oracle oracle = oracle_router(residual, weight, bias, hidden, tokens, 20);
    const auto device_branch = from_device_bf16(d_branch, oracle.branch.size());
    const auto device_post = from_device_f32(d_post, oracle.post.size());
    const auto device_mix = from_device_f32(d_mix, oracle.mix.size());
    const std::vector<double> merged =
        oracle_merge(residual, device_branch, device_post, device_mix, hidden, tokens);
    int failures = 0;
    const PointwiseCriterion criterion{1.0e-4, 3.0e-3};
    failures += verify_pointwise("mhc branch", device_branch, oracle.branch, criterion);
    failures += verify_pointwise("mhc post", device_post, oracle.post, criterion);
    failures += verify_pointwise("mhc mix", device_mix, oracle.mix, criterion);
    failures += verify_pointwise("mhc merged", from_device_bf16(d_merged, merged.size()), merged,
                                 criterion);
    return failures;
}

} // namespace


int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = 0;
    for (const int tokens : {1, 2, 8, 128, 1024}) { failures += run(4096, tokens, 7U + tokens); }
    failures += run(4096, 7, 99U);
    std::cout << (failures == 0 ? "mHC op tests passed" : "mHC op tests failed") << '\n';
    return failures == 0 ? 0 : 1;
}
