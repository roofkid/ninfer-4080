#include <ninfer/ops/mhc.h>

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kStreams        = 2;
constexpr int kRouterOutputs  = 2 * kStreams + kStreams * kStreams;
constexpr int kThreads        = 256;
constexpr float kNormEpsilon  = 1.0e-6F;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_down_sync(0xffffffff, value, offset); }
    return value;
}

__device__ __forceinline__ float sigmoid(float value) { return 1.0F / (1.0F + expf(-value)); }

__device__ __forceinline__ void sinkhorn_2x2(float (&logits)[2][2], int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int row = 0; row < 2; ++row) {
            const float maximum = fmaxf(logits[row][0], logits[row][1]);
            const float sum = expf(logits[row][0] - maximum) + expf(logits[row][1] - maximum);
            const float lse = maximum + logf(sum);
            logits[row][0] -= lse;
            logits[row][1] -= lse;
        }
        for (int column = 0; column < 2; ++column) {
            const float maximum = fmaxf(logits[0][column], logits[1][column]);
            const float sum = expf(logits[0][column] - maximum) + expf(logits[1][column] - maximum);
            const float lse = maximum + logf(sum);
            logits[0][column] -= lse;
            logits[1][column] -= lse;
        }
    }
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) { logits[row][column] = expf(logits[row][column]); }
    }
}

__global__ void mhc_router_kernel(const __nv_bfloat16* residual, const float* weight,
                                  const float* bias, int hidden, int tokens, int iterations,
                                  float pre_scale, __nv_bfloat16* branch, float* post, float* mix) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid & 31;
    const int warp         = tid >> 5;
    const int warps        = static_cast<int>(blockDim.x) >> 5;
    const int width        = kStreams * hidden;
    const __nv_bfloat16* x = residual + static_cast<std::int64_t>(token) * width;

    __shared__ float warp_sums[8];
    __shared__ float inverse_norm;
    __shared__ float logits[kRouterOutputs];
    __shared__ float pre[kStreams];
    __shared__ float post_shared[kStreams];
    __shared__ float mix_shared[kStreams][kStreams];

    float square_sum = 0.0F;
    for (int index = tid; index < width; index += kThreads) {
        const float value = __bfloat162float(x[index]);
        square_sum += value * value;
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) { warp_sums[warp] = square_sum; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0F;
        for (int index = 0; index < warps; ++index) { total += warp_sums[index]; }
        inverse_norm = rsqrtf(total / static_cast<float>(width) + kNormEpsilon);
    }
    __syncthreads();

    // Eight warps, one FP32 router row each.
    if (warp < kRouterOutputs) {
        const float* row = weight + static_cast<std::int64_t>(warp) * width;
        float dot        = 0.0F;
        for (int index = lane; index < width; index += 32) {
            dot += (__bfloat162float(x[index]) * inverse_norm) * row[index];
        }
        dot = warp_sum(dot);
        if (lane == 0) { logits[warp] = dot + bias[warp]; }
    }
    __syncthreads();

    if (tid == 0) {
        for (int stream = 0; stream < kStreams; ++stream) {
            pre[stream]          = sigmoid(logits[stream]);
            post_shared[stream]  = 2.0F * sigmoid(logits[kStreams + stream]);
        }
        float block[kStreams][kStreams];
        for (int row = 0; row < kStreams; ++row) {
            for (int column = 0; column < kStreams; ++column) {
                block[row][column] = logits[2 * kStreams + row * kStreams + column];
            }
        }
        sinkhorn_2x2(block, iterations);
        for (int row = 0; row < kStreams; ++row) {
            for (int column = 0; column < kStreams; ++column) {
                mix_shared[row][column] = block[row][column];
            }
        }
        for (int stream = 0; stream < kStreams; ++stream) {
            post[static_cast<std::int64_t>(stream) * tokens + token] = post_shared[stream];
        }
        for (int index = 0; index < kStreams * kStreams; ++index) {
            mix[static_cast<std::int64_t>(index) * tokens + token] =
                mix_shared[index / kStreams][index % kStreams];
        }
    }
    __syncthreads();

    float pre_bf16[kStreams];
    for (int stream = 0; stream < kStreams; ++stream) {
        pre_bf16[stream] = __bfloat162float(__float2bfloat16_rn(pre[stream]));
    }
    __nv_bfloat16* out = branch + static_cast<std::int64_t>(token) * hidden;
    for (int feature = tid; feature < hidden; feature += kThreads) {
        float value = 0.0F;
        for (int stream = 0; stream < kStreams; ++stream) {
            value += pre_bf16[stream] *
                     __bfloat162float(x[static_cast<std::int64_t>(stream) * hidden + feature]);
        }
        if (pre_scale != 1.0F) { value *= pre_scale; }
        out[feature] = __float2bfloat16_rn(value);
    }
}

__global__ void mhc_merge_kernel(const __nv_bfloat16* residual, const __nv_bfloat16* branch,
                                 const float* post, const float* mix, int hidden, int tokens,
                                 __nv_bfloat16* out) {
    const std::int64_t total = static_cast<std::int64_t>(kStreams) * hidden * tokens;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t token  = index / (kStreams * hidden);
        const int element         = static_cast<int>(index % (kStreams * hidden));
        const int stream          = element / hidden;
        const int feature         = element % hidden;
        const __nv_bfloat16* source = residual + token * (kStreams * hidden);
        float mixed = 0.0F;
        for (int other = 0; other < kStreams; ++other) {
            const float coefficient = __bfloat162float(
                __float2bfloat16_rn(mix[(static_cast<std::int64_t>(stream) * kStreams + other) *
                                            tokens +
                                        token]));
            mixed += coefficient *
                     __bfloat162float(source[static_cast<std::int64_t>(other) * hidden + feature]);
        }
        const float mixed_bf16 = __bfloat162float(__float2bfloat16_rn(mixed));
        const float post_bf16  = __bfloat162float(
            __float2bfloat16_rn(post[static_cast<std::int64_t>(stream) * tokens + token]));
        const float branch_value = __bfloat162float(branch[token * hidden + feature]);
        out[index] = __float2bfloat16_rn(mixed_bf16 + post_bf16 * branch_value);
    }
}


__global__ void mhc_enter_kernel(const __nv_bfloat16* hidden, int width, int tokens,
                                 __nv_bfloat16* streams) {
    const std::int64_t total = static_cast<std::int64_t>(width) * tokens;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t token  = index / width;
        const int feature         = static_cast<int>(index % width);
        streams[token * (kStreams * width) + feature] = hidden[index];
        streams[token * (kStreams * width) + width + feature] = hidden[index];
    }
}

__global__ void mhc_leave_kernel(const __nv_bfloat16* streams, int width, int tokens,
                                 __nv_bfloat16* hidden) {
    const std::int64_t total = static_cast<std::int64_t>(width) * tokens;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t token = index / width;
        const int feature        = static_cast<int>(index % width);
        const __nv_bfloat16* source = streams + token * (kStreams * width);
        float sum = 0.0F;
        for (int stream = 0; stream < kStreams; ++stream) {
            sum += __bfloat162float(source[stream * width + feature]);
        }
        hidden[index] = __float2bfloat16_rn(sum / static_cast<float>(kStreams));
    }
}

void require_bf16(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::BF16) { throw std::invalid_argument(std::string(label) + " must be BF16"); }
}

void require_fp32(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::FP32) { throw std::invalid_argument(std::string(label) + " must be FP32"); }
}

void require_contiguous(const Tensor& tensor, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
}

} // namespace

void mhc_router(const Tensor& residual, const Tensor& weight, const Tensor& bias, int iterations,
                float pre_scale, Tensor& branch, Tensor& post, Tensor& mix,
                cudaStream_t stream) {
    require_bf16(residual, "mHC residual");
    require_fp32(weight, "mHC router weight");
    require_fp32(bias, "mHC router bias");
    require_bf16(branch, "mHC branch");
    require_fp32(post, "mHC post");
    require_fp32(mix, "mHC mix");
    require_contiguous(residual, "mHC residual");
    require_contiguous(branch, "mHC branch");
    if (residual.ne[0] <= 0 || residual.ne[1] <= 0 || residual.ne[0] % kStreams != 0) {
        throw std::invalid_argument("mHC residual shape is invalid");
    }
    if (iterations < 1 || !std::isfinite(pre_scale) || pre_scale <= 0.0F) {
        throw std::invalid_argument("mHC router settings are invalid");
    }
    const int width  = residual.ne[0];
    const int hidden = width / kStreams;
    const int tokens = residual.ne[1];
    if (weight.ne[0] != kRouterOutputs || weight.ne[1] != width) {
        throw std::invalid_argument("mHC router weight shape is invalid");
    }
    if (bias.ne[0] != kRouterOutputs) { throw std::invalid_argument("mHC router bias shape is invalid"); }
    if (branch.ne[0] != hidden || branch.ne[1] != tokens) {
        throw std::invalid_argument("mHC branch shape is invalid");
    }
    if (post.ne[0] != kStreams || post.ne[1] != tokens || mix.ne[0] != kStreams * kStreams ||
        mix.ne[1] != tokens) {
        throw std::invalid_argument("mHC routing output shape is invalid");
    }
    mhc_router_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<const float*>(weight.data),
        static_cast<const float*>(bias.data), hidden, tokens, iterations, pre_scale,
        static_cast<__nv_bfloat16*>(branch.data), static_cast<float*>(post.data),
        static_cast<float*>(mix.data));
    CUDA_CHECK(cudaGetLastError());
}

void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post, const Tensor& mix,
               Tensor& out, cudaStream_t stream) {
    require_bf16(residual, "mHC merge residual");
    require_bf16(branch, "mHC merge branch");
    require_fp32(post, "mHC merge post");
    require_fp32(mix, "mHC merge mix");
    require_bf16(out, "mHC merge output");
    require_contiguous(residual, "mHC merge residual");
    require_contiguous(out, "mHC merge output");
    if (residual.ne[0] <= 0 || residual.ne[1] <= 0 || residual.ne[0] % kStreams != 0) {
        throw std::invalid_argument("mHC merge residual shape is invalid");
    }
    const int width  = residual.ne[0];
    const int hidden = width / kStreams;
    const int tokens = residual.ne[1];
    if (branch.ne[0] != hidden || branch.ne[1] != tokens || post.ne[0] != kStreams ||
        post.ne[1] != tokens || mix.ne[0] != kStreams * kStreams || mix.ne[1] != tokens ||
        out.ne[0] != width || out.ne[1] != tokens) {
        throw std::invalid_argument("mHC merge tensor shape is invalid");
    }
    const std::int64_t total = static_cast<std::int64_t>(width) * tokens;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (total + kThreads - 1) / kThreads, 65535));
    mhc_merge_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(branch.data), static_cast<const float*>(post.data),
        static_cast<const float*>(mix.data), hidden, tokens,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void mhc_enter(const Tensor& hidden, Tensor& streams, cudaStream_t stream) {
    require_bf16(hidden, "mHC enter hidden");
    require_bf16(streams, "mHC enter streams");
    require_contiguous(hidden, "mHC enter hidden");
    require_contiguous(streams, "mHC enter streams");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || streams.ne[0] != kStreams * hidden.ne[0] ||
        streams.ne[1] != hidden.ne[1]) {
        throw std::invalid_argument("mHC enter shape is invalid");
    }
    const std::int64_t total = static_cast<std::int64_t>(hidden.ne[0]) * hidden.ne[1];
    const int blocks = static_cast<int>(std::min<std::int64_t>((total + kThreads - 1) / kThreads,
                                                              65535));
    mhc_enter_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), hidden.ne[0], hidden.ne[1],
        static_cast<__nv_bfloat16*>(streams.data));
    CUDA_CHECK(cudaGetLastError());
}

void mhc_leave(const Tensor& streams, Tensor& hidden, cudaStream_t stream) {
    require_bf16(streams, "mHC leave streams");
    require_bf16(hidden, "mHC leave hidden");
    require_contiguous(streams, "mHC leave streams");
    require_contiguous(hidden, "mHC leave hidden");
    if (streams.ne[0] <= 0 || streams.ne[1] <= 0 || streams.ne[0] % kStreams != 0 ||
        hidden.ne[0] != streams.ne[0] / kStreams || hidden.ne[1] != streams.ne[1]) {
        throw std::invalid_argument("mHC leave shape is invalid");
    }
    const std::int64_t total = static_cast<std::int64_t>(hidden.ne[0]) * hidden.ne[1];
    const int blocks = static_cast<int>(std::min<std::int64_t>((total + kThreads - 1) / kThreads,
                                                              65535));
    mhc_leave_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(streams.data), hidden.ne[0], hidden.ne[1],
        static_cast<__nv_bfloat16*>(hidden.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
