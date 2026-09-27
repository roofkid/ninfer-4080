#include "ops/common/rowsplit_a8_quantize.h"

#include "core/device.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

// Eight lanes per (token, group), eight values per lane; items run token-fastest so that the
// scales of a group are written contiguously.
__global__ void __launch_bounds__(kThreads)
    a8_g64_quantize_kernel(const __nv_bfloat16* __restrict__ x, std::int32_t k,
                           std::int32_t tokens, std::int64_t items, std::int8_t* __restrict__ q,
                           float* __restrict__ scale) {
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const std::int64_t item   = thread >> 3;
    const int part            = static_cast<int>(thread & 7);
    const bool live           = item < items;
    const std::int64_t index  = live ? item : 0;
    const int token           = static_cast<int>(index % tokens);
    const int group           = static_cast<int>(index / tokens);
    const std::int64_t offset = static_cast<std::int64_t>(token) * k + group * 64 + part * 8;

    const uint4 raw = load_ldg<uint4>(x + offset);
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    float value[8];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
        value[2 * i]      = pair.x;
        value[2 * i + 1]  = pair.y;
        amax              = fmaxf(amax, fmaxf(fabsf(pair.x), fabsf(pair.y)));
    }
#pragma unroll
    for (int offset_lanes = 1; offset_lanes < 8; offset_lanes <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset_lanes));
    }
    const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
    unsigned packed[2]  = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int code = max(-127, min(127, __float2int_rn(value[i] * inverse)));
        packed[i >> 2] |= (static_cast<unsigned>(code) & 0xffu) << (8 * (i & 3));
    }
    if (!live) { return; }
    *reinterpret_cast<uint2*>(q + offset) = make_uint2(packed[0], packed[1]);
    if (part == 0) { scale[static_cast<std::int64_t>(group) * tokens + token] = amax / 127.0f; }
}

} // namespace

void a8_g64_quantize(const Tensor& x, A8G64Activation& out, cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[2] != 1 || x.ne[3] != 1 || k <= 0 ||
        k % 64 != 0 || tokens <= 0 || out.q.dtype != DType::I8 || out.q.ne[0] != k ||
        out.q.ne[1] != tokens || out.scale.dtype != DType::FP32 || out.scale.ne[0] != tokens ||
        out.scale.ne[1] != k / 64 ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15) != 0 ||
        (reinterpret_cast<std::uintptr_t>(out.q.data) & 15) != 0) {
        throw std::invalid_argument("a8_g64_quantize: invalid activation or destination");
    }
    const std::int64_t items   = static_cast<std::int64_t>(tokens) * (k / 64);
    const std::int64_t threads = items * 8;
    const unsigned blocks      = static_cast<unsigned>((threads + kThreads - 1) / kThreads);
    a8_g64_quantize_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), k, tokens, items,
        static_cast<std::int8_t*>(out.q.data), static_cast<float*>(out.scale.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
