#include "ops/context_kv_materialize/q4_launch.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kLayers  = static_cast<int>(kContextKVMaterializeLayers);
constexpr int kHidden  = 5120;
constexpr int kRows    = 1024;
constexpr int kHeadDim = 128;

struct StoreLayerView {
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

__device__ __forceinline__ int context_column(int column, int width, int prefix) {
    return width == prefix ? column : column / prefix * width + column % prefix;
}

// Normalize and rotate one 128-wide key head, then write it at the ring slot for its absolute
// position. The math matches the fused W8 route; only the raw-key source representation differs.
__device__ __forceinline__ void store_key_head(const __nv_bfloat16* input, StoreLayerView layer,
                                               const std::int32_t* positions,
                                               const std::int32_t* slots, int column, int width,
                                               int head) {
    const int lane = threadIdx.x & 31;
    const int j    = lane * 2;
    float x0 = __bfloat162float(input[j]), x1 = __bfloat162float(input[j + 1]);
    float y0 = __bfloat162float(input[j + 64]), y1 = __bfloat162float(input[j + 65]);
    float sum           = warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / 128.0f + 1.e-6f);
    x0 *= inverse * __bfloat162float(layer.key_norm[j]);
    x1 *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    float sin0, cos0, sin1, cos1;
    dflash_rope_sincos(positions, column, j, &sin0, &cos0);
    dflash_rope_sincos(positions, column, j + 1, &sin1, &cos1);
    const auto dst = 128LL * ((positions[column] & 2047) + (long long)layer.padded_capacity *
                                                               (head + 8 * slots[column / width]));
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);
}

__global__ __launch_bounds__(256) void q4_context_key_store_kernel(
    const __nv_bfloat16* __restrict__ key, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ slots,
    StoreLayerView layer, int width, int batch, int min_count, int max_count) {
    const int packed_column = static_cast<int>(blockIdx.x);
    const int physical      = context_column(packed_column, width, max_count);
    const int request       = physical / width;
    const int count         = counts[request];
    if (count < min_count || count > max_count || (physical % width) >= count) return;
    const int head = threadIdx.x >> 5;
    const __nv_bfloat16* input =
        key + static_cast<std::int64_t>(physical) * kRows + head * kHeadDim;
    store_key_head(input, layer, positions, slots, physical, width, head);
}

__global__ __launch_bounds__(256) void q4_context_value_store_kernel(
    const __nv_bfloat16* __restrict__ value, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ slots,
    StoreLayerView layer, int width, int batch, int min_count, int max_count) {
    const int packed_column = static_cast<int>(blockIdx.x);
    const int physical      = context_column(packed_column, width, max_count);
    const int request       = physical / width;
    const int count         = counts[request];
    if (count < min_count || count > max_count || (physical % width) >= count) return;
    const auto* input = value + static_cast<std::int64_t>(physical) * kRows;
    for (int row = threadIdx.x; row < kRows; row += blockDim.x) {
        const auto dst = row % kHeadDim + static_cast<std::int64_t>(kHeadDim) *
                                              ((positions[physical] & 2047) +
                                               static_cast<std::int64_t>(layer.padded_capacity) *
                                                   (row / kHeadDim + 8 * slots[request]));
        layer.cache_v[dst] = __float2half_rn(__bfloat162float(input[row]));
    }
}

void store_layer_views(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    StoreLayerView (&out)[kLayers]) {
    for (int index = 0; index < kLayers; ++index) {
        const ContextKVMaterializeLayerView& source = layers[static_cast<std::size_t>(index)];
        out[index]                                  = {
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(source.cache.k.data),
            static_cast<__half*>(source.cache.v.data),
            static_cast<std::int32_t>(source.cache.padded_capacity),
        };
    }
}

} // namespace

void context_kv_materialize_q4_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch, cudaStream_t stream) {
    const int width   = context.ne[1];
    const int batch   = context.ne[2];
    const int columns = width * batch;
    StoreLayerView layer_views[kLayers];
    store_layer_views(layers, layer_views);
    const Tensor context_flat = context.view({kHidden, columns});
    Tensor scratch_flat       = scratch.view({kRows, columns});
    const dim3 store_grid(static_cast<unsigned>(envelope.max_count) * static_cast<unsigned>(batch));
    const auto* positions_data = static_cast<const std::int32_t*>(positions.data);
    const auto* counts_data    = static_cast<const std::int32_t*>(counts.data);
    const auto* slots_data     = static_cast<const std::int32_t*>(state_slots.data);
    for (int layer = 0; layer < kLayers; ++layer) {
        // The scratch holds one layer's projection at a time, so each store launch owns exactly
        // the layer whose projection it consumes.
        const ContextKVMaterializeLayerView& weights = layers[static_cast<std::size_t>(layer)];
        linear(context_flat, weights.key_weight, scratch_flat, stream);
        q4_context_key_store_kernel<<<store_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.data), positions_data, counts_data,
            slots_data, layer_views[layer], width, batch, static_cast<int>(envelope.min_count),
            static_cast<int>(envelope.max_count));
        linear(context_flat, weights.value_weight, scratch_flat, stream);
        q4_context_value_store_kernel<<<store_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.data), positions_data, counts_data,
            slots_data, layer_views[layer], width, batch, static_cast<int>(envelope.min_count),
            static_cast<int>(envelope.max_count));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
