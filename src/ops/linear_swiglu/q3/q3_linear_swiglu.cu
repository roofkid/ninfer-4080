#include "ops/linear_swiglu/q3/q3_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/q3/q3_launch.h"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = kGateUpRows / 2;
constexpr std::int32_t kChunkCols  = 64;
constexpr std::int32_t kGemvColumns = 8;

// proj is a column-major [2*rows, cols] FP32 plane; out is column-major [rows, cols] BF16.
__global__ void q3_swiglu_epilogue_kernel(const float* __restrict__ proj,
                                          __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                          std::int32_t cols) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * cols;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t row = static_cast<std::int32_t>(index % rows);
        const std::int32_t col = static_cast<std::int32_t>(index / rows);
        const float gate       = proj[static_cast<std::int64_t>(col) * (2 * rows) + row];
        const float up         = proj[static_cast<std::int64_t>(col) * (2 * rows) + rows + row];
        out[index]             = __float2bfloat16_rn(silu(gate) * up);
    }
}

// Fused decode route for the MTP width: one warp computes output row `row` from gate row `row`
// and up row `row + rows` of the [2*rows, K] parent. The K loop mirrors the qualified
// q3_rowsplit_gemv_kernel: two consecutive 128-code groups per iteration, the low half-warp on
// the even group and the high half-warp on the odd group, so every code byte is fetched once.
// Both rows reuse the activation reads and only lane 0 rounds silu(gate)*up to BF16, so decode
// never materializes the FP32 projection plane.
template <int MaxCols>
__global__ __launch_bounds__(256, 2) void q3_rowsplit_gemv_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    static_assert(MaxCols >= 1 && MaxCols <= 8);
    constexpr int kWarps      = 8;
    constexpr int kGroupSize  = Q3RowSplitStorage::kGroupK;
    constexpr int kGroupBytes = Q3RowSplitStorage::kCodeBytesPerGroup;

    const std::int32_t warp = static_cast<std::int32_t>(threadIdx.x) >> 5;
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x) & 31;
    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.x) * kWarps + warp;
    if (row >= rows) { return; }

    const std::int32_t groups_per_row = padded_k / kGroupSize;
    const std::uint8_t* gate_codes =
        codes + static_cast<std::int64_t>(row) * groups_per_row * kGroupBytes;
    const std::uint8_t* up_codes =
        gate_codes + static_cast<std::int64_t>(rows) * groups_per_row * kGroupBytes;
    const std::uint8_t* gate_scales = scales + static_cast<std::int64_t>(row) * groups_per_row * 2;
    const std::uint8_t* up_scales =
        gate_scales + static_cast<std::int64_t>(rows) * groups_per_row * 2;

    const std::int32_t group_lane = lane >> 4;
    const std::int32_t local      = lane & 15;
    const std::int32_t byte0      = 3 * local;
    const std::int32_t code_base  = 8 * local;

    float gate_accum[MaxCols];
    float up_accum[MaxCols];
#pragma unroll
    for (int c = 0; c < MaxCols; ++c) {
        gate_accum[c] = 0.0F;
        up_accum[c]   = 0.0F;
    }

    for (std::int32_t group = 0; group < groups_per_row; group += 2) {
        const std::int32_t active_group = group + group_lane;
        float gate_weights[8];
        float up_weights[8];
        std::int32_t kbase = 0;
        if (active_group < groups_per_row) {
            const std::uint8_t* gate_ptr = gate_codes + active_group * kGroupBytes;
            const std::uint8_t* up_ptr   = up_codes + active_group * kGroupBytes;
            const std::uint32_t gate_window =
                static_cast<std::uint32_t>(gate_ptr[byte0]) |
                (static_cast<std::uint32_t>(gate_ptr[byte0 + 1]) << 8) |
                (static_cast<std::uint32_t>(gate_ptr[byte0 + 2]) << 16);
            const std::uint32_t up_window =
                static_cast<std::uint32_t>(up_ptr[byte0]) |
                (static_cast<std::uint32_t>(up_ptr[byte0 + 1]) << 8) |
                (static_cast<std::uint32_t>(up_ptr[byte0 + 2]) << 16);
            const float gate_scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(gate_scales + active_group * 2)));
            const float up_scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(up_scales + active_group * 2)));
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                gate_weights[c] =
                    static_cast<float>(q3_signed_code((gate_window >> (3 * c)) & 0x7u)) * gate_scale;
                up_weights[c] =
                    static_cast<float>(q3_signed_code((up_window >> (3 * c)) & 0x7u)) * up_scale;
            }
            kbase = active_group * kGroupSize + code_base;
        } else {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                gate_weights[c] = 0.0F;
                up_weights[c]   = 0.0F;
            }
        }
#pragma unroll
        for (int col = 0; col < MaxCols; ++col) {
            if (col >= cols) { continue; }
            const uint4 packed =
                *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + kbase);
            const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
            const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
            const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&packed.z);
            const __nv_bfloat162 p3 = *reinterpret_cast<const __nv_bfloat162*>(&packed.w);
            const float2 f0 = __bfloat1622float2(p0);
            const float2 f1 = __bfloat1622float2(p1);
            const float2 f2 = __bfloat1622float2(p2);
            const float2 f3 = __bfloat1622float2(p3);
            gate_accum[col] +=
                gate_weights[0] * f0.x + gate_weights[1] * f0.y + gate_weights[2] * f1.x +
                gate_weights[3] * f1.y + gate_weights[4] * f2.x + gate_weights[5] * f2.y +
                gate_weights[6] * f3.x + gate_weights[7] * f3.y;
            up_accum[col] += up_weights[0] * f0.x + up_weights[1] * f0.y + up_weights[2] * f1.x +
                             up_weights[3] * f1.y + up_weights[4] * f2.x + up_weights[5] * f2.y +
                             up_weights[6] * f3.x + up_weights[7] * f3.y;
        }
    }

#pragma unroll
    for (int col = 0; col < MaxCols; ++col) {
        float gate = gate_accum[col];
        float up   = up_accum[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate += __shfl_down_sync(0xffffffffu, gate, offset);
            up += __shfl_down_sync(0xffffffffu, up, offset);
        }
        if (lane == 0 && col < cols) {
            out[static_cast<std::int64_t>(col) * rows + row] =
                __float2bfloat16_rn(silu(gate) * up);
        }
    }
}

} // namespace

std::size_t q3_linear_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q3 linear_swiglu workspace: invalid token interval");
    }
    if (max_tokens <= kGemvColumns) { return 0; }
    const std::int64_t columns = std::min<std::int32_t>(max_tokens, kChunkCols - 1);
    return static_cast<std::size_t>(2 * kOutputRows) * static_cast<std::size_t>(columns) *
           sizeof(float);
}

void q3_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    if (w.qtype != QType::Q3G128_F16S || w.n != kGateUpRows || w.k != 5120 ||
        w.padded_shape[1] != 5120) {
        throw std::invalid_argument("q3 linear_swiglu: unsupported weight");
    }
    const std::int32_t columns = x.ne[1];
    if (columns <= 0 || out.ne[1] != columns) {
        throw std::invalid_argument("q3 linear_swiglu: invalid token extent");
    }
    if (columns <= kGemvColumns) {
        const dim3 grid(static_cast<unsigned>((kOutputRows + 7) / 8), 1u, 1u);
        q3_rowsplit_gemv_swiglu_kernel<8><<<grid, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            kOutputRows, x.ne[0], columns, static_cast<std::int32_t>(w.padded_shape[1]));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto scope  = ws.scope();
    Tensor proj = ws.alloc(DType::FP32, {2 * kOutputRows, std::min(columns, kChunkCols)});
    constexpr int kThreads = 256;
    for (std::int32_t begin = 0; begin < columns; begin += kChunkCols) {
        const std::int32_t chunk = std::min(kChunkCols, columns - begin);
        const auto* x_offset =
            static_cast<const std::uint8_t*>(x.data) +
            static_cast<std::int64_t>(begin) * w.k * static_cast<std::int64_t>(sizeof(std::uint16_t));
        Tensor x_chunk(const_cast<std::uint8_t*>(x_offset), DType::BF16, {w.k, chunk});
        Tensor proj_chunk(proj.data, DType::FP32, {2 * kOutputRows, chunk});
        launch_q3_mma_r32_c64_f32(x_chunk, w, proj_chunk, stream);
        auto* out_offset =
            static_cast<__nv_bfloat16*>(out.data) +
            static_cast<std::int64_t>(begin) * kOutputRows;
        const std::int64_t total = static_cast<std::int64_t>(kOutputRows) * chunk;
        const int blocks         = static_cast<int>(std::min<std::int64_t>(
            (total + kThreads - 1) / kThreads, 4096));
        q3_swiglu_epilogue_kernel<<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(proj.data), out_offset, kOutputRows, chunk);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
