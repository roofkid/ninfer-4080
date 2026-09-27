#pragma once

// Q3G128_F16S RowSplit x BF16 staged small-T GEMV.
//
// out[Cols, Rows] = W[Rows, K] * x[K, Cols] for one to eight token columns.
//
// One warp owns its weight rows' full K extent and walks it in stages of eight 128-code groups.
// The code bytes of the next stages are staged into shared memory with cp.async, so the DRAM
// latency of the weight stream is covered by the pipeline instead of by the consuming
// instructions, and every eight-code window is decoded from shared memory. Each lane owns window
// `l` (`lane & 15`) of a group: bytes 3l..3l+2 of the group's 48-byte code block. The lane
// decodes its window once, keeps the weights in registers, and applies the columns' activation
// values to them, so one activation read serves a whole window and the decode is not repeated per
// column.
//
// The 32-lane warp covers two adjacent groups per iteration, lane half selecting the group:
//
//   * plain (`RowsPerLane == 1`): one row per warp. Each lane decodes the same window of its
//     half's group and every lane of a half warp accumulates a different window of that row.
//   * fused (`RowsPerLane == 2`): the gate row and its matching up row of one output row. Each
//     lane decodes the same window of both rows, so the two rows share one activation read and
//     the epilogue publishes silu(gate) * up with a single BF16 rounding.
//
// Groups are consumed in ascending K, which keeps the plain route's per-output accumulation
// order identical to the direct warp-per-row GEMV it replaces - a lane accumulates its window of
// groups 0, 2, 4, ... - and one warp reduction publishes each output with the Op's single BF16
// rounding.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::q3_gemv_staged {

constexpr int kWarpsPerCta = 8;
constexpr int kThreads     = kWarpsPerCta * 32;
constexpr int kGroupK      = Q3RowSplitStorage::kGroupK;
constexpr int kGroupBytes  = Q3RowSplitStorage::kCodeBytesPerGroup;

// Production schedule: stages of eight 128-code groups over a three-deep cp.async pipeline.
constexpr int kProductionGroupsPerStage = 8;
constexpr int kProductionPipelineStages = 3;

// Warp-private staging: [warp][row in stage][pipeline stage][16-byte code vector].
template <int StageRows_, int GroupsPerStage_, int PipelineStages_>
struct StageTiles {
    static constexpr int kStageRows      = StageRows_;
    static constexpr int kGroupsPerStage = GroupsPerStage_;
    static constexpr int kPipelineStages = PipelineStages_;
    static constexpr int kVecsPerStageRow =
        GroupsPerStage_ * kGroupBytes / static_cast<int>(sizeof(uint4));
    static_assert(kVecsPerStageRow * static_cast<int>(sizeof(uint4)) ==
                  GroupsPerStage_ * kGroupBytes);
    static_assert(GroupsPerStage_ % 2 == 0, "a stage must hold whole group pairs");
    static_assert(PipelineStages_ >= 2 && PipelineStages_ <= 7,
                  "the staged GEMV pipeline depth must fit cp.async wait-group immediates");

    __align__(16) uint4 codes[kWarpsPerCta][StageRows_][PipelineStages_][kVecsPerStageRow];
};

// One eight-code window: three consecutive code bytes of one 128-code group.
__device__ __forceinline__ std::uint32_t q3_gemv_window(const std::uint8_t* window_bytes) {
    return static_cast<std::uint32_t>(window_bytes[0]) |
           (static_cast<std::uint32_t>(window_bytes[1]) << 8) |
           (static_cast<std::uint32_t>(window_bytes[2]) << 16);
}

// Decodes one eight-code window and multiplies it by the group's FP16 scale, exactly as the
// direct route does.
__device__ __forceinline__ void q3_gemv_decode_window(const std::uint8_t* window_bytes,
                                                      std::uint16_t scale_bits,
                                                      float (&weights)[8]) {
    const std::uint32_t window = q3_gemv_window(window_bytes);
    const float scale          = __half2float(__ushort_as_half(scale_bits));
#pragma unroll
    for (int code = 0; code < 8; ++code) {
        weights[code] = static_cast<float>(q3_signed_code((window >> (3 * code)) & 0x7u)) * scale;
    }
}

// Stages the next `kGroupsPerStage` groups of the warp's rows. Groups beyond the row's padded
// extent and rows beyond the weight extent are zero filled, so a partial final stage decodes to
// zero weights instead of reading out of bounds.
template <class Tiles>
__device__ __forceinline__ void q3_gemv_issue_stage(
    Tiles& tiles, int warp, int buffer, const std::uint8_t* const* row_codes,
    const bool* row_live, std::int32_t stage_group0, std::int32_t groups_per_row,
    const std::uint8_t* safe_source, int lane) {
    constexpr int kRows           = Tiles::kStageRows;
    constexpr int kVecsPerStageRow = Tiles::kVecsPerStageRow;
#pragma unroll
    for (int item = lane; item < kRows * kVecsPerStageRow; item += 32) {
        const int row  = item / kVecsPerStageRow;
        const int slot = item - row * kVecsPerStageRow;
        const std::int32_t group = stage_group0 + slot / 3;
        uint4* destination       = &tiles.codes[warp][row][buffer][slot];
        if (row_live[row] && group < groups_per_row) {
            cp_async<16, Cache::cg>(destination,
                                    row_codes[row] + static_cast<std::int64_t>(group) * kGroupBytes +
                                        (slot % 3) * 16);
        } else {
            cp_async_zfill<16>(destination, safe_source, 0);
        }
    }
    cp_commit();
}

// Consumes one stage: the lane half selects the group of the pair, and each lane decodes its
// window of one row (plain) or of the gate/up pair (fused). One activation window then feeds
// every row the lane owns.
template <int MaxCols, bool FullK, int RowsPerLane, class Tiles>
__device__ __forceinline__ void q3_gemv_consume_stage(
    const Tiles& tiles, int warp, int buffer, std::int32_t stage_group0,
    std::int32_t groups_per_row, const std::uint16_t* const* row_scales,
    const __nv_bfloat16* __restrict__ x, std::int32_t k, std::int32_t cols, int lane,
    float (&accumulator)[RowsPerLane][MaxCols]) {
    constexpr int kGroupsPerStage = Tiles::kGroupsPerStage;
    const int half      = lane >> 4;
    const int window    = lane & 15;
    const int byte0     = 3 * window;
    const int code_base = 8 * window;

#pragma unroll
    for (int iteration = 0; iteration < kGroupsPerStage / 2; ++iteration) {
        const int stage_group    = iteration * 2 + half;
        const std::int32_t group = stage_group0 + stage_group;
        if (group >= groups_per_row) { continue; }
        const std::int32_t kbase = group * kGroupK + code_base;

        float weights[RowsPerLane][8];
#pragma unroll
        for (int row = 0; row < RowsPerLane; ++row) {
            const std::uint8_t* group_bytes =
                reinterpret_cast<const std::uint8_t*>(tiles.codes[warp][row][buffer]) +
                stage_group * kGroupBytes + byte0;
            q3_gemv_decode_window(group_bytes, row_scales[row][group], weights[row]);
        }

#pragma unroll
        for (int col = 0; col < MaxCols; ++col) {
            if (col >= cols) { continue; }
            if constexpr (FullK) {
                const uint4 packed =
                    *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + kbase);
                const float2 f0 = bf16x2_bits_to_float2(packed.x);
                const float2 f1 = bf16x2_bits_to_float2(packed.y);
                const float2 f2 = bf16x2_bits_to_float2(packed.z);
                const float2 f3 = bf16x2_bits_to_float2(packed.w);
#pragma unroll
                for (int row = 0; row < RowsPerLane; ++row) {
                    const float* w = weights[row];
                    accumulator[row][col] +=
                        w[0] * f0.x + w[1] * f0.y + w[2] * f1.x + w[3] * f1.y + w[4] * f2.x +
                        w[5] * f2.y + w[6] * f3.x + w[7] * f3.y;
                }
            } else {
                const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(col) * k + kbase;
                float partial[RowsPerLane];
#pragma unroll
                for (int row = 0; row < RowsPerLane; ++row) { partial[row] = 0.0F; }
#pragma unroll
                for (int code = 0; code < 8; ++code) {
                    if (kbase + code < k) {
                        const float activation = __bfloat162float(xrow[code]);
#pragma unroll
                        for (int row = 0; row < RowsPerLane; ++row) {
                            partial[row] += weights[row][code] * activation;
                        }
                    }
                }
#pragma unroll
                for (int row = 0; row < RowsPerLane; ++row) {
                    accumulator[row][col] += partial[row];
                }
            }
        }
    }
}

template <int MaxCols, bool FullK, bool Fused, int GroupsPerStage, int PipelineStages,
          int LaunchBoundsMinBlocks>
__device__ __forceinline__ void q3_gemv_run(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t out_rows, std::int32_t k, std::int32_t cols, std::int32_t padded_k,
    std::int32_t pair_stride, std::int32_t weight_rows) {
    using Tiles                = StageTiles<Fused ? 2 : 1, GroupsPerStage, PipelineStages>;
    constexpr int kRowsPerLane = Tiles::kStageRows;
    constexpr int kStageRows   = Tiles::kStageRows;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ Tiles tiles;

    const std::int32_t groups_per_row = padded_k / kGroupK;
    const std::int32_t row_a          = static_cast<std::int32_t>(blockIdx.x) * kWarpsPerCta + warp;
    const std::int32_t row_b          = row_a + pair_stride;
    bool live[kStageRows];
    live[0] = row_a < out_rows;
    if constexpr (Fused) { live[1] = row_b < weight_rows; }
    std::int32_t rows[kStageRows];
    rows[0] = row_a;
    if constexpr (Fused) { rows[1] = row_b; }

    const std::uint8_t* row_codes[kStageRows];
    const std::uint16_t* row_scales[kStageRows];
#pragma unroll
    for (int row = 0; row < kStageRows; ++row) {
        row_codes[row] = codes + static_cast<std::int64_t>(live[row] ? rows[row] : 0) *
                                    groups_per_row * kGroupBytes;
        row_scales[row] = reinterpret_cast<const std::uint16_t*>(scales) +
                          static_cast<std::int64_t>(live[row] ? rows[row] : 0) * groups_per_row;
    }

    float accumulator[kRowsPerLane][MaxCols];
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
#pragma unroll
        for (int col = 0; col < MaxCols; ++col) { accumulator[row][col] = 0.0F; }
    }

    const std::int32_t stages = (groups_per_row + GroupsPerStage - 1) / GroupsPerStage;

#pragma unroll
    for (int prefetch = 0; prefetch < PipelineStages - 1; ++prefetch) {
        if (prefetch < stages) {
            q3_gemv_issue_stage(tiles, warp, prefetch % PipelineStages, row_codes, live,
                                prefetch * GroupsPerStage, groups_per_row, codes, lane);
        } else {
            cp_commit();
        }
    }

    for (std::int32_t stage = 0; stage < stages; ++stage) {
        const std::int32_t fetch = stage + PipelineStages - 1;
        if (fetch < stages) {
            q3_gemv_issue_stage(tiles, warp, fetch % PipelineStages, row_codes, live,
                                fetch * GroupsPerStage, groups_per_row, codes, lane);
        } else {
            cp_commit();
        }
        cp_wait<PipelineStages - 1>();
        __syncwarp();
        q3_gemv_consume_stage<MaxCols, FullK, kRowsPerLane>(
            tiles, warp, stage % PipelineStages, stage * GroupsPerStage, groups_per_row,
            row_scales, x, k, cols, lane, accumulator);
        __syncwarp();
    }

#pragma unroll
    for (int col = 0; col < MaxCols; ++col) {
        if (col >= cols) { break; }
        float gate = accumulator[0][col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate += __shfl_down_sync(0xffffffffu, gate, offset);
        }
        if constexpr (Fused) {
            float up = accumulator[1][col];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                up += __shfl_down_sync(0xffffffffu, up, offset);
            }
            if (lane == 0 && live[0]) {
                out[static_cast<std::int64_t>(col) * out_rows + row_a] =
                    __float2bfloat16_rn(silu(gate) * up);
            }
        } else {
            if (lane == 0 && live[0]) {
                out[static_cast<std::int64_t>(col) * out_rows + row_a] = __float2bfloat16(gate);
            }
        }
    }
}

template <int MaxCols, bool FullK, bool Fused, int GroupsPerStage, int PipelineStages,
          int LaunchBoundsMinBlocks>
__global__ __launch_bounds__(kThreads, LaunchBoundsMinBlocks) void q3_gemv_staged_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t out_rows, std::int32_t k, std::int32_t cols, std::int32_t padded_k,
    std::int32_t pair_stride, std::int32_t weight_rows) {
    q3_gemv_run<MaxCols, FullK, Fused, GroupsPerStage, PipelineStages, LaunchBoundsMinBlocks>(
        x, codes, scales, out, out_rows, k, cols, padded_k, pair_stride, weight_rows);
}

} // namespace ninfer::ops::detail::q3_gemv_staged
