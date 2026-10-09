#pragma once

// Q3G128_F16S RowSplit x int8 activation (A8) small-T tensor-core GEMM for one to sixteen token
// columns.
//
// Semantics. The activation arrives quantized by rowsplit_a8_quantize: per token and 64-code
// group, scale = amax / 127 and q = rint(x * (127 / amax)) clamped to [-127, 127] in FP32. A
// weight code c in [-4, 3] is an exact int8 MMA operand, so each 64-code group's integer dot
// product d_g = sum(c * q) is exact in int32 (one m16n8k32 per 32-code half, chained into the
// same accumulator, seeded with the integer-magic bias so d_g is one FADD away from the sum's
// bits). The output is sum_g (w_scale_128(g / 2) * scale_g) * d_g with one FP32 product per group
// and one fused multiply-add per group, before the Problem's single BF16 output rounding. Each of
// the eight warps owns one 64-code activation group of every 512-code stage, so the per-warp
// group order is ascending in K and the final step is the same eight-way partial tree the A16
// small-T route uses.
//
// Structure. One CTA owns 32 weight rows and one token tile, 8 columns for the decode and MTP
// verify widths and a native 16 columns for the DFlash2 and wide n-gram verify windows. A stage is
// 512 codes: four 128-code weight groups and eight 64-code activation groups, one per warp. The
// whole stage's codes, weight scales,
// quantized activation and activation scales are copied with cp.async into a three-buffer ring.
//
// The A fragment is spread straight from the staged codes: each lane extracts the four 12-bit
// fields its m16n8k32 A registers need (rows gid/gid+8 plus 16*mt, codes 32*ks + 16*(k&1) + 4*lid
// of its warp's 64-code step) and runs the same `spread4` sign-extension the tall A8 engine uses.
// There is no decoded int8 tile, so the decode -> ldmatrix -> barrier round trip disappears: each
// warp reads only the staged codes and its own `x` fragments, and the end-of-loop barrier is the
// stage's only block-wide synchronization. The stage ring holds four 128-code weight groups for
// the shared-memory footprint the A16 route needs for two, halving the stage count and doubling
// the per-row cp.async burst (192 bytes against 96).

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q3/q3_rowsplit_a8_codec.cuh"
#include "ops/linear/q3/q3_rowsplit_small_t_mma.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::q3_small_t_a8 {

using q3_small_t::kMaxTokens;
using q3_small_t::kM16Tiles;
using q3_small_t::kRows;
using q3_small_t::kThreads;
using q3_small_t::kTokens;
using q3_small_t::kTokensPerB;
using q3_small_t::kWarps;
using q3_small_t::kWideTokens;
using q3_small_t::Q3SmallTLinearProblem;
using q3_small_t::Q3SmallTSwiGluProblem;

// One stage: four 128-code weight groups = eight 64-code activation groups = eight 64-code steps.
// The dispatch admits only exact-K problems whose K is a whole number of stages.
constexpr int kStageK               = 512;
constexpr int kStepK                = 64;
constexpr int kStepsPerStage        = kStageK / kStepK;
constexpr int kWeightGroupsPerStage = kStageK / Q3RowSplitStorage::kGroupK;
constexpr int kActGroupsPerStage    = kStageK / 64;
constexpr int kStageCodeBytes       = kWeightGroupsPerStage * Q3RowSplitStorage::kCodeBytesPerGroup;

constexpr int kCodeBuffers       = 3;
constexpr int kOutstandingGroups = 1;
static_assert(kCodeBuffers == kOutstandingGroups + 2, "the ring keeps two ready stages ahead");

// 1.5 * 2^23 as FP32 bits: kFloatMagic + d reinterpreted as a float is 12582912 + d exactly for
// |d| < 2^22 (a group's |d| <= 64 * 4 * 127 = 32512).
constexpr int kFloatMagic        = 0x4B400000;
constexpr float kFloatMagicValue = 12582912.0f;

template <int StageTokens>
struct Stage {
    __align__(16) std::uint8_t codes[kRows][kStageCodeBytes];
    __align__(16) std::int8_t x[StageTokens][kStageK];
    __align__(16) std::uint16_t wscale[kRows][kWeightGroupsPerStage];
    // The quantized activation's group scales are stored group-major, exactly as
    // a8_g64_quantize writes and the tall A8 engine reads them: group g of token t lives at
    // x_scale[g * tokens + t]. The staged tile keeps that order, which leaves each group's run
    // at an arbitrary float offset, so the staging copies one token's scale at a time with
    // 4-byte cp.async and a 16-byte copy would silently drop a misaligned source.
    __align__(16) float xscale[kActGroupsPerStage][StageTokens];
};

// The stage ring is the only working storage: A fragments are spread straight from the staged
// codes, so no decoded tile exists. The union holds the K-split partial sums after the loop.
template <int StageTokens>
struct Storage {
    union {
        Stage<StageTokens> work[kCodeBuffers];
        __align__(16) float partial[kWarps][kM16Tiles][32][4];
    };
};

template <int StageTokens>
__host__ __device__ constexpr int stage_bytes() {
    return kCodeBuffers * (kRows * kStageCodeBytes + StageTokens * kStageK +
                           kRows * kWeightGroupsPerStage * 2 +
                           StageTokens * kActGroupsPerStage * 4);
}
static_assert(stage_bytes<4>() * 2 + 2 * 1024 <= 102400, "two CTAs must fit four activation rows");
static_assert(stage_bytes<8>() * 2 + 2 * 1024 <= 102400, "two CTAs must fit eight activation rows");
static_assert(stage_bytes<kWideTokens>() * 2 + 2 * 1024 <= 102400,
              "two CTAs must fit sixteen activation rows");
static_assert(stage_bytes<kWideTokens>() <= 48 * 1024,
              "the wide tile must fit static shared memory");

// 16-byte chunk c of the 64-code step with swizzle line `line` sits at chunk c ^ ((line >> 1) & 3).
__device__ __forceinline__ unsigned swizzled_chunk(unsigned chunk, int line) {
    return (chunk ^ (static_cast<unsigned>(line >> 1) & 3u)) << 4;
}

// One 12-bit field (four codes) at `bit` of a packed 24-byte 64-code step. The field always spans
// exactly two bytes, so two byte loads replace an unaligned 32-bit one.
__device__ __forceinline__ unsigned field12(const std::uint8_t* base, int bit) {
    const int byte = bit >> 3;
    const unsigned raw =
        static_cast<unsigned>(base[byte]) | (static_cast<unsigned>(base[byte + 1]) << 8);
    return (raw >> (bit & 7)) & 0xfffu;
}

template <class Problem, int TileTokens, int StageTokens>
__global__ void __launch_bounds__(kThreads, 2)
    q3_small_t_a8_mma_kernel(const std::int8_t* __restrict__ qx,
                             const float* __restrict__ x_scale, Problem problem, std::int32_t k,
                             std::int32_t tokens) {
    static_assert(TileTokens == kTokens || TileTokens == kWideTokens);
    static_assert(StageTokens == 4 || StageTokens == 8 || StageTokens == kWideTokens);
    static_assert(StageTokens <= TileTokens);
    constexpr int kTileHalves = TileTokens / kTokensPerB;
    __shared__ Storage<StageTokens> storage;

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int row_block = static_cast<int>(blockIdx.x);
    const int live      = min(TileTokens, tokens);
    const int stages    = k / kStageK;

    // Whole-stage staging: the codes of one row's stage are 192 contiguous bytes (twelve 16-byte
    // vectors) and its four group scales are one 8-byte vector. The quantized activation's stage is
    // 32 16-byte chunks per token and eight FP32 group scales per token.
    const auto issue_stage = [&](int stage, int buffer) {
        constexpr int kChunksPerRow = kStageCodeBytes / 16;
        Stage<StageTokens>& target  = storage.work[buffer];
        for (int item = tid; item < kRows * kChunksPerRow; item += kThreads) {
            const int row   = item / kChunksPerRow;
            const int chunk = item % kChunksPerRow;
            cp_async<16, Cache::cg>(
                &target.codes[row][chunk * 16],
                problem.code_row(row_block, row) + static_cast<std::int64_t>(stage) * kStageCodeBytes +
                    chunk * 16);
        }
        if (tid < kRows) {
            cp_async<8>(&target.wscale[tid][0],
                        problem.scale_row(row_block, tid) + stage * kWeightGroupsPerStage);
        }
        // One 16-byte activation chunk per item. Eight rows fit one item per thread exactly, four
        // rows use the lower half and the wide tile walks two; spelling the three shapes out keeps
        // the narrow instantiations free of the branch a guard would add and of the speculative
        // unrolling a runtime-bounded loop produces.
        if constexpr (StageTokens * 32 == kThreads) {
            const int item  = tid;
            const int token = item >> 5;
            const int chunk = item & 31;
            const int step  = chunk >> 2;
            const int part  = chunk & 3;
            const int line  = token * kStepsPerStage + step;
            const bool ok   = token < live;
            const std::int8_t* src =
                qx + static_cast<std::int64_t>(ok ? token : 0) * k + stage * kStageK + chunk * 16;
            cp_async_zfill<16>(&target.x[token][step * kStepK + swizzled_chunk(part, line)], src,
                               ok ? 16 : 0);
        } else if constexpr (StageTokens * 32 < kThreads) {
            if (tid < StageTokens * 32) {
                const int item  = tid;
                const int token = item >> 5;
                const int chunk = item & 31;
                const int step  = chunk >> 2;
                const int part  = chunk & 3;
                const int line  = token * kStepsPerStage + step;
                const bool ok   = token < live;
                const std::int8_t* src =
                    qx + static_cast<std::int64_t>(ok ? token : 0) * k + stage * kStageK + chunk * 16;
                cp_async_zfill<16>(&target.x[token][step * kStepK + swizzled_chunk(part, line)], src,
                                   ok ? 16 : 0);
            }
        } else {
#pragma unroll
            for (int item = tid; item < StageTokens * 32; item += kThreads) {
                const int token = item >> 5;
                const int chunk = item & 31;
                const int step  = chunk >> 2;
                const int part  = chunk & 3;
                const int line  = token * kStepsPerStage + step;
                const bool ok   = token < live;
                const std::int8_t* src =
                    qx + static_cast<std::int64_t>(ok ? token : 0) * k + stage * kStageK + chunk * 16;
                cp_async_zfill<16>(&target.x[token][step * kStepK + swizzled_chunk(part, line)], src,
                                   ok ? 16 : 0);
            }
        }
        if (tid < kActGroupsPerStage * StageTokens) {
            const int group = tid / StageTokens;
            const int token = tid % StageTokens;
            const bool ok   = token < live;
            // The plane is group-major with stride `tokens`, so a group's run starts at an
            // arbitrary float offset; a 4-byte copy keeps the source aligned.
            const float* src = x_scale +
                               static_cast<std::int64_t>(stage * kActGroupsPerStage + group) *
                                   tokens +
                               (ok ? token : 0);
            cp_async_zfill<4>(&target.xscale[group][token], src, ok ? 4 : 0);
        }
        cp_commit();
    };
    float acc[kTileHalves][kM16Tiles][4] = {};

    // Prologue: two stages fill the ring, so the loop always has the next stage complete and the
    // stage after it in flight.
#pragma unroll
    for (int prefetch = 0; prefetch < kCodeBuffers - 1; ++prefetch) {
        if (prefetch < stages) {
            issue_stage(prefetch, prefetch);
        } else {
            cp_commit();
        }
    }
    cp_wait<0>();
    __syncthreads();

    // The warp's 64-code weight step of every stage lives at this byte offset of its 192-byte code
    // row; the A fragment fields are addressed relative to it.
    const int step_byte =
        (warp >> 1) * Q3RowSplitStorage::kCodeBytesPerGroup + (warp & 1) * 24;
    for (std::int32_t stage = 0; stage < stages; ++stage) {
        const int buffer = static_cast<int>(stage % kCodeBuffers);
        // A fragments are spread straight from this stage's codes, and the buffer being refilled
        // (stage - 1 mod kCodeBuffers) was fully read in the previous iteration, so the
        // end-of-loop barrier is the only block-wide synchronization the stage needs.
        if (stage + kCodeBuffers - 1 < stages) {
            issue_stage(static_cast<int>(stage) + kCodeBuffers - 1,
                        static_cast<int>((stage + kCodeBuffers - 1) % kCodeBuffers));
        } else {
            cp_commit();
        }

        const Stage<StageTokens>& source = storage.work[buffer];
        int g[kTileHalves][kM16Tiles][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            // One A fragment set per warp step serves every token half; only the B fragment of each
            // eight-column half changes.
            unsigned b[kTileHalves][2];
#pragma unroll
            for (int half = 0; half < kTileHalves; ++half) {
                const int token = min((lane & 7) + kTokensPerB * half, StageTokens - 1);
                const int line  = token * kStepsPerStage + warp;
                const unsigned off = static_cast<unsigned>(line * kStepK) +
                                     swizzled_chunk(static_cast<unsigned>(2 * ks + ((lane >> 3) & 1)),
                                                    line);
                ldmatrix_x2(b[half][0], b[half][1], smem_addr(&source.x[0][0]) + off);
            }
#pragma unroll
            for (int mt = 0; mt < kM16Tiles; ++mt) {
                const int row0            = gid + mt * 16;
                const std::uint8_t* base0 = &source.codes[row0][step_byte];
                const std::uint8_t* base1 = &source.codes[row0 + 8][step_byte];
                const int bit_lo          = 96 * ks + 12 * lid;
                const int bit_hi          = bit_lo + 48;
                const unsigned a0         = q3_a8::spread4(field12(base0, bit_lo));
                const unsigned a2         = q3_a8::spread4(field12(base0, bit_hi));
                const unsigned a1         = q3_a8::spread4(field12(base1, bit_lo));
                const unsigned a3         = q3_a8::spread4(field12(base1, bit_hi));
#pragma unroll
                for (int half = 0; half < kTileHalves; ++half) {
                    if (ks == 0) {
                        mma_s8_from(g[half][mt][0], g[half][mt][1], g[half][mt][2], g[half][mt][3], a0,
                                    a1, a2, a3, b[half][0], b[half][1], kFloatMagic, kFloatMagic,
                                    kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(g[half][mt][0], g[half][mt][1], g[half][mt][2], g[half][mt][3], a0, a1,
                               a2, a3, b[half][0], b[half][1]);
                    }
                }
            }
        }

        // One FP32 product and one fused multiply-add per 64-code group, ascending in K within
        // the warp's own group sequence.
        // One FP32 product and one fused multiply-add per 64-code group, ascending in K within
        // the warp's own group sequence. Each eight-column half scales its own columns.
#pragma unroll
        for (int half = 0; half < kTileHalves; ++half) {
            const int group = warp;
            const int tok0  = min(kTokensPerB * half + 2 * lid, StageTokens - 1);
            const int tok1  = min(kTokensPerB * half + 2 * lid + 1, StageTokens - 1);
            const float ts0 = source.xscale[group][tok0];
            const float ts1 = source.xscale[group][tok1];
#pragma unroll
            for (int mt = 0; mt < kM16Tiles; ++mt) {
                const float rs0 =
                    __half2float(__ushort_as_half(source.wscale[mt * 16 + gid][group >> 1]));
                const float rs1 =
                    __half2float(__ushort_as_half(source.wscale[mt * 16 + gid + 8][group >> 1]));
                acc[half][mt][0] =
                    fmaf(rs0 * ts0, __int_as_float(g[half][mt][0]) - kFloatMagicValue, acc[half][mt][0]);
                acc[half][mt][1] =
                    fmaf(rs0 * ts1, __int_as_float(g[half][mt][1]) - kFloatMagicValue, acc[half][mt][1]);
                acc[half][mt][2] =
                    fmaf(rs1 * ts0, __int_as_float(g[half][mt][2]) - kFloatMagicValue, acc[half][mt][2]);
                acc[half][mt][3] =
                    fmaf(rs1 * ts1, __int_as_float(g[half][mt][3]) - kFloatMagicValue, acc[half][mt][3]);
            }
        }
        // Leave the pipeline's in-flight window outstanding; the sync publishes the stages the
        // next iterations consume and proves every warp is done with this stage's tile.
        if (stage + kOutstandingGroups < stages) {
            cp_wait<kOutstandingGroups>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
    }

    // K-split reduction: odd warps publish, even warps fold their partner and republish, warp 0
    // owns the final sum. Each eight-column half reduces and emits independently.
#pragma unroll
    for (int half = 0; half < kTileHalves; ++half) {
        const auto store_partial = [&](int split) {
#pragma unroll
            for (int t = 0; t < kM16Tiles; ++t) {
                *reinterpret_cast<float4*>(&storage.partial[split][t][lane][0]) =
                    make_float4(acc[half][t][0], acc[half][t][1], acc[half][t][2], acc[half][t][3]);
            }
        };
        if ((warp & 1) != 0) { store_partial(warp); }
        __syncthreads();
        if ((warp & 1) == 0) {
#pragma unroll
            for (int t = 0; t < kM16Tiles; ++t) {
                const float4 partner =
                    *reinterpret_cast<const float4*>(&storage.partial[warp + 1][t][lane][0]);
                acc[half][t][0] += partner.x;
                acc[half][t][1] += partner.y;
                acc[half][t][2] += partner.z;
                acc[half][t][3] += partner.w;
            }
            if (warp != 0) { store_partial(warp); }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
#pragma unroll
                for (int t = 0; t < kM16Tiles; ++t) {
                    const float4 value =
                        *reinterpret_cast<const float4*>(&storage.partial[split][t][lane][0]);
                    acc[half][t][0] += value.x;
                    acc[half][t][1] += value.y;
                    acc[half][t][2] += value.z;
                    acc[half][t][3] += value.w;
                }
            }
            problem.emit(row_block, kTokensPerB * half, lane, acc[half], tokens);
        }
        if (half + 1 < kTileHalves) { __syncthreads(); }
    }
}

// Launches one CTA per row block over an activation quantized by rowsplit_a8_quantize for this `k`
// and `tokens` (1..16): the narrow tile owns 1..8 columns and the wide tile 9..16, so every code
// byte of a row block is loaded once. The weight must have k == padded_k and a whole number of
// 512-code stages.
template <class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const std::int8_t* qx,
            const float* x_scale, std::int32_t k, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 0 || tokens > kMaxTokens) {
        throw std::invalid_argument("q3 small-T A8 MMA: unsupported token extent");
    }
    if (k <= 0 || (k % kStageK) != 0) {
        throw std::invalid_argument("q3 small-T A8 MMA: K is not a whole number of stages");
    }
    const unsigned grid = static_cast<unsigned>(row_blocks);
    if (tokens <= kTokens) {
        if (tokens <= 4) {
            q3_small_t_a8_mma_kernel<Problem, kTokens, 4><<<grid, kThreads, 0, stream>>>(
                qx, x_scale, problem, k, tokens);
        } else {
            q3_small_t_a8_mma_kernel<Problem, kTokens, kTokens><<<grid, kThreads, 0, stream>>>(
                qx, x_scale, problem, k, tokens);
        }
    } else {
        q3_small_t_a8_mma_kernel<Problem, kWideTokens, kWideTokens><<<grid, kThreads, 0, stream>>>(
            qx, x_scale, problem, k, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::q3_small_t_a8
