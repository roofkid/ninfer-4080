#pragma once

// Q3G128_F16S RowSplit x BF16 small-T tensor-core GEMM for one to sixteen token columns.
//
// The route exists because the decode and MTP widths are too narrow for the prefill engines
// (one weight re-read per 64-column tile) and the warp-per-row GEMV leaves the tensor core idle
// and spends one FMA per weight code and token. Four properties distinguish it from those
// routes:
//
//   * whole-group 16-byte staging: every 48-byte 128-code group is copied into shared memory as
//     three 16-byte vectors, so a row's weight stream is contiguous DRAM traffic instead of the
//     strided 12-byte quarter reads the tall tiles use;
//   * a three-buffer staging ring: every stage's codes and activation travel together, and the
//     loop issues two stages ahead, so one stage's copies are always in flight while the previous
//     stage is decoded and multiplied;
//   * a small CTA footprint: 32 weight rows x one token tile per CTA, 8 columns for the decode
//     and MTP verify widths and a native 16 columns for the DFlash2 and wide n-gram windows, which
//     keeps several CTAs resident per SM: staging four activation rows needs about 31 KiB and keeps
//     three; the eight-row shape needs about 37 KiB and keeps two; the sixteen-row shape needs
//     about 38 KiB and keeps two;
//   * K-split: the eight warps split each 256-code stage, each warp owning two 16-code slices,
//     and a shared-memory tree reduces their accumulators once per CTA.
//
// Semantics. Each thread decodes one 12-byte quarter (32 codes) of one 128-code group into the
// same `float(code) * float(scale)` -> single BF16 rounding representation the A16 tall routes
// use, and each FP32 accumulator receives one m16n8k16 MMA per 16-wide K slice. The K-split
// partial sums are added once at the end, so the accumulation order differs from the
// warp-per-row GEMV route; the Op's criterion is the authority for both (the documented A16
// criterion applies).
//
// Layout: the decoded tile stores, per 128-code group, 16 16-byte chunks per row at
// `chunk ^ (row & 7)` within the group, which is the swizzle the tall A16 engine uses and keeps
// the ldmatrix A reads conflict-free.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"
#include "ops/linear/q3/q3_rowsplit_tall_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::q3_small_t {

// CTA tile: 32 physical weight rows, one 8- or 16-token column tile, K walked in 256-code
// (two-group) stages. Every registered Q3 parent's padded K is a multiple of 256.
constexpr int kRows           = 32;
// Tokens per CTA tile: eight for the decode and MTP verify widths, sixteen for the DFlash2 and
// wide n-gram verify windows, where a second eight-column tile would duplicate the whole weight
// stream. One CTA owns one row block and one tile, so 9..16 columns load every code byte once.
constexpr int kTokens   = 8;
constexpr int kWideTokens = 16;
constexpr int kTokensPerB = 8;
constexpr int kMaxTokens  = kWideTokens;
constexpr int kWarps          = 8;
constexpr int kThreads        = kWarps * 32;
constexpr int kGroupK         = 256;
constexpr int kGroupsPerStage = kGroupK / Q3RowSplitStorage::kGroupK;
constexpr int kCodeBytesPerGroup = Q3RowSplitStorage::kCodeBytesPerGroup;
constexpr int kStageCodeBytes = kGroupsPerStage * kCodeBytesPerGroup;
constexpr int kSlicesPerStage = kGroupK / 16;
constexpr int kSlicesPerWarp  = kSlicesPerStage / kWarps;
constexpr int kM16Tiles       = 2;
constexpr int kDecodeItemsPerStage = kRows * kGroupsPerStage * 4;

// Staging ring: three buffers keep two stages of copies in flight plus the one being decoded, and
// every stage's codes and activation travel together so the ring has a single owner. The wide tile
// stages sixteen activation rows and decodes the full 32x256 tile, so its ring is one shallower:
// two buffers leave two CTAs resident per SM inside the 48 KiB static shared-memory limit.
template <int StageTokens>
__host__ __device__ constexpr int code_buffers() {
    return StageTokens == kWideTokens ? 2 : 3;
}
template <int StageTokens>
__host__ __device__ constexpr int outstanding_groups() {
    return code_buffers<StageTokens>() - 2;
}
static_assert(kSlicesPerStage % kWarps == 0);
static_assert(kDecodeItemsPerStage == kThreads, "one decode quarter per thread per stage");
static_assert(kStageCodeBytes == 96);
// Per-CTA staging bytes for `StageTokens` activation rows.
template <int StageTokens>
__host__ __device__ constexpr int stage_bytes() {
    return code_buffers<StageTokens>() *
               (kStageCodeBytes * kRows + kGroupsPerStage * kRows * 2 + StageTokens * kGroupK * 2) +
           kRows * kGroupK * 2;
}
static_assert(stage_bytes<4>() * 3 + 3 * 1024 <= 102400, "three CTAs must fit four activation rows");
static_assert(stage_bytes<8>() * 2 + 2 * 1024 <= 102400, "two CTAs must fit eight activation rows");
static_assert(stage_bytes<kWideTokens>() * 2 + 2 * 1024 <= 102400,
              "two CTAs must fit sixteen activation rows");
static_assert(stage_bytes<kWideTokens>() <= 48 * 1024, "the wide tile must fit static shared memory");

template <int StageTokens>
struct Stage {
    __align__(16) std::uint8_t codes[kRows][kStageCodeBytes];
    __align__(16) __nv_bfloat16 x[StageTokens][kGroupK];
    std::uint16_t scale[kRows][kGroupsPerStage];
};

// The decoded weight tile is only live between the decode and the MMAs of one stage; the same
// storage holds the K-split partial sums afterwards.
template <int StageTokens>
struct Storage {
    union {
        struct {
            Stage<StageTokens> stage[code_buffers<StageTokens>()];
            __align__(16) __nv_bfloat16 decoded[kRows][kGroupK];
        } work;
        __align__(16) float partial[kWarps][kM16Tiles][32][4];
    };
};

// Plain 32-row output block: `out` is the token-major BF16 [N, T] destination.
struct Q3SmallTLinearProblem {
    static constexpr bool kFused = false;

    const std::uint8_t* codes;
    const std::uint16_t* scales;
    __nv_bfloat16* out;
    std::int32_t out_ld;
    std::int32_t groups_per_row;

    __device__ __forceinline__ const std::uint8_t* code_row(int row_block, int local_row) const {
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * kRows + local_row;
        return codes + grow * groups_per_row * kCodeBytesPerGroup;
    }
    __device__ __forceinline__ const std::uint16_t* scale_row(int row_block, int local_row) const {
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * kRows + local_row;
        return scales + grow * groups_per_row;
    }
    __device__ __forceinline__ void emit(int row_block, int token0, int lane,
                                         const float (&acc)[kM16Tiles][4], int tokens) const {
        const int gid  = lane >> 2;
        const int lid  = lane & 3;
        const int col0 = token0 + 2 * lid;
        for (int t = 0; t < kM16Tiles; ++t) {
            const int row0 = row_block * kRows + t * 16 + gid;
            if (col0 < tokens) {
                out[static_cast<std::int64_t>(col0) * out_ld + row0] =
                    __float2bfloat16_rn(acc[t][0]);
                out[static_cast<std::int64_t>(col0) * out_ld + row0 + 8] =
                    __float2bfloat16_rn(acc[t][2]);
            }
            if (col0 + 1 < tokens) {
                out[static_cast<std::int64_t>(col0 + 1) * out_ld + row0] =
                    __float2bfloat16_rn(acc[t][1]);
                out[static_cast<std::int64_t>(col0 + 1) * out_ld + row0 + 8] =
                    __float2bfloat16_rn(acc[t][3]);
            }
        }
    }
};

// Folded gate/up SwiGLU: the stacked [2*intermediate, K] parent supplies tile rows 0-15 as the
// gate rows and rows 16-31 as the matching up rows of one 16-row output block, so the epilogue
// publishes the Problem's single `silu(gate) * up` BF16 rounding.
struct Q3SmallTSwiGluProblem {
    static constexpr bool kFused = true;

    const std::uint8_t* codes;
    const std::uint16_t* scales;
    __nv_bfloat16* out;
    std::int32_t intermediate;
    std::int32_t groups_per_row;

    __device__ __forceinline__ const std::uint8_t* code_row(int row_block, int local_row) const {
        const int out_row = row_block * 16 + (local_row & 15);
        const std::int64_t grow =
            (local_row >= 16) ? static_cast<std::int64_t>(intermediate) + out_row : out_row;
        return codes + grow * groups_per_row * kCodeBytesPerGroup;
    }
    __device__ __forceinline__ const std::uint16_t* scale_row(int row_block, int local_row) const {
        const int out_row = row_block * 16 + (local_row & 15);
        const std::int64_t grow =
            (local_row >= 16) ? static_cast<std::int64_t>(intermediate) + out_row : out_row;
        return scales + grow * groups_per_row;
    }
    __device__ __forceinline__ void emit(int row_block, int token0, int lane,
                                         const float (&acc)[kM16Tiles][4], int tokens) const {
        const int gid  = lane >> 2;
        const int lid  = lane & 3;
        const int col0 = token0 + 2 * lid;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row_block * 16 + gid + half * 8;
            if (col0 < tokens) {
                out[static_cast<std::int64_t>(col0) * intermediate + row] =
                    __float2bfloat16_rn(silu(acc[0][2 * half]) * acc[1][2 * half]);
            }
            if (col0 + 1 < tokens) {
                out[static_cast<std::int64_t>(col0 + 1) * intermediate + row] =
                    __float2bfloat16_rn(silu(acc[0][2 * half + 1]) * acc[1][2 * half + 1]);
            }
        }
    }
};

// TileTokens is 8 or 16 columns and StageTokens is 4, 8 or 16 activation rows: four rows are
// enough for the decode and MTP verify widths and keep three CTAs resident, eight rows serve the
// wider verify windows, and sixteen rows give the 9..16-column windows one CTA per row block.
template <class Problem, int TileTokens, int StageTokens>
__global__ void __launch_bounds__(kThreads, TileTokens == kTokens ? (StageTokens == 4 ? 3 : 2) : 2)
    q3_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x, Problem problem, std::int32_t k,
                          std::int32_t padded_k, std::int32_t tokens) {
    static_assert(TileTokens == kTokens || TileTokens == kWideTokens);
    static_assert(StageTokens == 4 || StageTokens == 8 || StageTokens == kWideTokens);
    static_assert(StageTokens <= TileTokens);
    constexpr int kTileHalves  = TileTokens / kTokensPerB;
    constexpr int kRing        = code_buffers<StageTokens>();
    constexpr int kOutstanding = outstanding_groups<StageTokens>();
    __shared__ Storage<StageTokens> storage;

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int row_block = static_cast<int>(blockIdx.x);
    const int live      = min(TileTokens, tokens);
    const std::int32_t stages = padded_k / kGroupK;

    // Whole-group staging: the codes of one row's stage are 96 contiguous bytes (six 16-byte
    // vectors) and the two group scales of a row are one 4-byte vector.
    const auto issue_stage = [&](int stage, int buffer) {
        constexpr int kChunksPerRow = kStageCodeBytes / 16;
        Stage<StageTokens>& target = storage.work.stage[buffer];
        if (tid < kRows * kChunksPerRow) {
            const int row   = tid / kChunksPerRow;
            const int chunk = tid % kChunksPerRow;
            cp_async<16, Cache::cg>(
                &target.codes[row][chunk * 16],
                problem.code_row(row_block, row) + static_cast<std::int64_t>(stage) * kStageCodeBytes +
                    chunk * 16);
        }
        // One 16-byte activation chunk per item. Eight rows fit one item per thread exactly, four
        // rows use the lower half and the wide tile walks two; spelling the three shapes out keeps
        // the narrow instantiations free of the branch a guard would add and of the speculative
        // unrolling a runtime-bounded loop produces.
        if constexpr (StageTokens * 32 == kThreads) {
            const int token = tid >> 5;
            const int chunk = tid & 31;
            const std::int32_t k0 = stage * kGroupK + chunk * 8;
            const bool ok = token < live && k0 < k;
            const __nv_bfloat16* src =
                x + static_cast<std::int64_t>(ok ? token : 0) * k + (ok ? k0 : 0);
            cp_async_zfill<16>(&target.x[token][(chunk ^ (token & 7)) * 8], src, ok ? 16 : 0);
        } else if constexpr (StageTokens * 32 < kThreads) {
            if (tid < StageTokens * 32) {
                const int token = tid >> 5;
                const int chunk = tid & 31;
                const std::int32_t k0 = stage * kGroupK + chunk * 8;
                const bool ok = token < live && k0 < k;
                const __nv_bfloat16* src =
                    x + static_cast<std::int64_t>(ok ? token : 0) * k + (ok ? k0 : 0);
                cp_async_zfill<16>(&target.x[token][(chunk ^ (token & 7)) * 8], src, ok ? 16 : 0);
            }
        } else {
            for (int item = tid; item < StageTokens * 32; item += kThreads) {
                const int token = item >> 5;
                const int chunk = item & 31;
                const std::int32_t k0 = stage * kGroupK + chunk * 8;
                const bool ok = token < live && k0 < k;
                const __nv_bfloat16* src =
                    x + static_cast<std::int64_t>(ok ? token : 0) * k + (ok ? k0 : 0);
                cp_async_zfill<16>(&target.x[token][(chunk ^ (token & 7)) * 8], src, ok ? 16 : 0);
            }
        }
        if (tid < kRows) {
            cp_async<4>(&target.scale[tid][0],
                        problem.scale_row(row_block, tid) + stage * kGroupsPerStage);
        }
        cp_commit();
    };

    const auto decode = [&](int buffer) {
        const Stage<StageTokens>& source = storage.work.stage[buffer];
        const int row = tid >> 3;
        const int gq  = tid & 7;
        const int g   = gq >> 2;
        const int q   = gq & 3;
        const std::uint8_t* src = &source.codes[row][g * kCodeBytesPerGroup + q * 12];
        const std::uint32_t raw0 = *reinterpret_cast<const std::uint32_t*>(src);
        const std::uint32_t raw1 = *reinterpret_cast<const std::uint32_t*>(src + 4);
        const std::uint32_t raw2 = *reinterpret_cast<const std::uint32_t*>(src + 8);
        const float scale = __half2float(__ushort_as_half(source.scale[row][g]));
        std::uint8_t* row_base =
            reinterpret_cast<std::uint8_t*>(&storage.work.decoded[row][0]) + g * 256;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t window = q3_tall::q3_tall_window(raw0, raw1, raw2, j);
            unsigned pairs[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float q0 =
                    static_cast<float>(q3_signed_code((window >> (6 * b)) & 0x7u)) * scale;
                const float q1 =
                    static_cast<float>(q3_signed_code((window >> (6 * b + 3)) & 0x7u)) * scale;
                const __nv_bfloat162 pair = __floats2bfloat162_rn(q0, q1);
                pairs[b]                 = *reinterpret_cast<const unsigned*>(&pair);
            }
            const int chunk = (q * 4 + j) ^ (row & 7);
            *reinterpret_cast<uint4*>(row_base + chunk * 16) =
                make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
        }
    };

    float acc[kTileHalves][kM16Tiles][4] = {};

    // Prologue: two stages fill the ring, so the loop always has the next stage complete and the
    // stage after it in flight.
#pragma unroll
    for (int prefetch = 0; prefetch < kRing - 1; ++prefetch) {
        if (prefetch < stages) {
            issue_stage(prefetch, prefetch);
        } else {
            cp_commit();
        }
    }
    cp_wait<0>();
    __syncthreads();

    for (std::int32_t stage = 0; stage < stages; ++stage) {
        const int buffer = static_cast<int>(stage % kRing);
        decode(buffer);
        // The decoded tile is visible to every warp, and the buffer this stage just read can be
        // refilled by the copies of stage + kRing.
        __syncthreads();
        if (stage + kRing - 1 < stages) {
            issue_stage(static_cast<int>(stage) + kRing - 1,
                        static_cast<int>((stage + kRing - 1) % kRing));
        } else {
            cp_commit();
        }

#pragma unroll
        for (int item = 0; item < kSlicesPerWarp; ++item) {
            const int slice = warp * kSlicesPerWarp + item;
            const int group = slice >> 3;
            const int ks    = slice & 7;
            // One A fragment pair serves every token half; only the B fragment of each eight-column
            // half changes, so the decoded weights and the code stream are read once per row block.
            unsigned b[kTileHalves][2];
#pragma unroll
            for (int half = 0; half < kTileHalves; ++half) {
                const int token = min((lane & 7) + kTokensPerB * half, StageTokens - 1);
                const unsigned off =
                    ((static_cast<unsigned>((lane >> 3) & 1) ^ static_cast<unsigned>(token & 7)) << 4) ^
                    (static_cast<unsigned>(slice) << 5);
                ldmatrix_x2(b[half][0], b[half][1],
                            smem_addr(&storage.work.stage[buffer].x[token][0]) + off);
            }
            const unsigned l7 = static_cast<unsigned>(lane & 7);
            const unsigned h  = static_cast<unsigned>(lane >> 4);
            const unsigned ax = ((h ^ l7) << 4) ^ (static_cast<unsigned>(ks) << 5);
#pragma unroll
            for (int t = 0; t < kM16Tiles; ++t) {
                const int a_row = t * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_addr(&storage.work.decoded[a_row][0]) +
                                static_cast<unsigned>(group * 256) + ax);
#pragma unroll
                for (int half = 0; half < kTileHalves; ++half) {
                    mma_bf16(acc[half][t][0], acc[half][t][1], acc[half][t][2], acc[half][t][3], a0, a1,
                             a2, a3, b[half][0], b[half][1]);
                }
            }
        }

        // Leave the pipeline's in-flight window outstanding; the sync publishes the stages the
        // next iterations consume and proves every warp is done with this stage's tile.
        if (stage + kOutstanding < stages) {
            cp_wait<kOutstanding>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
    }

    // K-split reduction: odd warps publish, even warps fold their partner and republish, warp 0
    // owns the final sum. Each eight-column half reduces and emits independently, so the decoded
    // tile and the code stream still serve both halves.
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

// Launches one CTA per row block over an activation of `tokens` columns (1..16): the narrow tile
// owns 1..8 columns and the wide tile 9..16, so every code byte of a row block is loaded once. The
// weight's padded K must be a whole number of 256-code stages and the logical K must not exceed it.
template <class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const __nv_bfloat16* x,
            std::int32_t k, std::int32_t padded_k, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 0 || tokens > kMaxTokens) {
        throw std::invalid_argument("q3 small-T MMA: unsupported token extent");
    }
    const unsigned grid = static_cast<unsigned>(row_blocks);
    if (tokens <= kTokens) {
        if (tokens <= 4) {
            q3_small_t_mma_kernel<Problem, kTokens, 4><<<grid, kThreads, 0, stream>>>(
                x, problem, k, padded_k, tokens);
        } else {
            q3_small_t_mma_kernel<Problem, kTokens, kTokens><<<grid, kThreads, 0, stream>>>(
                x, problem, k, padded_k, tokens);
        }
    } else {
        q3_small_t_mma_kernel<Problem, kWideTokens, kWideTokens><<<grid, kThreads, 0, stream>>>(
            x, problem, k, padded_k, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::q3_small_t
