#pragma once

// Q3G128_F16S RowSplit x BF16 pipelined tensor-core GEMM for prefill widths (one CTA per SM).
//
// A CTA of eight warps owns 128 weight rows and `Tokens` (64 or 128) tokens and walks K one
// 64-code half-group per step. Raw 3-bit windows are loaded into registers one step ahead, decoded
// to BF16 (float(code) * float(scale), the same dequantization and the same single BF16 rounding
// as the staged 32x64 tile), double-buffered in shared memory and multiplied with m16n8k16 BF16
// MMAs into FP32 accumulators, one MMA per 16-wide K slice in ascending K. The 128-code group's
// one FP16 scale serves both of its halves. Every output therefore sees the same instruction
// sequence on the same operands as the staged tile, whatever the CTA and warp tiling, and the
// outputs are bit-identical.
//
// Pipeline: the decoded weight tile and the activation tile are both double-buffered, so a step
// needs one barrier. Each thread decodes one 32-code quarter of one weight row per step, from
// three 32-bit code words loaded a step earlier. Warps 0-3 multiply before they decode the next
// step and warps 4-7 after, so each SM sub-partition runs one warp's MMAs while the other warp
// decodes. Tiles are launched token-tile fastest, so the CTAs of one row block run together and
// share its code bytes in L2.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::q3_tall {

constexpr int kRows    = 128;
constexpr int kThreads = 256;
constexpr int kStepK   = 64;

template <int Tokens>
struct Q3TallConfig {
    static_assert(Tokens == 128 || Tokens == 64);
    static constexpr int kWarpsN   = Tokens / 32;
    static constexpr int kWarpsM   = 8 / kWarpsN;
    static constexpr int kWarpRows = kRows / kWarpsM;
    static constexpr int MT        = kWarpRows / 16;

    struct Stage {
        __nv_bfloat16 a[kRows * kStepK];  // 128-byte rows, 16-byte chunks XORed with row & 7
        __nv_bfloat16 x[Tokens * kStepK]; // the same swizzle per token
    };
    static constexpr std::size_t kSharedBytes = 2 * sizeof(Stage);
};

// One decode thread's weight row: the row's code plane offset by the thread's 12-byte quarter and
// the row's FP16 128-group scales.
struct Q3RowSource {
    const std::uint8_t* codes;
    const std::uint16_t* scales;
};

// Epilogue staging row stride in bf16: 8 elements of padding keep the fragment stores
// conflict-free and the 16-byte row reads aligned.
template <int OutRows>
constexpr int kOutLd = OutRows + 8;

// The 24-bit window `j` of the 12 code bytes a thread owns: eight consecutive 3-bit codes.
__device__ __forceinline__ std::uint32_t q3_tall_window(std::uint32_t w0, std::uint32_t w1,
                                                        std::uint32_t w2, int j) {
    switch (j) {
    case 0: return w0 & 0x00ffffffu;
    case 1: return (w0 >> 24) | ((w1 & 0x0000ffffu) << 8);
    case 2: return (w1 >> 16) | ((w2 & 0x000000ffu) << 16);
    default: return w2 >> 8;
    }
}

// Fragment C of an m16n8 tile: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] are
// row gid + 8.
template <class Value>
__device__ __forceinline__ void stage_fragment(__nv_bfloat16* staged, int ld, int row, int token,
                                               Value value) {
    staged[token * ld + row]           = value(0);
    staged[(token + 1) * ld + row]     = value(1);
    staged[token * ld + row + 8]       = value(2);
    staged[(token + 1) * ld + row + 8] = value(3);
}

template <int Tokens, class Problem>
__global__ void __launch_bounds__(kThreads, 1)
    q3_rowsplit_tall_mma_kernel(const __nv_bfloat16* __restrict__ x, Problem problem,
                                std::int32_t logical_k, std::int32_t padded_k, std::int32_t tokens,
                                std::int32_t token_tiles) {
    using Cfg   = Q3TallConfig<Tokens>;
    using Stage = typename Cfg::Stage;
    constexpr int MT = Cfg::MT;
    static_assert(Tokens * kOutLd<Problem::kOutRows> * sizeof(__nv_bfloat16) <=
                  Cfg::kSharedBytes);

    extern __shared__ __align__(128) std::uint8_t q3_tall_smem[];
    Stage* stages = reinterpret_cast<Stage*>(q3_tall_smem);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int wm        = warp % Cfg::kWarpsM;
    const int wn        = warp / Cfg::kWarpsM;
    const bool pong     = warp >= 4;
    const int tile      = static_cast<int>(blockIdx.x);
    const int row_block = tile / token_tiles;
    const int token0    = tile % token_tiles * Tokens;
    const int live      = min(Tokens, tokens - token0);
    const int steps     = padded_k / kStepK;
    const int groups    = padded_k / Q3RowSplitStorage::kGroupK;

    // Decode item: weight row tid / 2 of the tile, code bytes 12 (tid % 2) .. + 11.
    const int my_row     = tid >> 1;
    const int my_quarter = tid & 1;
    const Q3RowSource src = problem.source(row_block, my_row, my_quarter, groups);
    std::uint32_t raw0 = 0;
    std::uint32_t raw1 = 0;
    std::uint32_t raw2 = 0;
    std::uint16_t raw_scale = 0;
    const auto load = [&](int step) {
        const std::uint8_t* ptr = src.codes + static_cast<std::int64_t>(step) * 24;
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw0) : "l"(ptr));
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw1) : "l"(ptr + 4));
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw2) : "l"(ptr + 8));
        raw_scale = __ldg(src.scales + (step >> 1));
    };
    const auto decode = [&](Stage& s) {
        const float scale      = __half2float(__ushort_as_half(raw_scale));
        std::uint8_t* row_base = reinterpret_cast<std::uint8_t*>(s.a) + my_row * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t window = q3_tall_window(raw0, raw1, raw2, j);
            unsigned pairs[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float q0 =
                    static_cast<float>(q3_signed_code((window >> (6 * b)) & 0x7u)) * scale;
                const float q1 =
                    static_cast<float>(q3_signed_code((window >> (6 * b + 3)) & 0x7u)) * scale;
                const __nv_bfloat162 pair = __floats2bfloat162_rn(q0, q1);
                pairs[b]                  = *reinterpret_cast<const unsigned*>(&pair);
            }
            const int chunk = (my_quarter * 4 + j) ^ (my_row & 7);
            *reinterpret_cast<uint4*>(row_base + chunk * 16) =
                make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
        }
    };

    // Activation staging: 16-byte chunks; tokens past `live` and chunks past the logical K read
    // token0 with zero fill.
    constexpr int kXChunks = Tokens * 8 / kThreads;
    const __nv_bfloat16* x_src[kXChunks];
    unsigned x_dst[kXChunks];
#pragma unroll
    for (int i = 0; i < kXChunks; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item >> 3;
        const int chunk = item & 7;
        const bool ok   = token < live;
        x_src[i] = x + static_cast<std::int64_t>(ok ? token0 + token : token0) * logical_k +
                   chunk * 8;
        x_dst[i] = static_cast<unsigned>(offsetof(Stage, x) + token * 128 +
                                         ((chunk ^ (token & 7)) << 4));
    }
    const auto issue_x = [&](int step, Stage& s) {
        const unsigned base = smem_addr(&s);
        const std::int32_t k0 = step * kStepK;
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            const int item  = tid + i * kThreads;
            const int token = item >> 3;
            const int chunk = item & 7;
            const int bytes = (token < live && k0 + chunk * 8 + 8 <= logical_k) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + x_dst[i]), "l"(x_src[i] + k0), "r"(bytes));
        }
    };

    // Fragment byte offsets per 16-wide K slice: A x4 per 16-row tile, B x4 per pair of 8-token
    // tiles. A row or token differs from the lane's line only in multiples of 8, so chunk 2 ks + c
    // of it sits at ((c ^ l7) << 4) ^ (ks << 5).
    unsigned a_off[4];
    unsigned b_off[4];
    {
        const unsigned l7    = static_cast<unsigned>(lane & 7);
        const unsigned a_row = static_cast<unsigned>(wm * Cfg::kWarpRows + (lane & 7) +
                                                      ((lane >> 3) & 1) * 8);
        const unsigned b_tok =
            static_cast<unsigned>(wn * 32 + (lane >> 4) * 8 + (lane & 7));
        const unsigned ax = (static_cast<unsigned>(lane >> 4) ^ l7) << 4;
        const unsigned bx = (static_cast<unsigned>((lane >> 3) & 1) ^ l7) << 4;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            a_off[ks] = static_cast<unsigned>(offsetof(Stage, a)) + a_row * 128u +
                        (ax ^ static_cast<unsigned>(ks << 5));
            b_off[ks] = static_cast<unsigned>(offsetof(Stage, x)) + b_tok * 128u +
                        (bx ^ static_cast<unsigned>(ks << 5));
        }
    }

    float acc[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mt][nt][e] = 0.0f; }
        }
    }
    const auto multiply = [&](const Stage& s) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            base + b_off[ks] + np * 16 * 128);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, base + a_off[ks] + mt * 16 * 128);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3], a0,
                             a1, a2, a3, b[nt][0], b[nt][1]);
                }
            }
        }
    };

    load(0);
    decode(stages[0]);
    if (steps > 1) { load(1); }
    issue_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        cp_wait<0>();
        // Stage `step` is visible and every warp is done reading step - 1, whose buffers refill.
        __syncthreads();
        if (step + 1 < steps) { issue_x(step + 1, stages[(step + 1) & 1]); }
        cp_commit();
        const Stage& s  = stages[step & 1];
        const auto next = [&] {
            if (step + 1 < steps) {
                decode(stages[(step + 1) & 1]);
                if (step + 2 < steps) { load(step + 2); }
            }
        };
        if (pong) {
            next();
            multiply(s);
        } else {
            multiply(s);
            next();
        }
    }

    // Epilogue: the problem stages bf16 outputs as [token][row] in shared memory, then writes
    // 16-byte row chunks of each live token.
    constexpr int kOutRows = Problem::kOutRows;
    constexpr int kLd      = kOutLd<kOutRows>;
    __syncthreads();
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(q3_tall_smem);
    problem.template stage_outputs<MT>(acc, staged, kLd, wm * Cfg::kWarpRows, wn * 32, lane);
    __syncthreads();
    constexpr int kChunks = kOutRows / 8;
#pragma unroll
    for (int i = 0; i < Tokens * kChunks / kThreads; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item / kChunks;
        const int chunk = item % kChunks;
        if (token < live) {
            problem.write(row_block, token0 + token, chunk * 8,
                          *reinterpret_cast<const uint4*>(&staged[token * kLd + chunk * 8]));
        }
    }
}

// Plain RowSplit Q3 linear: full 128-row output blocks of a token-major BF16 [N,T] output.
struct Q3LinearProblem {
    static constexpr int kOutRows = kRows;
    const std::uint8_t* codes;
    const std::uint16_t* scales;
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ Q3RowSource source(int row_block, int row, int quarter, int groups) const {
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * kRows + row;
        return {codes + grow * groups * Q3RowSplitStorage::kCodeBytesPerGroup + quarter * 12,
                scales + grow * groups};
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                stage_fragment(staged, ld, warp_row + mt * 16 + (lane >> 2),
                               warp_token + nt * 8 + 2 * (lane & 3),
                               [&](int e) { return __float2bfloat16_rn(acc[mt][nt][e]); });
            }
        }
    }
    __device__ void write(int row_block, int token, int row, uint4 values) const {
        *reinterpret_cast<uint4*>(&out[static_cast<std::int64_t>(token) * out_ld +
                                       row_block * kRows + row]) = values;
    }
};

// Folded gate/up SwiGLU over the stacked [2*intermediate, K] Q3 parent: row block b holds
// output rows 64 b .. 64 b + 63; tile rows 0-31 and 64-95 are their gate rows and tile rows 32-63
// and 96-127 the matching up rows, so a 64-row warp tile pairs its first and second halves.
struct Q3SwiGluProblem {
    static constexpr int kOutRows = 64;
    const std::uint8_t* codes;
    const std::uint16_t* scales;
    __nv_bfloat16* out;
    std::int32_t intermediate;

    __device__ Q3RowSource source(int row_block, int row, int quarter, int groups) const {
        const int block   = row >> 6;
        const int local   = row & 63;
        const int out_row = row_block * 64 + block * 32 + (local & 31);
        const bool up     = (local >> 5) != 0;
        const std::int64_t grow = up ? intermediate + out_row : out_row;
        return {codes + grow * groups * Q3RowSplitStorage::kCodeBytesPerGroup + quarter * 12,
                scales + grow * groups};
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
        static_assert(MT == 4, "the folded pairing needs 64-row warp tiles");
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                stage_fragment(staged, ld, warp_row / 2 + mt * 16 + (lane >> 2),
                               warp_token + nt * 8 + 2 * (lane & 3), [&](int e) {
                                   return __float2bfloat16_rn(silu(acc[mt][nt][e]) *
                                                              acc[mt + 2][nt][e]);
                               });
            }
        }
    }
    __device__ void write(int row_block, int token, int row, uint4 values) const {
        *reinterpret_cast<uint4*>(&out[static_cast<std::int64_t>(token) * intermediate +
                                       row_block * 64 + row]) = values;
    }
};

// Launches `row_blocks` 128-row blocks by ceil(tokens / Tokens) token tiles, token tile fastest.
// The weight's padded K must be a multiple of 64 and the weight must hold whole 128-row blocks.
template <int Tokens, class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const __nv_bfloat16* x,
            std::int32_t logical_k, std::int32_t padded_k, std::int32_t tokens,
            cudaStream_t stream) {
    constexpr std::size_t kSmem = Q3TallConfig<Tokens>::kSharedBytes;
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(q3_rowsplit_tall_mma_kernel<Tokens, Problem>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const std::int32_t token_tiles = (tokens + Tokens - 1) / Tokens;
    const unsigned grid = static_cast<unsigned>(static_cast<std::int64_t>(row_blocks) * token_tiles);
    q3_rowsplit_tall_mma_kernel<Tokens, Problem>
        <<<grid, kThreads, kSmem, stream>>>(x, problem, logical_k, padded_k, tokens, token_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::q3_tall
