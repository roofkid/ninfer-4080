#pragma once

// Q3G128_F16S RowSplit x int8 activation (A8) pipelined tensor-core GEMM for widths at least 129
// columns (one CTA per SM).
//
// Semantics. The activation arrives quantized by rowsplit_a8_quantize (per token and 64-column
// group g: scale s = amax / 127, q = rint(x / s) in [-127, 127]). A weight code c in [-4, 3] is
// an exact int8 MMA operand, so every group's integer dot product d_g = sum(c * q) over its 64
// codes is exact in int32. The output is sum_g (w_scale_128(g / 2) * s_g) * d_g in FP32 with one
// FP32 product per 64-group and one fused multiply-add per group in ascending K, before the
// Problem's single BF16 output rounding. The 128-code weight group's one FP16 scale serves both
// of its halves, exactly as in the A16 tall engine.
//
// Structure: as q3_rowsplit_tall_mma.cuh (which also supplies the Problems), but the decoded
// weight tile holds int8 codes and the multiply is m16n8k32 s8 MMAs. A CTA of eight warps owns
// 128 weight rows and `Tokens` (128 or 64) tokens and walks K one 64-code half-group per step.
// The decoded code tile, the activation tile (with their row and token scales) are
// double-buffered, so a step needs one barrier. Every thread decodes one 32-code half of one
// weight row per step from three code words loaded a step earlier. Each step's int32 sums start
// at kFloatMagic, so d_g is one exact FADD away from the sum's bits. Warps 0-3 multiply, then
// decode and apply the step's FP32 update; warps 4-7 first apply the previous step's update and
// decode, then multiply, so each SM sub-partition runs one warp's MMAs while the other warp does
// its integer and FP32 work. Tiles are launched token-tile fastest, so the CTAs of one row block
// run together and share its code bytes in L2.

#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q3/q3_rowsplit_a8_codec.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"
#include "ops/linear/q3/q3_rowsplit_tall_mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::q3_tall_a8 {

using q3_tall::kRows;
using q3_tall::kStepK;
using q3_tall::kThreads;
using q3_tall::Q3LinearProblem;
using q3_tall::Q3RowSource;
using q3_tall::Q3SwiGluProblem;

// 1.5 * 2^23 as FP32 bits: kFloatMagic + d reinterpreted as a float is 12582912 + d exactly for
// |d| < 2^22 (a group's |d| <= 64 * 4 * 127 = 32512).
constexpr int kFloatMagic        = 0x4B400000;
constexpr float kFloatMagicValue = 12582912.0f;

template <int Tokens>
struct Config {
    static_assert(Tokens == 128 || Tokens == 64);
    static constexpr int kWarpsN   = Tokens / 32;
    static constexpr int kWarpsM   = 8 / kWarpsN;
    static constexpr int kWarpRows = kRows / kWarpsM;
    static constexpr int MT        = kWarpRows / 16;

    // Code and activation rows are 64 bytes; the 16-byte chunk c of line l sits at chunk
    // c ^ ((l >> 1) & 3), so eight consecutive lines of one chunk cover all 32 banks.
    struct Stage {
        std::uint8_t a[kRows * kStepK];
        std::uint8_t x[Tokens * kStepK];
        float row_scale[kRows];
        float token_scale[Tokens];
    };

    template <class Problem>
    static constexpr std::size_t shared_bytes() {
        return std::max(2 * sizeof(Stage), static_cast<std::size_t>(Tokens) *
                                               q3_tall::kOutLd<Problem::kOutRows> *
                                               sizeof(__nv_bfloat16));
    }
};

__device__ __forceinline__ unsigned swizzled(int line, int chunk) {
    return static_cast<unsigned>(line * 64 + ((chunk ^ ((line >> 1) & 3)) << 4));
}


template <int Tokens, class Problem>
__global__ void __launch_bounds__(kThreads, 1)
    q3_rowsplit_tall_a8_kernel(const std::int8_t* __restrict__ qx,
                               const float* __restrict__ x_scale, Problem problem,
                               std::int32_t k, std::int32_t tokens, std::int32_t token_tiles) {
    using Cfg   = Config<Tokens>;
    using Stage = typename Cfg::Stage;
    constexpr int MT = Cfg::MT;
    extern __shared__ __align__(128) std::uint8_t q3_tall_a8_smem[];
    Stage* stages = reinterpret_cast<Stage*>(q3_tall_a8_smem);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int wm        = warp % Cfg::kWarpsM;
    const int wn        = warp / Cfg::kWarpsM;
    const bool pong     = warp >= 4;
    const int tile      = static_cast<int>(blockIdx.x);
    const int row_block = tile / token_tiles;
    const int token0    = tile % token_tiles * Tokens;
    const int live      = min(Tokens, tokens - token0);
    const int steps     = k / kStepK;
    const int groups    = k / Q3RowSplitStorage::kGroupK;

    // Decode item: weight row tid / 2 of the tile, code bytes 12 (tid % 2) .. + 11, i.e. the
    // half-group's 32 codes `half`; the row's 128-code group scale is shared by both halves.
    const int my_row      = tid >> 1;
    const int my_half     = tid & 1;
    const Q3RowSource src = problem.source(row_block, my_row, my_half, groups);
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
        unsigned out[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t window = q3_tall::q3_tall_window(raw0, raw1, raw2, j);
            out[2 * j]                 = q3_a8::spread4(window & 0x0fffu);
            out[2 * j + 1]             = q3_a8::spread4(window >> 12);
        }
        *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 2 * my_half)) =
            make_uint4(out[0], out[1], out[2], out[3]);
        *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 2 * my_half + 1)) =
            make_uint4(out[4], out[5], out[6], out[7]);
        if (my_half == 0) { s.row_scale[my_row] = __half2float(__ushort_as_half(raw_scale)); }
    };

    // Activation staging: 16-byte chunks and one token scale per thread < Tokens; tokens past
    // `live` read token0 with zero fill.
    constexpr int kXChunks = Tokens * 4 / kThreads;
    const std::int8_t* x_src[kXChunks];
    unsigned x_dst[kXChunks];
    int x_bytes[kXChunks];
#pragma unroll
    for (int i = 0; i < kXChunks; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item >> 2;
        const int chunk = item & 3;
        const bool ok   = token < live;
        x_src[i]   = qx + static_cast<std::int64_t>(ok ? token0 + token : token0) * k + chunk * 16;
        x_dst[i]   = static_cast<unsigned>(offsetof(Stage, x)) + swizzled(token, chunk);
        x_bytes[i] = ok ? 16 : 0;
    }
    const bool stages_scale = tid < Tokens;
    const float* scale_src  = x_scale + (tid < live ? token0 + tid : token0);
    const unsigned scale_dst =
        static_cast<unsigned>(offsetof(Stage, token_scale)) + static_cast<unsigned>(tid) * 4u;
    const int scale_bytes = tid < live ? 4 : 0;
    const auto issue_x    = [&](int step, Stage& s) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + x_dst[i]), "l"(x_src[i] + step * kStepK), "r"(x_bytes[i]));
        }
        if (stages_scale) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :
                         : "r"(base + scale_dst),
                           "l"(scale_src + static_cast<std::int64_t>(step) * tokens),
                           "r"(scale_bytes));
        }
    };

    // Fragment byte offsets per 32-wide K slice ks: A x4 per 16-row tile (rows 0-7 / 8-15 of
    // chunk 2 ks, then of chunk 2 ks + 1), B x4 per pair of 8-token tiles (tokens of the first
    // tile at chunks 2 ks and 2 ks + 1, then of the second). Tiles differ from the lane's line in
    // multiples of 8, which keep the swizzle.
    unsigned a_off[2];
    unsigned b_off[2];
    {
        const int a_row = wm * Cfg::kWarpRows + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int b_tok = wn * 32 + (lane >> 4) * 8 + (lane & 7);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            a_off[ks] = static_cast<unsigned>(offsetof(Stage, a)) +
                        swizzled(a_row, 2 * ks + (lane >> 4));
            b_off[ks] = static_cast<unsigned>(offsetof(Stage, x)) +
                        swizzled(b_tok, 2 * ks + ((lane >> 3) & 1));
        }
    }

    const auto multiply = [&](const Stage& s, int (&g)[MT][4][4]) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            base + b_off[ks] + np * 16 * kStepK);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, base + a_off[ks] + mt * 16 * kStepK);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int* d = g[mt][nt];
                    if (ks == 0) {
                        mma_s8_from(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1],
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
    };
    float row_scale[MT][2], token_scale[4][2];
    const auto load_scales = [&](const Stage& s) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            row_scale[mt][0] = s.row_scale[wm * Cfg::kWarpRows + mt * 16 + gid];
            row_scale[mt][1] = s.row_scale[wm * Cfg::kWarpRows + mt * 16 + gid + 8];
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            token_scale[nt][0] = s.token_scale[wn * 32 + nt * 8 + 2 * lid];
            token_scale[nt][1] = s.token_scale[wn * 32 + nt * 8 + 2 * lid + 1];
        }
    };
    float acc[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mt][nt][e] = 0.0f; }
        }
    }
    // Fragment C: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] are row gid + 8.
    const auto update = [&](const int (&g)[MT][4][4]) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float unit = row_scale[mt][e >> 1] * token_scale[nt][e & 1];
                    acc[mt][nt][e]   = fmaf(unit, __int_as_float(g[mt][nt][e]) - kFloatMagicValue,
                                            acc[mt][nt][e]);
                }
            }
        }
    };

    load(0);
    decode(stages[0]);
    if (steps > 1) { load(1); }
    issue_x(0, stages[0]);
    cp_commit();
    int g[MT][4][4];
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
            if (step > 0) { update(g); }
            next();
            multiply(s, g);
            load_scales(s);
        } else {
            multiply(s, g);
            next();
            load_scales(s);
            update(g);
        }
    }
    if (pong) { update(g); }

    // Epilogue: the problem stages bf16 outputs as [token][row] in shared memory, then writes
    // 16-byte row chunks of each live token.
    constexpr int kOutRows = Problem::kOutRows;
    constexpr int kLd      = q3_tall::kOutLd<kOutRows>;
    __syncthreads();
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(q3_tall_a8_smem);
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

// Launches `row_blocks` 128-row blocks by ceil(tokens / Tokens) token tiles, token tile fastest,
// over an activation quantized by rowsplit_a8_quantize for this `k` and `tokens`. The problem's
// weights must have k == padded_k, a multiple of 64.
template <int Tokens, class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const std::int8_t* qx,
            const float* x_scale, std::int32_t k, std::int32_t tokens, cudaStream_t stream) {
    constexpr std::size_t kSmem = Config<Tokens>::template shared_bytes<Problem>();
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(q3_rowsplit_tall_a8_kernel<Tokens, Problem>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const std::int32_t token_tiles = (tokens + Tokens - 1) / Tokens;
    const unsigned grid = static_cast<unsigned>(static_cast<std::int64_t>(row_blocks) * token_tiles);
    q3_rowsplit_tall_a8_kernel<Tokens, Problem>
        <<<grid, kThreads, kSmem, stream>>>(qx, x_scale, problem, k, tokens, token_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::q3_tall_a8
