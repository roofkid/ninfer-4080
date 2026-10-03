#pragma once

// INT8-native GQA prompt kernel for the registered Qwen3.6 head geometries on sm_89. QK stays
// INT8 through m16n8k32.s8 Tensor Cores; V alone is dequantized to FP16 for the PV product.
//
// Warp specialization with a one-tile software pipeline. Eight producer warps (four 16-row
// tiles x two Bc column halves) own QK, the online softmax and the K tiles; eight worker warps
// own the FP32 output accumulator (two 16-row tiles x one 64-dimension group each) and the V
// tile. While the workers multiply P(t) by V(t), the producers score tile t + 1 and hold its
// probabilities in registers, so the INT8 QK and the FP16 PV Tensor Core work of consecutive
// tiles overlap. The roles meet at two one-sided named barriers per tile (PFree: PV(t) has
// read P(t); PReady: P(t + 1) is published), so neither side waits through the other's decode
// work. Producers issue K(t + 2) as soon as every producer has finished reading K(t + 1) (the
// named-barrier max exchange), and wait for it at the start of the next scoring pass. Workers
// issue V(t + 1) at the start of PV(t) and dequantize it to FP16 right after PV(t), off the
// producers' scoring path. Packed K codes (rk4v4, rk4v4-e8) land by cp.async in the unused
// upper half of each packed V row and each producer expands the chunks it issued; packed V
// codes stay packed in shared memory until the FP16 dequantizer.
//
// Consumer Ada runs f32-accumulate HMMA at half rate, so each 64-key PV tile accumulates in
// packed FP16 at full rate and is folded into the FP32 accumulator once per tile.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kCausalPromptI8Warps         = 16;
inline constexpr int kCausalPromptI8Threads       = kCausalPromptI8Warps * 32;
inline constexpr int kCausalPromptI8ProducerWarps = 8;
inline constexpr int kCausalPromptI8WorkerWarps   = kCausalPromptI8Warps - kCausalPromptI8ProducerWarps;
inline constexpr int kCausalPromptI8Br            = 64;
inline constexpr int kCausalPromptI8Bc            = 64;
inline constexpr int kCausalPromptI8Groups        = kCausalPromptHeadDim / kKVCacheInt8Group;
inline constexpr int kCausalPromptI8DB16          = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptI8RowTiles      = kCausalPromptI8Br / 16;

inline constexpr int kCausalPromptI8QBytes = kCausalPromptI8Br * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8QScaleBytes =
    kCausalPromptI8Br * kCausalPromptI8Groups * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8KBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VStageBytes =
    kCausalPromptI8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8PBytes =
    kCausalPromptI8Br * kCausalPromptI8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8ScaleBytes =
    2 * kCausalPromptI8Bc * kCausalPromptI8Groups * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8StatsBytes =
    2 * kCausalPromptI8Br * static_cast<int>(sizeof(float));
// Block-max and row-sum exchange slots of the two producer column halves of each row tile.
inline constexpr int kCausalPromptI8PairStatsBytes =
    2 * 2 * kCausalPromptI8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8SmemBytes =
    kCausalPromptI8QBytes + kCausalPromptI8QScaleBytes + kCausalPromptI8KBytes + kCausalPromptI8VBytes +
    kCausalPromptI8VStageBytes + kCausalPromptI8PBytes + kCausalPromptI8ScaleBytes +
    kCausalPromptI8StatsBytes + kCausalPromptI8PairStatsBytes;

static_assert(kCausalPromptI8Groups == 4);
static_assert(kCausalPromptI8ProducerWarps == 2 * kCausalPromptI8RowTiles);
static_assert(kCausalPromptI8WorkerWarps * 2 * 64 == kCausalPromptI8RowTiles * kCausalPromptHeadDim);
static_assert(kCausalPromptI8SmemBytes == 93696);

// Ada throttles on the conversion pipe, so the FP16 codes are built with byte permutes: bias
// each code to unsigned, splice it under exponent 2^10 (0x64xx is 1024 + byte), and subtract
// the bias back. Integer halves are exact, and the scale product is one correctly rounded FP16
// multiply, as for a converted code.
__device__ __forceinline__ int4 causal_prompt_i8_dequant_f16x8(const std::int8_t* codes8,
                                                             __half scale) {
    const int2 raw       = load_vec<int2>(codes8);
    const __half2 s2     = __halves2half2(scale, scale);
    const __half2 magic2 = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const unsigned x0    = static_cast<unsigned>(raw.x) ^ 0x80808080u;
    const unsigned x1    = static_cast<unsigned>(raw.y) ^ 0x80808080u;
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned src   = i < 2 ? x0 : x1;
        const unsigned pair  = __byte_perm(src, 0x64646464u, (i & 1) ? 0x7352u : 0x7150u);
        const __half2 code2  = __hsub2(*reinterpret_cast<const __half2*>(&pair), magic2);
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Eight packed signed INT4 codes (nibble i is dimension d + i, the kv_cache_pack_i4 order) to
// eight scaled FP16 values, bit-identical to unpacking to INT8 and dequantizing those.
__device__ __forceinline__ int4 causal_prompt_i4_dequant_f16x8(std::uint32_t raw, __half scale) {
    const __half2 s2     = __halves2half2(scale, scale);
    const __half2 magic2 = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    const unsigned x     = raw ^ 0x88888888u; // nibble -> code + 8
    const unsigned lo    = x & 0x0f0f0f0fu;          // bytes: d0 d2 d4 d6
    const unsigned hi    = (x >> 4) & 0x0f0f0f0fu;   // bytes: d1 d3 d5 d7
    const unsigned t0    = __byte_perm(lo, hi, 0x5140u); // d0 d1 d2 d3
    const unsigned t1    = __byte_perm(lo, hi, 0x7362u); // d4 d5 d6 d7
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned src   = i < 2 ? t0 : t1;
        const unsigned pair  = __byte_perm(src, 0x64646464u, (i & 1) ? 0x7352u : 0x7150u);
        const __half2 code2  = __hsub2(*reinterpret_cast<const __half2*>(&pair), magic2);
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// 512 threads x 128 registers is the whole register file; shared memory caps occupancy at one
// CTA per SM either way.
template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Root = false, typename Metadata>
__global__ __maxnreg__(128) void causal_attention_prompt_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D               = kCausalPromptHeadDim;
    constexpr int Br              = kCausalPromptI8Br;
    constexpr int Bc              = kCausalPromptI8Bc;
    constexpr int DB16            = kCausalPromptI8DB16;
    constexpr int Groups          = kCausalPromptI8Groups;
    constexpr int GroupKc         = kKVCacheInt8Group / 32;
    constexpr int ColSplit        = 2;
    constexpr int ProducerWarps   = kCausalPromptI8ProducerWarps;
    constexpr int ProducerThreads = ProducerWarps * 32;
    constexpr int WorkerThreads   = kCausalPromptI8WorkerWarps * 32;
    constexpr int QKNtL           = Bc / 8 / ColSplit;
    constexpr int WorkerRowTiles  = 2;
    constexpr int PVNt            = kKVCacheInt8Group / 8;
    constexpr int PVNtPass        = 4;
    constexpr int PVKs            = Bc / 16;
    // Packed V codes occupy the first D / 2 bytes of each staged V row; packed or E8-root keys
    // are staged behind them until the producers expand them into the INT8 K tile.
    constexpr int kPackedKStage   = D / 2;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;
    constexpr unsigned kMaxBarrier     = 1;
    constexpr unsigned kKReadyBarrier  = 2;
    constexpr unsigned kWorkerBarrier  = 3;
    // Producer/worker handoff barriers, one side syncing and the other only arriving:
    //   PFree:  workers arrive when PV(t) has read P(t) and alpha(t).
    //   PReady: producers arrive when P(t + 1) and alpha(t + 1) are published.
    constexpr unsigned kPFreeBarrier   = 4;
    constexpr unsigned kPReadyBarrier  = 5;
    // Padded FP32 output rows for the rotated-V epilogue, staged over the then idle K, V, V FP16
    // and P tiles (contiguous from k_i8).
    constexpr int kOutStride = D + 8;
    static_assert(Br * kOutStride * static_cast<int>(sizeof(float)) <=
                  kCausalPromptI8KBytes + kCausalPromptI8VBytes + kCausalPromptI8VStageBytes +
                      kCausalPromptI8PBytes);

    static_assert(GroupKc == 2);
    static_assert(PVNt == 8);
    static_assert(!(PackedK || E8Root) || PackedV, "staged packed keys share the packed V rows");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kCausalPromptI8QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kCausalPromptI8QScaleBytes);
    std::int8_t* v_i8 = k_i8 + kCausalPromptI8KBytes;
    __half* v_f16     = reinterpret_cast<__half*>(v_i8 + kCausalPromptI8VBytes);
    __half* p_s       = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                  kCausalPromptI8VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kCausalPromptI8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    float* pair_m_s      = final_l_s + Br;
    float* pair_l_s      = pair_m_s + 2 * Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);
    float* o_f32         = reinterpret_cast<float*>(k_i8);

    const int q_block     = static_cast<int>(blockIdx.x);
    const int q_head      = static_cast<int>(blockIdx.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const bool producer   = warp < ProducerWarps;
    const int q0          = q_block * Br;
    const int kv_head     = q_head / Geometry::GroupSize;
    const int tokens      = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kCausalPromptI8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Leading key blocks whose every key is visible to every row of this CTA tile
    // ((kb + 1) * Bc - 1 <= base_pos + q0). Those blocks stage and score without
    // causal guards; the boundary blocks after them keep the exact masked path.
    const int n_full_blocks = (q0 + Br <= tokens) ? min(key_blocks, (base_pos + q0 + 1) / Bc) : 0;

    // Producers stage and expand K (thread index ptid); workers stage and dequantize V (wtid).
    const int ptid = tid;
    const int wtid = tid - ProducerThreads;

    auto issue_k_tile = [&](int kb, auto full_tag) {
        constexpr bool FullTile = decltype(full_tag)::value;
        const int tile_k0       = kb * Bc;
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        if (ptid < Bc) {
            const int key_l = ptid;
            __half* kd      = &k_scale_s[key_l * Groups];
            if (FullTile || tile_k0 + key_l <= max_query_abs) {
                const std::int64_t off =
                    kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
            }
        }
#pragma unroll
        for (int chunk = ptid; chunk < Bc * (D / 16); chunk += ProducerThreads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            std::int8_t* kd = &k_i8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
            if (FullTile || tile_k0 + key_l <= max_query_abs) {
                if constexpr (E8Root) {
                    const std::int64_t koff =
                        paged_kv_page_head_offset<64, Geometry::KVHeads>(physical_page, kv_head) +
                        static_cast<std::int64_t>(key_l) * 64 + (d / 4);
                    ninfer::ops::cp_async<4>(&v_i8[key_l * D + kPackedKStage + d / 4],
                                             &reinterpret_cast<const std::uint8_t*>(cache_k)[koff]);
                } else if constexpr (PackedK) {
                    const std::int64_t koff =
                        kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key_l);
                    ninfer::ops::cp_async<8>(&v_i8[key_l * D + kPackedKStage + d / 2],
                                             &reinterpret_cast<const std::uint8_t*>(cache_k)[koff]);
                } else {
                    const std::int64_t off =
                        kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
                    cp_async<16, Cache::cg>(kd, &cache_k[off]);
                }
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    // After its own cp.async groups complete, each producer expands exactly the packed key
    // chunks it issued; the K-ready named barrier then publishes the INT8 tile.
    auto expand_k_tile = [&](int kb) {
        if constexpr (PackedK || E8Root) {
            const int tile_k0 = kb * Bc;
#pragma unroll
            for (int chunk = ptid; chunk < Bc * (D / 16); chunk += ProducerThreads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                if (tile_k0 + key_l > max_query_abs) { continue; }
                std::int8_t* kd = &k_i8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
                if constexpr (E8Root) {
                    const std::uint32_t src4 =
                        load_vec<std::uint32_t>(&v_i8[key_l * D + kPackedKStage + d / 4]);
                    int8_t dec8_0[8], dec8_1[8];
                    e8_root_decode_8d_int8(static_cast<uint8_t>(src4 & 0xFF),
                                           static_cast<uint8_t>((src4 >> 8) & 0xFF), dec8_0);
                    e8_root_decode_8d_int8(static_cast<uint8_t>((src4 >> 16) & 0xFF),
                                           static_cast<uint8_t>((src4 >> 24) & 0xFF), dec8_1);
                    *reinterpret_cast<uint64_t*>(&kd[0]) = *reinterpret_cast<const uint64_t*>(dec8_0);
                    *reinterpret_cast<uint64_t*>(&kd[8]) = *reinterpret_cast<const uint64_t*>(dec8_1);
                } else {
                    kv_cache_unpack_i4x16(
                        reinterpret_cast<const std::uint8_t*>(&v_i8[key_l * D + kPackedKStage + d / 2]),
                        kd);
                }
            }
        }
    };

    // Workers stage V codes and scales. Keys past the last query stay unstaged: the FP16
    // dequantizer writes zeros for them without reading the codes.
    auto issue_v_tile = [&](int kb, auto full_tag) {
        constexpr bool FullTile = decltype(full_tag)::value;
        const int tile_k0       = kb * Bc;
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        if (wtid < Bc && (FullTile || tile_k0 + wtid <= max_query_abs)) {
            const std::int64_t off =
                kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, 0, wtid);
            ninfer::ops::cp_async<8>(&v_scale_s[wtid * Groups], &cache_v_scale[off]);
        }
#pragma unroll
        for (int chunk = wtid; chunk < Bc * (D / 16); chunk += WorkerThreads) {
            const int key_l = chunk / (D / 16);
            const int d     = (chunk - key_l * (D / 16)) * 16;
            if (FullTile || tile_k0 + key_l <= max_query_abs) {
                if constexpr (PackedV) {
                    const std::int64_t voff =
                        kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key_l);
                    ninfer::ops::cp_async<8>(&v_i8[key_l * D + d / 2], &cache_v[voff]);
                } else {
                    const std::int64_t off =
                        kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
                    cp_async<16, Cache::cg>(&v_i8[key_l * D + d],
                                            &reinterpret_cast<const std::int8_t*>(cache_v)[off]);
                }
            }
        }
        ninfer::ops::cp_commit();
    };

    // The workers expand the staged V codes into the swizzled FP16 tile between their PV passes,
    // off the producers' scoring path.
    auto dequant_v_tile = [&](int kb, auto full_tag) {
        constexpr bool FullTile = decltype(full_tag)::value;
        const int tile_k0       = kb * Bc;
#pragma unroll 2
        for (int chunk = wtid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
            if (FullTile || tile_k0 + key_l <= max_query_abs) {
                const __half vs = v_scale_s[key_l * Groups + (d >> 6)];
                if constexpr (PackedV) {
                    store_vec(dst, causal_prompt_i4_dequant_f16x8(
                                       load_vec<std::uint32_t>(&v_i8[key_l * D + d / 2]), vs));
                } else {
                    store_vec(dst, causal_prompt_i8_dequant_f16x8(&v_i8[key_l * D + d], vs));
                }
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto issue_k_block = [&](int kb) {
        if (kb < n_full_blocks) {
            issue_k_tile(kb, std::true_type{});
        } else {
            issue_k_tile(kb, std::false_type{});
        }
    };

    // Tile 0 staging overlaps the Q quantization.
    if (producer) {
        issue_k_block(0);
    } else {
        if (n_full_blocks > 0) {
            issue_v_tile(0, std::true_type{});
        } else {
            issue_v_tile(0, std::false_type{});
        }
    }

    // Quantize Q cooperatively. One warp owns one (row, 64-d group) at a time.
    for (int unit = warp; unit < Br * Groups; unit += kCausalPromptI8Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kKVCacheInt8Group + lane;
        const int d1  = d0 + 32;
        float x0      = 0.0f;
        float x1      = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row)]);
            x1 = __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d1, q0 + row)]);
            if constexpr (RotateK) { kv_cache_hadamard64(x0, x1, FullMask); }
        }
        float absmax    = fmaxf(fabsf(x0), fabsf(x1));
        absmax          = warp_max(absmax, FullMask);
        const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
        causal_prompt_store_byte_swizzled(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }
    if (!producer) { ninfer::ops::cp_wait<0>(); }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // ---- producer state: Q fragments and scales, softmax statistics, P(t + 1) in registers.
    const int p_row_base = (warp / ColSplit) * 16;
    const int col_half   = warp % ColSplit;
    const int row0       = p_row_base + gid;
    const int row1       = row0 + 8;
    unsigned qf[Groups][GroupKc][4];
    float qs0[Groups];
    float qs1[Groups];
    unsigned p_row0[QKNtL];
    unsigned p_row1[QKNtL];
    float alpha_r0   = 0.0f;
    float alpha_r1   = 0.0f;
    float running_m0 = -CUDART_INF_F;
    float running_m1 = -CUDART_INF_F;
    float running_l0 = 0.0f;
    float running_l1 = 0.0f;
    const float scale_l2 = scale * Log2E;
    if (producer) {
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            qs0[grp] = q_scale[row0 * Groups + grp];
            qs1[grp] = q_scale[row1 * Groups + grp];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(qf[grp][kk][0], qf[grp][kk][1], qf[grp][kk][2], qf[grp][kk][3],
                            smem_addr(&q_b16[(p_row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(p_row_base + a_rowoff, acol)]));
            }
        }
    }

    // Scores key block kb into p_row0/p_row1 and alpha_r0/alpha_r1, and issues K(kb + 1) once
    // every producer has read K(kb). One body, two instantiations: FullTile compiles the
    // interior-block path with no causal masking and no softmax zero-selects.
    auto produce = [&](int kb, auto full_tag) {
        constexpr bool FullTile = decltype(full_tag)::value;
        const int k0            = kb * Bc;
        ninfer::ops::cp_wait<0>();
        expand_k_tile(kb);
        asm volatile("bar.sync %0, %1;" ::"n"(kKReadyBarrier), "n"(ProducerThreads) : "memory");

        float score[QKNtL][4];
#pragma unroll
        for (int ntl = 0; ntl < QKNtL; ++ntl) {
            score[ntl][0] = score[ntl][1] = score[ntl][2] = score[ntl][3] = 0.0f;
        }
        // Key-tile-outer: one 8-byte load brings a key's four group scales, and one x4 ldmatrix
        // both k-steps of a group. Each score still sums its groups in ascending order.
#pragma unroll
        for (int ntl = 0; ntl < QKNtL; ++ntl) {
            const int nt   = col_half * QKNtL + ntl;
            const int keya = nt * 8 + 2 * lid;
            const uint2 ks_a = load_vec<uint2>(&k_scale_s[keya * Groups]);
            const uint2 ks_b = load_vec<uint2>(&k_scale_s[(keya + 1) * Groups]);
            const __half* ks_ah = reinterpret_cast<const __half*>(&ks_a);
            const __half* ks_bh = reinterpret_cast<const __half*>(&ks_b);
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                // Lanes 16-31 address the second k-step of the group.
                const int brow = nt * 8 + b_rin;
                const int bcol = (grp * GroupKc + (lane >> 4)) * 16 + b_koff;
                unsigned bf[GroupKc][2];
                ldmatrix_x4(bf[0][0], bf[0][1], bf[1][0], bf[1][1],
                            smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    mma_s8(c0, c1, c2, c3, qf[grp][kk][0], qf[grp][kk][1], qf[grp][kk][2],
                           qf[grp][kk][3], bf[kk][0], bf[kk][1]);
                }
                const float ks0 = __half2float(ks_ah[grp]);
                const float ks1 = __half2float(ks_bh[grp]);
                score[ntl][0]   = __fmaf_rn(qs0[grp] * ks0, static_cast<float>(c0), score[ntl][0]);
                score[ntl][1]   = __fmaf_rn(qs0[grp] * ks1, static_cast<float>(c1), score[ntl][1]);
                score[ntl][2]   = __fmaf_rn(qs1[grp] * ks0, static_cast<float>(c2), score[ntl][2]);
                score[ntl][3]   = __fmaf_rn(qs1[grp] * ks1, static_cast<float>(c3), score[ntl][3]);
            }
        }

        if constexpr (!FullTile) {
            const int qabs0 = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1 = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            // A boundary block can still be fully visible for a tail CTA whose
            // n_full_blocks collapsed to zero; keep the per-block skip.
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            if (!full_score_tile) {
#pragma unroll
                for (int ntl = 0; ntl < QKNtL; ++ntl) {
                    const int nt   = col_half * QKNtL + ntl;
                    const int key0 = k0 + nt * 8 + 2 * lid;
                    const int key1 = key0 + 1;
                    score[ntl][0]  = key0 <= qabs0 ? score[ntl][0] : -CUDART_INF_F;
                    score[ntl][1]  = key1 <= qabs0 ? score[ntl][1] : -CUDART_INF_F;
                    score[ntl][2]  = key0 <= qabs1 ? score[ntl][2] : -CUDART_INF_F;
                    score[ntl][3]  = key1 <= qabs1 ? score[ntl][3] : -CUDART_INF_F;
                }
            }
        }
        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
#pragma unroll
        for (int ntl = 0; ntl < QKNtL; ++ntl) {
            bm0 = fmaxf(bm0, fmaxf(score[ntl][0], score[ntl][1]));
            bm1 = fmaxf(bm1, fmaxf(score[ntl][2], score[ntl][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        if (lid == 0) {
            pair_m_s[col_half * Br + row0] = bm0;
            pair_m_s[col_half * Br + row1] = bm1;
        }
        // Every producer has now finished reading K(kb) and its scales.
        asm volatile("bar.sync %0, %1;" ::"n"(kMaxBarrier), "n"(ProducerThreads) : "memory");
        bm0 = fmaxf(pair_m_s[row0], pair_m_s[Br + row0]);
        bm1 = fmaxf(pair_m_s[row1], pair_m_s[Br + row1]);
        if (kb + 1 < key_blocks) { issue_k_block(kb + 1); }

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        alpha_r0               = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        alpha_r1               = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        float bl0              = 0.0f;
        float bl1              = 0.0f;
#pragma unroll
        for (int ntl = 0; ntl < QKNtL; ++ntl) {
            float p00, p01, p10, p11;
            if constexpr (FullTile) {
                p00 = exp2_approx(__fmaf_rn(score[ntl][0], scale_l2, -nm0_scaled));
                p01 = exp2_approx(__fmaf_rn(score[ntl][1], scale_l2, -nm0_scaled));
                p10 = exp2_approx(__fmaf_rn(score[ntl][2], scale_l2, -nm1_scaled));
                p11 = exp2_approx(__fmaf_rn(score[ntl][3], scale_l2, -nm1_scaled));
            } else {
                p00 = score[ntl][0] > -CUDART_INF_F
                          ? exp2_approx(__fmaf_rn(score[ntl][0], scale_l2, -nm0_scaled))
                          : 0.0f;
                p01 = score[ntl][1] > -CUDART_INF_F
                          ? exp2_approx(__fmaf_rn(score[ntl][1], scale_l2, -nm0_scaled))
                          : 0.0f;
                p10 = score[ntl][2] > -CUDART_INF_F
                          ? exp2_approx(__fmaf_rn(score[ntl][2], scale_l2, -nm1_scaled))
                          : 0.0f;
                p11 = score[ntl][3] > -CUDART_INF_F
                          ? exp2_approx(__fmaf_rn(score[ntl][3], scale_l2, -nm1_scaled))
                          : 0.0f;
            }
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const __half2 h0 = __floats2half2_rn(p00, p01);
            const __half2 h1 = __floats2half2_rn(p10, p11);
            p_row0[ntl]      = *reinterpret_cast<const unsigned*>(&h0);
            p_row1[ntl]      = *reinterpret_cast<const unsigned*>(&h1);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);
        // Each column half accumulates its own partial row sum; the shared block max makes the
        // alpha sequences identical, so the halves add linearly and are combined after the loop.
        running_l0 = __fmaf_rn(running_l0, alpha_r0, bl0);
        running_l1 = __fmaf_rn(running_l1, alpha_r1, bl1);
        running_m0 = nm0;
        running_m1 = nm1;
    };
    auto produce_block = [&](int kb) {
        if (kb < n_full_blocks) {
            produce(kb, std::true_type{});
        } else {
            produce(kb, std::false_type{});
        }
    };
    auto publish_p = [&]() {
#pragma unroll
        for (int ntl = 0; ntl < QKNtL; ++ntl) {
            // The swizzle keeps even/odd column pairs adjacent, so store one half2.
            const int col0 = (col_half * QKNtL + ntl) * 8 + 2 * lid;
            *reinterpret_cast<unsigned*>(&p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)]) =
                p_row0[ntl];
            *reinterpret_cast<unsigned*>(&p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)]) =
                p_row1[ntl];
        }
        if (col_half == 0 && lid == 0) {
            alpha_s[row0] = alpha_r0;
            alpha_s[row1] = alpha_r1;
        }
    };
    auto dequant_block = [&](int kb) {
        if (kb < n_full_blocks) {
            dequant_v_tile(kb, std::true_type{});
        } else {
            dequant_v_tile(kb, std::false_type{});
        }
    };

    // The two roles run separate loops so the register allocator can give the producer state
    // (Q fragments, statistics, P) and the worker accumulator the same registers. They meet at
    // two one-sided barriers per tile (the waiting side syncs, the other side only arrives):
    //   PReady(t): producers publish P(t), alpha(t); workers wait before PV(t).
    //   PFree(t):  workers finish PV(t);        producers wait before publishing P(t + 1).
    // An arrival for tile t + 1 always follows the other side's wait for tile t, so the hardware
    // barrier never mixes two tiles' arrivals.
    auto block_barrier = []() { asm volatile("bar.sync 0;" ::: "memory"); };
    auto sync_handoff  = [](auto id_tag) {
        asm volatile("bar.sync %0, %1;" ::"n"(decltype(id_tag)::value),
                     "n"(kCausalPromptI8Threads)
                     : "memory");
    };
    auto arrive_handoff = [](auto id_tag) {
        asm volatile("bar.arrive %0, %1;" ::"n"(decltype(id_tag)::value),
                     "n"(kCausalPromptI8Threads)
                     : "memory");
    };
    using PFree  = std::integral_constant<unsigned, kPFreeBarrier>;
    using PReady = std::integral_constant<unsigned, kPReadyBarrier>;
    if (producer) {
        produce_block(0);
        publish_p();
        arrive_handoff(PReady{});
#pragma unroll 1
        for (int kb = 0; kb + 1 < key_blocks; ++kb) {
            produce_block(kb + 1);
            sync_handoff(PFree{}); // PV(kb) has read P(kb) and alpha(kb).
            publish_p();
            arrive_handoff(PReady{}); // P(kb + 1), alpha(kb + 1) published.
        }
        if (lid == 0) {
            pair_l_s[col_half * Br + row0] = running_l0;
            pair_l_s[col_half * Br + row1] = running_l1;
        }
        block_barrier();
        if (col_half == 0 && lid == 0) {
            final_l_s[row0] = pair_l_s[row0] + pair_l_s[Br + row0];
            final_l_s[row1] = pair_l_s[row1] + pair_l_s[Br + row1];
        }
        block_barrier();
    } else {
        // ---- worker state: rows [32 * row_pair, +32) x dimensions [64 * d_slice, +64).
        const int wwarp      = warp - ProducerWarps;
        const int d_slice    = wwarp & 3;
        const int w_row_base = (wwarp >> 2) * (16 * WorkerRowTiles);
        float acc[WorkerRowTiles][PVNt][4];
#pragma unroll
        for (int r = 0; r < WorkerRowTiles; ++r) {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[r][n][i] = 0.0f; }
            }
        }

        auto pv = [&]() {
#pragma unroll
            for (int r = 0; r < WorkerRowTiles; ++r) {
                const float alpha0 = alpha_s[w_row_base + r * 16 + gid];
                const float alpha1 = alpha_s[w_row_base + r * 16 + gid + 8];
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
                    acc[r][n][0] *= alpha0;
                    acc[r][n][1] *= alpha0;
                    acc[r][n][2] *= alpha1;
                    acc[r][n][3] *= alpha1;
                }
            }
            // Two passes of four n tiles keep the FP16 tile accumulators at 16 registers beside
            // the 64-register FP32 accumulator; the P fragments are reloaded per pass.
#pragma unroll
            for (int n0 = 0; n0 < PVNt; n0 += PVNtPass) {
                unsigned tacc[WorkerRowTiles][PVNtPass][2];
#pragma unroll
                for (int r = 0; r < WorkerRowTiles; ++r) {
#pragma unroll
                    for (int n = 0; n < PVNtPass; ++n) { tacc[r][n][0] = tacc[r][n][1] = 0u; }
                }
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned pf[WorkerRowTiles][4];
                    const int pcol = k * 16 + a_coloff;
#pragma unroll
                    for (int r = 0; r < WorkerRowTiles; ++r) {
                        const int prow = w_row_base + r * 16 + a_rowoff;
                        ldmatrix_x4(pf[r][0], pf[r][1], pf[r][2], pf[r][3],
                                    smem_addr(&p_s[prow * Bc + causal_prompt_p_swz<Bc>(prow, pcol)]));
                    }
#pragma unroll
                    for (int n = 0; n < PVNtPass; n += 2) {
                        // Lanes 16-31 address the next n tile: one x4 load feeds n and n + 1.
                        const int vrow = k * 16 + b_koff + b_rin;
                        const int vcol = (d_slice * PVNt + n0 + n + (lane >> 4)) * 8;
                        unsigned vf[4];
                        ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                                      smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
#pragma unroll
                        for (int r = 0; r < WorkerRowTiles; ++r) {
                            mma_f16_f16acc(tacc[r][n][0], tacc[r][n][1], pf[r][0], pf[r][1],
                                           pf[r][2], pf[r][3], vf[0], vf[1]);
                            mma_f16_f16acc(tacc[r][n + 1][0], tacc[r][n + 1][1], pf[r][0],
                                           pf[r][1], pf[r][2], pf[r][3], vf[2], vf[3]);
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < WorkerRowTiles; ++r) {
#pragma unroll
                    for (int n = 0; n < PVNtPass; ++n) {
                        const __half2 lo = *reinterpret_cast<const __half2*>(&tacc[r][n][0]);
                        const __half2 hi = *reinterpret_cast<const __half2*>(&tacc[r][n][1]);
                        acc[r][n0 + n][0] += __half2float(lo.x);
                        acc[r][n0 + n][1] += __half2float(lo.y);
                        acc[r][n0 + n][2] += __half2float(hi.x);
                        acc[r][n0 + n][3] += __half2float(hi.y);
                    }
                }
            }
        };

        // V(0) codes landed before the Q barrier.
        dequant_block(0);
        sync_handoff(PReady{});
#pragma unroll 1
        for (int kb = 0; kb < key_blocks; ++kb) {
            const bool has_next = kb + 1 < key_blocks;
            if (has_next) {
                if (kb + 1 < n_full_blocks) {
                    issue_v_tile(kb + 1, std::true_type{});
                } else {
                    issue_v_tile(kb + 1, std::false_type{});
                }
            }
            pv();
            if (has_next) {
                arrive_handoff(PFree{});
                // Every worker has finished PV(kb) (the FP16 V tile is free) and its own V(kb + 1)
                // copies have landed.
                ninfer::ops::cp_wait<0>();
                asm volatile("bar.sync %0, %1;" ::"n"(kWorkerBarrier), "n"(WorkerThreads) : "memory");
                dequant_block(kb + 1);
                sync_handoff(PReady{}); // P(kb + 1) and the V(kb + 1) FP16 tile published.
            }
        }
        block_barrier(); // pair row sums written
        block_barrier(); // final row sums written

#pragma unroll
        for (int r = 0; r < WorkerRowTiles; ++r) {
            const int orow0    = w_row_base + r * 16 + gid;
            const int orow1    = orow0 + 8;
            const float inv_l0 = final_l_s[orow0] > 0.0f ? __frcp_rn(final_l_s[orow0]) : 0.0f;
            const float inv_l1 = final_l_s[orow1] > 0.0f ? __frcp_rn(final_l_s[orow1]) : 0.0f;
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                acc[r][n][0] *= inv_l0;
                acc[r][n][1] *= inv_l0;
                acc[r][n][2] *= inv_l1;
                acc[r][n][3] *= inv_l1;
            }
            if constexpr (RotateV) {
                // Stage the normalized FP32 rows; the H64 inverse runs from shared memory below.
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
                    const int d0 = (d_slice * PVNt + n) * 8 + 2 * lid;
                    store_vec(&o_f32[orow0 * kOutStride + d0], make_float2(acc[r][n][0], acc[r][n][1]));
                    store_vec(&o_f32[orow1 * kOutStride + d0], make_float2(acc[r][n][2], acc[r][n][3]));
                }
            } else {
                __nv_bfloat16* out_row0 =
                    orow0 < tile_rows
                        ? out + causal_prompt_q_row_offset<Geometry>(q_head, q0 + orow0)
                        : nullptr;
                __nv_bfloat16* out_row1 =
                    orow1 < tile_rows
                        ? out + causal_prompt_q_row_offset<Geometry>(q_head, q0 + orow1)
                        : nullptr;
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
                    const int d0 = (d_slice * PVNt + n) * 8 + 2 * lid;
                    if (out_row0 != nullptr) {
                        *reinterpret_cast<unsigned*>(&out_row0[d0]) =
                            pack_bf16x2(acc[r][n][0], acc[r][n][1]);
                    }
                    if (out_row1 != nullptr) {
                        *reinterpret_cast<unsigned*>(&out_row1[d0]) =
                            pack_bf16x2(acc[r][n][2], acc[r][n][3]);
                    }
                }
            }
        }
        if constexpr (RotateV) {
            // Undo the cache's H64 in FP32 before the only BF16 rounding: one warp transforms
            // one (row, 64-dimension group) at a time, lane l holding dimensions l and l + 32.
            asm volatile("bar.sync %0, %1;" ::"n"(kWorkerBarrier), "n"(WorkerThreads) : "memory");
#pragma unroll 1
            for (int unit = wwarp; unit < tile_rows * Groups; unit += kCausalPromptI8WorkerWarps) {
                const int row = unit / Groups;
                const int d0  = (unit - row * Groups) * kKVCacheInt8Group + lane;
                float x0      = o_f32[row * kOutStride + d0];
                float x1      = o_f32[row * kOutStride + d0 + 32];
                kv_cache_hadamard64(x0, x1, FullMask);
                __nv_bfloat16* out_row = out + causal_prompt_q_row_offset<Geometry>(q_head, q0 + row);
                out_row[d0]            = __float2bfloat16(x0);
                out_row[d0 + 32]       = __float2bfloat16(x1);
            }
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kCausalPromptI8Threads);
}

} // namespace ninfer::ops
