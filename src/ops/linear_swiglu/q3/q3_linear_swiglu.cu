#include "ops/linear_swiglu/q3/q3_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/linear/q3/q3_dispatch.h"
#include "ops/linear/q3/q3_launch.h"
#include "ops/linear/q3/q3_rowsplit_gemv_staged.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = kGateUpRows / 2;
constexpr std::int32_t kInputCols  = 5120;
constexpr std::int32_t kChunkCols  = 64;
constexpr std::int32_t kGemvColumns = 8;
constexpr std::int32_t kTallColumns = 64;

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

// The fused decode route is the staged Q3 GEMV with the gate/up problem: one warp owns one
// output row together with its matching up row (`linear/q3/q3_rowsplit_gemv_staged.cuh`). It
// never materializes the FP32 projection plane and rounds silu(gate) * up once.


} // namespace

std::size_t q3_linear_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens,
                                                      LinearPolicy policy) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q3 linear_swiglu workspace: invalid token interval");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("q3 linear_swiglu workspace: Q3 admits only A16 or A8");
    }
    std::size_t bytes = 0;
    if (policy == LinearPolicy::AllowA8 && max_tokens >= kQ3A8MinTokens) {
        bytes = q3_a8_workspace_capacity_bytes(kInputCols, max_tokens);
    }
    if (max_tokens > kGemvColumns && min_tokens < kTallColumns) {
        const std::int64_t columns = std::min<std::int32_t>(max_tokens, kChunkCols - 1);
        bytes = std::max(bytes, static_cast<std::size_t>(2 * kOutputRows) *
                                    static_cast<std::size_t>(columns) * sizeof(float));
    }
    return bytes;
}

void q3_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                               LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream) {
    if (w.qtype != QType::Q3G128_F16S || w.n != kGateUpRows || w.k != 5120 ||
        w.padded_shape[1] != 5120) {
        throw std::invalid_argument("q3 linear_swiglu: unsupported weight");
    }
    const std::int32_t columns = x.ne[1];
    if (columns <= 0 || out.ne[1] != columns) {
        throw std::invalid_argument("q3 linear_swiglu: invalid token extent");
    }
    if (q3_uses_a8(w.n, w.k, w.padded_shape[1], policy, columns)) {
        auto scope = ws.scope();
        A8G64Activation act = allocate_a8_g64_activation(ws, w.k, columns);
        a8_g64_quantize(x, act, stream);
        launch_q3_mma_tall_a8_swiglu_r64_c128(act, w, out, stream);
        return;
    }
    if (columns <= kGemvColumns) {
        const dim3 grid(
            static_cast<unsigned>((kOutputRows + q3_gemv_staged::kWarpsPerCta - 1) /
                                   q3_gemv_staged::kWarpsPerCta),
            1u, 1u);
        q3_gemv_staged::q3_gemv_staged_kernel<8, true, true,
                                              q3_gemv_staged::kProductionGroupsPerStage,
                                              q3_gemv_staged::kProductionPipelineStages, 3>
            <<<grid, q3_gemv_staged::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales),
                static_cast<__nv_bfloat16*>(out.data), kOutputRows, x.ne[0], columns,
                static_cast<std::int32_t>(w.padded_shape[1]), kOutputRows, kGateUpRows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (columns >= kTallColumns) {
        launch_q3_mma_tall_swiglu_r64_c128(x, w, out, stream);
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
