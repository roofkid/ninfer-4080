#include "targets/sensenova_loop_9b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::sensenova_loop_9b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS sensenova_loop_9b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

#include "core/layout.h"

#include <ninfer/ops/attn_input_proj.h>
#include <ninfer/ops/gdn_gating_proj.h>
#include <ninfer/ops/gdn_input_proj.h>
#include <ninfer/ops/linear.h>
#include <ninfer/ops/linear_add.h>
#include <ninfer/ops/linear_pair.h>
#include <ninfer/ops/linear_swiglu.h>
#include <ninfer/ops/mtp_pack.h>
#include <ninfer/ops/residual_add.h>
#include <ninfer/ops/silu_mul.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights&) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    const std::int32_t channels = 2 * TextConfig::key_dim + TextConfig::value_dim;
    return static_cast<std::size_t>(channels) * batch * width * sizeof(std::uint16_t);
}

std::size_t gdn_record_workspace_bytes(const Tensor&, const Variant::GdnProjectionWeights&) { return 1; }

std::size_t post_mixer_workspace_bytes(std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            QType::Q4G64_F16S, 2 * TextConfig::intermediate, TextConfig::hidden, kTextPolicy,
            first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            QType::Q5G64_F16S, TextConfig::hidden, TextConfig::intermediate, kTextPolicy, first,
            last));
    }
    return layout.peak_bytes(1);
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid attention projection row view");
    }
    const std::uint64_t groups     = static_cast<std::uint64_t>(block.padded_shape[1]) /
                                 static_cast<std::uint64_t>(block.group);
    const std::uint64_t low_group  = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row   = groups * low_group;
    const std::uint64_t high_row  = groups * high_group;
    const std::uint64_t scale_row = groups * 2;
    Weight out                    = block;
    out.qdata                     = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh = high_group == 0 ? nullptr
                                : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    if (draft_window == 3) { add_shifted(1029, draft_window + 1); }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile>
Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t, std::uint32_t) {
    throw std::logic_error("SenseNova-Loop-9B has no registered DFlash backend");
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    const Weight query_weight = row_view(weights.query_key, 0, TextConfig::query_size);
    const Weight key_weight   = row_view(weights.query_key, TextConfig::query_size,
                                         TextConfig::kv_size);
    const Weight gate_weight  = row_view(weights.gate_value, 0, TextConfig::query_size);
    const Weight value_weight = row_view(weights.gate_value, TextConfig::query_size,
                                         TextConfig::kv_size);
    ops::linear(hidden, query_weight, query, stream);
    ops::linear(hidden, key_weight, key, stream);
    ops::linear(hidden, gate_weight, gate, stream);
    ops::linear(hidden, value_weight, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, kTextPolicy, workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    const Weight query_weight = row_view(weights.packed, 0, TextConfig::query_size);
    const Weight key_weight =
        row_view(weights.packed, TextConfig::query_size, TextConfig::kv_size);
    const Weight gate_weight = row_view(weights.packed,
                                        TextConfig::query_size + TextConfig::kv_size,
                                        TextConfig::query_size);
    const Weight value_weight =
        row_view(weights.packed, 2 * TextConfig::query_size + TextConfig::kv_size,
                 TextConfig::kv_size);
    ops::linear(hidden, query_weight, query, stream);
    ops::linear(hidden, key_weight, key, stream);
    ops::linear(hidden, gate_weight, gate, stream);
    ops::linear(hidden, value_weight, value, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const Variant::GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    ops::gdn_input_proj(hidden, weights.input_projection.query_key,
                        weights.input_projection.value_z, qkv, output_gate_flat, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const Variant::GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view =
        output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    ops::gdn_input_proj_conv_snapshot(
        hidden, weights.input_projection.query_key, weights.input_projection.value_z, conv_weight,
        conv_states, valid_columns, initial_slot, snapshot_base_slot, query, key, value,
        output_gate_view, leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(
    const Tensor& hidden, const Variant::GdnProjectionWeights& weights, const Tensor& conv_weight,
    const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view =
        output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    auto workspace_scope = workspace.scope();
    const std::size_t record_workspace_bytes =
        gdn_record_workspace_bytes(hidden, weights);
    const DeviceSpan storage = workspace.alloc_bytes(record_workspace_bytes);
    WorkspaceArena leaf_workspace(storage);
    ops::gdn_input_proj_conv_record(
        hidden, weights.input_projection.query_key, weights.input_projection.value_z, conv_weight,
        conv_states, valid_columns, initial_slots, conv_record, query, key, value,
        output_gate_view, leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, kTextPolicy, workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Variant::GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps,
                              weights.control_projection.a_projection,
                              weights.control_projection.b_projection, weights.a_log,
                              weights.dt_bias, workspace, hidden, g, beta, execution);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, const ::ninfer::ops::SparseMoeHints&,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, kTextPolicy, workspace, stream);
    ops::linear_add(activation, weights.down, residual, kTextPolicy, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                    TextConfig::query_size, kTextPolicy, first,
                                                    last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    // The two-parent Q4/Q5 split always materializes its projection into a BF16 staging matrix
    // (query + key + value channels) before the convolution snapshot epilogue.
    const std::int32_t channels = 2 * TextConfig::key_dim + TextConfig::value_dim;
    return static_cast<std::size_t>(channels) * batch_size * last * sizeof(std::uint16_t);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                    TextConfig::value_dim, kTextPolicy, first,
                                                    last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        TextConfig::gdn_value_heads, TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return post_mixer_workspace_bytes(first, last);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::sensenova_loop_9b::detail
