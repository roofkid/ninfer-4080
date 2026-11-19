#include "targets/sensenova_loop_9b/impl/load/bindings.h"


#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

using artifact::NumericFormat;
using artifact::TensorPlacement;

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       TensorPlacement placement = TensorPlacement::Device) {
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}

artifact::ObjectHandle bind_bf16(artifact::Binder& binder, std::string_view name,
                                 std::initializer_list<std::uint64_t> shape,
                                 TensorPlacement placement = TensorPlacement::Device) {
    return artifact::bind_tensor(binder, name, NumericFormat::BF16, shape, placement);
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    return DensePostMixerPayload{
        .gate_up = materialized_weight(materialized, plan.gate_up, 2 * TextConfig::intermediate,
                                       TextConfig::hidden),
        .down    = materialized_weight(materialized, plan.down, TextConfig::hidden,
                                       TextConfig::intermediate),
    };
}


} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    if (weights_profile != WeightsProfile::GroupwiseInt) {
        throw std::invalid_argument("sensenova_loop_9b: invalid weights profile");
    }
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding =
        bind_weight(binder, "text/token_embedding", NumericFormat::Q6G64_F16S,
                    {TextConfig::output_rows, TextConfig::hidden});
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                     {TextConfig::hidden});
    out.output_head = bind_weight(binder, "text/output_head", NumericFormat::Q6G64_F16S,
                                  {TextConfig::output_rows, TextConfig::hidden});

    for (std::size_t physical = 0; physical < TextConfig::physical_layers; ++physical) {
        TextLayerPlan& target    = out.text_layers[physical];
        const std::string prefix = "text/layers/" + std::to_string(physical) + "/";
        target.input_norm = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                         NumericFormat::BF16,
                                                         {TextConfig::hidden});
        target.is_full_attention = TextConfig::is_full_attention(static_cast<int>(physical));
        if (target.is_full_attention) {
            target.attention.query_key =
                bind_weight(binder, prefix + "attention/query_key", NumericFormat::Q4G64_F16S,
                            {static_cast<std::uint64_t>(TextConfig::query_size +
                                                        TextConfig::kv_size),
                             TextConfig::hidden});
            target.attention.gate_value =
                bind_weight(binder, prefix + "attention/gate_value", NumericFormat::Q5G64_F16S,
                            {static_cast<std::uint64_t>(TextConfig::query_size +
                                                        TextConfig::kv_size),
                             TextConfig::hidden});
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16,
                {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16,
                {TextConfig::head_dim});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", NumericFormat::Q5G64_F16S,
                            {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_log", NumericFormat::FP32,
                {static_cast<std::uint64_t>(TextConfig::gdn_value_heads)});
            target.gdn.dt_bias = artifact::bind_device_tensor(
                binder, prefix + "gdn/dt_bias", NumericFormat::FP32,
                {static_cast<std::uint64_t>(TextConfig::gdn_value_heads)});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16,
                {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.a_projection =
                bind_weight(binder, prefix + "gdn/a_projection", NumericFormat::BF16,
                            {static_cast<std::uint64_t>(TextConfig::gdn_value_heads),
                             TextConfig::hidden});
            target.gdn.b_projection =
                bind_weight(binder, prefix + "gdn/b_projection", NumericFormat::BF16,
                            {static_cast<std::uint64_t>(TextConfig::gdn_value_heads),
                             TextConfig::hidden});
            target.gdn.query_key =
                bind_weight(binder, prefix + "gdn/query_key", NumericFormat::Q4G64_F16S,
                            {2 * TextConfig::key_dim, TextConfig::hidden});
            target.gdn.value_z =
                bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                            {2 * TextConfig::value_dim, TextConfig::hidden});
            target.gdn.norm = artifact::bind_device_tensor(
                binder, prefix + "gdn/norm", NumericFormat::BF16,
                {TextConfig::gdn_key_head_dim});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S,
                            {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S,
                        {2 * TextConfig::intermediate, TextConfig::hidden});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S,
                        {TextConfig::hidden, TextConfig::intermediate});
        if (physical >= 4 && physical < 28) {
            target.mhc_attn = MhcRouterPlan{
                .weight = artifact::bind_device_tensor(
                    binder, prefix + "mhc/attn/weight", NumericFormat::FP32,
                    {8, 2 * TextConfig::hidden}),
                .bias = artifact::bind_device_tensor(binder, prefix + "mhc/attn/bias",
                                                      NumericFormat::FP32, {8}),
            };
            target.mhc_ffn = MhcRouterPlan{
                .weight = artifact::bind_device_tensor(
                    binder, prefix + "mhc/ffn/weight", NumericFormat::FP32,
                    {8, 2 * TextConfig::hidden}),
                .bias = artifact::bind_device_tensor(binder, prefix + "mhc/ffn/bias",
                                                      NumericFormat::FP32, {8}),
            };
        }
    }

    const TensorPlacement mtp_placement =
        features.mtp() ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        WeightPlan{.object = bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S,
                                      {TextConfig::hidden, TextConfig::mtp_input_rows}),
                   .format = NumericFormat::W8G32_F16S};
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16,
                                      {TextConfig::hidden});
    out.mtp.hidden_norm = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.input_norm =
        bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.query_key_gate_value =
        WeightPlan{.object = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                      NumericFormat::W8G32_F16S,
                                      {TextConfig::mtp_attention_input_rows, TextConfig::hidden}),
                   .format = NumericFormat::W8G32_F16S};
    out.mtp.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.output = WeightPlan{.object = bind_mtp("mtp/layer/attention/output",
                                                   NumericFormat::W8G32_F16S,
                                                   {TextConfig::hidden, TextConfig::query_size}),
                                .format = NumericFormat::W8G32_F16S};
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.mlp.gate_up =
        WeightPlan{.object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S,
                                      {2 * TextConfig::intermediate, TextConfig::hidden}),
                   .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{.object = bind_mtp("mtp/layer/mlp/down",
                                                     NumericFormat::W8G32_F16S,
                                                     {TextConfig::hidden,
                                                      TextConfig::intermediate}),
                                  .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {TextConfig::hidden});

    const TensorPlacement vision_placement =
        features.vision ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2 =
        artifact::bind_tensor(binder, "vision/merger/fc2", NumericFormat::W8G32_F16S,
                              {TextConfig::hidden, 4608}, vision_placement);
    out.vision_merger_fc2_bias =
        artifact::bind_tensor(binder, "vision/merger/fc2_bias", NumericFormat::BF16,
                              {TextConfig::hidden}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend              = qwen3_6::take_frontend_resources(backing, plan.frontend);
    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    runtime.token_embedding =
        materialized_weight(backing, plan.token_embedding, TextConfig::output_rows,
                            TextConfig::hidden);
    runtime.final_norm = artifact::materialized_tensor(backing, plan.final_norm,
                                                       NumericFormat::BF16, {TextConfig::hidden});
    runtime.output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                              TextConfig::hidden);

    for (std::size_t physical = 0; physical < TextConfig::physical_layers; ++physical) {
        const TextLayerPlan& source = plan.text_layers[physical];
        if (source.is_full_attention) {
            FullAttentionWeights& target =
                runtime.full_layers.at(static_cast<std::size_t>(
                    TextConfig::physical_full_index(static_cast<int>(physical))));
            target.input_norm = artifact::materialized_tensor(
                backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.projection.query_key =
                materialized_weight(backing, source.attention.query_key,
                                    TextConfig::query_size + TextConfig::kv_size,
                                    TextConfig::hidden);
            target.projection.gate_value =
                materialized_weight(backing, source.attention.gate_value,
                                    TextConfig::query_size + TextConfig::kv_size,
                                    TextConfig::hidden);
            target.query_norm = artifact::materialized_tensor(
                backing, source.attention.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.key_norm = artifact::materialized_tensor(
                backing, source.attention.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.output = materialized_weight(backing, source.attention.output,
                                                TextConfig::hidden, TextConfig::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = runtime.gdn_layers.at(static_cast<std::size_t>(
                TextConfig::physical_gdn_index(static_cast<int>(physical))));
            target.input_norm = artifact::materialized_tensor(
                backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.projection.a_log = artifact::materialized_tensor(
                backing, source.gdn.a_log, NumericFormat::FP32,
                {TextConfig::gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(
                backing, source.gdn.dt_bias, NumericFormat::FP32,
                {TextConfig::gdn_value_heads});
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16,
                {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            target.projection.control_projection.a_projection =
                materialized_weight(backing, source.gdn.a_projection, TextConfig::gdn_value_heads,
                                    TextConfig::hidden);
            target.projection.control_projection.b_projection =
                materialized_weight(backing, source.gdn.b_projection, TextConfig::gdn_value_heads,
                                    TextConfig::hidden);
            target.projection.input_projection.query_key =
                materialized_weight(backing, source.gdn.query_key, 2 * TextConfig::key_dim,
                                    TextConfig::hidden);
            target.projection.input_projection.value_z =
                materialized_weight(backing, source.gdn.value_z, 2 * TextConfig::value_dim,
                                    TextConfig::hidden);
            target.norm = artifact::materialized_tensor(backing, source.gdn.norm,
                                                        NumericFormat::BF16,
                                                        {TextConfig::gdn_key_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, TextConfig::hidden,
                                                TextConfig::value_dim);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
        if (source.mhc_attn && source.mhc_ffn) {
            runtime.mhc_attn[physical] = qwen3_6::MhcRouterWeights{
                .weight = artifact::materialized_tensor(backing, source.mhc_attn->weight,
                                                        NumericFormat::FP32,
                                                        {8, 2 * TextConfig::hidden}),
                .bias   = artifact::materialized_tensor(backing, source.mhc_attn->bias,
                                                        NumericFormat::FP32, {8}),
            };
            runtime.mhc_ffn[physical] = qwen3_6::MhcRouterWeights{
                .weight = artifact::materialized_tensor(backing, source.mhc_ffn->weight,
                                                        NumericFormat::FP32,
                                                        {8, 2 * TextConfig::hidden}),
                .bias   = artifact::materialized_tensor(backing, source.mhc_ffn->bias,
                                                        NumericFormat::FP32, {8}),
            };
        }
    }

    if (plan.features.mtp()) {
        auto& mtp = runtime.mtp.emplace();
        mtp.input_projection = materialized_weight(backing, plan.mtp.input_projection,
                                                   TextConfig::hidden, TextConfig::mtp_input_rows);
        mtp.embedding_norm   = artifact::materialized_tensor(
            backing, plan.mtp.embedding_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.hidden_norm = artifact::materialized_tensor(
            backing, plan.mtp.hidden_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.input_norm = artifact::materialized_tensor(
            backing, plan.mtp.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.attention.packed =
            materialized_weight(backing, plan.mtp.query_key_gate_value,
                                TextConfig::mtp_attention_input_rows, TextConfig::hidden);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, TextConfig::query_size);
        mtp.attention.key         = row_view(mtp.attention.packed, TextConfig::query_size,
                                             TextConfig::kv_size);
        mtp.attention.output_gate = row_view(mtp.attention.packed,
                                             TextConfig::query_size + TextConfig::kv_size,
                                             TextConfig::query_size);
        mtp.attention.value       = row_view(
            mtp.attention.packed, 2 * TextConfig::query_size + TextConfig::kv_size,
            TextConfig::kv_size);
        mtp.query_norm = artifact::materialized_tensor(
            backing, plan.mtp.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.key_norm = artifact::materialized_tensor(
            backing, plan.mtp.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.output = materialized_weight(backing, plan.mtp.output, TextConfig::hidden,
                                         TextConfig::query_size);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(
            backing, plan.mtp.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2 = materialized_weight(
            backing, WeightPlan{.object = plan.vision_merger_fc2,
                                .format = NumericFormat::W8G32_F16S},
            TextConfig::hidden, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(
            backing, plan.vision_merger_fc2_bias, NumericFormat::BF16, {TextConfig::hidden});
    }
}

} // namespace ninfer::targets::sensenova_loop_9b::detail
