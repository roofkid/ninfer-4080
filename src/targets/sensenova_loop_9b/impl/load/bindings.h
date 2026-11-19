#pragma once

#include <ninfer/targets/sensenova_loop_9b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "core/tensor.h"
#include "targets/sensenova_loop_9b/impl/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::sensenova_loop_9b::detail {

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct MhcRouterPlan {
    artifact::ObjectHandle weight;
    artifact::ObjectHandle bias;
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
};

struct FullAttentionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    WeightPlan a_projection;
    WeightPlan b_projection;
    WeightPlan query_key;
    WeightPlan value_z;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    std::optional<MhcRouterPlan> mhc_attn;
    std::optional<MhcRouterPlan> mhc_ffn;
};

struct MtpPlan {
    WeightPlan input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    WeightPlan token_embedding;
    std::array<TextLayerPlan, TextConfig::physical_layers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    MtpPlan mtp;
    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    SplitGdnControlProjectionPayload control_projection;
    SplitGdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<SplitAttentionProjectionPayload, GdnProjectionPayload,
                       DensePostMixerPayload, MtpAttentionPayload, DensePostMixerPayload,
                       qwen3_6::DFlash2Weights, TextConfig::full_attention_layers(),
                       TextConfig::gdn_layers()>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail
