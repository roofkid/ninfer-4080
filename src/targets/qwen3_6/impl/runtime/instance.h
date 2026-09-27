#pragma once

#ifndef NINFER_QWEN36_VARIANT
#    error "NINFER_QWEN36_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN36_RUNTIME_NS
#    error "NINFER_QWEN36_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/ops/attention_geometry.h>
#include <ninfer/ops/softmax_attention.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <algorithm>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using Variant                        = NINFER_QWEN36_VARIANT;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan            = qwen3_6::SequencePlan<Variant>;
using SequencePlanner         = qwen3_6::SequencePlanner<Variant>;
using RequestBasePlan         = qwen3_6::RequestBasePlan<Variant>;
using AdmissionCandidate      = qwen3_6::AdmissionCandidate<Variant>;
using PressurePlanningSession = qwen3_6::PressurePlanningSession<Variant>;
using PressureTargetHandle    = qwen3_6::PressureTargetHandle;
using ResourcePlan            = qwen3_6::ResourcePlan<Variant>;
using PersistentBackfillProof = qwen3_6::PersistentBackfillProof<Variant>;
using SequenceHandle          = qwen3_6::SequenceHandle<Variant>;
using ContinuationHandle      = qwen3_6::ContinuationHandle<Variant>;
using SharedPrefixHandle      = qwen3_6::SharedPrefixHandle<Variant>;
using CaptureOffer            = qwen3_6::CaptureOffer<Variant>;
using CaptureAssessment       = qwen3_6::CaptureAssessment;
using ActiveCaptureResult     = qwen3_6::ActiveCaptureResult<Variant>;
using MaterializationResult   = qwen3_6::MaterializationResult<Variant>;
using PendingBatch            = qwen3_6::PendingBatch<Variant>;
using PrefillProgress         = qwen3_6::PrefillProgress<Variant>;
using StartResult             = qwen3_6::StartResult<Variant>;
using CommitResult            = qwen3_6::CommitResult<Variant>;
using DiscardResult           = qwen3_6::DiscardResult<Variant>;
using FinishResult            = qwen3_6::FinishResult<Variant>;
using AbortResult             = qwen3_6::AbortResult<Variant>;
using ReleaseResult           = qwen3_6::ReleaseResult<Variant>;
using ContractAccess          = qwen3_6::detail::RuntimeContractAccess<Variant>;
using Program                 = qwen3_6::Program<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

// Profile ends for rounds verifying `verify_drafts` drafts with MTP depth K. The wide family
// splits wherever the causal attention route changes, so one captured executable covers each
// range and no update crosses a route boundary.
// The same preferred-end partition the Variant profile helpers use: contiguous ranges covering
// [0,max_frontier].
inline std::vector<GraphExecutionProfile> mtp_profile_ranges(
    std::uint32_t max_frontier, const std::vector<std::uint32_t>& preferred_ends) {
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

inline std::vector<GraphExecutionProfile> mtp_wide_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t verify_window,
    std::uint32_t batch_size, KvCacheStorage kv_storage) {
    if (capacity == 0 || draft_window == 0 || verify_window <= draft_window ||
        verify_window > kMtpVerifyMaximumDrafts || batch_size == 0) {
        throw std::invalid_argument("invalid MTP wide graph dimensions");
    }
    const ops::AttentionHeadGeometry attention{TextConfig::head_dim, TextConfig::query_heads,
                                               TextConfig::kv_heads};
    const std::uint32_t block = verify_window + 1;
    const auto topology = [&](std::uint32_t frontier) {
        const auto visible = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(frontier) + block));
        return ops::causal_softmax_attention_topology_class(
            attention, kv_storage, {1, visible}, static_cast<std::int32_t>(block),
            static_cast<std::int32_t>(batch_size));
    };
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, verify_window + draft_window);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    // Split a range where the attention class changes (it changes once, at a visible-key limit),
    // so short contexts keep their cheaper route instead of the class at the range's maximum.
    std::vector<std::uint32_t> split_ends;
    const auto ranges = mtp_profile_ranges(capacity - 1, ends);
    for (const GraphExecutionProfile range : ranges) {
        if (topology(range.min) != topology(range.max)) {
            std::uint32_t low  = range.min;
            std::uint32_t high = range.max;
            while (low + 1 < high) {
                const std::uint32_t middle = low + (high - low) / 2;
                (topology(middle) == topology(range.min) ? low : high) = middle;
            }
            split_ends.push_back(low);
        }
        split_ends.push_back(range.max);
    }
    std::vector<GraphExecutionProfile> profiles =
        mtp_profile_ranges(capacity - 1, split_ends);
    for (GraphExecutionProfile& profile : profiles) { profile.topology_class = topology(profile.max); }
    return profiles;
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
