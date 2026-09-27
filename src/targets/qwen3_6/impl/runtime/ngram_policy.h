#pragma once

#include "targets/qwen3_6/impl/runtime/ngram_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {

// A round verifies the full n-gram window V only when some row's usable draft reaches the MTP
// depth k plus this margin; otherwise it keeps the MTP width k+1 and its cost.
inline constexpr std::uint32_t kNgramWideRoundMargin = 3;

/**
 * Chain draft for one row: the MTP proposal followed by the pool's continuation of
 * ledger + proposal. Writes the proposal and then at most out.size() - proposal.size() pool
 * tokens to `out`, and returns the total. A pool extension shorter than `min_extension` is
 * dropped. `context` is caller scratch of at least match_tokens + proposal.size() tokens: only
 * the last n tokens of ledger + proposal enter the lookup. Performs no allocation.
 */
[[nodiscard]] inline std::size_t chain_ngram_drafts(const NgramDraftPool& pool,
                                                    std::span<const TokenId> ledger,
                                                    std::span<const TokenId> proposal,
                                                    std::size_t min_extension,
                                                    std::span<TokenId> context,
                                                    std::span<TokenId> out) {
    const std::size_t n = pool.match_tokens();
    if (proposal.size() > out.size() || context.size() < n + proposal.size()) {
        throw std::invalid_argument("n-gram chain draft buffers are too small");
    }
    std::copy(proposal.begin(), proposal.end(), out.begin());
    const std::size_t from_ledger = std::min(ledger.size(), n);
    std::copy(ledger.end() - static_cast<std::ptrdiff_t>(from_ledger), ledger.end(),
              context.begin());
    std::copy(proposal.begin(), proposal.end(),
              context.begin() + static_cast<std::ptrdiff_t>(from_ledger));
    std::size_t extension =
        pool.propose(context.first(from_ledger + proposal.size()), out.subspan(proposal.size()));
    if (extension < min_extension) { extension = 0; }
    return proposal.size() + extension;
}

/**
 * Verify drafts for an MTP round with n-gram chaining. `usable` holds each row's draft count
 * after the budget and context clamps (at most `verify_window`). Returns `verify_window` when
 * some row reaches depth + kNgramWideRoundMargin, otherwise `depth`.
 */
[[nodiscard]] inline std::uint32_t ngram_round_verify_drafts(std::span<const std::uint32_t> usable,
                                                             std::uint32_t depth,
                                                             std::uint32_t verify_window) {
    if (depth == 0 || verify_window < depth) {
        throw std::invalid_argument("n-gram round needs 1 <= depth <= verify window");
    }
    const bool wide = std::any_of(usable.begin(), usable.end(), [&](std::uint32_t count) {
        return count >= depth + kNgramWideRoundMargin;
    });
    return wide ? verify_window : depth;
}

struct DraftSourceCounts {
    std::uint32_t mtp_drafted    = 0;
    std::uint32_t mtp_accepted   = 0;
    std::uint32_t ngram_drafted  = 0;
    std::uint32_t ngram_accepted = 0;
};

/**
 * Attributes a verified row to its sources. The first `mtp_drafts` positions of a chain draft
 * come from MTP and the rest from the pool, so of `accepted` leading drafts the first
 * min(accepted, MTP part) are MTP's.
 */
[[nodiscard]] inline DraftSourceCounts draft_source_counts(std::uint32_t extent,
                                                           std::uint32_t mtp_drafts,
                                                           std::uint32_t accepted) {
    if (accepted > extent) {
        throw std::invalid_argument("accepted drafts exceed the verified extent");
    }
    const std::uint32_t mtp = std::min(mtp_drafts, extent);
    DraftSourceCounts counts;
    counts.mtp_drafted    = mtp;
    counts.ngram_drafted  = extent - mtp;
    counts.mtp_accepted   = std::min(accepted, mtp);
    counts.ngram_accepted = accepted - counts.mtp_accepted;
    return counts;
}

} // namespace ninfer::targets::qwen3_6
