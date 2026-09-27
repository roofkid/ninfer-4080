#include "targets/qwen3_6/impl/runtime/ngram_policy.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::targets::qwen3_6::chain_ngram_drafts;
using ninfer::targets::qwen3_6::draft_source_counts;
using ninfer::targets::qwen3_6::kNgramWideRoundMargin;
using ninfer::targets::qwen3_6::NgramDraftPool;
using ninfer::targets::qwen3_6::ngram_round_verify_drafts;

constexpr std::int32_t kQwenTokenDomain = 248077;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

std::vector<TokenId> chain(const NgramDraftPool& pool, std::span<const TokenId> ledger,
                           std::span<const TokenId> proposal, std::size_t min_extension,
                           std::size_t window) {
    std::vector<TokenId> context(pool.match_tokens() + proposal.size());
    std::vector<TokenId> out(window);
    out.resize(chain_ngram_drafts(pool, ledger, proposal, min_extension, context, out));
    return out;
}

// A span seen earlier continues after the MTP proposal when the proposal matches it: the lookup
// key is the last n tokens of ledger + proposal, not of the ledger alone.
void test_chain_extends_after_the_proposal() {
    NgramDraftPool pool({.match_tokens = 3, .entries = 1U << 16, .token_domain = kQwenTokenDomain});
    const std::vector<TokenId> earlier{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    pool.observe(earlier, 0);

    const std::vector<TokenId> ledger{90, 91, 10, 11};
    const std::array<TokenId, 2> proposal{12, 13};
    const auto drafts = chain(pool, ledger, proposal, 1, 8);
    check(drafts == std::vector<TokenId>({12, 13, 14, 15, 16, 17, 18, 19}),
          "chain must keep the proposal and extend it from ledger + proposal");

    // The window bounds the extension; the proposal is never truncated by the policy.
    check(chain(pool, ledger, proposal, 1, 4) == std::vector<TokenId>({12, 13, 14, 15}),
          "chain must stop at the verify window");

    // No MTP proposal (fallback round): the pool continues the ledger itself.
    const std::vector<TokenId> ledger_only{90, 11, 12, 13};
    check(chain(pool, ledger_only, {}, 1, 3) == std::vector<TokenId>({14, 15, 16}),
          "chain without a proposal must continue the ledger");

    // A proposal that diverges from the pool gets no extension.
    const std::array<TokenId, 2> diverging{12, 77};
    check(chain(pool, ledger, diverging, 1, 8) == std::vector<TokenId>({12, 77}),
          "a diverging proposal must not be extended");
}

void test_short_extensions_are_dropped() {
    NgramDraftPool pool({.match_tokens = 2, .entries = 1U << 12, .token_domain = kQwenTokenDomain});
    const std::vector<TokenId> earlier{1, 2, 3, 4};
    pool.observe(earlier, 0);
    const std::vector<TokenId> ledger{9, 1};
    const std::array<TokenId, 1> proposal{2};
    check(chain(pool, ledger, proposal, 2, 8) == std::vector<TokenId>({2, 3, 4}),
          "an extension at the minimum length must be kept");
    check(chain(pool, ledger, proposal, 3, 8) == std::vector<TokenId>({2}),
          "an extension below the minimum length must be dropped");
}

void test_short_history_and_buffers() {
    NgramDraftPool pool({.match_tokens = 4, .entries = 1U << 12, .token_domain = kQwenTokenDomain});
    const std::vector<TokenId> earlier{1, 2, 3, 4, 5};
    pool.observe(earlier, 0);
    // Fewer than n tokens of ledger + proposal: no lookup, the proposal alone.
    const std::vector<TokenId> ledger{2};
    const std::array<TokenId, 2> proposal{3, 4};
    check(chain(pool, ledger, proposal, 1, 8) == std::vector<TokenId>({3, 4}),
          "a context shorter than n must yield only the proposal");

    std::array<TokenId, 6> context{};
    std::array<TokenId, 1> small_out{};
    check(throws_invalid([&] {
              (void)chain_ngram_drafts(pool, ledger, proposal, 1, context, small_out);
          }),
          "an output smaller than the proposal must be rejected");
    std::array<TokenId, 8> out{};
    check(!throws_invalid([&] {
              (void)chain_ngram_drafts(pool, ledger, proposal, 1, context, out);
          }),
          "a context of n + proposal tokens is enough");
    check(throws_invalid([&] {
              (void)chain_ngram_drafts(pool, ledger, proposal, 1,
                                       std::span<TokenId>(context).first(5), out);
          }),
          "a context shorter than n + proposal must be rejected");
}

void test_round_width_policy() {
    const std::uint32_t k = 3;
    const std::uint32_t v = 15;
    const std::array<std::uint32_t, 3> mtp_only{3, 2, 0};
    check(ngram_round_verify_drafts(mtp_only, k, v) == k, "MTP-only rows keep width k+1");
    const std::array<std::uint32_t, 2> below{k + kNgramWideRoundMargin - 1, 1};
    check(ngram_round_verify_drafts(below, k, v) == k,
          "a draft below the margin must not widen the round");
    const std::array<std::uint32_t, 3> one_wide{1, k + kNgramWideRoundMargin, 0};
    check(ngram_round_verify_drafts(one_wide, k, v) == v, "one long draft widens the round");
    check(ngram_round_verify_drafts({}, k, v) == k, "an empty batch keeps width k+1");
    check(throws_invalid([] { (void)ngram_round_verify_drafts({}, 4, 3); }),
          "a verify window below the depth must be rejected");
}

void test_source_attribution() {
    auto counts = draft_source_counts(8, 3, 5);
    check(counts.mtp_drafted == 3 && counts.ngram_drafted == 5 && counts.mtp_accepted == 3 &&
              counts.ngram_accepted == 2,
          "acceptance past the proposal belongs to the pool");
    counts = draft_source_counts(8, 3, 2);
    check(counts.mtp_accepted == 2 && counts.ngram_accepted == 0,
          "acceptance inside the proposal belongs to MTP");
    // The round clamp can cut the proposal itself.
    counts = draft_source_counts(2, 3, 2);
    check(counts.mtp_drafted == 2 && counts.ngram_drafted == 0 && counts.mtp_accepted == 2,
          "a clamped proposal counts only its verified part");
    counts = draft_source_counts(4, 0, 4);
    check(counts.mtp_drafted == 0 && counts.ngram_drafted == 4 && counts.ngram_accepted == 4,
          "a pool-only draft belongs to the pool");
    check(throws_invalid([] { (void)draft_source_counts(2, 2, 3); }),
          "acceptance beyond the extent must be rejected");
}

// Direct pool contract: the same window observes and proposes, a different tag misses, and the
// encoded window rolls with the drafted tokens.
void test_pool_roundtrip() {
    NgramDraftPool pool({.match_tokens = 4, .entries = 1U << 16, .token_domain = kQwenTokenDomain});
    const std::vector<TokenId> history{5, 6, 7, 8, 9, 10, 11, 12};
    pool.observe(history, 0);
    check(pool.occupied() > 0 && pool.memory_bytes() == (1U << 16) * sizeof(std::uint32_t),
          "pool reports its occupancy and fixed table size");

    const std::vector<TokenId> context{5, 6, 7, 8};
    std::array<TokenId, 4> out{};
    check(pool.propose(context, out) == 4 && out == std::array<TokenId, 4>({9, 10, 11, 12}),
          "propose walks the observed continuation");

    // A window whose n-gram was never observed misses at its first step.
    const std::vector<TokenId> unseen{5, 6, 7, 99};
    std::fill(out.begin(), out.end(), TokenId{});
    check(pool.propose(unseen, out) == 0, "an unseen window misses");

    // Observation ahead of the history and out-of-domain continuations are rejected.
    check(throws_invalid([&] { pool.observe(history, history.size() + 1); }),
          "observation past the history must be rejected");
    check(throws_invalid([&] {
              NgramDraftPool small(
                  {.match_tokens = 1, .entries = 16, .token_domain = 4});
              const std::vector<TokenId> bad{1, 7};
              small.observe(bad, 0);
          }),
          "a continuation outside the token domain must be rejected");

    // Invalid specifications are rejected before any table is built.
    check(throws_invalid([] { NgramDraftPool p({.match_tokens = 0, .entries = 16, .token_domain = 8}); }),
          "a zero match length must be rejected");
    check(throws_invalid([] { NgramDraftPool p({.match_tokens = 8, .entries = 0, .token_domain = 8}); }),
          "a zero-entry pool must be rejected");
    check(throws_invalid([] { NgramDraftPool p({.match_tokens = 8, .entries = 16, .token_domain = 0}); }),
          "a zero token domain must be rejected");
}

} // namespace

int main() {
    test_chain_extends_after_the_proposal();
    test_short_extensions_are_dropped();
    test_short_history_and_buffers();
    test_round_width_policy();
    test_source_attribution();
    test_pool_roundtrip();
    if (failures != 0) {
        std::cerr << failures << " n-gram policy check(s) failed\n";
        return 1;
    }
    std::cout << "ngram policy: PASS\n";
    return 0;
}
