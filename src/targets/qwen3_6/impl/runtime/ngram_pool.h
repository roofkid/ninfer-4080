#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct NgramPoolSpec {
    // n: tokens in the lookup key.
    std::uint32_t match_tokens = 0;
    // Fixed slot count; memory is entries * 4 bytes, allocated once at construction.
    std::size_t entries = 0;
    // Continuation ids must be in [0,token_domain).
    std::int32_t token_domain = 0;
};

/**
 * Host n-gram draft pool in the style of llama.cpp `ngram-mod`.
 *
 * One fixed table maps the hash of an n-token window to the token that most recently followed
 * it. Walking the table from the last n context tokens yields a variable-length draft without a
 * draft model; greedy verification of the draft is lossless, so a wrong entry costs only width.
 *
 * Window hash: h = sum_i u32(w[i]) * M^(n-1-i) mod 2^64 (M = 6364136223846793005), which rolls
 * in O(1) per drafted token. Slot = fmix64(h) mod entries. Each 32-bit entry stores
 * (tag << 18) | (token + 1), where tag is the top 14 bits of fmix64(h) and 0 means empty; a
 * lookup whose tag differs is a miss, which suppresses most slot collisions without extra
 * memory.
 *
 * The pool is not synchronized. It owns no per-lane state: a lane's history is its token ledger,
 * and the caller reports what is new through observe(history, first_new).
 */
class NgramDraftPool {
public:
    static constexpr std::uint64_t kLcgMultiplier      = 6364136223846793005ULL;
    static constexpr std::uint32_t kTokenBits          = 18;
    static constexpr std::uint32_t kTagBits            = 32 - kTokenBits;
    static constexpr std::uint32_t kTokenMask          = (1U << kTokenBits) - 1U;
    static constexpr std::int32_t kMaximumTokenDomain  = static_cast<std::int32_t>(kTokenMask);
    static constexpr std::uint32_t kMaximumMatchTokens = 64;

    explicit NgramDraftPool(const NgramPoolSpec& spec)
        : spec_(validated(spec)), drop_(power(kLcgMultiplier, spec.match_tokens)),
          table_(spec.entries, 0U) {}

    NgramDraftPool(const NgramDraftPool&)            = delete;
    NgramDraftPool& operator=(const NgramDraftPool&) = delete;
    NgramDraftPool(NgramDraftPool&&)                 = delete;
    NgramDraftPool& operator=(NgramDraftPool&&)      = delete;

    [[nodiscard]] std::uint32_t match_tokens() const noexcept { return spec_.match_tokens; }

    [[nodiscard]] std::size_t entries() const noexcept { return table_.size(); }

    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return table_.size() * sizeof(std::uint32_t);
    }

    [[nodiscard]] std::size_t occupied() const noexcept { return occupied_; }

    /**
     * Records every n-gram of `history` whose continuation index is in [first_new, size): after
     * a prefill that reused P prompt tokens pass (prompt, P); after committing C decode tokens
     * pass (ledger, ledger.size() - C). Later records overwrite earlier ones. Throws
     * std::invalid_argument, leaving the pool unchanged, when first_new exceeds the history or a
     * recorded continuation is outside the token domain. Performs no allocation.
     */
    void observe(std::span<const TokenId> history, std::size_t first_new) {
        const std::size_t n = spec_.match_tokens;
        if (first_new > history.size()) {
            throw std::invalid_argument("n-gram pool observation starts past its history");
        }
        const std::size_t first_value = first_new > n ? first_new : n;
        for (std::size_t i = first_value; i < history.size(); ++i) {
            if (history[i] < 0 || history[i] >= spec_.token_domain) {
                throw std::invalid_argument("n-gram pool continuation is outside the domain");
            }
        }
        if (first_value >= history.size()) { return; }
        std::uint64_t window = window_hash(history.subspan(first_value - n, n));
        for (std::size_t i = first_value; i < history.size(); ++i) {
            const std::uint64_t mixed = mix(window);
            std::uint32_t& entry      = table_[static_cast<std::size_t>(mixed % table_.size())];
            if (entry == 0U) { ++occupied_; }
            entry  = encode(mixed, history[i]);
            window = roll(window, history[i - n], history[i]);
        }
    }

    /**
     * Walks the pool from the last n tokens of `context` and writes at most out.size() drafted
     * tokens to `out`, stopping at the first miss. Returns the draft length, which is zero when
     * the context is shorter than n. Performs no allocation and does not modify the pool.
     */
    [[nodiscard]] std::size_t propose(std::span<const TokenId> context,
                                      std::span<TokenId> out) const noexcept {
        const std::size_t n = spec_.match_tokens;
        if (context.size() < n || out.empty()) { return 0; }
        std::uint64_t window = window_hash(context.last(n));
        std::size_t count    = 0;
        while (count < out.size()) {
            const std::uint64_t mixed = mix(window);
            const std::uint32_t entry = table_[static_cast<std::size_t>(mixed % table_.size())];
            if (entry == 0U || (entry >> kTokenBits) != tag(mixed)) { break; }
            const auto token         = static_cast<TokenId>((entry & kTokenMask) - 1U);
            const std::size_t oldest = context.size() + count - n;
            const TokenId leaving =
                oldest < context.size() ? context[oldest] : out[oldest - context.size()];
            out[count++] = token;
            window       = roll(window, leaving, token);
        }
        return count;
    }

    void clear() noexcept {
        for (std::uint32_t& entry : table_) { entry = 0U; }
        occupied_ = 0;
    }

    [[nodiscard]] static std::uint64_t window_hash(std::span<const TokenId> window) noexcept {
        std::uint64_t value = 0;
        for (const TokenId token : window) { value = value * kLcgMultiplier + bits(token); }
        return value;
    }

    // MurmurHash3 fmix64 finalizer.
    [[nodiscard]] static constexpr std::uint64_t mix(std::uint64_t value) noexcept {
        value ^= value >> 33U;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33U;
        value *= 0xc4ceb9fe1a85ec53ULL;
        value ^= value >> 33U;
        return value;
    }

    [[nodiscard]] static constexpr std::size_t slot(std::uint64_t window,
                                                    std::size_t entries) noexcept {
        return static_cast<std::size_t>(mix(window) % entries);
    }

    [[nodiscard]] static constexpr std::uint32_t tag(std::uint64_t mixed) noexcept {
        return static_cast<std::uint32_t>(mixed >> (64U - kTagBits));
    }

private:
    [[nodiscard]] static NgramPoolSpec validated(const NgramPoolSpec& spec) {
        if (spec.match_tokens == 0 || spec.match_tokens > kMaximumMatchTokens) {
            throw std::invalid_argument("n-gram pool match length must be in [1,64]");
        }
        if (spec.entries == 0) {
            throw std::invalid_argument("n-gram pool needs at least one entry");
        }
        if (spec.token_domain <= 0 || spec.token_domain > kMaximumTokenDomain) {
            throw std::invalid_argument("n-gram pool token domain must be in [1,2^18-1]");
        }
        return spec;
    }

    [[nodiscard]] static constexpr std::uint64_t bits(TokenId token) noexcept {
        return static_cast<std::uint32_t>(token);
    }

    [[nodiscard]] static constexpr std::uint64_t power(std::uint64_t base,
                                                       std::uint32_t exponent) noexcept {
        std::uint64_t value = 1;
        for (std::uint32_t i = 0; i < exponent; ++i) { value *= base; }
        return value;
    }

    [[nodiscard]] static constexpr std::uint32_t encode(std::uint64_t mixed,
                                                        TokenId token) noexcept {
        return (tag(mixed) << kTokenBits) | (static_cast<std::uint32_t>(token) + 1U);
    }

    [[nodiscard]] std::uint64_t roll(std::uint64_t window, TokenId leaving,
                                     TokenId entering) const noexcept {
        return window * kLcgMultiplier + bits(entering) - bits(leaving) * drop_;
    }

    NgramPoolSpec spec_;
    std::uint64_t drop_ = 0;
    std::vector<std::uint32_t> table_;
    std::size_t occupied_ = 0;
};

} // namespace ninfer::targets::qwen3_6
