#pragma once

#include "ninfer/types.h"

#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

[[nodiscard]] inline NgramDraftMode parse_ngram_mode(std::string_view value) {
    if (value == "off") { return NgramDraftMode::Off; }
    if (value == "chain") { return NgramDraftMode::Chain; }
    throw std::invalid_argument("invalid n-gram mode: " + std::string(value));
}

[[nodiscard]] inline const char* ngram_mode_name(NgramDraftMode mode) noexcept {
    switch (mode) {
    case NgramDraftMode::Off:
        return "off";
    case NgramDraftMode::Chain:
        return "chain";
    }
    return "unknown";
}

[[nodiscard]] inline bool is_ngram_cli_flag(std::string_view flag) noexcept {
    return flag == "--ngram" || flag == "--ngram-max" || flag == "--ngram-n" ||
           flag == "--ngram-min" || flag == "--ngram-pool-mib";
}

// Applies one flag accepted by is_ngram_cli_flag; ranges are checked by
// validate_speculative_cli_options.
inline void apply_ngram_cli_option(std::string_view flag, std::string_view value,
                                   NgramOptions& ngram) {
    if (flag == "--ngram") {
        ngram.mode = parse_ngram_mode(value);
        return;
    }
    std::uint64_t number = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size() ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(flag) + " requires an unsigned integer");
    }
    const auto count = static_cast<std::uint32_t>(number);
    if (flag == "--ngram-max") {
        ngram.max_drafts = count;
    } else if (flag == "--ngram-n") {
        ngram.match_tokens = count;
    } else if (flag == "--ngram-min") {
        ngram.min_drafts = count;
    } else if (flag == "--ngram-pool-mib") {
        ngram.pool_bytes = number << 20U;
    } else {
        throw std::invalid_argument("unknown n-gram flag: " + std::string(flag));
    }
}

// The n-gram chain extends the MTP proposal, so it is on by default for `--spec mtp` and has no
// effect elsewhere. Every other backend clears it here, before validation, so the engine plan,
// request logs and metrics all report the effective mode.
inline void normalize_speculative_options(SpeculativeOptions& options) {
    if (options.backend != SpeculativeBackend::Mtp) {
        options.ngram = NgramOptions{.mode = NgramDraftMode::Off};
    }
}

inline void validate_ngram_cli_options(const SpeculativeOptions& options) {
    const NgramOptions& ngram = options.ngram;
    if (ngram.mode == NgramDraftMode::Off) {
        const NgramOptions defaults;
        if (ngram.max_drafts != defaults.max_drafts || ngram.match_tokens != defaults.match_tokens ||
            ngram.min_drafts != defaults.min_drafts || ngram.pool_bytes != defaults.pool_bytes) {
            throw std::invalid_argument(
                "--ngram-max, --ngram-n, --ngram-min and --ngram-pool-mib require --ngram chain");
        }
        return;
    }
    if (options.backend != SpeculativeBackend::Mtp) {
        // normalize_speculative_options clears non-MTP n-gram options before validation.
        return;
    }
    if (ngram.max_drafts < options.draft_tokens + 3 || ngram.max_drafts > 15) {
        throw std::invalid_argument("--ngram-max must be in [--draft-tokens + 3, 15]");
    }
    if (ngram.match_tokens == 0 || ngram.match_tokens > 64) {
        throw std::invalid_argument("--ngram-n must be in [1,64]");
    }
    if (ngram.min_drafts == 0 || ngram.min_drafts > ngram.max_drafts) {
        throw std::invalid_argument("--ngram-min must be in [1, --ngram-max]");
    }
    if (ngram.pool_bytes == 0 || ngram.pool_bytes > (4096ULL << 20U)) {
        throw std::invalid_argument("--ngram-pool-mib must be in [1,4096]");
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    validate_ngram_cli_options(options);
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
