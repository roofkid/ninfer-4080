// Lossless check of MTP rounds extended by n-gram drafts, on a real artifact.
//
// Greedy verification of a one-hot draft is exact, but a round's width and partition change the
// floating-point summation order, so greedy text may flip at a near-tie. MTP alone already differs
// from the non-speculative route that way (and further with quantized KV, whose decode, verify and
// scoring routes read the cache differently), so the n-gram criterion is relative to MTP:
//   (a) with BF16 KV (blocking), every prompt where MTP+n-gram diverges from the non-speculative
//       route before MTP alone does is a new divergence, and it must be a tie: the two candidates
//       within kTieNats under the scoring route, an independent evaluation of the same prefix;
//       with quantized KV the divergences are reported only;
//   (b) divergences per 1000 compared tokens are reported for both configurations;
//   (c) the same configuration in two fresh Engines produces identical token ids (blocking).
// The engine currently verifies at most the MTP width K (the wider n-gram verify window is
// disabled pending the root cause recorded in the plan's Stage 5c.3 note), so this run cannot
// yet require wide rounds; it protects the lossless criterion of the enabled path.
//
// Environment: NINFER_TEST_ARTIFACT (required), NINFER_NGRAM_DRAFT_TOKENS (MTP depth, default 3),
// NINFER_NGRAM_KV_DTYPE (bf16|int8|fp8|rk4v4-e8, default bf16), NINFER_NGRAM_MAX_NEW (default 384).

#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

// Scored log-probabilities differ by the difference of two BF16 logits: one BF16 step is 1/16 nat
// for |logit| in [8,16), the binade the observed 1/16-quantized gaps come from. A tie is at most
// one step apart.
constexpr double kTieNats = 0.0625;

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    return static_cast<std::uint32_t>(std::stoul(value));
}

ninfer::KvCacheStorage kv_under_test() {
    const char* value           = std::getenv("NINFER_NGRAM_KV_DTYPE");
    const std::string_view text = value == nullptr || *value == '\0' ? "bf16" : value;
    if (text == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (text == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    if (text == "rk4v4-e8") { return ninfer::KvCacheStorage::RK4V4E8; }
    throw std::invalid_argument("NINFER_NGRAM_KV_DTYPE: unsupported KV-cache storage");
}

// Raw continuation prompts, so every route sees identical token ids without a chat template. Most
// carry structure the output repeats (code, records, tables, restated text), where the pool drafts.
std::vector<std::string> prompts() {
    std::string code = "import math\n\n\n";
    for (const char* name : {"area", "perimeter", "diagonal", "scale", "rotate"}) {
        code += "def rectangle_" + std::string(name) +
                "(width: float, height: float) -> float:\n"
                "    \"\"\"Return the " +
                name +
                " of a rectangle.\"\"\"\n"
                "    if width < 0 or height < 0:\n"
                "        raise ValueError(\"negative size\")\n"
                "    return math.hypot(width, height)\n\n\n";
    }
    code += "def rectangle_";

    std::string records = "[\n";
    const char* names[] = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot"};
    for (int i = 0; i < 6; ++i) {
        records += "  {\"id\": " + std::to_string(i + 1) + ", \"name\": \"" + names[i] +
                   "\", \"enabled\": true, \"port\": " + std::to_string(8080 + i) +
                   ", \"tags\": [\"edge\", \"cache\"]},\n";
    }
    records += "  {\"id\": 7, \"name\": \"golf\",";

    const std::string paragraph =
        "The build finished without warnings, the tests passed on the first run, and the release "
        "notes were updated before the tag was pushed to the shared repository.";
    const std::string restated = "Original: " + paragraph + "\nCopy 1: " + paragraph +
                                 "\nCopy 2: " + paragraph + "\nCopy 3:";

    std::string table = "| Step | Command | Expected result |\n|---|---|---|\n";
    for (int i = 1; i <= 6; ++i) {
        table += "| " + std::to_string(i) + " | `cmake --build build --target step" +
                 std::to_string(i) + "` | step" + std::to_string(i) + " builds without errors |\n";
    }
    table += "| 7 |";

    return {code,
            records,
            restated,
            table,
            "Write a short story about a lighthouse keeper.\n\nStory:",
            "Explain how a transformer language model generates text, step by step.\n\n1."};
}

ninfer::EngineOptions generation_options(const char* artifact, bool speculative, bool ngram) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 4096;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.use_cuda_graph  = env_u32("NINFER_NGRAM_GRAPH", 1) != 0;
    options.kv_cache        = kv_under_test();
    if (speculative) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = env_u32("NINFER_NGRAM_DRAFT_TOKENS", 3);
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        if (ngram) {
            options.speculative.ngram.mode       = ninfer::NgramDraftMode::Chain;
            options.speculative.ngram.max_drafts = env_u32("NINFER_NGRAM_MAX_DRAFTS", 15);
        }
    }
    return options;
}

struct Run {
    std::vector<std::vector<ninfer::TokenId>> outputs;
    std::uint64_t ngram_drafted  = 0;
    std::uint64_t ngram_accepted = 0;
    std::uint64_t wide_rounds    = 0;
    std::uint64_t rounds         = 0;
};

Run generate_all(ninfer::Engine& engine,
                 const std::vector<std::vector<ninfer::TokenId>>& prompt_ids) {
    Run run;
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = env_u32("NINFER_NGRAM_MAX_NEW", 384);
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    for (const auto& ids : prompt_ids) {
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(ids, false), request);
        run.outputs.push_back(result.generated_token_ids);
        run.ngram_drafted += result.speculative.ngram_drafted_tokens;
        run.ngram_accepted += result.speculative.ngram_accepted_tokens;
        run.wide_rounds += result.speculative.wide_rounds;
        run.rounds += result.speculative.rounds + result.speculative.fallback_steps;
    }
    return run;
}

std::optional<std::size_t> first_divergence(const std::vector<ninfer::TokenId>& a,
                                            const std::vector<ninfer::TokenId>& b) {
    const std::size_t common = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < common; ++i) {
        if (a[i] != b[i]) { return i; }
    }
    if (a.size() != b.size()) { return common; }
    return std::nullopt;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    std::vector<std::vector<ninfer::TokenId>> prompt_ids;
    Run ngram_first;
    Run ngram_second;
    Run mtp_only;
    Run reference;
    // The pool is shared by every request of an Engine, so determinism compares two fresh Engines
    // that see the same request sequence.
    {
        ninfer::Engine engine(generation_options(artifact, true, true));
        for (const std::string& text : prompts()) { prompt_ids.push_back(engine.tokenize_text(text)); }
        ngram_first = generate_all(engine, prompt_ids);
    }
    {
        ninfer::Engine engine(generation_options(artifact, true, true));
        ngram_second = generate_all(engine, prompt_ids);
    }
    {
        ninfer::Engine engine(generation_options(artifact, true, false));
        mtp_only = generate_all(engine, prompt_ids);
    }
    {
        ninfer::Engine engine(generation_options(artifact, false, false));
        reference = generate_all(engine, prompt_ids);
    }

    int failures = 0;
    // (c) Determinism: the same configuration twice gives identical ids.
    if (ngram_first.outputs != ngram_second.outputs) {
        std::cerr << "FAIL: the n-gram configuration is not deterministic across two runs\n";
        ++failures;
    }
    // Wide verify windows are disabled in the engine (see the plan's Stage 5c.3 note), so the pool
    // cannot extend a round past the MTP width yet. The criterion here is the lossless one:
    // enabling the chain must not change greedy output, and any divergence from the
    // non-speculative route must be a scoring tie.
    if (ngram_first.rounds == 0) {
        std::cerr << "FAIL: the speculative rounds did not run\n";
        ++failures;
    }


    ninfer::EngineOptions scoring_options;
    scoring_options.artifact_path = artifact;
    scoring_options.purpose       = ninfer::EnginePurpose::CausalScoring;
    scoring_options.max_context   = 4096;
    scoring_options.kv_cache      = kv_under_test();
    ninfer::Engine scorer(scoring_options);

    const bool blocking = kv_under_test() == ninfer::KvCacheStorage::BFloat16;
    // (b) Divergences of each configuration against the non-speculative route.
    std::vector<std::optional<std::size_t>> ngram_at(prompt_ids.size());
    std::vector<std::optional<std::size_t>> mtp_at(prompt_ids.size());
    for (const auto& [label, run, at] :
         {std::tuple<const char*, const Run*, std::vector<std::optional<std::size_t>>*>{
              "mtp+ngram", &ngram_first, &ngram_at},
          std::tuple<const char*, const Run*, std::vector<std::optional<std::size_t>>*>{
              "mtp", &mtp_only, &mtp_at}}) {
        std::size_t compared    = 0;
        std::size_t divergences = 0;
        for (std::size_t p = 0; p < prompt_ids.size(); ++p) {
            (*at)[p] = first_divergence(reference.outputs[p], run->outputs[p]);
            compared += (*at)[p] ? *(*at)[p] + 1 : reference.outputs[p].size();
            divergences += (*at)[p] ? 1U : 0U;
        }
        std::cout << label << ": " << divergences << " divergences in " << compared
                  << " compared tokens (" << std::setprecision(3)
                  << (compared == 0 ? 0.0 : 1000.0 * static_cast<double>(divergences) / compared)
                  << " per 1000)\n";
        for (std::size_t p = 0; p < prompt_ids.size(); ++p) {
            if ((*at)[p]) {
                std::cout << "  " << label << " prompt " << p << " diverged at token " << *(*at)[p]
                          << " reference " << reference.outputs[p][*(*at)[p]] << " vs "
                          << run->outputs[p][*(*at)[p]] << "\n";
            }
        }
    }

    // (a) Divergences the n-gram adds over MTP alone must be ties. The baseline MTP divergences
    // are scored too, so the report shows the route-vs-route gap distribution the tie bound
    // must cover.
    const auto score_divergence = [&](const char* label, std::size_t p, std::size_t at,
                                      const std::vector<ninfer::TokenId>& output) {
        const auto& expected = reference.outputs[p];
        if (at >= expected.size() || at >= output.size()) {
            std::cerr << "FAIL: " << label << " prompt " << p << " output length differs\n";
            return -1.0;
        }
        std::vector<ninfer::TokenId> context = prompt_ids[p];
        context.insert(context.end(), expected.begin(),
                       expected.begin() + static_cast<std::ptrdiff_t>(at));
        const auto target = static_cast<std::uint32_t>(context.size());
        std::vector<ninfer::TokenId> with_reference = context;
        with_reference.push_back(expected[at]);
        std::vector<ninfer::TokenId> with_actual = context;
        with_actual.push_back(output[at]);
        const float reference_logprob = scorer.score_tokens(with_reference, target).back();
        const float actual_logprob    = scorer.score_tokens(with_actual, target).back();
        const double gap =
            std::abs(static_cast<double>(reference_logprob) - static_cast<double>(actual_logprob));
        const bool tie = gap <= kTieNats;
        std::cout << label << ": prompt " << p << " token " << at << ": reference "
                  << expected[at] << " (" << std::setprecision(6) << reference_logprob << "), "
                  << output[at] << " (" << actual_logprob << "), gap " << gap << " nats"
                  << (tie ? " (tie)" : " (not a tie)") << '\n';
        return gap;
    };

    std::size_t added = 0;
    for (std::size_t p = 0; p < prompt_ids.size(); ++p) {
        if (!mtp_at[p]) { continue; }
        const double gap = score_divergence("mtp baseline", p, *mtp_at[p], mtp_only.outputs[p]);
        (void)gap;
    }
    for (std::size_t p = 0; p < prompt_ids.size(); ++p) {
        const std::optional<std::size_t> at = ngram_at[p];
        if (!at || (mtp_at[p] && *mtp_at[p] <= *at)) { continue; }
        ++added;
        const double gap = score_divergence("n-gram adds", p, *at, ngram_first.outputs[p]);
        if (blocking && (gap < 0.0 || gap > kTieNats)) { ++failures; }
        }
    std::cout << "n-gram added " << added << " divergence(s) over MTP alone"
              << (blocking ? "" : " (quantized KV: reported, not blocking)") << '\n';
    std::cout << "mtp+ngram pool: drafted " << ngram_first.ngram_drafted << ", accepted "
              << ngram_first.ngram_accepted << ", wide rounds " << ngram_first.wide_rounds
              << " of " << ngram_first.rounds << "\n";

    if (failures != 0) {
        std::cerr << failures << " n-gram lossless check(s) failed\n";
        return 1;
    }
    std::cout << "ngram lossless: PASS\n";
    return 0;
}
