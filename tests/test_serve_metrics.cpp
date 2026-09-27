#include "serve/serve_metrics.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <string>

namespace {

using ninfer::serve::GenerationMetrics;
using ninfer::serve::GenerationOutcome;
using ninfer::serve::ServeMetrics;

int check(bool ok, const char* label) {
    if (!ok) { std::printf("FAIL %s\n", label); }
    return ok ? 0 : 1;
}

std::map<std::string, double> parse(const std::string& body) {
    std::map<std::string, double> values;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        const auto space = line.find(' ');
        if (space == std::string::npos) { continue; }
        values[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return values;
}

GenerationOutcome outcome(int prompt, std::uint32_t cached, int completion, double prefill_s,
                          double decode_s, std::uint64_t drafted, std::uint64_t accepted) {
    GenerationOutcome out;
    out.prompt_tokens                        = prompt;
    out.completion_tokens                    = completion;
    out.metrics.prefix_cache_hit_tokens      = cached;
    out.metrics.prefill_seconds              = prefill_s;
    out.metrics.decode_seconds               = decode_s;
    out.metrics.speculative_draft_tokens     = drafted;
    out.metrics.speculative_accepted_tokens  = accepted;
    // A third of the drafts and of the accepted drafts came from the n-gram pool.
    out.metrics.speculative_ngram_draft_tokens    = drafted / 3;
    out.metrics.speculative_ngram_accepted_tokens = accepted / 3;
    return out;
}

} // namespace

int main() {
    int failures = 0;

    ServeMetrics metrics;
    // The four llamacpp counters flow straight from the Engine's live totals.
    ninfer::RuntimeStats live;
    const auto empty = parse(metrics.render(1, live, 0));
    failures += check(empty.at("llamacpp:prompt_tokens_total") == 0.0, "starts at zero");
    const auto never = metrics.last_completed();
    failures += check(never.prompt_tokens == 0 && never.cached_tokens == 0,
                      "last completed starts at zero");
    failures += check(empty.at("ninfer:requests_total") == 0.0, "requests start at zero");
    failures += check(empty.at("llamacpp:requests_processing") == 0.0, "idle processing");
    failures += check(empty.at("llamacpp:requests_deferred") == 0.0, "idle deferred");

    // Admission reserves request lifetimes before preparation/submission. Metrics use that
    // authoritative count directly, including work not yet visible in an engine slot.
    const auto busy = parse(metrics.render(1, live, 2));
    failures += check(busy.at("llamacpp:requests_processing") == 1.0, "one processing");
    failures += check(busy.at("llamacpp:requests_deferred") == 1.0, "one deferred");
    const auto drained = parse(metrics.render(1, live, 0));
    failures += check(drained.at("llamacpp:requests_processing") == 0.0, "drained processing");
    failures += check(drained.at("llamacpp:requests_deferred") == 0.0, "drained deferred");

    // Cold request: whole prompt computed.
    metrics.record(outcome(1000, 0, 200, 0.5, 4.0, 300, 150));
    const auto cold = metrics.last_completed();
    failures += check(cold.prompt_tokens == 1000 && cold.cached_tokens == 0,
                      "last completed after cold request");
    // Warm request: 900 of 1200 prompt tokens served from the prefix cache -
    // only the 300 computed tokens may count toward the prompt counter.
    metrics.record(outcome(1200, 900, 100, 0.1, 2.0, 150, 75));
    const auto warm = metrics.last_completed();
    failures += check(warm.prompt_tokens == 1200 && warm.cached_tokens == 900,
                      "last completed after warm request");

    live.computed_prefill_tokens = 1300;
    live.prefill_seconds_total   = 0.6;
    live.committed_decode_tokens = 300;
    live.decode_seconds_total    = 6.0;
    const auto values = parse(metrics.render(1, live, 0));
    failures += check(values.at("llamacpp:prompt_tokens_total") == 1300.0, "live prefill tokens");
    failures += check(values.at("llamacpp:prompt_seconds_total") == 0.6, "live prefill seconds");
    failures += check(values.at("llamacpp:tokens_predicted_total") == 300.0, "live decode tokens");
    failures += check(values.at("llamacpp:tokens_predicted_seconds_total") == 6.0,
                      "live decode seconds");
    failures += check(values.at("ninfer:requests_total") == 2.0, "request count");
    failures += check(values.at("ninfer:prefix_cache_hit_tokens_total") == 900.0, "cache hits");
    failures += check(values.at("ninfer:draft_tokens_total") == 450.0, "draft tokens");
    failures += check(values.at("ninfer:draft_accepted_tokens_total") == 225.0, "accepted tokens");
    failures += check(values.at("ninfer:ngram_draft_tokens_total") == 150.0, "n-gram draft tokens");
    failures += check(values.at("ninfer:ngram_draft_accepted_tokens_total") == 75.0,
                      "n-gram accepted tokens");

    // A cache hit reported larger than the prompt must clamp, not underflow.
    metrics.record(outcome(10, 50, 1, 0.0, 0.1, 0, 0));
    const auto residue = metrics.last_completed();
    failures += check(residue.prompt_tokens == 10 && residue.cached_tokens == 10,
                      "last completed cache clamped to prompt");

    std::printf("%s serve metrics\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
