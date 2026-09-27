# RTX 4080 bring-up and 3-bit GSQ artifact

**Status: ACTIVE — this file is the resume point for the work.** Created 2026-09-26 from the
feasibility session on `rtx4090-port`. It is a temporary plan, not a permanent reference: delete it
when the work is finished or abandoned (AGENTS.md, "Change consistency").

**Progress (2026-09-27, session 6 + session 7): Stages 0–5 are recorded and Stage 5c.1 is complete
with both engine numbers measured.** Stage 4's gate passed on the 4080: the 100K + vision + MTP3
profile
validates memory before listening, the retrieval and vision probes are exact, MTP3 acceptance at
98K depth is 74.3%, and `/metrics` and `/slots` cross-check against the request timings. §11
records the host-memory investigation behind the Windows "shared GPU memory" figure. Stage 5's
measurements are recorded in §11 (PPL 4.596525 quick; 48.3 tok/s MTP3 decode on mixed text at
98K; 74.3% acceptance). The quality reference the gate named cannot run on a 16 GB card and no
larger card is available, so it is closed by taking the publisher's numbers for these exact
weights at face value, with the verbatim repack proof as the local check (§11, Stage 5). The
llama.cpp comparison is measured too. Session 5 closed most of the shallow-decode gap (fused
small-T Q3 SwiGLU GEMV, uniform 8-code MMA weight decode): the shallow round went from 59.5 to
40.1 ms (beellama ~39.9) and 100K prefill from 733.7 to 870.3 tok/s (beellama 1042.8). Session 6
landed **5c.1, the pipelined tall Q3 A16 GEMM**: the prefill route now walks 64-code half-groups
with one CTA per SM and the SwiGLU epilogue folds gate/up, dropping the chunked FP32 plane for
widths of at least 64 tokens. `pp32768` went 1014.98 -> 1406.34 tok/s (+38.5%) and `pp100000`
871.02 -> 1166.48 tok/s (+33.9%), which meets 5c.2's 1043 tok/s gate and beats beellama's 1042.8
at 100K; shallow MTP3 decode is unchanged (73.2 tok/s, 40.0 ms/round). The op bench gains
1.28–1.40x at chunk-aligned widths but regresses at some mid widths, so the route thresholds
still need a tail-aware pass (§11). **D10's measured axes are now met** (shallow decode parity,
100K prefill ahead), so the next step is a maintainer decision: 5c.2 as headroom, 5c.3 for a
decode lead, the Q3 small-T GEMV schedule port (§11 decode bandwidth audit), or Stage 6
publication. Session 7 then ported the whole 5c.3 n-gram surface (pool/
policy, verify-window split, CLI/serve flags, log schema 21, metrics, real lossless test) but
found a wrong correction logit in the wide verify route on the real artifact; **wide n-gram
rounds are disabled in the engine (the planner keeps `verify_window == draft_window`) with the exact
the ruled-out components recorded in the 5c.3 result block, so the enabled path stays lossless.
The full `ctest` set passes on the current build (127 tests, 115 passed, 12 expected skips, 0
failed).
The maintainer's D9 publication decision is now unblocked.

Environment for this plan: the `Dockerfile.dev` image in this repository. It is the sandbox the
maintainer hands to pi, with the host RTX 4080 passed through:

```bash
docker build -f Dockerfile.dev -t ninfer-4080-dev:pi .
docker run --rm -it --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/work" -v "$HOME/ninfer-models:/models" \
  ninfer-4080-dev:pi
```

Resume by reading this file top to bottom, then running §10.

## 1. Goal and completion conditions

Run Qwen3.8-27B in NInfer on one RTX 4080 (16 GB, `sm_89`, 76 SMs): **100,000-token context, single
lane, vision enabled, MTP speculative decoding**, with a 3-bit body converted **verbatim** from
`ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ`.

Complete when, on the 4080:

1. The new artifact loads and serves that profile end to end (vision probe, MTP3 decode, exact
   retrieval at 100K, engine memory validation passes before listening).
2. The full `ctest` suite matches the 4090 result set (the tip `81b68a20` gate was 109 passed,
   11 expected skips, 0 failed) with no `cudaErrorCooperativeLaunchTooLarge`, and the documented
   `--prefill-chunk` range (through 2688) still launches.
3. The weight-quality gate passes: `ninfer-perplexity` on the new artifact is within the agreed
   margin of the published `groupwise-int` artifact on the same corpus, `int8` KV
   (`ninfer-perplexity` does not expose the E8 modes; this gate is about weights).
4. Decode and prefill are measured at 100K with the method in `bench/README.md` and recorded.
5. The fork's docs carry the change: artifact reference, model card, ledger rows, README fork
   section. Publication as an RTX 4080 fork is the maintainer's decision (D9).

## 2. Decisions taken in the 2026-09-26 session

| # | Decision |
|---|---|
| D1 | Deliverable is a **new 3-bit `.ninfer` artifact**, not a GGUF/IQ3_S loader. |
| D2 | Source is `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` @ `b5ce0b76f60020a875dee4f6ec9d934cca4121e4`, repacked **verbatim** (no requantization), so the publisher's task numbers apply to the represented weights. The release carries no MTP, so the 12 `mtp/*` objects come from `Qwen/Qwen3.8-27B` @ `1d4bf0f2` (one 3.16 GiB shard). |
| D3 | The 3-bit group geometry follows the source: **group 128**, scales copied unchanged. See the Stage 2 decision point below. |
| D4 | Vision stays enabled. A text-only profile is cheaper but was declined. |
| D5 | Quality gate is perplexity only; the maintainer then validates in daily use. |
| D6 | 100,000-token context, single lane, `rk4v4-e8` KV. |
| D7 | The maintainer authorised the registry amendment (`tensor-formats.md` §10 currently excludes Q3 and alternate group sizes) and a ledger row. |
| D8 | TDD at the seams in §5, one vertical slice at a time. |
| D9 | If it works, this is published as an RTX 4080 fork. |
| D10 | Publication requires NInfer to beat beellama at comparable quality; Stage 6 is blocked until the shallow-decode and prefill gaps close. |

**Stage 2 decision point (D3):** the weights are identical either way; only the storage geometry
differs. `Q3G128_F16S` keeps the source groups (tightest, 3.125 bpw, but a second group geometry in
a registry that deliberately keeps one). `Q3G64_F16S` duplicates each source scale into its two
64-wide halves (weights bit-identical, +0.37 GiB, one group geometry). Default is D3; if the 100K
profile needs the extra slack, take the G64 form.

## 3. Facts established (evidence)

### 3.1 Why a 3-bit body is the only thing that fits

Sizes derive from the repo's own packing rules (`tools/artifact/layouts.py`); the arithmetic model reproduces
the published 16.96 GiB artifact to 0.01 GiB, so it is calibrated to the byte.

| Recipe | Weights (GiB) | + 100K `rk4v4-e8` (1.98) + 0.6–1.3 runtime |
|---|---:|---:|
| Published `groupwise-int` (W8 vocab, Q4/Q5 body, MTP, draft, vision) | 16.95 | 19.5–20.2 |
| All-Q4 body + Q4 vocab + MTP + draft, no vision (registered-format floor) | 13.89 | 16.5–17.2 |
| **Q3 body + Q4 vocab + MTP + draft, no vision** | **11.27** | **13.9–14.6** |
| **Q3 body + Q4 vocab + MTP + draft + vision** | **11.54** | **14.1–14.8** |

A 4080 has ≈15.7 GiB usable. The published artifact does not load at all; no all-Q4 recipe fits; the
Q3 profile fits with roughly 1 GiB of slack. KV cost from the fork's tables: `rk4v4-e8`
20.3 KiB/token (1.98 GiB at 100K), `int8` 38.5 KiB/token (3.61 GiB at 100K). The maintainer's
"kvarN 5/5" is int8-class or larger; nothing KV-related needs building.

### 3.2 The GSQ source checkpoint (`ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ`)

`config.json` → `quantization_config` (`compressed-tensors`, `format: pack-quantized`):

- `group_0`: 3-bit int, **group_size 128**, symmetric, targets `.*linear_attn.*`, `.*self_attn.*`,
  `.*mlp.*`, weights only (no activation quantization).
- `group_1`: 4-bit int, **group_size 64**, symmetric, targets `.*embed_tokens.*` and `lm_head$`.
- `ignore`: `.*visual.*`, `.*mtp.*`, `in_proj_a`, `in_proj_b` — stored as BF16.
- 2003 tensors, `total_size` 12,676,923,136 bytes (11.81 GiB), 3 shards.

Per-object storage (verified from safetensors headers, not inferred):

| Object | dtype | shape | Meaning |
|---|---|---|---|
| `<x>.weight_packed` | `I32` | `[N, K*bits/32]` | codes packed 32 bits/word, tightly along K |
| `<x>.weight_scale` | `BF16` | `[N, K/group]` | one scale per group, **bf16** |
| `<x>.weight_shape` | `I64` | `[2]` | original `[N,K]` |

Example: `layers.5.mlp.gate_proj.weight_packed` `I32 [17408,480]` (5120·3/32), `weight_scale`
`BF16 [17408,40]` (5120/128). `embed_tokens.weight_packed` `I32 [248320,640]`, `weight_scale`
`BF16 [248320,80]` (5120/64). Decode rule is `value = code × scale`.

Verify the code domain empirically at Stage 3: symmetric int-3 is expected to use codes in
`[-4,3]` with `scale = max_abs/3`, but the validator must accept exactly the domain the source
uses rather than an assumed one.

Consequences for the plan:

- The text body is **one new scheme over one new plane packing**; the vocabulary endpoints are the
  already-registered `Q4G64_F16S`, and the ignored tensors map onto existing BF16/FP32 objects.
- **Verbatim** means: copy codes and scales, undo nothing. Row fusion (`q/k/v/gate/up` → the
  artifact's fused row orders) is row selection/concatenation of (codes, scale) pairs, which
  preserves verbatimness exactly. The only numeric risk is the bf16→fp16 scale conversion
  (Stage 3 asserts exactness, and fails loudly per group if any scale is not representable).
- **MTP is not in the GSQ release.** Its `model.safetensors.index.json` lists 15 `mtp.*` tensors,
  but no shard contains them: the index has 2003 entries against 1988 real tensors, and the 15-entry
  difference is exactly the MTP set. The publisher's model card is right, and the stale index is a
  trap: `ShardReader.has("mtp.fc.weight")` answers true while the read then fails. MTP comes from the
  official checkpoint (3.3) through the existing `W8G32` recipe.
- **No draft head**: `text/draft_head` + `text/draft_head_token_ids` are derived by the converter
  from the committed frequency fixture `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`
  plus the tokenizer, not read from either checkpoint.
- Vision is BF16 in the source and uncalibrated (the publisher's release did not include vision
  calibration); it is quantized by the converter the same way the official recipe quantizes the
  official BF16 vision tower.

### 3.3 The official checkpoint (`Qwen/Qwen3.8-27B`) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

1199 index entries, 55,562,855,904 bytes (51.75 GiB) over 18 shards, 15 `mtp.*` tensors, no `draft*`
object. Only `model-00018-of-00018.safetensors` (3.16 GiB: the 15 `mtp.*` tensors plus
`lm_head.weight` in BF16) and `model.safetensors.index.json` (112 KB) are required — every other
artifact object comes from the GSQ release. `tools/convert/common/safetensors.py` opens shards
lazily, so a single-shard official source directory suffices provided the converter's preflight
validation accepts a declared subset (Stage 3). The full download is optional, useful only as a
BF16 reference.

### 3.4 The 4080 platform gap

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:71` derives device-wide
  cooperative-launch budgets from *the RTX 4090's 128 SMs* and guards them with a `static_assert`.
  On 76 SMs the budget is overstated and the driver rejects the launch with
  `cudaErrorCooperativeLaunchTooLarge` on the first prefill wide enough to reach it. Recorded as
  open in the port ledger (`Don-Chad 7afc8e17`, UDP `45a5ae57`).
- `DeviceContext::multiprocessor_count()` exists (`src/core/device.h:38`) and nothing in `src/ops`
  consumes it yet.
- `--prefill-chunk`/residency guidance (≤2688) and the retuned `sm_89` attention prefill schedule
  were measured on 128 SMs.
- No 4080 context-cost preset exists; the planner falls back to the generic default.
- Everything else is arch-correct for the 4080: the gate is `compute_capability() != 89`
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h:708`), FP4 paths are stubbed off, and the KV
  pool sizes itself from available VRAM (`src/runtime/engine/kv_capacity.cpp`).

### 3.5 Expectation setting

The 4080 is 76/128 SMs and 716.8/1008 GB/s. Decode ≈0.7× the published 4090 rows (148.6 tok/s code
at 81% acceptance, 106.5 bench, 50.5 no-spec) ⇒ roughly 100–110 tok/s code and ~35 tok/s no-spec.
Prefill is SM-bound and the 4090 already trails llama.cpp by 16–24% on full prompts, so on 76 SMs
**prefill will trail llama.cpp more, not less**. The win is decode; cold-prefill TTFT at 100K should
be expected to be worse than the current llama.cpp setup.

## 4. Implementation map

| Area | Change |
|---|---|
| `docs/maintainer/tensor-formats.md` | Register the new scheme; amend §10's Q3 exclusion; record the admission evidence (publisher task numbers, the exact source revision). |
| `tools/artifact/numeric.py` | New `QuantFormat` entry (3-bit, group 128 or 64). |
| `tools/artifact/layouts.py` | New base-plane geometry (3 bits/weight), encode/decode oracle, exact-size rules. |
| `tools/convert/qwen3_8_27b/` | GSQ source adapter (`pack-quantized` → codes/scales), `recipe_gsq3.py`, `convert_gsq3.py`, identity (`weights_id`, `recipe_id`), draft head/MTP/vision/resource reuse. `GatherRows` needs a packed variant that gathers `weight_packed` and `weight_scale` rows together; the 12 `mtp/*` objects read from the official source. |
| `src/…` (core/targets) | Persistent-format enum + validation + layout + binding + the GEMM/decode path for 3-bit codes. |
| `src/ops/gdn_gating_proj/bf16/…plan.cpp` | Device-wide budgets from `multiprocessor_count()`; bounds legal at the minimum supported SM count. |
| Tests | Python artifact/converter suites, a C++ linear op oracle suite, the artifact reader, E2E on the real file. |
| Docs | Artifact reference (`qwen3.8-27b-artifact.md` or a peer), model card, README fork section, ledger rows. |

## 5. Test seams (confirm before writing tests)

Per the `tdd` skill, tests go at these public seams and nowhere else:

1. **Python artifact codec** — `tools.artifact.numeric`/`layouts` public encode/decode functions,
   tested against an independently computed FP64 oracle (mirroring the existing `tests/artifact`
   style). Covers plane geometry, padding, code domains, and the verbatim scale copy.
2. **Converter CLI + artifact inspection** — `python -m tools.convert.qwen3_8_27b.convert_gsq3 …`
   and `python -m tools.artifact.inspect …`, tested against an independent decode of the source
   shards (object inventory, per-object byte sizes, and value-for-value equality with the source).
3. **Op contract** — the public `linear` path with the new format against a naive FP64 oracle
   (`tests/ops/linear/test_q3_a16.cpp` beside the existing `test_q4_a16.cpp`), same tolerance
   policy as the other quantized widths.
4. **Engine route** — the public binaries on the real artifact: `ninfer` CLI and `ninfer-serve`
   (100K + vision + MTP3 profile), and `ninfer-perplexity` for the weight gate.
5. **Platform gates** — `ctest` as-is plus the op benches (`ninfer_gdn_gating_proj_bench`,
   `bench/ops/`) for the SM-count behaviour, and a host unit test on the residency/route plan for
   the 76-SM bounds.

## 6. Stages

Each stage ends with its gate; do not start the next stage before the gate passes. Commit only when
the maintainer asks.

### Stage 0 — environment and 4080 baseline (no product change)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 -m pytest tests/artifact tests/convert tests/test_bench_matrix.py tests/test_serve_corpus.py
nvcc -O3 -std=c++17 -arch=sm_89 tools/hbm_bandwidth_probe.cu -o /tmp/hbm_bandwidth_probe
/tmp/hbm_bandwidth_probe   # device bandwidth sanity; see the probe's header comment
./build/bench/ninfer_gdn_gating_proj_bench --help
```

Also record: `nvidia-smi`, `nvcc --version`, driver version, free VRAM at idle, and the whole
`ctest` result set. Downloads (pinned revisions, explicit paths — no globs, no "latest"):

```bash
hf download ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ --revision b5ce0b76f60020a875dee4f6ec9d934cca4121e4 --local-dir /models/gsq3
# The GSQ release has no MTP (its index lists 15 phantom `mtp.*` entries), so fetch only the one
# official shard that carries them. Add `--include "config.json"` if the converter preflight wants it.
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --include "model-00018-of-00018.safetensors" --include "model.safetensors.index.json" \
  --local-dir /models/qwen3.8-27b-bf16
```

Gate: the suite's pass/skip/fail set is recorded; the failing tests, if any, are explained (the
expected platform failure is the gating-projection cooperative launch, whose reproducer is the
bench above; the compile-time catalog guard validates against the 4090's budget, so it cannot catch
the 76-SM case, which is part of Stage 1).

(If ccache misbehaves with nvcc, drop the three launcher flags; nothing else changes.)

### Stage 1 — 4080 platform enablement

Parameterize the gating-projection resident-CTA budgets by `multiprocessor_count()`, re-derive the
route bounds for 76 SMs (with 76: split8 admits 2 CTAs/SM → 152 device-wide, split4/2 → 76, so the
cooperative split-K ceilings drop from 2688 columns to 1536), introduce the minimum supported SM
count, and make the compile-time catalog guard assert against *that* count rather than the 4090's.

Gate: host unit test of the plan bounds for 76/128 SMs; `ctest` on the 4080 unchanged; the
gating-projection bench runs cooperative splits without a launch rejection up to its new bounds;
`ninfer-serve` prefills a synthetic length past the old bound without error.

### Stage 2 — the 3-bit scheme (TDD)

Vertical slices, each test first: numeric registration → plane geometry → Python encode/decode
oracle → artifact reader/binder → C++ op test → 4080 kernel bench.

Gate: Python oracle and C++ op tests pass against the FP64 oracle; the artifact reader accepts a
synthetic Q3 object and rejects malformed planes; the 4080 `linear` bench records the Q3 rates.

### Stage 3 — converter and artifact

Source adapter plus recipe, reusing the existing writer, verification, draft-head, MTP, vision, and
resource machinery. Identity: a new `weights_id`/`recipe_id` and filename
(`qwen3_8_27b_gsq3.ninfer`), resolved by the artifact registry.

Gate: converter inventory tests; `tools.artifact.inspect --objects` matches the object plan; an
independent decode of the artifact equals an independent decode of the source shards
value-for-value; every scale is fp16-exact; the conversion report is written.

### Stage 4 — engine bring-up on the 4080

```bash
./build/apps/ninfer-serve /models/qwen3_8_27b_gsq3.ninfer \
  --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 \
  --max-concurrency 1 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking \
  --host-kv-mib 4096
```

The 4 GiB host KV keeps the pinned host pool at ~5.5 GiB on the 16 GiB host without losing
long-anchor reuse; host state slots stay at the default 8 (§11's Stage 4 entry has the
measurements). A separate pass at `--prefill-chunk 2688` covers the deferred Stage 1 item.

Gate: startup memory validation passes before listening; MTP3 acceptance recorded; a 100K
retrieval probe (single needle, five needles, exact code detail) is exact; the vision probe answers
its oracle facts; `/metrics` and `/slots` report truthfully.

### Stage 5 — quality and performance

```bash
./build/apps/ninfer-perplexity /models/qwen3_8_27b_gsq3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8
```

Also run the same corpus on the published `groupwise-int` artifact and, if the numbers are
comparable, on the maintainer's llama.cpp IQ3_S configuration with the same text. Then decode and
prefill at 100K with the documented method (`bench/README.md`), plus the depth sweep the README
publishes, always with the exact command, artifact hash, and date.

Gate: the weight-quality margin agreed with the maintainer is met; decode improves on the current
llama.cpp setup; prefill is recorded honestly even if it regresses.

### Stage 5b — performance parity with beellama (D10)

Measured gaps (2026-09-26): shallow MTP3 decode 44.0 vs 71.0 tok/s, and ~59.5 vs ~39.9 ms per
MTP3 round at 8K context; 98K prefill 734.7–737.3 vs 1042.8 tok/s. At 98K depth NInfer already
has the faster round (~67 vs ~72 ms) but accepts fewer drafts (74.3% vs 78.5–81.5%).

Method: attribute each gap with `ninfer_bench --profile-measured` under Nsight Systems before
changing code. Decode: `tg512 --spec mtp --draft-tokens 3 --lm-head-draft` at 8K `rk4v4-e8`;
prefill: `pp32768` with the same KV. Target: a shallow round below ~37 ms (at the measured
2.62 tok/round, 44 tok/s then beats 71) and 100K prefill at or above 1043 tok/s.

Candidates to confirm or reject by profile: the single 32x64 Q3 MMA schedule (weight rows
re-read per 64-token column tile), the small-T Q3 GEMV path, the chunked FP32 swiglu plane in
`linear_swiglu`, and per-round fixed overhead (draft loop, sampling, graph replays).

Gate: whole-inference shallow decode and 100K prefill beat the beellama figures above on the
same card and workload; acceptance quality is reported alongside.

**Stage 5b progress (2026-09-26, session 5): the shallow decode gap was one route choice.** Every
one of the 64 Q3 `linear_swiglu` MLP parents ran the prefill MMA schedule (32x64 tile) at the
T=4 verify: 500 us per call against ~275 us for the qualified GEMV on the same shape, 32 ms of
a ~63 ms round. A fused small-T GEMV (`q3_rowsplit_gemv_swiglu_kernel`, warp-per-row, both
halves in one pass, no FP32 plane, zero-capacity workspace) now routes T<=8. Evidence:
`ninfer_linear_swiglu_q3_a16_test` passes; bench `tg512` 36.7 -> 54.1 tok/s at the same
acceptance (233 rounds, 39.9%); the CLI thinking-off scenario decodes at 72.9 tok/s over 14
rounds (40.1 ms/round against beellama's ~39.9, 71.0 tok/s). Profiles:
`profiles/nsys/gsq3-shallow-tg32-{nodes,after}.nsys-rep`.

**Prefill attribution (2026-09-26, session 5).** `pp32768 --kv-dtype rk4v4-e8` = 850.9 tok/s,
38.3 s of kernel time: Q3 MMA 34.1 s (89%), INT8 prompt attention 3.0 s (7.8%), GDN chunked ops
~0.7 s, norms and casts ~0.4 s. Two experiments followed:

- A 32x128 MMA tile with dynamic shared memory (1 CTA/SM) regressed every T above 64 (T=256:
  2042 against 1921 us), so the 32x64 tile stays.
- The weight-decode loop was the real cost: the old per-pair extraction issued two divergent
  byte loads per code. A uniform 8-code window decode (one 3-byte load, eight constant shifts,
  one 16-byte store per lane, two rows per warp) is 1.23x faster at every T (T=256: 1921 ->
  1542 us, 47.5 -> 58.4 TFLOPS). The now-unused pair decoder is removed.

End to end: `pp32768` 850.9 -> 1016.4 tok/s; `pp100000 --prefill-chunk 2688` 733.7 -> 870.3
tok/s against beellama's 1042.8, so the prefill gap narrowed from 42% to 17%. Profile:
`profiles/nsys/gsq3-prefill-pp32768.nsys-rep`.

### Stage 5c — ported prefill and speculation work (fresh-context plan)

**Why this exists.** D10 blocks publication until NInfer is faster than beellama at comparable
quality. Stage 5b closed the shallow decode gap (40.1 vs 39.9 ms per MTP3 round; 72.9 vs 71.0
tok/s on the CLI scenario) but prefill still trails: `pp100000` 870.3 vs 1042.8 tok/s. The
JGamboa RTX 4090 fork measured four features that target exactly this gap, all Apache-2.0 and
from the same upstream lineage. Nothing cherry-picks: their branch sits on a newer upstream
layout (`src/models/qwen3_5/...`) than this tree (`src/targets/...`), so each item is an
adaptation port. Read the fork files, keep this tree's ownership boundaries, and record each
landed port in [port-ledger.md](port-ledger.md).

**Fork access (read-only).**

```bash
git clone --depth 60 --branch feat/bonsai-ternary \
  https://github.com/JGamboa/ninfer-4090-windows /tmp/jgamboa-eval
```

| Commit | Subject | Fork files to read first |
|---|---|---|
| `4ba151c` | pipeline the Q4/Q5 prefill GEMMs with one CTA per SM | `src/ops/common/rowsplit_tall_mma.cuh` (482 lines) and the Q4/Q5 plan diffs |
| `956b169` | run the Q4/Q5 prefill GEMMs on int8 tensor cores under AllowA8 | `src/ops/common/rowsplit_a8_quantize.{h,cu}`, `src/ops/common/rowsplit_tall_a8_mma.cuh` (357 lines), `tests/ops/*_a8*` |
| `7b6ed55` | move V dequant to the workers in the int8 prompt kernel | `src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh` (their 788 -> 814 lines) |
| `375542a` | chain host n-gram drafts after the MTP proposal | `src/models/qwen3_5/program/speculative/ngram_policy.h`, `ngram_pool.h` |
| `535f9c1` | split the MTP verify window from the proposal depth | `include/ninfer/ops/mtp_round.h`, `src/ops/{kernel,launcher,wrapper}/mtp_round.*` |
| `be3b680` | expose n-gram drafts in the CLI, server, logs and metrics | `src/serve/{serve_options,request_log,serve_metrics}.*` |

Do not port the Bonsai/ternary (`t5_*`) work, the Windows build, or their artifacts. The 4080
artifact stays `out/qwen3_8_27b_gsq3.ninfer`; none of these changes touch artifact bytes.

**Baseline the fresh session must reproduce first** (GPU free; beellama may be stopped):

```bash
# decode: shallow MTP3 (expect ~54.2 tok/s, 39.9% acceptance, 233 rounds)
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer -n 512 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-ctx 8192 --kv-dtype rk4v4-e8 \
  --warmup 1 -r 1
# prefill: 32K (expect ~1016 tok/s) and 100K (expect ~870 tok/s)
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer -p 32768 \
  --max-ctx 40960 --kv-dtype rk4v4-e8 --warmup 1 -r 1
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer \
  --corpus profiles/bench/bench_corpus_131072.ids -p 100000 --max-ctx 102400 \
  --prefill-chunk 2688 --kv-dtype rk4v4-e8 --warmup 1 -r 1
# op baseline for the Q3 MMA and its tests
./build/bench/ninfer_linear_bench --qtype Q3 --n 34816 --k 5120 --sweep 9:513:8 --policy a16
./build/tests/ninfer_linear_q3_a16_test && ./build/tests/ninfer_linear_swiglu_q3_a16_test
```

Profiler note: `ncu` cannot read counters in this WSL2 container (`ERR_NVGPUCTRPERM`); use
Nsight Systems. The bundled importer needs `libdw1` (already installed) and lives at
`/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter`:
if `nsys profile` fails to write a `.nsys-rep`, run
`QdstrmImporter -i x.qdstrm -o x.nsys-rep -f`. Existing captures: `profiles/nsys/gsq3-*`.

**5c.1 Q3 tall A16 GEMM (DONE 2026-09-27, session 6; evidence below).**

- Fork: `4ba151c`. Ours: `src/ops/linear/q3/q3_rowsplit_gemm_mma.{cuh,cu}` (the current 32x64
  tile: 2 CTAs/SM, 4 warps, all warps decode then MMA, two barriers per K step).
- Design to port: one CTA per SM; 8 warps own 128 weight rows and `Tokens` (64 or 128) tokens;
  walk K in 64-value steps; decoded weights and activations double-buffered; each thread decodes
  one half-group from registers loaded one step earlier; warps 0-3 multiply before decoding the
  next step, warps 4-7 decode before multiplying; one barrier per step; token tile fastest in
  the grid so a row block's CTAs share code bytes in L2.
- Q3 adaptation: our weight group is 128, so keep the 64-value step and use the 128-group's
  fp16 scale for both halves; decode with the uniform 8-code window already in
  `decode_weight` into bf16. Preserve the accumulation contract: one `m16n8k16` MMA per 16-wide
  K slice in ascending K, fp32 accumulators - that is what makes the port bit-identical.
- Acceptance: a byte-compare test proving the new route equals the current c64 route at
  T = 9..513; `ninfer_linear_q3_a16_test` and `ninfer_linear_swiglu_q3_a16_test` pass; the
  `ninfer_linear_bench` T sweep improves; then `pp32768`/`pp100000` record the engine gain.
  Expect roughly +20-30% if the fork's 26-29% bf16 result transfers.
- If the bit-identical invariant cannot be held, stop: the registered A16 contract is the
  authority, not the new kernel.

**5c.1 result (2026-09-27, session 6).** Landed as `src/ops/linear/q3/q3_rowsplit_tall_mma.{cuh,cu}`
with the dispatch in `q3_dispatch.cpp`: widths below 64 columns keep the staged 32x64 route,
64..127 the 64-token tall tile, and 128+ the 128-token tile; `q3_linear_swiglu.cu` routes 64+
columns to the folded SwiGLU problem and reports zero workspace for that range. Evidence:

- `tests/ops/linear/test_q3_a16_tall.cpp` (new) byte-compares both tall tiles against the staged
  route at T = 9..513 step 8 on `[1024,5120]`, at 64/129/512 on `[14336,5120]`, and at
  9/65/129/512 on the padded `[4096,4304 -> 4352]` shape; all byte-identical. The three Q3 op
  suites pass (`linear`, `linear_swiglu`, `linear_add`). The full `ctest` set has not been re-run
  after the change.
- Op bench (`ninfer_linear_bench --sweep 9:513:8 --warmup 2 --repeat 10`, median us, staged ->
  tall): `34816x5120` T=257 1961 -> 1578, T=513 3597 -> 2571; `16384x5120` T=385 1308 -> 1009;
  `14336x5120` T=385 1141 -> 867; `5120x17408` T=513 1830 -> 1441; `5120x6144` T=513 653 -> 510.
  CSVs are `profiles/bench/5c-tall/{staged,tall}_n*_k*.csv`. The staged/tall split is 1.28-1.40x
  at 128-aligned widths of 385+, but the 40-block shapes lose at T=129 (0.72x) because the last
  128-token tile has one live token; a tail-aware split is the obvious follow-up (the engine's
  steady-state chunk sizes are multiples of 128).
- End to end (session-6 log `profiles/bench/5c-tall/session6-tall-engine.log`): `pp32768`
  1014.98 -> **1406.34 tok/s** (+38.5%); decode `tg512` 53.73 -> 54.05 tok/s (unchanged, as
  expected). `pp100000 --prefill-chunk 2688` 871.02 -> **1166.48 tok/s** (+33.9%; session-7 log
  `profiles/bench/5c-tall/session7-pp100000.log`), so **5c.2's 1043 tok/s gate is already met
  and 100K prefill now beats beellama's 1042.8**. The shallow CLI scenario is unchanged at
  73.2 tok/s / 40.0 ms per MTP3 round (session 5: 72.9 / 40.1). Full `ctest` on the tall build:
  125 tests, 113 passed, 12 expected skips, 0 failed.

**5c.2 Q3 A8 int8 prefill (gate already met by 5c.1: 1166.48 >= 1043; now optional headroom).**

- Fork: `956b169`. Read `rowsplit_a8_quantize.{h,cu}` and `rowsplit_tall_a8_mma.cuh` in full,
  plus the `q4_linear_swiglu_plan.cpp` / `q5_linear_add_plan.cpp` and paired-projection diffs.
- Our state: `AllowA8` is admitted for Q4/Q5/Q6/W8 but routes to the same bf16 kernels
  (`src/ops/linear/q4/q4_dispatch.cpp` and peers); Q3 throws for `AllowA8`
  (`src/ops/linear/q3/q3_dispatch.cpp`). `mma_s8` (`m16n8k32.s8`) already exists in
  `src/ops/common/mma.cuh` and is used by attention. Every GSQ3 prefill family runs through
  `linear` and the fused `linear_swiglu`, so two routes are needed, not the fork's four.
- Contract to document and test (mirror fork `op-development.md` 6.4):
  activation per token and 64-K group: `amax = max|x|`, `s = amax/127`,
    `q = rint(x * 127/amax)` clamped to `[-127,127]`, FP32 RNE, zero group -> zero scale/codes;
  weight code `c` in `[-4,3]` is an exact int8 operand and each 64-value group's int32 dot
  `d_g` is exact; output = `sum_g (w_scale_128(g/2) * s_g) * d_g`, one FP32 fma per 64-group in
  ascending K, before the Op's single bf16 output rounding. One 128-wide weight group spans two
  activation groups.
- Kernel: adapt the tall A8 kernel. Decode 3-bit windows to int8 (32 codes = 12 bytes per
  half-group), seed each group's s8 accumulator with `0x4B400000` and recover `d` with
  `__int_as_float(g) - 12582912.0f` (guard `|d| <= 64*4*127`, well inside the exact range),
  then `fmaf(w_scale * s_g, d, acc)`. Keep 5c.1's ping-pong warp roles and double buffering.
- Wiring: route selection in `q3_dispatch` (A16 below 129 columns, A8 at 129+, matching the
  fork's threshold); a workspace for the quantized activation (`I8 [K,T]` + `FP32 [T,K/64]`,
    ~5.4 KB per token; the 100K profile has ~930 MiB slack); thread the policy through
  `linear_workspace_capacity_bytes` (it calls `select_q3_launch` for min and max T and would
  throw); the GSQ3 execution leaves pass `AllowA8` at prefill widths and `A16Only` at decode.
  Decode (T<=8 GEMV) and the MTP verify (T=4) stay A16.
- Tests: new `ninfer_linear_q3_a8_test` and `ninfer_linear_swiglu_q3_a8_test` against an FP64
  oracle that applies the documented quantization; existing A16 suites unchanged; add the A8
  provider case to `ninfer_attn_input_proj_test`/`ninfer_gdn_input_proj_test` when the leaves
  expose the policy.
- Gate: `pp100000 --prefill-chunk 2688` at or above 1043 tok/s (met by 5c.1), `pp32768` recorded,
  decode unchanged, full `ctest` clean, `ninfer-perplexity --quick` unchanged or better.

**5c.3 N-gram drafts chained after MTP (decode lead on real workloads).**

- Fork: `375542a` (policy, pool, tests), `535f9c1` (verify window split), `be3b680` (flags,
  logs, metrics). Our ops files share the fork's paths and pre-split contents
  (`include/ninfer/ops/mtp_round.h` differs only trivially), so `535f9c1` is the closest thing
  to a cherry-pick in this plan. Our engine layer differs: the round lives in
  `src/targets/qwen3_6/impl/runtime/mtp_impl.h` (211 lines) and proposals come from
  `card.mtp_propose_batch`.
- Sequence: (a) port `ngram_policy.h`/`ngram_pool.h` and the unit test; (b) apply the verify
  window split - round buffers/envelopes/workspace sized for V <= 15 verify drafts while the
  MTP head keeps k = 1..5 (ours documents `1<=K<=5`); (c) chain the pool proposal after
  `card.mtp_propose_batch` per row and use the wide verify window only when a row's extension
  reaches `k + kNgramWideRoundMargin` (3), so prose keeps the MTP-only cost; (d) `--ngram
  chain`, metrics counters, request-log fields, and a lossless real test.
- Acceptance: greedy output md5-identical with and without n-gram; `--ngram chain` on
  `examples/cli/messages/scenario_code_python.json` (or a committed edit-style transcript)
  shows a real acceptance-length gain; prose decode unchanged within noise; ctest clean.

**5c.3 result (2026-09-27, session 7): the port is in place but its wide verify window is
disabled pending a fix.** Landed: `ngram_pool.h`/`ngram_policy.h` + `test_ngram_policy`, the
`535f9c1` verify-window split (op contract, kernel, launcher, wrapper, oracle test, round state,
envelope/profile/workspace/record plumbing), the engine chaining in `decode_mtp_batch`, the CLI
and serve flags, request-log schema 21 with the n-gram counters, and the real lossless test
`ninfer_qwen3_6_27b_ngram_real_test`. The narrow path is correct (real test passes with 0 added
divergences and 0 wide rounds).

The blocker: with wide rounds enabled, the real artifact's target verify produced a wrong
correction logit in one column. Isolated repro and evidence: prompt 3 (the table), round E=467
width 7 extent 6, ids `[5653 1870 1137 5480 2923 16 23]` at positions `[467..473]`; the verify
argmax at column 2 was 4075 (raw logit 19.75) while the model's own scoring route gives 5480
(`-0.00352` vs `-6.00352` nats), so the correct draft 5480 was rejected. Columns 0,1,3..6 were
correct. What was ruled out: the attention op at width 7/window 474 (new masked and unmasked
oracle cases pass), the GDN replay-record op (width sweep 2..16 passes), the Q3 GEMV and fused
SwiGLU GEMV at T=5/6/7 (new cases pass), the W8 lm_head at T=7 (test covers 1..128), the fold/
record width selection (verified by construction), CUDA graphs (fails without them), OOB memory
and shared-memory races (compute-sanitizer memcheck and racecheck: 0 errors). The remaining
suspect area is the assembled multi-column verify state (a single-column hidden corruption).
The engine now plans `verify_window == draft_window`, so the wide graph family and record planes are
test state that a round verifies at most the MTP width. Do not re-enable wide rounds until the
column-level corruption is explained.

**5c.4 Prompt attention worker V-dequant (last, modest).**

- Fork `7b6ed55` against a 788-line file; ours is 642 lines at the same path
  (`src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh`), so port the ideas, not the
  patch: workers dequantize V(t+1) to FP16 after PV(t) behind a worker-only barrier; replace
  the two per-tile block barriers with one-sided named barriers (`PFree`/`PReady`); bytewise
  INT4 -> INT8 unpack. Their result is -15..-25% on the kernel; in our `pp32768` profile prompt
  attention is 7.8% of prefill (more at 100K), so expect low single digits overall plus some
  depth-decode help.
- Acceptance: `ninfer_softmax_attention_test` (full and `--rk4v4-e8-only`) passes; the append
  bench improves at 8K-128K; engine prefill recorded.

**Order and dependencies.** 5c.1 -> 5c.2 (A8 reuses 5c.1's skeleton). 5c.3 is engine-side and
independent; it can run in parallel with either kernel port. 5c.4 last. Each port lands with
its tests, a bench number, and a `port-ledger.md` row; do not update the model card, README, or
publication docs until D10 is met. Rough effort: 5c.1 one session, 5c.2 two to three, 5c.3 one
to two, 5c.4 one.

**5c.5 DFlash2 instead of MTP (conditional candidate; not committed).**

The engine half already exists in this tree: `DFlashConfig::supported = true` for the 27B
(`src/targets/qwen3_6_27b/impl/config.h:75`), `bind_dflash2` activates whenever an artifact carries
`dflash2/feature_projection` (`src/targets/qwen3_6_27b/impl/load/bindings.cpp:450`),
`ninfer_qwen3_8_27b_dflash2_real_test` is built, and `--spec dflash2 --draft-tokens K` is the whole
public surface. The GSQ3 converter is what lacks the suffix (`convert_gsq3.py`; the 5090 identities
already take `--dflash2-model`), so the artifact work is a recipe/verifier extension plus the
`z-lab/Qwen3.8-27B-DFlash2` BF16 source (~3.9 GB, pinned `50307d4c`).

Fit at the 100K profile (registered formats, measured slack):

- companion 66 objects = **2.074 GiB** (1.838 GB `W8G32_F16S` matrices + 0.389 GB BF16, of which
  the two selector codebooks are 0.254 GB), plus 40 MiB persistent ring state per device state
  slot;
- DFlash2 and MTP are mutually exclusive, so the comparison baseline drops MTP (0.420 GiB): net
  **+1.69 GiB** over the profile Stage 4 validated, which ended with 0.912 GiB
  `available_after_startup_bytes`. **It does not fit as converted.** The options are a maintainer
  decision: text-only artifact (-0.28 GiB) plus a context cap near 75K, or requantizing the
  companion (Q4G64 on the W8 matrices -0.86 GiB, int8 selector -0.12 GiB) with new
  bindings/routes and a draft-quality gate. K does not change the weight or ring-state cost (only
  proposal/verify transients grow with it).

Speed, against MTP3 on the same card:

- 4090 (port-ledger, `dc370fb6295a`): K=3 149.8/104.0 tok/s code/prose vs MTP3 140/107 (+7%/-3%);
  K=5 190.3/94.2 (+36%/-12%). 4090D 48 GB K=3: 133.3/111.0 vs 122.0/97.5 (+9%/+14%).
- 5090 (G3/GD): code 4.77 vs 3.29 tokens/round, structured 6.65 vs 3.68, story 2.09 vs 2.14;
  published decode 191.4/84.0/267.3 vs 200.3/130.4/224.4 tok/s; the nvfp4 profile gains more.
- 4080 arithmetic today: the W=8 verify sits at the Q3 GEMV's worst point (T=8 = 1.98x T1 against
  T=4 = 1.37x), so the round grows 40.6 -> ~54 ms while code-style tokens/round grow ~1.45x:
  roughly +10% code, +35% structured, -25% prose. **After the Q3 small-T schedule port in §11**
  the verify is T-flat and the tokens/round term dominates: +35..70% code/structured, prose parity.
  N-gram chaining (5c.3, already ported, no extra weights) buys the same tokens/round upside, so
  measure that first.

Shared gating item: DFlash2 always verifies wide (`W=K+1`, up to 16), and the 4090 reported the
same symptom class as our 5c.3 blocker - "K=3 differs from K=5 at temperature 0 on `rk4v4-e8`:
quality gate before any default change". Nothing about DFlash2 is trustworthy until the 5c.3
column corruption is explained, and the memory question above needs a decision before the converter
work starts. Until then this stays a candidate, not a stage.

**Open questions for the maintainer.**

- **A8 permission (session-6 finding; recommendation).** In this tree the artifact carries no
  activation policy at all (`artifact-container.md` §4/§10 exclude execution policy; there is no
  `activation_policy` in `src/artifact` or `tools/artifact`). Policies are chosen in
  `src/targets/qwen3_6_27b/impl/variant.cpp:54 text_policy()`, and the op-domain docs list which
  policies each format admits. So there is no artifact byte to change and no second identity to
  create: 5c.2 is a public-doc/registry admission plus a `text_policy` case and the A8 route.
  Recommendation: proceed on the single `gsq3` artifact as the question proposes.
- **n-gram pool defaults (session-6 finding; recommendation).** The fork's defaults are
  `match_tokens = 8`, `pool_bytes = 16 MiB` (4,194,304 entries), `max_drafts = 15`,
  `min_drafts = 1`, with `kNgramWideRoundMargin = 3` (`include/ninfer/types.h` `NgramOptions`
  and `ngram_policy.h`). They are cheap in host memory and the margin keeps prose at the
  MTP-only width. Recommendation: adopt them, keep the same CLI knobs for later tuning.

### Stage 6 — documentation and publication

Held until Stage 5b passes (D10). Then: artifact reference + model card + README fork section +
ledger rows; state the source revision, the verbatim claim and what it does and does not cover,
the memory profiles that were validated, and the known prefill limitation on 76 SMs.

## 7. Where each gate runs

| Gate | Host CPU | 4080 GPU |
|---|---|---|
| Python artifact/converter suites, `py_compile` | yes | yes |
| Host plan/geometry unit tests | yes | yes |
| CUDA compile (`sm_89`), `ptxas -v` budgets | yes (nvcc in this image, no GPU needed) | yes |
| `ctest` kernel/op/engine tests | no | yes |
| Real-artifact E2E, perplexity, benches | no | yes |

There is no GPU-free substitute for the kernel, E2E, and perplexity gates.

## 8. Risks and open items

- **Q3 decode throughput is unknown.** A new unpack path may cost more per weight than the existing
  Q4/Q5 paths; the op bench decides, and a decode-specialized path is the fallback.
- **≈1 GiB of slack** at 100K with vision on 16 GB. Any growth (extra lane, larger vision
  scratchpad, DFlash2) does not fit. Re-check against `kv_capacity` boot lines on the real card.
- **bf16 → fp16 scales** must be exact; the converter fails per group rather than rounding.
- **MTP provenance**: the official checkpoint is required for the 12 `mtp/*` objects. Only
  `model-00018-of-00018.safetensors` is needed, and the converter's preflight must accept that
  single-shard source. If it does not, fetch the full checkpoint rather than teaching the converter
  a partial-index hack.
- **Vision is uncalibrated in GSQ**; the converter quantizes BF16 words, and the vision oracle
  probes are the only local evidence.
- **Prefill on 76 SMs** now beats beellama at 100K (1166.48 vs 1042.8 tok/s; 32K 1406.34). The
  only remaining prefill regression is at some mid widths (T=129 on 40-block weights) where the
  tall route's last 128-token tile is nearly empty; the tail-aware split is the fix if it matters.
- **Host driver** must support CUDA 13.1; verify `nvidia-smi` before Stage 0.
- **Publication** needs the license/attribution notes of both upstream sources.
- Perplexity margin (closed 2026-09-26, maintainer): the reference artifact cannot run on a
  16 GB card and no larger card is available; the publisher's numbers for the represented
  weights are taken at face value and no comparison is attempted. Still open: whether a
  text-only sibling artifact should be produced.

## 9. Deliberately out of scope

GGUF/IQ3_S/K-quant/I-quant loading; RCO per-tensor allocation search; runtime weight repacking;
DFlash2 on the 4080; multi-lane and preemption; the 5090 tree; Windows.

## 10. Resume checklist

1. Build the image and start it with `--gpus all` (§ header). Confirm `nvidia-smi` sees the 4080 and
   that `nvcc --version` reports 13.1.
2. Read §2 (decisions) and §5 (seams); confirm or amend the Stage 2 geometry decision.
3. Run Stage 0's commands and record the baseline in this file (append a "Stage 0 result" line with
   the date, the ctest counts, and the bench numbers).
4. Work the stages in order; update this file at each gate, including failures and the reason, so
   the next session does not repeat the investigation.
5. Delete this file when the work is done or abandoned, and move the durable outcome into the
   artifact reference, the model card, and the port ledger.
6. If the task is the remaining performance work, read **Stage 5c** first. 5c.1 is complete and
   both engine numbers are measured (32K 1406.34, 100K 1166.48 tok/s); D10's measured axes are
   met. The next item is the maintainer's call: 5c.2 as headroom, 5c.3 for a decode lead, the Q3
   small-T GEMV schedule port (§11 decode bandwidth audit), or Stage 6 publication. 5c.5 (DFlash2)
   is a conditional candidate only: it does not fit the 100K profile as converted and is gated on
   the 5c.3 wide-verify fix (§6).

## 11. Stage log

### Stage 0 — environment and baseline (2026-09-26, session 2)

Host: `nvidia-smi` → RTX 4080, 16,376 MiB, 76 SMs, idle 882 MiB, driver 615.71.08 / KMD
616.92 / CUDA UMD 13.4. `nvcc` 13.1.115 (V13.1.115). Image from `Dockerfile.dev`, models bind-mounted at
`/models` (both pinned sources present).

- Build: 711 targets, Release + Ninja + ccache, no errors.
- `ctest`: 120 tests, **107 passed, 12 skipped, 1 failed**. Failure:
  `ninfer_gdn_gating_proj_test` → `cudaErrorCooperativeLaunchTooLarge` at
  `bf16_gdn_gating_proj_kernels.cu:310`. Skips: 27–33 (real-artifact tests; no `.ninfer` file in
  this container), 78/79 (softmax attention NVFP4/K8V4), 85/86 (KV append NVFP4/K8V4), 97
  (linear NVFP4 A4). The plan's 4090 reference (109/11/0) differs only by one artifact-dependent
  test that had a file on the 4090 host; no 4080 test skipped for an SM-count reason.
- Python: `pytest tests/artifact tests/convert tests/test_bench_matrix.py tests/test_serve_corpus.py`
  → 76 passed, 3 skipped.
- HBM probe (`-arch=sm_89`): best bus rates write 420.2, read 395.9, copy 385.1 GB/s. The probe's
  default 1792 GB/s peak constant is the 5090 calibration; the 4080's advertised peak is 716.8 GB/s,
  so measured sustained traffic is ~54–59% of peak in this sandbox.
  **Correction (2026-09-27, re-measured): that run caught the card before its memory clock was up.**
  The card idles at 405 MHz memory and the probe's eight-launch warmup is too short on this host.
  Re-running the same binary on the idle card reports write 641.8, read **664.0**, copy 577.3 GB/s bus
  (92.6% of the 716.8 spec), and the engine's own kernels match (Q4 draft-head GEMV 664 GB/s, Q4 main
  head 600 GB/s; §11 decode bandwidth audit). Treat 395.9 as a bad measurement, not a device
  capability: the "~54–59% of peak" reading is withdrawn.

**Stage 0 gate finding (root cause of the one failure).** The runtime cooperative capacity in
`bf16_gdn_gating_proj_kernels.cu` (`cooperative_resident_ctas_per_sm`) overstates two
specializations of the sm_89 build, so the upstream token-tile partitioner does not split enough
and the driver rejects the grid. Measured with `cuobjdump -res-usage` on the built cubins and
reproduced with the bench: 27B split-4/2 use 512 threads × 74 registers → **1 CTA/SM** (the table
says 2), so T=2048/2688 (grid 96/126 > 76) are rejected; 35B split-8/4/2 use 256 threads × 74
registers → **3 CTA/SM** (the table says 4), so T≥1024 (grid 256 > 228) is rejected. The
compile-time table in `bf16_gdn_gating_proj_plan.cpp` already carries the correct numbers, but
nothing consumes it at runtime: the two tables are duplicated and had drifted. Stage 1 unifies
them (plan §4, §6).

### Stage 1 — 4080 platform enablement (2026-09-26, session 2)

Fix: the per-SM residency facts now live once in
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_residency.h`, with the `cuobjdump` evidence for
each cooperative specialization. The launcher reads them through the same constexpr table the
route catalog uses, and the corrections are 27B split-4/2 → 1 CTA/SM and 35B split-8/4/2 → 3
CTA/SM. `bf16_gdn_gating_proj_plan.cpp` derives its 27B route endpoints from the shared ceilings,
asserts whole-grid residency on the 128-SM reference, and asserts that every cooperative route
seats one token tile on the 76-SM minimum device. The 35B perf-chosen bounds are unchanged and
validated against the same facts.

Deliberate reading of the plan's “ceilings drop from 2688 to 1536”: the route table keeps the
128-SM performance policy and the launcher partitions a longer token range into 1536-column
slices, instead of shrinking the route. Shrinking it would move 1537–2688 to `MmaUnsplit`, whose
sequential FP32 accumulation misses the registered 1.4e-6 criterion (~1e-5, documented at the
T=2689 onset) and would break the documented prefill-chunk guidance. `catalog_can_partition` is
the guard that keeps the partition path (not the unsplit fallback) viable on the 4080.

Gate evidence:

- New host test `ninfer_gdn_gating_proj_residency_test` passes: per-SM facts, and single-launch
  ceilings at 76/128 SMs (27B split8 768/1280, split2 1536/2688; 35B split8 896/1536, split4
  1792/3072, split2 3648/6144).
- `ninfer_gdn_gating_proj_test` passes on the 4080 in 8.3 s (was aborting after 14 s).
- Full `ctest`: **121 tests, 108 passed, 12 skipped, 0 failed**; the same skip set as Stage 0.
- Bench sweeps with no rejection: 27B and 35B, `control` and `norm`, T = 1…4097. Partitioning
  appears exactly at the derived ceilings (nodes=2 from 769, 1537, 897, 1793, 3649).
- serve prefill past the old bound is deferred to Stage 4 (no `.ninfer` artifact exists yet); the
  unsplit path past 2688 is exercised by the op test and bench at T=4097.

Ledger/portal open item `Don-Chad 7afc8e17` (sm_89 form) is resolved in the working tree; the
durable ledger row lands with Stage 6.

### Stage 2 — the 3-bit scheme (2026-09-26, session 2)

Geometry decision (D3): **`Q3G128_F16S`**, the source grouping, is registered. The source is
repacked verbatim, so the extra group geometry is the only way to keep 3.125 b/w; the G64
duplicate-scale alternative would grow the artifact without changing represented values.

**Source packing pinned from the publisher, not inferred.** `compressed-tensors`
`pack_to_int32` (vllm-project/compressed-tensors, `compressors/pack_quantized/helpers.py`) packs
each 32-element block as a dense little-endian bitstream of `value + 2^(b-1)`, and the GSQ
`GumbelQuantizerInt` grid is the full two's-complement `[-2^(b-1), 2^(b-1)-1]` with `w = q *
scale`. The probe over `layers.5.mlp.gate_proj` confirms LSB-first: every 128-code group decodes
with max `+3` and min `-4`, and the bf16 scales are finite and positive. So the registered
scheme is codes `[-4, 3]`, one FP16 multiplier per 128 codes, reconstruction `code * scale`.

Implemented in this stage:

- `tools/artifact/numeric.py`: `Q3G128_F16S = QuantFormat(3, 128, -4, 3)`, registry and exports.
- `tools/artifact/layouts.py`: the `row-split-k128-v1` base plane for 3-bit codes — 48 bytes per
  128-code group as one dense little-endian stream (code `i` at bits `3i..3i+2`) — plus the
  encoder/decoder/dequantizer and a separate-group K-padding path. Independent bit-level oracle
  tests cover plane bytes (known vector `\xac\x8f\x68`), round trips, partial groups, gathers,
  and FP64 reconstruction (`tests/artifact/test_layouts.py`).
- `tensor-formats.md` and `storage-layouts.md`: the scheme is registered (ten formats, five
  grouped widths), section 10's Q3 exclusion is lifted for this one width/group pair, and the
  canonical encoder profile now covers five schemes.
- C++: `NumericFormat::Q3G128_F16S`, `QType::Q3G128_F16S`, reader parsing, row-split geometry
  (48-byte groups), typed binding. `test_ninfer_artifact_reader` accepts the synthetic Q3 object
  and rejects a wrong encoded size.
- `src/ops/linear/q3/`: GEMV for T=1…8 (warp-per-row, two groups per iteration, `uint4`
  activations) and a bf16 `m16n8k16` MMA schedule (32×64 tile, cp.async two-stage) for T>8.
  `ninfer_linear_q3_a16_test` passes the FP64 oracle on all five Qwen3.8-27B Q3 parent shapes
  plus `[4096,4304]` (partial final group) at T=1…128.
- `ninfer_linear_bench`: `q3`/`q3g128_f16s` accepted; suite entries for the five GSQ3 parents.

**Measured rates on the 4080** (`34816×5120` unless noted, cold L2, 20 samples):

| Route | T | median | Note |
|---|---:|---:|---|
| GEMV | 1 | 204.8 µs (340 GB/s) | Q4 GEMV on the same shape: 201.7 µs |
| GEMV | 4 | 274.4 µs | MTP3 width |
| GEMV | 8 | 398.3 µs | MTP7 width |
| MMA | 57 | 569.3 µs | one 64-token tile |
| MMA | 121 | 1005.6 µs / 42.9 TFLOPs | two tiles: 2× weight traffic |
| MMA | 256 | 1799.2 µs / 50.7 TFLOPs | four tiles |

Known limitation for the performance stage: the single 32×64 schedule re-reads every weight row
once per 64-token column tile, so T just above a multiple of 64 nearly doubles the time. A wider
tile (BN=128, dynamic shared memory) is the identified first fix. Also deferred to Stage 3/4: the
converter recipe (source adapter, fused row orders, MTP/vision/draft reuse, identity) and the
execution-leaf support for Q3 in `linear_swiglu`, `linear_add`, `attn_input_proj`,
`gdn_input_proj`, and the GDN convolution snapshot/record leaves.

Gate evidence: Python oracle/round-trip tests pass; `ninfer_linear_q3_a16_test` passes; the reader
accepts Q3 and rejects a malformed plane; the bench records the rates above. Full `ctest`: **122
tests, 110 passed, 12 skipped, 0 failed** (Stage 2 added the residency test and the Q3 linear
test).

### Stage 3 — converter and artifact (2026-09-26, session 3)

**The publisher packing is a shifted unsigned bitstream, not a bare two's complement plane.**
`compressed-tensors` packs `value + 2^(bits-1)` as a dense little-endian 32-bit-word stream
(LSB-first from the start of the row), while the registered artifact plane stores the signed code
in two's complement. The exact transform is therefore the XOR of the sign bit of every code
(`0b100` per 3-bit field, `0b1000` per 4-bit field). Verified on the real source with the lm_head
cross-check against the official BF16 shard: the shifted decode reconstructs the represented matrix
(rel-L2 0.114 for a 4-bit/64 matrix), while a two's-complement decode is catastrophically wrong
(rel-L2 3.12). The adapter never decodes or requantizes a code; it flips bits and copies.

**Scale-width finding (plan §3.2 risk R4).** 218 of the 240,271,360 source multipliers are not
binary16-exact. All 218 are bf16 subnormals whose fp16 rounding error is at most `2**-25` and
whose rounded result is still binary16 subnormal; the artifact therefore stores the rounded word
and the converter fails above that bound or on any normal-range inexact word. This is the
artifact's only value deviation from the source. The conversion report and the verifier both
carry the audit.

Delivered:

- `tools/convert/qwen3_8_27b/gsq3_source.py` — pack-quantized adapter (read/validate, sign-bit
  flip, scale audit, row select/concat, row-split payload encode) with an independent bit-level
  decoder and tests against a hand-built known vector (`\xac\x8f\x68`), round trips at both widths,
  subnormal rounding, and malformed-source rejection.
- `inventory_gsq3.py`, `recipe_gsq3.py` — 1124 object plan and recipes: Q3G128_F16S for the 320
  body matrices, Q4G64_F16S for embedding/output head/draft head, the groupwise MTP and Vision
  plans, and identical row fusions (query/key, gate/value, value/z, gate/up). No DFlash2 objects.
- `convert_gsq3.py` — pinned-source preflight (config scheme, source geometry, single-shard
  official subset, frontend hashes with official-first fallback), streaming writer, report with
  source revisions and the scale audit. Canonical output `out/qwen3_8_27b_gsq3.ninfer`:
  **12,023,113,728 bytes in 106.3 s**, identity `qwen3.8-27b/gsq3`, recipe
  `qwen3_8_27b_gsq3-v1`, 1124 objects (1118 tensors: 320 Q3, 57 Q4, 54 Q5, 1 Q6, 7 W8, 582 BF16,
  96 FP32, 1 I32).
- `verify_gsq3.py` — verifier-local row plans that rebuilds every packed plane from the source
  shards and compares bytes word-for-word, then compares every binary16 scale against the bf16
  source with the subnormal bound. Result on the artifact: structure 1124/1118/6; packed 323
  objects, 4,527,104 rows, 240,271,360 groups, **base bytes equal 323/323, scales equal 323/323,
  218 rounded, max error 2.98e-8**; 3 direct probes, 5 quantized probes (45 groups), draft 131072
  rows, 6 resources. `transformers` is absent in the container, so the AutoProcessor/GenerationConfig
  construction is reported as `transformers-absent` while the resource bytes and official hashes
  are still enforced.
- C++ profile: `WeightsProfile::Qwen38Gsq3`, `resolve_weights` for `qwen3.8-27b/gsq3`, GSQ3
  bindings (split attention and GDN parents, Q3 outputs, Q4 vocab endpoints), and Variant
  workspace capacities. Execution leaves: `linear_add` composes linear+residual_add;
  `attn_input_proj` and `gdn_input_proj` project one parent at a time through the qualified Q3
  linear route and publish column-wise row copies; the GDN convolution snapshot/record paths
  reuse the existing composed projected-plane machinery with the same Q3 projection lambda;
  `linear_swiglu` uses a fused Q3 MMA epilogue path — the two halves accumulate in FP32 through a
  column-chunked `[2*17408, chunk]` plane (chunk 64, 17.8 MiB peak) and the single BF16 rounding
  is the output store. `ninfer_linear_swiglu_q3_a16_test`, `ninfer_linear_add_q3_a16_test`, and
  the Q3 cases added to `ninfer_attn_input_proj_test`, `ninfer_gdn_input_proj_test`,
  `ninfer_gdn_input_proj_conv_snapshot_test`, and `ninfer_gdn_input_proj_conv_record_test` all
  pass. The first `linear_swiglu` composition (two BF16 projections) missed the registered A16
  criterion at the T=9 negative-activation replay by 0.7% (rel-L2 3.32e-3 vs 3.3e-3); the fused
  FP32 route is the fix.

Open for Stage 4/5: the chunked FP32 swiglu plane is correctness-first (a register-resident
two-half MMA epilogue is the identified performance step); the engine route (serve, 100K, MTP3,
vision) and the weight-quality gate remain; `transformers` is needed to run the frontend parser
check.

**Engine smoke on the 4080 (Stage 4 preview, same session).** During the first real load the Q4
vocabulary endpoints exposed two missing production paths: `embedding` had no Q4 table route and
the Q4 linear resolver had no `n=248320` schedule for the full output head. Both are now
implemented (`embed_gather_q4_kernel`; the head schedule mirrors the T regions of the other Q4
shapes with `launch_q4_mma_r64_c128` above T=16) and covered by the embedding and linear Q4
tests. With those, `ninfer-serve` on `out/qwen3_8_27b_gsq3.ninfer` with
`--max-context 2048 --kv-capacity 2048 --kv-dtype rk4v4-e8 --max-concurrency 1` starts
successfully: weights 10.2 GiB in 34.6 s, CUDA graphs 2.6 s, engine ready 42.4 s,
`available_after_startup_bytes=4.34 GiB`; one greedy chat completion produced coherent
reasoning at 20.3 tok/s (59 prompt tokens in 0.52 s). This is a load/generation smoke, not the
Stage 4 gate: the 100K + vision + MTP3 profile, exact retrieval, and the memory validation at
that profile are still open.

Gate evidence (final): full `ctest` **124 tests, 112 passed, 12 skipped, 0 failed** (the same skip
set as Stage 0); Python `tests/artifact tests/convert tests/test_bench_matrix.py
tests/test_serve_corpus.py` **94 passed, 5 skipped**, with the GSQ3 real-source and real-artifact
tests enabled; `git diff --check` clean.

### Stage 4 — engine bring-up on the 4080 (2026-09-26, session 4)

Profile: the §6 command with `--host-kv-mib 4096`, artifact `out/qwen3_8_27b_gsq3.ninfer`.
Gate evidence command (manual smoke; server stays alive while it runs):

```bash
python3 -m tools.smoke.serve_retrieval \
  --base-url http://127.0.0.1:8080 --model qwen3.8-27b \
  --output out/stage4-4080/retrieval.json
```

**Startup.** weights 11.2 GiB in 30.8 s (372 MiB/s), CUDA graphs 2.7 s, engine ready 38.6 s;
`kv_capacity_tokens=102400`, `runtime_reservation_bytes=2,747,221,760`,
`available_after_startup_bytes=978,825,216` (933 MiB). The engine validates capacity and finishes
CUDA-graph capture and warmup before it prints `listening`, so the memory gate precedes serving.

**Retrieval and vision.** All five probes pass on the real artifact: the single passphrase
(`QZ-7391-KESTREL`), the five vault codes, the three exact code-block facts
(`BEARING_WINDOW=37`, `phase_mdeg`, `int`), and the chart oracle (`NUMBER=731; CIRCLES=3;
SIDE=left`). A cold 98K prefill runs at 735.9–737.3 tok/s with `--prefill-chunk 1024` (TTFT
2m13.6s); once one session has completed, a rewrite reuses 98,226 tokens (99.9%, long anchor)
with TTFT 430–490 ms. MTP3 at 98K depth accepts 176/237 drafts (74.3%) at 48.3 tok/s decode;
the short retrieval outputs accept 90–100%. `/metrics` deltas equal the summed request `timings`
exactly (prompt, prefix-cache, request, draft, and accepted counters); `/slots` reports the busy
request as prompt 98,278 / reused 98,226 and the idle retained session as depth 98,269 with its
digest.

**`--prefill-chunk 2688` (deferred Stage 1 item).** Cold 98K prefill at 747.0 tok/s with no
`cudaErrorCooperativeLaunchTooLarge`; the 76-SM 1536-column partition path runs without error,
and the chunk-2688 runtime reservation (2.83 GB) still validates before listening.

**Host memory and the Windows "shared GPU memory" figure (maintainer observation).**
`nvidia-smi` reports 15.1 GiB used, while Windows Task Manager shows ~9 GB shared. The shared
figure is the engine's pinned host pools — 8.00 GiB Host KV + 1.15 GiB Host state — not sysmem
fallback of device allocations: container `Shmem` tracks it exactly. The GPU itself is fully
resident and at full boost while serving: 15,144 MiB used (933 MiB reserved slack), SM 2651 MHz,
memory 10801 MHz, 317 W average / 320.7 W peak, 100% utilization. A forced-eviction experiment
(an external 8 GiB CUDA allocation) dropped `nvidia-smi` to 11.5 GiB while idle, but a cached
decode (48.5 -> 48.3 tok/s) and the following cold prefill (735.9 tok/s) were unchanged: the
driver faults pages back on demand and the rates do not move. Reducing the pools to
`--host-kv-mib 2048 --host-state-slots 2` cuts `Shmem` to 2.5 GiB and runs the same rates, but
2 GiB of host KV holds only one deep 100K checkpoint: with a second retained session present,
the system-boundary long anchor is evicted and every rewrite re-prefills (observed: five cold
98K prefills in one suite). `--host-kv-mib 4096` with the default 8 host state slots keeps
`Shmem` at 5.5 GiB and reuse intact with a prior deep session present, so it is the recommended
4080 profile; the pinned host pools are a deliberate checkpoint-capacity cost, not a leak.

The retrieval/vision harness (`tools/smoke/serve_retrieval.py`) is new and documented in
`tests/README.md`. It samples `/metrics` and `/slots` during a decode-heavy request and fails
when the metrics deltas disagree with the summed request timings.

### Stage 5 — quality and performance (2026-09-26, session 4, partial)

**Weight quality (measured; reference comparison not measurable).** `ninfer-perplexity` on the
real artifact with the §6 command (`--corpus eval/corpora/perplexity-1m/manifest.json --quick
--kv-dtype int8`): overall PPL **4.596525** over 261,167 scored tokens at 448.5 tok/s (9m42s
scoring). Domains: chinese_reference 5.6999, english_long_form 7.2113, english_reference
6.3999, ninfer_code 1.6835. The report is `profiles/perplexity/gsq3-quick/report.json`.

The reference the gate named (`groupwise-int`, ~19 GiB of weights) cannot load on a 16 GB card
and no larger card is available. Decision (2026-09-26, maintainer): take the publisher's numbers
for the represented weights at face value and make no comparison. The artifact is a verified
verbatim repack (Stage 3: every packed plane byte-equal to an independent decode of the source
shards; the only value deviation is 218 subnormal bf16 scales rounded to fp16, max error
2.98e-8), the publisher evaluated exactly these weights (AIME 2025 100.00 vs 100.00 base, GPQA
Diamond 91.41 vs 89.90 base; GSQ model card), and the local PPL is kept only as an absolute
regression anchor, not as a margin claim.

**Performance at 100K (bench/README.md method).** `ninfer_bench` depth sweep on
`out/qwen3_8_27b_gsq3.ninfer` with `--kv-dtype rk4v4-e8 --max-ctx 102400 --spec mtp
--draft-tokens 3 --lm-head-draft`, 2 reps + 1 warmup per point. The committed corpus is
65,536 tokens, shorter than a 100K prefill, so the run used a local 131,072-token
tiled/rotated copy of it (`profiles/bench/bench_corpus_131072.ids`; hashes in
`profiles/bench/gsq3-4080-depth-mtp3.meta.txt`). Report:
`profiles/bench/gsq3-4080-depth-mtp3.json`.

| P | prefill t/s | decode out t/s | MTP acceptance |
|---:|---:|---:|---:|
| 8,192 | 906.8 | 66.6 | 100% |
| 32,768 | 852.1 | 64.7 | 100% |
| 65,536 | 791.7 | 62.0 | 100% |
| 100,000 | 733.7 | 59.4 | 100% |

The 100% acceptance is a fixture property, not a model result: the committed corpus is
explicitly tiled, so the greedy continuation from a slice repeats earlier text and the draft
head is never wrong. The serve measurements on mixed text are the honest decode rows: 48.3
tok/s at 98K depth with 74.3% acceptance (the LRU code request), 62.0 tok/s on the 456-token
vision prompt with 91.7% acceptance, and 44.0 tok/s at an 8K context with 53.8% acceptance on
the CLI Python package scenario. Prefill agrees across routes (735.9–737.3 tok/s serve at 98K,
733.7 bench at 100K, 747.0 at `--prefill-chunk 2688`).

**llama.cpp comparison (measured 2026-09-26).** The maintainer's beellama server
(`http://host.docker.internal:9931/v1`, `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, IQ3_S
3.44 bpw, kvarn5, `n_ctx=100096`, MTP on) ran the same workloads:

| Workload | beellama | NInfer GSQ3 |
|---|---|---|
| LRU code, ~98K depth, MTP3, 256 tokens | prefill 1042.8 tok/s; decode 46.9–48.0 tok/s; draft accepted 78.5–81.5% | prefill 734.7–737.3 tok/s; decode 48.3 tok/s; draft accepted 74.3% |
| Python package scenario, shallow, 512 tokens | decode 71.0 tok/s; draft accepted 60.7% | decode 44.0 tok/s; draft accepted 53.8% |

Method: the depth row is the same request text (serve_retrieval document + LRU prompt) sized
with each server's tokenizer (97,752 llama tokens vs 98,278 NInfer tokens), `temperature 0`,
thinking off; beellama numbers are its server `timings` over a cold-prefill run and a
cached-prompt repeat (files under `profiles/bench/beellama-4080-*`), NInfer's are
`retrieval5.json`. The shallow row uses `examples/cli/messages/scenario_code_python.json`; the
NInfer side is the CLI summary, the beellama side its server timings.

Reading: at 98K depth the two engines decode at parity (NInfer 48.3 against 46.9–48.0) while
llama.cpp prefills 42% faster (1042.8 against 734.7–737.3); shallow, beellama decodes 61%
faster (71.0 against 44.0) and accepts more drafts (60.7% against 53.8%). Per-round time is
~40 ms for beellama against ~60 ms for NInfer at shallow depth, and ~72 ms against ~67 ms at
98K. So the Stage 5 decode gate is met at depth and not met shallow, and the shallow gap plus
the prefill deficit are recorded as measured; kernel attribution is still open (the Stage 2
single 32x64 MMA schedule and the small-T GEMV paths are the identified tuning steps).

### Stage 5c.1 — Q3 tall A16 GEMM (2026-09-27, session 6, GPU session cut short)

Landed the port described in §6: `src/ops/linear/q3/q3_rowsplit_tall_mma.{cuh,cu}` (128 weight
rows x 64/128 tokens per CTA, 64-code half-group steps, one barrier per step, registers one step
ahead), the folded gate/up SwiGLU problem, the route thresholds in `q3_dispatch.cpp` and
`q3_linear_swiglu.cu`, and the byte-compare test `tests/ops/linear/test_q3_a16_tall.cpp`. All
Q3 op suites pass and the new test is byte-identical to the staged route. Bench and end-to-end
numbers are in the §6 result block. The session ended with the host shutdown while the
`pp100000` run was in flight. Resume on the GPU with, in order:

1. `ctest --test-dir build --output-on-failure` — the full set has not run since the tall route
   landed (only the Q3 op suites and the byte-compare test have).
2. The `pp100000 --prefill-chunk 2688` command from §6, recorded against the 870.3 tok/s
   pre-5c.1 baseline.
3. If the 100K gain is materially smaller than the 32K one, add the tail-aware split: at
   `t >= 128`, launch the 128-token tile over the first `floor(t/128)*128` columns and the
   staged 32x64 route over the remainder, then re-run the byte-compare test and the sweep.
4. An nsys profile of `pp32768` to confirm the tall kernel holds the expected share of prefill.

The work is uncommitted in the working tree until the maintainer asks for a commit.

### Stage 5c.1 verification (2026-09-27, session 7, GPU back)

The interrupted measurements were completed on the tall build:

- `pp100000 --prefill-chunk 2688`: **1166.48 tok/s** against the 871.02 pre-5c.1 baseline
  (+33.9%) and beellama's 1042.8; log `profiles/bench/5c-tall/session7-pp100000.log`. 5c.2's
  named gate (1043) is therefore already met, so that port is now headroom rather than the gate
  closer.
- Full `ctest`: 125 tests, 113 passed, 12 expected skips, 0 failed, 516.6 s.
- Shallow CLI scenario (greedy, `--no-thinking`, MTP3): 73.2 tok/s over 14 rounds, decode phase
  560 ms (40.0 ms/round) against session 5's 72.9 tok/s / 40.1 ms; output and acceptance are
  unchanged, as expected for a prefill-only route. Logs `profiles/bench/5c-tall/session7-code-cli-
greedy.{out,err}`.
- D10's measured axes are met. The next work item is the maintainer's call: 5c.2 headroom, 5c.3
  decode lead, or Stage 6 publication. The tail-aware split noted in §6 remains the optional
  fix for the mid-width regressions.

### Stage 5c.3 — n-gram drafts (2026-09-27, session 7, blocked on a wide-verify bug)

The port itself is complete and tested except for the wide verify window:

- `ngram_pool.h`, `ngram_policy.h`, `test_ngram_policy` (pool, policy, source attribution, and
  round-width policy) pass.
- `535f9c1`'s verify-window split is ported end to end: `include/ninfer/ops/mtp_round.h`,
  `src/ops/{kernel,launcher,wrapper}/mtp_round.*`, `round_state.{h,cpp}`, `schedule.h`, `mtp_impl.h`,
  the graph families/allowance, the ReplaySSM narrowed views and fold, and the workspace sizes.
  `ninfer_linear_q3_a16_test` and the other op suites are unchanged; the `mtp_round` oracle test now
  covers every `(V,k)` pair.
- `decode_mtp_batch` chains the pool after the MTP proposal; the CLI and server expose
  `--ngram chain`/`--ngram-max`/`--ngram-n`/`--ngram-min`/`--ngram-pool-mib`; request-log schema
  21 and the metrics counters carry the n-gram share; `docs/cli.md`, `docs/serving.md`, and
  `README.md` are updated.
- `tests/targets/qwen3_6_27b/test_engine_ngram_real.cpp` passes on the real artifact with 0 added
  divergences and the speculative rounds exercised.

**The wide verify window is disabled.** With it enabled, one wide round on the real artifact
produced a wrong correction token:

- Round E=467, width 7, extent 6, proposal 3. verify ids `[5653 1870 1137 5480 2923 16 23]`,
  positions `[467..473]`. The chain was correct; the verify rejected draft 5480 at column 2 and
  emitted 4075, whose scoring-route log-prob is -6.00 against 5480's -0.0035 (raw verify logits at
  column 2: 4075 19.75, 591 17.75, 5480 17.63). Columns 0,1,3,4,5,6 were correct.
- Ruled out by direct tests: the attention op at width 7/window 474 (new masked and unmasked oracle
  cases added to `ninfer_softmax_attention_test` pass), the GDN replay-record op (width sweep 2..16
  passes), the Q3 GEMV and fused SwiGLU GEMV at T=5/6/7 (new token cases pass), the W8 lm_head at
  T=7 (the suite covers 1..128), the fold/record width selection (by construction, and the narrowed
  views are exercised), CUDA graphs (the failure reproduces with graphs off), OOB memory and
  shared-memory races (`compute-sanitizer --tool memcheck` and `--tool racecheck`: 0 errors).
- Remaining suspect: the assembled multi-column verify state, i.e. a single-column hidden
  corruption in the T=7 verify stack. A minimal repro was used during the investigation: one engine,
  prompt 3, `--ngram-max 6`, graphs off, 384 tokens; the divergence is at generated index 311. That
  probe was temporary and has been removed.
- The planner keeps `verify_window == draft_window` (wide planes are not materialized); the real
  and docs state that a round verifies at most the MTP width. Full `ctest`: 127 tests, 115 passed,
  12 expected skips, 0 failed.

Resume options for the next session: (a) explain the column-2 corruption (e.g. by dumping the
per-layer hidden states of the failing verify round and comparing them with a prefill of the same
prefix), or (b) drop the n-gram feature if the wide path is not worth it, since D10 is already met
without it. Do not re-enable wide rounds before the corruption is explained.

### Decode bandwidth audit (2026-09-27, session 7 follow-up)

Prompted by the maintainer's challenge to the ~400 GB/s figure. With the corrected calibration
(read 664 GB/s, see the Stage 0 correction), the shallow decode round was re-derived from
`profiles/nsys/gsq3-shallow-tg32-after.sqlite`. The capture holds 22 verify rounds (5632 plain +
1408 SwiGLU Q3 launches = 256 + 64 per round) in 893 ms of wall, i.e. 40.6 ms per round:

| Path | Measured | Payload | Rate |
|---|---:|---:|---:|
| Q3 plain GEMV (T=4 verify) | 20.05 ms/round | 5.046 GB/round | 252 GB/s |
| Q3 fused SwiGLU GEMV (T=4) | 12.37 ms/round | 4.456 GB/round | 360 GB/s |
| Q4 draft head (3 x T=1) | 1.61 ms/round | 1.07 GB/round | 664 GB/s |
| Q4 main head (T=4) | 1.13 ms/launch | 0.675 GB | 600 GB/s |
| W8 MTP gate_up (34816x5120) | 0.30 ms/launch | 0.178 GB | 604 GB/s |

Op bench, same shape `34816x5120`, cold L2, this card: Q3 T=1 202 us (346 GB/s), T=4 275 us
(254 GB/s), T=8 398 us (176 GB/s); Q4 T=1 204 us (465 GB/s), T=4 214 us (444 GB/s). Q3 and Q4 move
the same number of codes and take the same time at T=1 even though Q4 moves 36% more bytes, so the
Q3 small-T path is code-processing/latency-bound, not byte-bound: per 8 codes it issues three scalar
byte loads straight from global (no staging, `__launch_bounds__(256,2)`, one warp per row, two CTAs
per SM), while the Q4 path stages 16-byte vectors with cp.async into shared memory and decodes from
shared.

Two separate gaps, both actionable: (a) the Q3 decode kernels run at ~half the read rate the Q4/W8
kernels already demonstrate on the same device; (b) each extra token costs ~9.4% of the T=1 time on
these shapes (T=4 = 1.35x T=1, T=8 = 1.97x), which is per-token activation work, not weights.
Streaming the 9.5 GB Q3 body at the Q4 same-shape rate (444 GB/s) would take ~21 ms and at the
device read rate ~14 ms, against 32.4 ms today: the 40.6 ms round becomes ~30 or ~23 ms, i.e.
~88-116 tok/s at the current 2.62 tok/round.

The port is a Q3 analogue of the Q4 `AsyncVector16`/shared-memory GEMV schedule - a 48-byte Q3
group is exactly three uint4, so the same staging shape works - decoding the eight-code windows
from shared, with two rows per warp considered to halve the per-token activation re-reads. This is
the top decode item; it is independent of the 5c.3 wide-verify bug and of the D10 publication
decision.

Note for the bench: `ninfer_linear_bench`'s `DRAM_%`/`READ_%` columns are calibrated to the 5090
constant (1792 GB/s, printed as `dram_spec_gbs`), so they read ~2.4x low on this card; use the raw
GB/s columns.
