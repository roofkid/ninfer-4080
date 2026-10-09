# RTX 4080 bring-up and 3-bit GSQ artifact

**Status: ACTIVE — this file is the resume point for the work.** Created 2026-09-26 from the
feasibility session on `rtx4090-port`. It is a temporary plan, not a permanent reference: delete it
when the work is finished or abandoned (AGENTS.md, "Change consistency").

**Progress (2026-09-27, sessions 6–10): Stages 0–5 are recorded, and Stages 5c.1 and 5c.2 are
complete with their engine numbers measured.** Stage 4's gate passed on the 4080: the 100K + vision + MTP3
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
found a wrong correction logit in the wide verify route on the real artifact, so the wide n-gram
rounds stayed disabled (the planner kept `verify_window == draft_window`). Session 15 traced the
flips to bf16-ulp trajectory sensitivity rather than a kernel defect, and session 19 re-enabled the
window (see the Stage 5c.3 block and the session-19 record in §11).
The full `ctest` set passes on the current build (127 tests, 115 passed, 12 expected skips, 0
failed).
**Session 8 (2026-09-27) took the decode-bandwidth audit's top item: the Q3 small-T GEMV now
runs one cp.async-staged schedule for both the plain and the fused gate/up problem.** The new
route is byte-identical to the direct route it replaces (a new byte-compare test covers every
registered Q3 parent shape at T=1..8) and is 14.2% faster at T=1 and 4.5% faster at T=4, the
MTP3 verify width; the engine decode is unchanged within noise (53.7 vs 53.8 tok/s on the §6
`tg512` run). Details, the measured schedule sweep, and the identified next step (a small-T
tensor-core or A8 route rather than more SIMT tuning) are in §11. The maintainer's D9
publication decision is unblocked.
**Session 9 (2026-09-27) landed 5c.2, the Q3 A8 int8 prefill route.** With `AllowA8` the exact-K
Q3 parents run the documented per-token 64-code activation quantization and m16n8k32 s8 MMAs
from 129 columns on; decode, MTP verification and padded-K problems keep A16. The new op suites
apply the documented quantization in their FP64 oracle. On the 4080 the route is ~1.9x the A16
tall GEMM at T=129..513, `pp32768` 1406.34 -> 2277.53 tok/s, `pp100000 --prefill-chunk 2688`
1166.48 -> 1675.39 tok/s, quick perplexity 4.596525 -> 4.596095, full `ctest` clean, and decode
unchanged (tg512 53.2-53.9 tok/s, same acceptance). The same kernel at T<=8 is slower than the
staged small-T GEMV (377-420 us against 174-264 us on 34816x5120), so A8 was **not** enabled at
decode widths and the decode prize still needs a dedicated small-T tensor-core design; details
and the measured cause are in §11. **Session 10 landed the small-T bf16 MMA decode route the audit
named**: T=2..8 and the fused gate/up from two columns on, 106.7 tok/s on the tg512 fixture,
71.9 tok/s on the real CLI scenario against 55.4, and 102.9 tok/s at 98K depth; §11 records the
design, the measured schedule space, and the A8 follow-up. **Session 11 landed the DFlash2
companion in the artifact and its K=7 profile** (greedy-lossless, ~93 tok/s shallow and
203/180 tok/s at 28K/56K depth against MTP3's 118/~112) at the stock-companion caps of 28K
context with vision and 56K without. **Session 12 (2026-09-28) landed option 4's matrix half:** the
gsq3 identity now carries the requantized Q4G64_F16S companion (selector and norms stay BF16),
which moves the DFlash2 caps on this card to 100,000 tokens without vision and 65,536 with it at
the same safety margin, keeps K=7 greedy-lossless, and measures 100.9 tok/s at 38.2% acceptance
on the code scenario (stock 93.4/35.7%) and 124.2/205.5/182.7 tok/s at 8K/28K/56K depth. The
8-bit selector codebook (option 4's remaining ~0.12 GiB) is the only unlanded slice; §11 has the
design, verification, and measured fit.

**Session 13 (2026-09-28) is a read-only audit: fresh per-kernel decode/prefill breakdowns, the A8
tall GEMM's measured streaming floor, and a ranked candidate list are in §11 ("Remaining-
performance audit"). The same session re-framed the 5c.3 wide-verify blocker: it is a small-T
multi-split E8 corruption that also occurs in narrow rounds, not a wide-window bug; §11
("Wide-verify root-cause update") has the evidence, the single-split localization, and the next
step. No product code changed.**

**Session 14 (2026-09-28) corrected the executed-path analysis of the same blocker: RK4V4E8
is the E8-lattice packed-int4 mode, the session-13 K-decode suspects belong to RK2V4E8 only, and
a 64M-sample device probe shows the warp-cooperative E8 projection agrees with the scalar and
independent nearest-E8 projectors; see "RK4V4E8 executed-path correction" at the end of §11.**

**Session 14's stock-test control then showed the gross deviation is not E8-specific: at default
settings int8 diverges by 20.97 nats and rk4v4-e8 by 6.75 where bf16 stays at 0.5, so the
shared quantized-KV small-T path is the target; see the control measurement at the end of §11.**

**Session 15 (2026-09-29) closed the 5c.3 blocker as trajectory sensitivity, not a defect: the
gross quantized-KV flips reproduce from a bf16-ulp-level route difference at one position that
grows to 8-14 nats over ~270 greedy steps in bf16 as well as int8, at a position where the
decode and prefill states agree to bf16 ulp and the op oracle passes. It also landed the plan's
named next step: the RK4V4E8 host codec + FP64 oracle + append byte parity in
`tests/ops/softmax_attention/causal_cache.cpp`, and the rotated-V double-rounding fix that suite
demands (`787766f`) for rk4v4/rk4v4-e8/rk2v4-e8, which the shipping 100K profile uses. Full
`ctest` is 132/119/13/0 and the `rk4v4-e8` n-gram real route passes. See the two session-15
sections at the end of §11. The wide-window re-enable and the pre-existing K=5 divergence are now
maintainer decisions, not open investigations. **Session 16 (2026-09-30) closed the session-13
audit's two decode candidates: the small-T tensor-core route now covers 9..16 columns with one
CTA per 8-column tile (DFlash2 K=15 39.6 -> 53.0 tok/s on the CLI scenario), and an A8 profile of
the same engine serves 2..16 columns with the documented activation quantization (op level
-15..-17% against A16, greedy engine +10..13%, DFlash2 K=7/K=15 texts byte-identical). §11 has the
design, the fixed group-major scale-staging bug, and the MTP3 near-tie drift that the decode A8
profile introduces.**

**Session 19 (2026-10-02) re-enabled the 5c.3 wide n-gram verify window** (the planner now sets
`verify_window = ngram.max_drafts` when `--ngram chain` is enabled) and made the real test require a
wide round. On bf16 with the default graph route: 80 wide rounds of 560, 0 n-gram-added divergences,
PASS; the structured-JSONL CLI scenario drops 68 -> 58 decode rounds (129.1 -> 139.8 tok/s) with
byte-identical output. The `--no-cuda-graph` route still shows the session-13..17 trajectory-flip
class; §11 has the record.

**Session 25 (2026-10-03) cut the Q3 A8 small-T in-kernel consume** (session-21 item b): A fragments
are now spread straight from the staged codes into the mma fragment layout, removing the decoded int8
tile, its ldmatrix read and the decode barrier. The outputs are byte-identical; the clean-flush op
gains 15-27% and the same-build DFlash2 K=7 CLI decode is 121.4 -> 136.1 tok/s (+12.1%). §11 has the
probe split, the new `ninfer_linear_bench --flush read` option, and the nsys numbers. The remaining
decode item from session 21 is the native T=9..16 single-pass tile for K=15.

**Session 26 (2026-10-03) landed 5c.4, the prompt attention worker V-dequant port** — the
session-13 audit's ranked candidate 5 and the last untouched prefill kernel. The fork's pipelined
producer/worker schedule (P in registers, workers own PV and V, one-sided `PFree`/`PReady` barriers)
plus the bytewise `kv_cache_unpack_i4x16` are in; the op bench gains `--kv-dtype rk4v4-e8`. Same-session
A/B on the 4080: append bench int8 -32..-35% at 8K-128K, engine `pp32768` 2314.1 -> 2470.0 tok/s
(+6.7%) and `pp100000 --prefill-chunk 2688` 1710.0 -> 1971.4 tok/s (+15.3%); §11 has the record and
the unchanged real-route checks.

**Session 27 (2026-10-03) prepared the Stage 6 publication.** Landed: the model card,
manifest, and hash list under `model-cards/Qwen3.8-27B-GSQ3-NInfer/`; the `gsq3` contract as
section 14 of `qwen3.8-27b-artifact.md`; `scripts/download-qwen38-gsq3.{sh,bat}`; the 4080-first
README with the three-command quick start (download -> docker run -> serve); and pull-first
launchers (`roofkid/ninfer-4080:gsq3`, source build as fallback) for both the MTP3 and DFlash2
profiles. Published 2026-10-03: the Hugging Face repository `roofkid/Qwen3.8-27B-GSQ3-NInfer`
(main `2359d374`) carries the artifact, and the Docker Hub image `roofkid/ninfer-4080:gsq3` is
pushed; `ghcr.io` is deferred.

**Session 28 (2026-10-09) landed session-21 item (c), the native T=9..16 small-T tile.** The
9..16-column verify windows (DFlash2 K=15, wide n-gram) now run one 16-column CTA per row block in
both the A16 and the A8 small-T engines, so every code byte is staged once; the 8-column route is
untouched. Same-binaries A/B on `34816x5120`, clean flush: A8 T=9..11 0.65x and T=12..16 0.69x,
A16 T=9..16 0.68-0.69x, T=2..8 at parity. The engine DFlash2 K=15 code scenario drops 465 -> 372 ms
(88.2 -> 110.2 tok/s, +25%) with the greedy output hash unchanged, K=7 and MTP3 are at parity, the
real DFlash2 and n-gram routes pass, and full `ctest` is 133/120/13/0. §11 has the design and the
codegen finding that kept the narrow kernels from regressing.

**Session 29 (2026-10-09) made the MTP n-gram chain the default.** `NgramOptions::mode` is now
`Chain`, so `--spec mtp` runs the session-7/19 host n-gram chain unless `--ngram off` is passed;
DFlash/DFlash2 clear the option at parse time so logs, metrics and the engine plan report the
effective mode. This is where the session-28 wide tile pays in the shipped profile: on the
structured-JSONL scenario MTP3 goes 145.8 -> 166.9 tok/s with an unchanged output hash, the tiled
corpus depth sweep goes 151/142/130/122 -> 361/385/332/303 tok/s (DFlash2 K=7: 168/265/241/213),
and the 100K + vision profile still validates with ~827 MiB free (-53 MiB). The one-shot code
scenario is unaffected (0 wide rounds). §11 has the design, memory and comparison record.

Environment for this plan: the `Dockerfile.dev` image in this repository. It is the sandbox the
maintainer hands to pi, with the host RTX 4080 passed through:

```bash
docker build -f Dockerfile.dev -t ninfer-4080-dev:pi .
docker run --rm -it --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/work" -v "$HOME/ninfer-models:/models" \
  ninfer-4080-dev:pi
```

For daily serving rather than development, `scripts/run-ninfer-4080.bat` (Windows host) and
`scripts/run-ninfer-4080.sh` build the product `Dockerfile` image (first run, and whenever the
checkout revision changes) and start the served 100K MTP3 profile with the artifact bind-mounted
read-only; every build stamps `org.ninfer.revision`, so pulling source changes reaches the served
binary without a manual rebuild. The dev container above stays the build and measurement
environment.

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
64-wide halves (weights bit-identical, +0.35 GiB, one group geometry).
**Decision (2026-09-27, maintainer): G128 stays.** The G64 duplication buys no accuracy (identical
represented weights) and costs both VRAM and stream bytes, so the single-geometry convenience does
not pay; revisit only if that registry rule becomes a binding constraint.

Quantified (2026-09-27, from the converted artifact): the Q3 body is 9.123 GB of code planes +
0.380 GB of fp16 scales = 9.503 GB. G64 leaves the code planes byte-identical and doubles only the
scale plane, so the cost is +0.380 GB = +0.354 GiB and +4.0% weight bytes; those bytes stream once
per verify round (decode) and once per prefill pass, so decode loses roughly 3% (73.2 -> ~71 tok/s)
and prefill roughly 3.5% (pp32768 1406 -> ~1358, pp100000 1166 -> ~1126 tok/s). Accuracy is
unchanged either way, because duplication keeps every represented weight at its source scale and
value; only a true per-64-code requantization would change quality (and would stop being a verbatim
repack of the publisher's weights).

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
Nsight Systems. The bundled importer needs `libdw.so.1` and lives at
`/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter`:
session 8 found this image without it and installed it with
`apt-get install -y --no-install-recommends libdw1t64`; that is a container change, not a repo
one, so re-check it after any image rebuild. If `nsys profile` fails to write a `.nsys-rep`, run
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

**5c.3 result (2026-09-27, session 7; wide window re-enabled session 19).** Landed: `ngram_pool.h`/`ngram_policy.h` + `test_ngram_policy`, the
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
The session-7 engine planned `verify_window == draft_window`, so the wide graph family and record
planes were not materialized and a round verified at most the MTP width; session 19 restored
`verify_window = ngram.max_drafts` and the real test requires a wide round.

**5c.4 Prompt attention worker V-dequant (DONE 2026-10-03, session 26; evidence below).**

- Fork `7b6ed55` ("move V dequant to the workers in the int8 prompt kernel"). Ours predated the
  fork's producer/worker pipeline, so this landed the whole schedule, not only the delta:
  producers keep P in registers and own QK/softmax/K; workers own the FP32 output accumulator,
  PV and V (issue + dequant); `PFree`/`PReady` one-sided named barriers replace the two per-tile
  block barriers; V(t+1) is dequantized by the workers right after PV(t), off the producers'
  scoring path; packed K codes land in the upper half of each packed V row and each producer
  expands the chunks it issued. The shared `kv_cache_unpack_i4x16` is bytewise (xor, add, xor,
  byte permutes). `ninfer_causal_softmax_attention_bench` gained `--kv-dtype rk4v4-e8` so the
  production KV mode is measurable.
- Acceptance met: `ninfer_softmax_attention_test` full and `--rk4v4-e8-only` pass; the append
  bench improves 32-35% at 8K-128K (int8; rk4v4-e8 measured after); engine prefill recorded
  (same-session A/B). Full `ctest` 133/120/13/0; the n-gram and DFlash2 real routes reproduce
  their recorded outputs.

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

**Session-11 outcome:** the artifact half is landed (stock companion, verified), and DFlash2 K=7 -
the window users are expected to run - is greedy-lossless and measured at the caps in §11. The
pre-existing K=5 divergence is reproduced on the old Q3 GEMV route as well and is deferred; the
requantized companion (the memory decision) was the next step. **Session-12 outcome:** the
requantized companion's matrix half is landed (Q4G64_F16S, 13,330,776,576 bytes) and the K=7
profile now fits 100,000 tokens text-only / 65,536 with vision at the same safety margin; the
8-bit selector codebook remains as the final ~0.12 GiB slice.

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

**Landed 2026-10-03 (session 27).** `model-cards/Qwen3.8-27B-GSQ3-NInfer/` (HF card,
manifest, licenses, `SHA256SUMS`), `qwen3.8-27b-artifact.md` section 14, the download scripts,
the 4080 README quick start, the published-image launchers, and the port-ledger publication
row. The Hugging Face model repository (`roofkid/Qwen3.8-27B-GSQ3-NInfer`, main `2359d374`) and
the Docker Hub image (`roofkid/ninfer-4080:gsq3`, tag `0.6.1-rtx4080`) are live; `ghcr.io` is
deferred. Remaining: the fork push of this commit.

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
6. If the task is the remaining performance work, read **Stage 5c** first. 5c.1 and 5c.2
   (sessions 6 and 9, §11) are complete: the Q3 A8 prefill route takes 32K to 2277.53 and 100K
   `--prefill-chunk 2688` to 1675.39 tok/s with the quality anchor unchanged (4.596095 quick).
   Session 10 then landed the decode axis the audit named: the dedicated small-T bf16 MMA route
   (whole-group staging, three-CTA staging ring, eight-warp K-split) replaces the staged GEMV
   from T=2 through T=8 and the fused gate/up variant from two columns on, taking the shallow
   CLI scenario 55.4 -> 71.9 tok/s and the depth sweep 59.4 -> 102.9 tok/s at 98K (same
   acceptance on the real scenario). Session 16 (§11) then landed the two decode levers the
   session-13 audit ranked: the small-T route now serves 9..16 columns (one CTA per 8-column tile;
   DFlash2 K=15 39.6 -> 53.0 tok/s) and an A8 profile of the same engine serves 2..16 columns
   (op level -15..-17%, greedy engine +10..13%, DFlash2 K=7/K=15 texts byte-identical). The decode
   A8 profile changes decode numerics by design and the MTP3 greedy text drifts one phrase at a
   near-tie, which is the one open maintainer call. 5c.3's wide verify window is re-enabled (session
   19); Stage 6 publication is otherwise unblocked, and 5c.5 (DFlash2) is past its artifact
   gate: session 11 (§11) landed the
   companion in the GSQ3 identity and its K=7 profile at 28K (vision) / 56K (text) context, which
   is greedy-lossless; session 12 (§11) then requantized the companion matrices to Q4G64_F16S,
   which puts the K=7 profile at 100,000 tokens text-only / 65,536 with vision at the same safety
   margin with unchanged-or-better decode. The 8-bit selector codebook is option 4's last slice;
   the pre-existing K=5 narrow-window divergence is documented and deferred.
   Session 15 (§11) closed the wide-verify blocker as trajectory sensitivity rather than a kernel
   defect, landed the RK4V4E8 oracle coverage + the rotated-V double-rounding fix, and left the
   wide-window re-enable (landed in session 19) and DFlash2 K=5 as maintainer calls. The landed work
   through session 25 is committed on `rtx4080-port`; new work stays uncommitted until the
   maintainer asks for a commit. Session 26 (§11) then landed 5c.4, the last untouched prefill
   kernel: the prompt attention worker V-dequant schedule takes the append bench 32-35% down at
   8K-128K and the engine same-session A/B to `pp32768` 2470 and `pp100000` 1971 tok/s, with the
   real n-gram/DFlash2 routes unchanged. No prefill item from the session-13 audit remains; the
   Session 28 (§11) then landed the last open decode item, the native T=9..16 small-T tile
   (session 21 item c): DFlash2 K=15 went 88.2 -> 110.2 tok/s at the same output hash, no narrow
   route regressed, and the session-21 next-step list is closed.

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

### Stage 5c.3 — n-gram drafts (2026-09-27, session 7; wide window re-enabled session 19)

The port itself is complete and tested; **the wide verify window is enabled** (session 19 restored
`verify_window = ngram.max_drafts` in the planner after session 15 explained the session-7 failure
as trajectory sensitivity; the real test now requires a wide round). Original session-7 record:

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

**The session-7 wide verify window was disabled.** With it enabled, one wide round on the real artifact
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
- The planner kept `verify_window == draft_window` (wide planes were not materialized) until session
  19. The historical `ctest` at the time: 127 tests, 115 passed, 12 expected skips, 0 failed.

Resume options for the next session: (a) explain the column-2 corruption (e.g. by dumping the
per-layer hidden states of the failing verify round and comparing them with a prefill of the same
prefix), or (b) drop the n-gram feature if the wide path is not worth it, since D10 is already met
without it. Do not re-enable wide rounds before the corruption is explained. **Session 13
partially answered (a): the corruption is not wide-window-specific and the small-T multi-split
E8 path is the culprit; see "Wide-verify root-cause update" below.**
**Session 15 answered (a) completely: the flips are the amplification of bf16-ulp route
differences and reproduce in bf16 as well as int8; see the session-15 section at the end of §11. The
wide window itself is not the cause and can be re-enabled as a maintainer decision.**

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

### Q3 small-T staged GEMV (2026-09-27, session 8)

**Landed.** The audit's top item: the direct warp-per-row Q3 GEMV (three scalar byte loads per
window straight from global, no staging) is replaced by a cp.async-staged schedule. New file
`src/ops/linear/q3/q3_rowsplit_gemv_staged.cuh` with `launch_q3_gemv_r8_c8_staged`; the fused
gate/up (SwiGLU) decode route now runs the same kernel with the `Fused` problem instead of its
own direct kernel (`q3_rowsplit_gemv_swiglu_kernel` removed).

Shape of the schedule: one warp owns its rows' full K extent and walks it in stages of eight
128-code groups. The next stages' code bytes are staged with `cp.async.cg` into warp-private
shared memory (three-deep pipeline) and each lane decodes its eight-code window (bytes 3l..3l+2
of the 48-byte group) from shared memory once per window, keeping the weights in registers while
the column loop applies the activation values. The 32-lane warp covers two adjacent groups per
iteration with lane half selecting the group. The plain problem puts one row per warp; the fused
problem gives each lane the gate and up window of the same group, so the pair shares one
activation read and the epilogue publishes `silu(gate) * up` with one BF16 rounding.

**Exactness.** The staged plain route is byte-identical to the direct one: the new
`tests/ops/linear/test_q3_a16_gemv.cpp` byte-compares `launch_q3_gemv_r8_c8` (kept as the
reference), `launch_q3_gemv_r8_c8_staged` and the dispatched route over all seven registered Q3
parent shapes at T=1..8, including the padded `[4096,4304]` shape. The fused staged route was
byte-compared against the replaced direct kernel with a standalone probe on the real artifact's
layer-0 `gate_up` weights at T=1..8 (identical, warm and cold); its in-tree evidence is the
oracle suite `ninfer_linear_swiglu_q3_a16_test` plus the Q3 `linear`/`linear_add`/input-projection
suites, which all pass.

**Op bench** (`ninfer_linear_bench --qtype Q3 --n 34816 --k 5120 --sweep 1:9:1 --warmup 3
--repeat 10`, cold L2, median µs; baseline = the direct route at the start of the session):

| T | direct | staged | change |
|---:|---:|---:|---:|
| 1 | 201.7 | 173.1 | -14.2% |
| 2 | 228.4 | 199.7 | -12.5% |
| 3 | 249.0 | 231.4 | -7.1% |
| 4 | 275.7 | 263.2 | -4.5% |
| 5 | 302.1 | 297.0 | -1.7% |
| 6 | 331.7 | 331.8 | 0.0% |
| 7 | 362.5 | 366.6 | +1.1% |
| 8 | 397.3 | 402.7 | +1.4% |

CSV: `profiles/bench/5c-small-t/q3_34816x5120.csv`. The widths the MTP route uses are 1..6,
where the staged route is equal or better.

**Fused route at T=4** on the real layer-0 `gate_up` parent (standalone probe, real weights,
cold L2 every sample): direct 261.3 µs -> staged 247.2 µs, at identical output bytes.

**Engine.** The §6 `tg512` command, three runs each on the same card and build flags: baseline
53.87 / 53.81 / 53.68 tok/s and staged 53.72 / 53.84 / 53.70 tok/s, all at acceptance
0.3991416309 over 233 rounds. The change is engine-neutral: the op-level gain is real, but the
round's Q3 time is dominated by the 256 small parent calls whose per-launch time is latency-bound
rather than stream-bound, and by the rest of the round.

**Measured schedule space (no further gain).** Each of these was run on the real shape: cp.async
pipeline depth 3..7, stage size 2/4/8/16 groups, launch-bounds occupancy from 24 to 48 warps/SM,
one row per warp vs the paired two-rows-per-warp layout, and `ca` vs `cg` staging. The best
configuration is the landed one (8 groups, 3 stages, `cg`, 48 warps/SM plain / 24 fused); the
others land within ~2% or worse. The staged pipeline alone streams at ~445 GB/s at T=4, the same
cold-cache ceiling the W8/Q4 small-T kernels see, but the full route lands at ~265 (plain) / 270
(fused) GB/s, so the consume is not overlapping the staging and the remaining 1.7x is not
reachable with this SIMT structure. The identified next candidates are a small-T tensor-core route
(8-column MMA tiles with the weights decoded to bf16 in shared, where the multiply leaves the
instruction stream) or the 5c.2 A8 activation route; both are larger pieces of work than this
port and neither is committed.

**Note on the audit's fused figure.** The audit's 360 GB/s for the fused route came from the
engine profile; measured back to back in one probe on the same real weights the direct fused
kernel reaches 256 GB/s and the staged one 270 GB/s, so the engine figure was not an isolated-op
measurement. The plain route's numbers do agree between the audit and the op bench.

### Q3 A8 prefill route (2026-09-27, session 9)

**Landed.** The 5c.2 port: the Q3G128_F16S prefill routes now quantize the activation per token
and 64-code group (`src/ops/common/rowsplit_a8_quantize.{h,cu}`, the fork's documented
contract) and multiply the weight codes with m16n8k32 s8 MMAs
(`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.{cuh,cu}`, adapted from the fork's `956b169`). The
plain `linear` and the folded gate/up `linear_swiglu` routes select it from 129 columns on when
the policy grants `AllowA8`; decode, MTP verification and padded-K problems keep A16. The Q3
execution leaves now pass `kQ3TextPolicy` (`AllowA8`) and the split attention/GDN pair wrappers,
`linear_add` and the workspace capacity queries thread the policy and the activation workspace.
Semantics and the oracle are documented in op-development.md 6.4; the two new suites
(`ninfer_linear_q3_a8_test`, `ninfer_linear_swiglu_q3_a8_test`) compare against an FP64 oracle
that applies exactly the documented quantization, and cover the 129-column boundary, one and
several token tiles and the padded-K fallback. Full `ctest`: 130 tests, 117 passed, 13 skipped,
0 failed.

**Op bench (plain route, 34816x5120, cold L2, median us; A16 = the tall route, A8 = the new
route).**

| T | A16 | A8 | change |
|---:|---:|---:|---:|
| 129 | 1233.9 | 649.3 | -47.4% |
| 256 | 1249.3 | 667.6 | -46.6% |
| 385 | 2278.4 | 1157.1 | -49.2% |
| 513 | 2744.3 | 1585.2 | -42.2% |

The T=385/513 rows use more token tiles than the 128-token tile needs, so a tail-aware split
remains the optional follow-up (the engine chunk widths are multiples of 128).

**Engine.** `pp32768` 1406.34 -> **2277.53 tok/s** (+61.9%); `pp100000 --prefill-chunk 2688`
1166.48 -> **1675.39 tok/s** (+43.6%; logs `profiles/bench/5c-a8/session9-pp{32768,100000}.log`).
Quick perplexity on the real artifact (`--corpus eval/corpora/perplexity-1m/manifest.json
--quick --kv-dtype int8`) 4.596525 -> **4.596095** over the same 261,167 tokens (report under
`profiles/perplexity/qwen3.8-27b/gsq3/int8-g64/ninfer-ppl-1m-v1/quick/`, log
`profiles/bench/5c-a8/session9-ppl-quick.log`), so the quantization does not move the quality
anchor. Decode is unchanged, as designed: the `tg512` run reports 53.18/53.89 tok/s at
acceptance 0.3991416309 over 233 rounds, against 53.87/53.81/53.68 on the same command before
the route (`profiles/bench/5c-a8/session9-tg512.log`), and T=4/6 dispatch to the staged A16
GEMV.

**A8 at decode widths: measured and rejected.** With the selection threshold temporarily at 1,
the tall A8 kernel on 34816x5120 runs T=1 377, T=2 376, T=4 381, T=6 379, T=8 378 us against the
staged A16 GEMV's 174/201/264/332/403 us, and the MTP verify width is T=4. The kernel is not
stream-bound: a cp.async variant that staged the 24-byte code windows with 8-byte copies three
steps deep measured 416-423 us over T<=8 (no gain), disabling the code transfer entirely left
175 us, disabling the activation staging changed nothing, and a single CTA still took ~57 us to
walk 80 steps. The transfers themselves move 66.8 MB of codes in ~240 us (~280 GB/s), and the
access pattern explains the gap: each step touches 128 different 128-byte lines (one per weight
row, 1920-byte row stride) and uses only 24 bytes of each, so the effective code traffic is
several times the plane; the staged GEMV avoids this by staging whole 48-byte groups with
16-byte copies at 16 warps/SM. A decode-side A8 route therefore needs a new small-T kernel
(whole-group staging, more CTAs per SM, K-split) rather than the tall tile, and none of that is
committed. The audit's ~1.6x projection holds only if the consume dominates; at these widths
the staging, not the consume, is the remaining wall.

### Q3 small-T bf16 MMA decode route (2026-09-27, session 10)

**Landed.** The audit's and the A8 measurement's identified next step: a dedicated small-T
tensor-core route, not a threshold change on an existing one.

- `src/ops/linear/q3/q3_rowsplit_small_t_mma.{cuh,cu}`: 32 weight rows x up to 8 token columns
  per CTA, K walked in 256-code (two-group) stages. Every 48-byte 128-code group is copied with
  16-byte `cp.async` vectors (whole-group staging); three staging buffers carry each stage's
  codes, activation and scales together and the loop issues two stages ahead; the eight warps
  split each stage into two 16-code slices, and their partials are reduced once per CTA. The
  decoded BF16 tile reuses the tall engine's `chunk ^ (row & 7)` swizzle for its ldmatrix A
  reads. `StageTokens` is 4 or 8: four activation rows need about 32 KiB per CTA and keep three
  CTAs per SM, eight rows need about 38 KiB and keep two.
- `q3_dispatch.cpp`: T=1 keeps the staged GEMV (~8% faster there), T=2..8 select the small-T
  route when N is a whole number of 32-row blocks, K is a multiple of 8, and the padded K is a
  whole number of 256-code stages; every other width is unchanged. `q3_linear_swiglu.cu`
  selects the fused gate/up variant from two columns on by the same facts.

**Semantics.** Each thread decodes one 12-byte quarter (32 codes) of one 128-code group into the
same `float(code) * float(scale)` -> single BF16 rounding the A16 tall routes use, and every
FP32 accumulator takes one m16n8k16 MMA per 16-wide K slice. The K-split partials are added once
at the end, so the accumulation order differs from the warp-per-row GEMV; the registered A16
criterion is the authority for both routes.

**Correctness evidence.** New `ninfer_linear_q3_a16_small_t_test`: route boundaries (T=1/2/8/9, a
padded *logical* tail that still fits whole 256-code stages, and a K whose padding stops at a
whole 128-code group, which keeps the GEMV), plus the shared FP64 oracle through the public
dispatch with the fixture's **Unit** scale pattern on `[1024,5120]` T=1..9, the
`[34816,5120]` gate_up shape at T=1/2/4/6/8, and the padded `[4096,4304]` shape at T=2/4/8.
The Unit pattern is the point of the suite: the default Q3 fixture's "Small" multipliers are
tiny powers of two whose `code * scale` products are all exactly representable in BF16, so a
route that rounds its decoded weights to BF16 passes the existing oracle without ever
exercising that rounding. The new linear and swiglu Unit-scale cases close that gap. All Q3
suites pass (`linear`, `swiglu`, `linear_add`, `gemv`, `tall`) and the full `ctest` is clean:
131 tests, 118 passed, 13 expected skips, 0 failed.

**Op bench** (`34816x5120`, cold L2, median us; CSV
`profiles/bench/5c-small-t-mma/q3_34816x5120.csv`; the small-T column is the same kernel the
dispatch selects from T=2):

| T | staged GEMV | small-T MMA | change |
|---:|---:|---:|---:|
| 1 | 169.0 | 184.3 | +9.1% (route not taken) |
| 2 | 194.6 | 186.4 | -4.2% |
| 3 | 225.3 | 186.4 | -17.3% |
| 4 | 257.0 | 188.4 | -26.7% |
| 5 | 288.8 | 180.2 | -37.6% |
| 6 | 323.6 | 180.2 | -44.3% |
| 7 | 359.4 | 181.2 | -49.6% |
| 8 | 393.2 | 181.3 | -53.9% |

The route comparison is `profiles/bench/5c-small-t-mma/session10-route-comparison.txt` (both
routes launched directly on the same real shape, medians of 30).

**Engine.** All figures on the pinned artifact, `rk4v4-e8`, MTP3 `--lm-head-draft`.

- `tg512` (the fixture the earlier sessions used): **53.67 -> 106.69 tok/s**; acceptance
  0.3991 -> 0.7873 over 152 rounds instead of 233 (log
  `session10-tg512.log`). The acceptance move is a trajectory change of that degenerate fixture
  (generation from BOS on the tiled corpus), not a quality claim; the plain `linear` and the
  fused swiglu routes are both oracle-valid, and the real workloads below do not move.
- CLI code scenario (`examples/cli/messages/scenario_code_python.json`, greedy, thinking off,
  64 tokens): **55.4 -> 71.9 tok/s** at 43.2% -> 43.8% acceptance (27 rounds both; run
  `session10-code-cli.{out,err}`).
- Depth sweep at `--prefill-chunk 1024` (tiled corpus, so acceptance is a fixture property):
  8K 124.9 tok/s, 32K 117.95, 98K 102.89 (the session-9 depth table recorded 66.6 / 64.7 / 59.4
  with the same corpus and flags). Prefill is unchanged (8192 2687.34, 32768 2272.84,
  98304 1610.39 tok/s).

**Where the remaining decode time goes (probe attribution, now deleted).** On `34816x5120`,
T=4: the code stream alone takes ~150 us (66.8 MB at ~460 GB/s), the decode adds ~25 us and the
MMAs ~10 us, against a ~100 us floor at the card's 664 GB/s read rate. The measured schedule
space that did not beat the landed one: direct B-fragment loads from L2 instead of an activation
tile (+19 us, the loads are uncoalesced 2-byte accesses), a five-deep code-only ring (the stream
stays at ~450 GB/s, so the limit is not memory-level parallelism), a two-CTA three-buffer ring
(-3 us over the landed design, within noise), and a two-row-block tile (the decode item count no
longer matches the CTA width). The identified next lever is the A8 profile for this kernel:
half-size decoded tiles (int8), a ~2.5x cheaper code decode, and m16n8k32 MMAs, at the cost of
the documented activation quantization and its separate launch; it is not committed.

Commands behind the numbers (`out/qwen3_8_27b_gsq3.ninfer`, 2026-09-27, RTX 4080):

```bash
./build/bench/ninfer_linear_bench --qtype Q3 --n 34816 --k 5120 --sweep 1:9:1   --warmup 10 --repeat 20 --csv-out profiles/bench/5c-small-t-mma/q3_34816x5120.csv
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer -n 512 --spec mtp   --draft-tokens 3 --lm-head-draft --max-ctx 8192 --kv-dtype rk4v4-e8 --warmup 1 -r 1
./build/apps/ninfer out/qwen3_8_27b_gsq3.ninfer \
  --messages examples/cli/messages/scenario_code_python.json --max-context 8192 --max-new 64 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer \
  --corpus profiles/bench/bench_corpus_131072.ids --kv-dtype rk4v4-e8 --max-ctx 102400 \
  -pg 8192,128\;32768,128\;98304,128 --spec mtp --draft-tokens 3 --lm-head-draft \
  --prefill-chunk 1024 --warmup 1 -r 1
```

The baseline figures in this section were re-measured on the same card and build flags with the
changes stashed (`git stash`), so they are not the session-9 logs.

### DFlash2 companion and the K=7 profile (2026-09-28, session 11)

**Landed.** The 5c.5 artifact half: the GSQ3 identity now carries the DFlash2 companion, and the
engine runs it on the 4080 with a reduced context.

- `tools/convert/qwen3_8_27b/inventory_gsq3.py` adds the shared 66-object `DFLASH2_TENSOR_SPECS`
  (21 `W8G32_F16S` matrices, 45 BF16 norms/convolutions/selector codebooks) exactly as the
  groupwise-int identity does; `recipe_gsq3.py` folds the shared companion recipes into its one
  recipe table; `convert_gsq3.py` takes `--dflash2-model` (config validation, base compatibility,
  source inventory of 66 recipes / 81 tensors, and the companion writer);
  `verify_gsq3.py` verifies the companion against its own checkpoint (all 45 BF16 objects
  word-for-word, the 21 W8 matrices on representative rows against an independent re-quantization).
- Converted and fully verified artifact `out/qwen3_8_27b_gsq3.ninfer`:
  **14,249,918,976 bytes, 1190 objects**, body unchanged (packed 323/323 base bytes equal,
  323/323 scales, 218 rounded, max error 2.98e-8), companion 45 direct + 21 quantized objects
  clean. Python suites: `tests/convert` + `tests/artifact` 89 passed, 5 skipped. Full `ctest`:
  131 tests, 118 passed, 13 expected skips, 0 failed.
- Engine: `ninfer_qwen3_8_27b_dflash2_real_test` passes at K=3/4/5/6/7/15 with B=1, CUDA graphs
  and the optimized selector, including its greedy-vs-ordinary identity check. B=8 needs
  6.1 GB of runtime reservation and does not fit this 16 GB card.

**The K=7 profile (what to use).** DFlash2 K=7 verifies 8 tokens, the widest window the session-10
small-T tensor-core route covers at full speed, and it is greedy-lossless on the real scenario
(256-token code generation md5-identical to MTP3: `54c59149db5f`).

| Workload | DFlash2 K=7 | MTP3 |
|---|---:|---:|
| CLI code scenario, 256 greedy tokens | 93.4 tok/s (35.7% acc) | 94.1 tok/s (71.4%) |
| depth 8K / 28K / 56K decode | 122.5 / 203.5 / 180.2 | 124.9 / 118.0 / ~112 |
| depth prefill | 2656 / 2309 / 1947 | 2687 / 2273 / 1610 |

The tiled-corpus depth rows accept nearly everything (a fixture property), so the wide window pays
off there; shallow greedy is at parity. Measured fit with the stock companion (RK4V4-E8):

- **57,344-token context without vision** (56K): 12.56 GiB weights, 1.80 GiB runtime reservation,
  636 MiB free after startup, 69 MiB planned slack. 61,440 fails by 70 MiB.
- **28,672 with vision** (28K): 12.8 GiB weights, 1.50 GiB runtime reservation, 525 MiB free,
  122 MiB planned slack. 30,720 fails by 16 MiB.

Launchers: `scripts/run-ninfer-4080-dflash2.{bat,sh}` (K=7, `NINFER_VISION=1` selects the 65,536
vision cap, `NINFER_CONTEXT` overrides); the MTP launcher keeps the 100K profile.

**The K=5 caveat (deferred, pre-existing).** Greedy DFlash2 K=5 on the code scenario diverges from
ordinary decoding (`2eb9e160cb96` against MTP3's `54c59149db5f`), and it reproduces with the Q3
small-T route forced off (`c539ce307cc3`), so it is a pre-existing DFlash2 wide-verify issue of the
5c.3 symptom class, not the session-10 kernel. K=3, 4, 6, 7 and 15 pass the real identity test; the
tg512 fixture's DFlash2 acceptance collapses at K=5/K=7 (13.9% / 21.0% against K=3's 66.9%) while
the real workloads do not. K=7 is the priority window and is unaffected; do not chase K=5 before it
matters.
**Session 15 update:** the divergence is the trajectory-sensitivity class described at the end of
§11, and the rotated-V double-rounding fix landed in session 15 changes this path's numerics. The
K=5 window remains deferred until it matters.

### DFlash2 Q4 companion (2026-09-28, session 12)

**Landed (option 4's matrix half).** The gsq3 identity requantizes the 21 companion matrices from
the stock `W8G32_F16S` grid to `Q4G64_F16S` (4.25 bpw); the dynamic-conv/norm vectors and the two
selector codebooks stay BF16. The artifact is **13,330,776,576 bytes over 1190 objects** (was
14,249,918,976), the predicted -0.856 GiB. The body re-verifies byte-for-byte (323/323 packed
objects, 323/323 scales equal, 218 rounded, max error 2.98e-8); the companion verifies 45 BF16
objects word-for-word plus 21 independent re-quantizations over 189 sampled groups. The shared
`dflash2_inventory.py` and the nvfp4 identity keep the W8 assignment.

- Converter: `inventory_gsq3.py` overrides the five matrix roles (feature projection, QKV,
  attention output, gate/up, down); `verify_gsq3.py` accepts the spec format and re-quantizes
  each matrix independently.
- Runtime: `bind_dflash2` picks the format from the weights profile; `q4_dispatch` gains
  `[5120,4096]`, `[5120,17408]` and `[5120,25600]` plus a K-generic T=1 GEMV; the three-output
  `attn_input_proj` projects the query/key/value row views through three qualified Q4 linears;
  `linear_dynamic_grouped_conv_add` shares its activation-free finish kernel and adds a Q4
  projection; `context_kv_materialize` gains a composed Q4 route (qualified Q4 linear plus two
  store kernels that own the norm/RoPE and the BF16->FP16 boundary). `linear_swiglu` already
  served `[34816,5120]` in Q4; the scratch planner now covers both companion formats.
- Verification: full `ctest` **131 of 131 pass**; new Q4 coverage for the three linear problems,
  the QKV split (T=1..128 eager and graph), the conv-add (W=2..16, B=1..8) and the composed
  context route (Tiny-scale Q4 weights, both formats in one binary). Real engine test: K=7 and
  K=15 at B=1 with CUDA graphs and the optimized selector, eager K=7 and `int8` KV all keep the
  ordinary-decoding identity. B=8 still does not fit (6.17 GiB runtime reservation); the full
  proposal head remains unsupported for the Q4 output head (pre-existing; the profile uses the
  optimized selector).
- Fit (RK4V4-E8, `--host-kv-mib 4096`, K=7, B=1): **100,000 text-only** starts with 11.7 GiB
  weights, `runtime_reservation_bytes=2,879,649,280`, 625 MiB `available_after_startup` and
  78 MiB planned slack, matching the stock companion's 57,344-token margin; **65,536 with vision**
  leaves 655 MiB free and 108 MiB slack. Hard limits: 104,448 text (106,496 fails by 18.5 MB) and
  71,680 with vision (73,728 fails by 17 MB).
- Draft quality: the K=7 code scenario (256 greedy tokens) runs **100.9 tok/s at 38.2% acceptance**
  against the stock 93.4 tok/s / 35.7%; the depth sweep is **124.2/205.5/182.7 tok/s at
  8K/28K/56K** against the stock 122.5/203.5/180.2, with prefill 2656/2308/1949 tok/s.

**Remaining (option 4's selector half).** The two BF16 selector codebooks (0.254 GB) hold the last
~0.12 GiB. No 8-bit codebook format is registered; `FP8_E4M3FN_ROW_BF16S` is the only row-scaled
8-bit candidate (-0.117 GiB, about +5.5K tokens) and needs a codebook decode path in
`candidate_selector_path` plus converter/verifier/op work. It is not required for the 100K text
profile and is deferred.

### Remaining-performance audit (2026-09-28, session 13)

Fresh Nsight Systems captures on the session-12 artifact (`out/qwen3_8_27b_gsq3.ninfer`,
13,330,776,576 B) with `--cuda-graph-trace=node`, exported as sqlite under
`profiles/nsys/audit-20260928/` (`mtp3`, `dflash7`, `pp32768`, `pp100000`). Kernel time is
CUPTI SM time; per-round figures divide by the Q3 launch count (256 plain + 64 SwiGLU/round).

**Decode rounds (tg512 fixture, `rk4v4-e8`).**

| Path | MTP3 T=4 round | DFlash2 K=7 T=8 round |
|---|---:|---:|
| Q3 plain small-T (256/round) | 14.95 ms | 16.76 ms |
| Q3 fused SwiGLU small-T (64/round) | 9.92 ms | 10.77 ms |
| Q4 DFlash2 companion (37 linears + SwiGLU + draft head) | - | 5.59 ms |
| W8 MTP layer | 2.23 ms | - |
| Q4 main head (1/round) + Q4 draft head (3/round) | 2.79 ms | - |
| GDN record/fold/conv/gating | ~1.3 ms | ~2.0 ms |
| decode attention (8K) | 0.27 ms | 0.35 ms |
| norms, adds, rope, casts | ~0.7 ms | ~0.5 ms |
| **total** | **~32.2 ms** | **~36.3 ms** |

The Q3 body is 77% (MTP3) / 76% (DFlash2 K=7) of the round and streams 9.5 GB at ~380 GB/s,
against the 664 GB/s read rate the Q4 draft head reaches on the same card. T=2..8 is ~187 us on
34816x5120 (T-flat); T=1 is 173 us; **T=9..16 falls back to the 32x64 staged tile at 407-438 us**
(2.2-2.3x), so DFlash2 K>7 and any wide n-gram window currently pay that route.

**Prefill passes (`--prefill-chunk` default / 2688).**

| Path | pp32768 | pp100000 chunk 2688 |
|---|---:|---:|
| Q3 A8 tall GEMM | 10.13 s (70.6%) | 29.29 s (49.0%) |
| prompt INT8 attention | 3.02 s (21.0%) | 25.17 s (42.1%) |
| GDN (state_passing, wy_wu, output, conv, gating) | ~0.7 s | ~2.5 s |
| A8 activation quantize | 0.13 s | 1.02 s |
| norms, adds, casts | ~0.2 s | ~1.1 s |

**The A8 tall GEMM is not DRAM-bound.** q3_rowsplit_tall_a8_kernel runs ~1.1 us per 64-code
step per CTA (T=129: 648 us over 7.2 waves, 80 steps; T=513: 1370 us), i.e. ~3 GB/s per CTA,
220 GB/s aggregate. A temporary ablation that kept only the three `ld.global.nc` code words, the
scale load and one smem store (decode, MMA, FP32 update, activation staging, barriers removed)
still measured 575 us at T=129 / 938 us at T=513: the load pattern itself is the wall. A
three-slot register pipeline (two steps of code loads in flight) made it slower (690 us), so the
throttle is not simple load latency - the per-row chunk is only 24 B at a 1920 B stride, versus
the 96 B whole-group stages that already reach ~450 GB/s in the small-T decode kernel and the
128 B+ rows of the Q4 draft head at 664 GB/s. The prototype was reverted; no repo change.

**Ranked candidates.**

1. Decode: A8 profile of the small-T kernel (half-size int8 decoded tile, m16n8k32), the lever
   session 10 named; upper bound is the ~450 GB/s staging ceiling unless the stage width grows
   with it (e.g. four-group 192 B stages).
2. Decode: small-T T=9..16 route - removes the 2.2x cliff that makes DFlash2 K>7 and the
   previously disabled wide n-gram verify window expensive (both landed, sessions 16 and 19).
3. Decode: ~~finish the 5c.3 corruption root cause~~ **DONE (session 15): the flips are
   trajectory sensitivity, not a defect — see the session-15 section at the end of §11.** The
   n-gram wide window was a maintainer call and is re-enabled in session 19; DFlash2 K=5 stays open.
4. ~~Prefill: restage the A8 tall code path with whole-group cp.async (48-192 B per row) instead of
   the 24 B register loads.~~ **Withdrawn (session 20): the staging structure is not the wall.**
5. ~~Prefill: prompt attention (42% of 100K); the 5c.4 port is unlanded.~~ **DONE (session 26,
   2026-10-03): the 5c.4 worker V-dequant schedule landed; append bench int8 -32..-35% at
   8K-128K, `pp100000 --prefill-chunk 2688` 1710 -> 1971 tok/s and `pp32768` 2314 -> 2470 tok/s
   (same-session A/B). §11 has the record. This was the last untouched prefill kernel.**
6. Optional: the documented tail-aware split for the 129-513-column A8 widths (last tile is
   nearly empty) and the GDN chunked ops (~4% at 100K).

### Wide-verify root-cause update (2026-09-28, session 13)

The session-7 blocker ("the configured wide window produced a wrong correction logit in one
column; the suspect area is the assembled multi-column verify state") is **re-framed**: the
corruption is not wide-window-specific, and the small-T INT8-family kernel's multi-split path is
the culprit. Evidence, all on `out/qwen3_8_27b_gsq3.ninfer` with the temporary debug harness and
dumps removed again (no repo change):

- **Wide rounds are clean on BF16.** With `verify_window = ngram.max_drafts` restored in the
  planner, `ninfer_qwen3_6_27b_ngram_real_test` (`NINFER_NGRAM_MAX_DRAFTS=6`, graphs off, BF16 KV)
  ran **76 wide rounds with 0 n-gram-added divergences over MTP** and passed. The exact session-7
  failing chain `[5653 1870 1137 5480 2923 16 23]` verifies correctly in isolation too.
- **The corruption reproduces in narrow rounds, RK4V4-E8 only.** The committed table prompt (raw
  text, the ngram test's `prompts()[3]`) with MTP (no n-gram) and `rk4v4-e8` diverges from the
  ordinary route at generated token 314 (plain 24 vs MTP 23) at frontier 470, column 2, position
  472: verify ids `[5480 2923 16 24]`, argmax `[2923 16 23 63]`, and the failing column's top
  logits are `23=20.125, 22=19.0, 21=18.625, 24=17.875` while the scoring route has 24 at -0.003
  (a wrong ordering of ~9 nats). MTP depth K=1/2/3 (widths 2/3/4) all diverge at the same token;
  K=4/5 (widths 5/6) do not reach the state. BF16, int8 and `rk4v4` (non-E8) are clean;
  `rk8v4` and `fp8` have their own flips at other states.
- **The ordinary route is correct with the same cache**, so the data is representable: only the
  verify kernel's decode of it is wrong. A per-layer hidden dump of the failing verify round
  versus the ordinary route at the same token/position shows the hidden identical through layers
  0-2 (GDN), a first small difference at layer 3 (first full-attention layer, rel-L2 0.0068 for
  both BF16 and E8), then a large E8-only jump at layers 7-9 (rel-L2 0.49 against 0.009 for BF16).
- **Forcing one split removes the corruption** (`causal_small_t_split_count` returning 1, host and
  device agree); forcing 2 or 4 splits keeps it. Forcing the prompt route for E8 widths <= 8 also
  removes it. Both experiments were reverted.
- What was already ruled out still stands (attention/GDN/linear op oracles, graphs, OOB, smem
  races). The remaining suspects are inside the small-T partial kernels' split partitioning and
  their interaction with the fused append: each split stages keys from `first_tile` (key-tile
  aligned) but only masks/scores its own `[split_start, split_end)` range, and only the owning
  split writes a current token's row. The reduce kernel recomputes the same active-split count, so
  a reducer mismatch is unlikely but not excluded.

Next step to finish the explanation: compare the partial buffers (or the per-split QK scores) of
the failing round for the split counts 1 and 8, for the one corrupted column; the E8 K-decode
`koff` (page-relative `(key & mask) * 64 + d/4`) and the append's `k_base` are the two places to
check against the split ranges first. Until then the wide window stays disabled and `rk4v4-e8`
decode stays as validated by the session-4 profile - the corruption is a close-call flipper, not
a systematic text breaker, but it must be fixed before the n-gram chain or DFlash2 K=5 are
trusted on this KV format.

**Superseded by session 15:** the corruption is trajectory sensitivity, not a split/append/codec
defect; see "Wide-verify root cause: trajectory sensitivity" at the end of §11. The wide window
was re-enabled in session 19 under the existing bf16 tie criterion.

### RK4V4E8 executed-path correction (2026-09-28, session 14)

The session-13 "next step" named code that RK4V4E8 never executes. `RK4V4E8` is the **E8
lattice + packed-int4** mode, not the cylinder/root mode:

- `kv_fork_mode_flags(RK4V4E8)` (`src/core/paged_kv_storage.h:147`) returns
  `{packed_v, rotate_k, rotate_v, packed_k, e8_lattice}`. The dispatch in
  `causal_attention_small_t_launch_for` therefore instantiates the kernel with `PackedK=true,
  E8Lattice=true, E8Root=false`; the `E8Root` branch (its `koff = ... + (key & mask) * 64 +
  d/4` K reads and the append's `k_base + s * 2` writes) belongs to `RK2V4E8` only.
- The RK4V4E8 plane geometry is `{U8, 128, FP16, 4}` for both planes
  (`paged_kv_storage.h:102`): 128 bytes of packed int4 per key. The kernel's K/V staging is
  `kv_cache_unpack_i4x16(kv_cache_i4_code_index(...))` - **byte-for-byte the same decode as
  `RotatedInt4KeyInt4ValueGroup64` (rk4v4)**, which session 13 measured clean, and the same
  staging as the `PackedK` branch of the standalone append.
- The only RK4V4E8-specific code is the encoder `e8_project_8d_warp` (nearest E8 point before
  the int4 rounding), called from the fused append (`small_t_i8.cuh`) and the standalone
  append (`kv_cache/append/kernel.cuh`). The prompt route uses it too.

**The projection is validated.** A throwaway device probe (2M random 64-dim groups per range,
64M comparisons) compared the warp-cooperative encode against both the scalar
`e8_project_8d_fast` and an independent nearest-E8 search (D8 u (D8 + 1/2)):

| input range | warp vs scalar code mismatches | scalar vs independent mismatches |
|---|---:|---:|
| [-1, 1] | 6 / 64M (exact ties, both points equidistant) | 0 / 64M |
| [-2.5, 2.5] | 0 / 64M | 0 / 64M |
| [-7, 7] (production scale) | 0 / 64M | 0 / 64M |

So the E8 encode is not the defect, and neither is the shared decode.

**Where that leaves the root cause.** The strongest surviving evidence is the split-count
dependence: forcing `causal_small_t_split_count` to 1 removes the divergence while 2 and 4 keep
it, and the code as read makes the partitions numerically equivalent to within the FP32 merge
order. A material difference (the failing column's ~2.25-nat swing) therefore implies a
partition-dependent data path or a race, not rounding. The session-13 per-layer dumps should also
be re-derived before further interpretation: in those files the E8 and BF16 verify taps are
bit-identical at layers 3, 6 and 7, which no KV-format-dependent full-attention output should
be, so regenerate them with the tap gate before trusting the layer localization.

Decisive next experiment: replay the failing attention call with real data. Dump the input Q and
the E8 K/V/scale bytes for the failing round (layer 3 is the first full-attention layer; frontier
473, the four columns) from the MTP route, then run the small-T op on that snapshot at split
counts 1/2/4/8 and compare each against an FP64 oracle computed from an independent decode of
the same cache bytes. That separates (a) partition-dependent cache/append bytes, (b) a decode
bug, and (c) tie-level numerics. Before bisecting further, land the missing committed coverage:
the fork's `8b202c4` adds an RK4V4E8 host codec + FP64 oracle + fused/standalone append byte
parity to `tests/ops/softmax_attention/causal_cache.cpp`, and our file's existing
`{7, 467, 512}` / `{8, 467, 512}` cases are exactly the failing width/depth class; port it and
add those cases. The stock repro entry point remains `ninfer_qwen3_6_27b_ngram_real_test` with
`NINFER_NGRAM_KV_DTYPE=rk4v4-e8` (prompt 3, token 314).

### Quantized-KV control measurement (2026-09-28, session 14)

The session-13 "E8-only" localization does not survive the stock test at default settings (wide
window disabled, `ninfer_qwen3_6_27b_ngram_real_test`, 6 prompts, 384 new tokens):

| KV dtype | MTP divergences | largest scoring-route gap |
|---|---:|---:|
| bf16 | 3 / 1347 compared tokens | 0.5 nats |
| int8 | 5 / 1156 | **20.97 nats** (prompt 3) |
| rk4v4-e8 | 5 / 1014 | **6.75 nats** (prompt 3) |

The bf16 flips are tie-scale (0.125-0.5 nats). The int8 gross case has the *plain* route's token
rated at -20.97 nats against the verify's -0.0004, i.e. the decode-side route disagrees with the
prefill scoring route by ~21 nats on a cache with no E8 projection, no rotation and no packed
nibbles. So the defect is shared with the simpler `Int8Group64` small-T path and E8 is not
required to reproduce it; the earlier "int8 clean" reading came from the temporary harness
(wide-window enabled, different trajectory) and is withdrawn. Logs: `/tmp/s14_ngram_{bf16,int8,e8}.log`
(same commands as the session-14 entry point).

The simpler codec is the better repro. `Int8Group64` has no E8 projection, no rotation and no
packed nibbles, so replaying the small-T vs prompt attention output on one cache isolates the
split-K path with far less noise than the E8 route. `ninfer_softmax_attention_test` already
covers `Int8Group64` against an FP64 oracle, so the failing configurations can be added there
directly.

```bash
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_gsq3.ninfer NINFER_NGRAM_KV_DTYPE=bf16 \
  ./build/tests/ninfer_qwen3_6_27b_ngram_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_gsq3.ninfer NINFER_NGRAM_KV_DTYPE=int8 \
  ./build/tests/ninfer_qwen3_6_27b_ngram_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_gsq3.ninfer NINFER_NGRAM_KV_DTYPE=rk4v4-e8 \
  ./build/tests/ninfer_qwen3_6_27b_ngram_real_test
```

### Wide-verify root cause: trajectory sensitivity, not a kernel defect (2026-09-29, session 15)

**The session-13/14 "column corruption" is explained.** It is not a defect in the small-T split
path, the fused append, the E8 codec or the wide verify window: greedy decoding under a lossy
KV cache is chaotically sensitive to bf16-ulp-level state differences, and the flips are the
amplification of those differences, not a corrupted memory or logit. The evidence, all on
`out/qwen3_8_27b_gsq3.ninfer`:

- **A reproducible single-prompt int8 case.** `ninfer_qwen3_6_27b_ngram_real_test` with
  `NINFER_NGRAM_ONLY`-style single-prompt reduction (prompt 3, 286 new greedy tokens) picks
  4277 at generated token 285 while the same engine's *scoring* route rates 4277 at -20.97 nats
  and 5480 at -0.0004. A fresh prefill of the identical 444-token prefix then decode gives 5480,
  and bf16 incremental decode also gives 5480.
- **Position bisect.** Feeding the correct prefix up to position k and decoding the rest
  incrementally: k = 13/14/15 corrupt (4277 at position 443), k = 16 clean (5480 at 26.75,
  4277 at 9.56). The runs differ only in whether position 174 was computed by a T=1 decode or
  by the 175-token prefill chunk. At position 174 the two routes agree to bf16 ulp: logits
  max-diff 0.31, normed-hidden cosine 0.9995, same argmax 5787. That single ulp-level difference
  becomes an 8-14 nat logit difference at 443 **in bf16 too** (5480: 28.12 vs 19.88, max logit
  diff 10.6), only bf16's flip does not cross the decision boundary while int8's does.
- **The same-position KV caches differ only from the appended position onward.** All 32
  K/V plane byte differences between the decode and prefill runs start at byte offset 142848 =
  page 2, offset 46, dim 0 — exactly the token appended at position 174. Nothing before it differs
  (prefixes are bit-identical), and the difference is the rounding of that token's inputs.
- **Op-level qualification passes at the failing class.** `ninfer_softmax_attention_test`'s
  int8 `{7, 467, 512}`/`{8, 467, 512}` cases (a1/a3, fused append and cache read) pass against the
  FP64 oracle, including new width-1/2/4 cases at windows 441-445 added for this investigation
  (removed again). Forcing the prompt route for every decode step (`causal_attention_resolve_route`
  override, also removed) left the int8 reference output bit-identical over 384 tokens, and forcing
  the split capacity to 1 moved the gross flips elsewhere rather than removing the class.

**Consequence.** The wide n-gram verify window does not corrupt anything by itself; it perturbs
the trajectory like any other width/partition change. The 5c.3 wide window can be re-enabled as a
maintainer decision: the bf16 tie criterion remains the contracting test, and quantized-KV
divergences should stay report-only. The same explanation covers DFlash2 K=5 and the E8 gross gap.

### Rotated-V double rounding fix and the RK4V4E8 oracle port (2026-09-29, session 15)

The plan's named next step: port the fork's `8b202c4` RK4V4E8 host codec + FP64 oracle + fused/
standalone append byte parity into `tests/ops/softmax_attention/causal_cache.cpp`, and the fix its
own 46 failures demand (`787766f`). Landed:

- **Test coverage** (the missing committed coverage): `test_cache_layout`, `HostCache`, the H64
  butterfly encoder, nearest-E8 projection, `make_cache`/`append_cache`/`cache_value`, the rotated
  ideal-attention (H64 on Q and on the output), `DeviceCache` upload/snapshot/verify with
  byte-exact K and V parity, `verify_cache`, `cache_name`, `attention_criterion`, the a1 fused/
  standalone append parity, RK4V4E8 in the shared geometry, batch and DFlash2 sweeps, a new
  `run_rk4v4e8_cases()` (T=1/T=13 at 8K and 128K), a new `--rk4v4-e8-only` entry and the
  `ninfer_softmax_attention_rk4v4_e8_test` ctest case.
- **Fix**: rotated-V caches (rk4v4, rk4v4-e8, rk2v4-e8) wrote the attention output in the H64
  domain in bf16 and a second kernel rotated it back, adding a second bf16 rounding. The small-T
  reduce and the prompt i8 epilogue now apply the inverse H64 to the FP32 result before the only
  BF16 rounding; the separate `kv_cache_inverse_rotate_output_kernel` launch is removed.
- **Evidence**: with the port and without the fix the new suite fails the reduction criterion by
  about one bf16 ulp across many cases (e.g. actual -1.02344 vs reference -1.01909); with the fix
  it passes. Full `ctest`: **132 tests, 119 passed, 13 expected skips, 0 failed** (was 131/118/13/0).
  Real routes: the n-gram real test on `rk4v4-e8` passes (4 divergences over 1263 tokens, one
  7-nat gross flip on prompt 2 — consistent with the trajectory-sensitivity explanation above),
  and the DFlash2 K=3/7/15 B=1 integration test passes unchanged (that binary uses its default
  bf16 KV, so it is a regression check, not E8 coverage).
- **Impact**: the E8 decode output is now closer to the FP64 oracle by ~1 bf16 ulp, and one kernel
  launch per attention call is removed. The suite's codec-quality report for this artifact's
  shape is rel_rmse 0.102 (the E8 store itself). The `ninfer-perplexity` weight-quality anchor
  is unaffected (int8 prompt attention).

### Small-T 9..16-column route and the A8 small-T decode engine (2026-09-30, session 16)

Both decode items the session-13 audit ranked first and second are landed. No artifact changed and
nothing is committed.

**1. The small-T tensor-core route now covers 9..16 columns (ranked candidate 2).**
`q3_rowsplit_small_t_mma.{cuh,cu}` launches one CTA per (row block, 8-column token tile), token
tile fastest in the grid, and the `Problem::emit` writes the tile's token offset. The dispatch
sends T=2..16 to the small-T engine and keeps the A16 staged 32x64 tile for 17..127 (the tall
engine takes over at 64 columns for whole 128-row blocks). The folded SwiGLU route covers 2..16
the same way, and `q3_linear_swiglu_workspace_capacity_bytes` now reserves the chunked FP32 plane
only for 17..63. `test_q3_a16_small_t.cpp` pins the new boundaries and runs the FP64 oracle across
T=1..16 on the registered parents; `test_q3_a16_tall.cpp` keeps its staged byte-equality check for
the widths the staged/tall engines own (T>16).

- Op bench `34816x5120` (cold L2, median us; baseline = the staged 32x64 route at the start of the
  session): T=9..14 437 -> 328, T=15/16 326-328 (CSV
  `profiles/bench/5c-decode-levers/tiles_a16_34816x5120.csv`). The residual 1.75x against the
  T<=8 rate (187 us) is the second tile's duplicated code staging; a native 16-column CTA would
  need a half-size decoded tile or a two-buffer ring to keep two CTAs resident per SM.
- Engine (CLI 256 sampled tokens, `examples/cli/messages/scenario_code_python.json`): MTP3 69.7
  and DFlash2 K=7 99.8 unchanged (their verify widths are 4 and 8), DFlash2 K=15 **39.6 -> 53.0**
  tok/s with the round at 60 ms against 84 ms. Logs
  `profiles/bench/5c-decode-levers/tiles_cli_*.log`.

**2. The A8 profile of the small-T engine (ranked candidate 1).** New
`src/ops/linear/q3/q3_rowsplit_small_t_a8_mma.{cuh,cu}` and the shared
`q3_rowsplit_a8_codec.cuh` (the tall A8 engine now takes its decode helpers from there). One CTA
owns 32 weight rows and one 8-column token tile; a stage is **512 codes** (four 128-code weight
groups = eight 64-code activation groups), so each of the eight warps owns exactly one activation
group per stage and everything a group needs — its int8 decoded A tile step and its activation B
step — comes from one 64-byte step of the stage. Whole-stage codes (192 B/row), weight scales, the
quantized activation and its group scales are staged with cp.async into a three-buffer ring; every
thread decodes two 12-byte quarters into an int8 tile with the tall engine's 64-byte swizzle;
m16n8k32 s8 MMAs seed each group sum with the integer magic and recover the exact int32 `d_g` with
one FADD; the per-group FP32 product and fma plus the A16 engine's eight-way partial reduction
produce the output. `q3_uses_a8` admits 2..16 columns for exact-K shapes whose K is a whole number
of 512-code stages; the tall engine keeps 129+, and the single-token decode, the 17..128 window and
padded-K shapes keep A16. The workspace queries, `q3_linear_swiglu`, `linear_add` and the input-
projection wrappers follow the policy.

**The bug that cost the session: the activation scale plane is group-major.**
`a8_g64_quantize` writes scale `[g * tokens + t]` and the tall A8 engine reads exactly that layout;
the small-T engine first read it token-major. The visible symptom was that even 64-code groups
matched the tall route and odd groups vanished. The fix keeps the group-major order in the smem
tile, which leaves every group's run starting at an arbitrary float offset, so the scales are
staged with **4-byte** cp.async per (group, token): a 16-byte copy from a misaligned address is
dropped silently, which zeroed every scale except group 0's (offset 0).

- `ninfer_linear_q3_a8_small_t_test` passes the documented-quantization FP64 oracle on all six
  registered parents at T=2..16 plus the padded-K, 256-code-stage and A16Only fallbacks; the other
  eight Q3 suites are unchanged.
- Op bench `34816x5120` (cold L2): T=2..8 A16 185-189 -> A8 157-159, T=9..16 A16 327 -> A8 278
  (-15..-17%, the separate activation quantization included; CSV
  `profiles/bench/5c-decode-levers/a8small_34816x5120.csv`).
- Engine with `--greedy` (identical rounds and acceptance): MTP3 91.6 -> **101.0** tok/s, DFlash2
  K=7 104.9 -> **118.2**, K=15 60.4 -> **67.1** (+10..13%). The greedy texts are byte-identical for
  K=7 and K=15; MTP3 drifts by one phrase at a near-tie (`"understand what's already there"`
  against `"understand what I'm working with"`), the documented-quantization class the decode A8
  profile introduces. Logs `profiles/bench/5c-decode-levers/{a16,a8small}_greedy_*.log`.
- This is the only remaining maintainer call from this session: the decode A8 profile changes
  decode numerics by design, so if the MTP3 near-tie drift is unwanted the `q3_uses_a8` small-T
  branch can be limited to prefill widths with a one-line change.
- Full `ctest` after both items: **133 tests, 120 passed, 13 expected skips, 0 failed** (session 15
  was 132/119/13/0; the new `ninfer_linear_q3_a8_small_t_test` is the addition). `git diff --check`
  is clean.

### Serving-launcher rebuild tracking and the decode re-measurement (2026-10-01, session 17)

**Launchers rebuild on checkout change.** All four 4080 launchers
(`scripts/run-ninfer-4080{,-dflash2}.{sh,bat}`) stamp every image build with
`--label org.ninfer.revision=<git rev-parse --short=12 HEAD>` (plus a `-dirty` suffix when the
worktree has changes) and rebuild when the image is missing or its label differs from the current
revision. Previously they built only on the first run, so a pulled decode improvement kept being
served by the old image; now re-running a launcher after a pull rebuilds it. `NINFER_IMAGE` still
selects the image; headers carry the current route, the measurement summary, and the DFlash2
draft-window correction (the small-T tensor cores cover K=1..15, widths 2..16, not just width 8).

**Depth re-measurement on the session-16 build** (`out/qwen3_8_27b_gsq3.ninfer`,
13,330,776,576 B; tiled corpus `profiles/bench/bench_corpus_131072.ids`, `rk4v4-e8`,
`--prefill-chunk 1024`, one repetition per point; logs
`profiles/bench/5c-decode-levers/session17_depth_{mtp3,dflash2_k7,dflash2_k15}.log`):

| Point | MTP3 decode | DFlash2 K=7 decode | MTP3 prefill | DFlash2 K=7 prefill |
|---|---:|---:|---:|---:|
| 8K | 130.16 | 136.02 | 2693.62 | 2672.12 |
| 28K / 32K | 126.19 (32K) | 232.13 (28K) | 2297.57 (32K) | 2333.56 (28K) |
| 56K / 98K | 109.36 (98K) | 204.94 (56K) | 1651.00 (98K) | 1986.37 (56K) |

The same points measured 125/118/103 (session 10, MTP3) and 124/205/183 (session 12, DFlash2
K=7), so the A8 small-T decode profile is visible in the product route, not only in the op bench.
The decode A8 profile stays enabled (no `q3_uses_a8` change): the maintainer's direction was to
reap the decode improvement, and the launcher headers state it. The session-16 MTP3 near-tie drift
is the accepted consequence of that choice.

**K=7 stays the DFlash2 window.** The route widening made depth a free choice again (session 11
picked K=7 because widths above 8 left the fast small-T route), so the K=15 point was re-measured
on the same three depths (log `profiles/bench/5c-decode-levers/session17_depth_dflash2_k15.log`):
79.42/197.21/153.66 tok/s at 8K/28K/56K with 23.3/76.0/61.2% acceptance, against K=7's
136.02/232.13/204.94 at 49.0/100/95.7%. K=15 loses at every point even where acceptance is
high. The structure makes that hard to escape: the Q3 small-T cost jumps at T=9 (the second
8-column tile) and is flat through T=16 (A8 op bench 157-159 us at T=2..8 against 278 us at
T=9..16), so K=8..14 pay the wide-window price with the second tile partly unused and the only
real candidates are K=7 and K=15; and K=15's extra accepted tokens do not cover the ~1.7x round
cost (at 28K it takes 11.6 tokens/round against K=7's 8.0; at 8K both need 29 rounds and K=15 is
simply 1.7x slower). K=7 remains selected; widen only if a workload's acceptance holds far past
position 8, which neither the code scenario nor the tiled corpus does.

### MBPP quality: the RCO weight-allocation confound (2026-10-01, session 18)

**Observation (maintainer).** MBPP pass rate 88% with beellama on
`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` at Q8_0 KV, against 84% with this artifact's DFlash2 K=7
profile (`rk4v4-e8`).

**The two systems do not carry the same weights.** The baseline GGUF is the RCO release
`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S: 3.50 bpw whole-file, a per-tensor RCO allocation
over GGUF types (IQ3_S 144, IQ4_XS 96, IQ3_XXS 78, Q4_K 39, IQ2_S 17, Q2_K 13, IQ2_XS 9, Q6_K 8,
IQ2_XXS 5, IQ1_M 1; `output.weight` Q4_K, `token_embd` IQ2_S), quantized with the shipped
importance matrix and published as task-lossless (LiveCodeBench v6 85.71 = BF16). This artifact
is a verbatim repack of the *different* uniform checkpoint `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ`
(3-bit G128 body, Q4G64 endpoints) — GSQ without the RCO per-tensor allocation. So a code-
benchmark delta between the two systems cannot be attributed to KV alone. The active differences,
cheapest first:

1. **Sampling.** The plan's beellama baseline ran `temperature 0`. The DFlash2 and MTP launchers
   serve `--temperature 1 --top-k 20 --top-p 0.95 --presence-penalty 0` as server defaults, and
   `resolve_sampling_overrides` lets only an explicit request override them; a harness that omits
   sampling gets temp-1 sampling from NInfer and whatever beellama's default is. Check the
   request `temperature` (the `--request-log-jsonl` records it) and pin greedy for both engines.
2. **KV storage.** Q8_0 (8-bit block-32, fp16 scale) vs `rk4v4-e8` (Hadamard-rotated 4-bit E8 K +
   4-bit V, 64-group scales); INT8-G64 is the local analogue of Q8_0. Session 14's control and
   session 15's trajectory-sensitivity closure show quantized KV flips whole greedy generations at
   near-ties, and `int8` is not clean either (20.97-nat gross flip against `rk4v4-e8`'s 6.75 and
   bf16's 0.5); only `bf16` is lossless. Task-level MBPP effect is unmeasured.
3. **Decode A8.** Session 16's int8 activation profile is enabled at decode widths 2..16, so it
   runs inside DFlash2 K=7 (W=8) and MTP3 (W=4). It moves PPL by -0.0004 and one DFlash2 K=7
   fixture stayed byte-identical, but it is the newest task-level-unvalidated numeric change.
4. **Spec backend.** DFlash2 K=7 is greedy-lossless against the engine's own routes; a `--spec
   mtp` vs `--spec dflash2` MBPP delta at fixed KV and greedy would be a verify defect.

**Isolation set** (same artifact and harness, one variable per run, greedy; `bf16`/`int8` fit
MBPP-scale contexts on the 16 GiB card):

```bash
# KV: bf16 (lossless) / int8 (Q8_0 analogue) / rk8v4 (8-bit K, 4-bit V) at the same spec
ninfer-serve out/qwen3_8_27b_gsq3.ninfer --max-context 16384 --kv-capacity 16384 \
  --kv-dtype bf16 --spec dflash2 --draft-tokens 7 --lm-head-draft --greedy --host-kv-mib 4096
# Backend: same KV, MTP3
ninfer-serve out/qwen3_8_27b_gsq3.ninfer --max-context 100000 --kv-capacity 100000 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --greedy --host-kv-mib 4096
# Decode A8 off: q3_uses_a8's small-T branch -> return false, rebuild, rerun the same command
```

Calibrate the weight hypothesis inside beellama before any artifact work: rerun the same MBPP
harness on the same repo's IQ3_XXS (3.00 bpw) and IQ2_S (2.75 bpw) GGUFs at Q8_0 KV. If they
fall to ~84, this artifact's uniform 3.125-bpw result is a weight-budget result, not a KV one.

**RCO cannot be added to inference; it is a quantization-time search.** The released pipeline
(IST-DASLab/RCO, IST-DASLab/GSQ) quantizes every tensor at every candidate type with GSQ into a
per-tensor database, runs the budget-constrained Riemannian search on task loss, then assembles
the chosen mix. Inference only consumes stored formats. A mixed-allocation `.ninfer` variant
would therefore need: the full BF16 source (51.75 GiB), the GSQ/RCO code and calibration data, a
search over *NInfer's* registered formats (Q3G128/Q4G64/Q5/Q6/W8 - GGUF I-quants/K-quants and
their imatrix scales cannot be reproduced here), a new recipe/identity, and its own PPL+MBPP
gate; the result would not byte-match the GGUF or inherit its 88%. The interesting variant is
equal-budget (same artifact size, better allocation), which keeps the 16 GB fit; matching IQ3_S's
3.50 bpw costs roughly +1 GB on the body and does not fit the DFlash2 100K profile (625 MiB free
+ 78 MiB slack). This is the "RCO per-tensor allocation search" plan §9 excluded, and it is a
maintainer scope decision. Cleanest alternative: ask ISTA-DASLab whether a compressed-tensors
GSQ-RCO checkpoint can be published, which would repack verbatim under D2.

**Recommendation (superseded 2026-10-02 by the apples-to-apples MBPP rerun).** The greedy,
matched-thinking rerun measures `rk4v4-e8` **90%** against beellama `kvarn5/5` 90% and `kvarn4/4`
92%: the earlier 84/88 separation was a harness artifact (sampling/thinking), and the
non-monotonic kvarn4/4 > kvarn5/5 ordering is itself within MBPP noise (~1.5-2 points SE). The KV
codec is not a quality limiter for this artifact; the RCO/weight campaign, the GGUF port, and the
KVarN port are off the critical path. `rk4v4-e8` is the accuracy profile and already fits the
DFlash2 100K profile unchanged; the KV byte/fit measurements below stand as capacity facts, not as
a quality fix.

**Launcher override (2026-10-01).** All four 4080 launchers now take `NINFER_KV_DTYPE`
(default `rk4v4-e8`) instead of a hardcoded `--kv-dtype`, and echo the served mode;
`scripts/check-linux-scripts.sh` covers the override. The DFlash2 launcher's POSIX and Windows
variants had drifted (`.sh` used `rk4v4-e8`, `.bat` used `int8` locally); the override makes
the bisect runs (`NINFER_KV_DTYPE=int8`, `rk8v4`, `rk4v4`) one env var on both hosts.

**GSQ-RCO port landed (2026-10-01, session 18).** Three findings shaped the converter:

- **V-head order.** llama.cpp's `conversion/qwen.py` (`_LinearAttentionVReorderBase`) stores
  the GDN value-side tensors in *tiled* `[group, K-head, dim]` order while the artifact (and
  the GSQ checkpoint) use the grouped `[K-head, group, dim]` order: `in_proj_qkv` V rows,
  `in_proj_z`, `in_proj_a/b`, `conv1d` V channels, `out_proj` columns, `A_log`, `dt_bias`. The
  port inverts the permutation for `gdn/value_z` (V and Z) and `gdn/output`; every other GDN
  object is copied from the GSQ3 artifact. Without the inversion the fused values measured
  cosine 0.05 against the GSQ3 artifact; with it, 0.96-0.97 (full attention and MLP 0.92-0.95).
- **Fused pairs are format-locked by the ops.** `attn_input_proj` requires (query_key,
  gate_value) = (Q3,Q3) or (Q4,Q5); `gdn_input_proj` requires (gdn/query_key, gdn/value_z) =
  (Q3,Q3) or (Q4,Q5); `linear_add` admits only Q3 or Q5 for `attention/output`, `gdn/output`,
  and `mlp/down`. The allocation is therefore resolved per layer to a route, not per tensor: a
  4-bit-or-wider contributor promotes the pair to (Q4,Q5), the linear_add sites and gate_up map
  to Q5/Q4. Final mix: 175 Q3G128 + 59 Q4G64 + 88 Q5G64; the mixed artifact is
  15,106,146,816 B and the all-Q3 RTN control 13,330,776,576 B.
- **Second-generation quantizer.** The registered max-scaled grid loses about a third more
  squared error than necessary at 3 bits, so the port uses a per-group clipping-factor search
  (`gsqrco_quantize.py`, same fp16-scale/codes/clamp/padding contract). Measured
  second-generation rel-L2: Q3 0.19-0.22, Q4 0.10-0.13, Q5 ~0.05.

New files: `inventory_gsqrco.py`, `gsqrco_source.py`, `gsqrco_quantize.py`, `convert_gsqrco.py`,
`verify_gsqrco.py`, and `tests/convert/qwen3_8_27b/test_gsqrco_{convert,quantize}.py`; the C++
profile is `WeightsProfile::Qwen38GsqRcoIq3S` (identity `gsqrco-iq3s`) with `Binder::tensor_format`,
a format-agnostic `bind_gsqrco_text_layers`, and Q5-aware site maxima in `variant.cpp`. Both
artifacts load and generate through the new profile; PPL and the maintainer's MBPP run are the
remaining gates.

**Port results (2026-10-01).** Both artifacts load through the new profile and generate coherent
greedy code on the CLI scenario (mixed: 66.6 tok/s, DFlash2 K=7 acceptance 24.6%; uniform
control: 123.5 tok/s, 46.1%). `verify_gsqrco` checked all 322 ported objects against an
independent gguf-py dequantization (worst rel-L2: Q3 0.2116, Q4 0.1133, Q5 0.0501) and all 862
copied objects byte-identical to the GSQ3 donor. Quick perplexity (int8 KV, same corpus, same
harness): GSQ3 **4.596095**, RCO port **4.685275**, uniform-RTN Q3 control **5.256053**. Reading:
the RCO allocation improves on an equal-quantizer uniform body by 10.9% PPL, but the
second-generation RTN+clipping quantizer loses more than the allocation recovers, so the port
still trails the GSQ3 artifact by 1.9%. The limiting factor is quantization quality (GSQ), not
the allocation or the artifact plumbing. Full `ctest`: 133 passed, 13 expected skips, 0 failed.
The maintainer's MBPP run on the mixed artifact is the remaining product question; a quality win
would need GSQ-quality re-quantization of the GGUF values or an official compressed-tensors
GSQ-RCO checkpoint.

**Follow-up (maintainer, 2026-10-01, corrected): the KV codec is the whole delta.** Controlled
MBPP on the same artifact, DFlash2 K=7 backend, and harness: `rk4v4-e8` (4.375 b/v) **84%**,
`int8` INT8-G64 (8.25 b/v) **88%**, beellama kvarn5/5 (5.375 b/v) 88%. The first reading recorded
in this file was inverted; the corrected reading exonerates everything held constant between
the two NInfer runs (weights - uniform GSQ3 vs the RCO GGUF, decode A8, sampling) and leaves
the KV store as the candidate cause, since with `int8` the uniform GSQ3 artifact appeared to match
beellama's RCO IQ3_S on MBPP. **That reading was harness-confounded; see the apples-to-apples
rerun below (`rk4v4-e8` 90% against beellama 90-92%).**

**rk8v4 bisect and the 100K fit (2026-10-02, maintainer + engine check; quality reading
superseded below).** First-pass MBPP with only the KV dtype changed: `rk8v4` (8-bit K, 4-bit V)
88%, `int8` 88%, beellama `kvarn5/5` 88%, against `rk4v4-e8` 84%. That read the 4-bit V as free
and the four points as the 4-bit K store; the apples-to-apples rerun below withdraws the reading.
Plane bytes per K or V vector per head (256-dim, four FP16 group-64 scales): `rk4v4-e8` 136 B,
`rk8v4` K 264 B / V 136 B, `int8` 264 B; with 17 K/V layers (16 full attention plus the MTP layer)
the per-token rates are 18,496 / 27,200 / 35,904 B, so `rk8v4` costs +8,704 B/token, +0.83 GiB at
102,400. The boot test on the current 13,330,776,576 B artifact (MTP3 + vision,
`--host-kv-mib 4096`, RTX 4080) shows that price is not payable at 100K: `rk8v4` fails startup by
**147 MiB** at `--prefill-chunk 1024` (requested 3,638,545,152 B against 3,484,262,400 B
available) and by 239 MiB at the launcher's chunk 2688; `rk4v4-e8` starts the same profile with
703 MiB planned slack, and 96,000 tokens boot with 19 MB slack, putting the MTP `rk8v4` cap near
**96.7K** (chunk 1024) / 93.2K (chunk 2688). The same arithmetic on DFlash2: `rk8v4` caps in the
low 70Ks and kvarn5/5 (had it been ported) near 83K, while any byte-neutral codec keeps 100K.
With `rk4v4-e8` at parity quality (see below), none of this requires a codec change; it only
bounds future experiments.

**Apples-to-apples MBPP rerun (2026-10-02, maintainer; supersedes the codec-quality readings
above).** With greedy decoding and the same xhigh thinking level on both engines: `rk4v4-e8`
**90%**, beellama `kvarn5/5` **90%**, beellama `kvarn4/4` **92%**. The 84/88 separation and the
codec ordering above were harness artifacts (sampling/thinking), and the kvarn4/4 > kvarn5/5
inversion is itself within MBPP noise (~1.5-2 points SE). The KV codec is therefore not a quality
limiter for this artifact; `rk4v4-e8` is at parity with the beellama reference and already fits
the DFlash2 100K profile, so the KVarN port loses its quality motivation. The plane-byte and fit
measurements in the note above remain valid as capacity facts.

**KVarN: no longer required (quality motivation closed 2026-10-02).** KVarN is Huawei CSL's
variance-normalized KV quantization: Hadamard rotation, Sinkhorn-like iterative variance
normalization over a 128-token tile, asymmetric RTN, independent K/V widths 2-8 (~0.375 b/v
metadata), with the incomplete tile and an exact sink kept in F16 (arXiv 2606.03458; Apache-2.0
vLLM fork; ported into `Anbeeld/beellama.cpp`, whose kvarn5/5 is the 88% reference). Its value was
88%-class accuracy below int8 bytes; the apples-to-apples rerun above closes that demand. On this
card's 100K accounting
(`rk4v4-e8` 20.3 KiB/token, `int8` 38.5 KiB/token), kvarn4/4 (4.375 b/v) is about the current
E8 size (~20.4 KiB/token, so every current profile fits unchanged) and kvarn5/5 (5.375 b/v) is
~25.1 KiB/token, +0.47 GiB at 100K; that fits the MTP 100K profile but not DFlash2, whose 100K
profile keeps only ~41 MiB planned slack and caps kvarn5/5 near 83K (see the fit note above). The
port is a cache-architecture change, not a codec swap: the encode normalizes a whole tile in two
passes and must keep the open tile exact until it seals, which touches the
paged cache, frontier/prefix reuse, host checkpoints, the fused append, every prompt/decode
attention path (including the small-T split-K and DFlash2's non-causal attention), and the
speculative block-commit semantics. Beellama's port spans ~20 files including
`ggml-cuda/kvarn.cu` and `fattn-tail.cuh`. With `rk4v4-e8` at 90% and fitting every target
profile, nothing here is on the critical path; revisit only if a new capacity or quality
requirement names it.

**GGUF-to-artifact port: feasible, no longer on the critical path.** It was the proposed weight
fix, and the `int8` control now shows the uniform GSQ3 weights are not the limiting factor.
Keep it as a documented option (e.g. if the RCO allocation is wanted for other reasons or a
handover to a mixed-precision artifact is planned): `gguf-py`'s `gguf.quants.dequantize` covers
every type in the allocation (Q2_K/Q4_K/Q6_K/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS),
the published `.rco-allocation.txt` pins the per-tensor map, and a streaming converter can
re-encode into the registered formats (IQ4_XS/Q4_K -> Q4G64, Q6_K -> Q6G64, IQ3_S/IQ3_XXS ->
Q3G128, F32/BF16 vectors as-is). It is a second-generation quantization, not verbatim, and the
IQ2_*/Q2_K/IQ1_M tensors have no registered counterpart (promote to Q3 or add a Q2 format).

**Runtime blocker for the port (2026-10-01).** `artifact::bind_tensor` requires an exact
per-object `NumericFormat`, and `bindings.cpp` passes `Q3G128_F16S` at every GSQ3 body-parent
call site, while `variant.cpp`'s per-site workspace queries assume one profile-wide qtype. A
mixed RCO artifact therefore cannot load under `Qwen38Gsq3`: it needs a new `WeightsProfile`
(proposed `Qwen38GsqRcoIq3S`, `weights_id` `gsqrco-iq3s`), a per-role format table shared by
the bindings and the workspace-capacity queries, a `resolve_weights` case in `package.cpp`,
and the Python source adapter/inventory/verifier. The GGUF download is at
`/models/qwen3.8-27b-gsq-rco/` (`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, 11.8 GB); the allocation
map is already on disk. Fit caveat: promoting the IQ2_*/Q2_K/IQ1_M body tensors to Q3 grows the
artifact, so the sizing pass must precede the C++ work.

**Port sizing pass (2026-10-01, measured against the real file).** The GGUF is complete in
`/models/qwen3.8-27b-gsq-rco/` (12,120,016,960 B, 866 tensors) and the `mmproj` vision tower is
fetching. Mapping every text-core object to the GGUF allocation (IQ4_XS/Q4_K -> Q4G64, Q6_K ->
Q6G64 where present, IQ3_S/IQ3_XXS -> Q3G128, and the IQ2_*/Q2_K/IQ1_M tensors promoted to
Q3G128; fused objects take the widest contributor) gives a text core of **11.431 GiB against
the current 10.157 GiB, +1.274 GiB**, with 133 objects at Q4G64 and 189 at Q3G128. All eight
body sites can reach Q4G64 somewhere, so the new profile's workspace queries can use Q4G64 as
the site maximum. Reuse plan: only the 322 text-core matrix objects change; vision, DFlash2,
draft head, MTP and resources are copied byte-for-byte from `out/qwen3_8_27b_gsq3.ninfer`
(same underlying values), which keeps the port a controlled test of the RCO body allocation.
Fit: the artifact grows to ~13.7 GiB, so the 100K DFlash2 profile needs a context cut;
MBPP-scale contexts are unaffected. `gguf` 0.19.0 is installed in the dev venv and the
reference dequantizer covers all ten types.

### N-gram wide verify window re-enabled (2026-10-02, session 19)

**Change.** `make_sequence_planner_impl` sets the MTP verify window to `ngram.max_drafts` when the
n-gram chain is enabled and to `draft_tokens` otherwise, so the wide graph family and record planes
session 7 planned are materialized. The session-7 failure was closed as bf16-ulp trajectory
sensitivity in session 15; this is the maintainer's re-enable decision. `program_impl.h`'s stale
"disabled" comment is removed, `test_engine_ngram_real.cpp` now requires at least one wide round,
and `docs/cli.md`/`docs/serving.md` drop the disabled caveat.

**Evidence** (`out/qwen3_8_27b_gsq3.ninfer`, RTX 4080; logs `/tmp/s19_ngram_*.log`,
`/tmp/s19_jsonl_*.err`, `/tmp/s19_ctest.log`):
- `ninfer_qwen3_6_27b_ngram_real_test`, bf16, default graphs on: **80 wide rounds of 560**, 958
  n-gram drafted / 555 accepted, **0 n-gram-added divergences** over MTP, two fresh engines
  token-identical, PASS. The n-gram and MTP-only runs' first divergences agree on all six prompts.
- Non-default overrides expose the session-13..17 bf16-ulp trajectory class. `NINFER_NGRAM_GRAPH=0`
  at max 15: 80 wide rounds of 563, 958/555, **1 added** (prompt 2 token 42, gap 5 nats); max 6 with
  graphs on: 126 wide rounds of 619, **1 added** (prompt 3 token 322, gap 16.4); max 6 graphs off:
  **2 added** (prompt 2 gap 5, prompt 3 gap 16.4). The n-gram run's own first divergences are
  identical in both graph modes at max 15; only the narrow MTP baseline's first flip moves (token 42
  with graphs on, token 270 with graphs off). The wide round's partition perturbs the state at a
  near-tie, and the scoring
  oracle evaluates the reference prefix, so a token chosen from a different basin can appear as a
  gross gap. The committed gate runs the default route (graphs on, max 15), which passes.
- `rk4v4-e8` (report-only): 65 wide rounds of 608, 778 drafted / 431 accepted, 0 added divergences,
  PASS.
- CLI `examples/cli/messages/scenario_structured_jsonl.json`, greedy, bf16, 256 tokens: n-gram
  58 rounds / 139.8 tok/s against MTP 68 rounds / 129.1 tok/s (-14.7% rounds, +8.3% decode), outputs
  byte-identical (`0f7c070916968adb5456c23e9d3fd718`). The code scenario's pool barely fires (1 wide
  round, 0 accepted) because its output is novel code; the pool's buy is restated structure.
- Full `ctest`: **133 tests, 120 passed, 13 expected skips, 0 failed** (unchanged from session 16);
  `git diff --check` clean.

**Residual.** Non-default decode routes (`--no-cuda-graph`, and `--ngram-max 6` on this build) can
report added non-tie flips under the scoring oracle. They are the accepted session-13..17 class
(the session-16 decode A8 profile drifts MTP-only bf16 too), not wide-window corruption; the
default product and test route (graphs on, max 15) passes.

### Staging-ceiling audit: the Q3 memory pattern is not the wall (2026-10-02, session 20)

**Question.** The session-13 decode audit read the Q3 body at ~380-445 GB/s and named the small-T
staging structure (32 rows x 192 B stages at a 1920 B row stride, 2 CTAs/SM, 3-deep cp.async ring)
as the limiting pattern. This session tested that hypothesis directly: a throwaway probe
(`profiles/bench/5c-decode-levers/staging_probe.cu`) runs the production staging pipeline with the
decode/MMA/activation removed and sweeps stage width (192/384 B), ring depth (2/3/5), rows per
CTA (16/32/64), resident CTAs per SM (1..5), warp-private vs CTA-shared pipelining, and full-row
bursts, against linear-read controls; the production kernel was re-measured through
`ninfer_linear_bench` with `flush_l2` temporarily changed from a 256 MiB memset to a read flush
(patch reverted after the run).

**Probe rates (34816 x 5120 codes + scales = 69.6 MB/pass, median of 30, clean L2 via read flush).**

| config | median | model GB/s |
|---|---:|---:|
| read_linear float4 (occupancy grid, 456 CTAs) | 103.4 us | 673 |
| stage_linear cp.async 4 KB stages | 104.3 us | 668 |
| stage_shared 192 B ring3 32r 2cta (production shape) | 107.5 us | 648 |
| stage_shared 192 B ring3 32r 1cta / maxcta (5/SM) | 107.5 us | 648 |
| stage_shared 384 B ring2/3 / ring5 / 16r / 64r | 107.5 us | 648 |
| stage_warp 384 B ring3 4r/warp (no block barriers) | 107.5 us | 648 |
| row_burst 16r / 32r full-K | 103.4 / 102.5 us | 673 / 679 |
| stage_shared 192 B + the production A8 decode loop | 109.6 us | 636 |

**Flush-mode effect (same kernels, mode 0 = memset flush, mode 1 = read flush).** read_linear 466
vs 646 GB/s; stage_shared 453 vs 627; the production Q3 A8 kernel in `ninfer_linear_bench` T=2/4
156.7 -> 140.3 us and T=5..8 157.7-158.7 -> 148.5-150.5 us; T=9..16 unchanged at 277.5 us. The memset flush
leaves a dirty L2 whose writebacks share DRAM with the timed stream and depresses streaming numbers
by 10-27% (the `a8small_34816x5120_cleanflush.csv` keeps the clean run).

**Conclusions.**

1. The staging structure is not the wall. Every structural variant lands within 1-2% of the pure
   float4 read (673 GB/s), and the production shape is only 4% below it. Restaging (wider stages,
   deeper ring, more CTAs, row bursts, warp-private) is a dead end.
2. The production A8 decode is effectively free once the pipeline overlaps: staging 107.5 ->
   staging + decode 109.6 us (+2%).
3. The production kernel's remaining cost is the MMA/x-staging/reduction path plus the separate
   `a8_g64_quantize` launch: 140.3 us clean against 108-110 us of staging + decode, i.e. ~32 us
   per big parent, not the weight stream.
4. `ninfer_linear_bench`'s memset flush understates streaming kernel rates, so recorded absolute
   GB/s (the 445 "wall") are not kernel ceilings. Engine-level Q3 remains ~23 ms/round (~413 GB/s)
   against ~16 ms at the clean op-level rate; a fresh nsys on the current build must split that
   gap between kernel consume, shape mix, quantize launches and wave tails.
5. The T=9..16 double tile is now fully explained: 2 x the ~140 us pipeline (277 measured), so a
   native 16-column single-pass tile is worth ~135 us per double-tile call (K=15 Q3 ~43 -> ~21 ms).

**Plan change.** The session-13 "restage the A8 tall/small-T code path" candidates are withdrawn.
Decode work is redirected to: (a) attribute and cut the ~32 us consume (fuse/overlap the activation
quantize, x staging, MMA/epilogue) with a fresh nsys on the current build; (b) the native T=9..16
single-pass tile (DFlash2 K=15 / wide windows); (c) the Q4 main-head single-pass at T=8
(`q4_dispatch.cpp` n=248320 takes `launch_q4_simt_r8_c4`, two 4-column passes; the draft-head
small-T route is single-pass). Recommendation for the maintainer: give `ninfer_linear_bench` a
read-flush variant (or switch the default) so streaming measurements stop carrying the dirty-L2
penalty.

**Evidence.** `profiles/bench/5c-decode-levers/staging_probe.cu`,
`staging_probe_clean_vs_dirty.log`, `a8small_34816x5120_cleanflush.csv`. Probe command:
`nvcc -O3 -std=c++17 -arch=sm_89 stage_probe.cu -o stage_probe && ./stage_probe 30`.

### Decode consume attribution and the A16 GDN stragglers (2026-10-02, session 21)

Step 1 from the session-20 redirect. Fresh nsys captures on the current build at 8K, tiled corpus,
`--cuda-graph-trace=node`: `profiles/nsys/session21-{dflash7-8k,mtp3-8k,linprof-q3a8-t4}.*`
(the `QdstrmImporter` needs `libdw1t64`, reinstalled after the image rebuild; container change only).
Per-round counts are keyed by the once-per-round selector/argmax kernels (76 DFlash2 rounds, 74
MTP3 rounds over the two bench runs).

**One public Q3 A8 call (N=34816, K=5120, T=4):** `a8_g64_quantize_kernel` **1.66 us** +
`q3_small_t_a8_mma_kernel` **137.49 us** (grid 1088, 256 threads, 68 registers). The separate
activation quantize is 1.2% of the call and is not the consume gap. Against the session-20 probe
(staging 107.5 us, staging + decode 109.6 us) the remaining ~30 us is the in-kernel x staging /
ldmatrix / MMA / reduction / epilogue path.

**DFlash2 K=7 per decode round** (round = 32.9 ms, 134.6 tok/s at 4.43 tok/round):

| route | calls | avg | ms/round |
|---|---:|---:|---:|
| A8 SwiGlu gate_up (grid 1088) | 64 | 149.6 us | 9.57 |
| A8 plain N=5120 (grid 160) | 128 | 68.6 us | 8.78 |
| A8 plain N=7168 (grid 224) | 32 | 33.6 us | 1.08 |
| **A16 gdn/value_z N=12288 (grid 384)** | 48 | 70.1 us | 3.36 |
| **A16 gdn/query_key N=4096 (grid 128)** | 48 | 25.7 us | 1.23 |
| a8 quantize (decode) | 224 | ~1.7 us | 0.38 |
| **Q3 body total** | 320 | | **24.4** |

MTP3 (T=4) per round: A8 SwiGlu 10.20 + A8 plain(N=5120) 9.47 + A8 plain(N=7168) 1.16 + A16
grid384 3.44 + A16 grid128 1.35 + quantize 0.4 = **25.6 ms** of a 30.5 ms round.

**Findings.**

1. The engine's Q3 kernels run at the clean op-bench rates (SwiGlu 149.6 us in-engine vs 150.5 us
   clean). There is no engine-level Q3 penalty; the session-20 "~23 ms vs ~16 ms" comparison used
   the staging-only floor, which is not the kernel floor while the consume is serialized. The
   current ceiling is ~24 ms/round until the consume is cut.
2. The remaining per-parent consume is ~30 us (T=4) to ~40 us (T=8 SwiGlu) over staging + decode,
   entirely inside `q3_small_t_a8_mma_kernel`. A probe iteration adding x staging then the MMA path
   would split it; the fix candidates are a warp-specialized pipeline or a leaner consume.
3. **30% of decode Q3 calls are A16 by construction.** The GDN conv record/snapshot projections on
   the split Q3 payload run A16: `gdn_input_proj_conv_record`'s Q3 two-parent branch hardcodes
   `LinearPolicy::A16Only` (`src/ops/wrapper/gdn_input_proj.cpp:1112`; the snapshot branch at 1178),
   and `Variant::gdn_input_projection_record/_snapshot` call the no-policy overloads
   (`src/targets/qwen3_6_27b/impl/variant.cpp:276/301`), while the fused payload path passes
   `text_policy(...)`. Op bench at T=8, A16 -> A8: N=4096 32.8 -> 29.7 us, N=12288 77.8 -> 67.6 us.
   Threading `kQ3TextPolicy` through the split record/snapshot forms and their workspace capacity
   queries should return ~0.6 ms/round on DFlash2 K=7 (~2%), at the cost of the decode-A8 numerics
   class already accepted in session 16.
4. **The small-N parents lose ~2-3.5 ms/round to wave quantization.** The A8 small-T kernel runs
   2 CTAs/SM (42,112 B static smem at StageTokens=4, 48,640 B at 8; 152 resident slots). The
   per-launch histograms are bimodal at almost exactly the two-wave times: N=5120/grid 160 (1.05
   waves) splits into ~35 us (K=6144) and ~93 us (K=17408) on DFlash2 (MTP3: 34/89 us), against
   single-wave ideals of ~24.4/69.0 us at the kernel's own 504 GB/s big-shape rate; N=7168/grid 224
   (1.47 waves) is ~31.6 us against ~27.5. That is 64 N=5120 layers + 32 N=7168 calls per round,
   ~2-3.5 ms of the 32.9 ms DFlash2 round. Fitting 3 CTAs/SM (<=34,133 B: 2-buffer ring at
   StageTokens=4; 2-buffer ring + half decoded tile at StageTokens=8) removes the second wave with
   no arithmetic change; the session-20 probe showed ring 2 stages as fast as ring 3 and occupancy
   irrelevant for a full-wave shape.

**Next.** (a) the Q4 main-head single-pass at T=8 (extend the Q4 small-T geometry to n=248320; the
session-21 ranking's item 3) — **landed session 22**; (b) split the in-kernel consume with one more
probe (x staging vs MMA) and try a warp-specialized/leaner consume — **landed session 25**
(direct-register A fragments, §11); (c) the T=9..16 single-pass tile for K=15 — **landed session 28**
(§11).

### Occupancy/wave-tail experiment: rejected (2026-10-02, session 22)

Session-21 item 1. Tested in `q3_rowsplit_small_t_a8_mma.cuh` (working tree only, reverted): two code
buffers plus a 16-row half decoded tile consumed in two m16 phases, taking static shared memory from
42,112/48,640 B to 25,344/29,696 B and occupancy from 2 to 3 CTAs/SM (228 slots). The occupancy probe
confirmed 3 blocks/SM for both StageTokens variants at 72-80 registers, and the A8 oracle suites
passed, so the restructure was arithmetically sound.

Measured, then reverted:

- Op bench (dirty flush, 34816x5120): T=4 156.7 -> 158.7 us; T=8 158.7 -> 162.8 us.
- Engine nsys (DFlash2 K=7, 8K, `profiles/nsys/session22-dflash7-8k.*`): Q3 A8 body 19.43 -> 21.18
  ms/round (+9.0%); big parent median 138.2 -> 140.4 us; N=5120 parents 35.8/93.0 -> 36.8/97.8 us;
  N=7168 31.6 -> 31.8 us. The A16 parents were unchanged (24.5/65.5 us).

Conclusion: the kernel is per-CTA issue/latency-bound, not wave-bound. Three CTAs/SM time-slice the
same SM throughput (aggregate ~430 GB/s is unchanged), while the two-deep ring and the extra
barriers cost more than the removed partial wave. The bimodal small-parent times are real, but the
tail is not reachable by trading ring depth for occupancy. The kernel is back to the committed
revision (`git checkout`; tests pass). Session-21 item 1 is closed.

### GDN record/snapshot A8 policy threading (2026-10-02, session 22, item 4)

Landed: the split Q3 GDN conv snapshot/record projections now take a compute policy. New
policy-taking two-parent overloads in `include/ninfer/ops/gdn_input_proj.h`; the wrapper's Q3
branches use the passed policy (`src/ops/wrapper/gdn_input_proj.cpp`); the Q3 branches of the
snapshot/record capacity queries admit A16Only/AllowA8; `Variant::gdn_input_projection_record` and
`_snapshot` and their workspace helpers pass `kQ3TextPolicy`. The no-policy two-parent forms remain
as A16Only delegates. `ninfer_gdn_input_proj_conv_record_test` now runs both policies;
record/snapshot/replay-fold suites pass. Engine capture (`profiles/nsys/session23-dflash7-8k.*`):
the A16 small-T kernel no longer appears in decode; grid128 24.5 -> 21.3 us, grid384 65.5 -> 56.1 us,
Q3 body 24.03 -> 23.33 ms/round (-0.70 ms; +2.2% on the 32.9 ms DFlash2 K=7 round).

**Numerics decision point.** A8 quantizes the projection activation per token and 64-code group
(about 0.4% per group) while A16 used bf16 activations; both keep the weight codes exact int8 with
fp32 group scales. It is the same documented A8 contract already covering the main decode linears
since session 16. On the real artifact (DFlash2 K=7, `scenario_code_python.json`, greedy) the A8
record changes the output hash 54c59149db5f -> 2eb9e160cb96; the only text difference is the first
sentence's near-tie phrase ("understand what I'm working with" vs "...what's already there"), and
the rest, including the code block, is byte-identical. That is the session-16 MTP3 near-tie class
reached by DFlash2 K=7, and the trade is explicit: keep the +0.70 ms/round with the drift, or revert
the two call sites to A16Only and keep DFlash2 K=7 at the reference hash. MBPP has not been rerun
with the record path on A8.

**Reverted 2026-10-02 (maintainer decision).** The kernel gain was real, but the end-to-end trade
was not acceptable: the A8-record trajectory lost acceptance length on the code scenario
(3.82 -> 3.42 tok/round, within sampling noise at that length but far exceeding the +0.70 ms/round
kernel gain if it persisted). The split call sites and their workspace helpers are back to
`A16Only`, and the policy-taking two-parent overloads, the capacity gates and the two-policy test
case are reverted with them - the path needs no policy plumbing until a measured acceptance A/B
justifies A8. Verified: `ninfer_gdn_input_proj_conv_record_test`,
`ninfer_gdn_input_proj_conv_snapshot_test` and `ninfer_gdn_replay_fold_test` pass, and DFlash2 K=7
on the code scenario is back to the reference output `54c59149db5f` (42 tokens, 40.3% acceptance,
118.6 tok/s).

### Q4 main-head single-pass at T=8 (2026-10-02, session 22, item 3)

Landed. `Q4DraftHeadGeometry<InputRows>` became `Q4SmallTGeometry<OutputRows, InputRows>` (the old
name is the 131072-row alias); the small-T launcher gained a 248320-row geometry and was renamed
`launch_q4_small_t_mma` (`q4_launch.h`, `q4_dispatch.cpp`). The main head now selects it for T=2..8;
T=9..15 keep `simt_r8_c4`, T=16 c8, T>=17 the tall MMA. `ninfer_linear_q4_a16_test` passes (its
[248320,5120] cases cover T=2/3/4/8).

Evidence:

- Op bench (dirty flush, 248320x5120): T=8 2338.8 -> 1185.8 us (2.0x, 573 GB/s); T=4 1201.2 ->
  1136.6 us; T=5/6/7 are single-pass now as well. T=9..10 unchanged (still three simt tiles).
- Engine nsys (`profiles/nsys/session24-dflash7-8k.*`): the 31040x2 simt head launch (2245 us) is
  replaced by a 15520x1 small-T launch at 1074.6 us, -1.17 ms/round; Q4 total 7.78 -> 6.87 ms/round.
- DFlash2 K=7 code scenario, greedy: 118.6 -> 122.5 tok/s (+3.3%) at the same acceptance (40.3%,
  3.82 tok/round) and the same output hash `54c59149db5f` - no trajectory change.

Follow-up not taken: the T=9..16 head window (K=15, wide n-gram) still runs the two-tile c8 route;
extending the main-head launcher table to 16 columns would make it single-pass too.

Integration evidence: `ninfer_qwen3_8_27b_dflash2_real_test 7 1 1 1` (K=7, B=1, graph, optimized)
passes with `accepted=20/20`; the default B=8 argument aborts on the runtime reservation, which is the
pre-existing 16 GB fit limit, not this change. Full rebuild plus `ctest` exit 0 with the same 13
expected skips and no `LastTestsFailed.log` update.

### In-kernel Q3 A8 consume cut: direct-register A fragments (2026-10-03, session 25)

**Probe split.** The session-21 next step (split the ~30 us of in-kernel consume) ran on a throwaway
`profiles/bench/5c-decode-levers/consume_probe.cu` that mirrors the A8 kernel with stage switches and
instantiates the production kernel as the reference. Under the clean read flush, T=4 production is
136.2 us against the 107.5 us staging floor, and the off-by-one variants (full minus x-staging /
decode / B ldmatrix / A ldmatrix / mma / epilogue / reduction) all collapse to 109-113 us: the
consume is one serialized chain (staged codes -> decode -> barrier -> ldmatrix -> mma -> epilogue),
and removing any link drops the kernel under the memory stream. A diagnostic warp sync in place of
the decode barrier is worth only 3-4 us, so the fix is not just fewer barriers.

**Fix: direct-register A fragments.** Each lane now extracts the four 12-bit fields its m16n8k32 A
registers need straight from the staged 24-byte step (rows gid/gid+8 plus 16*mt, bits
96*ks + 48*(k&1) + 12*lid; two byte loads, a shift/mask, and the same `spread4`) and feeds the mma.
The decoded int8 tile, its ldmatrix read and the decode->MMA barrier disappear; the end-of-loop
barrier is the stage's only block-wide synchronization, and the ring still stages into buffer
stage-1 while the current stage's codes are read. The arithmetic, K order, mma sequence and
eight-way reduction are unchanged, so outputs are byte-identical: the probe's 34816x5120 T=4/T=8
byte-compare against the production kernel is IDENTICAL. `Storage` drops the decoded tile.

Evidence (RTX 4080, `out/qwen3_8_27b_gsq3.ninfer`; logs under `profiles/bench/5c-consume/` and
`profiles/nsys/session25-*`):
- Probe clean flush: T=4 136.2 -> 111.6 us (-18.1%); T=8 136.2 -> 113.5 us (-16.7%).
- `ninfer_linear_bench --flush read` (new option, the session-20 recommendation) on 34816x5120 A8:
  T=2 140.3 -> 116.7, T=4 141.3 -> 119.8, T=8 149.5 -> 118.8, T=9..13 277.5 -> 203.8 (-27%),
  T=14..16 258.1 -> 190.5 (-26%). The old dirty-flush numbers move only 2-3 us at T<=8 because the
  memset writeback masks the consume.
- Engine, same build, DFlash2 K=7 CLI code scenario greedy: decode 338 -> 301 ms, 121.4 -> 136.1
  tok/s (+12.1%), output hash `54c59149db5f` and 40.3% acceptance (3.82 tok/round) unchanged.
  Depth sweep: 8K 136.0 -> 155.4, 28K 232.1 -> 262.3, 56K 204.9 -> 228.7 tok/s; prefill unchanged.
- nsys `session25-dflash7-8k` (76 rounds): A8 SwiGLU 149.6 -> 136.6 us, plain N=5120 68.6 -> 61.0,
  N=7168 33.6 -> 30.5; per-round A8 body 19.43 -> 17.53 ms.
- `ctest` 133/120/13/0; all Q3 op suites pass; `ninfer_qwen3_8_27b_dflash2_real_test 7 1 1 1`
  accepted=20/20; the n-gram real route passes with 80 wide rounds and 0 added divergences;
  `git diff --check` clean.

Session-21 item (b) is closed. Item (c) wanted the native single-pass 16-column tile for the
double-tile cost at T=9..16 (203.8 us against 118.8 at T=8); **session 28 (§11) landed it** at
140.3/139.3 us (A8/A16-class op numbers are in the session-28 record). Bench tooling now takes
`--flush read|memset`; use
`--flush read` for streaming claims.

### Prompt attention worker V-dequant: 5c.4 landed (2026-10-03, session 26)

**Deliverable.** The session-13 audit's ranked candidate 5: `src/ops/softmax_attention/dense/
causal_cache/prompt_i8.cuh` — 42% of the pp100000 kernel time and "the largest untouched prefill
kernel". Accepted evidence is the op test, the append bench at 8K-128K, and the engine prefill.

**Port.** The fork's `7b6ed55` schedule. Our kernel predated the fork's pipeline (all 16 warps did
PV, P lived in shared memory, and every tile paid two full-CTA barriers), so the whole structure
was adopted, not only the commit delta:

- eight producer warps (four 16-row tiles x two Bc column halves) own QK, the online softmax, K
  staging and packed-K expansion, and keep P(t+1) in registers;
- eight worker warps (two 16-row tiles x one 64-dimension group) own the FP32 accumulator, PV,
  and V issue + dequant;
- `PFree`/`PReady` one-sided named barriers (`bar.arrive`/`bar.sync`, count 512) replace the two
  per-tile `__syncthreads()`: producers score tile t+1 while workers PV tile t, and the workers
  dequantize V(t+1) right after PV(t) behind a worker-only barrier, off the scoring path;
- packed K codes (rk4v4, rk4v4-e8) land by cp.async in the upper half of each packed V row and
  each producer expands the chunks it issued;
- QK walks key tiles outermost (one 8-byte scale load and one x4 ldmatrix per group, groups still
  summed in ascending order) and the shared `kv_cache_unpack_i4x16` is bytewise exact
  (xor/add/xor/byte permutes) — both from the same commit;
- `ninfer_causal_softmax_attention_bench` gained `--kv-dtype rk4v4-e8` so the production KV mode
  is measurable (bench/README.md updated).

**Op bench** (`ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4
--kv-dtype int8 --batch 1 --tokens 1024 --context ... --execution eager --cache cold --warmup 3
--repeat 5`, two rounds; CSVs `profiles/bench/5c-4/{base,after}_int8_r*.csv`):

| context | before | after r1 / r2 | delta |
|---:|---:|---:|---:|
| 8K | 2677.8 us | 1809.4 / 1805.0 | -32.4% |
| 16K | 5096.4 us | 3416.1 / 3409.6 | -33.0% |
| 32K | 9751.5 us | 6583.3 / 6583.9 | -32.5% |
| 64K | 18830.3 us | 12682.2 / 12985.0 | -32.7% |
| 128K | 38164.5 us | 24666.1 / 24792.4 | -35.4% |

`rk4v4-e8` after (same command; CSVs `..._rk4v4-e8_r1/r2.csv`): 2123.0/2138.0, 4013.1/4009.0,
7790.6/7835.0, 14825.5/14664.0, 29753.3/29603.0 us at the same five contexts.

**Engine, same-session A/B** (stash the port, rebuild, measure, restore; `ninfer_bench`,
`rk4v4-e8`, warmup 1, one repetition; logs `profiles/bench/5c-4/s26_pp32768.log` and
`s26_pp100000.log`):

- `pp32768`: **2314.07 -> 2469.95/2470.48 tok/s (+6.7%)** — against the plan's previously
  recorded 2277.53, so the prompt-attention share explains the delta.
- `pp100000 --prefill-chunk 2688`: **1709.99 -> 1971.36/1971.13/1968.17 tok/s (+15.3%)** — the
  100K profile now clears the session-9 1675.39 record by 17.7%.

**Numerics and behavior.** `ninfer_softmax_attention_test` passes full and `--rk4v4-e8-only`;
full `ctest` 133 tests, 120 passed, 13 expected skips, 0 failed; `git diff --check` clean.
`ninfer_qwen3_6_27b_ngram_real_test` (bf16) passes with exactly the session-19 counts — 80 wide
rounds of 560, 958 drafted / 555 accepted, 0 n-gram-added divergences — and
`ninfer_qwen3_8_27b_dflash2_real_test 7 1 1 1` reports `accepted=20/20`, so the prompt-kernel
restructure leaves the recorded real-route outputs unchanged. `ninfer-perplexity --quick`
(`--kv-dtype int8`) reports 4.596095 overall with the same per-domain numbers as the session-9
anchor, so the weight-quality gate is untouched.

**Residual.** No prefill item from the session-13 audit remains. The one open decode item was
session-21 item (c), the native T=9..16 single-pass tile for K=15 / wide n-gram windows; **session
28 (§11) landed it**.

**Byteshape allocation transplant (2026-10-09, exploratory; no plan stage).** The published
`byteshape/Qwen3.8-27B-GGUF` `Qwen3.8-27B-IQ3_S-3.23bpw.gguf` was used as an allocation prior
only: `inventory_byteshape.py` maps its per-tensor GGML types (IQ3_XXS and IQ2_XXS to Q3G128,
IQ4_XS to Q4G64, Q5_K to Q5G64, Q6_K to Q6G64) through the shared fused-object route resolution,
and `convert_byteshape.py` quantizes every text-core matrix once from the official BF16
checkpoint with the shared clipping-search encoder; Vision, MTP, the draft head and the Q4
DFlash2 companion are the registered first-generation objects. New identity
`qwen3.8-27b/byteshape-iq3s` with `WeightsProfile::Qwen38ByteshapeIq3s` (shares the gsqrco
binder, Q4 endpoints and Q4 companion), plus `verify_byteshape.py` and
`tests/convert/qwen3_8_27b/test_byteshape_convert.py`.

- `out/qwen3_8_27b_byteshape_iq3s.ninfer`: 13,741,441,536 B; ported mix 271 Q3G128 / 23 Q4G64 /
  28 Q5G64; conversion 437 s; allocation digest `86875f2fd2c0de4e…` in the report.
- Quick PPL (int8 KV, same corpus and harness): **4.810153** against GSQ3 4.596095 (+4.65%) and
  the RCO port 4.685275 (+2.7%); per domain chinese 5.833, english-long 7.696, english-reference
- **Same-file check** (`wikitext/00.txt`, 65,304 scored tokens, context 4096 / stride 2048, bf16 KV;
  llama.cpp on the same file with default KV and 15 context chunks): GSQ3 **6.3968**, RCO port
  **6.4560**, byteshape mix **6.6706**, uniform control **7.0755**; llama.cpp GSQ-RCO GGUF
  **6.2188**, llama.cpp byteshape GGUF **6.5740**. Within each harness the ordering is the same:
  the first-generation byteshape transplant lands next to its source GGUF (6.67 vs 6.57) and
  behind both ISTA artifacts; the RCO file is natively better than byteshape at 3.50 vs 3.23 bpw.
  The quick-corpus 4.81 is not comparable to a wikitext-only number (four streams, code domain
  1.77).
  6.670, code 1.774.
- **Encoder control** (`--uniform-body`: the GSQ3 allocation at 13,330,776,576 B, same encoder)
  scores **5.092467**. At identical allocation and size the publisher's GSQ codes beat the local
  clipping-search encoder by 0.496 PPL (10.8%), while the byteshape mix itself buys 0.282 PPL
  (5.5%) over uniform Q3 for +0.41 GB. The gap is encoder quality, not allocation — the standing
  "GSQ-quality re-quantization" finding, now measured first-generation.
- `verify_byteshape` passes over all 1184 tensors (direct formats exact, resource payloads equal
  to source; worst relative L2: Q3 0.2239, Q4 0.1469, Q5 0.0650, Q6 0.0247, W8 0.0084); report
  `profiles/perplexity/byteshape-verify.json`. The gsq3 and gsqrco identities still load under the
  new profile table.
- Result: the transplant loads, scores and is coherent, but it does not beat the existing GSQ3
  artifact, and no further allocation transplant is warranted before an encoder improvement. The
  uniform control is `out/qwen3_8_27b_byteshape_q3_control.ninfer`; both identities remain local
  to this tree.

### Native T=9..16 single-pass small-T tile (2026-10-09, session 28; session-21 item c)

**Deliverable.** The 9..16-column verify windows (DFlash2 K=15, wide n-gram) ran two 8-column
small-T CTAs per row block, so each code byte was staged twice and the weight stream doubled. Both
small-T engines now own the window in one CTA.

**Design.** `q3_small_t_mma_kernel` and `q3_small_t_a8_mma_kernel` take `TileTokens` (8 or 16) and
`StageTokens` (4, 8 or 16); one CTA owns one row block and one tile, so 9..16 columns launch one CTA
per row block (`launch_q3_mma_small_t_r32_c16` / `..._a8_r32_c16` and the two folded-SwiGLU c16
wrappers; the 8-column wrappers now assert their 1..8 domain). The B fragment is loaded per
8-column half and the A fragment pair is reused, so only the activation traffic grows with the tile.
The A16 kernel's 16-row activation makes its ring one shallower (two buffers, 39,168 B) to keep the
full 32x256 decoded tile inside the 48 KiB static shared limit at two CTAs/SM; the A8 kernel keeps
its three-buffer ring (45,312 B, two CTAs/SM). Padded-K shapes that already took the small-T route
take the wide tile at 9..16 too.

**Codegen finding.** Expressing the activation staging as one runtime-bounded loop (`for (item =
tid; item < StageTokens*32; item += kThreads)`) made nvcc speculate on out-of-range iterations
(tid+256, tid+512) and branch around each staging copy; that cost the *narrow* A16 kernels 2-11%
(T=5..8 172 -> 191 us) with no semantic change. Spelling the three shapes out - 8 rows take exactly
one item per thread with no guard, 4 rows guard the lower half, 16 rows walk two - restores the
baseline codegen (T=8 170 us against the pre-change binary's 171 us). The session-20 estimate of
"~135 us saved per double-tile call" was optimistic: the wide tile lands at 140 us against 204 at
T=16, i.e. ~64 us per call, because the activation stream is re-read by every row block and doubles
with the tile width (the A8 path reads it as int8, hence T=9..11 at 132 us).

**Evidence** (RTX 4080, `out/qwen3_8_27b_gsq3.ninfer`; the pre-change binary is kept as the A/B
control, and CSVs/logs are under `profiles/bench/5c-c16/`):

- `ninfer_linear_bench --qtype Q3 --policy a8|a16 --n 34816 --k 5120 --sweep 2:17:1 --warmup 3
  --repeat 10 --flush read`, baseline and after runs back to back in one clock window: A8 T=2..8
  0.98-1.00x, T=9..11 0.65x, T=12..16 0.69x; A16 T=2..4 0.99x, T=5..8 0.99-1.00x, T=9..16 0.68-0.69x.
- `test_q3_a16_small_t` and `test_q3_a8_small_t` pin the 16-column tile at T=9..16 and keep the
  FP64 / documented-quantization oracle coverage; each suite now also byte-compares the wide tile's
  first eight columns against the 8-column route (same codes and the same per-column MMA sequence,
  so a schedule change must be exactly equal). All nine Q3 suites pass.
- Engine, `scenario_code_python.json`, greedy, `rk4v4-e8`, same-binaries A/B: DFlash2 K=15 465 ->
  372 ms, 88.2 -> 110.2 tok/s (+25%), output hash `54c59149db5f` and 4.20 tok/round unchanged;
  DFlash2 K=7 300 -> 299 ms; MTP3 358 -> 360 ms (same rounds, acceptance and hash; within the
  scenario's ~1% run-to-run wobble, while the op-level T=4 points measure equal-or-faster).
- `ninfer_qwen3_8_27b_dflash2_real_test 15 1 1 1` ok (accepted=20/35) and `7 1 1 1` ok (20/20);
  `ninfer_qwen3_6_27b_ngram_real_test` on this artifact with `rk4v4-e8` reports 60 wide rounds of
  645 and `ngram lossless: PASS` (its usual quantized-KV report; the bf16 fixture is not present in
  this container).
- Full `ctest` 133/120/13/0 twice; `git diff --check` clean.

Session-21 item (c) is closed, and with it the session-21 next-step list.

### MTP n-gram chain on by default (2026-10-09, session 29)

**Decision.** The host n-gram chain (session 7 port, session 19 wide window) is now the default for
`--spec mtp`: `NgramOptions::mode = Chain`, `--ngram off` opts out. The model is that the chain
extends MTP proposals only; every other speculative backend clears the options at parse time
(`product::normalize_speculative_options`, called by the CLI, serve and bench parsers before
validation) so request logs, metrics and the engine plan report the effective mode. The engine
planning normalizes again for SDK callers (`layouts_impl.h` clears a non-MTP `ngram` and validates
the ranges only for MTP); the pool is no longer allocated for other backends.

**Why it is the default.** Structured and repetitive generation is a common user workload, and the
chain is free when it cannot help: a round widens to the 9..16 window only when some row's pool
extension reaches `draft_tokens + 3`, and the session-28 wide tile cut that round's cost ~31%.

- `scenario_structured_jsonl.json`, greedy, `rk4v4-e8`, 256 tokens: MTP3 (chain, default) 166.9
  tok/s in 57 rounds (6 wide, 44 n-gram-accepted) against 145.8 tok/s in 67 rounds with
  `--ngram off`; output hash unchanged. `scenario_code_python.json` is a no-op (0 wide rounds,
  113.8 vs 114.1 tok/s) because the answer has no long suffix repeat.
- Tiled-corpus depth sweep (`ninfer_bench`, `rk4v4-e8`, `--prefill-chunk 1024`, one warmup and one
  repetition; logs `profiles/bench/5c-ngram-default/`): MTP3 361.0/384.7/332.0/302.0 tok/s at
  8K/32K/64K/98K against 150.5/141.7/130.3/122.2 with `--ngram off`; DFlash2 K=7
  168.1/264.6/241.1/213.5. The repeated corpus amplifies the chain by construction, so the README
  and model-card tables now carry the chain column and the `--ngram off` control.
- Memory (RTX 4080, 100K + vision + MTP3, `--host-kv-mib 4096`): the wide graph family adds
  118 MB to the runtime reservation (2.747 -> 2.865 GB) and takes 53 MB of the after-startup slack
  (922 -> 867 MB, ~827 MiB free); the profile still validates and listens.
- Tooling: `ninfer_bench` gained `--ngram off|chain` and reports the mode in its config line;
  schema v15 adds `verify_window`, `wide_rounds`, `ngram_drafted_tokens` and
  `ngram_accepted_tokens` to each repetition (the C++ support test, the matrix tool and
  `bench/README.md` moved to v15).
- Tests: CLI/serve option suites cover the MTP default, the `--ngram off` opt-out and the
  non-MTP normalization; the full `ctest` is 133/120/13/0.

**DFlash2 comparison on the same build** (greedy): structured JSONL 272.6 tok/s at K=15 and 239.4
at K=7 against MTP3's 166.9 — DFlash2 is still the faster backend where it fits, and K=15's output
matches MTP3's while K=7 takes a different valid branch (the unspecified `tags` values), the
session-15 trajectory class. On the one-shot code scenario DFlash2 K=7 stays ahead (136.5 vs
113.8) and K=15 is behind (109.8). The chain's value is closing part of the structured gap with no
extra weight memory and no context penalty.

**Launcher pin.** The pull-first 4080 launchers (`scripts/run-ninfer-4080.{sh,bat}`) and the
documented container invocations in `README.md` and the model card pass `--ngram chain` explicitly:
the published image predates the default and is served as pulled (no `org.ninfer.revision` label, so
the launcher never rebuilds it), while the flag is valid on both the published and the current
binary. The DFlash2 launchers stay untouched because the chain is MTP-only and the published
binary rejects the flag with `--spec dflash2`. The launcher headers now carry the session-29 depth
sweep instead of the pre-chain numbers.

**Image republished (2026-10-09).** `roofkid/ninfer-4080:0.6.2-rtx4080` (digest
`sha256:bfbf286f429651295ed73feb8ef4f52a7bac30237d995c7e8dfeb37bb79867f9`) and the refreshed
`gsq3` tag carry the session-28/29 binaries; the artifact is unchanged (`c6f2707...`), so the
Hugging Face refresh was the card and manifest only. The launchers' explicit `--ngram chain` stays
valid on both image revisions. README and port ledger now name the new tag.
