# NInfer-4080

**Qwen3.8-27B on a single 16 GB RTX 4080, at 100K context, with vision and speculative decoding.**

NInfer-4080 runs the
[ISTA-DASLab Qwen3.8-27B 3-bit GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ)
checkpoint on one NVIDIA GeForce RTX 4080 using more of the hardware than the general-purpose
engines do: up to **2,754 tok/s prefill** and **385 tok/s generation** are measured on this card
(sweep below), with the full **100,000-token context, vision, and MTP3/DFlash2 speculative
decoding** profiles resident at once. The artifact and a binaries-only container image are
published, so there is nothing to convert before you run it.

It is an `sm_89` port of [NInfer-4090](https://github.com/sergiuszm/ninfer-4090), which derives
from [NInfer-3090](https://github.com/Don-Chad/ninfer-3090) and
[Neroued/ninfer](https://github.com/Neroued/ninfer), a specialized C++20/CUDA inference engine.
This fork registers the `Q3G128_F16S` 3-bit weight scheme and the `gsq3` weights profile because
no registered Q4-or-wider allocation fits a 16 GB card at 100K context. The engine itself is
inherited: paged KV, compatible-prefix reuse, CUDA Graphs, speculative decoding, reasoning-effort
control, the OpenAI/Anthropic-compatible APIs, and the ReplaySSM state transactions all work as
documented in [docs/](docs/).

## TL;DR

- Runs ISTA-DASLab's 3-bit GSQ Qwen3.8-27B at 100K context on one RTX 4080 16 GB.
- Up to 2,754 tok/s prefill and 385 tok/s generation, measured on the card.
- 3.125 bpw Text body repacked verbatim from the publisher; Q4 vocabulary endpoints; DFlash2
  companion requantized to Q4 and enabled.
- Artifact: [roofkid/Qwen3.8-27B-GSQ3-NInfer](https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer)
  (12.4 GiB). Image: `roofkid/ninfer-4080:gsq3` (Docker Hub, tags `gsq3` and `0.6.1-rtx4080`).
- Three commands: download, `docker run`, serve.

## How to use it

Requirements: an RTX 4080 (16 GB, `sm_89`) with a CUDA 13.1-or-newer driver, and Docker with the
NVIDIA Container Toolkit. A source build needs CUDA 13.1 and only accepts
`CMAKE_CUDA_ARCHITECTURES=89`.

### 1. Download the artifact

```bash
./scripts/download-qwen38-gsq3.sh          # .bat on Windows
```

The script fetches `qwen3_8_27b_gsq3.ninfer` from the pinned Hugging Face revision, resumes an
interrupted download, and verifies the SHA-256. The artifact lands in `models/`.

### 2. Start the server

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v "$PWD/models/qwen3_8_27b_gsq3.ninfer:/models/model.ninfer:ro" \
  roofkid/ninfer-4080:gsq3 \
  ninfer-serve /models/model.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 \
  --host-kv-mib 4096 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

`scripts/run-ninfer-4080.sh` (or `.bat`) does the same in one step and pulls the image for you.
`scripts/run-ninfer-4080-dflash2.{sh,bat}` serves the DFlash2 K=7 profile instead. Both pick the
artifact up from `out/` or `models/`, accept `NINFER_IMAGE`, `NINFER_PORT`, `NINFER_BIND`,
`NINFER_KV_DTYPE`, `NINFER_CONTEXT`, and `NINFER_API_KEY`, and fall back to a local source build
when the image cannot be pulled (force one with `NINFER_BUILD=1`).

### 3. Serve a request

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

The server speaks OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages; see
[HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md). For the CLI without Docker, build
from source (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON`,
then `cmake --build build --parallel`) and run `./build/apps/ninfer models/qwen3_8_27b_gsq3.ninfer
--prompt "..." --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft`.

## Measured results on the RTX 4080

Conditions: single request, `rk4v4-e8` KV, `--prefill-chunk 1024`, the 131,072-token tiled corpus,
one warmup and one measured repetition per point. Acceptance is a tiled-corpus fixture property
(repeated text), not a model result; the MTP3 column is the default profile including the n-gram
chain, which the repeated corpus amplifies, and `--ngram off` is its control.

| Depth | Prefill t/s | MTP3 decode t/s | MTP3 `--ngram off` t/s | DFlash2 K=7 decode t/s |
|---:|---:|---:|---:|---:|
| 8K | 2,754.2 | 361.8 | 150.5 | 167.6 |
| 32K | 2,460.0 | 385.1 | 141.7 | 264.4 |
| 64K | 2,152.0 | 332.8 | 130.3 | 241.3 |
| 98K | 1,917.2 | 302.9 | 122.2 | 213.2 |

At the documented 100K prefill profile (`--prefill-chunk 2688`) the same build measures
**1,971.4 tok/s**, and 2,470.0 tok/s at 32,768 tokens. Against the llama.cpp IQ3_S reference on
this card (1,043 tok/s at 100K), the fork's 100K prefill now leads, and decode leads at every
depth. In real use the maintainer sees the 2K+ prefill rates on long prompts and roughly 150-200
tok/s decode on coding, about 100 tok/s on prose. Raw session logs and method: the port's working
record is [docs/maintainer/rtx-4080-plan.md](docs/maintainer/rtx-4080-plan.md) section 11 while
the port is active; the same numbers are repeated in the model card.

## Profiles and fit

- **100K + vision + MTP3** (`--max-context 102400`, `--host-kv-mib 4096`, `rk4v4-e8`): about
  11.2 GiB of device weights; the KV and runtime reservation validate before the server listens,
  leaving roughly 0.8 GiB free. The n-gram chain is on by default for MTP3 (`--ngram off` disables
  it).
- **DFlash2 K=7**: validated at 100,000 tokens text-only and 65,536 tokens with vision at the same
  safety margin (`--spec dflash2 --draft-tokens 7 --lm-head-draft`).
- KV modes: `rk4v4-e8` is the 100K accuracy profile. `int8` and `rk8v4` trade context for
  accuracy and fit below 100K. The pinned host pools hold one deep checkpoint so rewrites reuse
  the prefix instead of re-prefilling.
- The official 16.96 GiB `groupwise-int` artifact does not load on this card; the 3-bit GSQ3
  artifact is the registered 4080 fit.

## Quality

The maintainer's own runs keep MBPP in the 90-92% range and HumanEval in the 95-96% range. The
weight-quality anchor on the fork's 1M-token corpus is 4.596095 quick perplexity (INT8 KV),
identical to the value recorded at conversion, and an apples-to-apples MBPP run put the artifact
at 90% against 90-92% for the beellama `kvarn5/5`/`kvarn4/4` reference (within noise). Small
deviations within measurement noise are possible, mainly from KV quantization: `rk4v4-e8` is the
validated 100K profile, and `int8` or `rk8v4` trade context for accuracy if you want the
conservative option.

## Background

After watching the community build NInfer for the 5090, 4090, and 3090, there was still nothing
for a 16 GB RTX 4080. This fork is that missing piece. It was built as a pet project by the
maintainer, who has 20+ years of software engineering and architecture experience but no GPU
kernel background: the work was approached from a requirements and product-owner perspective,
with a focus on engineering practices, measurable changes, and business decisions rather than
hand-judging kernel code. The guiding principles were: fit the 16 GB card; start from the GSQ
checkpoint the maintainer already used daily (its reasoning quality is discussed in the
[ByteShape GSQ article](https://byteshape.com/blogs/Qwen3.8-27B/#96-gb-rtx-pro-6000)); use
DFlash2 speculative decoding; reach 100K+ context; beat the general-purpose engines on prefill
and decode; and re-measure quality after every change. Agent tooling drove the implementation
(Pi as the harness with DeepSeek V4.1 Flash), at a total model spend of about $13.

## What I learned

- General-purpose engines leave a lot of performance on the table. Seeing memory throughput
  measurements in the 200 GB/s range against the card's 720 GB/s theoretical maximum was the
  moment the project's real headroom became obvious.
- Specialized inference engines are a lasting trend (vllm-radiance for R9700, the NInfer
  variants for CUDA, Splash for Metal), and with software creation getting cheaper there will be
  more of them.
- The whole port cost roughly 2 billion tokens for about $13 of off-hours agent work. Cheap
  cache-read pricing matters far more than expected; the electricity to produce the same tokens
  locally can cost more than the API bill did.
- Prefill speed changes the day-to-day experience completely. Once the prefill gains landed,
  prompts streamed immediately instead of waiting 5-10 seconds for an uncached system prompt,
  and a 100K context window fills in about 40 seconds.
- KV compression has come a long way. The old fear of "high" 4-bit-style compression was
  unfounded: after benchmarking, `rk4v4-e8` here is indistinguishable from kvarn5/5 for the
  maintainer, with no "fast garbage" effect, and it is what makes 100K fit. At these speeds the
  maintainer went back to `xhigh` thinking.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_gsq3.ninfer` |
| Size | 13,330,776,576 bytes (12.41 GiB) |
| SHA-256 | `c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7` |
| NInfer identity | `qwen3.8-27b` / `gsq3` / `qwen3_8_27b` |
| Source | `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` @ `b5ce0b76`, Apache-2.0 |

The 323 packed matrices from the publisher's checkpoint (the 320 Text-body matrices, the token
embedding, the full output head, and the draft head) are a **lossless repack**: every code and
group scale is copied unchanged, with only the publisher's unsigned `code + 4` bit-plane layout
inverted into two's-complement fields. The only value deviation is 218 of 240,271,360 bf16 scales
that are not binary16-exact; all are subnormals rounded with an error bounded by `2**-25` (max
`2.98e-8`). The publisher's task evaluations describe the represented Text and vocabulary weights.
The MTP layer, Vision tower, and DFlash2 companion come from the official BF16 sources and are
quantized by the converter. The full model card, provenance table, and verification evidence are
in [model-cards/Qwen3.8-27B-GSQ3-NInfer](model-cards/Qwen3.8-27B-GSQ3-NInfer/README.md).

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7' \
  'qwen3_8_27b_gsq3.ninfer' | sha256sum --check
```

## What this fork adds

- **The 3-bit GSQ artifact and its routes.** `Q3G128_F16S` (symmetric 3-bit codes, one FP16 scale
  per 128 values, 3.125 bpw) in the artifact codec, the converter/verifier, the C++ binder, and
  the `Qwen38Gsq3` profile, plus Q3 execution leaves for every Text site.
- **Q3 prefill routes**: the pipelined tall A16 GEMM, the A8 int8 tensor-core route from 129
  columns, and a fused SwiGLU epilogue (pp32768 1,015 → 2,470 tok/s across sessions 6/9).
- **Q3 decode routes**: the staged small-T GEMV, the small-T bf16 tensor-core route for 2..16
  columns, an A8 profile of it, the direct-register A-fragment consume cut, and the prompt
  attention worker-V-dequant pipeline (`pp100000` 1,710 → 1,971 tok/s; decode +12% at K=7).
- **4080 platform enablement**: the GDN gating-projection cooperative budgets now derive from the
  runtime SM count (76 on this card) instead of the 4090's 128, and the residency tables are
  shared with the launcher instead of duplicated.
- **DFlash2 companion** requantized to `Q4G64_F16S` and bound in the `gsq3` identity, which
  raised the K=7 caps from 57K to 100,000 tokens text-only.
- **Launchers and this publication**: `scripts/run-ninfer-4080.sh`, `.bat`, the artifact download
  scripts, and the published image.

## Known limits on the RTX 4080

- One process, one GPU, one resident model, bounded FIFO admission, no preemption, and no
  multi-GPU execution. The 100K profile runs one lane.
- The artifact is accepted only by this fork (`weights_id = gsq3` is not registered upstream).
- CUDA 13.1-or-newer userspace; the image is built for `sm_89` only.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Reasoning effort

Qwen3.8-27B has three trained reasoning depths plus an off switch. OpenAI Chat Completions
accepts a top-level `reasoning_effort` field (`low`, `medium`, `xhigh`) and a top-level
`enable_thinking` boolean; hidden reasoning returns separately as `message.reasoning_content`.
Only those three levels are accepted, plus `none` to turn thinking off. `high`, `minimal`, and
`max` are rejected as `reasoning_effort_not_supported`, so a client that offers a `high` setting
must map it to `xhigh`. A token budget for reasoning is separate from the effort level. It is
set only through the Anthropic Messages path (`thinking.budget_tokens`) or server-wide with
`--default-thinking-budget N`; the OpenAI paths have no field for it. Without a budget,
reasoning is bounded only by the request's `max_tokens`, which is what the model card
recommends, and the `model_thinking_tokens` field of the request JSONL reads zero, because that
counter runs only under a budget. The `chat_template_kwargs` request field of llama.cpp is not
supported and is rejected. For the CLI, pass `--reasoning-effort` or `--no-thinking`. Sampling
defaults come from the model card and switch with the thinking mode: `temperature=1.0`,
`top_p=0.95`, `top_k=20` in thinking mode; `temperature=0.7`, `top_p=0.80`, `top_k=20`,
`presence_penalty=1.5` in non-thinking mode.

## Serving APIs

OpenAI Chat Completions, OpenAI Responses with streaming and local continuation state, Anthropic
Messages, prompt-rendered function tools with parsed tool calls, compatible-prefix reuse, and
JSONL request logs. See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## Community

If you have another 16 GB RTX 4xxx card (4070 Ti Super, 4060 Ti 16 GB, 4080 Super, ...) I would
like to know whether this works there and what speeds you see, because it is hard to judge how
tied to the 4080 the tuning is. Issues and measurement reports are welcome on the fork's tracker.
If you have a 4080, enjoy.

## Upstream and credits

Thanks to everyone who worked on NInfer before this fork; you provided a stable base. A special
shout-out to [sergiuszm](https://github.com/sergiuszm/ninfer-4090) for the NInfer-4090 SM_89
heavy lifting, and to ISTA-DASLab for GSQ - cheers to Austria from Germany.

- [Neroued/ninfer](https://github.com/Neroued/ninfer) - the engine, developed for the RTX 5090
  (`sm_120a`), Apache-2.0.
- [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) - the RTX 4090 fork this
  branch descends from on `rtx4080-port`.
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - the SM86 compatibility layer,
  ReplaySSM integration, and Qwen3.8 runtime support.
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) - the rotated and
  E8-lattice KV-cache modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`) and the E8 codecs, merged
  with authorship preserved.
- [ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ) -
  the 3-bit quantized checkpoint the artifact repacks, Apache-2.0.
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) - the base model (MTP, vision
  calibration, tokenizer), Apache-2.0.
- [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) - the DFlash2
  draft companion, Apache-2.0.

## Support

If this fork is useful to you and you would like to support its continued development, you can
[buy me a coffee](https://buymeacoffee.com/roofkid). The upstream engine project also accepts
support at [ko-fi.com/neroued](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
