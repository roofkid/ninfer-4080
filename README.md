# NInfer-4080

NInfer-4080 runs **Qwen3.8-27B** on one 16 GB NVIDIA GeForce RTX 4080 from a **3-bit GSQ
artifact**, with the full **100,000-token context, vision, and MTP3 speculative decoding**
profile resident at once. It is an `sm_89` port of
[NInfer-4090](https://github.com/sergiuszm/ninfer-4090), which derives from
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090) and
[Neroued/ninfer](https://github.com/Neroued/ninfer), a specialized C++20/CUDA inference engine.

This fork registers the `Q3G128_F16S` 3-bit weight scheme, the `gsq3` weights profile, and a
verbatim repack of the [ISTA-DASLab Qwen3.8-27B 3-bit GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ)
checkpoint, because no registered Q4-or-wider allocation fits a 16 GB card at 100K context. The
engine itself is inherited: paged KV, compatible-prefix reuse, CUDA Graphs, MTP and DFlash2
speculative decoding, reasoning-effort control, the OpenAI/Anthropic-compatible APIs, and the
ReplaySSM state transactions all work as documented in [docs/](docs/).

## Three commands

The published artifact is [roofkid/Qwen3.8-27B-GSQ3-NInfer](https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer)
(12.4 GiB, SHA-256 verified). The published image carries only the binaries; the artifact stays a
mounted file.

```bash
# 1. Download the artifact (resumable, SHA-256 checked).
./scripts/download-qwen38-gsq3.sh          # .bat on Windows

# 2. Start the server (published image from Docker Hub).
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

# 3. Serve a request.
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

`scripts/run-ninfer-4080.sh` (or `.bat`) runs the same image and profile in one step. It pulls the
image, uses the artifact from `out/` or `models/`, forces a source build with `NINFER_BUILD=1`, and
serves on `NINFER_BIND:NINFER_PORT` (0.0.0.0:8080 by default). The image is `roofkid/ninfer-4080`
on Docker Hub, tagged `gsq3` and `0.6.1-rtx4080`.

Building from source instead: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-DNINFER_BUILD_APPS=ON` then `cmake --build build --parallel`; the build requires CUDA 13.1, a
recent driver, and only accepts `CMAKE_CUDA_ARCHITECTURES=89`.

## Measured results on the RTX 4080

Conditions: single request, `rk4v4-e8` KV, `--prefill-chunk 1024`, the 131,072-token tiled corpus,
one warmup and one measured repetition per point. Acceptance is a tiled-corpus fixture property
(repeated text), not a model result.

| Depth | Prefill t/s | MTP3 decode t/s | DFlash2 K=7 decode t/s |
|---:|---:|---:|---:|
| 8K | 2,719.9 | 151.2 | 166.7 |
| 32K | 2,424.9 | 141.7 | 262.3 |
| 64K | 2,125.5 | 130.7 | 239.1 |
| 98K | 1,895.1 | 122.3 | 212.7 |

At the documented 100K prefill profile (`--prefill-chunk 2688`) the same build measures
**1,971.4 tok/s**, and 2,470.0 tok/s at 32,768 tokens. Against the llama.cpp IQ3_S reference on
this card (1,043 tok/s at 100K), the fork's 100K prefill now leads, and decode leads at every
depth. Raw session logs and method: the port's working record is
[docs/maintainer/rtx-4080-plan.md](docs/maintainer/rtx-4080-plan.md) section 11 while the port is
active; the same numbers are repeated in the model card.

## Profiles and fit

- **100K + vision + MTP3** (`--max-context 102400`, `--host-kv-mib 4096`, `rk4v4-e8`): about
  11.2 GiB of device weights; the KV and runtime reservation validate before the server listens,
  leaving roughly 0.9 GiB free.
- **DFlash2 K=7**: validated at 100,000 tokens text-only and 65,536 tokens with vision at the same
  safety margin (`--spec dflash2 --draft-tokens 7 --lm-head-draft`).
- KV modes: `rk4v4-e8` is the 100K accuracy profile. `int8` and `rk8v4` trade context for
  accuracy and fit below 100K. The pinned host pools hold one deep checkpoint so rewrites reuse
  the prefix instead of re-prefilling.
- The official 16.96 GiB `groupwise-int` artifact does not load on this card; the 3-bit GSQ3
  artifact is the registered 4080 fit.

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

## Upstream and credits

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

NInfer is a personal project developed out of interest. If you find it useful and would like to
support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
