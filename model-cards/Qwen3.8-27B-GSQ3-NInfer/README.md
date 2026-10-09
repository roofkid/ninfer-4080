---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-4080
  - 3-bit
  - gsq
---

# Qwen3.8-27B GSQ3 for NInfer (RTX 4080)

This model card is the version-controlled source for
[roofkid/Qwen3.8-27B-GSQ3-NInfer](https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer).

The repository contains a verbatim repack of the 3-bit
[ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ)
checkpoint into the native [NInfer](https://github.com/roofkid/ninfer-4080) `.ninfer` artifact
format, sized to run the full 100,000-token, vision, and speculative-decoding profile on a single
16 GB **RTX 4080**. It is intended only for the RTX 4080 NInfer fork; it is not a Transformers
checkpoint, Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_gsq3.ninfer` |
| Size | 13,330,776,576 bytes (12.41 GiB) |
| SHA-256 | `c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `gsq3` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,190 (1,184 tensors and 6 resources) |

The Text body is 320 matrices in the registered `Q3G128_F16S` scheme: symmetric 3-bit codes in
`[-4, 3]`, one FP16 multiplier per 128-value group, 3.125 bits per weight. The token embedding,
full output head, and optimized draft head use `Q4G64_F16S`; the MTP layer and Vision tower follow
NInfer's registered `groupwise-int` recipes; the DFlash2 companion was requantized to
`Q4G64_F16S`. The file also carries the
tokenizer, chat-template, generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7' \
  'qwen3_8_27b_gsq3.ninfer' | sha256sum --check
```

## What "verbatim" covers

The 323 packed matrices that come from the publisher's checkpoint — the 320 Text-body matrices,
the token embedding, the full output head, and the draft head (a row gather of the output head) —
are a lossless repack: every code and every group scale was copied unchanged, with only the
publisher's shifted bit-plane transform inverted (`compressed-tensors` stores the unsigned
`code + 2^(bits-1)` form; the artifact stores two's-complement fields). No code was requantized.
The only value deviation from the source is **218 of 240,271,360** multipliers: bf16 subnormal
words whose FP16 rounding error is at most `2**-25` (max observed `2.98e-8`), audited per group in
the conversion report. The publisher's task evaluations therefore describe the represented Text
and vocabulary weights.

The MTP layer (official BF16 checkpoint), the Vision tower (BF16 in the GSQ release), and the
DFlash2 companion (BF16 from `z-lab`) are quantized by NInfer's converter, so the verbatim claim
does not extend to them.

## Requirements

- The [RTX 4080 fork](https://github.com/roofkid/ninfer-4080) of NInfer, branch `rtx4080-port`,
  built from source (`sm_89`; the fork's CMake accepts only `CMAKE_CUDA_ARCHITECTURES=89`), or
  the published container image below. Upstream NInfer and the RTX 3090/4090 forks do not
  register the `gsq3` weights profile or the `Q3G128_F16S` scheme and reject this file.
- NVIDIA GeForce RTX 4080 (16 GB, `sm_89`);
- 64-bit Linux with a CUDA 13.1-or-newer driver (the container image carries the userspace CUDA
  13.1 runtime);
- the launcher profile uses the `rk4v4-e8` KV cache, which is what makes the 100K profile fit.

## Download and run

```bash
hf download roofkid/Qwen3.8-27B-GSQ3-NInfer qwen3_8_27b_gsq3.ninfer --local-dir models
```

The turnkey profile is the `rtx4080-port` fork's `scripts/run-ninfer-4080.sh` (or `.bat`), which
serves 100,000 tokens with vision and MTP3. The equivalent container invocation is:

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

Without Docker, the CLI is:

```bash
./build/apps/ninfer models/qwen3_8_27b_gsq3.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 --max-new 8192 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Performance at 100K on the RTX 4080

Measured with `ninfer_bench` on the current `rtx4080-port` build, `rk4v4-e8`,
`--prefill-chunk 1024`, the fork's 131,072-token tiled corpus, one warmup and one measured
repetition per point. Acceptance is a tiled-corpus fixture property (repeated text), not a
model result; the MTP3 column is the default profile including the n-gram chain, which the
repeated corpus amplifies, and `--ngram off` is its control.

| Depth | Prefill t/s | MTP3 decode t/s | MTP3 `--ngram off` t/s | DFlash2 K=7 decode t/s |
|---:|---:|---:|---:|---:|
| 8K | 2,754.2 | 361.8 | 150.5 | 167.6 |
| 32K | 2,460.0 | 385.1 | 141.7 | 264.4 |
| 64K | 2,152.0 | 332.8 | 130.3 | 241.3 |
| 98K | 1,917.2 | 302.9 | 122.2 | 213.2 |

At the documented 100K prefill profile (`--prefill-chunk 2688`) the same build measures
**1,971.4 tok/s**, and 2,470.0 tok/s at 32,768 tokens. MTP3 and DFlash2 accept 3 and 7 draft
tokens respectively; the DFlash2 K=7 profile is greedy-lossless against the engine's own routes
in the fork's real-artifact tests, and the MTP3 route is covered by the same suite.

## Memory profile

- **100K + vision + MTP3** (`--max-context 102400`, `--host-kv-mib 4096`, `rk4v4-e8`): about
  11 GiB of device weights, KV plus runtime reservation validated before the server starts
  listening, with roughly 0.8 GiB of headroom after startup. The n-gram chain is on by default
  for MTP3 (`--ngram off` disables it).
- **DFlash2 K=7**: validated at 100,000 tokens text-only and 65,536 tokens with vision at the
  same safety margin.
- The pinned host pools (`--host-kv-mib 4096`, default host state slots) hold a deep 100K
  checkpoint so rewrites reuse the prefix instead of re-prefilling.

Accuracy profiles that need more KV bytes (`int8`, `rk8v4`) fit at shorter contexts; `rk4v4-e8`
is the accuracy profile validated at 100K.

## Quality anchors

- Quick perplexity on the fork's 1M-token corpus with INT8 KV: **4.596095** overall, identical
  to the value recorded when the artifact was converted.
- MBPP with greedy decoding and the same harness as the llama.cpp reference: **90%** at
  `rk4v4-e8`, against 90–92% for the beellama `kvarn5/5`/`kvarn4/4` reference (within noise).
- The publisher evaluated the represented values as task-lossless (the GSQ model card's AIME
  2025 / GPQA-Diamond figures apply).

## Limits

- The artifact is accepted only by the RTX 4080 fork at the runtime revision listed below; the
  file is versioned with that engine and rejects on any other.
- One RTX 4080 and one CUDA device; one resident model; startup-fixed capacities only.
- No preemption, multi-GPU execution, CPU/GPU offload, or large-scale batching; a 100K profile
  runs one lane.
- NInfer does not execute generated tool calls, and context allocation is subject to GPU memory
  and the selected KV-cache type.

## Provenance

| Field | Value |
|---|---|
| Quantized source | `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` |
| Quantized source revision | `b5ce0b76f60020a875dee4f6ec9d934cca4121e4` |
| Base model | `Qwen/Qwen3.8-27B` |
| Base revision (vocabulary, MTP) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| DFlash2 source | `z-lab/Qwen3.8-27B-DFlash2` |
| DFlash2 revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Conversion recipe | `qwen3_8_27b_gsq3-v1` |
| Converter repository | `https://github.com/roofkid/ninfer-4080` |
| Converter revision | `ce85711b5f71296c4027f839284e41acd9a80669` |
| Minimum runtime revision | `db4a7d9ee4c03adb26cf1958463b41c5a59db39b` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The conversion verifier rebuilt every packed plane from the source shards and compared them
word-for-word with the artifact: **323 packed objects, 4,527,104 rows, 240,271,360 groups,
base bytes equal 323/323, scales equal 323/323** (218 rounded subnormals). The artifact
identity, object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer/blob/main/artifact-manifest.json).

## License

This NInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base model and the
[3-bit GSQ checkpoint](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ) are also
licensed under Apache-2.0, and the DFlash2 companion comes from
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (Apache-2.0).
Users remain responsible for complying with the licenses and applicable laws.

If this artifact is useful to you, you can support the maintainer at
[Buy Me a Coffee](https://buymeacoffee.com/roofkid).
