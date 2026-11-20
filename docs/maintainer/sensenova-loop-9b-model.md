# SenseNova-Loop-9B Model Reference

This reference records the delivered Text, loop, mHC, YaRN, MTP, Vision, and persistent-state
semantics for the registered `sensenova-loop-9b/groupwise-int` target on `sm_89`. Fixed
dimensions are encoded in `src/targets/sensenova_loop_9b/impl/config.h`; artifact assignment and
binding are defined in [`sensenova-loop-9b-artifact.md`](sensenova-loop-9b-artifact.md).

## 1. Identity

| Field | Value |
|---|---|
| NInfer model / weights / target | `sensenova-loop-9b` / `groupwise-int` / `sensenova_loop_9b` |
| Source | `sensenova/SenseNova-Loop-9B` @ `3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6` |
| Architectures | `SenseNovaLoopForConditionalGeneration`, `qwen3_5_text`, `qwen3_5_vision` |
| Modalities | Text and image; video is not advertised |
| License / access | release repository is gated, account access required |

The release `modeling_sensenova_loop.py` (SHA-256
`26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b`) is the authority for
loop and mHC semantics. The Preview repository is not release semantics: its config uses default
RoPE and different position limits.

## 2. Text decoder

| Field | Value |
|---|---:|
| hidden size | 4096 |
| physical layers | 32 |
| execution slots | 56 |
| intermediate size | 12288 |
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| full-attention interval | 4 |
| physical full-attention layers | 8 (`3,7,11,15,19,23,27,31`) |
| physical GDN layers | 24 |
| execution full-attention slots | 14 |
| execution GDN slots | 42 |
| RMSNorm epsilon | `1e-6` |
| native context | 524288 (launch profile validated to 262144) |
| MTP layers | 1 |

Layer dimensions:

| Component | Value |
|---|---:|
| query / KV heads | 16 / 4 |
| attention head width | 256 |
| Q / K / V widths | 4096 / 1024 / 1024 |
| rotated dimensions per head | 64 |
| GDN Q/K heads × width | 16 × 128 = 2048 |
| GDN V heads × width | 32 × 128 = 4096 |
| GDN convolution channels / taps | 8192 / 4 |
| GDN A/B gate width | 32 each |

Full attention uses zero-centered `(1+w)` Q/K RMSNorm, partial NeoX MRoPE with sections
`[11,11,10]`, an output gate, and scale `1/sqrt(256)`. Every layer has a dense SwiGLU MLP.

### 2.1 Loop execution plan

The execution order is:

```text
0, 1, 2, 3,
4, 5, ..., 27,
4, 5, ..., 27,
28, 29, 30, 31
```

The plan is compile-time data in `qwen3_6::ExecutionPlan`. Physical weights are materialized once
and indexed by physical layer. KV planes, GDN recurrent/convolution state, replay records,
checkpoints, graph state, and checkpoint provenance are indexed by execution slot. Visit 1 and
visit 2 of a loop-block layer share weights but never share state. Repo tests cover the mapping,
state counts, and the `slot 4` / `slot 28` isolation.

### 2.2 mHC residual streams

Physical layers `4..27` are multi-stream hyper-connection (mHC) layers on both visits. Each has
two FP32 routers (`mhc_attn` before the mixer and `mhc_ffn` before the MLP) with:

```text
weight FP32 [8, 8192]
bias   FP32 [8]
streams = 2, iterations = 20, fixed_identity = false, pre_scale = 1.0
```

For residual `[T,2,H]` the router evaluates normalization, eight logits, sigmoid `pre`, doubled
sigmoid `post`, and a 20-iteration log-space Sinkhorn `mix` in FP32. `pre` and `mix` are rounded to
BF16 before their matmuls and `post` is rounded to BF16 before the merge update. The merge is
`mix @ residual + post[:,None] * branch[:,None,:]` with the residual mix rounded to BF16 before the
additive update.

`enter` duplicates the single stream into two identical streams at physical layer 4; `leave`
averages the streams at physical layer 27. The reference treats the intervening representation as
a flattened `[T,2H]` residual; the kernel stores the equivalent `[2H,T]` feature-major tensor.
`src/ops/mhc/` owns `mhc_router`, `mhc_merge`, `mhc_enter`, and `mhc_leave`; `tests/ops/test_mhc.cpp`
qualifies them against an independent FP64 oracle at T=1..1024.

### 2.3 YaRN RoPE

The release config requires YaRN at every position:

```text
rope_type = yarn, factor = 2, original_max_position_embeddings = 262144,
beta_fast = 32, beta_slow = 1, attention_factor = 0.1*ln(2)+1 = 1.069314718...
```

The scaled inverse frequencies are applied to all three MRoPE axes. The YaRN path evaluates the
frequency ramp and phase in FP64 so the 262144/524288 positions remain accurate; cos/sin are then
multiplied by the attention factor. Unscaled Qwen variants keep the existing RoPE path unchanged.
`tests/ops/test_rope.cpp` covers positions `0, 8192, 262143, 262144, 524281, 524287` for both
3-axis MRoPE and 1-D text.

## 3. MTP

The MTP is a separate checkpoint (`mtp/` in the AWQ release) with one full-attention dense layer, hidden 4096,
FFN 12288, 16 Q / 4 KV, head width 256, partial MRoPE, and `mtp_use_dedicated_embeddings=false`.
The stem is `x = fc([RMSNorm(embedding), RMSNorm(target_hidden)])`. The release MTP has no
optimized shortlist head; target verification uses the full text `lm_head`. The separate
`mtp.norm.weight` is the draft final norm. The AWQ MTP carries an extra `main_norm` tensor; it is
not consumed.

The Engine path is MTP3 plus the model-agnostic n-gram chain. DFlash2 is not registered because no
official SenseNova draft exists.

## 4. Vision

The release Vision tower is 27 layers × 1152 hidden, 4304 intermediate, 16 heads × 72,
patch 16, temporal 2, spatial merge 2, 2304 positions, GELU-tanh, and a 4608-wide merger whose
output is 4096. Video remains out of scope even though the release config carries a video token.
Vision is not deepstack.

## 5. Frontend

The release template digest is
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. It differs from both Qwen
templates: assistant history after the last query always renders the `<think>` block, and tool
argument serialization stringifies non-container values (Python `True`/`False`/`None`) while
containers remain JSON. The tokenizer adds no BOS token and uses `<|endoftext|>` as pad.
`tokenizer_config.json` has no `added_tokens_decoder`; added tokens come from `tokenizer.json`.

## 6. Measured evidence and limits

Measured on the RTX 4080 (16 GiB, 76 SMs, compute 8.9, driver 621.24) with artifact
`sensenova_loop_9b.ninfer`:

| Check | Result |
|---|---|
| 256K Text, int8 KV | starts; 1.68 GiB free after startup |
| 256K Text + Vision, int8 KV | starts; 1.27 GiB free after startup |
| 256K Text + Vision + MTP3, int8 KV | starts; 0.35 GiB available after startup |
| 253K-token prefill | 1.72k tok/s |
| 3.5K-token prefill | 2.62k–2.65k tok/s |
| 3.5K decode, MTP0 | ~36 tok/s |
| 3.5K decode, MTP3 + n-gram | ~58 tok/s |
| Quick corpus PPL, int8 KV | 8.55, versus 9.15 for the prior BF16-sourced artifact |

The first two capacity checks use the non-speculative profile; adding MTP3 with the n-gram chain
lowers the 256K + Vision startup headroom to about 0.35 GiB. That is the profile the repository
launcher serves, and it stays one lane.

The delivered target does **not** meet the plan's aspirational 80/55 MTP0, 180/120 MTP3, or
4,000/2,000 prefill bars. Those remain performance work, not correctness claims. The 256K decode
depth was not benchmarked end-to-end; 253K-token prefill and 256K startup were. Vision uses the
proven Qwen Q4/Q5/Q6/W8 vision roles rather than BF16 vision weights; this was required to keep the
vision route on qualified kernels and leaves more headroom for the 256K profile.

The 2+2 prompt is sensitive to Q4/Q5 quantization and can echo the prompt where BF16 continues with
`4`; short-prompt token equality is not a quality criterion. The documented quality checks are the
quick-corpus perplexity comparison and the external MBPP run, not exact greedy text.
