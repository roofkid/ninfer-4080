# SenseNova-Loop-9B RTX 4080 Target Plan

Status: active implementation plan. This is a working record, not a delivered contract.
Replace it with model/artifact references and remove it when the target is complete or abandoned.

Plan date: 2026-11-18

## Implementation status (2026-11-20)

The target is implemented and runs on the RTX 4080. Delivered:

- compile-time 56-slot execution plan with physical-weight/shared-slot state separation;
- mHC router/merge/enter/leave Ops and FP64 oracle tests;
- YaRN RoPE with FP64 frequency/phase evaluation and 0..524287 tests;
- `sensenova_loop_9b` package, registry entry, groupwise-int converter and independent verifier;
- 16/4 attention geometry and 4096/12288 linear profiles for the target;
- 42-slot ReplaySSM fold geometry, MTP binding and MTP3 + n-gram path;
- SenseNova chat-template digest/semantics and tokenizer variants;
- Vision path with the 4096-wide merger.

Measured: 256K Text+int8 KV starts with 1.68 GiB free; 256K+Vision starts with 1.27 GiB free;
253K-token prefill 1.72k tok/s; 3.5K decode ~36 tok/s MTP0 and ~58 tok/s MTP3+n-gram; quick
corpus PPL 8.55 for the canonical AWQ source versus 9.15 for the prior BF16-sourced artifact. The
maintainer MBPP comparison found no regression, so the AWQ source is now the default artifact.

Remaining gaps: the plan's aspirational decode/prefill bars are not met; 256K-depth decode was not
benchmarked (253K prefill was); Vision uses the proven quantized Qwen roles instead of BF16 tower
weights; the full multi-prompt BF16/AWQ quality campaign and 200K vision fallback were not needed
because 256K+Vision fits. Durable semantics are in
[`sensenova-loop-9b-model.md`](sensenova-loop-9b-model.md) and
[`sensenova-loop-9b-artifact.md`](sensenova-loop-9b-artifact.md).


## 0. Fresh-context start here

Read this document first, then verify the environment before changing code.

```bash
cd /work
git status -sb
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version --format=csv,noheader
free -h
```

Expected baseline when this plan was written:

- Branch: `rtx4080-port`
- HEAD: `2469b5752672ad597055af61fe28e8e7ced20056`
- GPU: NVIDIA GeForce RTX 4080, 16,376 MiB, 76 SMs, compute capability 8.9,
  driver 621.24, CUDA toolkit 13.1.115
- CMake accepts only `CMAKE_CUDA_ARCHITECTURES=89`; there is no `sm_120a` build in this checkout
- GPU free memory at plan time: 15,254 MiB (about 14.9 GiB)
- Host: WSL2, Intel i7-14700KF, 28 threads; the user is raising the WSL memory limit to 32 GB
  for the one-time BF16 CPU reference
- HF access: `hf auth whoami` returns `user=roofkid`

Do not assume the 5090/NVFP4 path is active. The sm_89 build excludes NVFP4 kernels and stubs
the NVFP4-valued KV-cache modes. The 4-bit profile here is groupwise-int Q4/Q5/Q6.

Useful project rules:

- Use `rg` instead of `grep`, `fd` instead of `find`.
- Use the selected Python 3.11 interpreter explicitly; check `python3 --version` first.
- Build with `cmake --build build -j` unless resource pressure requires otherwise.
- Do not create commits unless the user asks.

## 1. Objective

Add a registered NInfer target for `sensenova/SenseNova-Loop-9B` to the RTX 4080 fork:

- 4-bit groupwise-int weights converted from the official AWQ checkpoint and re-encoded
- 256K context with q8/int8 KV as the hard text baseline
- 256K with vision and int8 KV as the stretch target
- If 256K + vision + int8 does not fit, reduce vision context to about 200K
- Text + image input, no video
- MTP3 with the n-gram chain first; DFlash2 later only if an official draft appears
- Full Engine parity: CLI, HTTP serving, CausalScoring, context cache/checkpoints, MTP, vision
- Single RTX 4080, one resident model, one lane at the 256K profile

Quality is preferred over the last few hundred MiB. If a profile does not fit, reduce vision
context before changing weight precision or KV precision. If 256K text with q8 does not fit,
stop and report the measured gap; do not silently switch to uniform Q4 or `rk4v4-e8`.

## 2. Decisions settled with the user

| Topic | Decision |
|---|---|
| Deliverable | Production target with tests, benchmarks, docs; not a design-only study |
| Hardware | RTX 4080 16 GB, `sm_89`; no 5090/NVFP4 work |
| Weight source | Official AWQ 4-bit checkpoint, pinned revision |
| Weight format | NInfer groupwise-int Q4/Q5/Q6 mixed artifact; mHC and GDN control scalars in FP32 |
| AWQ | Artifact source; the BF16 release remains the numerical comparison reference; no AWQ runtime format |
| Context | 256K q8 text is the hard baseline; 256K + vision + q8 is best-effort |
| Vision fallback | Reduce vision context to ~200K, keeping q8 and Q4/Q5 weights |
| Modalities | Text + image; video out of scope |
| Speculation | MTP3 + n-gram chain first; DFlash2 deferred |
| Family | Extend the existing Qwen3.6/Qwen3.5 hybrid family with a compile-time execution plan |
| Parity | Full Engine parity by completion, staged |
| Speed | MTP0 ≥80/≥55 tok/s and MTP3 without n-gram ≥180/≥120 tok/s at 8K/256K; prefill ≥4,000/≥2,000 tok/s; one lane |
| Quality | No worse than the prior BF16-sourced artifact on quick perplexity; task eval within noise |
| Reference | BF16 CPU capture on this WSL host after it reaches 32 GB, plus official AWQ on the 4080 |

The 256K + q8 + vision budget is tight: rough arithmetic is 7.2 GiB of KV, 5.7 GiB of
Q4/Q5/Q6 weights, and roughly 2 GiB of runtime/vision. It may fit only with a very small
headroom. Startup validation decides; the target does not advertise 256K + vision until it has
been measured on the real allocator.

## 3. Pinned model facts

### 3.1 Source identity

Release repository: `sensenova/SenseNova-Loop-9B`

- Revision: `3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6` (pin this; do not use moving `main`)
- Main index total size: 18,832,231,392 bytes
- Main shards: 11 `model-*.safetensors` files
- MTP is a separate `mtp.safetensors`, 486,582,880 bytes; it is not in the main index
- Main index has 857 keys; 96 mHC tensors; 333 `model.visual.*` tensors
- The repository is gated (`gated: auto`); the account `roofkid` has access
- `modeling_sensenova_loop.py` is byte-identical between Preview and release

Resource hashes from the release snapshot:

| Resource | SHA-256 |
|---|---|
| `config.json` | `614dbd44b3a7fba8cd3adf5118b91ee40d51765b874de9dfadf67ea4e16ba68e` |
| `modeling_sensenova_loop.py` | `26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b` |
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `model.safetensors.index.json` | `a478440d3817d761ef6a8088f063131e02197b0d8699262165d4626c5d577510` |
| `preprocessor_config.json` | `3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2` |
| `generation_config.json` | `f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f` |

Preview repository: `sensenova/SenseNova-Loop-9B-Preview`, revision
`89d7970926e215f8da61726700cd026ccffbb6cb`. It is useful for early comparison but its config
uses default RoPE and 262,144 positions, so it is not the release semantics.

Official AWQ: `sensenova/SenseNova-Loop-9B-AWQ`, revision
`8327226ffd2e33e34cb5466a0e6cccd4419bde40`. It is public and includes a separate `mtp/`
directory. The canonical artifact is converted from this checkpoint; the BF16 release remains the
numerical comparison reference.

Do not use `/tmp` as a persistent source location. The reference capture must record the
revision and hashes above. A local source snapshot can live under `/models/SenseNova-Loop-9B` if
the user prefers.

### 3.2 Release config

```text
architectures                 SenseNovaLoopForConditionalGeneration
model_type                    sensenova_loop
text_config.model_type        qwen3_5_text
hidden_size                   4096
intermediate_size             12288
num_hidden_layers             32
full_attention_interval       4
num_attention_heads           16
num_key_value_heads           4
head_dim                      256
attention_bias                false
attn_output_gate              true
linear_conv_kernel_dim        4
linear_key_head_dim           128
linear_value_head_dim         128
linear_num_key_heads          16
linear_num_value_heads        32
vocab_size                    248320
rms_norm_eps                  1e-6
max_position_embeddings       524288
rope_theta                    1e7
partial_rotary_factor         0.25
mrope_interleaved             true
mrope_section                 [11, 11, 10]
rope_type                     yarn
factor                        2.0
original_max_position_embeddings 262144
tie_word_embeddings           false
use_cache                     true
mtp_num_hidden_layers         1
mtp_use_dedicated_embeddings  false
```

Loop and mHC:

```json
"loop_settings": {
  "start": 4,
  "end": 28,
  "counts": [2],
  "eval_count": 2
},
"mhc": {
  "start": 4,
  "end": 28,
  "streams": 2,
  "iterations": 20,
  "fixed_identity": false,
  "pre_scale": 1.0
}
```

Vision:

```text
depth                          27
hidden_size                    1152
intermediate_size              4304
num_heads                      16
head_dim                        72
patch_size                      16
temporal_patch_size             2
spatial_merge_size              2
num_position_embeddings         2304
out_hidden_size                 4096
deepstack_visual_indexes        []
hidden_act                      gelu_pytorch_tanh
```

Special token IDs:

```text
eos_token_id                   248046
pad_token_id                   248044
image_token_id                 248056
video_token_id                 248057
vision_start_token_id          248053
vision_end_token_id            248054
generation eos                 [248044, 248046]
tokenizer-addressable domain   0..248076 (248077 IDs)
```

The release tree does include a `video_preprocessor_config.json`, but the model card advertises
text and images only. Video remains out of scope for this target even though a `video_token_id`
exists.

### 3.3 Execution plan

Physical layer types:

- Full attention: `3, 7, 11, 15, 19, 23, 27, 31`
- Gated DeltaNet: every other physical layer `0..31`
- First entry block: `0, 1, 2, 3`
- Looped block: `4..27` inclusive, 24 physical layers, executed twice
- Exit block: `28, 29, 30, 31`

Execution order:

```text
0, 1, 2, 3,
4, 5, ..., 27,
4, 5, ..., 27,
28, 29, 30, 31
```

That is **56 execution slots**:

- 42 GDN execution slots
- 14 full-attention execution slots
- 14 full-attention KV planes
- 42 GDN recurrent-state slots
- 8 physical full-attention weight sets
- 24 physical GDN weight sets

Every visit has its own cache/recurrent state. Visit 2 of physical layer 4 does not read visit 1's
KV or recurrent state as its own history; it has maintained its own state since the previous token.
The weights are shared, the state is not.

### 3.4 mHC semantics

mHC is active only for physical layers `4..27`, and therefore on both loop visits. Each such
layer has two routers: `mhc_attn` before the mixer and `mhc_ffn` before the MLP. The authoritative
source is the release `modeling_sensenova_loop.py` (hash above); do not rely on the Preview config
for RoPE or execution semantics. The exact math is:

```text
enter(residual, physical):
  if physical == 4:
    residual = hidden.unsqueeze(-2).expand(T, 2, H).contiguous()
  else:
    residual = hidden.unflatten(-1, (2, H))

router(residual):
  flat       = residual.flatten(-2).float()                  # [T, 2H]
  normalized = flat * rsqrt(mean(flat^2, -1, keepdim=True) + 1e-6)
  logits     = normalized @ weight.T + bias                  # [T, 8]
  pre        = sigmoid(logits[..., :2])                      # [T, 2]
  post       = 2 * sigmoid(logits[..., 2:4])                 # [T, 2]
  mix        = sinkhorn(logits[..., 4:].unflatten(-1, (2, 2)), 20)
  branch     = matmul(pre.to(residual.dtype), residual)      # [T, H], BF16
  if pre_scale != 1:
    branch = branch * pre_scale

sinkhorn(logits, 20):
  for 20 iterations:
    logits = logits - logsumexp(logits, dim=-1, keepdim=True)
    logits = logits - logsumexp(logits, dim=-2, keepdim=True)
  return exp(logits)

merge(residual, branch, post, mix):
  mixed   = matmul(mix.to(residual.dtype), residual)         # [T, 2, H], BF16
  update  = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
  return mixed + update

leave(residual, physical):
  if physical == 27:
    return residual.mean(-2)                                 # [T, H]
  return residual.flatten(-2)                                # [T, 2H]
```

Router tensors per router:

```text
weight  FP32  [8, 8192]   (2 * streams + streams * streams = 8)
bias    FP32  [8]
```

There are 24 physical loop layers × 2 routers = 48 physical routers, executed on both visits
(96 router calls per forward). The router weights are not quantized. `fixed_identity=false` and
`streams=2` for this checkpoint; reject other settings rather than silently approximating.

Important rounding boundaries from the reference:

- The normalization and logits are FP32.
- `branch` is a BF16 matmul result using BF16 `pre` and BF16 residual.
- `mix` is cast to BF16 before the residual matmul.
- `post` is cast to the branch dtype before the elementwise update.
- The reference uses `torch.autocast(..., enabled=False)` around this math.

## 4. Current engine baseline

### 4.1 Existing family

The shared family under `src/targets/qwen3_6` already implements the Qwen3.5-style hybrid
decoder: three GDN layers plus one full-attention layer per group, Q/K norm, output gate,
partial MRoPE, dense SwiGLU, GDN recurrent state, vision tower, MTP, and CUDA graph decode.

The target package `src/targets/qwen3_6_27b` currently also owns the Qwen3.8 identity and the
RTX 4080 `Qwen38Gsq3`/`Qwen38GsqRcoIq3S` weight profiles. `src/targets/qwen3_6_35b_a3b` is the
MoE peer.

Relevant facts:

- Family algorithms: `src/targets/qwen3_6/impl/runtime/`, `impl/state/`, `impl/vision/`
- Family public types: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/`
- Target package pattern: `src/targets/qwen3_6_27b/{CMakeLists.txt,export,impl}`
- Registry: `src/targets/registry.cpp`
- CMake target sources: `src/CMakeLists.txt`
- Qwen3.6 model semantics: `docs/maintainer/qwen3.6-27b-model.md`
- Qwen3.6 artifact contracts: `docs/maintainer/qwen3.6-27b-artifact.md`
- 4080 port record: `docs/maintainer/rtx-4080-plan.md`

### 4.2 What already matches SenseNova

- GDN + full-attention 3:1 hybrid topology
- Q/K zero-centered RMSNorm, output gate, partial NeoX RoPE, MRoPE sections `[11, 11, 10]`
- Dense SwiGLU MLP
- Vision depth/hidden/intermediate/heads/patch/merge/position table
- MTP stem shape and one-layer full-attention draft pattern
- Groupwise-int Q4/Q5/Q6, int8 KV, `rk4v4-e8` KV, CUDA graph decode, paged KV
- Frontend tokenizer/thinking/tool-call machinery (with a new template variant)
- CausalScoring, context cache, checkpoint/resource scheduler

### 4.3 What does not match

- 32 physical layers vs the Qwen 64/64/… shapes
- hidden 4096, FFN 12288, 16 Q / 4 KV, 32 GDN V-heads
- 56 execution slots with shared weights and per-visit state
- mHC routers and stream math
- YaRN ×2 on the release config (required at all positions, not only 512K)
- New chat-template digest and small rendering differences
- MTP `main_norm`/source layout needs verification
- 16/4 attention geometry is not registered
- 32-V-head GDN gating/input profiles are not registered
- 4096/12288 linear shapes need 4080 tile selection

### 4.4 Op gaps

Full attention:

- `include/ninfer/ops/softmax_attention.h` documents only `[256,24,4]` and `[256,16,2]`.
- `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp` validates only
  `(24,4)` and `(16,2)`.
- Need `[D=256, Hq=16, Hkv=4]` (group 4) for the target and for MTP target verification.
- Add the geometry to prompt, chunked, small-T, append/attend, and every KV codec used by the
  4080 profile: BF16, `Int8Group64`, `rk4v4-e8` (`RK4V4E8`), and any other enabled codec.

GDN:

- `gdn_gating_proj` registered resident shapes are 48×5120 and 32×2048; add 32×4096.
- `gdn_input_proj` needs split Q4 `query_key [4096,4096]` and Q5 `value_z [8192,4096]`.
  Prefer the split form; it avoids inventing a fused parent shape.
- GDN output `[4096,4096]`, value heads 32, key heads 16, conv channels 8192, state value
  width 4096.
- `gated_delta_net` already documents Hv in `{32,48}`; verify the 32-head path at T=1 and
  prefill widths, with the 4080 76-SM residency.

Linear:

- 4096/12288 matrices are generic but need 4080 tile/schedule measurements. Existing 4080
  profiles were tuned for 5120/17408 and 2048.

Vision:

- The tower is already 27×1152×4304; only the merger output width changes 5120 → 4096.
- Use BF16 vision weights initially for quality. If 256K + vision does not fit, reduce vision
  context before quantizing vision.

## 5. Architecture design

### 5.1 Compile-time execution plan seam

Do not duplicate weights for the second loop. Do not fork the Program. Add a compile-time
execution plan to the family and keep physical weights and execution state separate.

Add a plan type in the family export/config layer, for example
`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/execution_plan.h`:

```cpp
template <int PhysicalLayers, int Slots, int PhysicalFull, int PhysicalGdn>
struct ExecutionPlan {
    static constexpr int physical_layers = PhysicalLayers;
    static constexpr int slots           = Slots;
    static constexpr int physical_full   = PhysicalFull;
    static constexpr int physical_gdn    = PhysicalGdn;

    static constexpr std::array<int, Slots> slot_physical; // compile-time
    static constexpr int slot_physical_layer(int slot);
    static constexpr bool slot_is_full(int slot);
    static constexpr int slot_full_index(int slot);        // full slot -> 0..13
    static constexpr int slot_gdn_index(int slot);         // gdn slot -> 0..41
    static constexpr int physical_full_index(int physical); // weight index
    static constexpr int physical_gdn_index(int physical);  // weight index
};
```

SenseNova plan:

```text
slot  0..3   -> physical 0..3
slot  4..27  -> physical 4..27   (first visit)
slot 28..51  -> physical 4..27   (second visit)
slot 52..55  -> physical 28..31
```

Existing Qwen variants get the identity plan (`slots == physical_layers`). Keep their behavior
and graph shapes unchanged.

Required changes:

1. `TextConfig` must expose separate counts:
   - `physical_layers`
   - `execution_slots`
   - `physical_full_layers`, `physical_gdn_layers`
   - `full_attention_slots`, `gdn_slots`
   - the plan mapping functions above
2. Weight arrays remain sized by physical full/GDN counts. They are bound once per physical
   layer and reused on both visits.
3. State/KV arrays, StateImage, ReplaySSM records, checkpoint copies, and resource projections
   are sized by execution slots, not physical layers.
4. All family layer loops iterate slots, not physical layers. The loop body resolves
   `physical = slot_physical_layer(slot)` for weights and `slot` for state/cache.
5. `TextContext` weight views (`full_`, `gdn_`) remain physical-indexed. State indices use
   `slot_full_index`/`slot_gdn_index`.
6. Full-attention cache layout has 14 logical planes; GDN state has 42 slots. Every visit is
   isolated. Append/attend, MTP verify, prefix capture, and checkpoint restore must use the
   slot index.
7. Graph capture and execution must trace the 56-slot order. A graph profile is a batch/token
   frontier, not a layer count. Verify the graph replays both visits and does not alias state.
8. `CausalScoring` uses the same slot loops and the same temporary state/KV.
9. Add a plan digest (config hash + slot mapping hash) to resident-prefix/checkpoint provenance
   so a state produced by a different plan cannot be reused.

Audit points before editing:

```bash
rg -n "TextConfig::layers|n_layers|full_attention_layers\\(\\)|gdn_layers\\(\\)|full_idx|gdn_idx" \
  src/targets/qwen3_6 src/targets/qwen3_6_27b src/targets/qwen3_6_35b_a3b
rg -n "for \\(.*layer|layer_index|layer_count" src/targets/qwen3_6/impl
```

Do not mechanically change every `layer` variable. Classify each use as one of:

- physical weight index
- execution slot index
- state/cache slot index
- artifact object name

### 5.2 New target package

Follow the package pattern:

```text
src/targets/sensenova_loop_9b/
  CMakeLists.txt
  export/ninfer/targets/sensenova_loop_9b/package.h
  impl/config.h
  impl/load/bindings.h
  impl/load/bindings.cpp
  impl/package.cpp
  impl/variant.h
  impl/variant.cpp
```

Identity:

```text
model_id     sensenova-loop-9b
weights_id   groupwise-int
target_key   sensenova_loop_9b
```

Register in `src/targets/registry.cpp` with the same `construct_registered` pattern as the Qwen
targets. Add `add_subdirectory(targets/sensenova_loop_9b)` in `src/CMakeLists.txt`.

Config facts:

```text
TextConfig::hidden                 4096
TextConfig::intermediate           12288
TextConfig::physical_layers        32
TextConfig::execution_slots        56
TextConfig::gdn_value_heads        32
TextConfig::gdn_key_heads          16
TextConfig::gdn_key_head_dim       128
TextConfig::gdn_value_head_dim     128
TextConfig::gdn_conv_kernel         4
TextConfig::query_heads            16
TextConfig::kv_heads                4
TextConfig::head_dim               256
TextConfig::rotary_dim              64
TextConfig::full_slots             14
TextConfig::gdn_slots              42
TextConfig::output_rows        248320
TextConfig::token_domain       248077
```

Do not copy Qwen's 48 V-head or 24 Q-head constants. The attention scale is `1/sqrt(256)=0.0625`;
the GDN scale is `1/sqrt(128)`.

Package sampling defaults: use the release `generation_config.json` values as the starting point
(`temperature=1.0`, `top_p=0.95`, `do_sample=true`) and record the thinking/non-thinking defaults
from the model card. Do not invent values that are not in the release card; if the card does not
distinguish, use one set and document it.

### 5.3 Artifact and converter

Add a converter under `tools/convert/sensenova_loop_9b/`:

```text
inventory.py     source key/shape inventory and expected object plan
recipe.py        role -> NumericFormat assignment and shapes
convert.py       source shards -> .ninfer
verify.py        independent decode and conversion report
mtp.py           inspect/map mtp.safetensors
```

Artifact identity: `sensenova-loop-9b/groupwise-int`.

Proposed object layout, following the Qwen artifact conventions:

```text
frontend/tokenizer.json
frontend/tokenizer_config.json
frontend/chat_template.jinja
frontend/generation_config.json
frontend/preprocessor_config.json
frontend/video_preprocessor_config.json   # include only if the frontend expects it
text/token_embedding
text/layers/{0..31}/...
text/layers/{4..27}/mhc/attn/weight
text/layers/{4..27}/mhc/attn/bias
text/layers/{4..27}/mhc/ffn/weight
text/layers/{4..27}/mhc/ffn/bias
text/final_norm
text/output_head
mtp/...                                  # from mtp.safetensors
vision/patch_embedding
vision/position_embedding
vision/blocks/{0..26}/...
vision/merger/...
```

The exact names must align with the binder. Do not store loop visits as duplicate weight objects.

Proposed numeric assignment for quality and fit:

| Role | Format |
|---|---|
| token embedding | `Q6G64_F16S` |
| full output head | `Q6G64_F16S` |
| full-attention Q/K | `Q4G64_F16S` |
| full-attention gate/V/output | `Q5G64_F16S` |
| GDN Q/K | `Q4G64_F16S` |
| GDN value/z and output | `Q5G64_F16S` |
| Text MLP gate/up | `Q4G64_F16S` |
| Text MLP down | `Q5G64_F16S` |
| norms, GDN convolution | BF16 |
| GDN `A_log`, `dt_bias` | FP32 |
| mHC weight/bias | FP32 |
| MTP matrices | `W8G32_F16S` |
| MTP norms | BF16 |
| Vision, including merger | BF16 initially for quality; if memory requires quantizing vision, use the proven Qwen vision roles (Q4/Q5/Q6 and W8 merger) |

Rationale: mHC is a small FP32 router (12 MiB total) and is quality-sensitive; GDN control
scalars are FP32; the text body uses the proven Q4/Q5/Q6 split. Vision BF16 costs about 0.55 GiB
more than Q4/Q5 but is the user's quality preference. If 256K + vision cannot fit, reduce vision
context first, not vision precision. Quantizing vision is a later, separately measured fallback.

Converter requirements:

- Read the AWQ main shards and `mtp/` index; dequantize GEMM matrices and the row-major embedding.
- Verify every source key/shape before writing.
- Use `tools/artifact/numeric.py` and `tools/artifact/layouts.py`; keep FP16 scales exact.
- Write a conversion report with source revision, hashes, object count, formats, and sizes.
- Independently decode the artifact and compare non-quantized tensors value-for-value.
- For quantized tensors, compare against an FP64 decode/naive projection on representative rows.
- The artifact must be self-identifying; no filename inference.

MTP source:

- Inspect `mtp.safetensors` keys and shapes before binding. The release main index has no `mtp.*`
  keys; the MTP is separate.
- The AWQ MTP config has `mtp.main_norm.weight` and `mtp.norm.weight`; `main_norm` is not consumed
  by the draft path. The binder uses `mtp.norm.weight` as `mtp/final_norm`.
- The release has `mtp_use_dedicated_embeddings=false`; reuse text embedding and the full
  `lm_head`. There is no optimized proposal head unless one is found in the source; do not
  invent one.
- MTP target verification uses the 56-slot schedule; the draft layer itself is one ordinary
  full-attention layer.

### 5.4 mHC Ops

Add public contracts under `include/ninfer/ops/mhc.h` and implementations under
`src/ops/mhc/`. Suggested API:

```cpp
struct MhcSettings {
    int streams;         // 2
    int iterations;      // 20
    bool fixed_identity; // false
    float pre_scale;     // 1.0
};

void mhc_router(const Tensor& residual, const Tensor& weight, const Tensor& bias,
                MhcSettings settings, Tensor& branch, Tensor& post, Tensor& mix,
                cudaStream_t stream);

void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post,
               const Tensor& mix, Tensor& out, cudaStream_t stream);
```

Implementation constraints:

- Residual is BF16 `[T, 2, H]` (or a documented `[T, 2H]` view).
- Weight is FP32 `[8, 2H]`; bias FP32 `[8]`.
- Normalize across the flattened 2H vector in FP32 with `+1e-6`.
- Compute eight logits in FP32; sigmoid; Sinkhorn in log space with 20 iterations.
- Produce `branch` BF16, `post` with the reference cast boundary, and `mix` in the form the
  merge op needs.
- Merge with BF16 matmul semantics matching the reference; do not silently keep everything in
  FP32.
- Reject `streams != 2`, `fixed_identity != false`, `iterations != 20`, or a range other than
  `[4,28)` for this target. The parser may support the fields; the target validates the exact
  checkpoint values.
- Handle T=1 decode and large prefill widths. For decode, one CTA per router is acceptable;
  measure launch overhead (96 router calls per token). For prefill, use a row-blocked schedule
  that amortizes the 8×2H weight read across tokens.
- Fuse `enter`/`leave` only if it does not change the reference rounding; initially keep them as
  elementwise views/copies.

Tests:

- Independent FP32/FP64 oracle for random `[T,2,H]` tensors; compare normalized logits, pre,
  post, mix, branch, merged output.
- Exact semantic cast boundaries: BF16 pre/mix, FP32 normalization/logits.
- Sinkhorn row/column sums.
- T=1, 2, 8, 128, 1024; H=4096; 20 iterations; `pre_scale` 1.0.
- Integration: one mHC decoder layer vs a PyTorch reference for both GDN and full-attention
  variants, with a fake per-visit cache.

### 5.5 YaRN RoPE

The release config uses `rope_type: yarn`, `factor=2.0`, `original_max_position_embeddings=262144`,
and `max_position_embeddings=524288`. YaRN is not a >256K-only feature: the reference rotary
embedding applies the YaRN frequency/attention scaling at every position. Running the release
weights with plain RoPE at 256K is a correctness bug.

Implement YaRN as an explicit RoPE mode:

- Add a `RopeScaling`/`YarnParameters` argument to the rope path. Keep the existing no-scaling
  behavior as the default for Qwen variants.
- Match the HF Qwen3.5 implementation for this config, including the `attention_factor`/mscale
  convention. Determine the exact defaults from Transformers 5.18 and the model code; do not
  assume only `position / factor`.
- Apply the same scaled inverse frequencies to the three MRoPE axes with sections `[11,11,10]`.
- Cover positions 0, 1, 8K, 262143, 262144, 524287 in unit tests.
- Compare against an independent formula and, where possible, the HF `rotary_emb` output.
- Document whether the cos/sin cache is precomputed; at 256K the partial-rotary cache is
  manageable, but the planner must include it in device memory.

### 5.6 Attention and GDN kernel profiles

Full attention 16/4:

- Update `require_causal_geometry` to admit `(16,4)`.
- Add route selection for group 4; do not reuse the 24/4 thresholds unchanged.
- Add prompt, chunked, small-T, append/attend, and graph-path instantiations.
- Cover BF16, int8, and `rk4v4-e8` KV; the 4080 profile must not regress the existing 24/4
  kernels.
- Add naive FP64 oracle tests for B=1..8, W=1..16, and prompt widths; compare all codecs.
- MTP target verification uses the same 16/4 geometry; add its envelope.

GDN 32 V-heads:

- Add `heads=32, input_rows=4096` to the gating-projection residency/plan.
- Add split Q4/Q5 input parents `[4096,4096]` and `[8192,4096]`.
- Add output `[4096,4096]`; conv channels `2*2048+4096=8192`; state width 4096.
- Verify `gated_delta_net` with Hv=32, Hq=16 at T=1 and prefill widths; the op documents
  support, but the 4080 route/residency needs a measured profile.
- Update ReplaySSM record layout for 42 GDN slots and MTP verify width.

Linear:

- Add 4096/12288 shapes to the 4080 linear/bench profiles.
- Q4/Q5/Q6 paths must support the new shapes; tune tile/split choices at T=1..16 and prefill
  chunks 512/1024/2688.
- Do not publish a performance number from a 5090 calibration; use the 4080 raw GB/s columns.

### 5.7 Loop integration

The execution skeleton for one slot:

```text
physical = plan.slot_physical(slot)
if physical in mHC range:
    residual = enter(hidden, physical, settings)
    branch, post, mix = router(residual, mhc_attn)
    branch = input_norm(branch)
    mixer_out = mixer(branch, slot_cache(physical, slot))
    residual = merge(residual, mixer_out, post, mix)
    branch, post, mix = router(residual, mhc_ffn)
    branch = post_attention_norm(branch)
    mlp_out = mlp(branch)
    hidden = leave(merge(residual, mlp_out, post, mix), physical, settings)
else:
    ordinary pre-norm mixer + MLP with single residual stream
```

- Layer 3 ends in `[T,H]`; layer 4 visit 1 expands to two streams; layer 27 visit 1 averages back
  to `[T,H]`; layer 28 is ordinary; layer 4 visit 2 expands again.
- The router branch for attention is `[T,H]`; the mixer sees one branch, not two.
- The same physical weights are used on both visits; the cache object passed to the mixer is the
  visit's slot, not the physical index.
- MTP target verification, prefill chunks, ordinary decode, and causal scoring all use the same
  slot loop.
- Graph capture must include the 56 slots and the mHC kernels; verify with a graph-vs-eager
  byte/tolerance comparison.

### 5.8 Vision and frontend

Vision:

- Reuse the family tower; change `VisionConfig::output_hidden` to 4096 and the merger fc2 shape
  to `[4096,4608]`.
- Keep the 27-layer tower, 16 heads, head_dim 72, patch 16, temporal 2, merge 2, 2304 positions.
- Bind BF16 vision weights initially. If the memory gate fails at 256K + vision, reduce vision
  context to ~200K before considering quantization.
- Update the visual-scatter handoff width and the media prompt positions.
- Tests: one fixed image vs the BF16 reference; image + text; multiple images; no video.

Frontend:

- Add a third chat-template digest/semantics for the SenseNova template
  (`a4aee8af...`). It is not either existing Qwen digest.
- Differences to implement:
  - The template has no `preserve_thinking` condition; assistant history after the last query
    always renders the `<think>` block.
  - Tool argument serialization stringifies non-container values; Qwen's fixture JSON-encodes
    scalars. Add a semantics flag rather than changing Qwen behavior.
  - No trailing newline at EOF is irrelevant to rendering.
- The `--preserve-thinking` option must either be rejected for this target or mapped to the
  template's behavior. Record the decision in the CLI help; do not silently change semantics.
- Tokenizer/resources: use the release `tokenizer.json`, `tokenizer_config.json`,
  `generation_config.json`, `chat_template.jinja`, and image preprocessor resource.
- Verify token domain 248077, special-token IDs, stop tokens, and tool parser against the
  release template.
- Add rendered-prompt tests against Jinja for text, thinking on/off, tool calls, images, and
  multi-turn histories.

### 5.9 MTP and n-gram

- Inspect and bind the separate `mtp.safetensors`.
- MTP config: 1 layer, hidden 4096, FFN 12288, 16 Q / 4 KV / 256, full attention, 1e-6 RMS,
  partial MRoPE, no dedicated embedding.
- Stem: `e = RMSNorm(embed(token), pre_fc_norm_embedding)`,
  `h = RMSNorm(target_hidden, pre_fc_norm_hidden)`, `x = fc([e,h])`.
- MTP layer: full attention + SwiGLU; its own 1-layer KV; final norm.
- The release MTP uses the full text `lm_head`; no optimized shortlist head. Use
  `ProposalHead::Full` and measure the extra head cost. Do not claim `--lm-head-draft` support
  unless an optimized head is actually present.
- MTP3 + n-gram chain first. The n-gram chain is model-agnostic but must use the target token
  domain and stop-token policy.
- DFlash2 deferred; do not add draft weights or schedules until an official SenseNova draft
  exists.
- Tests: MTP logits vs BF16 reference, acceptance/correction, target verify with 56 slots,
  checkpoint rollback, n-gram chain, no-output-change invariant for greedy.

### 5.10 Memory plan

Approximate device budget (GiB; validate with the real allocator, not this table):

| Component | 256K int8 | 200K int8 | 256K `rk4v4-e8` (not selected) |
|---|---:|---:|---:|
| KV, 14 full layers × 4 heads × 256 | 7.22 | 5.64 | 3.72 |
| Q4/Q5/Q6 text + MTP + mHC | ~5.7 | ~5.7 | ~5.7 |
| Vision BF16 (0.4B) | ~0.8 | ~0.8 | ~0.8 |
| GDN state 42 slots + images | ~0.1–0.3 | ~0.1–0.3 | ~0.1–0.3 |
| Workspace, graphs, MTP, page tables | ~1.5–2.5 | ~1.5–2.5 | ~1.5–2.5 |
| **Approximate total** | **~15.3–16.5** | **~13.7–14.9** | **~11.8–13.0** |

At plan time the card had 15,254 MiB free (14.9 GiB). The 256K + int8 + vision total is right at
or above that. This is why the vision-context fallback exists. The startup path must:

- Plan weights before loading and check `cudaMemGetInfo`.
- Materialize weights, then resolve KV capacity against actual free memory.
- Refuse to listen if the requested profile does not fit; print the measured component budget.
- Keep a small safety margin for transient allocations; do not rely on the last few MiB.
- Record the actual free bytes after startup in logs/metrics.

Host memory:

- Target `--host-kv-mib 4096` and the default host state slots.
- The WSL host is being raised to 32 GB; the current 15.5 GiB is insufficient for the BF16
  reference and may be tight for host checkpoints.
- The 256K profile uses one lane; shorter contexts may use more lanes when the planner proves
  capacity.

KV codecs:

- `int8` is the selected q8 profile.
- `rk4v4-e8` remains available as a fallback/debug mode, but the user chose quality over
  automatic KV precision reduction; do not switch silently.
- Do not use NVFP4 KV on `sm_89`; the build stubs it.

## 6. Verification plan

### 6.1 BF16 CPU reference

The 4080 cannot hold the 19.3 GB BF16 model, and the current WSL 15.5 GiB cannot either. After
the user raises WSL to 32 GB, run a one-time CPU capture on this machine:

```text
tools/reference/sensenova_loop_bf16.py
```

Requirements:

- Pin `transformers==5.18.0`, `torch==2.11.0`, `safetensors`, and the release custom code.
- Load `AutoModelForImageTextToText` / `AutoModelForCausalLM` with `trust_remote_code=True`,
  `dtype=torch.bfloat16`, `low_cpu_mem_usage=True`, CPU device.
- Use `torch.inference_mode()` and a small fixed prompt set:
  - short text, thinking on
  - short text, thinking off
  - tool-call prompt
  - one image + text
  - a 2-token continuation to inspect per-visit cache state
- Capture:
  - last-token logits
  - hidden states after the embedding and each of the 56 visits (57 states)
  - mHC branch/post/mix for selected layers/tokens
  - DynamicCache per execution slot after the first and second steps (KV and GDN state)
  - vision embeddings/merger output for the image prompt
  - MTP logits for one prompt if the reference model exposes them
- Save arrays plus a manifest with the release revision, config hash, tokenizer hash, prompt
  token IDs, and CPU thread count.
- Do not require generation throughput; this is a correctness capture.

If the custom model's `output_hidden_states=True` does not expose the states needed, call the
text model's `forward` directly with `output_hidden_states=True` and use the returned tuple; the
release implementation documents one state per visit. For cache inspection, use the
`_loop_execution_plan` and `_VisitCache` mapping in the custom code, or run one forward per visit
with a controlled cache object.

### 6.2 Official AWQ cross-check

- Run the official AWQ on the 4080 in a separate environment if its dependencies conflict.
- Use the same prompts/tokenizer and greedy decoding.
- Compare last-token logits, top-1 agreement, and generated text.
- This is a 4-bit reference, not BF16. It tests loop/mHC/cache behavior and quantized quality,
  but not exact BF16 arithmetic.
- Use it to set the quantized-artifact quality gate: quick perplexity no worse than AWQ and task
  eval within noise.

### 6.3 Op-level oracles

Every new floating-point Op gets an independent naive FP32/FP64 oracle. Packed inputs are
decoded with their stored scales.

- mHC router/merge: normalization, logits, sigmoid, Sinkhorn, branch, merge; semantic BF16 casts.
- YaRN: scaled inverse frequencies, cos/sin, MRoPE axes, attention factor.
- Full attention 16/4: naive causal GQA over K/V for all codecs; append + attend.
- GDN 32 V-heads: naive delta rule, conv, recurrent state, gating.
- Linear Q4/Q5/Q6 at 4096/12288; independent FP64 dequantized matmul.
- MTP stem/layer at 4096/12288/16-4-256.
- Vision merger 4096 output.
- Conversion inventory and artifact decode.

### 6.4 End-to-end

- Text: greedy fixed prompts; compare to BF16 CPU logits within a documented quantization
  tolerance; record max/mean |Δlogit| and top-1 agreement. Provisional gate: mean |Δ| ≤0.1,
  max ≤1.0, top-1 agreement ≥95% on short prompts; adjust only with evidence.
- Loop/cache: verify the first and second visits have distinct KV/GDN state; a checkpoint
  save/restore at 128K/200K/256K continues with identical outputs.
- Vision: fixed image vs BF16 reference; image + text; multiple images; no video.
- MTP: draft logits/acceptance; greedy output equals MTP0; n-gram output invariant.
- CausalScoring: fixed corpus logprobs vs BF16 reference.
- Serving: OpenAI Responses/Chat Completions, Anthropic Messages, streaming, usage, tool calls,
  thinking channels; affected schema tests.
- CLI: text, image, thinking, long decode, context cache.
- Perplexity: quick corpus; compare AWQ and BF16 (CPU) where possible.

### 6.5 Memory and performance gates

Memory:

- 256K text, int8 KV, Q4/Q5/Q6 weights: startup must pass; record free bytes.
- 256K + vision, int8 KV: attempt; if it does not fit, validate 200K + vision.
- 128K/64K with vision; multi-lane only if the plan proves capacity.
- Checkpoint/restore at the maximum validated context.

Performance (one lane, 4080, exact commands/date/artifact hash):

| Depth | MTP0 | MTP3 no n-gram | Prefill |
|---|---:|---:|---:|
| 8K | ≥80 tok/s | ≥180 tok/s | ≥4,000 tok/s |
| 256K | ≥55 tok/s | ≥120 tok/s | ≥2,000 tok/s |

Also report n-gram-chain decode separately. Target the 4080's measured bandwidth, not the 5090
constant. Use `ninfer_bench`, the 4080 launcher profile, and `profiles/bench`/`nsys` for
attribution. An Op microbenchmark is not an end-to-end result.

## 7. Work order and gates

### Stage 0 — reference and source freeze

- Raise WSL to 32 GB; verify `free -h` shows ~31 GiB.
- Download/pin the release BF16 snapshot; verify hashes and index.
- Inspect `mtp.safetensors`; record keys/shapes and the `main_norm` mapping.
- Build the BF16 CPU capture script; generate the manifest and arrays for the fixed prompts.
- Run the official AWQ on the 4080 for the same prompts.

Gate: BF16 logits/hidden/cache/MTP/vision reference files exist and are recorded with the source
revision; MTP key mapping is documented; AWQ baseline is recorded.

### Stage 1 — family execution plan

- Add the compile-time plan seam; identity plans for Qwen3.6/3.8/35B.
- Convert state/KV/resource/graph loops to slots while preserving physical weight arrays.
- Keep existing model outputs byte-identical where possible.

Gate: existing `ctest` passes; Qwen 27B/35B real-artifact tests and benchmarks are unchanged;
a synthetic 32-layer/56-slot plan unit test covers slot mapping, state counts, and cache
isolation.

### Stage 2 — mHC and YaRN Ops

- Implement mHC router/merge/enter/leave with tests.
- Implement YaRN RoPE with tests against the independent/HF reference.
- No target integration yet.

Gate: FP64 oracles pass at T=1..1024; YaRN covers 0..524287; no existing RoPE behavior changes.

### Stage 3 — target package and artifact

- Add config/bindings/package/registry/CMake.
- Add converter inventory/recipe/convert/verify.
- Build the Q4/Q5/Q6 artifact; inspect object count/sizes.
- Load on the 4080 with a short context; run a text-only prompt.
- Compare against the BF16 CPU reference and AWQ.

Gate: artifact loads; text-only logits meet the provisional tolerance; conversion report and
artifact identity are correct.

### Stage 4 — new kernel profiles

- Add 16/4 attention for all required codecs and MTP verify.
- Add 32-V-head/4096-hidden GDN profiles.
- Add 4096/12288 linear tuning.
- Qualify each against the independent oracle; measure at T=1..16 and prefill widths.

Gate: all op tests pass; 4080 benchmarks record the new shapes; no regression to existing
24/4 and 48-head profiles.

### Stage 5 — loop/mHC end-to-end

- Integrate mHC into prefill, ordinary decode, MTP verify, causal scoring, and graph capture.
- Wire 56-slot state/KV/checkpoints/resource projections.
- Run text prompts, multi-turn, thinking, tool calls.
- Verify per-visit cache isolation and checkpoint restore.

Gate: BF16-vs-engine logits/token agreement within the documented tolerance; MTP3 greedy output
equals MTP0; graph and eager paths agree; 8K speed bar passes.

### Stage 6 — vision, frontend, MTP/n-gram, serving

- Vision merger 4096 and BF16 vision weights.
- SenseNova chat-template semantics/digest and tokenizer resources.
- MTP stem/layer binding; full lm_head; n-gram chain.
- CLI/HTTP/CausalScoring/context cache.
- Image and tool tests; serving schema tests.

Gate: image prompts vs BF16; no video; MTP acceptance recorded; full serving tests pass.

### Stage 7 — 256K fit and publication

- Attempt 256K text + int8; then 256K + vision + int8; fallback 200K vision.
- Validate checkpoint/restore and long-context retrieval.
- Run 8K/32K/64K/128K/200K/256K performance.
- Write model card, artifact reference, README/performance updates.
- Remove this plan or fold durable facts into the model/artifact references.

Gate: 256K text q8 passes; vision profile is either 256K or the measured 200K fallback;
performance bar passes; docs and tests are current.

## 8. Risks and open items

1. **256K + int8 + vision may not fit.** The measured budget decides. Fallback is 200K vision,
   not lower weight/KV precision. If 256K text + int8 itself fails, stop and report.
2. **YaRN is mandatory at all positions.** Do not treat it as a 512K-only feature. A plain-RoPE
   target may load and generate garbage.
3. **MTP source layout is not yet inspected.** `main_norm` may be an extra stem/final norm; bind
   it explicitly. The full lm_head makes MTP draft tokens expensive; measure before promising
   the MTP3 bar.
4. **mHC launch overhead.** 96 router calls per forward may add hundreds of microseconds. Fuse
   or batch if the measured overhead threatens the speed bar; do not change rounding.
5. **Group-4 attention coverage.** The 4080 has multiple KV codecs and prompt/verify routes; add
   only the routes the target uses, but test every enabled codec.
6. **Q4/Q5 error is applied twice in the loop block.** Compare against AWQ and BF16; if quality
   is poor, consider selective Q5/Q6 for the loop block or a better quantization recipe before
   changing the architecture.
7. **56-slot graph/state size.** Graph capture and checkpoint copies grow; measure memory and
   startup time. A graph profile is not a layer count.
8. **Frontend template differences.** The template is not either Qwen digest; test rendered
   prompts. `--preserve-thinking` semantics must be decided and documented.
9. **Host RAM.** The BF16 reference needs 32 GB WSL; the current 15.5 GiB is insufficient.
   Host KV pools and conversion staging must not exhaust it.
10. **No BF16 end-to-end on the 4080.** BF16 correctness is CPU/op-level plus AWQ cross-check;
    state that limitation in the model card.
11. **512K is not in launch scope.** YaRN makes the math available, but the 16 GB card cannot
    hold 512K int8 with 14 KV planes. Do not advertise it without a measured profile.
12. **Naming debt.** The shared family is named `qwen3_6` but owns Qwen3.5-style hybrid
    semantics. Keep the namespace for this work to avoid a broad rename; document the ownership
    in the new target/reference. A later rename is a separate refactor.

## 9. Key commands

Build and test:

```bash
cd /work
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Python checks:

```bash
python3 --version
python3 -m py_compile tools/convert/sensenova_loop_9b/*.py
python3 -m pytest tests/artifact tests/convert -q
```

Source download (after `hf auth whoami` succeeds):

```bash
hf download sensenova/SenseNova-Loop-9B \
  --revision 3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6 \
  --local-dir /models/SenseNova-Loop-9B
```

Expected runtime profile (target, not yet validated):

```bash
./build/apps/ninfer-serve /models/sensenova_loop_9b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
  --host-kv-mib 4096 \
  --spec mtp --draft-tokens 3 --ngram chain \
  --vision
```

Benchmarks:

```bash
./build/apps/ninfer_bench ...   # use the exact 4080 suite/commands from bench/README.md
```

Record the date, artifact hash, source revision, GPU, driver, and exact command with every
published number.

## 10. Definition of done

- `sensenova-loop-9b/groupwise-int` loads through the public Engine on the RTX 4080.
- Text + image, thinking, tool calls, MTP3 + n-gram, context cache/checkpoints, CausalScoring,
  CLI, and HTTP serving work.
- 256K text with int8 KV meets the memory and speed bar; vision is validated at 256K or the
  measured ~200K fallback.
- BF16 CPU reference/AWQ cross-check and all new op oracles pass.
- No known in-scope blocker remains.
- Model card, artifact reference, README/performance page, and tests are updated.
- This plan is removed or its durable facts are folded into the model/artifact references.

## 11. Resume checklist

1. `git status -sb`; confirm branch/HEAD and no unexpected changes.
2. `nvidia-smi`; confirm the 4080 and free memory.
3. `free -h`; confirm WSL has 32 GB before starting the BF16 capture.
4. `hf auth whoami`; confirm access without printing the token.
5. Read sections 2, 3.4, 5.1, 5.4, 5.5, and 7.
6. Start at the first stage whose gate has not passed; do not skip the reference or the
   execution-plan seam.
7. Update this file at each gate with the measured result, or replace the relevant section with
   the durable reference when the plan is retired.
