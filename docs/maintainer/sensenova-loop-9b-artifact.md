# SenseNova-Loop-9B Artifact Reference

This reference defines the `sensenova-loop-9b/groupwise-int` `.ninfer` object contract. Common
framing is defined in [`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and loop/mHC mathematics in
[`sensenova-loop-9b-model.md`](sensenova-loop-9b-model.md).

## 1. Identity and source freeze

```text
model_id     sensenova-loop-9b
weights_id   groupwise-int
target_key   sensenova_loop_9b
source       sensenova/SenseNova-Loop-9B-AWQ
revision     8327226ffd2e33e34cb5466a0e6cccd4419bde40
```

Pinned source hashes:

| Resource | SHA-256 |
|---|---|
| `config.json` | `f62caa6744b50c25d13348a572403c2f597526f46a27da407ac0705a01ffcebb` |
| `model.safetensors.index.json` | `36a2da515feace45ffefaa158c1dab741cd73128707f20cfb43fd0fd94fd7a2b` |
| `mtp/model.safetensors.index.json` | `6b10fe824edbf051205aeb1b4ed62a118386c30f4ec83b4eb5d463e82c21bc23` |
| `mtp/model-00001-of-00001.safetensors` | `bca1271c68a0584c31449dc66b8c3826e9428c3b31ded6358b745ed172abff4d` |
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `generation_config.json` | `94d008428d534f4a633708eae6a3e72b1eae5326b6096d896daa53d1648e07ee` |
| `video_preprocessor_config.json` (BF16 release) | `bd96830cce1312679aa1f2ea43ed17da20e8ebbcea2398cfa64e2d275a96fdd7` |

The AWQ main index has three `model-*.safetensors` shards. MTP is
`mtp/model-00001-of-00001.safetensors` (486,599,352 bytes) with its own index. The AWQ release
omits `video_preprocessor_config.json`; the converter takes that one frontend resource from
`sensenova/SenseNova-Loop-9B` revision `3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6` because this
target does not support video. AWQ's extra `mtp.main_norm.weight` is not consumed;
`mtp.norm.weight` is the draft final norm.

## 2. Artifact size and object counts

The converted artifact contains **834 objects**: 6 raw frontend resources and 828 tensors.

| Component | Tensors |
|---|---:|
| Text core | 483 |
| MTP | 12 |
| Vision | 333 |
| **Total** | **828** |

Formats: `BF16` 406, `FP32` 144, `Q4G64_F16S` 118, `Q5G64_F16S` 150,
`Q6G64_F16S` 3, `W8G32_F16S` 7. There are no `I32`, NVFP4, FP8, or Q3 objects.

The current artifact is 6,248,091,904 bytes with SHA-256
`e793869f6b4edb98ae1a8f08899a28f938ed77652796f5eaf572785cc48267e0`. The size and hash describe the
generated file; identity is serialized and never inferred from filename, hash, or object count.

## 3. Text objects

All matrix shapes are logical `[N,K] = [output rows,input columns]`; quantization groups along K.
Layer numbers are unpadded decimal physical layer indices. There are no duplicate objects for the
second loop visit.

```text
text/token_embedding                         Q6G64_F16S [248320,4096]
text/layers/{0..31}/input_norm               BF16       [4096]
text/layers/{3,7,...,31}/attention/query_key Q4G64_F16S [5120,4096]  (Q 4096 + K 1024)
text/layers/{3,7,...,31}/attention/gate_value Q5G64_F16S [5120,4096] (gate 4096 + V 1024)
text/layers/{3,7,...,31}/attention/query_norm BF16      [256]
text/layers/{3,7,...,31}/attention/key_norm   BF16      [256]
text/layers/{3,7,...,31}/attention/output     Q5G64_F16S [4096,4096]
text/layers/{0,1,2,4,...30}/gdn/a_log         FP32       [32]
text/layers/{0,1,2,4,...30}/gdn/dt_bias       FP32       [32]
text/layers/{0,1,2,4,...30}/gdn/convolution   BF16       [4,8192]
text/layers/{0,1,2,4,...30}/gdn/a_projection  BF16       [32,4096]
text/layers/{0,1,2,4,...30}/gdn/b_projection  BF16       [32,4096]
text/layers/{0,1,2,4,...30}/gdn/query_key     Q4G64_F16S [4096,4096] (Q 2048 + K 2048)
text/layers/{0,1,2,4,...30}/gdn/value_z       Q5G64_F16S [8192,4096] (V 4096 + Z 4096)
text/layers/{0,1,2,4,...30}/gdn/norm          BF16       [128]
text/layers/{0,1,2,4,...30}/gdn/output        Q5G64_F16S [4096,4096]
text/layers/{0..31}/post_attention_norm       BF16       [4096]
text/layers/{0..31}/mlp/gate_up               Q4G64_F16S [24576,4096]
text/layers/{0..31}/mlp/down                  Q5G64_F16S [4096,12288]
text/layers/{4..27}/mhc/attn/weight           FP32       [8,8192]
text/layers/{4..27}/mhc/attn/bias             FP32       [8]
text/layers/{4..27}/mhc/ffn/weight            FP32       [8,8192]
text/layers/{4..27}/mhc/ffn/bias              FP32       [8]
text/final_norm                               BF16       [4096]
text/output_head                              Q6G64_F16S [248320,4096]
```

The full-attention `query_key` and `gate_value` parents use the same head-interleaved Q-projection
row order as the Qwen family: per head `[query 256, gate 256]`; the K and V rows are appended after
the Q/gate rows. GDN `query_key` concatenates Q then K; `value_z` concatenates the release
`in_proj_qkv` value rows then `in_proj_z`.

## 4. MTP objects

```text
mtp/input_projection                         W8G32_F16S [4096,8192]
mtp/embedding_norm                           BF16       [4096]
mtp/hidden_norm                              BF16       [4096]
mtp/layer/input_norm                         BF16       [4096]
mtp/layer/attention/query_key_gate_value     W8G32_F16S [10240,4096] (Q 4096,K 1024,G 4096,V 1024)
mtp/layer/attention/query_norm               BF16       [256]
mtp/layer/attention/key_norm                 BF16       [256]
mtp/layer/attention/output                   W8G32_F16S [4096,4096]
mtp/layer/post_attention_norm                BF16       [4096]
mtp/layer/mlp/gate_up                        W8G32_F16S [24576,4096]
mtp/layer/mlp/down                           W8G32_F16S [4096,12288]
mtp/final_norm                               BF16       [4096]
```

Source mapping: `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`,
`mtp.pre_fc_norm_hidden.weight`, `mtp.layers.0.*`, and `mtp.norm.weight`. The AWQ MTP carries an
extra `main_norm` tensor; it is not consumed. `mtp.norm.weight` is the draft final norm. MTP
reuses the text embedding and full `lm_head`.

## 5. Vision objects

The Vision tower uses the shared Qwen3.5 Vision inventory and the proven quantized vision roles
rather than BF16 tower weights. This is a memory/route decision; all fixed dimensions remain the
release tower. Merger output is 4096.

```text
vision/patch_embedding             Q6G64_F16S [1152,1536]
vision/patch_embedding_bias        BF16       [1152]
vision/position_embedding          BF16       [2304,1152]
vision/layers/{0..26}/attention/qkv        Q4G64_F16S [3456,1152]
vision/layers/{0..26}/attention/qkv_bias   BF16       [3456]
vision/layers/{0..26}/attention/output     Q5G64_F16S [1152,1152]
vision/layers/{0..26}/attention/output_bias BF16      [1152]
vision/layers/{0..26}/mlp/fc1              Q4G64_F16S [4304,1152]
vision/layers/{0..26}/mlp/fc1_bias         BF16       [4304]
vision/layers/{0..26}/mlp/fc2              Q5G64_F16S [1152,4304]
vision/layers/{0..26}/mlp/fc2_bias         BF16       [1152]
vision/layers/{0..26}/norm1/{weight,bias}  BF16       [1152]
vision/layers/{0..26}/norm2/{weight,bias}  BF16       [1152]
vision/merger/fc1                  W8G32_F16S [4608,4608]
vision/merger/fc1_bias             BF16       [4608]
vision/merger/fc2                  W8G32_F16S [4096,4608]
vision/merger/fc2_bias             BF16       [4096]
vision/merger/norm/{weight,bias}   BF16       [1152]
```

## 6. Frontend resources

Exactly six raw resources are embedded in the canonical order:

```text
frontend/tokenizer.json
frontend/tokenizer_config.json
frontend/chat_template.jinja
frontend/generation_config.json
frontend/preprocessor_config.json
frontend/video_preprocessor_config.json
```

The template digest is `a4aee8af...`; the frontend resolves it as
`ChatTemplateSemantics::SenseNovaLoop`. The tokenizer config has no `added_tokens_decoder`;
added tokens come from `tokenizer.json`.

## 7. Converter

The canonical artifact is converted from the official AWQ checkpoint. `awq.py` dequantizes the
AWQ GEMM matrices and row-major 4-bit token embedding, keeps the AWQ-adjusted RMSNorm weights,
casts direct tensors to the registered payload dtypes, and re-encodes the same object inventory
and numeric assignment. `inventory.py` defines the complete object order and formats; `recipe.py`
maps every object to its logical source.

```bash
cd /work
/opt/venv/bin/python -m tools.convert.sensenova_loop_9b.awq \
  --model /models/SenseNova-Loop-9B-AWQ \
  --out out/sensenova_loop_9b.ninfer \
  --resource-fallback /models/SenseNova-Loop-9B \
  --source-revision 8327226ffd2e33e34cb5466a0e6cccd4419bde40
/opt/venv/bin/python -m tools.convert.sensenova_loop_9b.verify \
  out/sensenova_loop_9b.ninfer --model /models/SenseNova-Loop-9B-AWQ --source awq
```

The AWQ release omits `video_preprocessor_config.json`; the converter takes that one frontend
resource from the BF16 release because this target does not support video. `convert.py` remains
the BF16-release reader for reference conversion; `awq.py` produces the canonical artifact.

On the RTX 4080 with `int8` KV and the fixed quick corpus (4,096-token context, 2,048 stride), the
AWQ-sourced artifact scored PPL 8.554851 versus 9.151547 for the prior BF16-sourced artifact. The
maintainer MBPP comparison found no regression against the prior artifact.

## 8. Load and runtime checks

```bash
# Text
./build/apps/ninfer out/sensenova_loop_9b.ninfer --prompt "Hello" \
  --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --max-new 8 --greedy

# 256K Text + Vision capacity
./build/apps/ninfer out/sensenova_loop_9b.ninfer \
  --messages /tmp/vision.json \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --vision --max-new 1 --greedy --no-thinking

# MTP3 + n-gram
./build/apps/ninfer out/sensenova_loop_9b.ninfer --prompt "..." \
  --max-context 512 --kv-capacity 512 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --ngram chain --max-new 8 --greedy
```

The target uses 14 KV planes and 42 GDN state slots, not 8/24 physical counts. A state or
checkpoint produced by a different plan must not be reused; the slot plan is compile-time target
identity.
