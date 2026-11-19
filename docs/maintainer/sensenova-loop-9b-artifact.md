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
source       sensenova/SenseNova-Loop-9B
revision     3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6
```

Pinned source hashes:

| Resource | SHA-256 |
|---|---|
| `config.json` | `614dbd44b3a7fba8cd3adf5118b91ee40d51765b874de9dfadf67ea4e16ba68e` |
| `modeling_sensenova_loop.py` | `26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b` |
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `model.safetensors.index.json` | `a478440d3817d761ef6a8088f063131e02197b0d8699262165d4626c5d577510` |
| `preprocessor_config.json` | `3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2` |
| `generation_config.json` | `f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f` |

The main index is sharded into 11 `model-*.safetensors` files; MTP is the separate
`mtp.safetensors` (486,582,880 bytes) and is not in the main index. The release has no
`text/draft_head` or optimized proposal head.

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

The converted release artifact used for qualification is 6,248,091,904 bytes with SHA-256
`c58e34b8b6cfec63914eeca797c611bf6f37374869a2f73166231cf8f17fb144`. The size and hash describe the
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
`mtp.pre_fc_norm_hidden.weight`, `mtp.layers.0.*`, and `mtp.norm.weight`. The release has no
`main_norm`; `mtp.norm.weight` is the draft final norm. MTP reuses the text embedding and full
`lm_head`.

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

```bash
cd /work
/opt/venv/bin/python -m tools.convert.sensenova_loop_9b.convert \
  --model /models/SenseNova-Loop-9B \
  --out out/sensenova_loop_9b.ninfer
/opt/venv/bin/python -m tools.convert.sensenova_loop_9b.verify \
  out/sensenova_loop_9b.ninfer --model /models/SenseNova-Loop-9B
```

`inventory.py` defines the complete object order and formats; `recipe.py` maps every object to
main-shard or `mtp.safetensors` sources; `convert.py` validates the release config and writes the
artifact plus `.conversion.json`; `verify.py` checks structural identity, exact direct payloads,
and representative quantized rows against the source.

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
