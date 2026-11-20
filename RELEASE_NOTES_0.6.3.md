# NInfer-4080 v0.6.3

This release adds the registered SenseNova-Loop-9B target to the RTX 4080 fork. The Qwen3.8-27B
`gsq3` target and its artifact are unchanged.

## Highlights

- Registers `sensenova-loop-9b/groupwise-int` as target `sensenova_loop_9b`.
- Compile-time 56-slot execution plan: 32 physical layers, the `4..27` loop block visited twice,
  14 KV planes, and 42 GDN state slots.
- mHC router, merge, enter, and leave Ops with FP64 oracles; YaRN RoPE evaluated in FP64 at every
  position.
- SenseNova 16/4 attention, 4096/12288 linear profiles, MTP3 plus the n-gram chain, the release
  chat template/tokenizer, and the 4096-wide vision merger.
- 256K text and 256K text plus vision start with int8 KV on the 16 GiB card; the 256K + vision +
  MTP3 launch profile starts with about 0.35 GiB available.
- The canonical artifact is converted from the official AWQ checkpoint: 6,248,091,904 bytes,
  SHA-256 `e793869f6b4edb98ae1a8f08899a28f938ed77652796f5eaf572785cc48267e0`.
- 253K-token prefill measured 1.72k tok/s; 3.5K decode about 36 tok/s MTP0 and 58 tok/s with
  MTP3 + n-gram; quick corpus PPL 8.55 versus 9.15 for the prior BF16-sourced artifact.

## Artifact and image

- Hugging Face: `roofkid/SenseNova-Loop-9B-NInfer`
- Docker Hub: `roofkid/ninfer-4080:0.6.3-rtx4080`, with the model aliases `sensenova` and `gsq3`.
  The `gsq3` alias moves to this image so Qwen3.8-27B gets the same release binaries;
  `0.6.2-rtx4080` remains unchanged.
- Download: `scripts/download-sensenova-loop-9b.{sh,bat}`
- Launcher: `scripts/run-sensenova-loop-9b.{sh,bat}`

The image carries only the binaries; the artifact is bind-mounted read-only.

## Limits

- The implementation plan's aspirational 80/55 MTP0, 180/120 MTP3, and 4k/2k prefill bars are not
  met. This is a functional 256K profile, not a performance claim.
- The 256K decode depth was not benchmarked end to end; 253K-token prefill and 256K startup were.
- Vision uses the qualified quantized Qwen Q4/Q5/Q6/W8 roles rather than BF16 tower weights.
- Video is not supported. The 256K + vision + MTP3 profile runs one lane.
