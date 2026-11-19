# SenseNova-Loop-9B NInfer

NInfer target for `sensenova/SenseNova-Loop-9B` on a single RTX 4080 (`sm_89`, 16 GiB).

| Field | Value |
|---|---|
| NInfer identity | `sensenova-loop-9b` / `groupwise-int` / `sensenova_loop_9b` |
| Source revision | `3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6` |
| Artifact | `sensenova_loop_9b.ninfer` |
| Artifact bytes | 6,248,091,904 |
| Artifact SHA-256 | `c58e34b8b6cfec63914eeca797c611bf6f37374869a2f73166231cf8f17fb144` |
| Text | 32 physical layers / 56 execution slots, 4096 hidden, 16/4/256 attention, 32-head GDN |
| Context | 262144 validated with int8 KV (524288 native limit) |
| Modalities | Text and image; video not supported |
| Speculation | MTP3 + n-gram chain; no optimized proposal head, no DFlash2 |

This is a 4-bit groupwise-int artifact converted from the official BF16 release. Text weights use
Q4/Q5/Q6, mHC routers and GDN control scalars are FP32, norms/convolution are BF16, MTP matrices
are W8, and the Vision tower uses the proven Qwen Q4/Q5/Q6/W8 roles with a 4096-wide merger.

The loop visits physical layers `4..27` twice. Weights are shared; KV planes, GDN state,
ReplaySSM records, checkpoints, and graph state are isolated per execution slot. mHC applies two
FP32 two-stream routers per physical layer on both visits. YaRN is applied at every position.

Validation on the RTX 4080:

- 256K Text + int8 KV starts with 1.68 GiB free; 256K + Vision starts with 1.27 GiB free.
- The 256K + Vision + MTP3 launch profile starts with about 0.35 GiB available after startup.
- 253K-token prefill measured 1.72k tok/s; 3.5K-token prefill 2.62k tok/s.
- 3.5K decode measured ~36 tok/s MTP0 and ~58 tok/s MTP3+n-gram.
- 49-token perplexity 5.32 versus 5.09 for BF16 CPU.

The performance targets in the implementation plan (80/55 MTP0, 180/120 MTP3, 4k/2k prefill) are
not met by this artifact; it is published as a functional target with measured results, not as a
performance claim. BF16 end-to-end inference is not possible on the 16 GiB card; correctness is
qualified by CPU BF16 references, op oracles, and the AWQ cross-check.

## Run

Both launchers are Docker launchers, following `scripts/run-ninfer-4080.{sh,bat}`. They mount
`sensenova_loop_9b.ninfer` from `out/` or `models/` read-only and serve the validated profile:

```bash
./scripts/run-sensenova-loop-9b.sh
```

The image carries only the binaries. `NINFER_IMAGE` selects it (default
`roofkid/ninfer-4080:sensenova`). A missing image is pulled; when the pull fails, the launcher
builds it from the checkout's Dockerfile and labels it with the checkout revision, so source
changes rebuild it. The published `roofkid/ninfer-4080:gsq3` image predates this target, so the
first run is expected to build. `NINFER_BUILD=1` forces a rebuild. Overrides: `NINFER_ARTIFACT`,
`NINFER_PORT` (8080), `NINFER_BIND` (0.0.0.0, set 127.0.0.1 to keep the port host-local),
`NINFER_API_KEY`, and `NINFER_KV_DTYPE` (default int8; `rk4v4-e8` is the fallback/debug mode).

The launcher runs this command inside the image with the artifact mounted at `/models`:

```bash
ninfer-serve /models/sensenova_loop_9b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --host-kv-mib 4096 \
  --spec mtp --draft-tokens 3 --ngram chain \
  --vision \
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0
```

See [model semantics](../../docs/maintainer/sensenova-loop-9b-model.md) and
[artifact contracts](../../docs/maintainer/sensenova-loop-9b-artifact.md).
