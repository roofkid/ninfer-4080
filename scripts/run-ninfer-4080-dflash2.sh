#!/usr/bin/env bash
#
# NInfer on the RTX 4080 with the DFlash2 draft model, Windows/Linux host through
# Docker Desktop or Docker.
#
# Builds (once) the product image from ./Dockerfile and serves the DFlash2 profile at
# the context that fits the 16 GiB card with the stock companion:
#
#   * 57,344-token context without vision, 28,672 with vision (measured caps; override
#     with NINFER_CONTEXT)
#   * DFlash2 block drafting with 7 draft tokens (verify width 8, the widest window on
#     the fast small-T tensor-core route), LM-head proposal selector
#   * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
#     presence/frequency penalties 0. This engine has no multiplicative repeat penalty;
#     the neutral value 1 is its implicit behavior.
#
# Measured on this card (docs/maintainer/rtx-4080-plan.md section 11): DFlash2 K=7
# decodes about 122/203/180 tok/s at 8K/28K/56K depth against MTP3's 125/118/103, is
# greedy-lossless against ordinary decoding, and needs the smaller context because the
# companion weights and ring state cost about 1.9 GiB more than MTP. MTP stays the
# 100K profile (scripts/run-ninfer-4080.sh) because it is smaller; DFlash2 is the
# deeper-context speed option.
#
# The image carries only the binaries; the artifact is bind-mounted read-only. Rebuild
# the image after pulling source changes:
#
#   docker build -f Dockerfile -t ninfer-4080:gsq3 .
#
# Environment overrides: NINFER_IMAGE (default ninfer-4080:gsq3),
# NINFER_ARTIFACT (default <repo>/out/qwen3_8_27b_gsq3.ninfer),
# NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to keep the
# port host-local), NINFER_VISION (1 enables media and selects the vision cap),
# NINFER_CONTEXT (explicit token cap overrides both), NINFER_API_KEY (optional).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${NINFER_IMAGE:-ninfer-4080:gsq3}"
artifact="${NINFER_ARTIFACT:-$root/out/qwen3_8_27b_gsq3.ninfer}"
host_port="${NINFER_PORT:-8080}"
bind_address="${NINFER_BIND:-0.0.0.0}"
vision="${NINFER_VISION:-0}"

command -v docker >/dev/null 2>&1 || {
    echo "docker was not found on PATH. Install Docker Desktop or docker." >&2
    exit 1
}
if [[ ! -f "$artifact" ]]; then
    echo "Missing $artifact" >&2
    echo "Convert the 3-bit GSQ artifact with its DFlash2 companion first (docs/maintainer/rtx-4080-plan.md)." >&2
    exit 1
fi
artifact_dir="$(cd -- "$(dirname -- "$artifact")" && pwd)"
artifact_name="$(basename -- "$artifact")"

if [[ "$vision" == "1" ]]; then
    context="${NINFER_CONTEXT:-28672}"
    vision_args=(--vision)
else
    context="${NINFER_CONTEXT:-57344}"
    vision_args=()
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Building $image from Dockerfile (first run only, several minutes)..."
    docker build -f "$root/Dockerfile" -t "$image" "$root"
fi

api_args=()
if [[ -n "${NINFER_API_KEY:-}" ]]; then
    api_args=(--api-key "$NINFER_API_KEY")
fi

echo "Serving DFlash2 K=7 at $context tokens on http://$bind_address:$host_port/v1"
exec docker run --rm \
    --gpus all \
    --add-host=host.docker.internal:host-gateway \
    -p "$bind_address:$host_port:8080" \
    -v "$artifact_dir:/models:ro" \
    "$image" \
    ninfer-serve "/models/$artifact_name" \
    --host 0.0.0.0 --port 8080 \
    --max-context "$context" --kv-capacity "$context" --kv-dtype rk4v4-e8 \
    --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
    --host-kv-mib 4096 \
    --spec dflash2 --draft-tokens 7 --lm-head-draft \
    "${vision_args[@]+"${vision_args[@]}"}" \
    --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
    --presence-penalty 0 --frequency-penalty 0 \
    "${api_args[@]+"${api_args[@]}"}"
