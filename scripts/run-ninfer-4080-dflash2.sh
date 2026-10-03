#!/usr/bin/env bash
#
# NInfer on the RTX 4080 with the DFlash2 draft model, Windows/Linux host through
# Docker Desktop or Docker.
#
# Pulls the published image and serves the DFlash2 profile at the context that fits the 16 GiB
# card with the requantized Q4 companion:
#
#   * 100,000-token context without vision, 65,536 with vision (measured caps; override
#     with NINFER_CONTEXT)
#   * DFlash2 block drafting with 7 draft tokens (K=7), LM-head proposal selector.
#     Session 16 put every DFlash2 width on the fast small-T route (K=1..15,
#     widths 2..16) with its A8 profile, but K=7 stays the measured window:
#     widths 9..16 pay a second 8-column tile (~1.7x round cost) and the extra
#     drafts do not add enough accepted tokens to pay for it (K=15 measures
#     79/197/154 tok/s at 8K/28K/56K, below these K=7 numbers at every depth).
#   * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
#     presence/frequency penalties 0. This engine has no multiplicative repeat penalty;
#     the neutral value 1 is its implicit behavior.
#
# Measured on this card with the session-26 build (docs/maintainer/rtx-4080-plan.md
# section 11; tiled corpus, rk4v4-e8, --prefill-chunk 1024, one repetition per point):
# DFlash2 K=7 decodes about 167/262/239/213 tok/s at 8K/32K/64K/98K depth against
# MTP3's 151/142/131/122, is greedy-lossless against ordinary decoding, and needs a
# smaller context than MTP because the companion weights and ring state cost about
# 0.8 GiB more. MTP stays the 100K profile (scripts/run-ninfer-4080.sh); DFlash2 is
# the deeper-context speed option.
#
# The image carries only the binaries; the artifact is bind-mounted read-only. Download the
# artifact first with scripts/download-qwen38-gsq3.sh (or .bat); the launcher looks for it in
# <repo>/out and then <repo>/models.
#
# Image resolution: an existing local image is used as-is. A missing image is pulled from the
# registry; if the pull fails and this checkout has a Dockerfile, the image is built from source.
# NINFER_BUILD=1 always builds from source. A locally built image carries the checkout revision
# (label org.ninfer.revision), so pulling source changes and re-running this launcher rebuilds
# that image instead of serving the previous build; published images have no label and are
# served as pulled.
#
# Environment overrides: NINFER_IMAGE (default roofkid/ninfer-4080:gsq3),
# NINFER_ARTIFACT (default <repo>/out or <repo>/models qwen3_8_27b_gsq3.ninfer),
# NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to keep the
# port host-local), NINFER_VISION (1 enables media and selects the vision cap),
# NINFER_CONTEXT (explicit token cap overrides both), NINFER_API_KEY (optional),
# NINFER_BUILD=1 (force a source build).
# NINFER_KV_DTYPE (default rk4v4-e8; int8 or rk8v4 trade context for MBPP-class accuracy).
set -euo pipefail
kv_dtype="${NINFER_KV_DTYPE:-rk4v4-e8}"

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${NINFER_IMAGE:-roofkid/ninfer-4080:gsq3}"
artifact="${NINFER_ARTIFACT:-}"
host_port="${NINFER_PORT:-8080}"
bind_address="${NINFER_BIND:-0.0.0.0}"
vision="${NINFER_VISION:-0}"

if [[ -z "$artifact" ]]; then
    if [[ -f "$root/out/qwen3_8_27b_gsq3.ninfer" ]]; then
        artifact="$root/out/qwen3_8_27b_gsq3.ninfer"
    else
        artifact="$root/models/qwen3_8_27b_gsq3.ninfer"
    fi
fi

command -v docker >/dev/null 2>&1 || {
    echo "docker was not found on PATH. Install Docker Desktop or docker." >&2
    exit 1
}
if [[ ! -f "$artifact" ]]; then
    echo "Missing $artifact" >&2
    echo "Download it first with scripts/download-qwen38-gsq3.sh (or .bat), or point" >&2
    echo "NINFER_ARTIFACT at an existing qwen3_8_27b_gsq3.ninfer copy." >&2
    exit 1
fi
artifact_dir="$(cd -- "$(dirname -- "$artifact")" && pwd)"
artifact_name="$(basename -- "$artifact")"

if [[ "$vision" == "1" ]]; then
    context="${NINFER_CONTEXT:-65536}"
    vision_args=(--vision)
else
    context="${NINFER_CONTEXT:-100000}"
    vision_args=()
fi

source_revision="$(git -C "$root" rev-parse --short=12 HEAD 2>/dev/null || true)"
if [[ -n "$source_revision" ]]; then
    dirty="$(git -C "$root" status --porcelain 2>/dev/null || true)"
    if [[ -n "$dirty" ]]; then
        source_revision="$source_revision-dirty"
    fi
fi

build_image=0
build_reason=""
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Pulling $image..."
    if ! docker pull "$image"; then
        if [[ -f "$root/Dockerfile" ]]; then
            build_image=1
            build_reason="pull failed and a source checkout is present"
        else
            echo "Could not pull $image and no Dockerfile is available to build it." >&2
            exit 1
        fi
    fi
elif [[ "${NINFER_BUILD:-0}" != "0" ]]; then
    build_image=1
    build_reason="NINFER_BUILD is set"
elif [[ -f "$root/Dockerfile" && -n "$source_revision" ]]; then
    image_revision="$(docker image inspect --format '{{ index .Config.Labels "org.ninfer.revision" }}' "$image" 2>/dev/null || true)"
    if [[ -n "$image_revision" && "$image_revision" != "$source_revision" ]]; then
        build_image=1
        build_reason="locally built image is from $image_revision, checkout is at $source_revision"
    fi
fi
if (( build_image )); then
    echo "Building $image from Dockerfile ($build_reason; several minutes)..."
    if [[ -n "$source_revision" ]]; then
        docker build --label "org.ninfer.revision=$source_revision" -f "$root/Dockerfile" -t "$image" "$root"
    else
        docker build -f "$root/Dockerfile" -t "$image" "$root"
    fi
fi

api_args=()
if [[ -n "${NINFER_API_KEY:-}" ]]; then
    api_args=(--api-key "$NINFER_API_KEY")
fi

echo "Serving DFlash2 K=7 at $context tokens ($kv_dtype KV) on http://$bind_address:$host_port/v1"
exec docker run --rm \
    --gpus all \
    --add-host=host.docker.internal:host-gateway \
    -p "$bind_address:$host_port:8080" \
    -v "$artifact_dir:/models:ro" \
    "$image" \
    ninfer-serve "/models/$artifact_name" \
    --host 0.0.0.0 --port 8080 \
    --max-context "$context" --kv-capacity "$context" --kv-dtype "$kv_dtype" \
    --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
    --host-kv-mib 4096 \
    --spec dflash2 --draft-tokens 7 --lm-head-draft \
    "${vision_args[@]+"${vision_args[@]}"}" \
    --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
    --presence-penalty 0 --frequency-penalty 0 \
    "${api_args[@]+"${api_args[@]}"}"
