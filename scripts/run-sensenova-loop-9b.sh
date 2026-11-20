#!/usr/bin/env bash
#
# SenseNova-Loop-9B on the RTX 4080, from a Windows host through Docker Desktop / WSL2 (or any
# Linux host with Docker and the NVIDIA Container Toolkit).
#
# Pulls the published image or builds it from this checkout, then serves the registered
# sensenova-loop-9b/groupwise-int artifact with the validated 256K profile:
#
#   * 262,144-token context, one lane, int8 KV (524,288 is the native limit)
#   * MTP speculative decoding, three draft tokens, n-gram chain; the release MTP has no
#     optimized shortlist head, so --lm-head-draft is not used
#   * vision enabled for image input (video is not supported by this target)
#   * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
#     presence/frequency penalties 0
#
# The image carries only the binaries; the artifact is bind-mounted read-only, so the container
# stays small and never contains model weights. The artifact is published at
# roofkid/SenseNova-Loop-9B-NInfer; download it with scripts/download-sensenova-loop-9b.sh (or
# .bat), or convert it with the converter (docs/maintainer/sensenova-loop-9b-artifact.md). The
# launcher looks for it in <repo>/out and then <repo>/models.
#
# Image resolution: an existing local image is used as-is. A missing image is pulled from the
# registry; if the pull fails and this checkout has a Dockerfile, the image is built from source.
# NINFER_BUILD=1 always builds from source. A locally built image carries the checkout revision
# (label org.ninfer.revision), so pulling source changes and re-running this launcher rebuilds
# that image instead of serving the previous build; published images have no label and are
# served as pulled. The published roofkid/ninfer-4080:sensenova tag (0.6.3-rtx4080) includes this
# target, so a pull normally serves the published binaries.
#
# Measured on this card (docs/maintainer/sensenova-loop-9b-model.md section 6): this 256K + Vision
# + MTP3 profile starts with about 0.35 GiB available after startup, so it stays one lane.
# 253K-token prefill is about 1.72k tok/s and 3.5K-token decode about 36 tok/s MTP0 and 58 tok/s
# MTP3 + n-gram. The plan's speed bars are not met, so this is a functional 256K profile, not a
# performance claim. The profile is fixed; edit this file to change it.
#
# The published port binds every host interface, so the profile is reachable from other machines
# on the network at http://<host-ip>:8080/v1 (allow the Docker Desktop firewall prompt on first
# use). Set NINFER_API_KEY to require a bearer token; the server only checks it when this is set.
#
# Environment overrides: NINFER_IMAGE (default roofkid/ninfer-4080:sensenova),
# NINFER_ARTIFACT (default <repo>/out or <repo>/models/sensenova_loop_9b.ninfer),
# NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to keep the port
# host-local), NINFER_API_KEY (optional), NINFER_BUILD=1 (force a source build),
# NINFER_KV_DTYPE (default int8; rk4v4-e8 is the fallback/debug mode).
set -euo pipefail

kv_dtype="${NINFER_KV_DTYPE:-int8}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${NINFER_IMAGE:-roofkid/ninfer-4080:sensenova}"
artifact="${NINFER_ARTIFACT:-}"
host_port="${NINFER_PORT:-8080}"
bind_address="${NINFER_BIND:-0.0.0.0}"

if [[ -z "$artifact" ]]; then
    if [[ -f "$root/out/sensenova_loop_9b.ninfer" ]]; then
        artifact="$root/out/sensenova_loop_9b.ninfer"
    else
        artifact="$root/models/sensenova_loop_9b.ninfer"
    fi
fi

command -v docker >/dev/null 2>&1 || {
    echo "docker was not found on PATH. Install Docker Desktop with the WSL2 backend." >&2
    exit 1
}
if [[ ! -f "$artifact" ]]; then
    echo "Missing $artifact" >&2
    echo "Download it with scripts/download-sensenova-loop-9b.sh, or point NINFER_ARTIFACT" >&2
    echo "at an existing sensenova_loop_9b.ninfer copy." >&2
    exit 1
fi
artifact_dir="$(cd -- "$(dirname -- "$artifact")" && pwd)"
artifact_name="$(basename -- "$artifact")"

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

echo "Serving SenseNova-Loop-9B ($kv_dtype KV) on http://$bind_address:$host_port/v1"
exec docker run --rm \
    --gpus all \
    --add-host=host.docker.internal:host-gateway \
    -p "$bind_address:$host_port:8080" \
    -v "$artifact_dir:/models:ro" \
    "$image" \
    ninfer-serve "/models/$artifact_name" \
    --host 0.0.0.0 --port 8080 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype "$kv_dtype" \
    --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 \
    --host-kv-mib 4096 \
    --spec mtp --draft-tokens 3 --ngram chain \
    --vision \
    --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
    --presence-penalty 0 --frequency-penalty 0 \
    "${api_args[@]+"${api_args[@]}"}"
