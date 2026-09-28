#!/usr/bin/env bash
#
# NInfer on the RTX 4080, from a Windows host through Docker Desktop / WSL2.
#
# Builds (once) the product image from ./Dockerfile and serves the registered 3-bit
# GSQ artifact with the documented 100K profile:
#
#   * 102,400-token context, one lane, rk4v4-e8 KV (the registered 4080 fit)
#   * MTP speculative decoding, three draft tokens, LM-head draft route
#   * vision tower enabled (the GSQ3 artifact carries it)
#   * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
#     presence/frequency penalties 0. This engine has no multiplicative repeat
#     penalty; the neutral value 1 is its implicit behavior.
#
# The image carries only the binaries; the artifact is bind-mounted read-only, so
# the container stays small and never contains model weights. The image is built
# from the checkout as it exists now, which is what carries the current decode
# route; rebuild it after pulling source changes:
#
#   docker build -f Dockerfile -t ninfer-4080:gsq3 .
#
# Measured on this card with the session-10 build (docs/maintainer/rtx-4080-plan.md
# section 11): decode about 125/118/103 tok/s at 8K/32K/98K depth and prefill about
# 2700/2270/1675 tok/s. The profile is fixed; edit this file to change it.
#
# The published port binds every host interface, so the profile is reachable from
# other machines on the network at http://<host-ip>:8080/v1 (allow the Docker
# Desktop firewall prompt on first use). Set NINFER_API_KEY to require a bearer
# token; the server only checks it when this is set.
#
# Environment overrides: NINFER_IMAGE (default ninfer-4080:gsq3),
# NINFER_ARTIFACT (default <repo>/out/qwen3_8_27b_gsq3.ninfer),
# NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to keep
# the port host-local), NINFER_API_KEY (optional).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${NINFER_IMAGE:-ninfer-4080:gsq3}"
artifact="${NINFER_ARTIFACT:-$root/out/qwen3_8_27b_gsq3.ninfer}"
host_port="${NINFER_PORT:-8080}"
bind_address="${NINFER_BIND:-0.0.0.0}"

command -v docker >/dev/null 2>&1 || {
    echo "docker was not found on PATH. Install Docker Desktop with the WSL2 backend." >&2
    exit 1
}
if [[ ! -f "$artifact" ]]; then
    echo "Missing $artifact" >&2
    echo "Convert or copy the 3-bit GSQ artifact there first (its recipe is in docs/maintainer/rtx-4080-plan.md)." >&2
    exit 1
fi
artifact_dir="$(cd -- "$(dirname -- "$artifact")" && pwd)"
artifact_name="$(basename -- "$artifact")"

if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Building $image from Dockerfile (first run only, several minutes)..."
    docker build -f "$root/Dockerfile" -t "$image" "$root"
fi

api_args=()
if [[ -n "${NINFER_API_KEY:-}" ]]; then
    api_args=(--api-key "$NINFER_API_KEY")
fi

echo "Serving the Qwen3.8-27B GSQ3 profile on http://$bind_address:$host_port/v1"
exec docker run --rm \
    --gpus all \
    --add-host=host.docker.internal:host-gateway \
    -p "$bind_address:$host_port:8080" \
    -v "$artifact_dir:/models:ro" \
    "$image" \
    ninfer-serve "/models/$artifact_name" \
    --host 0.0.0.0 --port 8080 \
    --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 \
    --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 \
    --host-kv-mib 4096 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision --preserve-thinking \
    --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 \
    --presence-penalty 0 --frequency-penalty 0 \
    "${api_args[@]+"${api_args[@]}"}"
