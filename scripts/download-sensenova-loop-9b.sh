#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/sensenova_loop_9b.ninfer"

mkdir -p -- "$model_dir"
# Pinned to the published revision; override with NINFER_REVISION. The SHA-256 below is the
# content pin, so a moved tag or branch fails the check instead of silently changing the artifact.
revision="${NINFER_REVISION:-main}"
expected_sha256='e793869f6b4edb98ae1a8f08899a28f938ed77652796f5eaf572785cc48267e0'

printf '%s\n' "Downloading SenseNova-Loop-9B NInfer model (revision $revision)..."
if ! curl -L -C - --fail --output "$model" \
  "https://huggingface.co/roofkid/SenseNova-Loop-9B-NInfer/resolve/$revision/sensenova_loop_9b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
if [[ -z "${NINFER_SKIP_SHA256:-}" ]] && command -v sha256sum >/dev/null 2>&1; then
  printf '%s\n' 'Verifying SHA-256...'
  actual_sha256="$(sha256sum -- "$model" | cut -d' ' -f1)"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected_sha256" "$actual_sha256" >&2
    exit 1
  fi
fi
printf 'Model ready: %s\n' "$model"
