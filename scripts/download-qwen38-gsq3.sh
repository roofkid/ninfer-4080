#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_8_27b_gsq3.ninfer"

mkdir -p -- "$model_dir"
# Pinned to the revision that published the artifact; override with NINFER_REVISION.
# The SHA-256 check below is the content pin.
revision="${NINFER_REVISION:-2359d374b0f6ed3cd400815e5114d32ce03b5899}"
expected_sha256='c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7'

printf '%s\n' "Downloading Qwen3.8-27B GSQ3 NInfer model (revision $revision)..."
if ! curl -L -C - --fail --output "$model" \
  "https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer/resolve/$revision/qwen3_8_27b_gsq3.ninfer"; then
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
