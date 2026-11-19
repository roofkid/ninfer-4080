#!/usr/bin/env bash
#
# SenseNova-Loop-9B on one RTX 4080 (sm_89, 16 GiB): the validated 256K profile.
#
# Serves the registered `sensenova-loop-9b/groupwise-int` artifact with:
#
#   * a 262,144-token context on int8 KV (524,288 is the native limit; 256K is
#     the validated launch profile)
#   * vision enabled for image input (video is not supported by this target)
#   * MTP3 speculative decoding with the n-gram chain; the release MTP has no
#     optimized shortlist head, so --lm-head-draft is not used
#   * one lane, 16 pending requests, 1,024-token prefill chunks, 4 GiB host KV
#
# Measured on this card (docs/maintainer/sensenova-loop-9b-model.md section 6):
# this 256K + Vision + MTP3 profile starts with about 0.35 GiB available after
# startup, so it stays one lane. 253K-token prefill is about 1.72k tok/s and
# 3.5K-token decode about 36 tok/s MTP0 and 58 tok/s MTP3 + n-gram. The plan's
# speed bars are not met, so this is a functional 256K profile, not a
# performance claim. The profile is fixed; edit this file to change it.
#
# The artifact is produced locally by the converter (see
# docs/maintainer/sensenova-loop-9b-artifact.md). The launcher looks for it in
# <repo>/out and then <repo>/models, and accepts an explicit path as its first
# argument.
#
# Environment overrides: NINFER_SERVER (default ninfer-serve beside this script,
# then <repo>/build/apps/ninfer-serve), NINFER_MODEL_DIR (default <repo>/out when
# it holds sensenova_loop_9b.ninfer, otherwise <repo>/models).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
server="${NINFER_SERVER:-$root/ninfer-serve}"
if [[ ! -x "$server" && -x "$root/../build/apps/ninfer-serve" ]]; then
  server="$root/../build/apps/ninfer-serve"
fi
model_dir="$root/../models"
if [[ -f "$root/../out/sensenova_loop_9b.ninfer" ]]; then
  model_dir="$root/../out"
fi
model="${1:-${NINFER_MODEL_DIR:-$model_dir}/sensenova_loop_9b.ninfer}"

if [[ ! -x "$server" ]]; then
  printf 'Missing executable: %s\n' "$server" >&2
  printf '%s\n' 'Set NINFER_SERVER or build build/apps/ninfer-serve.' >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'Missing model: %s\n' "$model" >&2
  printf '%s\n' 'Convert it with tools/convert/sensenova_loop_9b, or pass the artifact path.' >&2
  exit 1
fi

printf '%s\n' 'Starting SenseNova-Loop-9B at http://127.0.0.1:8080/v1'
printf '%s\n' 'Profile: one lane, 256K context, int8 KV, vision, MTP3 + n-gram chain'
exec "$server" "$model" \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --host-kv-mib 4096 \
  --spec mtp --draft-tokens 3 --ngram chain \
  --vision
