#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done
for windows_script in "$root"/*.bat "$root"/*.ps1; do
  counterpart="${windows_script%.*}.sh"
  if [[ ! -x "$counterpart" ]]; then
    printf 'Missing Bash counterpart: %s\n' "$counterpart" >&2
    exit 1
  fi
  case "$windows_script" in
    */run-ninfer-4080*.bat)
      # The 4080 pair must agree on the served KV selector and the image-revision
      # rebuild contract; a Windows-only drift has shipped before.
      grep -q 'NINFER_KV_DTYPE' "$windows_script"
      grep -q 'org.ninfer.revision' "$windows_script"
      grep -q 'NINFER_KV_DTYPE' "$counterpart"
      grep -q 'org.ninfer.revision' "$counterpart"
      ;;
  esac
done

cat > "$tmp/ninfer-serve" <<'SERVER'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$NINFER_TEST_ARGS"
SERVER
chmod +x "$tmp/ninfer-serve"
touch "$tmp/qwen3_8_27b.ninfer" "$tmp/qwen3_6_35b_a3b.ninfer"

launchers=(
  'run-qwen38-c1.sh:qwen3_8_27b.ninfer:--max-context:65536'
  'run-qwen38-c8.sh:qwen3_8_27b.ninfer:--max-concurrency:8'
  'run-qwen38-vision.sh:qwen3_8_27b.ninfer:--vision:--spec'
  'run-qwen36-35b-vision.sh:qwen3_6_35b_a3b.ninfer:--vision:--no-thinking'
)
for entry in "${launchers[@]}"; do
  IFS=: read -r script model expected value <<< "$entry"
  args="$tmp/${script%.sh}.args"
  NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$args" \
    "$root/$script" "$tmp/$model" >/dev/null
  grep -Fx -- "$expected" "$args" >/dev/null
  grep -Fx -- "$value" "$args" >/dev/null
done

# The 4080 launchers build the product image once per checkout revision. Stub
# docker so the build decision and the served profile are checkable here.
mkdir -p -- "$tmp/docker-bin" "$tmp/docker-state/images" "$tmp/4080"
cat > "$tmp/docker-bin/docker" <<'DOCKER'
#!/usr/bin/env bash
# docker stub: image state lives in $DOCKER_STUB_DIR/images/<name>; build
# records the org.ninfer.revision label; run records its argv.
set -euo pipefail
state="$DOCKER_STUB_DIR"
name() { local n="${1##*/}"; printf '%s' "${n//:/__}"; }
case "${1:-}" in
  image)
    [[ "${2:-}" == inspect ]] || exit 0
    shift 2
    format=""
    image=""
    while (( $# )); do
      case "$1" in
        --format) format="$2"; shift 2 ;;
        *) image="$1"; shift ;;
      esac
    done
    [[ -f "$state/images/$(name "$image")" ]] || exit 1
    if [[ -n "$format" ]]; then cat "$state/images/$(name "$image")"; fi
    ;;
  build)
    shift
    label=""
    tag=""
    while (( $# )); do
      case "$1" in
        --label) label="${2#org.ninfer.revision=}"; shift 2 ;;
        -t) tag="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '%s' "$label" > "$state/images/$(name "$tag")"
    printf 'build %s %s\n' "$tag" "$label" >> "$state/log"
    ;;
  run)
    shift
    printf '%s\n' "$@" > "$state/run-args"
    printf 'run\n' >> "$state/log"
    ;;
esac
DOCKER
chmod +x "$tmp/docker-bin/docker"
: > "$tmp/4080/artifact.ninfer"

launch_4080() {
  PATH="$tmp/docker-bin:$PATH" DOCKER_STUB_DIR="$tmp/docker-state" \
    NINFER_ARTIFACT="$tmp/4080/artifact.ninfer" "$root/$1" >/dev/null 2>&1
}

# First run builds and serves; a matching image label reuses the image.
launch_4080 run-ninfer-4080.sh
[[ "$(grep -c '^build ' "$tmp/docker-state/log")" == 1 ]]
launch_4080 run-ninfer-4080.sh
[[ "$(grep -c '^build ' "$tmp/docker-state/log")" == 1 ]]
grep -Fx -- '--spec' "$tmp/docker-state/run-args" >/dev/null
grep -Fx -- 'mtp' "$tmp/docker-state/run-args" >/dev/null
grep -Fx -- '--vision' "$tmp/docker-state/run-args" >/dev/null

# A stale label rebuilds; the DFlash2 launcher serves its own profile.
printf 'stale' > "$tmp/docker-state/images/ninfer-4080__gsq3"
launch_4080 run-ninfer-4080.sh
[[ "$(grep -c '^build ' "$tmp/docker-state/log")" == 2 ]]
launch_4080 run-ninfer-4080-dflash2.sh
[[ "$(grep -c '^build ' "$tmp/docker-state/log")" == 2 ]]
grep -Fx -- 'dflash2' "$tmp/docker-state/run-args" >/dev/null
grep -Fx -- '7' "$tmp/docker-state/run-args" >/dev/null
grep -Fx -- '--max-context' "$tmp/docker-state/run-args" >/dev/null

# NINFER_KV_DTYPE selects the served KV mode; the default stays rk4v4-e8.
PATH="$tmp/docker-bin:$PATH" DOCKER_STUB_DIR="$tmp/docker-state" \
  NINFER_ARTIFACT="$tmp/4080/artifact.ninfer" NINFER_KV_DTYPE=int8 \
  "$root/run-ninfer-4080-dflash2.sh" >/dev/null 2>&1
grep -Fx -- '--kv-dtype' "$tmp/docker-state/run-args" >/dev/null
grep -Fx -- 'int8' "$tmp/docker-state/run-args" >/dev/null
launch_4080 run-ninfer-4080.sh
grep -Fx -- 'rk4v4-e8' "$tmp/docker-state/run-args" >/dev/null

mkdir -- "$tmp/bin" "$tmp/models"
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
while (( $# )); do
  if [[ "$1" == '--output' ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
: > "$output"
CURL
chmod +x "$tmp/bin/curl"
# The stub curl writes an empty file, so the pinned SHA-256 check is skipped here.
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 "$root/download-qwen38.sh" >/dev/null
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" "$root/download-qwen36-35b-vision.sh" >/dev/null
[[ -f "$tmp/models/qwen3_8_27b.ninfer" ]]
[[ -f "$tmp/models/qwen3_6_35b_a3b.ninfer" ]]

printf '%s\n' 'Linux script checks passed.'
