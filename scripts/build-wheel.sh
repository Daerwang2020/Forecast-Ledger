#!/usr/bin/env bash
# Build from an internal temporary staging area so AppleDouble files on external
# drives cannot enter package metadata or archives.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
output_dir="${1:-$project_dir/dist}"
python_bin="${TS_REPRO_PYTHON:-python3}"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/ts-repro-build.XXXXXX")"

mkdir -p "$output_dir"
rsync -a \
  --exclude='._*' \
  --exclude='build/' \
  --exclude='*.egg-info/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.bin' \
  "$project_dir/" "$stage_dir/"
"$python_bin" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$output_dir" "$stage_dir"
echo "wheel written to $output_dir"
