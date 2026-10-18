#!/bin/bash
# Build the complete qfn38-fp8-4xb70 image from source.
#
#   bash build.sh [TAG]        # or: IMAGE=localhost/qfn38:dev bash build.sh
#   Default tag: localhost/qfn38-fp8-4xb70:latest (a positional TAG overrides
#   the IMAGE env var). The intermediate base image is <TAG-root>:base.
#   bash build.sh clean        # remove all build artifacts (repo prep)
#
# Stages:
#   1. base image   — official vLLM 0.30.0 XPU image + python tree overlay
#                     (base-image/fs/vllm) + hash-locked Intel driver debs
#   2. kernel build — vllm-xpu-kernels v0.1.14 from git + 7-site int64 offset
#                     cast (scripts/patch_conv1d.py) → _xpu_C.abi3.so + SSM lib
#   3. final image  — stage 1 + stage 2 artifacts + platform-detection check
#
# Artifacts land in ./build_out (compiled kernels) and ./venv (build venv,
# reused on re-runs; delete freely).
#
# Prerequisites: podman (or set CONTAINER_ENGINE=docker), network access,
# ~40 GB disk. NPROC defaults to 16 — do not raise it on 128 GB hosts
# (the SYCL AOT compile OOMs at 32).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-${IMAGE:-localhost/qfn38-fp8-4xb70:latest}}"
BASE_TAG="${TAG%:*}:base"
NPROC="${NPROC:-16}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
DEVEL_IMAGE="docker.io/intel/deep-learning-essentials:2026.1.2-devel-ubuntu24.04"

cd "$SCRIPT_DIR"

# Repo prep: remove everything the build creates so only source remains
# (driver debs are re-fetched on the next build; images are not touched).
if [ "${1:-}" = "clean" ]; then
  echo "cleaning build artifacts..."
  rm -rf build_out venv
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  rm -f driver-debs/*.deb
  echo "clean done: build_out/, venv/, __pycache__/, driver-debs/*.deb removed"
  exit 0
fi

mkdir -p build_out

echo "=========================================================="
echo "Stage 0/3: driver debs (fetch if missing, hash-verify all)"
echo "=========================================================="
python3 scripts/fetch_debs.py

echo "=========================================================="
echo "Stage 1/3: base image (official vLLM 0.30.0 + python tree)"
echo "=========================================================="
$CONTAINER_ENGINE build --no-cache -f base-image/Dockerfile \
  -t "$BASE_TAG" .

echo "=========================================================="
echo "Stage 2/3: kernel build (vllm-xpu-kernels + int64 cast)"
echo "=========================================================="
# The package dir is mounted read-write at /cache inside the build container:
# the kernel script reads scripts/patch_conv1d.py and driver-debs/ from there
# and writes build_out/ back.
$CONTAINER_ENGINE run --rm \
  -v "$SCRIPT_DIR:/cache" \
  -e NPROC="$NPROC" \
  --entrypoint bash "$DEVEL_IMAGE" /cache/scripts/build_kernels.sh

echo "=========================================================="
echo "Stage 3/3: final image"
echo "=========================================================="
cat > build_out/Dockerfile.final <<EOF
FROM $BASE_TAG
COPY build_out/_xpu_C.abi3.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
COPY build_out/libgdn_attn_kernels_xe_2.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
EOF
$CONTAINER_ENGINE build --no-cache -f build_out/Dockerfile.final -t "$TAG" .

# Verify platform detection (catches a missing $ORIGIN RUNPATH):
$CONTAINER_ENGINE run --rm --group-add keep-groups \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=0,1,2,3 \
  --entrypoint bash "$TAG" \
  -c "cd /tmp && python3 -c \"
from vllm.platforms import current_platform
dt = current_platform.device_type
assert dt == 'xpu', f'platform detection failed: {dt!r}'
print('platform:', dt)
import torch
print('devices:', torch.xpu.device_count())
\""

echo ""
echo "=========================================================="
echo "BUILD OK: $TAG"
echo "  base:     vllm/vllm-openai-xpu@e4446310 (v0.30.0+xpu)"
echo "  python:   #54129 head + QSA XPU fixes (base-image/fs/vllm)"
echo "  driver:   26.35.39758.10 (hash-locked debs)"
echo "  kernels:  vllm-xpu-kernels v0.1.14 + 7-site int64 cast"
echo "  MTP:      Qwen4ExpMTP (checkpoint module), prefix caching ON"
echo "=========================================================="
