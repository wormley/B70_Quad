#!/bin/bash
# Build the patched vllm-xpu-kernels (int64 conv/ssm state offsets) inside the
# Intel deep-learning-essentials devel container. Produces:
#   build_out/_xpu_C.abi3.so         (main kernels lib, int64-fixed, RUNPATH $ORIGIN)
#   build_out/libgdn_attn_kernels_xe_2.so (SSM lib, int64-fixed)
# All source from git; no binaries carried.
set -euo pipefail
NPROC=${NPROC:-16}   # 32 OOMs on a 128 GB host
# MOUNT is where build.sh bind-mounts this package inside the build container;
# everything below is relative to it. Override only if you change build.sh.
MOUNT=${MOUNT:-/cache}
OUT="$MOUNT/build_out"
mkdir -p "$OUT"

echo "=== [1/5] toolchain ==="
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.1
export PATH="$CMPLR_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CMPLR_ROOT/lib:${LD_LIBRARY_PATH:-}"
[ -x "$CMPLR_ROOT/bin/icpx" ] || { echo "FATAL: icpx not found"; exit 1; }

echo "=== [2/5] python/torch (matches runtime image ABI) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3.12-venv python3.12-dev patchelf >/dev/null 2>&1 || true
if [ ! -x "$MOUNT/venv/bin/python" ]; then
  python3.12 -m venv "$MOUNT/venv"
  "$MOUNT/venv/bin/pip" install --quiet torch==2.13.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
fi
PY="$MOUNT/venv/bin/python"
$PY -c "import torch; assert torch.__version__.startswith('2.13.0'), torch.__version__" \
  && echo "torch 2.13.0+xpu OK"

echo "=== [3/5] install ocloc + matching IGC (AOT device-code link) ==="
dpkg -i \
    "$MOUNT"/driver-debs/intel-igc-core-2_2.41.5+22716_amd64.deb \
        "$MOUNT"/driver-debs/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
        "$MOUNT"/driver-debs/intel-ocloc_26.35.39758.10-0_amd64.deb >/dev/null 2>&1 || true
ldconfig
command -v ocloc >/dev/null && echo "ocloc OK"

echo "=== [4/5] clone + patch vllm-xpu-kernels v0.1.14 ==="
mkdir -p /workspace && cd /workspace
if [ ! -d vllm-xpu-kernels ]; then
  git clone --depth 1 --branch v0.1.14 \
    https://github.com/vllm-project/vllm-xpu-kernels.git
fi
cd vllm-xpu-kernels
git log -1 --format="source at %h %s"
# the 7-site int64 cast (idempotent):
python3 "$MOUNT"/scripts/patch_conv1d.py .
grep -c "static_cast<int64_t>" csrc/xpu/gdn_attn/causal_conv1d.hpp \
                             csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp \
                             csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp

echo "=== [5/5] build (full: _xpu_C + all kernel libs) ==="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=xpu \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake \
  -DVLLM_PYTHON_EXECUTABLE=$PY \
  -DVLLM_PYTHON_PATH=$($PY -c 'import sys;print(":".join(sys.path))') \
  -DBUILD_SYCL_TLA_KERNELS=ON -DVLLM_XPU_ENABLE_XE2=ON \
  -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF -DBASIC_KERNELS_ENABLED=OFF \
  -DFA2_KERNELS_ENABLED=OFF -DMOE_KERNELS_ENABLED=OFF \
  -DGDN_KERNELS_ENABLED=ON -DMQA_LOGITS_KERNELS_ENABLED=OFF \
  -DXPU_SPECIFIC_KERNELS_ENABLED=OFF -DXPUMEM_ALLOCATOR_ENABLED=OFF
cmake --build build --target _xpu_C -j"$NPROC" 2>&1 | tail -3
cmake --build build --target gdn_attn_kernels_xe_2 -j"$NPROC" 2>&1 | tail -3

ls -la build/_xpu_C.abi3.so build/libgdn_attn_kernels_xe_2.so
patchelf --set-rpath '$ORIGIN' build/_xpu_C.abi3.so
readelf -d build/_xpu_C.abi3.so | grep RUNPATH
cp build/_xpu_C.abi3.so "$OUT/_xpu_C.abi3.so"
cp build/libgdn_attn_kernels_xe_2.so "$OUT/libgdn_attn_kernels_xe_2.so"
sha256sum "$OUT"/*.so
echo "KERNELS_BUILD_OK"
