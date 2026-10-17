#!/bin/bash
# Build the patched vllm-xpu-kernels (int64 conv/ssm state offsets) inside the
# Intel deep-learning-essentials devel container, using the repo's canonical
# wheel build (setup.py -> cmake; build knobs default ON = full build).
# Produces (written back to this package dir, via the /cache mount):
#   build_out/_xpu_C.abi3.so              (main kernels lib, int64-fixed, RUNPATH $ORIGIN)
#   build_out/libgdn_attn_kernels_xe_2.so (SSM lib, int64-fixed)
# All source from git; no binaries carried.
set -euo pipefail
NPROC=${NPROC:-16}   # 32 OOMs on a 128 GB host
# MOUNT is where build.sh bind-mounts this package inside the build container;
# everything below is relative to it. Override only if you change build.sh.
MOUNT=${MOUNT:-/cache}
OUT="$MOUNT/build_out"
mkdir -p "$OUT"

echo "=== [1/6] toolchain ==="
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.1
export PATH="$CMPLR_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CMPLR_ROOT/lib:${LD_LIBRARY_PATH:-}"
[ -x "$CMPLR_ROOT/bin/icpx" ] || { echo "FATAL: icpx not found"; exit 1; }
icpx --version | head -1

echo "=== [2/6] python venv + torch 2.13.0+xpu (runtime ABI) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3.12-venv python3.12-dev patchelf >/dev/null 2>&1 || true
if [ ! -x "$MOUNT/venv/bin/python" ]; then
  python3.12 -m venv "$MOUNT/venv"
fi
PY="$MOUNT/venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet torch==2.13.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
"$PY" -c "import torch; assert torch.__version__.startswith('2.13.0'), torch.__version__" \
  && echo "torch 2.13.0+xpu OK"
# build requirements (pyproject [build-system] + runtime bits of setup.py)
"$PY" -m pip install --quiet "cmake>=3.26" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0" wheel regex jinja2 psutil build

echo "=== [3/6] install ocloc + matching IGC (AOT device-code link) ==="
dpkg -i \
    "$MOUNT"/driver-debs/intel-igc-core-2_2.41.5+22716_amd64.deb \
    "$MOUNT"/driver-debs/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
    "$MOUNT"/driver-debs/intel-ocloc_26.35.39758.10-0_amd64.deb >/dev/null 2>&1 || true
ldconfig
command -v ocloc >/dev/null && echo "ocloc OK"

echo "=== [4/6] clone + patch vllm-xpu-kernels v0.1.14 ==="
mkdir -p /workspace && cd /workspace
if [ ! -d vllm-xpu-kernels ]; then
  git clone -q --depth 1 --branch v0.1.14 \
    https://github.com/vllm-project/vllm-xpu-kernels.git
fi
cd vllm-xpu-kernels
git log -1 --format="source at %h %s"
# the 7-site int64 cast (idempotent — VERIFYs after patching):
python3 "$MOUNT"/scripts/patch_conv1d.py .

echo "=== [5/6] wheel build (full: all kernel libs, defaults ON) ==="
export MAX_JOBS="$NPROC"
"$PY" -m build --wheel --no-isolation --outdir "$OUT/wheel" .
ls -la "$OUT"/wheel/*.whl

echo "=== [6/6] extract + RUNPATH fix ==="
rm -rf "$OUT/whx" && mkdir -p "$OUT/whx"
"$PY" - <<PYX
import zipfile, glob, os
whl = glob.glob("$OUT/wheel/*.whl")[0]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.endswith("_xpu_C.abi3.so") or n.endswith("libgdn_attn_kernels_xe_2.so"):
            z.extract(n, "$OUT/whx")
            print("extracted", n)
PYX
XPU_C=$(find "$OUT/whx" -name "_xpu_C.abi3.so" | head -1)
GDN=$(find "$OUT/whx" -name "libgdn_attn_kernels_xe_2.so" | head -1)
[ -n "$XPU_C" ] && [ -n "$GDN" ] || { echo "FATAL: libs not found in wheel"; exit 1; }
# without $ORIGIN RUNPATH vLLM platform detection fails
# (libgrouped_gemm_xe_default.so: cannot open shared object file)
patchelf --set-rpath '$ORIGIN' "$XPU_C"
readelf -d "$XPU_C" | grep RUNPATH
cp "$XPU_C" "$OUT/_xpu_C.abi3.so"
cp "$GDN" "$OUT/libgdn_attn_kernels_xe_2.so"
sha256sum "$OUT"/_xpu_C.abi3.so "$OUT"/libgdn_attn_kernels_xe_2.so
echo "KERNELS_BUILD_OK"
