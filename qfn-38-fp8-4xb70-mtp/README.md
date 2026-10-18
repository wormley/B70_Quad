# QFN-38 FP8 on 4× Intel Arc Pro B70 — MTP-enabled, 262K context

Reproducible build and serving recipe for running **Qwen3.8-Flash-Next-FP8**
at **262,144-token context** on **4× Intel Arc Pro B70 (32 GB each)** with
multi-token prediction (MTP1) speculative decoding and prefix caching enabled.

No pre-built binaries are carried: the build starts from the official vLLM
XPU image, applies source patches, and compiles the kernels from git.

## Requirements

- Linux host with **4× Intel Arc Pro B70** (or compatible Battlemage dGPUs, 32 GB each)
- **Podman** (or Docker) and network access (docker.io, github.com,
  download.pytorch.org, github.com/intel/compute-runtime releases)
- ~40 GB free disk for the build; ~60 GB RAM recommended at 262K context
- The model checkpoint **Qwen3.8-Flash-Next-FP8** (official release; contains
  the MTP1 module — no separate draft model needed)
- Build time: ~20 min (attention-kernel templates are pruned to the subset
  this model can reach; `KERNEL_CONF=full bash build.sh` restores the
  unpruned ~40-60 min build). Capped at 16 parallel jobs — higher OOMs on
  128 GB hosts.

## Build Instructions

```bash
bash build.sh              # produces localhost/qfn38-fp8-4xb70:latest
```

Driver debs need not be present: stage 0 downloads any missing ones from the
pinned upstream URLs and hash-verifies everything. Carried files are used
as-is (offline-friendly).

`bash build.sh clean` removes all build artifacts (`build_out/`, `venv/`,
`__pycache__/`, fetched `driver-debs/*.deb`) so the directory is pure source
for committing; everything is regenerated or re-fetched on the next build.

Three stages, all from source:

1. **Base image** — official `vllm/vllm-openai-xpu@sha256:e4446310…` (v0.30.0,
   torch 2.13.0+xpu, triton 3.7.2+xpu, vllm_xpu_kernels 0.1.14.1) + the vLLM
   python tree overlay (`base-image/fs/vllm/`) + the hash-locked Intel GPU
   userspace driver **26.35.39758.10** (`driver-debs/`, verified against
   `driver-packages.lock.json` by `install_driver.py`).
2. **Kernel build** — `git clone vllm-xpu-kernels v0.1.14` + the 7-site int64
   offset-cast patch (`scripts/patch_conv1d.py`) → `_xpu_C.abi3.so` +
   `libgdn_attn_kernels_xe_2.so`, compiled inside
   `intel/deep-learning-essentials:2026.1.2-devel` (icpx 2026.1, ocloc, AOT
   device-code link). `NPROC=16` by default; do not raise it on 128 GB hosts.
3. **Final image** — base + patched kernels, then a smoke check that vLLM
   platform detection works (verifies the `$ORIGIN` RUNPATH fix).

## Run Instructions

```bash
BASE=/path/to/workdir        # cache + tmp (must persist across reboots)
MODEL=/path/to/Qwen3.8-Flash-Next-FP8

podman run -d --name qfn38 --group-add keep-groups \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host \
  --shm-size=2g --ulimit memlock=-1:-1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_AFFINITY_MASK=0,1,2,3 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=7200 \
  -v "$MODEL:/model:ro" -v "$BASE/cache:/cache:rw" \
  localhost/qfn38-fp8-4xb70:latest /model --served-model-name qwen38 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --offload-backend uva --cpu-offload-gb 16 --cpu-offload-params w13_weight w2_weight \
  --language-model-only --max-num-batched-tokens 4096 \
  --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.88 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --port 18100 --api-key abc
```

Wait for `/health` (≈5 min: Triton JIT + PLE mmap warm-up), then test:

```bash
curl -s http://127.0.0.1:18100/v1/chat/completions -H "Authorization: Bearer abc" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Compute 17*23. Reply with only the number."}],"max_tokens":8}'
# expect "391"
```

Profile notes (each matters):

- `--cpu-offload-gb 16` with MTP, or `14` without MTP: the KV-pool size band
  is load-bearing. Below ~6.5 GB of KV pool the scheduler fragments deep
  prefills and 262K requests slow down 4× (see Open Issues).
- Prefix caching is ON by default and validated at 262K (requires the int64
  kernel fix below).
- For +10% decode add `VLLM_XPU_ENABLE_XPU_GRAPH=1` and remove
  `--enforce-eager` (graphs validated at TP=4, including with MTP).
- The PLE table is served directly from the checkpoint safetensors via mmap
  (`VLLM_PLE_MMAP=1`) — no separate 48 GB table on disk is needed.
- Startup takes ~5 min. Do not trust HTTP 200 alone; gate on exact retrieval
  (the needle check above or your own).

## Our Benchmarks

4× Arc Pro B70, TP=4, exact-retrieval needles at 4K/16K/64K/131K/262K —
20/20 ladder rows PASS across four configurations, 0 UR errors, 0 kernel
faults during requests.

### Prefill (tok/s = prompt tokens / TTFT)

| context | eager | graphs | MTP1 | MTP1 + graphs |
|---:|---:|---:|---:|---:|
| 4,112 | 512 | 500 | 481 | 463 |
| 16,400 | 544 | 541 | 490 | 489 |
| 65,558 | 535 | 534 | 483 | 482 |
| 131,086 | 521 | 520 | 472 | 470 |
| 262,030 | 494 | 494 | 449 | 447 |

### Decode (tok/s, 128-token generations at ~2K context, warm mean)

| eager | graphs | MTP1 | MTP1 + graphs |
|---:|---:|---:|---:|
| 16.5 | 18.2 (+10%) | 17.8 (+8–17%) | **20.3 (+23%)** |

MTP1 acceptance length ≈ 1.89 (drafted 213, accepted 190). The decode gains
from graphs and MTP compose near-multiplicatively.

## Fixes applied

| # | fix | origin |
|---|---|---|
| 1 | mmap-PLE: serve the PLE table from the checkpoint shards via `VLLM_PLE_MMAP=1` (removes the 48 GB host table and the host-RAM blowup) | **upstream** vLLM PR [#54129](https://github.com/vllm-project/vllm/pull/54129) — python tree overlay is its head commit `50a061f7` |
| 2 | built-in UVA expert offload (`--offload-backend uva --cpu-offload-gb/--cpu-offload-params`) replaces all custom expert-placement code | **upstream** vLLM (present in the v0.30.0 line the base tree is built on) |
| 3 | QSA XPU bring-up: pad the indexer decode kernel's head axis to N≥16 and mask padded heads; add a non-CUDA `_topk` fallback (torch.topk over visible blocks); allow `xpu` in the paged-QSA sparse-attention guard | **ours** — small patches to `vllm/models/qwen4_exp/` (in `base-image/fs/vllm/`) |
| 4 | Qwen4Exp XPU gate: drop the `is_xpu()` `NotImplementedError` in `qwen4_exp/__init__.py` | **community** B70 patch series (Qwen4Exp XPU enablement) |
| 5 | Intel GPU userspace driver **26.35.39758.10** (compute-runtime + IGC 2.41.5 + ocloc), hash-locked — removes the compute-runtime host-mirroring bug that corrupted long-context serving | **official** [intel/compute-runtime release](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10); selection motivated by our fault analysis |
| 6 | **int64 state-offset fix (the 4 GiB bug)**: 7 sites in vllm-xpu-kernels (`causal_conv1d.hpp`, `chunk_causal_conv1d_tiled_xe2.hpp`, `chunk_causal_conv1d_xe2.hpp`) computed conv/SSM state offsets as `id × stride` in **int32**, wrapping at 2^31 elements. With prefix caching ON the mamba conv-state pool is packed per KV block, so deep-context block ids cross the wrap → wild reads/writes → `-ENOENT` fault storms or silent corruption. This is why prefix caching could not be enabled at 262K. The conv kernels are header-only and compile into `_xpu_C.abi3.so`, so the fix requires the full kernels rebuild this script performs. | **ours** — patch in `scripts/patch_conv1d.py`; umbrella upstream issue [vllm-xpu-kernels #291](https://github.com/vllm-project/vllm-xpu-kernels/issues/291) ("Replace int with size_t"); **still unfixed in v0.1.15** |
| 7 | `patchelf --set-rpath '$ORIGIN'` on the rebuilt `_xpu_C.abi3.so` — without it the image fails vLLM platform detection (`libgrouped_gemv_xe_default.so: cannot open shared object file` → "Failed to infer device type") | **ours** — build fix, applied automatically in `build.sh` |
| 8 | MTP1: no code changes — the checkpoint ships the full MTP1 module and the tree resolves `qwen4_exp → qwen4_exp_mtp → Qwen4ExpMTP`; mamba/QSA state shapes already honor `num_speculative_tokens` | **upstream** |

## Open Issues

1. **KV-pool size band (deep-prefill collapse with MTP).** With a KV pool
   below ~6.5 GB the scheduler advances deep prefills in ~1,100-token steps
   instead of 4,096 while each step pays a fixed O(context) cost → 262K TTFT
   grows ~4× (e.g. 2,522 s vs 587 s). Workaround (in the Run Instructions):
   keep the pool out of the band (`--cpu-offload-gb 16` with MTP, `14`
   without). Root cause measured but not fixed; worth an upstream report.
2. **Startup fault bursts.** Some boots log a burst of `xe … Fault response:
   Unsuccessful -ENOENT/-EINVAL` lines during engine startup. Benign so far
   (engine healthy, outputs exact, zero faults during requests) but
   unexplained.
3. **Residual silent-corruption class** (upstream, cf. vllm-xpu-kernels
   #559): one 64K needle returned truncated output once in many ladders
   (allocator-state dependent, no faults, did not recur). Treat occasional
   wrong answers at long context as possible.
4. **XPU graphs are experimental upstream** (logged warning says
   "single-GPU only"); they work at TP=4 in this stack, including with MTP,
   but this is off the upstream-supported path.
5. The int32-offset bug remains in **upstream vllm-xpu-kernels v0.1.15** —
   rebuild from v0.1.14 + this patch until upstream merges the int64 fixes.
