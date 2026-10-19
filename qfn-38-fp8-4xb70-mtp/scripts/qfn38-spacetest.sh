#!/bin/bash
# qfn38-spacetest.sh — A/B output-correctness test for the qfn38-fp8-4xb70 image.
# Runs the identical spacing/corruption battery against the image twice —
# leg A: exactly the README profile (MTP1 speculative decoding on),
# leg B: the same flags with MTP removed — then compares.
# Deterministic (temperature 0), unique nonce per request, contexts <= 32K.
set -u
BASE=/home/llm/qwen38-tests/qfn38-spacetest
IMAGE=localhost/qfn38-fp8-4xb70:latest
PORT=18260
SPEC='--speculative-config={"method": "mtp", "num_speculative_tokens": 1}'
mkdir -p "$BASE/cache" "$BASE/tmp"

log() { echo "[spacetest $(date -u +%H:%M:%S)] $*" | tee -a "$BASE/test.log"; }

run_leg() { # $1=leg-name  $2=spec-arg-or-empty
  local leg="$1" spec="$2" name="qwen38-space-$1"
  local dir="$BASE/$leg"
  mkdir -p "$dir"
  log "LEG $leg starting (spec=${spec:-none})"
  dmesg -T > "$dir/dmesg-before.log"
  local args=(run -d --name "$name" --group-add keep-groups
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host
    --shm-size=2g --ulimit memlock=-1:-1
    -e VLLM_PLE_MMAP=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e ZE_AFFINITY_MASK=0,1,2,3
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1
    -v /home/llm/Qwen3.8-Flash-Next-FP8:/model:ro -v "$BASE/cache:/cache:rw"
    "$IMAGE" /model --served-model-name qwen38
    --tensor-parallel-size 4 --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --offload-backend uva --cpu-offload-gb 16 --cpu-offload-params w13_weight w2_weight
    --language-model-only --max-num-batched-tokens 4096
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"preserve_thinking":true}'
    --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.88
    --port "$PORT" --api-key abc)
  [ -n "$spec" ] && args+=("$spec")
  podman "${args[@]}" > "$dir/podman.out" 2>&1
  echo "$name" > "$dir/contname"
  local ok=0
  for i in $(seq 1 250); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
    # bail out early if the container died instead of spinning blindly
    local st
    st=$(podman inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo gone)
    if [ "$st" != "running" ]; then
      log "LEG $leg container state=$st after $((i*4))s — aborting wait"
      break
    fi
    sleep 4
  done
  if [ "$ok" != 1 ]; then
    log "LEG $leg server never became healthy"
    podman ps -a --format '{{.Names}} {{.Status}}' | grep "$name"
    podman logs "$name" > "$dir/container.log" 2>&1
    dmesg -T > "$dir/dmesg-after.log"
    return 2
  fi
  log "LEG $leg healthy; running battery"
  python3 "$BASE/qfn38_space_battery.py" "$PORT" "leg-$leg" \
    2>&1 | tee "$dir/battery.log"
  local rc=${PIPESTATUS[0]}
  podman logs "$name" > "$dir/container.log" 2>&1
  # graphs engagement check: must NOT find the disabled-warning
  if grep -q "XPU Graph is disabled by environment variable" "$dir/container.log"; then
    log "LEG $leg GRAPHS-NOT-ENGAGED (disabled warning present)"
  else
    grep -ciE "capture.*(graph|shape)" "$dir/container.log" >/dev/null &&
      log "LEG $leg graphs: no disabled-warning; $(grep -ciE 'Capturing' "$dir/container.log") capture log lines"
  fi
  podman stop "$name" >/dev/null 2>&1
  podman logs "$name" > "$dir/container.log" 2>&1
  dmesg -T > "$dir/dmesg-after.log"
  grep -oE "UR_RESULT_ERROR_[A-Z_]+" "$dir/container.log" "$dir/battery.log" 2>/dev/null \
    | sort | uniq -c > "$dir/ur-errors.txt"
  if [ -s "$dir/ur-errors.txt" ]; then log "LEG $leg UR ERRORS:"; cat "$dir/ur-errors.txt"; fi
  local faults
  faults=$(diff <(grep -cE "Fault response|Engine reset" "$dir/dmesg-before.log") \
                <(grep -cE "Fault response|Engine reset" "$dir/dmesg-after.log") >/dev/null \
           && echo none || echo "NEW-FAULTS-SEE-DIR")
  log "LEG $leg battery_rc=$rc dmesg_faults=$faults"
  return $rc
}

dmesg -T > "$BASE/dmesg-teststart.log"

run_leg mtp "$SPEC";      rcA=$?
run_leg nomtp "";         rcB=$?

log "==== COMPARISON ===="
for f in "$BASE"/mtp/battery.log "$BASE"/nomtp/battery.log; do
  [ -f "$f" ] && grep -E "SUMMARY" "$f"
done
norm() { sed 's/leg-mtp/LEG/g; s/leg-nomtp/LEG/g'; }
if [ -f "$BASE/mtp/battery.log" ] && [ -f "$BASE/nomtp/battery.log" ]; then
  diff <(norm < "$BASE/mtp/battery.log") <(norm < "$BASE/nomtp/battery.log") > "$BASE/ab-diff.txt" && \
    log "A/B battery logs IDENTICAL (after tag normalization)" || \
    log "A/B battery logs DIFFER — see $BASE/ab-diff.txt"
  grep -h "PARITY_TEXT" "$BASE"/mtp/battery.log "$BASE"/nomtp/battery.log > "$BASE/parity.txt" 2>/dev/null
  if [ -s "$BASE/parity.txt" ]; then
    if [ "$(sed -n 1p "$BASE/parity.txt" | sed 's/^leg-mtp //')" = \
         "$(sed -n 2p "$BASE/parity.txt" | sed 's/^leg-nomtp //')" ]; then
      log "PARITY paragraph: outputs IDENTICAL across legs"
    else
      log "PARITY paragraph: outputs DIFFER across legs (see parity.txt)"
    fi
  fi
fi
dmesg -T > "$BASE/dmesg-testend.log"
log "TEST DONE rcA=$rcA rcB=$rcB"
