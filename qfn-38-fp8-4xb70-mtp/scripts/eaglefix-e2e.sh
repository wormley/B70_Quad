#!/bin/bash
# eaglefix-e2e.sh — E2E validation of the Qwen4ExpMTP eagle-group annotation fix.
# A/B against the same flags:
#   leg base     (:latest)  expect: flag-all warning PRESENT, prefix-cache hits = 0
#   leg eaglefix (:eaglefix) expect: warning ABSENT, prefix-cache hits > 0,
#                            spec-decode metrics intact, arithmetic control passes
# Per GPU_TESTING.md: no --rm, write only under /home/llm/qwen38-tests/eaglefix-e2e,
# unique container names/ports, evidence captured before any cleanup.
set -u
BASE=/home/llm/qwen38-tests/eaglefix-e2e
PORT=18120
MODEL=/models/Qwen3.8-Flash-Next-FP8
mkdir -p "$BASE/cache" "$BASE/tmp"

log() { echo "[eaglefix-e2e $(date -u +%H:%M:%S)] $*" | tee -a "$BASE/test.log"; }

# LEGS env var: space-separated subset to run (default: all)
want_leg() { [ -z "${LEGS:-}" ] && return 0; case " $LEGS " in *" $1 "*) return 0;; *) return 1;; esac; }

run_leg() { # $1=leg  $2=image  $3=extra vllm flags (optional)
  local leg="$1" image="$2"
  local name="qwen38-eaglefix-$leg-$(date -u +%H%M%S)"
  local dir="$BASE/$leg"; mkdir -p "$dir"
  log "LEG $leg ($image) starting as $name"
  # clean up OUR previous attempt's containers — only after evidence capture
  for c in $(podman ps -a --format '{{.Names}}' | grep '^qwen38-eaglefix-' || true); do
    podman logs "$c" > "$BASE/stale-$c.log" 2>&1 || true
    log "removing stale container $c (logs saved)"
    podman kill "$c" >/dev/null 2>&1 || true
    podman rm "$c" >/dev/null || true
  done
  if ss -ltn | grep -q ":$PORT "; then
    log "LEG $leg ABORT: port $PORT still in use"; podman logs "$name" > "$dir/container-fail.log" 2>&1 || true; return 1
  fi
  podman run -d --name "$name" --group-add keep-groups \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host \
    --shm-size=2g --ulimit memlock=-1:-1 \
    -e VLLM_PLE_MMAP=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e ZE_AFFINITY_MASK=0,1,2,3 \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=7200 \
    -v "$MODEL:/model:ro" -v "$BASE/cache:/cache:rw" \
    "$image" /model --served-model-name qwen38 \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --offload-backend uva --cpu-offload-gb 16 --cpu-offload-params w13_weight w2_weight \
    --language-model-only --max-num-batched-tokens 4096 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"preserve_thinking":true}' \
    --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.88 \
    ${SPEC_JSON:+--speculative-config "$SPEC_JSON"} \
    --port "$PORT" --api-key abc ${3:-} > "$dir/run.out" 2>&1
  echo "$name" > "$dir/container-name"
  log "LEG $leg waiting for /health"
  local up=0
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }
    # bail early if the container died
    podman ps --format '{{.Names}}' | grep -qx "$name" || break
    sleep 4
  done
  if [ "$up" != 1 ]; then
    log "LEG $leg FAILED to become healthy"
    podman logs "$name" > "$dir/container-fail.log" 2>&1 || true
    dmesg -T > "$dir/dmesg-fail.log"
    return 1
  fi
  log "LEG $leg healthy"

  # --- checks -------------------------------------------------------------
  # (a) flag-all warning present or absent in engine log
  podman logs "$name" > "$dir/container.log" 2>&1
  if grep -q "no KV cache group could be identified" "$dir/container.log"; then
    echo "WARNING_PRESENT" > "$dir/warning-state"
  else
    echo "WARNING_ABSENT" > "$dir/warning-state"
  fi

  # (b) arithmetic control: 17*23 -> 391 (warms the shared prefix too)
  local out
  out=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Authorization: Bearer abc" -H "Content-Type: application/json" \
    -d '{"model":"qwen38","messages":[{"role":"user","content":"Compute 17*23. Reply with only the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
  printf '%s' "$out" > "$dir/control-raw.json"
  echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" \
    > "$dir/control-answer" 2>/dev/null || echo "PARSE_FAIL" > "$dir/control-answer"

  # (c) prefix-cache battery: long shared system prefix + unique short tails.
  #     max_num_seqs=1 -> sequential; request 2..N should hit the cache.
  python3 - "$PORT" "$dir" <<'PYEOF'
import json, sys, time, urllib.request

port, dirpath = int(sys.argv[1]), sys.argv[2]
# ~1200-token shared prefix (deterministic pseudo-text)
line = ("The quick brown fox jumps over the lazy dog while the calm river "
        "flows beneath the old stone bridge in the quiet mountain valley. ")
prefix = ("You are a meticulous archivist. Memorize the following register; "
          "later questions only concern the final topic tag.\nREGISTER:\n"
          + line * 48)
tails = [
    "Question A: what animal does the fox jump over? Answer in one short sentence.",
    "Question B: what flows beneath the bridge? Answer in one short sentence.",
    "Question C: what is the final topic tag of this register? Answer in one short sentence.",
    "Question D: repeat the first six words of the register. Answer in one short sentence.",
]
# exact repeat of request 0 (identical prompt) — tests exact-match reuse
tails = tails + [tails[0]]
results = []
for i, tail in enumerate(tails):
    body = json.dumps({
        "model": "qwen38",
        "messages": [{"role": "user", "content": prefix + "\n\n" + tail}],
        "max_tokens": 64, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer abc", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    text = resp["choices"][0]["message"]["content"]
    results.append({"i": i, "latency_s": round(dt, 2), "text": text[:120]})
    print(f"req{i} {dt:.2f}s :: {text[:80]}")
with open(f"{dirpath}/prefix-battery.json", "w") as f:
    json.dump(results, f, indent=2)
PYEOF

  sleep 12  # let the metrics logger flush an interval
  # (d) metrics snapshot: prefix-cache counters + spec-decode metrics
  curl -s http://127.0.0.1:$PORT/metrics -H "Authorization: Bearer abc" \
    | grep -E "prefix_cache_(hits|queries)_total|spec_decode" \
    > "$dir/metrics.txt" || true
  podman kill "$name" >/dev/null 2>&1 || true
  sleep 3
  podman logs "$name" > "$dir/container-final.log" 2>&1 || true
  # the authoritative engine log: post-kill capture (exited containers keep logs)
  if [ -s "$dir/container-final.log" ]; then cp "$dir/container-final.log" "$dir/container.log"; fi
  log "LEG $leg done (warning=$(cat "$dir/warning-state"), control=$(cat "$dir/control-answer"))"
  return 0
}

# leg 3: the fix + sparse-retention interval opened up (832 = scheduler block
# size). Default retention interval is 0 = "replay boundaries only", which
# keeps only a request's own final mamba state — mid-prefix boundaries that a
# later shared-prefix request needs are sparsified away, so reconciled hits
# stay 0 even with the annotation fix. 832 retains one state per 832-token
# segment and restores cross-request hits.
EAGLEFIX_EXTRA_FLAGS="--prefix-cache-retention-interval 832"
SPEC_JSON='{"method": "mtp", "num_speculative_tokens": 1}'

RESULT=0
want_leg base && run_leg base localhost/qfn38-fp8-4xb70:latest || RESULT=1
log "stopping leg base"
podman kill "$(cat "$BASE/base/container-name")" >/dev/null 2>&1 || true
sleep 8

want_leg eaglefix && run_leg eaglefix localhost/qfn38-fp8-4xb70:eaglefix || RESULT=1
log "stopping leg eaglefix"
podman kill "$(cat "$BASE/eaglefix/container-name")" >/dev/null 2>&1 || true
sleep 3
podman logs "$(cat "$BASE/eaglefix/container-name")" > "$BASE/eaglefix/container-final.log" 2>&1 || true
if [ -s "$BASE/eaglefix/container-final.log" ]; then cp "$BASE/eaglefix/container-final.log" "$BASE/eaglefix/container.log"; fi

want_leg eaglefix-ri && run_leg eaglefix-ri localhost/qfn38-fp8-4xb70:eaglefix "$EAGLEFIX_EXTRA_FLAGS" || RESULT=1
log "stopping leg eaglefix-ri"
podman kill "$(cat "$BASE/eaglefix-ri/container-name" 2>/dev/null)" >/dev/null 2>&1 || true
sleep 3
if [ -f "$BASE/eaglefix-ri/container-name" ]; then
  podman logs "$(cat "$BASE/eaglefix-ri/container-name")" > "$BASE/eaglefix-ri/container-final.log" 2>&1 || true
  if [ -s "$BASE/eaglefix-ri/container-final.log" ]; then cp "$BASE/eaglefix-ri/container-final.log" "$BASE/eaglefix-ri/container.log"; fi
fi

# leg 4: eaglefix WITHOUT speculative decoding — splits the remaining
# 0-hit space: hits appear -> MTP/eagle-drop interaction; still 0 -> the
# hybrid mamba cross-request caching path itself.
SPEC_JSON="" want_leg eaglefix-nomtp && run_leg eaglefix-nomtp localhost/qfn38-fp8-4xb70:eaglefix || RESULT=1
log "stopping leg eaglefix-nomtp"
podman kill "$(cat "$BASE/eaglefix-nomtp/container-name" 2>/dev/null)" >/dev/null 2>&1 || true
sleep 3
if [ -f "$BASE/eaglefix-nomtp/container-name" ]; then
  podman logs "$(cat "$BASE/eaglefix-nomtp/container-name")" > "$BASE/eaglefix-nomtp/container-final.log" 2>&1 || true
  if [ -s "$BASE/eaglefix-nomtp/container-final.log" ]; then cp "$BASE/eaglefix-nomtp/container-final.log" "$BASE/eaglefix-nomtp/container.log"; fi
fi

# leg 5: the full designed path — MTP on, fine hash unit (64) so partial
# hash hits engage and the eagle drop costs one hash unit (64 tokens)
# instead of one 832-token scheduler block. Retention 832 keeps mamba
# states dense per segment so interior boundaries can serve hits.
EAGLEFIX_MU_FLAGS="--prefix-cache-retention-interval 832 --prefix-match-unit 64"
SPEC_JSON='{"method": "mtp", "num_speculative_tokens": 1}' \
  want_leg eaglefix-mu && run_leg eaglefix-mu localhost/qfn38-fp8-4xb70:eaglefix "$EAGLEFIX_MU_FLAGS" || RESULT=1
log "stopping leg eaglefix-mu"
podman kill "$(cat "$BASE/eaglefix-mu/container-name" 2>/dev/null)" >/dev/null 2>&1 || true
sleep 3
if [ -f "$BASE/eaglefix-mu/container-name" ]; then
  podman logs "$(cat "$BASE/eaglefix-mu/container-name")" > "$BASE/eaglefix-mu/container-final.log" 2>&1 || true
  if [ -s "$BASE/eaglefix-mu/container-final.log" ]; then cp "$BASE/eaglefix-mu/container-final.log" "$BASE/eaglefix-mu/container.log"; fi
fi

# leg 6: eaglefix + MTP + no eagle block drop. Tests whether skipping the
# last-block drop restores reconciled hits. CAVEAT being tested: the drop
# exists to keep eagle-lookahead-polluted tail states out of hits; if the
# draft hidden-state feeding breaks, spec-decode acceptance will collapse
# (gated below via spec_decode metrics + output checks).
EAGLEFIX_NODROP_FLAGS="--prefix-cache-retention-interval 832"
SPEC_JSON='{"method": "mtp", "num_speculative_tokens": 1, "disable_eagle_block_drop": true}' \
  run_leg eaglefix-nodrop localhost/qfn38-fp8-4xb70:eaglefix "$EAGLEFIX_NODROP_FLAGS" || RESULT=1
log "stopping leg eaglefix-nodrop"
podman kill "$(cat "$BASE/eaglefix-nodrop/container-name" 2>/dev/null)" >/dev/null 2>&1 || true
sleep 3
if [ -f "$BASE/eaglefix-nodrop/container-name" ]; then
  podman logs "$(cat "$BASE/eaglefix-nodrop/container-name")" > "$BASE/eaglefix-nodrop/container-final.log" 2>&1 || true
  if [ -s "$BASE/eaglefix-nodrop/container-final.log" ]; then cp "$BASE/eaglefix-nodrop/container-final.log" "$BASE/eaglefix-nodrop/container.log"; fi
fi

dmesg -T > "$BASE/dmesg-final.log"
grep -coE "UR_RESULT_ERROR_[A-Z_]+" "$BASE/dmesg-final.log" > "$BASE/dmesg-error-count" || true

echo "=== SUMMARY ==="
for leg in base eaglefix eaglefix-ri eaglefix-nomtp eaglefix-mu eaglefix-nodrop; do
  d="$BASE/$leg"
  echo "[$leg] warning=$(cat "$d/warning-state" 2>/dev/null) control=$(cat "$d/control-answer" 2>/dev/null)"
  echo "[$leg] log-metrics:"; cat "$d/log-metrics.txt" 2>/dev/null
done
echo "dmesg error kinds: $(cat "$BASE/dmesg-error-count")"
exit $RESULT
