# EAGLEFIX-RESULTS — Qwen4ExpMTP eagle-group annotation fix + prefix-cache/MTP interaction

Date: 2026-10-19 · Host: Apollo (4× Arc Pro B70) · Evidence: `/home/llm/qwen38-tests/eaglefix-e2e/`
Unit test: `scripts/test_eagle_annotation.py` (inside image, all 5 cases PASS)
E2E harness: `scripts/eaglefix-e2e.sh` (A/B legs, shared ~1.2K-token prefix battery + exact repeat + `17*23 → 391` control)

## What was fixed (and verified)

**Bug:** with `method=mtp`, vLLM could not identify the draft model's KV cache
group (the Qwen4ExpMTP draft layer reuses the target's full-attention
`Qwen4ExpDecoderLayer` — no spec marker; the positional fallback was gated to
`deepseek_v4` only). The coordinator then flagged **every** group — including
the 4 Mamba groups — as a draft group ("flag-all fallback"), silently
disabling cross-request prefix-cache reuse (0.0% hit rate) and blinding the
external KV offload tier.

**Fix:** `_is_qwen4_exp_eagle()` + `_EAGLE_LAST_LAYER_DRAFT_ARCHITECTURES` in
`vllm/v1/core/kv_cache_utils.py` — generalize the last-layer positional
fallback to drafts whose architectures include `Qwen4ExpMTP`. In
`base-image/fs/vllm/` (every fresh build) and as overlay `eaglefix-ctx/`
(`localhost/qfn38-fp8-4xb70:eaglefix` = `:latest` + the one file).

## A/B results (identical flags + battery per leg)

| leg | image / config | flag-all warning | prefix hits (tokens) | hit rate | outputs |
|---|---|---:|---:|---:|---|
| base | `:latest`, MTP | **present** | 0 | 0.0% | correct (391) |
| eaglefix | `:eaglefix`, MTP | **absent** | 0 | 0.0% | correct (391) |
| eaglefix-ri | + `--prefix-cache-retention-interval 832` | absent | 0 | 0.0% | correct (391) |
| eaglefix-mu | + `--prefix-match-unit 64` (+ ri) | absent | 0 | 0.0% | correct (391) |
| eaglefix-nomtp | MTP **off** | absent | **2496** | **32.9–39.6%** | correct (391) |
| eaglefix-nodrop | + `disable_eagle_block_drop:true` | present* | **2496** | 21.9–39.6% | **CORRUPT** |

\* `disable_eagle_block_drop` disables `use_eagle_block_drop()`, which also
skips the annotation — pre-existing gating, unrelated to the fix.

Corruption observed on hit-serving requests with the drop disabled:
`"beneath thethe bridge"` (duplicated token), `"final topic tag is nature"`
(wrong answer; correct is "the quiet mountain valley"). Latency also regressed
(req1 16.6 s vs 4.35 s baseline). The output gates caught it — HTTP 200 alone
proves nothing.

## Conclusion (root-cause chain, fully evidenced)

1. **Annotation bug — FIXED by us.** Without it: flag-all fallback mislabels
   the Mamba groups as draft groups; warning fires; reuse disabled; offload
   tier blind. With it: warning gone, groups annotated correctly. Necessary
   and correct, kept.
2. **Structural limitation — NOT config-fixable.** With MTP on this hybrid
   (GDN/Mamba + full-attn) model, the reconciled cross-request hit is always 0:
   - the eagle last-block drop backs every hit off by one unit, which always
     lands **below** the nearest 832-token mamba state boundary, and
   - mamba can only serve full-boundary states under eagle: partial-tail
     registration is gated `not use_eagle` (`scheduler.py`,
     `need_mamba_block_aligned_split`) — **correctly**, because leg 6 proves
     the tail state is real pollution: dropping the drop restores hits but
     corrupts outputs.
   Fine-grained hash units (`--prefix-match-unit 64`) and mamba retention
   (`--prefix-cache-retention-interval 832`) were tested and do not change
   this (legs 3/5).
3. **Deployment guidance.** MTP (≈2× decode via spec decode, ~95% acceptance)
   and cross-request prefix reuse (TTFT for shared-prefix traffic) are
   mutually exclusive on this stack today. Choose per workload: keep
   `--speculative-config` for decode-bound serving; drop it when serving
   heavy shared-prefix/multi-turn traffic (32.9–39.6% hit rates observed,
   ~3.8 s TTFT on a 1.2K-token cached prefix vs 4.15 s cold).

## Upstream-report material

- The annotation gap (`_annotate_eagle_groups` covers DSpark via spec marker
  and DeepseekV4 via positional rule, nothing else) — our fix generalizes the
  positional rule; the `FIXME(yifan)` comment upstream acknowledges the hack.
- The MTP×hybrid structural finding with the leg-6 corruption evidence is the
  strong part of an upstream report: the eagle drop and the `not use_eagle`
  partial-tail gate interact to zero out cross-request reuse for hybrid
  models, and neither can be safely bypassed.

## Hygiene

- dmesg clean after all legs (`UR_RESULT_ERROR_*` count: 0), no reboot required.
- No numerics touched; the fix is scheduler-layer python only.
- Unit test runs GPU-less inside the image:
  `podman run --rm --entrypoint python3 <img> /opt/eaglefix/test_eagle_annotation.py`
