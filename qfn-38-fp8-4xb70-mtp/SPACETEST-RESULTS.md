# SPACETEST-RESULTS.md — small-context output-correctness A/B test

**Question:** the user reports output glitches (missing spaces / glued tokens)
at contexts ≤ 32K. Is the serving stack responding properly?

**Test:** `scripts/qfn38-spacetest.sh` + `scripts/qfn38_space_battery.py`
(~110 deterministic requests per leg, temperature 0, streaming, unique nonce
per request so prefix-cache reuse cannot mask anything):

- **Exact-output tasks ×3 reps each** (byte-compared against canonical
  expected strings — the direct probe for missing spaces):
  verbatim copy with punctuation/apostrophes, comma list `1..30`,
  hyphenated spelling `n-e-c-e-s-s-a-r-y`, spaced alphabet, `17*23 → 391`.
- **Context ladder:** needle retrieval + verbatim copy embedded at ~4K / ~16K /
  ~32K.
- **Parity paragraphs** in two styles: battery-style (`enable_thinking:false`)
  and user-style (server `preserve_thinking:true` template + qwen3 reasoning
  parser, request identical to the user's clients), full text captured.
- **A/B isolation:** identical battery against the same image with MTP1 on
  (leg `mtp`) and off (leg `nomtp`).

**Profile note:** legs match the user's launch (`/home/llm/vllm-qfnfp8.sh`):
`VLLM_XPU_ENABLE_XPU_GRAPH=1` (graphs on — verified engaged, no
"disabled by environment variable" warning; 6–13 capture log lines),
tool-call parser qwen3_xml, reasoning parser qwen3, `preserve_thinking:true`,
offload 16 / util 0.88 / batch 4096, MTP1 on leg `mtp`. (Earlier runs 1–2
used the README profile without the graph env = eager; results below are
run 3, the user's profile.)

## Result — glitches did NOT reproduce

| leg | pass/fail | notes |
|---|---|---|
| `mtp` (user profile, MTP1 on, graphs) | **22/22 PASS** | all exact tasks byte-perfect, needle 4K/16K/32K OK |
| `nomtp` (same, MTP off) | 21/22 | single FAIL = model **refused** at 16K (see below) |

- Every spacing-sensitive task (`COPY`, `LIST`, `SPELL`, `ALPHA`, arithmetic)
  returned **byte-identical, correctly-spaced text in both legs, 3/3 reps** —
  zero missing spaces, zero glued tokens at any tested context.
- Verbatim copy recovered exactly at 16K and 32K (`COPY16K`/`COPY32K` PASS in
  both legs), i.e. even deep in the filler the detokenizer path is intact.
- Decode under MTP+graphs is clean: `17*23 → 391` at ~0.7 s TTFT, all reps.
- 0 `UR_RESULT_*` errors, dmesg fault count unchanged before/after each leg,
  host RAM healthy, no reboots needed.

**The one FAIL is not a glitch:** `NEED16K` in the `nomtp` leg returned
`"I cannot provide the secret access code."` while `NEED4K`/`NEED32K` passed
in the same leg and `mtp` passed 16K. Same refusal in run 2 — deterministic
model-behavior refusal at that sampling state, not text corruption. Worth
knowing, harmless for the stack question.

## Caveats recorded

1. **Cross-boot nondeterminism at temperature 0:** the parity paragraph
   differs between legs and between runs 2/3, but every variant is
   well-formed English with correct spacing. This is TP/numeric
   nondeterminism, not the missing-space class. Same-boot same-prompt outputs
   were always identical.
2. **`USERPARITY` outputs truncate mid-sentence at max_tokens 220** because
   with `preserve_thinking:true` the budget is partly consumed by thinking;
   the captured answer fragments were still correctly spaced in both legs.
3. The `GLUE:` annotations in the logs are detector false positives on real
   long words (`communication`, `infrastructure`, `groundbreaking`).

## Conclusion

The image `localhost/qfn38-fp8-4xb70:latest` (same build as the user's
`registry.wormley.com/qfn38-fp8-4xb70:latest`) **responds properly at ≤ 32K**
under the user's exact serving profile, with and without MTP: no missing
spaces, no glued tokens, exact retrieval, no kernel faults.

## Where the evidence is

- Apollo: `/home/llm/qwen38-tests/qfn38-spacetest/`
  (`mtp/`, `nomtp/` = run 3 legs; `mtp-eager/`, `nomtp-eager/` = run 2 eager
  legs; `mtp-run1/`, `nomtp-run1/` = run 1 partial; `parity.txt`,
  `ab-diff.txt`, `dmesg-teststart/testend.log`).
- Reboot: one, logged in `reboot.log` (2026-10-17T21:36Z) — caused by an
  orchestrator bug (`cache/` mountpoint missing) before any GPU work ran.

## Suggested next step if glitches persist for the user

The failure did not reproduce server-side with deterministic requests, so the
next discriminator is the client path: capture the **raw SSE stream** from the
user's real client (before any UI rendering) for a glitched response, and
replay it verbatim against the server. If the raw stream is correct, the
glitch is in client-side rendering/incremental-detokenizer handling of the
qwen3 reasoning-parser deltas; if the raw stream is broken, send the exact
request payload + `x-request-id` and we bisect from there.