#!/usr/bin/env python3
# qfn38_space_battery.py — spacing/corruption battery for the qfn38-fp8-4xb70 stack.
# Usage: python3 qfn38_space_battery.py <port> <tag>
#
# Hunts the "missing spaces / glued tokens" glitch class at small context:
#   - exact-output tasks (verbatim copy, comma list, hyphen spelling, spaced
#     alphabet, arithmetic) compared against canonical expected strings,
#     3 reps each with a unique nonce per request (no prefix-cache reuse);
#   - needle retrieval + verbatim copy embedded at ~4K / ~16K / ~32K context;
#   - one fixed parity paragraph (identical prompt across A/B legs) printed in
#     full so the orchestrator can diff MTP-on vs MTP-off outputs;
#   - glued-word detector (>=13-letter alphabetic tokens) on every output.
# All requests: temperature 0, stream:true, enable_thinking:false.
import json, sys, time, re, urllib.request

PORT, TAG = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:%s" % PORT
H = {"Content-Type": "application/json", "Authorization": "Bearer abc"}
F = "Routine inventory notes and maintenance logs from regional depots. "
NPASS = NFAIL = 0

COPY_TEXT = "The quick brown fox jumps over the lazy dog; it isn't 8 o'clock yet, is it?"
COPY_TASK = ("Output the following text exactly, with unchanged spacing and "
             "punctuation, and nothing else:\n\n" + COPY_TEXT)
LIST_EXPECT = ", ".join(str(i) for i in range(1, 31))
SPELL_EXPECT = "n-e-c-e-s-s-a-r-y"
ALPHA_EXPECT = " ".join("abcdefghijklmnopqrstuvwxyz")
NEEDLE_Q = "Question: What is the secret access code? Reply with only the code."


def say(name, ok, detail=""):
    global NPASS, NFAIL
    NPASS, NFAIL = NPASS + (1 if ok else 0), NFAIL + (0 if ok else 1)
    print("%s %-12s %s%s" % (TAG, name, "PASS" if ok else "FAIL",
                             (" | " + detail) if detail else ""), flush=True)


def glue(out):
    w = [t for t in re.findall(r"[A-Za-z]+", out) if len(t) >= 13]
    return (" GLUE:" + ",".join(w[:3])) if w else ""


def req(payload, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers=H), timeout=timeout)


def chat(msg, mx=48, timeout=3600):
    payload = {"model": "qwen38", "messages": [{"role": "user", "content": msg}],
               "temperature": 0, "max_tokens": mx, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; text = ""; ntok = 0; ptok = 0; err = None
    try:
        r = req(payload, timeout)
        for rl in r:
            line = rl.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except Exception:
                continue
            if obj.get("usage"):
                ntok = obj["usage"].get("completion_tokens", ntok)
                ptok = obj["usage"].get("prompt_tokens", ptok)
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if delta.get("content"):
                    if first is None:
                        first = time.time()
                    text += delta["content"]
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:120])
    return text, (first - t0 if first else 0), ntok, ptok, err


def count(t):
    r = urllib.request.urlopen(urllib.request.Request(URL + "/tokenize",
        data=json.dumps({"model": "qwen38", "prompt": t}).encode(),
        headers=H), timeout=600)
    return json.load(r)["count"]


def build(reps, code, nonce, question):
    b = F * reps; h = len(b) // 2
    return ("[%s] " % nonce + b[:h] + "\nThe secret access code is " + code
            + ".\n" + b[h:] + "\n" + question)


def sized(nonce, target, question=NEEDLE_Q):
    probe = build(1, "X", nonce, question)
    two = build(2, "X", nonce, question)
    per_rep = max(1, count(two) - count(probe))  # tokens per filler repetition
    lo, hi = 1, max(2, int(target / per_rep) + 4)  # bounded: ~target tokens max
    for _ in range(22):
        m = (lo + hi) // 2
        if count(build(m, "X", nonce, question)) < target:
            lo = m + 1
        else:
            hi = m
    code = "AZ" + nonce
    return build(lo, code, nonce, question), code


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip())


# ---- 1. short-context exact tasks, 3 reps each (unique nonce per request) ----
exact = [
    ("ARITH",  "Compute 17*23. Reply with only the number.", 8,   None, "391"),
    ("COPY",   COPY_TASK,  64, COPY_TEXT, None),
    ("LIST",   "List the numbers from 1 to 30 in order, separated by commas and "
               "single spaces, like '1, 2, 3'. Output only the list.", 160,
               LIST_EXPECT, None),
    ("SPELL",  "Spell the word 'necessary' letter by letter, separated by "
               "hyphens. Output only the letters and hyphens.", 32,
               SPELL_EXPECT, None),
    ("ALPHA",  "Write the lowercase English alphabet from a to z, separated by "
               "single spaces. Output only that.", 64, ALPHA_EXPECT, None),
]
for name, msg, mx, expect, contains in exact:
    for rep in range(3):
        nonce = "%s-r%d" % (name.lower(), rep)
        full = "[tag:%s] " % nonce + msg
        out, ttft, ntok, ptok, err = chat(full, mx=mx)
        if err:
            say("%s#%d" % (name, rep), False, "REQ_ERR " + err)
            continue
        g = norm(out)
        ok = (contains in g) if contains else (g == expect)
        det = ("ttft=%.1fs out=%r" % (ttft, g[:120]))
        if not ok and contains is None:
            det += " expected=%r" % expect
        say("%s#%d" % (name, rep), ok, det + glue(out))

# ---- 2. context ladder: needle + verbatim copy at ~4K/16K/32K ----
for tgt in (4096, 16384, 32768):
    k = tgt // 1024
    msg, code = sized("nd%d" % k, tgt)
    out, ttft, ntok, ptok, err = chat(msg, mx=24)
    if err:
        say("NEED%dK" % k, False, "REQ_ERR " + err)
    else:
        say("NEED%dK" % k, code in norm(out),
            "ptok=%d ttft=%.1fs out=%r%s" % (ptok or tgt, ttft,
                                             norm(out)[:60], glue(out)))
for tgt in (16384, 32768):
    k = tgt // 1024
    q = "Now ignore ALL of the text above completely. " + COPY_TASK
    msg, code = sized("cp%d" % k, tgt, question=q)
    out, ttft, ntok, ptok, err = chat(msg, mx=64)
    if err:
        say("COPY%dK" % k, False, "REQ_ERR " + err)
    else:
        g = norm(out)
        say("COPY%dK" % k, g == COPY_TEXT,
            "ptok=%d ttft=%.1fs out=%r%s%s" % (ptok or tgt, ttft, g[:100],
                                               "" if g == COPY_TEXT
                                               else " expected=%r" % COPY_TEXT,
                                               glue(out)))

# ---- 3. parity paragraph: identical prompt across A/B legs ----
PARITY = ("Write a short paragraph of five sentences about the history of the "
          "telephone. Use normal spacing and punctuation.")
out, ttft, ntok, ptok, err = chat(PARITY, mx=180)
if err:
    say("PARITY", False, "REQ_ERR " + err)
else:
    print("%s PARITY_TEXT >>>%s<<<" % (TAG, out), flush=True)
    say("PARITY", True, "ntok=%d ttft=%.1fs%s" % (ntok, ttft, glue(out)))

# ---- 3b. user-style parity: server template (preserve_thinking) + parsers,
#         request identical to the user's clients (no enable_thinking override) ----
    payload = {"model": "qwen38", "messages": [{"role": "user", "content": PARITY}],
               "temperature": 0, "max_tokens": 220, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; text = ""; ntok = 0; err = None
    try:
        r = req(payload, 3600)
        for rl in r:
            line = rl.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except Exception:
                continue
            if obj.get("usage"):
                ntok = obj["usage"].get("completion_tokens", ntok)
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                c = delta.get("content") or ""
                if c and first is None:
                    first = time.time()
                text += c
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:120])
    if err:
        say("USERPARITY", False, "REQ_ERR " + err)
    else:
        print("%s USERPARITY_TEXT >>>%s<<<" % (TAG, text), flush=True)
        say("USERPARITY", bool(text), "ntok=%d ttft=%.1fs%s" % (
            ntok, (first - t0) if first else 0, glue(text)))

print("%s SUMMARY pass=%d fail=%d" % (TAG, NPASS, NFAIL), flush=True)
sys.exit(0 if NFAIL == 0 else 1)
