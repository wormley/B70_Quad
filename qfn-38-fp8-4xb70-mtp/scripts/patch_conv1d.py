#!/usr/bin/env python3
"""Apply the int64 offset-cast fix to vllm-xpu-kernels conv-state kernels.

Several sites compute `conv_states + <state_id> * conv_states_stride_0` with
`int` id and `int` stride; the product overflows int32 once the conv-state
pool spans more than 2^31 elements (the state pool packed per KV block
reaches that at deep context). The sibling SSM kernel (gated_delta_rule.hpp)
already casts to int64 at every equivalent site — this brings the conv
kernels in line, across ALL variants:

  csrc/xpu/gdn_attn/causal_conv1d.hpp                      (3 sites)
  csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp  (2 sites — the
                                            variant actually dispatched on BMG)
  csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp        (2 sites)

Usage: patch_conv1d.py <repo-root | header-file>
Idempotent: re-running reports all sites already clean.
"""
import pathlib
import re
import sys

FIXED = "static_cast<int64_t>"

FILES = [
    "csrc/xpu/gdn_attn/causal_conv1d.hpp",
    "csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp",
    "csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp",
]

PATTERNS = [
    ("    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;",
     "    T* conv_states_ptr = conv_states + static_cast<int64_t>(states_id) * conv_states_stride_0;  // [4gbfix] int64: id*stride overflows int32 on large state pools"),
    ("      T* st = conv_states + states_id * conv_states_stride_0;",
     "      T* st = conv_states + static_cast<int64_t>(states_id) * conv_states_stride_0;  // [4gbfix] int64"),
    ("                            ? conv_states + state_id * conv_states_stride_0",
     "                            ? conv_states + static_cast<int64_t>(state_id) * conv_states_stride_0  // [4gbfix] int64"),
]

BUGGY_RE = re.compile(
    r"states_id \* conv_states_stride_0|state_id \* conv_states_stride_0")


def main(path: str) -> int:
    """Patch every conv-state id*stride product in all kernel variants.

    ``path`` may be a repo root (patches all FILES) or a single header file.
    """
    root = pathlib.Path(path)
    targets = [root] if root.is_file() else [root / f for f in FILES]
    for tgt in targets:
        if not tgt.exists():
            print(f"MISSING: {tgt} — cannot patch", flush=True)
            return 1
        text = tgt.read_text()
        orig = text
        for old, new in PATTERNS:
            c = text.count(old)
            if c:
                text = text.replace(old, new)
                print(f"  {tgt.name}: patched {c} site(s)")
        if text != orig:
            tgt.write_text(text)
        else:
            print(f"  {tgt.name}: no changes needed")
    bad = []
    for tgt in targets:
        for i, line in enumerate(tgt.read_text().splitlines(), 1):
            if BUGGY_RE.search(line) and FIXED not in line:
                bad.append(f"{tgt.name}:{i}: {line.strip()}")
    if bad:
        for b in bad:
            print("REMAINING:", b)
        print("VERIFY: FAILED")
        return 1
    print("VERIFY: OK — all conv-state offset products are int64 in all variants")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
