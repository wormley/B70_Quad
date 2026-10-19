#!/usr/bin/env python3
"""Regression test: Qwen4ExpMTP draft-group annotation (prefix-cache fix).

Verifies that _annotate_eagle_groups() flags the Qwen4ExpMTP draft layer's
KV cache group (rule 2: last-layer positional fallback) instead of leaving
every group unannotated -- which made the KVCacheCoordinator fall back to
flagging ALL groups (including the Mamba groups) as draft groups and
silently disabled cross-request prefix-cache reuse (0.0% hit rate).

Pure config/spec logic; no GPU required. Run inside the image:

    podman run --rm -v "$PWD:/work:ro" --entrypoint python3 \\
        localhost/qfn38-fp8-4xb70:latest /work/scripts/test_eagle_annotation.py

Exits 0 on success, 1 on failure.
"""

import logging
import sys
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import (
    _annotate_eagle_groups,
    _is_deepseek_v4_eagle,
    _is_qwen4_exp_eagle,
    _warn_if_unannotated_eagle_mamba,
)


def annotate(config, kv_cache_spec, groups):
    # The real call-site gate (get_kv_cache_groups):
    gate = _is_deepseek_v4_eagle(config) or _is_qwen4_exp_eagle(config)
    _annotate_eagle_groups(config, kv_cache_spec, groups, gate)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

LOGGER = logging.getLogger("vllm.v1.core.kv_cache_utils")


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def make_vllm_config(
    *,
    spec_method: str | None = "mtp",
    draft_architectures: tuple[str, ...] = ("Qwen4ExpMTP",),
    target_model_type: str = "qwen4_exp",
    disable_eagle_block_drop: bool = False,
):
    spec_config = None
    if spec_method is not None:
        draft_model_config = SimpleNamespace(architectures=list(draft_architectures))
        spec_config = SimpleNamespace(
            method=spec_method,
            draft_model_config=draft_model_config,
            disable_eagle_block_drop=disable_eagle_block_drop,
            use_eagle=lambda: spec_method in ("eagle", "eagle3", "mtp", "dflash", "dspark"),
            use_eagle_block_drop=lambda: (
                spec_method in ("eagle", "eagle3", "mtp", "dflash", "dspark")
                and not disable_eagle_block_drop
            ),
        )
    return SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=target_model_type)),
    )


def make_kv_cache_spec():
    """Layer registration order: backbone first, MTP draft layer LAST.

    Mirrors the qfn-38-fp8-4xb70-mtp deployment: the Qwen4ExpMTP draft layer
    is a full-attention Qwen4ExpDecoderLayer, so its spec is identical to the
    target's full-attention layers and the packed grouping path merges it
    into the same full-attention group.
    """
    fa_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    mamba_spec = MambaSpec(
        shapes=((4, 8),),
        dtypes=(torch.bfloat16,),
        block_size=16,
        mamba_cache_mode="align",
    )
    return {
        "model.layers.0.self_attn.attn": fa_spec,
        "model.layers.1.linear_attn": mamba_spec,
        "model.layers.2.self_attn.attn": fa_spec,
        "model.layers.3.linear_attn": mamba_spec,
        # The MTP draft layer, registered after every backbone layer:
        "model.layers.4.self_attn.attn": fa_spec,
    }


def make_groups():
    """Groups as the packed grouping path builds them for make_kv_cache_spec():
    group 0 = full attention (target layers + merged draft layer),
    groups 1-2 = Mamba (align mode)."""
    spec = make_kv_cache_spec()
    fa_spec = spec["model.layers.0.self_attn.attn"]
    mamba_spec = spec["model.layers.1.linear_attn"]
    return [
        KVCacheGroupSpec(
            [
                "model.layers.0.self_attn.attn",
                "model.layers.2.self_attn.attn",
                "model.layers.4.self_attn.attn",
            ],
            fa_spec,
        ),
        KVCacheGroupSpec(["model.layers.1.linear_attn"], mamba_spec),
        KVCacheGroupSpec(["model.layers.3.linear_attn"], mamba_spec),
    ]


def run_case(name, config, expect_full_attn_eagle, expect_warning_substring=None):
    capture = WarningCapture()
    LOGGER.addHandler(capture)
    try:
        groups = make_groups()
        kv_cache_spec = make_kv_cache_spec()
        annotate(config, kv_cache_spec, groups)
        _warn_if_unannotated_eagle_mamba(config, groups)

        assert groups[0].is_eagle_group == expect_full_attn_eagle, (
            f"{name}: full-attn group is_eagle_group={groups[0].is_eagle_group}, "
            f"expected {expect_full_attn_eagle}"
        )
        mamba_flagged = [i for i, g in enumerate(groups[1:], start=1) if g.is_eagle_group]
        assert not mamba_flagged, f"{name}: Mamba groups wrongly flagged: {mamba_flagged}"

        if expect_warning_substring is None:
            assert not capture.messages, f"{name}: unexpected warning: {capture.messages}"
        else:
            assert any(
                expect_warning_substring in m for m in capture.messages
            ), f"{name}: expected warning containing {expect_warning_substring!r}, got {capture.messages}"
    finally:
        LOGGER.removeHandler(capture)
    print(f"  PASS {name}")


def main():
    print("Qwen4ExpMTP eagle-group annotation regression test")

    # 1. The bug scenario, fixed: qwen4_exp + mtp + Qwen4ExpMTP draft.
    #    Full-attention group (holding the draft layer) must be flagged,
    #    Mamba groups must NOT be flagged, and the flag-all warning must be gone.
    run_case(
        "qwen4_exp + mtp (Qwen4ExpMTP draft)",
        make_vllm_config(),
        expect_full_attn_eagle=True,
    )

    # 2. Same target but an unrecognized draft architecture: the old
    #    (fallback) behavior must be preserved -- nothing annotated,
    #    flag-all warning emitted.
    run_case(
        "mtp with unrecognized draft arch (fallback preserved)",
        make_vllm_config(draft_architectures=("SomeOtherMTP",)),
        expect_full_attn_eagle=False,
        expect_warning_substring="no KV cache group could be identified",
    )

    # 3. DeepseekV4 rule-2 path unchanged (regression guard).
    run_case(
        "deepseek_v4 positional fallback intact",
        make_vllm_config(target_model_type="deepseek_v4", draft_architectures=()),
        expect_full_attn_eagle=True,
    )

    # 4. No speculative decoding: nothing annotated, no warning.
    run_case(
        "no speculative config",
        make_vllm_config(spec_method=None),
        expect_full_attn_eagle=False,
    )

    # 5. Rule 1 (spec marker) still wins when present, independent of rule 2.
    #    The dspark-style marker is only on MLAAttentionSpec; emulate by
    #    flagging through a draft arch we do not cover and a spec_config that
    #    already carries an annotated group -- annotation must not clobber it.
    groups = make_groups()
    groups[2].is_eagle_group = True  # pretend rule 1 flagged a mamba group
    config = make_vllm_config(draft_architectures=("SomeOtherMTP",))
    capture = WarningCapture()
    LOGGER.addHandler(capture)
    try:
        annotate(config, make_kv_cache_spec(), groups)
        _warn_if_unannotated_eagle_mamba(config, groups)
        assert groups[0].is_eagle_group is False
        assert groups[2].is_eagle_group is True
        assert not capture.messages, (
            "unexpected warning: the warn function must stay silent once any "
            f"group is annotated: {capture.messages}")
    finally:
        LOGGER.removeHandler(capture)
    print("  PASS pre-annotated group not clobbered / silent (zero-annotation rule)")

    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
