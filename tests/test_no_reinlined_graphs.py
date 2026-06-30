"""Guard: no tracked python graph-builder may re-inline a stale graph signature.

The repo keeps getting bitten by cargo-culted graph dicts (CLAUDE.md "Building generation graphs"). Three tells:
a re-inlined ModelSamplingFlux node (a pixel-identical no-op at ~1MP), ConditioningZeroOut as the negative
(grainy on RAW / `_cfg_pp` samplers), and a hardcoded stock VAE. This fails fast if any reappears in the tracked
`scripts/` builders, where the canonical recipe lives. It deliberately does NOT scan the gitignored `internal/`
harnesses (a scratchpad where the literals are legitimate -- e.g. the VAE-comparison experiments) nor the
`example_workflows/*.json` deliverables (ConditioningZeroOut + stock VAE are correct there for Turbo workflows);
discipline for those lives in CLAUDE.md. The node matchers tolerate quote/whitespace variation so a reformat
can't sneak one past, and generate.py owns the stock-VAE name as its documented STOCK_VAE fallback constant, so
it is exempt from that one check.
"""
import re
from pathlib import Path

import pytest

_TEXTS = {f.name: f.read_text()  # read each tracked script once
          for f in sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))}


def _node(class_name):
    return re.compile(r"""["']class_type["']\s*:\s*["']""" + class_name + r"""["']""")


# (matcher, the forbidden thing + the fix, filename allowed to legitimately contain it)
_SIGNATURES = [
    (_node("ModelSamplingFlux"), "re-inlined ModelSamplingFlux node -- use the canonical builders", None),
    (_node("ConditioningZeroOut"), "ConditioningZeroOut negative -- use a real empty CLIPTextEncode (breaks _cfg_pp)", None),
    (re.compile(r"qwen_image_vae\.safetensors"), "hardcoded stock qwen_image_vae -- use resolve_vae / DEFAULT_VAE", "generate.py"),
]


@pytest.mark.parametrize("pattern,forbidden,allow", _SIGNATURES,
                         ids=["model_sampling_flux", "conditioning_zero_out", "stock_vae"])
def test_no_tracked_builder_reinlines_a_stale_signature(pattern, forbidden, allow):
    bad = [name for name, text in _TEXTS.items() if name != allow and pattern.search(text)]
    assert not bad, f"{forbidden}: {bad}"
