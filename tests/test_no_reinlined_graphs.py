"""Guard: no tracked script may re-inline a stale graph signature.

The repo keeps getting bitten by cargo-culted graph dicts (CLAUDE.md "Building generation graphs"). The three
tells are: a re-inlined ModelSamplingFlux node (a pixel-identical no-op at ~1MP), ConditioningZeroOut as the
negative (grainy on RAW / `_cfg_pp` samplers), and a hardcoded stock VAE. This fails fast if any of them
reappears in scripts/, locking the public surface against regression. The canonical builders never emit them;
generate.py mentions ModelSamplingFlux only in prose explaining its removal, and owns the stock-VAE name as the
documented STOCK_VAE fallback constant. The node matchers tolerate quote/whitespace variation so a reformat
(single quotes, no space after the colon) can't sneak a node past.
"""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_STOCK_VAE = re.compile(r"qwen_image_vae\.safetensors")


def _node(class_name):
    return re.compile(r"""["']class_type["']\s*:\s*["']""" + class_name + r"""["']""")


def _scripts():
    return sorted(SCRIPTS.glob("*.py"))


def test_no_tracked_script_reinlines_model_sampling_flux():
    bad = [f.name for f in _scripts() if _node("ModelSamplingFlux").search(f.read_text())]
    assert not bad, f"re-inlined ModelSamplingFlux node in {bad} -- use the canonical builders"


def test_no_tracked_script_uses_conditioning_zero_out():
    bad = [f.name for f in _scripts() if _node("ConditioningZeroOut").search(f.read_text())]
    assert not bad, f"ConditioningZeroOut negative in {bad} -- use a real empty CLIPTextEncode (breaks _cfg_pp)"


def test_no_tracked_script_hardcodes_the_stock_vae():
    # generate.py legitimately names the stock VAE as the documented STOCK_VAE fallback; every other tracked
    # script must resolve via resolve_vae / DEFAULT_VAE, never the literal filename.
    bad = [f.name for f in _scripts() if f.name != "generate.py" and _STOCK_VAE.search(f.read_text())]
    assert not bad, f"hardcoded stock qwen_image_vae in {bad} -- use resolve_vae / DEFAULT_VAE"
