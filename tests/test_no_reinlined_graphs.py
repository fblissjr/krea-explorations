"""Guard: no tracked script may re-inline a ModelSamplingFlux node.

ModelSamplingFlux is a pixel-identical no-op at ~1MP (the 1.15 flow-shift lives in Krea2's model config), so a
re-inlined `ModelSamplingFlux` node is the tell-tale of a cargo-culted stale graph — the exact drift this repo
keeps hitting. This fails fast if one reappears in `scripts/`, locking the public surface against regression.
The canonical builders (`scripts/canonical_workflows.py`, `scripts/generate.py`) never emit the node;
`generate.py` mentions it only in prose explaining its removal, never as a node literal.
"""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_no_tracked_script_reinlines_model_sampling_flux():
    offenders = [f.name for f in sorted(SCRIPTS.glob("*.py"))
                 if '"class_type": "ModelSamplingFlux"' in f.read_text()]
    assert not offenders, f"re-inlined ModelSamplingFlux node in: {offenders} — use the canonical builders"
