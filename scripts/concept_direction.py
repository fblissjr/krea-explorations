#!/usr/bin/env python
"""Measure concept directions for the **Krea 2 Concept Inject** node from A/B prompt pairs.

For each named concept, ``direction = mean(encode(positives)) - mean(encode(negatives))`` in Krea 2's
12-layer conditioning space, saved as ``<name>.npy`` (shape ``12 x 2560``). Feed it to the *Krea 2 Concept
Inject* node (mode ``amplify`` to dial up a present axis, ``add`` to inject, ``project_out`` to remove). Works
on ANY axis the encoder already represents -- expression, style, lighting, pose, an attribute.

    uv run --active python scripts/concept_direction.py examples/concept_directions.json --out <comfyui_models>/concept_dirs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import orjson

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from krea2_explorations.contrast_directions import pooled_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("concepts", help='json: {"name": {"pos": ["..."], "neg": ["..."]}, ...}')
    ap.add_argument("--out", default=str(REPO / "data" / "concept_dirs"), help="output dir for <name>.npy")
    ap.add_argument("--text-encoder", default="qwen3vl_4b_bf16.safetensors")
    ap.add_argument("--dry-run", action="store_true", help="validate the concepts json without loading the model")
    a = ap.parse_args()

    concepts = orjson.loads(Path(a.concepts).read_bytes())
    for name, spec in concepts.items():
        if not spec.get("pos") or not spec.get("neg"):
            raise SystemExit(f"concept {name!r} needs non-empty 'pos' and 'neg' prompt lists")
    print(f"concepts: {list(concepts)}{'  [DRY RUN]' if a.dry_run else ''}")
    if a.dry_run:
        return

    out = Path(a.out).resolve()  # resolve relative to cwd BEFORE load_clip_cpu chdirs to the ComfyUI root
    from krea2_clip import load_clip_cpu, pooled_conditioning  # comfy, loaded lazily

    clip = load_clip_cpu(text_encoder=a.text_encoder)
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in concepts.items():
        pos = np.stack([pooled_conditioning(clip, p) for p in spec["pos"]])  # (n_pos, 12*2560)
        neg = np.stack([pooled_conditioning(clip, p) for p in spec["neg"]])
        d = pooled_direction(pos, neg)  # 12 bands -- matches Krea2's conditioning layout / the node's loader
        np.save(out / f"{name}.npy", d)
        print(f"  {name:16} ||d||={np.linalg.norm(d):8.2f}  -> {name}.npy", flush=True)
    print(f"saved -> {out}/  (use as direction_path in 'Krea 2 Concept Inject', mode=amplify)")


if __name__ == "__main__":
    main()
