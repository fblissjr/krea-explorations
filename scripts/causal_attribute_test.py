#!/usr/bin/env python
"""Causal attribute test: generate with vs without an attribute (same seed/settings, only the
attribute clause differs) and see whether the model actually renders it.

This is the causal companion to the difference-of-means probe: that probe showed benign attributes
*survive* the layer-fusion in the conditioning; this checks they actually appear in the image. Uses a
deterministic euler/simple + fixed seed + the flow shift (via generate.build_graph), so the only
variable is the attribute clause.

    uv run python scripts/causal_attribute_test.py            # Turbo, 1024, seed 42
    uv run python scripts/causal_attribute_test.py --preset raw --unet krea2_raw_fp8_scaled.safetensors
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import PRESETS, build_graph, run  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

BASE = "a close-up portrait photograph of a young woman, "
# name -> (without-clause, with-clause)
PAIRS = {
    "expression": ("with a neutral calm closed-mouth expression",
                   "laughing with a wide joyful open-mouthed smile"),
    "wet": ("her face and hair completely dry",
            "her face and hair soaking wet, dripping with water"),
    "blush": ("with an even pale skin tone",
              "with deeply flushed blushing bright-red cheeks"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/attribute_directions/causal")
    ap.add_argument("--preset", choices=list(PRESETS), default="turbo")
    ap.add_argument("--unet", default="krea2_turbo_fp8_scaled.safetensors")
    ap.add_argument("--clip", default="qwen3vl_4b_fp8_scaled.safetensors")
    ap.add_argument("--vae", default="qwen_image_vae.safetensors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = PRESETS[a.preset]
    paths = {}
    for name, (without, with_) in PAIRS.items():
        for tag, clause in (("without", without), ("with", with_)):
            g = build_graph(BASE + clause, unet=a.unet, clip=a.clip, vae=a.vae,
                            steps=p["steps"], cfg=p["cfg"], seed=a.seed,
                            width=a.size, height=a.size, filename_prefix=f"causal_{name}_{tag}")
            dst = out / f"{name}_{tag}.png"
            ok = run(g, str(dst), server=a.server)
            print(f"{name:>11} {tag:>8}: {'ok' if ok else 'FAILED'}")
            paths[(name, tag)] = dst

    # grid: one row per attribute, columns without | with
    th, lab = 420, 26
    names = list(PAIRS)
    sheet = Image.new("RGB", (2 * th, len(names) * (th + lab)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for r, name in enumerate(names):
        for c, tag in enumerate(("without", "with")):
            x, y = c * th, r * (th + lab)
            sheet.paste(Image.open(paths[(name, tag)]).convert("RGB").resize((th, th)), (x, y + lab))
            d.rectangle([x, y, x + th, y + lab], fill=(40, 40, 40))
            d.text((x + 5, y + 4), f"{name}: {tag}", fill=(235, 235, 235), font=font)
    grid = out / "causal_grid.png"
    sheet.save(grid)
    print(f"grid -> {grid}")


if __name__ == "__main__":
    main()
