#!/usr/bin/env python
"""Stock vs projector-rebalanced: what does the deep-layer projector-rebalance lever change?

Generates each attribute prompt with the stock projector and with a deep-layer-boosted projector (the
nova452-style "rebalance" lever, emitted as a tiny `.diff` LoRA) at one or more strengths — same
seed/settings otherwise. Observed: the benign attributes render either way; boosting the deep layers
mainly shifts detail / contrast / intensity (the deep layers carry the fine detail), rather than gating
whether an attribute appears.

    uv run python scripts/stock_vs_rebalanced.py                 # strengths 1 and 4
    uv run python scripts/stock_vs_rebalanced.py --strengths 1,4,7
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate import PRESETS, build_graph, run  # noqa: E402

from krea2_explorations.projector_lora import make_projector_lora  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

COMFY = Path(__file__).resolve().parents[3]  # .../ComfyUI
# nova452 "rebalance" node default per-layer weights — a widely-used deep-boost config.
DEEP_BOOST_GAINS = [1, 1, 1, 1, 1, 1, 1, 2.5, 5.0, 1.1, 4.0, 1.0]
BASE = "a close-up portrait photograph of a young woman, "
PROMPTS = {
    "expression": BASE + "laughing with a wide joyful open-mouthed smile",
    "wet": BASE + "her face and hair soaking wet, dripping with water",
    "blush": BASE + "with deeply flushed blushing bright-red cheeks",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/attribute_directions/stock_vs_rebalanced")
    ap.add_argument("--ckpt", default=str(COMFY / "models/diffusion_models/krea2_turbo_bf16.safetensors"),
                    help="checkpoint to read the original projector from (for the diff)")
    ap.add_argument("--loras-dir", default=str(COMFY / "models/loras"))
    ap.add_argument("--unet", default="krea2_turbo_fp8_scaled.safetensors")
    ap.add_argument("--clip", default="qwen3vl_4b_fp8_scaled.safetensors")
    ap.add_argument("--vae", default="qwen_image_vae.safetensors")
    ap.add_argument("--strengths", default="1,4", help="comma list of rebalance LoRA strengths")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    strengths = [float(s) for s in a.strengths.split(",") if s.strip()]
    lora_name = "krea2_rebalance_deepboost.safetensors"
    make_projector_lora(a.ckpt, DEEP_BOOST_GAINS, str(Path(a.loras_dir) / lora_name),
                        metadata={"note": "nova452 deep-boost rebalance (projector lever)"})
    print(f"emitted {lora_name} gains={DEEP_BOOST_GAINS}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = PRESETS["turbo"]
    cols = [("stock", None, 0.0)] + [(f"rebal x{s:g}", lora_name, s) for s in strengths]
    paths = {}
    for name, prompt in PROMPTS.items():
        for label, lora, strn in cols:
            g = build_graph(prompt, unet=a.unet, clip=a.clip, vae=a.vae,
                            steps=p["steps"], cfg=p["cfg"], seed=a.seed, width=a.size, height=a.size,
                            lora=lora, lora_strength=strn, filename_prefix=f"svr_{name}")
            dst = out / f"{name}_{label.replace(' ', '_')}.png"
            ok = run(g, str(dst), server=a.server)
            print(f"{name:>11} {label:>10}: {'ok' if ok else 'FAILED'}")
            paths[(name, label)] = dst

    th, lab = 400, 26
    names = list(PROMPTS)
    ncol = len(cols)
    sheet = Image.new("RGB", (ncol * th, len(names) * (th + lab)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for r, name in enumerate(names):
        for c, (label, _, _) in enumerate(cols):
            x, y = c * th, r * (th + lab)
            sheet.paste(Image.open(paths[(name, label)]).convert("RGB").resize((th, th)), (x, y + lab))
            d.rectangle([x, y, x + th, y + lab], fill=(40, 40, 40))
            d.text((x + 5, y + 4), f"{name}: {label}", fill=(235, 235, 235), font=font)
    grid = out / "stock_vs_rebalanced_grid.png"
    sheet.save(grid)
    print(f"grid -> {grid}")


if __name__ == "__main__":
    main()
