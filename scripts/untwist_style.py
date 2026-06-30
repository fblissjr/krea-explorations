#!/usr/bin/env python
"""Drive our Krea2 untwisting-RoPE node over the ComfyUI API and build a base-vs-untwist grid.

Training-free style transfer: a reference image's style is shared into each generation via untwisted
shared attention (no LoRA, no training). Sweeps several prompts x [base, untwist@low_scale...] and writes
a comparison contact sheet.

Needs a running ComfyUI with the krea2-explorations pack loaded (so the `Krea2UntwistStyleReference` node
exists) and the reference image present in ComfyUI's input dir (use --comfy-input-dir to copy it there).

    uv run python scripts/untwist_style.py --comfy-input-dir <comfyui_input_dir> --out-dir data/untwist_weirdguys
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from krea2_explorations.image_grid import build_contact_sheet  # noqa: E402

# reuse the proven API submit/poll helper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import run, resolve_vae, resolve_clip  # noqa: E402

PROMPTS = [
    "a portrait of a person",
    "a cat sitting",
    "a robot",
    "a small house",
]


def build_untwist_graph(prompt, *, ref_image_name, unet, clip, vae, low_scale, high_scale, beta,
                        untwist=True, steps=8, cfg=1.0, seed=42, width=1024, height=1024,
                        sampler="euler", scheduler="simple", filename_prefix="krea2_untwist"):
    # No ModelSamplingFlux: the 1.15 flow-shift is in Krea2's model config (pixel-identical no-op at ~1MP).
    # This is a specialized graph -- untwist needs a VAE-encoded reference latent -- so it is not build_graph.
    # For an ordinary render call scripts/canonical_workflows.py; do not copy this skeleton.
    g = {"ckpt": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}}
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    # always a real (empty) negative, never ConditioningZeroOut (see scripts/generate.py)
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}}
    g["latent"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}

    model_src = ["ckpt", 0]
    if untwist:
        g["refimg"] = {"class_type": "LoadImage", "inputs": {"image": ref_image_name}}
        g["refscale"] = {"class_type": "ImageScale",
                         "inputs": {"image": ["refimg", 0], "upscale_method": "lanczos",
                                    "width": width, "height": height, "crop": "center"}}
        g["refenc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["refscale", 0], "vae": ["vae", 0]}}
        g["untwist"] = {"class_type": "Krea2UntwistStyleReference",
                        "inputs": {"model": ["ckpt", 0], "reference_latent": ["refenc", 0],
                                   "low_scale": low_scale, "high_scale": high_scale, "beta": beta}}
        model_src = ["untwist", 0]

    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model_src, "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0], "seed": seed, "steps": steps, "cfg": cfg,
                               "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}}
    g["vaedec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["vaedec", 0], "filename_prefix": filename_prefix}}
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-image", default="data/train_styles/style_weirdguys/Subject_rgb.png",
                    help="local path to the RGB reference image (repo-relative ok)")
    ap.add_argument("--comfy-input-dir", help="ComfyUI input dir to copy the reference into (so LoadImage sees it)")
    ap.add_argument("--ref-name", default="weirdguys_ref.png", help="basename LoadImage will use")
    ap.add_argument("--out-dir", default="data/untwist_weirdguys")
    ap.add_argument("--unet", default="krea2_turbo_bf16.safetensors")
    ap.add_argument("--clip", default=None, help="CLIP/encoder filename; default = bf16 if present, else fp8")
    ap.add_argument("--vae", default=None, help="VAE filename; default = krea2RealVae if present, else stock")
    ap.add_argument("--low-scales", default="2.0,3.0,4.0", help="comma-separated low_scale sweep")
    ap.add_argument("--high-scale", type=float, default=1.05)
    ap.add_argument("--beta", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    vae = a.vae or resolve_vae(a.server)  # krea2RealVae if the server has it, else stock
    clip = a.clip or resolve_clip(a.server)  # bf16 encoder if the server has it, else fp8

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.comfy_input_dir:
        dst = Path(a.comfy_input_dir) / a.ref_name
        shutil.copyfile(a.ref_image, dst)
        print(f"copied reference -> {dst}")

    low_scales = [float(s) for s in a.low_scales.split(",")]
    cols = ["base"] + [f"low={ls:g}" for ls in low_scales]

    grid_rows = []
    for r, prompt in enumerate(PROMPTS):
        row = []
        for c, col in enumerate(cols):
            untwist = col != "base"
            ls = low_scales[c - 1] if untwist else 0.0
            g = build_untwist_graph(prompt, ref_image_name=a.ref_name, unet=a.unet, clip=clip, vae=vae,
                                    low_scale=ls, high_scale=a.high_scale, beta=a.beta,
                                    untwist=untwist, seed=a.seed, filename_prefix=f"untw_{r}_{c}")
            dst = out / f"cell_{r}_{c}.png"
            ok = run(g, dst, server=a.server, harness="untwist_style", arm=f"{col}_p{r}", seed=a.seed, prompt=prompt)
            print(f"{'ok ' if ok else 'FAIL'} prompt[{r}] {col} -> {dst}")
            row.append(str(dst) if ok else None)
            time.sleep(0.2)
        grid_rows.append(row)

    grid = build_contact_sheet(grid_rows, out / "untwist_weirdguys_grid.png",
                               col_labels=cols, row_labels=[p[:28] for p in PROMPTS])
    print("grid ->", grid)


if __name__ == "__main__":
    main()
