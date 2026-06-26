#!/usr/bin/env python
"""Generate a Krea 2 image through the ComfyUI API, with the proper flow-match shift.

Talks to a running ComfyUI server over HTTP, so it needs no ComfyUI path — just the
model filenames as they appear in your ComfyUI models dirs.

Key points baked in:
- ModelSamplingFlux (base_shift 0.5, max_shift 1.15) reproduces Krea's resolution-aware `mu`.
- Deterministic euler/simple + fixed seed by default, so A/B comparisons isolate your change.
- Presets: `turbo` (8 steps, CFG off) and `raw` (28 steps, CFG 4.5). RAW also needs `--unet` set
  to a RAW checkpoint.

Examples:
    uv run python scripts/generate.py "a red fox in snow" --out out.png
    uv run python scripts/generate.py --prompt-file p.txt --preset raw \
        --unet krea2_raw_fp8_scaled.safetensors --out raw.png
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PRESETS = {"turbo": dict(steps=8, cfg=1.0), "raw": dict(steps=28, cfg=4.5)}


def build_graph(prompt, *, unet, clip, vae, negative="", steps=8, cfg=1.0, seed=42,
                width=1024, height=1024, sampler="euler", scheduler="simple",
                base_shift=0.5, max_shift=1.15, filename_prefix="krea2_gen"):
    """Build a ComfyUI API graph for a single Krea 2 txt2img with the flow shift applied."""
    g = {
        "ckpt": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingFlux",
                  "inputs": {"model": ["ckpt", 0], "max_shift": max_shift,
                             "base_shift": base_shift, "width": width, "height": height}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "vaedec": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["vaedec", 0], "filename_prefix": filename_prefix}},
    }
    # Real negative conditioning when CFG is on; zeroed-out when CFG is off (Turbo).
    if cfg > 1.0:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
    else:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0], "seed": seed, "steps": steps,
                               "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                               "denoise": 1.0}}
    return g


def run(graph, out_path, server="http://127.0.0.1:8188", timeout=600):
    """Submit a graph and save the first SaveImage output to out_path. Returns True on success."""
    req = urllib.request.Request(server + "/prompt",
                                 data=json.dumps({"prompt": graph}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    for _ in range(timeout):
        time.sleep(1)
        h = json.loads(urllib.request.urlopen(server + f"/history/{pid}", timeout=30).read())
        if pid in h:
            imgs = h[pid].get("outputs", {}).get("save", {}).get("images", [])
            if not imgs:
                return False
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            Path(out_path).write_bytes(urllib.request.urlopen(server + "/view?" + q, timeout=120).read())
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="prompt text (or use --prompt-file)")
    ap.add_argument("--prompt-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="turbo")
    ap.add_argument("--unet", default="krea2_turbo_fp8_scaled.safetensors")
    ap.add_argument("--clip", default="qwen3vl_4b_fp8_scaled.safetensors")
    ap.add_argument("--vae", default="qwen_image_vae.safetensors")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, help="override preset steps")
    ap.add_argument("--cfg", type=float, help="override preset cfg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--base-shift", type=float, default=0.5)
    ap.add_argument("--max-shift", type=float, default=1.15)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    prompt = a.prompt or Path(a.prompt_file).read_text().strip()
    p = PRESETS[a.preset]
    steps = int(a.steps or p["steps"])
    cfg = float(a.cfg if a.cfg is not None else p["cfg"])
    g = build_graph(prompt, unet=a.unet, clip=a.clip, vae=a.vae, negative=a.negative,
                    steps=steps, cfg=cfg, seed=a.seed, width=a.width, height=a.height,
                    sampler=a.sampler, scheduler=a.scheduler,
                    base_shift=a.base_shift, max_shift=a.max_shift)
    ok = run(g, a.out, server=a.server)
    print(f"{'saved' if ok else 'FAILED'} {a.out}  (preset={a.preset} steps={steps} cfg={cfg} "
          f"{a.width}x{a.height} seed={a.seed})")


if __name__ == "__main__":
    main()
