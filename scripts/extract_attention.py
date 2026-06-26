#!/usr/bin/env python
"""Extract Krea2's multilayer-feature-aggregation (layer-fusion) attention.

Loads the Krea2 CLIP to get the 12 selected-layer hidden states for each prompt, instantiates the DiT's
``txtfusion`` module from the checkpoint weights, and recomputes the layerwise-block attention (how each
selected layer attends to the others when the model combines them). CPU-only so it does not contend with a
running ComfyUI for VRAM. Saves per-prompt ``.npy`` + cross-prompt and per-head figures.

Usage (paths default to this repo's location under ComfyUI/custom_nodes/):
    uv run python scripts/extract_attention.py
    uv run python scripts/extract_attention.py --prompt "a red bicycle" --prompt "an anime sunset"
    uv run python scripts/extract_attention.py --comfy-root /path/to/ComfyUI --out ./data/attention
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]            # .../custom_nodes/krea2-explorations
DEFAULT_COMFY = Path(os.environ.get("COMFYUI_ROOT", REPO.parents[1]))  # .../ComfyUI
SELECT_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
DEFAULT_PROMPTS = [
    "A close-up portrait of a young woman with an intense gaze, lilies in the foreground, crimson background.",
    "A dynamic anime painting of a joyful girl in a sailor uniform, windblown hair, vibrant blue background.",
    "A minimalist flat-color ligne-claire illustration of a figure wading through shallow ocean waves.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY, help="ComfyUI root (for importing comfy).")
    p.add_argument("--text-encoder", default="qwen3vl_4b_bf16.safetensors", help="filename in models/text_encoders/")
    p.add_argument("--dit", default="krea2_turbo_bf16.safetensors", help="filename in models/diffusion_models/ (for txtfusion weights)")
    p.add_argument("--prompt", action="append", dest="prompts", help="prompt(s); repeatable. Defaults to 3 varied prompts.")
    p.add_argument("--out", type=Path, default=REPO / "data" / "attention", help="output dir")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # comfy lives in the ComfyUI repo (not an installed package); force CPU before importing it.
    sys.path.insert(0, str(args.comfy_root))
    os.chdir(args.comfy_root)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()

    import numpy as np
    import torch
    from einops import rearrange
    from safetensors import safe_open
    import comfy.sd
    import comfy.ops
    from comfy.ldm.krea2.model import TextFusionTransformer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from krea2_explorations import attention_stats as ast

    torch.set_grad_enabled(False)
    te_path = args.comfy_root / "models" / "text_encoders" / args.text_encoder
    dit_path = args.comfy_root / "models" / "diffusion_models" / args.dit
    labels = [f"L{L}" for L in SELECT_LAYERS]

    print(f"loading Krea2 CLIP (CPU) from {te_path.name} ...", flush=True)
    clip = comfy.sd.load_clip(ckpt_paths=[str(te_path)], clip_type=comfy.sd.CLIPType.KREA2)

    print(f"loading txtfusion weights from {dit_path.name} ...", flush=True)
    tf = TextFusionTransformer(12, 2560, 20, 4, False, 20, device=torch.device("cpu"),
                               dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    sd = {}
    with safe_open(str(dit_path), framework="pt") as f:
        for k in f.keys():
            if k.startswith("txtfusion."):
                sd[k[len("txtfusion."):]] = f.get_tensor(k).float()
    missing, unexpected = tf.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"txtfusion load mismatch: missing={missing} unexpected={unexpected}"
    tf.eval()
    proj = tf.projector.weight.detach().reshape(-1).numpy()
    np.save(out / "projector_weights.npy", proj)

    def attn_w(block, x):
        a = block.attn
        xn = block.prenorm(x)
        q = rearrange(a.wq(xn), "n l (h d) -> n h l d", h=a.heads)
        k = rearrange(a.wk(xn), "n l (h d) -> n h l d", h=a.kvheads)
        q, k = a.qknorm(q, k)
        return torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(a.headdim), dim=-1)

    means = {}
    per_head = {}
    for name, prompt in zip([f"p{i}" for i in range(len(prompts))], prompts):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].float()
        b, seq, fused = cond.shape
        x = cond.reshape(b, seq, 12, fused // 12).reshape(b * seq, 12, fused // 12)
        a0 = attn_w(tf.layerwise_blocks[0], x)
        x1 = tf.layerwise_blocks[0](x.contiguous(), mask=None)
        a1 = attn_w(tf.layerwise_blocks[1], x1)
        a0n, a1n = a0.numpy(), a1.numpy()
        assert abs(ast.head_token_average(a1n).sum(axis=1).mean() - 1.0) < 1e-3, "attention rows must sum to ~1"
        m0, m1 = ast.head_token_average(a0n), ast.head_token_average(a1n)
        means[name] = (m0, m1)
        per_head[name] = ast.per_head_average(a1n)
        np.save(out / f"attn_{name}_b0_mean.npy", m0)
        np.save(out / f"attn_{name}_b1_mean.npy", m1)
        rank = ast.hub_ranking(m1, labels)
        print(f"  {name}: seq={seq}  block1 hub ranking: {[f'{l}:{v:.3f}' for l, v in rank[:4]]}", flush=True)

    # cross-prompt figure (rows = prompt, cols = block0/block1, head+token averaged)
    n = len(prompts)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n), squeeze=False)
    for r, name in enumerate(means):
        for c, (m, t) in enumerate([(means[name][0], "block0"), (means[name][1], "block1")]):
            ax = axes[r][c]
            im = ax.imshow(m, cmap="viridis")
            ax.set_xticks(range(12)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(12)); ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"prompt {r} — {t}", fontsize=10); ax.set_xlabel("attends TO", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Krea2 layer-fusion attention across prompts")
    fig.tight_layout(); fig.savefig(out / "attention_cross_prompt.png", dpi=100)

    # per-head figure for prompt 0, block 1
    ph = per_head["p0"]
    H = ph.shape[0]; cols = 5; rows = (H + cols - 1) // cols
    fig2, ax2 = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for h in range(rows * cols):
        a = ax2.flat[h]
        if h < H:
            a.imshow(ph[h], cmap="viridis"); a.set_title(f"head {h}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig2.suptitle("Per-head layerwise-block-1 attention (prompt 0)")
    fig2.tight_layout(); fig2.savefig(out / "attention_per_head.png", dpi=100)
    print(f"wrote figures + .npy to {out}", flush=True)


if __name__ == "__main__":
    main()
