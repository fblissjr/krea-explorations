#!/usr/bin/env python
"""Measure each tapped layer's TRUE contribution to Krea2's projector output (CPU).

The projector output (the aggregated text vector) is ``output = sum_i W_i * slot_i``, where ``slot_i``
is layer i's representation AT THE PROJECTOR INPUT (i.e. after the 2 layerwise attention blocks), and
``W_i`` is the learned projector weight. For each of the 12 selected layers, averaged over prompts and
tokens, this reports:

  - slot_norm  = ||slot_i||            -- the SENSITIVITY: d(output)/d(W_i) = slot_i, so ||slot_i|| is
                                          how hard W_i can swing the output (the "lever strength").
  - |weight|   = |W_i|
  - term_mag   = |W_i| * ||slot_i||    -- static contribution magnitude (the honest version of the
                                          earlier |w|*encoder-norm proxy, now post-attention).
  - drop_imp   = 1 - cos(output, output - W_i*slot_i)  -- CAUSAL: how much dropping band i rotates the
                                          aggregated vector (accounts for the contrastive cancellation).

Unlike the earlier proxy, slot_i here is the real attention-mixed slot, not the raw encoder hidden state.

    uv run --active python scripts/projector_contribution.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", REPO.parents[1]))
SELECT_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PROMPTS = [
    "A close-up portrait of a young woman with an intense gaze, crimson background.",
    "A dynamic anime painting of a joyful girl in a sailor uniform, blue background.",
    "A minimalist flat-color illustration of a figure wading through ocean waves.",
    "A still life photograph of a ceramic vase on a wooden table, soft daylight.",
]


def main():
    sys.path.insert(0, str(COMFY))
    os.chdir(COMFY)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()

    import numpy as np
    import torch
    from safetensors import safe_open
    import comfy.sd
    import comfy.ops
    from comfy.ldm.krea2.model import TextFusionTransformer

    torch.set_grad_enabled(False)
    te = COMFY / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
    dit = COMFY / "models" / "diffusion_models" / "krea2_turbo_bf16.safetensors"
    labels = [f"L{L}" for L in SELECT_LAYERS]

    print("loading CLIP + txtfusion (CPU) ...", flush=True)
    clip = comfy.sd.load_clip(ckpt_paths=[str(te)], clip_type=comfy.sd.CLIPType.KREA2)
    tf = TextFusionTransformer(12, 2560, 20, 4, False, 20, device=torch.device("cpu"),
                               dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    sd = {}
    with safe_open(str(dit), framework="pt") as f:
        for k in f.keys():
            if k.startswith("txtfusion."):
                sd[k[len("txtfusion."):]] = f.get_tensor(k).float()
    tf.load_state_dict(sd, strict=False)
    tf.eval()
    W = tf.projector.weight.detach()[0].float()  # (12,)

    slot_l, term_l, drop_l = [], [], []
    for prompt in PROMPTS:
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].float()  # (1, seq, 30720)
        b, seq, _ = cond.shape
        x = cond.reshape(b, seq, 12, 2560).reshape(b * seq, 12, 2560).contiguous()
        x = tf.layerwise_blocks[0](x, mask=None)
        x = tf.layerwise_blocks[1](x.contiguous(), mask=None)  # (T, 12, 2560) = projector input
        terms = x * W.view(1, 12, 1)            # (T, 12, 2560)
        output = terms.sum(dim=1)               # (T, 2560)
        slot_norm = x.norm(dim=2).mean(dim=0)   # (12,)  sensitivity
        term_mag = terms.norm(dim=2).mean(dim=0)  # (12,)
        on = output.norm(dim=1)
        drop = torch.empty(12)
        for i in range(12):
            od = output - terms[:, i, :]
            cos = (output * od).sum(1) / (on * od.norm(dim=1) + 1e-8)
            drop[i] = (1.0 - cos).mean()
        slot_l.append(slot_norm.numpy()); term_l.append(term_mag.numpy()); drop_l.append(drop.numpy())

    slot = np.mean(slot_l, 0); term = np.mean(term_l, 0); drop = np.mean(drop_l, 0)
    Wn = W.numpy()
    sf, tf_, df = slot / slot.sum(), term / term.sum(), drop / drop.sum()

    print(f"\n{'layer':5} {'|weight|':>8} {'slot_norm':>9} {'sens%':>6} {'term_mag':>9} {'term%':>6} {'drop_imp':>8} {'drop%':>6}")
    order = np.argsort(-term)
    for i in order:
        print(f"{labels[i]:5} {abs(Wn[i]):8.3f} {slot[i]:9.2f} {sf[i]*100:5.1f} "
              f"{term[i]:9.2f} {tf_[i]*100:5.1f} {drop[i]:8.4f} {df[i]*100:5.1f}")
    print("\nsens% = lever strength (||slot||); term% = static contribution; drop% = causal (rotation if removed)")


if __name__ == "__main__":
    main()
