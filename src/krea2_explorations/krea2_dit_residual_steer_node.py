"""ComfyUI node: steer the DiT image-residual toward a measured per-block direction inside chosen blocks.

Adds a measured direction to the image-token residual inside chosen blocks. The direction is a per-block
(num_blocks x features) array -- e.g. the difference between two model variants' residuals, precomputed to an
.npz -- added proportionally to each token's own magnitude (scale * ||token|| * dir_hat) so it stays roughly
in-distribution (RMSNorm-safe). Applied on the conditioned pass only.

``comfy``/torch imported lazily; the steer helper is numpy/torch-agnostic for unit testing.
"""

from __future__ import annotations

from .krea2_attn_bias_node import parse_blocks


def steer_image_residual(img, dir_hat, scale):
    """Add scale * per-token-norm * dir_hat to each image token. img: (..., feat). numpy/torch-agnostic."""
    norm = (img * img).sum(-1, keepdims=True) ** 0.5
    return img + scale * norm * dir_hat


class Krea2DiTResidualSteer:
    """Steer the DiT image-residual toward a measured per-block direction inside chosen blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks": ("STRING", {"default": "18-26", "tooltip": "DiT blocks to steer, e.g. '18-26'"}),
                "scale": ("FLOAT", {"default": 1.5, "min": -10.0, "max": 10.0, "step": 0.25,
                                    "tooltip": "fraction of each token's own magnitude to push along the direction"}),
            },
            "optional": {
                "dir_path": ("STRING", {"default": "data/dit_residual_dirs.npz",
                                        "tooltip": "npz with per-block directions (num_blocks x features)"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Add a measured per-block direction to the DiT image-residual inside chosen blocks. "
                   "Proportional/in-distribution, conditioned-pass only.")

    def apply(self, model, blocks, scale, dir_path="data/dit_residual_dirs.npz"):
        import uuid
        from pathlib import Path
        from types import MethodType

        import numpy as np
        import torch

        p = Path(dir_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / dir_path
        z = np.load(p)
        blk_dirs = {int(b): z["dirs"][i] for i, b in enumerate(z["blocks"])}  # block -> (6144,)
        m = model.clone()
        blk_list = parse_blocks(blocks, len(m.get_model_object("diffusion_model.blocks")))
        scale = float(scale)
        state = {"txtlen": 0, "active": False, "last_t": None}
        cache = {}  # block -> unit dir tensor on device

        def make_fwd(orig, idx):
            def fwd(self, x, *args, **kwargs):
                out = orig(x, *args, **kwargs)
                if state["active"] and state["txtlen"] > 0 and idx in blk_dirs:
                    try:
                        if idx not in cache:
                            d = torch.from_numpy(blk_dirs[idx]).to(device=out.device, dtype=torch.float32)
                            cache[idx] = d / (d.norm() + 1e-8)
                        tl = state["txtlen"]
                        img = out[:, tl:, :].float()
                        out[:, tl:, :] = steer_image_residual(img, cache[idx], scale).to(out.dtype)
                    except Exception as e:
                        if not state.get("warned"):
                            print(f"[Krea2DiTResidualSteer] error: {e}", flush=True)
                            state["warned"] = True
                return out
            return fwd

        for i in blk_list:
            blk = m.get_model_object(f"diffusion_model.blocks.{i}")
            m.add_object_patch(f"diffusion_model.blocks.{i}.forward", MethodType(make_fwd(blk.forward, i), blk))

        def unet_wrapper(apply_model, params):
            t = float(params["timestep"].flatten()[0])
            cc = params["c"].get("c_crossattn")
            state["txtlen"] = int(cc.shape[1]) if cc is not None else 0
            state["active"] = state["last_t"] is None or t != state["last_t"]  # cond pass
            state["last_t"] = t
            return apply_model(params["input"], params["timestep"], **params["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2DiTResidualSteer": Krea2DiTResidualSteer}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2DiTResidualSteer": "Krea 2 DiT Residual Steer"}

__all__ = ["Krea2DiTResidualSteer", "steer_image_residual",
           "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
