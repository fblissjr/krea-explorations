"""ComfyUI node: training-free style transfer on Krea2 via untwisting-RoPE shared attention.

Our own implementation (not the AshmoTV fork). Three pieces, each grounded in our measurements:
  1. correct [32,48,48] band schedule              -> rope_untwist.krea2_freq_scale_vector
  2. clean Krea2 image-attention patch (image      -> krea2_untwist_attn.make_untwist_forward
     DiT blocks only; txtfusion left alone)
  3. lean reference injection: renoise the          -> the unet wrapper below
     reference to the current sigma each step,
     batch it behind the target, share its
     (untwisted) IMAGE keys/values. No RF-inversion.

torch/comfy are imported lazily inside apply() so the module stays importable without the runtime.
"""
from __future__ import annotations

AXIS0_MODES = ["flat_low", "curve", "constant"]


class Krea2UntwistStyleReference:
    """Patch a Krea2 MODEL for training-free style transfer from a reference latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_latent": ("LATENT", {"tooltip": "Style source. VAEEncode your reference image; "
                                                "it is interpolated to the generation's latent size."}),
                "low_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Style strength: amplifies LOW-freq (semantic/style) reference attention."}),
                "high_scale": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Structure copy: higher = more of the reference's layout bleeds through."}),
                "beta": ("FLOAT", {"default": 50.0, "min": 0.1, "max": 200.0, "step": 0.5,
                    "tooltip": "Steepness of the high->low band curve."}),
            },
            "optional": {
                "axis0_mode": (AXIS0_MODES, {"default": "flat_low",
                    "tooltip": "Krea2's temporal axis (first 32 dims); flat_low matches the paper default."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for the reference renoise -> reproducible shared attention."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Training-free style transfer from a reference latent via untwisting-RoPE shared "
                   "attention, built for Krea2's real [32,48,48] RoPE. No LoRA, no training.")

    def apply(self, model, reference_latent, low_scale, high_scale, beta, axis0_mode="flat_low", seed=0):
        import uuid
        from types import MethodType

        import torch
        import torch.nn.functional as F

        from .krea2_untwist_attn import is_image_attention_name, make_untwist_forward
        from .rope_untwist import krea2_freq_scale_vector

        m = model.clone()

        blocks = m.get_model_object("diffusion_model.blocks")
        n_blocks = len(blocks)
        try:
            head_dim = int(getattr(blocks[0].attn, "headdim", 128))
        except Exception:
            head_dim = 128

        scale_vec = torch.tensor(
            krea2_freq_scale_vector(head_dim, high_scale=high_scale, low_scale=low_scale,
                                    beta=beta, axis0_mode=axis0_mode),
            dtype=torch.float32,
        )

        # 2) reversibly replace each IMAGE-block attention forward (not txtfusion).
        fwd = make_untwist_forward(scale_vec)
        patched = 0
        for i in range(n_blocks):
            if not is_image_attention_name(f"blocks.{i}.attn"):
                continue
            key = f"diffusion_model.blocks.{i}.attn"
            attn = m.get_model_object(key)
            m.add_object_patch(f"{key}.forward", MethodType(fwd, attn))
            patched += 1
        if patched == 0:
            raise RuntimeError("No Krea2 image-block attention found (diffusion_model.blocks.*.attn). "
                               "Is this a Krea2 model loaded via UNETLoader?")

        ref = reference_latent["samples"].detach()
        model_sampling = m.get_model_object("model_sampling")

        # 3) renoise-the-reference unet wrapper: batch the reference behind the target each step.
        def unet_wrapper(apply_model, params):
            x = params["input"]
            t = params["timestep"]
            c = dict(params["c"])
            B = x.shape[0]

            r = ref.to(device=x.device, dtype=x.dtype)
            if r.shape[0] > 1:
                r = r[:1]
            if r.shape[-2:] != x.shape[-2:]:
                hw = tuple(x.shape[-2:])
                if r.ndim == 5:  # [B,C,T,H,W] -> merge T into batch for 2D interpolate, then restore
                    b, ch, tt, h, w = r.shape
                    r = r.movedim(2, 1).reshape(b * tt, ch, h, w)
                    r = F.interpolate(r, size=hw, mode="bilinear", align_corners=False)
                    r = r.reshape(b, tt, ch, *hw).movedim(1, 2)
                else:
                    r = F.interpolate(r, size=hw, mode="bilinear", align_corners=False)
            r = r.repeat(B, *([1] * (r.ndim - 1)))  # rank-agnostic (Krea2 latents are 5D)

            g = torch.Generator(device=x.device).manual_seed(int(seed))
            noise = torch.randn(r.shape, generator=g, device=x.device, dtype=x.dtype)
            r_noised = model_sampling.noise_scaling(t, noise, r)

            x_aug = torch.cat([x, r_noised], dim=0)
            t_aug = torch.cat([t, t], dim=0)

            cc = c.get("c_crossattn")
            txtlen = int(cc.shape[1]) if cc is not None else 0
            for key, val in list(c.items()):
                if torch.is_tensor(val) and val.shape[:1] == (B,):
                    c[key] = torch.cat([val, val], dim=0)

            to = dict(c.get("transformer_options", {}))
            to["krea2_untwist"] = {"enabled": True, "target_bsz": B, "txtlen": txtlen}
            c["transformer_options"] = to

            out = apply_model(x_aug, t_aug, **c)
            return out[:B]

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2UntwistStyleReference": Krea2UntwistStyleReference}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2UntwistStyleReference": "Krea 2 Untwist Style Reference"}
