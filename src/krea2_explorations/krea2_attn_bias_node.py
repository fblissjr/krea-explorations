"""ComfyUI node: bias image->text cross-attention toward chosen text-token positions in chosen DiT blocks.

A targeted attention-routing control / interpretability probe for Krea2's DiT. Image tokens attend to the
text-conditioning tokens with varying strength across blocks; this node adds an additive bias to the attention
logits for image-query -> chosen text-key positions in chosen blocks, so you can study or strengthen how much
the image attends to specific words (e.g. tokens the model otherwise under-attends).

Implementation: wrap each chosen block's ``attn.forward`` and inject an additive mask into the SAME call (no
q/k/v re-derivation -> faithful; render is unchanged at bias 0). Applied on the conditioned pass only (first
call per denoise step; the key positions index the positive prompt). ``comfy``/torch imported lazily.
"""

from __future__ import annotations


def parse_blocks(spec, n):
    """'5-10' / '5,6,7' / '5-10,18-26' -> sorted unique block indices within [0,n)."""
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


def parse_positions(spec):
    return [int(x) for x in str(spec).replace(" ", "").split(",") if x != ""]


class Krea2AttnBias:
    """Bias image->text attention toward chosen text-token positions in chosen DiT blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks": ("STRING", {"default": "5-10", "tooltip": "DiT blocks to bias, e.g. '5-10' or '5,6,7'"}),
                "key_positions": ("STRING", {"default": "", "tooltip": "comma text-token positions to boost "
                                  "(from attn_tokens.py); empty = boost ALL text keys"}),
                "bias": ("FLOAT", {"default": 4.0, "min": -20.0, "max": 20.0, "step": 0.5,
                                   "tooltip": "additive attention-logit bias on image->target-text (in nats)"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Add an attention bias making image patches attend to chosen text-token positions in chosen "
                   "DiT blocks. Targeted attention-routing control / interpretability probe.")

    def apply(self, model, blocks, key_positions, bias):
        import uuid
        from types import MethodType

        import torch

        m = model.clone()
        blk_list = parse_blocks(blocks, len(m.get_model_object("diffusion_model.blocks")))
        cols = parse_positions(key_positions)
        bias = float(bias)
        state = {"txtlen": 0, "active": False, "last_t": None}

        def make_fwd(orig):
            def fwd(self, x, freqs=None, mask=None, transformer_options={}):
                if state["active"] and state["txtlen"] > 0:
                    L = x.shape[1]
                    tl = state["txtlen"]
                    bm = torch.zeros((1, 1, L, L), device=x.device, dtype=x.dtype)
                    kcols = [c for c in cols if 0 <= c < tl] if cols else list(range(tl))
                    if kcols:
                        idx = torch.tensor(kcols, device=x.device)
                        bm[:, :, tl:, idx] = bias  # image-query rows -> chosen text-key cols
                        mask = bm if mask is None else mask + bm
                return orig(x, freqs=freqs, mask=mask, transformer_options=transformer_options)
            return fwd

        for i in blk_list:
            attn = m.get_model_object(f"diffusion_model.blocks.{i}.attn")
            m.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward", MethodType(make_fwd(attn.forward), attn))

        def unet_wrapper(apply_model, params):
            t = float(params["timestep"].flatten()[0])
            cc = params["c"].get("c_crossattn")
            state["txtlen"] = int(cc.shape[1]) if cc is not None else 0
            state["active"] = state["last_t"] is None or t != state["last_t"]  # cond pass (first call per step)
            state["last_t"] = t
            return apply_model(params["input"], params["timestep"], **params["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2AttnBias": Krea2AttnBias}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2AttnBias": "Krea 2 Attention Bias"}

__all__ = ["Krea2AttnBias", "parse_blocks", "parse_positions",
           "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
