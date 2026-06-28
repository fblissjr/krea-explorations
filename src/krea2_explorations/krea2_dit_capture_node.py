"""ComfyUI node: capture per-block DiT activations to disk for interpretability.

Reversible object-patches (same pattern as the untwist node) record, per (block, step), one summary:

  residual   mean-pooled image-token residual                  -> per-block change between two model variants
  spatial    per-image-token residual L2 norm (reshape to HxW)  -> WHERE in the frame a change concentrates
  attention  image-query -> text-key attention mass             -> how much the image attends to the conditioning

CFG-correct: ComfyUI calls the unet wrapper per-cond, and at cfg>1 the batched row order is [uncond, cond] (or
two unbatched calls). So we (a) increment the denoise step on TIMESTEP CHANGE (not per call), (b) capture only
the FIRST call of each timestep = the conditioned pass, (c) pool the LAST batch row. State resets per apply().
Run the same prompt+seed twice (e.g. with/without a LoRA) with distinct run_tag, then diff per block.

``comfy``/torch imported lazily; the pooling helpers are numpy/torch-agnostic for unit testing.
"""

from __future__ import annotations

CAPTURE_MODES = ["residual", "spatial", "attention"]


def image_residual_summary(out, txtlen):
    """Mean-pool the image-token residual (tokens after txtlen), COND row (last). numpy/torch-agnostic."""
    return out[:, txtlen:, :].mean(1)[-1]


def image_residual_spatial(out, txtlen):
    """Per-image-token residual L2 norm, COND row (last) -> (imglen,). Reshape to HxW downstream."""
    img = out[-1, txtlen:, :]  # (imglen, feat), cond row
    img = img.float() if hasattr(img, "float") else img  # fp8/bf16 squares overflow to inf -> upcast first
    return (img * img).sum(-1) ** 0.5


def reduce_attn_to_text(attn_img_rows, txtlen):
    """Given softmax attn weights for image-query rows (..., n_img, n_keys), return per-text-key received mass
    (txtlen,) averaged over image queries (and any leading head/batch dims). numpy/torch-agnostic."""
    txt = attn_img_rows[..., :txtlen]  # (..., n_img, txtlen)
    return txt.mean(-2).reshape(-1, txtlen).mean(0)  # avg over image queries, then over heads/batch


class Krea2DiTCapture:
    """Capture per-block DiT activations (residual / spatial / attention) to disk for EXP5. Reversible patch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "out_dir": ("STRING", {"default": "data/dit_capture",
                                       "tooltip": "written to <out_dir>/<run_tag>/ ; the run_tag dir is CLEARED first"}),
                "run_tag": ("STRING", {"default": "run", "tooltip": "e.g. 'a' vs 'b' for the per-block diff"}),
                "mode": (CAPTURE_MODES, {"default": "residual"}),
            },
            "optional": {
                "attn_query_sample": ("INT", {"default": 512, "min": 32, "max": 8192,
                                              "tooltip": "attention mode: # image-query rows sampled (bounds memory)"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Capture per-block DiT image residuals (residual/spatial) or image->text attention "
                   "(attention) to disk, conditioned-pass only, per denoise step. Run twice (e.g. with/without "
                   "a LoRA) and diff per block.")

    def apply(self, model, out_dir, run_tag, mode="residual", attn_query_sample=512):
        import shutil
        import uuid
        from pathlib import Path
        from types import MethodType

        import numpy as np
        import torch

        m = model.clone()
        blocks = m.get_model_object("diffusion_model.blocks")
        n = len(blocks)
        outp = Path(out_dir) / run_tag
        if outp.exists():
            shutil.rmtree(outp)  # H2: never average across stale runs/configs
        outp.mkdir(parents=True, exist_ok=True)
        # fresh state per apply() (C3): step counts denoise steps; capture gates the cond pass.
        state = {"step": -1, "txtlen": 0, "capture": False, "last_t": None, "warned": False}

        def _save(name, arr):
            np.save(outp / name, np.asarray(arr, dtype=np.float32))

        if mode in ("residual", "spatial"):
            summary = image_residual_summary if mode == "residual" else image_residual_spatial

            def make_fwd(orig, idx):
                def fwd(self, x, *args, **kwargs):
                    out = orig(x, *args, **kwargs)
                    if state["capture"]:
                        try:
                            if state["txtlen"] <= 0 and not state["warned"]:
                                print("[Krea2DiTCapture] WARN txtlen<=0 -> pooling text+image", flush=True)
                                state["warned"] = True
                            _save(f"b{idx:02d}_s{state['step']:02d}.npy",
                                  summary(out, state["txtlen"]).float().cpu().numpy())
                        except Exception as e:  # don't break the render, but surface once
                            if not state["warned"]:
                                print(f"[Krea2DiTCapture] capture error: {e}", flush=True)
                                state["warned"] = True
                    return out
                return fwd

            for i in range(n):
                blk = m.get_model_object(f"diffusion_model.blocks.{i}")
                m.add_object_patch(f"diffusion_model.blocks.{i}.forward", MethodType(make_fwd(blk.forward, i), blk))

        else:  # attention: faithful Attention.forward replica (same q,k,v -> render unchanged) + capture mass
            from einops import rearrange
            import torch.nn.functional as F
            from comfy.ldm.flux.math import apply_rope
            from comfy.ldm.modules.attention import optimized_attention_masked

            def make_attn_fwd(idx):
                def fwd(self, x, freqs=None, mask=None, transformer_options={}):
                    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
                    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
                    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
                    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
                    q, k = self.qknorm(q, k)
                    if freqs is not None:
                        q, k = apply_rope(q, k, freqs)
                    if self.kvheads != self.heads:
                        rep = self.heads // self.kvheads
                        k = k.repeat_interleave(rep, dim=1)
                        v = v.repeat_interleave(rep, dim=1)
                    if state["capture"]:
                        try:
                            tl = state["txtlen"]
                            qc, kc = q[-1].float(), k[-1].float()  # cond row -> (H, L, D)
                            L = qc.shape[1]
                            img0 = max(tl, 0)
                            n_img = L - img0
                            if tl > 0 and n_img > 0:
                                idxs = torch.linspace(img0, L - 1, min(attn_query_sample, n_img)).long()
                                qs = qc[:, idxs, :]  # (H, S, D) sampled image queries
                                scores = (qs @ kc.transpose(-2, -1)) / (qc.shape[-1] ** 0.5)  # (H, S, L)
                                attn = scores.softmax(dim=-1)
                                per_txt = reduce_attn_to_text(attn, tl)  # (tl,) text tokens the image attends to
                                mass = float(attn[..., :tl].sum(-1).mean())  # avg image-query mass on text
                                _save(f"attn_b{idx:02d}_s{state['step']:02d}.npy", per_txt.cpu().numpy())
                                _save(f"mass_b{idx:02d}_s{state['step']:02d}.npy", [mass])
                        except Exception as e:
                            if not state["warned"]:
                                print(f"[Krea2DiTCapture] attn capture error: {e}", flush=True)
                                state["warned"] = True
                    out = optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                                     transformer_options=transformer_options)
                    return self.wo(out * F.sigmoid(gate))
                return fwd

            for i in range(n):
                attn = m.get_model_object(f"diffusion_model.blocks.{i}.attn")
                m.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward", MethodType(make_attn_fwd(i), attn))

        def unet_wrapper(apply_model, params):
            t = float(params["timestep"].flatten()[0])
            cc = params["c"].get("c_crossattn")
            state["txtlen"] = int(cc.shape[1]) if cc is not None else 0
            is_new = state["last_t"] is None or t != state["last_t"]  # first call of a denoise step = cond pass
            if is_new:
                state["step"] += 1
            state["capture"] = is_new
            state["last_t"] = t
            return apply_model(params["input"], params["timestep"], **params["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2DiTCapture": Krea2DiTCapture}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2DiTCapture": "Krea 2 DiT Capture"}

__all__ = ["Krea2DiTCapture", "image_residual_summary", "image_residual_spatial", "reduce_attn_to_text",
           "CAPTURE_MODES", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
