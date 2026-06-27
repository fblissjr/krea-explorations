"""Krea2 image-attention untwisting patch — piece 2 of our own untwisting-RoPE node.

Mirrors the real Krea2 ``Attention.forward`` (comfy/ldm/krea2/model.py):
    wq/wk/wv,gate -> rearrange B L (H D)->B H L D -> qknorm(q,k) -> apply_rope(q,k,freqs)
    -> GQA repeat_interleave -> optimized_attention_masked -> wo(out * sigmoid(gate))

On top, when a reference is present in the batch (items ``[target_bsz:2*target_bsz]``) and untwisting is
enabled, the TARGET branch additionally attends to the reference's IMAGE keys/values, with the reference
keys multiplied by our ``[32,48,48]`` frequency-scale vector (see ``rope_untwist``). High-freq bands are
attenuated (kill positional copying), low-freq amplified (semantic/style) -> shared attention transfers
*style* without copying layout.

Scoped to the image DiT blocks only (``blocks.<i>.attn``); NOT ``txtfusion.{layerwise,refiner}_blocks``
(the community fork's over-patch). Text tokens are never touched.

Requires torch; this module is imported only in the ComfyUI runtime / tests, never at package import time
(the project package stays torch-free).
"""
from __future__ import annotations

import torch


def is_image_attention_name(name: str) -> bool:
    """True only for SingleStreamDiT image blocks ('blocks.<i>.attn'); excludes txtfusion attention,
    which shares the same Attention class (the fork's bug was matching it too)."""
    n = name.lower()
    return n.endswith(".attn") and n.startswith("blocks.") and "txtfusion" not in n


def untwist_reference_image_keys(
    k: torch.Tensor, scale_vec: torch.Tensor, img_slice: tuple[int, int], target_bsz: int
) -> torch.Tensor:
    """Reference branch's IMAGE keys, scaled per-frequency. ``k``: [2B,H,S,D] post-rope+GQA.

    Reads (does not mutate) the reference slice ``k[target_bsz:2*target_bsz, :, img_s:img_e, :]`` and
    multiplies it by ``scale_vec`` broadcast over the head_dim (D) axis.
    """
    img_s, img_e = img_slice
    sv = scale_vec.to(device=k.device, dtype=k.dtype).view(1, 1, 1, -1)
    return k[target_bsz:2 * target_bsz, :, img_s:img_e, :] * sv


def build_shared_kv(
    k: torch.Tensor, v: torch.Tensor, scale_vec: torch.Tensor,
    img_slice: tuple[int, int], target_bsz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target-branch K/V with the untwisted reference IMAGE K/V appended along the sequence axis.

    Returns ``(k_t, v_t)`` each shaped ``[target_bsz, H, S + n_img, D]``. The target's own K/V are left
    intact; only the reference's image tokens (untwisted keys, raw values) are appended.
    """
    img_s, img_e = img_slice
    ref_k = untwist_reference_image_keys(k, scale_vec, img_slice, target_bsz)
    ref_v = v[target_bsz:2 * target_bsz, :, img_s:img_e, :]
    k_t = torch.cat([k[:target_bsz], ref_k], dim=2)
    v_t = torch.cat([v[:target_bsz], ref_v], dim=2)
    return k_t, v_t


def make_untwist_forward(scale_vec: torch.Tensor, attention_fn=None):
    """Build a replacement ``Attention.forward`` (bound via ``types.MethodType``) for an image block.

    Reads its runtime config from ``transformer_options['krea2_untwist']``:
      ``enabled`` (bool), ``target_bsz`` (int), ``img_slice`` (img_s, img_e).
    When disabled / no reference in the batch, it reproduces the stock forward exactly. ``attention_fn``
    defaults to comfy's ``optimized_attention_masked`` (imported lazily so this module loads without comfy).
    """
    if attention_fn is None:
        from comfy.ldm.modules.attention import optimized_attention_masked as attention_fn  # noqa: N806
    from comfy.ldm.flux.math import apply_rope
    from einops import rearrange

    def forward(self, x, freqs=None, mask=None, transformer_options={}):  # noqa: B006 (matches stock sig)
        cfg = (transformer_options or {}).get("krea2_untwist") or {}
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

        active = bool(cfg.get("enabled")) and cfg.get("target_bsz", 0) > 0 \
            and x.shape[0] >= 2 * cfg.get("target_bsz", 0)
        if not active:
            out = attention_fn(q, k, v, self.heads, mask=mask, skip_reshape=True,
                               transformer_options=transformer_options)
            return self.wo(out * torch.sigmoid(gate))

        tb = int(cfg["target_bsz"])
        # combined sequence is [text (txtlen) | image (rest)]; share only the reference's IMAGE keys.
        img_slice = (int(cfg["txtlen"]), k.shape[2])
        k_t, v_t = build_shared_kv(k, v, scale_vec, img_slice, tb)
        out_t = attention_fn(q[:tb], k_t, v_t, self.heads, mask=None, skip_reshape=True,
                             transformer_options=transformer_options)
        # reference (and any extra batch items) denoise against their own K/V, unchanged
        out_rest = attention_fn(q[tb:], k[tb:], v[tb:], self.heads, mask=None, skip_reshape=True,
                                transformer_options=transformer_options)
        out = torch.cat([out_t, out_rest], dim=0)
        return self.wo(out * torch.sigmoid(gate))

    return forward
