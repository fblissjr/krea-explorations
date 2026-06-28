"""ComfyUI node: route Krea 2 DiT attention through SageAttention -- our own override, no KJNodes.

Why this exists instead of leaning on KJNodes' ``PathchSageAttentionKJ``:

- The patch primitive is *core ComfyUI*, not a KJ invention. ``comfy/ldm/modules/attention.py``
  (``wrap_attn``) reads ``transformer_options["optimized_attention_override"]`` and, if present, calls it as
  ``override(func, q, k, v, heads, mask=..., skip_reshape=..., ...)`` in place of the default attention. All
  KJNodes does is set that key to a sage-calling closure. So we owe it nothing -- setting the same key
  ourselves is ~10 lines and gives us telemetry + a mask/seq guard the KJ node doesn't expose.
- The KJ node is author-flagged ``EXPERIMENTAL`` and "can't be disabled without running the node again"
  (it mutates ``model_options`` outside the reversible patcher). Owning the override keeps that risk out of
  any production krea2 path.

What this targets: Krea 2's main DiT self-attn (``comfy/ldm/krea2/model.py:87``) -- 4096 image + text
tokens, head_dim 128, **mask=None** in the denoise loop (``model.py:270`` passes ``None``). That unmasked,
large-seq call is where sage's win lives, so by default we sage exactly that and fall back to ComfyUI's own
attention for everything else (the tiny masked refiner-text blocks, any short attention). That isolates the
measurement variable to the one call that matters.

Known non-exploited lever (deliberately out of scope here): krea2 is GQA (48 Q / 12 KV) but the impl
``repeat_interleave``s KV to 48 heads *before* this call (``model.py:83-86``), so the override only ever sees
48-head MHA -- sage's native ``q_per_kv`` GQA support is moot. Capturing that needs a deeper patch of
``Attention.forward`` itself; left for a real node once the workload profile says it's worth it.

``torch`` / ``comfy`` / ``sageattention`` are imported lazily inside the closure so this module stays
importable (and the decision logic unit-testable) without the ComfyUI runtime or a GPU.
"""

from __future__ import annotations

import sys

# pv_accum_dtype per mode -- mirrors KJNodes' mode table so numbers are comparable across the two paths.
# "auto" routes through the sageattn() dispatcher (sm89 + CUDA>=12.8 -> fp8_cuda++ == fp32+fp16).
# "sdpa" forces ComfyUI's attention_pytorch on every call -- a true SDPA baseline arm that overrides a
# global --use-sage-attention launch flag, so the sage-vs-SDPA A/B works in one running server.
SAGE_MODES = ["auto", "fp8_cuda++", "fp8_cuda", "fp16_cuda", "fp16_triton", "sdpa", "disabled"]

# telemetry dedup: one stderr line per (mode, kernel, q-shape) seen, process-wide.
_seen_fire: set[tuple] = set()


def _reset_telemetry_for_test() -> None:
    """Test-only: clear the fired-once dedup set."""
    _seen_fire.clear()


def _should_sage(mask, seq_len: int, min_seq_len: int, sage_masked: bool) -> bool:
    """Pure decision: does this attention call get routed to sage, or fall back to ComfyUI's default?

    Default policy isolates the load-bearing call: sage only the unmasked, large-seq self-attn (the krea2
    DiT image stream); let small attentions and the masked refiner-text blocks use the default path. Kept
    free of torch so it is unit-testable without a GPU.
    """
    if mask is not None and not sage_masked:
        return False
    if seq_len < min_seq_len:
        return False
    return True


def make_sage_override(sage_mode: str = "auto", min_seq_len: int = 1024,
                       sage_masked: bool = False, verbose: bool = True, nvtx: bool = False):
    """Build the ``optimized_attention_override`` closure for ``sage_mode``.

    Returns a callable ``(func, q, k, v, heads, ...) -> out`` matching ComfyUI's override contract
    (``comfy/ldm/modules/attention.py`` ``wrap_attn``). On any shape we don't handle, or any call the policy
    declines, we call ``func`` -- ComfyUI's own attention -- so we never silently corrupt or drop a call.
    """
    import torch
    from sageattention import sageattn  # noqa: F401  (resolved lazily in the server env)

    try:
        from sageattention import get_last_dispatched_kernel
    except Exception:  # older sage without the telemetry accessor
        get_last_dispatched_kernel = lambda: None  # noqa: E731

    def _sage_call(q, k, v, tensor_layout, mask):
        if sage_mode in ("auto", "disabled"):
            return sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
        # Explicit kernels mirror KJNodes' mode->pv_accum table.
        if sage_mode == "fp8_cuda++":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                                pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
        if sage_mode == "fp8_cuda":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                                pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
        if sage_mode == "fp16_cuda":
            from sageattention import sageattn_qk_int8_pv_fp16_cuda
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                                 pv_accum_dtype="fp32", tensor_layout=tensor_layout)
        if sage_mode == "fp16_triton":
            from sageattention import sageattn_qk_int8_pv_fp16_triton
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=mask, is_causal=False,
                                                   tensor_layout=tensor_layout)
        raise ValueError(f"unknown sage_mode {sage_mode!r}")

    def _override(func, q, k, v, heads, mask=None, attn_precision=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
        if sage_mode == "sdpa":
            # True-SDPA baseline arm: force ComfyUI's pytorch attention on every call, overriding a global
            # --use-sage-attention flag, so we can measure sage's e2e delta against real SDPA in one server.
            from comfy.ldm.modules.attention import attention_pytorch
            return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                     skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
        # seq_len + dim_head from whichever layout we were handed (both bound on every path).
        if skip_reshape:        # q is [B, H, L, D]
            b, _, seq_len, dim_head = q.shape
            tensor_layout = "HND"
        else:                   # q is [B, L, H*D]
            b, seq_len, hidden = q.shape
            dim_head = hidden // heads
            tensor_layout = "NHD"

        if not _should_sage(mask, seq_len, min_seq_len, sage_masked):
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        in_dtype = v.dtype
        qq, kk, vv = q, k, v
        if qq.dtype == torch.float32 or kk.dtype == torch.float32 or vv.dtype == torch.float32:
            qq, kk, vv = qq.to(torch.float16), kk.to(torch.float16), vv.to(torch.float16)
        if not skip_reshape:    # split heads to the NHD layout sage expects
            qq, kk, vv = (t.view(b, -1, heads, dim_head) for t in (qq, kk, vv))
        m = mask
        if m is not None:
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.ndim == 3:
                m = m.unsqueeze(1)

        try:
            if nvtx:
                torch.cuda.nvtx.range_push(f"krea2_sage[{sage_mode}] L={seq_len} H={heads} D={dim_head}")
            out = _sage_call(qq, kk, vv, tensor_layout, m).to(in_dtype)
        except Exception as exc:  # any kernel reject -> fall back, never corrupt the render
            if verbose:
                sys.stderr.write(f"[krea2-sage] FALLBACK ({type(exc).__name__}: {exc}) "
                                 f"L={seq_len} H={heads} D={dim_head} -> default attention\n")
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()

        # match ComfyUI's output contract for the layout we were given
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            out = out.transpose(1, 2) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)

        if verbose:
            key = (sage_mode, get_last_dispatched_kernel(), tuple(q.shape))
            if key not in _seen_fire:
                _seen_fire.add(key)
                sys.stderr.write(f"[krea2-sage] FIRED mode={sage_mode} kernel={key[1]} "
                                 f"q={tuple(q.shape)} layout={tensor_layout} masked={mask is not None}\n")
        return out

    return _override


class Krea2SageAttention:
    """Patch a Krea 2 model so its DiT self-attn runs on SageAttention (our override, not KJNodes).

    Insert between the model loader (or ModelSamplingFlux) and the sampler. ``auto`` lets sage's dispatcher
    pick -- on sm89 + CUDA>=12.8 that is the fp8_cuda++ (fp32+fp16) kernel. The console prints one
    ``[krea2-sage] FIRED ... kernel=...`` line per unique shape so you can confirm sage actually ran and on
    which kernel (rung-1 of the evidence ladder), plus sage's own ``[INFO] sage routing: ...`` line.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sage_mode": (SAGE_MODES, {
                    "default": "auto",
                    "tooltip": "'auto' = sageattn() dispatcher (sm89 -> fp8_cuda++). 'disabled' = no patch "
                               "(passthrough). Explicit modes mirror KJNodes' pv_accum table.",
                }),
            },
            "optional": {
                "min_seq_len": ("INT", {
                    "default": 1024, "min": 0, "max": 1 << 20,
                    "tooltip": "Only sage attention with seq_len >= this; shorter calls use default attention. "
                               "Targets the big DiT self-attn (4096+ tokens), skips tiny text attentions.",
                }),
                "sage_masked": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also route masked calls through sage (sm89 fp8++ supports masks). Default off "
                               "keeps the masked refiner-text blocks on the default path to isolate the win.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print one FIRED line per unique shape (proof sage ran + which kernel).",
                }),
                "nvtx": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Wrap each sage call in an NVTX range for `nsys` timeline attribution.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Krea2/optimize"
    DESCRIPTION = ("Route Krea 2 DiT self-attn through SageAttention via the core ComfyUI "
                   "optimized_attention_override hook (no KJNodes dependency). Default policy sages only the "
                   "unmasked large-seq self-attn and falls back to default attention otherwise.")

    def patch(self, model, sage_mode, min_seq_len=1024, sage_masked=False, verbose=True, nvtx=False):
        if sage_mode == "disabled":
            return (model,)
        import uuid

        m = model.clone()
        override = make_sage_override(sage_mode=sage_mode, min_seq_len=int(min_seq_len),
                                      sage_masked=bool(sage_masked), verbose=bool(verbose), nvtx=bool(nvtx))
        # transformer_options is the per-model dict ComfyUI threads down to every attention call.
        m.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = override
        # Bump patches_uuid so ComfyUI's sampler cache treats the patched model as changed and does NOT
        # reuse an unpatched baseline latent -- an override closure does not alter the model hash on its
        # own, so without this the A/B silently returns the cached SDPA result (same precedent as the
        # projector-rebalance node). Caught 2026-06-28 by a bit-identical off-vs-on image diff.
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2SageAttention": Krea2SageAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SageAttention": "Krea 2 Sage Attention"}

__all__ = ["Krea2SageAttention", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "make_sage_override", "_should_sage", "_reset_telemetry_for_test", "SAGE_MODES"]
