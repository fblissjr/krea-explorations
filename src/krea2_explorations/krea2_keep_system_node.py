"""ComfyUI node: encode Krea 2 text conditioning WITHOUT stripping the system turn.

Krea 2's text encoder (``comfy/text_encoders/krea2.py`` ``Krea2TEModel.encode_token_weights``) slices the
conditioning from the *second* ``<|im_start|>`` (the user turn) onward, so a system-role prompt is discarded
before the DiT and can only influence the result indirectly (the surviving tokens attended back over it).

That method already takes a ``template_end`` argument that, when set to anything other than ``-1``, skips the
auto-strip and cuts exactly there (``template_end=0`` keeps the whole sequence). We feed it through a tiny,
reversible *per-instance* hook so a system prompt becomes a **direct** conditioning write-point — a steering
vector you can drive from inside ComfyUI. No re-tokenization, no per-mode drop-count tables, no global state:
just the existing escape hatch, exposed as a node.

``comfy`` is imported lazily (inside ``_ensure_patched``) so this module stays importable/testable without
the ComfyUI runtime.
"""

from __future__ import annotations

import functools

# per-instance attribute read by the patched method; None -> Comfy's original strip behavior.
_ATTR = "_krea2_template_end"
_HOOK_FLAG = "_krea2_keep_system_hook"
_PATCHED = False


def _wrap(orig):
    """Wrap ``encode_token_weights`` so a per-instance ``_ATTR`` can supply ``template_end``.

    Backward compatible: with no attribute set (or set to ``None``) the default ``-1`` is preserved, so
    untouched callers get Comfy's original auto-strip. An explicit non-default ``template_end`` from a
    caller always wins. Kept as a standalone factory so the logic is unit-testable without comfy/torch.
    """

    @functools.wraps(orig)
    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        if template_end == -1:
            override = getattr(self, _ATTR, None)
            if override is not None:
                template_end = override
        return orig(self, token_weight_pairs, template_end=template_end)

    return encode_token_weights


def _ensure_patched():
    """Idempotently install the hook on ``Krea2TEModel.encode_token_weights`` (lazy; needs comfy)."""
    global _PATCHED
    if _PATCHED:
        return
    import comfy.text_encoders.krea2 as krea2

    cls = krea2.Krea2TEModel
    if not getattr(cls.encode_token_weights, _HOOK_FLAG, False):
        wrapped = _wrap(cls.encode_token_weights)
        setattr(wrapped, _HOOK_FLAG, True)
        cls.encode_token_weights = wrapped
    _PATCHED = True


class Krea2EncodeKeepSystem:
    """Encode Krea 2 text conditioning while keeping the system turn (a usable ComfyUI node).

    Drop-in alternative to ``CLIPTextEncode`` for Krea 2: outputs ``CONDITIONING`` that includes the system
    turn, so a system-role prompt actually reaches the DiT. To set a system prompt, pass a full
    ``<|im_start|>system ... <|im_end|>\\n<|im_start|>user ... <|im_end|>\\n<|im_start|>assistant\\n`` string
    as ``text`` (Krea 2's tokenizer tokenizes it verbatim via its skip-template route).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True, "dynamicPrompts": True,
                    "tooltip": "Prompt text. For a system prompt, pass a full <|im_start|>system ... "
                               "<|im_end|><|im_start|>user ...<|im_end|><|im_start|>assistant\\n string; "
                               "the system turn is kept (not stripped).",
                }),
            },
            "optional": {
                "template_end": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Leading conditioning tokens to drop. 0 = keep everything incl. the system "
                               "turn. Comfy's default auto-strips to the user turn (use this node to override).",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Encode Krea 2 text conditioning WITHOUT stripping the system turn, so a system-role "
                   "prompt becomes a direct conditioning write-point (steering vector). Drop-in for "
                   "CLIPTextEncode.")

    def encode(self, clip, text, template_end=0):
        _ensure_patched()
        tokens = clip.tokenize(text)
        model = clip.cond_stage_model
        prev = getattr(model, _ATTR, None)
        setattr(model, _ATTR, int(template_end))
        try:
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        finally:
            setattr(model, _ATTR, prev)  # reversible: never leaks to other encodes/CLIPs
        return ([[cond, {"pooled_output": pooled}]],)


NODE_CLASS_MAPPINGS = {"Krea2EncodeKeepSystem": Krea2EncodeKeepSystem}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2EncodeKeepSystem": "Krea 2 Encode (keep system turn)"}

__all__ = ["Krea2EncodeKeepSystem", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "_wrap", "_ATTR"]
