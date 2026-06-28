"""ComfyUI node: targeted concept amplify / inject / project-out on Krea2 conditioning.

A *targeted* generalization of the bypass LoRA. The bypass is a blunt magnitude boost of the whole deep band
(L26/29/32) at the projector. This operates on the conditioning that *feeds* the projector and aims at a
single measured concept direction d (a 12x2560 difference-of-means axis, flattened to 30720 to match Krea2's
``(B, seq, 12*2560)`` conditioning):

  amplify     cond += scale * (cond . d̂) d̂   -- boost the component ALREADY present (the bypass, but targeted).
                                                 Mathematically can't conjure an ABSENT (unpaired) concept ->
                                                 doubles as a test of the concept-pairing model.
  add         cond += scale * d̂               -- inject the direction regardless of what's present.
  subtract    cond -= scale * d̂
  project_out cond -= (cond . d̂) d̂            -- remove the concept entirely.

Direction is loaded from a .npy/.npz (key 'd') of shape (12,2560) or (30720,) or (2560,) (broadcast to all
12 bands). ``comfy``/torch imported lazily so the math stays unit-testable in the project venv.
"""

from __future__ import annotations


def _unit(d):
    return d / (((d * d).sum()) ** 0.5 + 1e-8)


def apply_direction(cond, d, scale, mode, normalize=True):
    """Apply a concept direction to a conditioning tensor. numpy/torch-compatible (last axis = features)."""
    if mode == "add":
        return cond + scale * (_unit(d) if normalize else d)
    if mode == "subtract":
        return cond - scale * (_unit(d) if normalize else d)
    if mode in ("amplify", "project_out"):
        dh = _unit(d)
        proj = (cond * dh).sum(-1)[..., None]  # per-token component magnitude along d̂
        if mode == "amplify":
            return cond + scale * proj * dh    # scale 1 == bypass's x2 on the present component
        return cond - proj * dh                # project_out
    raise ValueError(f"unknown mode: {mode}")


def _load_direction(path, feat_dim, bands, torch, np):
    """Load (12,2560)/(30720,)/(2560,) -> flat (feat_dim,) torch float vector, layer-major."""
    if not path:
        raise ValueError("Krea2ConceptInject: 'direction_path' is empty -- point it at a .npy/.npz concept "
                         "direction (make one with scripts/concept_direction.py).")
    arr = np.load(path)
    if hasattr(arr, "files"):  # npz
        arr = arr["d"] if "d" in arr.files else arr[arr.files[0]]
    arr = np.asarray(arr, dtype="float32")
    band_dim = feat_dim // bands
    if arr.ndim == 2 and arr.shape == (bands, band_dim):
        arr = arr.reshape(-1)                       # layer-major flatten -> matches encoder
    elif arr.ndim == 1 and arr.shape[0] == band_dim:
        arr = np.tile(arr, bands)                   # single-layer direction -> broadcast to all bands
    elif not (arr.ndim == 1 and arr.shape[0] == feat_dim):
        raise ValueError(f"direction shape {arr.shape} not in {{({bands},{band_dim}),({feat_dim},),({band_dim},)}}")
    return torch.from_numpy(arr)


class Krea2ConceptInject:
    """Targeted concept amplify/inject/project-out on Krea2 conditioning (the bypass, generalized + aimed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "direction_path": ("STRING", {"default": "", "tooltip": ".npy/.npz concept direction "
                                   "(12x2560 / 30720 / 2560), e.g. a difference-of-means axis."}),
                "mode": (["amplify", "add", "subtract", "project_out"], {"default": "amplify"}),
                "scale": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1,
                          "tooltip": "amplify: 1.0 == bypass x2 on the present component. +/- to push either way."}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "unit-normalize the direction first "
                              "(scale in unit terms). Ignored by amplify/project_out (always use d̂)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Targeted version of the bypass: amplify/inject/remove a single measured concept direction "
                   "on Krea2 conditioning. 'amplify' boosts the present component (can't conjure absent ones).")

    def apply(self, conditioning, direction_path, mode, scale, normalize=True):
        import numpy as np
        import torch

        out, d_cpu = [], None
        for cond, meta in conditioning:
            if d_cpu is None:  # feat dim is constant across one CONDITIONING -> load the direction once
                d_cpu = _load_direction(direction_path, cond.shape[-1], 12, torch, np)
            d = d_cpu.to(device=cond.device, dtype=cond.dtype)
            out.append([apply_direction(cond, d, float(scale), mode, bool(normalize)), meta.copy()])
        return (out,)


NODE_CLASS_MAPPINGS = {"Krea2ConceptInject": Krea2ConceptInject}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2ConceptInject": "Krea 2 Concept Inject (targeted bypass)"}

__all__ = ["Krea2ConceptInject", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "apply_direction", "_load_direction"]
