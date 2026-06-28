"""ComfyUI node: targeted concept amplify / inject / project-out on Krea2 conditioning.

A *targeted* deep-band magnitude lever. The blunt version boosts the whole deep band (L26/29/32) at the
projector; this operates on the conditioning that *feeds* the projector and aims at a
single measured concept direction d (a 12x2560 difference-of-means axis, flattened to 30720 to match Krea2's
``(B, seq, 12*2560)`` conditioning):

  amplify     cond += scale * (cond . d̂) d̂   -- boost the component ALREADY present (a magnitude move, aimed).
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
            return cond + scale * proj * dh    # scale 1 doubles the present component (x2)
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


def _pool_cond_vec(cond):
    """Mean a conditioning tensor over all but the last (feature) axis -> (feat,). numpy/torch-agnostic."""
    return cond.reshape(-1, cond.shape[-1]).mean(0)


def _concept_direction(positive, negative):
    """Difference-of-means concept direction (feat,) from two ComfyUI CONDITIONINGs. numpy/torch-agnostic.

    Each CONDITIONING is a list of ``[cond_tensor, meta]``; pool every entry over tokens+batch and average,
    then subtract negative from positive. Equivalent to ``scripts/concept_direction.py`` for the usual
    single-batch case (the CLI pools per-prompt over tokens; this pools over tokens and batch together).
    Callers must pass float tensors (``build`` casts first) so the difference doesn't lose small components to
    fp16/bf16 cancellation.
    """
    def _pool(conditioning):
        vecs = [_pool_cond_vec(cond) for cond, _meta in conditioning]
        acc = vecs[0]
        for v in vecs[1:]:
            acc = acc + v
        return acc / len(vecs)

    return _pool(positive) - _pool(negative)


class Krea2ConceptDirection:
    """Build a concept direction in-graph from positive/negative prompts (feeds Krea 2 Concept Inject)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "prompt(s) WITH the concept"}),
                "negative": ("CONDITIONING", {"tooltip": "matched prompt(s) WITHOUT it"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "", "tooltip": "optional .npy to also save the direction "
                              "(12x2560) for reuse / building a library."}),
            },
        }

    RETURN_TYPES = ("KREA2_CONCEPT_DIR",)
    RETURN_NAMES = ("direction",)
    FUNCTION = "build"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Difference-of-means concept direction from positive/negative prompts -> feed it to "
                   "Krea 2 Concept Inject. No model load (operates on the conditioning you already encoded).")

    def build(self, positive, negative, save_path=""):
        # cast to float BEFORE pooling: difference-of-means of fp16/bf16 conditioning loses small components
        pos = [[cond.float(), meta] for cond, meta in positive]
        neg = [[cond.float(), meta] for cond, meta in negative]
        d = _concept_direction(pos, neg).detach().to(device="cpu").float()  # flat (feat,)
        if save_path:
            import numpy as np
            arr = d.numpy()
            if arr.shape[0] % 12 == 0:
                arr = arr.reshape(12, -1)  # the (12, 2560) layout scripts/concept_direction.py writes
            np.save(save_path, arr)
        return (d,)


class Krea2ConceptInject:
    """Targeted concept amplify/inject/project-out on Krea2 conditioning (a deep-band magnitude lever, aimed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mode": (["amplify", "add", "subtract", "project_out"], {"default": "amplify"}),
                "scale": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1,
                          "tooltip": "amplify: 1.0 doubles the present component (x2). +/- to push either way."}),
            },
            "optional": {
                "direction": ("KREA2_CONCEPT_DIR", {"tooltip": "direction from Krea 2 Concept Direction "
                              "(takes priority over direction_path)."}),
                "direction_path": ("STRING", {"default": "", "tooltip": ".npy/.npz concept direction "
                                   "(12x2560 / 30720 / 2560); used only if no 'direction' is connected."}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "unit-normalize the direction first "
                              "(scale in unit terms). Ignored by amplify/project_out (always use d̂)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("A deep-band magnitude lever, aimed: amplify/inject/remove a single measured concept direction "
                   "on Krea2 conditioning. 'amplify' boosts the present component (can't conjure absent ones).")

    def apply(self, conditioning, mode, scale, direction=None, direction_path="", normalize=True):
        import numpy as np
        import torch

        out, d_cpu = [], None
        for cond, meta in conditioning:
            if d_cpu is None:  # feat dim is constant across one CONDITIONING -> resolve the direction once
                if direction is not None:  # in-graph direction (from Krea2ConceptDirection) wins over the path
                    d_cpu = direction.reshape(-1).detach().to(device="cpu").float()
                    if d_cpu.shape[0] != cond.shape[-1]:
                        raise ValueError(f"Krea2ConceptInject: direction length {d_cpu.shape[0]} != "
                                         f"conditioning feature dim {cond.shape[-1]}")
                else:
                    d_cpu = _load_direction(direction_path, cond.shape[-1], 12, torch, np)
            # I1 (code review): compute the 30720-dim norm + projection in float32, then cast back.
            # In bf16/fp16 the ||d|| and per-token dot drift, mis-scaling the edit (and breaking scale-1 == x2).
            d = d_cpu.to(device=cond.device)  # d_cpu is float32
            edited = apply_direction(cond.float(), d, float(scale), mode, bool(normalize))
            out.append([edited.to(dtype=cond.dtype), meta.copy()])
        return (out,)


NODE_CLASS_MAPPINGS = {"Krea2ConceptInject": Krea2ConceptInject, "Krea2ConceptDirection": Krea2ConceptDirection}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2ConceptInject": "Krea 2 Concept Inject",
                             "Krea2ConceptDirection": "Krea 2 Concept Direction"}

__all__ = ["Krea2ConceptInject", "Krea2ConceptDirection", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "apply_direction", "_load_direction", "_concept_direction", "_pool_cond_vec"]
