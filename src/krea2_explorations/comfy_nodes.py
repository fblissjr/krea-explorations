"""ComfyUI node: live reweighting of Krea2's learned txtfusion.projector.

This is the interactive version of everything else in the repo. It patches the model's
``diffusion_model.txtfusion.projector.weight`` ([1,12] per-layer combiner / "multilayer feature
aggregation") via the ModelPatcher, so you get per-layer control with a strength slider and no checkpoint
rewrite / no LoRA file. ``effective = orig*(1 + strength*(gain-1))`` matches the ``.diff`` LoRA exactly:
strength 1 = the chosen gains, 0 = no change, higher pushes further.

``comfy`` is imported lazily inside ``apply`` so this module stays importable (and testable) without the
ComfyUI runtime.
"""

from __future__ import annotations

from .projector import N_BANDS, PROJECTOR_KEY, SELECT_LAYERS
from .projector_lora import PRESETS, effective_weights, resolve_gains

# attribute path relative to ModelPatcher.model (BaseModel) -> the DiT's projector weight
OBJECT_KEY = "diffusion_model." + PROJECTOR_KEY  # diffusion_model.txtfusion.projector.weight
PRESET_CHOICES = sorted(PRESETS) + ["custom"]


class Krea2ProjectorRebalance:
    """Reweight Krea2's 12 Qwen3-VL projector layers live, as a model patch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (PRESET_CHOICES, {
                    "default": "uniform",
                    "tooltip": "'uniform' = no change; 'custom' uses per_layer_weights. Use solo_band to "
                               "isolate a single layer.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1,
                    "tooltip": "1.0 = chosen gains exactly; 0 = no change; higher pushes further. "
                               "Magnitude is held by the model's RMSNorm.",
                }),
            },
            "optional": {
                "per_layer_weights": ("STRING", {
                    "default": "1,1,1,1,1,1,1,1,1,1,1,1", "multiline": False,
                    "tooltip": "12 comma-separated gains (shallow->deep layer). Used when preset='custom'.",
                }),
                "solo_band": ("INT", {
                    "default": -1, "min": -1, "max": N_BANDS - 1,
                    "tooltip": "-1 = off. 0..11 isolates one selected Qwen3-VL layer (zeros the rest) for the "
                               "hidden-state explorer; use strength 1.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Reweight Krea2's learned txtfusion.projector (per-layer combiner) live via a "
                   "model patch — no checkpoint rewrite, no LoRA file.")

    def apply(self, model, preset, strength, per_layer_weights="", solo_band=-1):
        import uuid

        import torch

        gains = resolve_gains(preset, per_layer_weights, solo_band)

        m = model.clone()
        try:
            cur = m.get_model_object(OBJECT_KEY)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message in the UI
            raise RuntimeError(
                f"Could not read {OBJECT_KEY!r}. Is this a Krea2 model loaded via UNETLoader? ({exc})"
            ) from exc

        orig = cur.detach().float().cpu().numpy().reshape(-1)
        if orig.shape[0] != N_BANDS:
            raise RuntimeError(
                f"projector has {orig.shape[0]} bands, expected {N_BANDS} — not a Krea2 model?"
            )

        new = effective_weights(orig, gains, strength)
        new_t = torch.tensor(new, dtype=cur.dtype, device=cur.device).reshape(cur.shape)
        # Parameter (not a bare tensor) so module.to(device) moves it with the rest of the model.
        m.add_object_patch(OBJECT_KEY, torch.nn.Parameter(new_t, requires_grad=False))
        m.patches_uuid = uuid.uuid4()
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2ProjectorRebalance": Krea2ProjectorRebalance}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2ProjectorRebalance": "Krea 2 Projector Rebalance"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "Krea2ProjectorRebalance", "SELECT_LAYERS"]
