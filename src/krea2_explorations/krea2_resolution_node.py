"""ComfyUI node: Krea2 resolution picker + workable-resolution snapping.

Krea 2's image grid must be divisible by **16** = VAE spatial factor (8) x DiT patch (2) (the divisor in
ModelSamplingFlux's `w*h/(8*8*2*2)`). It was trained around a ~1 MP area (1024^2); off-area resolutions also
shift the flow `mu`, so the sensible "workable" set is the ~1 MP aspect buckets, all /16. (NB: the Qwen3-VL
model here is the *text* encoder — it does not constrain image size; only the VAE + patch do.)

This node gives the standard ~1 MP buckets, or snaps an arbitrary width/height to the nearest /16 (optionally
rescaling to ~1 MP first). Outputs width/height ints to feed EmptyLatentImage. Pure-logic snap (no torch).
"""

from __future__ import annotations

# ~1 MP aspect buckets, every dim divisible by 16, ordered square -> wide (each landscape paired with its
# portrait rotation). Labels are the *exact* reduced ratio of the dims -- a test enforces label == w/h, so a
# bucket like 1216x832 is 19:13 (1.46:1), NOT the 3:2 it was once mislabeled as (true 3:2 is 1248x832).
BUCKETS = {
    "1024x1024 (1:1)":  (1024, 1024),
    "1120x896 (5:4)":   (1120, 896),
    "896x1120 (4:5)":   (896, 1120),
    "1152x896 (9:7)":   (1152, 896),
    "896x1152 (7:9)":   (896, 1152),
    "1152x864 (4:3)":   (1152, 864),
    "864x1152 (3:4)":   (864, 1152),
    "1216x832 (19:13)": (1216, 832),
    "832x1216 (13:19)": (832, 1216),
    "1248x832 (3:2)":   (1248, 832),
    "832x1248 (2:3)":   (832, 1248),
    "1344x768 (7:4)":   (1344, 768),
    "768x1344 (4:7)":   (768, 1344),
    "1280x720 (16:9)":  (1280, 720),
    "720x1280 (9:16)":  (720, 1280),
    "1568x672 (21:9)":  (1568, 672),
    "672x1568 (9:21)":  (672, 1568),
    "1536x640 (12:5)":  (1536, 640),
    "640x1536 (5:12)":  (640, 1536),
    "custom (snap w/h)": None,
}


def snap_resolution(width, height, multiple=16, target_mp=None):
    """Snap (width, height) to the nearest multiple of `multiple`; if target_mp is set, first rescale to that
    megapixel area preserving aspect. Returns (w, h) ints, each >= `multiple`. Pure function (unit-testable)."""
    w, h = float(width), float(height)
    if target_mp:
        scale = (target_mp * 1_000_000.0 / max(w * h, 1.0)) ** 0.5
        w, h = w * scale, h * scale
    sw = max(multiple, int(round(w / multiple)) * multiple)
    sh = max(multiple, int(round(h / multiple)) * multiple)
    return sw, sh


class Krea2Resolution:
    """Pick a ~1 MP Krea2 bucket, or snap a custom width/height to the nearest workable (/16) resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(BUCKETS), {"default": "1024x1024 (1:1)"}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 1,
                                  "tooltip": "used when preset = 'custom (snap w/h)'"}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 1}),
                "snap_to_1mp": ("BOOLEAN", {"default": True,
                                "tooltip": "custom mode: rescale to ~1 MP area (Krea2's trained regime) before /16 snap"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "latent")
    FUNCTION = "resolve"
    CATEGORY = "conditioning/Krea2"
    DESCRIPTION = ("Krea2 resolution: pick a ~1 MP bucket, or snap a custom width/height to the nearest "
                   "workable /16 (VAE 8x x DiT patch 2x). Outputs width/height plus an empty latent, so it "
                   "drops in for EmptyLatentImage.")

    def resolve(self, preset, width=1024, height=1024, snap_to_1mp=True, batch_size=1):
        w, h = (BUCKETS[preset] if BUCKETS.get(preset) is not None
                else snap_resolution(width, height, 16, target_mp=1.0 if snap_to_1mp else None))
        # lazy imports so the module (and snap_resolution/BUCKETS) stays usable without torch/comfy
        import torch
        try:
            import comfy.model_management as mm
            device = mm.intermediate_device()
        except Exception:
            device = "cpu"
        latent = {"samples": torch.zeros([batch_size, 4, h // 8, w // 8], device=device)}
        return (w, h, latent)


NODE_CLASS_MAPPINGS = {"Krea2Resolution": Krea2Resolution}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2Resolution": "Krea 2 Resolution"}

__all__ = ["Krea2Resolution", "snap_resolution", "BUCKETS",
           "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
