"""krea2-explorations: tools for understanding and surgically editing Krea2 conditioning.

See docs/findings.md for the background and findings.
"""

from . import attention_stats, projector, projector_lora, safetensors_patch
from .projector import PROJECTOR_KEY, SELECT_LAYERS, read_projector, scale_projector
from .projector_lora import (
    LORA_KEY,
    PRESETS,
    make_band_isolation_loras,
    make_preset_lora,
    make_projector_lora,
)

__all__ = [
    "safetensors_patch",
    "projector",
    "projector_lora",
    "attention_stats",
    "read_projector",
    "scale_projector",
    "make_projector_lora",
    "make_preset_lora",
    "make_band_isolation_loras",
    "PROJECTOR_KEY",
    "SELECT_LAYERS",
    "LORA_KEY",
    "PRESETS",
]
