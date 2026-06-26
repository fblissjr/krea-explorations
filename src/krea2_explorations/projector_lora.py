"""Emit tiny ComfyUI ``.diff`` LoRAs that reweight Krea2's learned text-layer aggregation.

Krea2 combines 12 selected Qwen3-VL encoder layers with a learned ``Linear(12 -> 1)`` projector
(``txtfusion.projector``; Krea calls this "multilayer feature aggregation"). These helpers emit a tiny
``diffusion_model.txtfusion.projector.diff`` patch that reweights those 12 layers. ComfyUI applies a
``.diff`` as ``weight = weight + strength * diff``, and each file is built as ``diff = orig*(gain-1)``,
so at strength 1 the effective projector is ``orig*gain`` and the loader's strength becomes a live knob.

Because it edits weights (not activations), the model's own downstream RMSNorm holds magnitude, so only
the layer-mix *direction* changes. Output loads via the stock ``LoraLoaderModelOnly`` (no custom node),
and each file is ~300 bytes, so per-layer exploration is 12 tiny files instead of 12 x 26 GB checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import safetensors_patch as sp
from .projector import N_BANDS, SELECT_LAYERS, read_projector

# ComfyUI key for an additive weight diff applied to the loaded diffusion model.
LORA_KEY = "diffusion_model.txtfusion.projector.diff"

# Named per-layer gain profiles (gain applied at LoRA strength 1). The real knobs for exploration are
# arbitrary `custom` gains and single-layer isolation (see make_band_isolation_loras); `uniform` is the
# identity baseline (no change).
PRESETS: dict[str, list[float]] = {
    "uniform": [1.0] * N_BANDS,
}


def effective_weights(orig, gains, strength: float = 1.0) -> np.ndarray:
    """Effective projector weights = ``orig * (1 + strength*(gain-1))`` per band.

    strength 1 -> ``orig*gain`` (the preset exactly); strength 0 -> ``orig`` (off); >1 pushes harder.
    This matches the ``.diff`` LoRA semantics so the file and the live node behave identically.
    """
    orig = np.asarray(orig, dtype=np.float32)
    gains = np.asarray(gains, dtype=np.float32)
    if gains.shape != (N_BANDS,):
        raise ValueError(f"gains must have length {N_BANDS}, got shape {tuple(gains.shape)}")
    return (orig * (1.0 + float(strength) * (gains - 1.0))).astype(np.float32)


def resolve_gains(preset: str = "balanced", per_layer_weights: str = "",
                  solo_band: int = -1, solo_gain: float = 1.0) -> list[float]:
    """Resolve UI-style inputs to a length-12 gain list. ``solo_band`` >= 0 overrides preset/custom."""
    if solo_band is not None and solo_band >= 0:
        if solo_band >= N_BANDS:
            raise ValueError(f"solo_band must be -1 or 0..{N_BANDS - 1}, got {solo_band}")
        g = [0.0] * N_BANDS
        g[solo_band] = float(solo_gain)
        return g
    if preset == "custom":
        vals = [float(x) for x in str(per_layer_weights).replace(";", ",").split(",") if x.strip() != ""]
        if len(vals) != N_BANDS:
            raise ValueError(f"custom per_layer_weights needs {N_BANDS} values, got {len(vals)}")
        return vals
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)} or 'custom'")
    return list(PRESETS[preset])


def _diff_from_gains(orig: np.ndarray, gains) -> np.ndarray:
    return (effective_weights(orig, gains, 1.0) - np.asarray(orig, dtype=np.float32)).astype(np.float32)


def make_projector_lora(src_ckpt, gains, out_path, metadata=None) -> np.ndarray:
    """Emit a projector ``.diff`` LoRA for arbitrary per-band ``gains`` (length 12).

    Returns the diff (float32, shape (12,)). At LoRA strength 1 the effective projector equals the
    source checkpoint's projector times ``gains``.
    """
    orig = read_projector(src_ckpt)
    diff = _diff_from_gains(orig, gains)
    md = {
        "krea2_explorations": "txtfusion.projector diff; effective = w*(1 + strength*(gain-1))",
        "gains": ",".join(f"{float(g):g}" for g in gains),
    }
    if metadata:
        md.update({str(k): str(v) for k, v in metadata.items()})
    sp.write_safetensors(out_path, {LORA_KEY: diff.reshape(1, N_BANDS)}, metadata=md)
    return diff


def make_preset_lora(src_ckpt, preset, out_path) -> np.ndarray:
    """Emit a LoRA for a named preset in ``PRESETS``."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return make_projector_lora(src_ckpt, PRESETS[preset], out_path, metadata={"preset": preset})


def make_band_isolation_loras(src_ckpt, out_dir, solo_gain: float = 1.0) -> list[Path]:
    """Emit 12 LoRAs, each isolating one selected Qwen3-VL layer (zero the other 11) at strength 1.

    With ``solo_gain`` you can also boost the kept band. Meant to be loaded at strength 1 to *see*
    what each individual hidden-state layer contributes to the image (the hidden-state explorer).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(N_BANDS):
        gains = [0.0] * N_BANDS
        gains[i] = float(solo_gain)
        p = out_dir / f"projector_solo_b{i:02d}_L{SELECT_LAYERS[i]:02d}.safetensors"
        make_projector_lora(src_ckpt, gains, p,
                            metadata={"isolate_band": i, "qwen_layer": SELECT_LAYERS[i], "solo_gain": solo_gain})
        paths.append(p)
    return paths
