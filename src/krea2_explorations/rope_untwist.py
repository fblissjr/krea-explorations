"""Untwisting-RoPE frequency control, parameterized for Krea2's REAL [32,48,48] RoPE axes.

Implements the paper's per-band polynomial schedule (arXiv 2602.05013):

    s_d = s_hf + (s_lf - s_hf) * d_tilde^beta

where d_tilde is the per-axis normalized band index (0 = highest freq -> 1 = lowest freq), beta the
curve steepness, s_hf/s_lf the high/low endpoints. High-freq bands carry spatial/positional alignment
(reference *copying*); low-freq bands carry semantic/style correspondence. Untwisting = scale the
reference keys per band: attenuate high-freq, amplify low-freq, so shared attention transfers *style*
without copying layout.

This is the part the community Krea2 fork got wrong: it feeds the builder a flat
``[head_dim/2, head_dim/2] = [64,64]`` split (a 2-axis Qwen/Flux assumption), whose sum still equals
head_dim so the guard never fires -- silently banding the wrong partition. Krea2's actual RoPE is the
3-axis ``[32,48,48]`` from comfy/ldm/krea2/model.py. We build the schedule against that.

Pure numpy (no torch) so the band math is unit-tested in the project venv; the ComfyUI node wraps the
returned vector as a torch tensor at use time.
"""
from __future__ import annotations

import numpy as np


def krea2_rope_axes(head_dim: int = 128) -> list[int]:
    """Krea2's real 3-axis RoPE split (comfy/ldm/krea2/model.py):
        axes = [headdim - 12*(headdim//16), 6*(headdim//16), 6*(headdim//16)]
    head_dim=128 -> [32, 48, 48]. Falls back to a single axis if the formula doesn't sum (non-128 heads).
    """
    q = head_dim // 16
    axes = [head_dim - 12 * q, 6 * q, 6 * q]
    return axes if sum(axes) == head_dim else [head_dim]


def krea2_freq_scale_vector(
    head_dim: int = 128,
    *,
    axes: list[int] | None = None,
    high_scale: float = 1.05,
    low_scale: float = 3.0,
    beta: float = 50.0,
    axis0_mode: str = "flat_low",
    axis0_scale: float = 0.0,
) -> np.ndarray:
    """Per-channel scale vector (length ``head_dim``) to multiply onto the reference keys.

    ``axes`` defaults to Krea2's real split. ``axis0_mode`` controls the first ("temporal") axis, whose
    position is constant for a still image: ``"flat_low"`` (default, matches the reference -- flat at
    ``low_scale``), ``"curve"`` (same polynomial as the spatial axes), or ``"constant"`` (``axis0_scale``).
    """
    axes = list(axes) if axes else krea2_rope_axes(head_dim)
    if sum(axes) != head_dim:
        axes = [head_dim]
    has_sep0 = len(axes) >= 2

    def curve(n_pairs: int) -> np.ndarray:
        if n_pairs <= 1:
            d = np.zeros(max(n_pairs, 1), dtype=np.float64)
        else:
            d = np.linspace(0.0, 1.0, n_pairs)
        return high_scale + (low_scale - high_scale) * (d ** float(beta))

    pieces: list[np.ndarray] = []
    for i, a in enumerate(axes):
        n_pairs = a // 2
        if n_pairs <= 0:
            pieces.append(np.ones(a))
            continue
        if has_sep0 and i == 0:
            if axis0_mode == "curve":
                ps = curve(n_pairs)
            elif axis0_mode == "constant":
                ps = np.full(n_pairs, float(axis0_scale))
            else:  # "flat_low" -- reference default
                ps = np.full(n_pairs, float(low_scale))
        else:
            ps = curve(n_pairs)
        piece = np.repeat(ps, 2)  # both dims of each rotated pair share a scale
        if a % 2:
            piece = np.concatenate([piece, np.ones(1)])
        pieces.append(piece)

    out = np.concatenate(pieces).astype(np.float32)
    if out.size >= head_dim:
        return out[:head_dim]
    return np.pad(out, (0, head_dim - out.size), constant_values=1.0)
