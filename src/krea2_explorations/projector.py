"""Read and reweight Krea2's learned text-layer aggregation (``txtfusion.projector``).

Krea2's text path selects 12 Qwen3-VL-4B encoder layers and the DiT combines them with a learned
``Linear(12 -> 1)`` named ``txtfusion.projector`` (weight shape ``[1, 12]``) -- Krea calls this
"multilayer feature aggregation". These helpers read those 12 learned weights and write a reweighted
copy of a checkpoint (scaling weight column ``i`` by gain ``g_i``).

Editing the weight directly keeps the conditioning in distribution -- the model's own downstream RMSNorm
holds magnitude, so only the layer-mix *direction* changes. It is also trivially reversible (restore 12
numbers) and interpretable.
"""

from __future__ import annotations

import numpy as np

from . import safetensors_patch as sp

PROJECTOR_KEY = "txtfusion.projector.weight"
N_BANDS = 12

# Which Qwen3-VL-4B hidden layer each band corresponds to (see comfy/text_encoders/krea2.py KREA2_TAP_LAYERS).
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def read_projector(path) -> np.ndarray:
    """Return the 12 learned projector weights as a 1-D float32 array, band-ordered."""
    w = sp.read_tensor(path, PROJECTOR_KEY)
    return np.asarray(w, dtype=np.float32).reshape(-1)


def scale_projector(src, dst, gains) -> np.ndarray:
    """Write a copy of ``src`` to ``dst`` with each projector band multiplied by ``gains[i]``.

    ``gains`` must have length 12. Returns the new weights as float32 (post bf16 round-trip, so the
    returned values match exactly what was written to ``dst``).
    """
    gains = np.asarray(gains, dtype=np.float32)
    if gains.shape != (N_BANDS,):
        raise ValueError(f"gains must have length {N_BANDS}, got shape {tuple(gains.shape)}")

    header, _ = sp.read_header(src)
    shape = header[PROJECTOR_KEY]["shape"]  # [1, 12]
    cur = read_projector(src)
    new = cur * gains

    sp.patch_tensor(src, dst, PROJECTOR_KEY, new.reshape(shape))
    return read_projector(dst)
