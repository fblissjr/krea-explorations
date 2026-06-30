"""Random matched-norm LoRA — the zero-information perturbation control.

Given a real trained LoRA (a flat ``{name: 2D array}`` mapping, as read from a
safetensors adapter), emit a random LoRA with the **same keys and shapes** but a
**random direction** whose **magnitude is matched** to the original. For each
``(lora_A, lora_B)`` pair the matched quantity is the effective weight delta
``B @ A`` (Frobenius norm) — the actual perturbation applied to the base weight —
not the factor norms, so the model sees the same-sized nudge in a random
direction. Tensors with no A/B partner are matched on their own Frobenius norm.

Use: if a *trained* benign LoRA unlocks a behaviour but this control does not,
the behaviour depends on the learned direction; if the random control unlocks it
too, it is pure magnitude/perturbation. Pure numpy so it imports anywhere.
"""
from __future__ import annotations

import numpy as np

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


def _fro(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a.astype(np.float64) ** 2)))


def match_norm_random(tensors: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return a random LoRA matched in magnitude to ``tensors``.

    ``rng`` is a ``numpy.random.Generator`` (e.g. ``np.random.default_rng(seed)``);
    the result is deterministic for a given seed. Output preserves input key order
    and per-tensor dtype.
    """
    pairs: dict[str, dict[str, str]] = {}
    for k in tensors:
        if k.endswith(_A_SUFFIX):
            pairs.setdefault(k[: -len(_A_SUFFIX)], {})["A"] = k
        elif k.endswith(_B_SUFFIX):
            pairs.setdefault(k[: -len(_B_SUFFIX)], {})["B"] = k

    result: dict[str, np.ndarray] = {}
    for pair in pairs.values():
        if "A" not in pair or "B" not in pair:
            continue
        ak, bk = pair["A"], pair["B"]
        a, b = tensors[ak], tensors[bk]
        target = _fro(b.astype(np.float64) @ a.astype(np.float64))
        ra = rng.standard_normal(a.shape)
        rb = rng.standard_normal(b.shape)
        delta = _fro(rb @ ra)
        scale = (target / delta) if delta > 0 else 0.0
        result[ak] = ra.astype(a.dtype)
        result[bk] = (rb * scale).astype(b.dtype)

    for k, v in tensors.items():
        if k in result:
            continue
        r = rng.standard_normal(v.shape)
        n = _fro(r)
        result[k] = (r * (_fro(v) / n)).astype(v.dtype) if n > 0 else r.astype(v.dtype)

    return {k: result[k] for k in tensors}  # preserve source key order
