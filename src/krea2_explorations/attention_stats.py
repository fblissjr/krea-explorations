"""Pure-numpy summarization of Krea2 layer-fusion attention (no torch / comfy needed).

Input attention tensors have shape ``[tokens, heads, n, n]`` where ``n`` = number of selected layers
(12 for Krea2) and the last two axes are (query layer -> key layer). These helpers average and rank to
answer "which selected layer is the attention hub" (the layer everyone attends to).
"""

from __future__ import annotations

import numpy as np


def head_token_average(attn: np.ndarray) -> np.ndarray:
    """``[tokens, heads, n, n]`` -> ``[n, n]`` averaged over tokens and heads."""
    attn = np.asarray(attn, dtype=np.float64)
    return attn.mean(axis=(0, 1))


def per_head_average(attn: np.ndarray) -> np.ndarray:
    """``[tokens, heads, n, n]`` -> ``[heads, n, n]`` averaged over tokens only."""
    attn = np.asarray(attn, dtype=np.float64)
    return attn.mean(axis=0)


def hub_strength(mean_map: np.ndarray) -> np.ndarray:
    """Per-key-layer 'hub strength' = how much, on average, the other layers attend TO it.

    ``mean_map`` is ``[n, n]`` with rows = FROM (query layer), cols = TO (key layer). Returns ``[n]``
    (mean over the query axis). The argmax is the attention hub.
    """
    return np.asarray(mean_map, dtype=np.float64).mean(axis=0)


def hub_ranking(mean_map: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    """Return ``[(label, strength), ...]`` sorted by hub strength, descending."""
    strength = hub_strength(mean_map)
    if len(labels) != strength.shape[0]:
        raise ValueError(f"labels ({len(labels)}) must match map size ({strength.shape[0]})")
    return sorted(((labels[i], float(strength[i])) for i in range(len(labels))), key=lambda kv: -kv[1])
