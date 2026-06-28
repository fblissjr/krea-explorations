"""Pairwise perceptual-diversity aggregation — one tested implementation.

Like ``image_grid``, this exists so experiment harnesses stop re-inlining the pairwise loop. The
distance is injected (``dist_fn(a, b) -> float``) so the module stays stdlib-only and imports in the
training venv too; callers supply a DreamSim/LPIPS-backed ``dist_fn`` (typically a cosine distance
over precomputed perceptual embeddings).

Higher mean pairwise distance = more diverse. Metric and protocol follow Gandikota & Bau,
"Distilling Diversity and Control in Diffusion Models" (arXiv:2503.10637): average pairwise DreamSim
distance over many same-prompt seeds.
"""
from __future__ import annotations

from itertools import combinations


def pairwise_distances(items, dist_fn):
    """``[dist_fn(items[i], items[j])]`` for every ``i < j``, upper-triangle row-major order.

    ``dist_fn`` is assumed symmetric; only the upper triangle is evaluated. Needs >= 2 items.
    """
    items = list(items)
    if len(items) < 2:
        raise ValueError(f"need at least 2 items to form a pair, got {len(items)}")
    return [float(dist_fn(a, b)) for a, b in combinations(items, 2)]


def pairwise_diversity(items, dist_fn):
    """Mean pairwise distance over all unordered pairs. Higher = more diverse."""
    dists = pairwise_distances(items, dist_fn)
    return sum(dists) / len(dists)


def diversity_table(arms, dist_fn):
    """Map ``{arm_name: items}`` to ``{arm_name: mean pairwise diversity}`` (each arm needs >= 2)."""
    return {name: pairwise_diversity(items, dist_fn) for name, items in arms.items()}
