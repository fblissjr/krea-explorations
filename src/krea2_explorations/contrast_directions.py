"""Conditioning-space contrast-direction analysis (pure numpy, no torch/comfy).

A *contrast direction* is the difference-of-means between two matched groups of conditioning
vectors -- e.g. prompts identical except for one attribute value. Given such directions these
helpers answer:

- Is a direction *real* (consistent across the individual pairs) or just averaged noise?
  ``pair_deltas`` + ``direction_consistency``.
- *Where* does it live across Krea2's 12 selected-layer bands? ``band_energy`` (raw) and
  ``relative_band_energy`` (magnitude-normalized -- undoes the ~48x hidden-state-norm growth).
- Is one direction *separable* from another (cosine ~0) or *coupled* (cosine ~+-1)? ``separability``;
  use ``band_normalize`` first to measure in a norm-equalized space (else L35's ~66% norm dominates).
- Remove a direction from a set of vectors (the ablation primitive). ``project_out``.

Conventions: a vector is 1D; a *group* is 2D ``[n_items, dim]``. Matched-pair helpers expect the two
groups row-aligned (item ``i`` of A pairs with item ``i`` of B).
"""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors; 0.0 if either is the zero vector."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def difference_of_means(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """``mean(A) - mean(B)`` over the item axis -- the contrast direction."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    return a.mean(axis=0) - b.mean(axis=0)


def pooled_direction(positives: np.ndarray, negatives: np.ndarray, n_bands: int = 12) -> np.ndarray:
    """A concept direction shaped for Krea 2's 12-layer conditioning axis (for ``Krea2ConceptInject``).

    ``mean(positives) - mean(negatives)`` (a difference-of-means contrast), reshaped to
    ``(n_bands, band_dim)``. Each group is a sequence of pooled conditioning vectors of length
    ``n_bands*band_dim`` -- one per prompt (produce them with ``scripts/krea2_clip.pooled_conditioning``).
    """
    return difference_of_means(positives, negatives).reshape(n_bands, -1)


def pair_deltas(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Per-pair ``A_i - B_i`` for row-aligned matched pairs -> ``[n_items, dim]``."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"matched groups must have equal shape, got {a.shape} vs {b.shape}")
    return a - b


def direction_consistency(deltas: np.ndarray) -> float:
    """Mean pairwise cosine between individual delta vectors.

    ~1 => every pair moves the same way (a real shared direction); ~0 => no shared direction, so a
    difference-of-means over these pairs would be averaging noise.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.shape[0]
    if n < 2:
        raise ValueError("need at least 2 deltas to measure consistency")
    norms = np.linalg.norm(deltas, axis=1)
    cosines: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] == 0.0 or norms[j] == 0.0:
                continue
            cosines.append(float(deltas[i] @ deltas[j] / (norms[i] * norms[j])))
    return float(np.mean(cosines)) if cosines else 0.0


def band_energy(direction: np.ndarray, n_bands: int = 12, band_dim: int = 2560) -> np.ndarray:
    """Fraction of squared-L2 energy of ``direction`` in each of ``n_bands`` contiguous blocks.

    Krea2 conditioning concatenates 12 selected-layer hidden states (each ``band_dim`` wide), so this
    localizes a contrast direction across the 12 layers. Returns ``[n_bands]`` summing to 1.
    """
    direction = np.asarray(direction, dtype=np.float64).ravel()
    expected = n_bands * band_dim
    if direction.shape[0] != expected:
        raise ValueError(f"direction dim {direction.shape[0]} != n_bands*band_dim {expected}")
    energy = (direction.reshape(n_bands, band_dim) ** 2).sum(axis=1)
    total = energy.sum()
    if total == 0.0:
        return np.zeros(n_bands)
    return energy / total


def relative_band_energy(
    direction: np.ndarray,
    baseline: np.ndarray,
    n_bands: int = 12,
    band_dim: int = 2560,
    normalize: bool = True,
) -> np.ndarray:
    """Per-band perturbation of ``direction`` relative to the per-band magnitude of ``baseline``.

    Raw ``band_energy`` is dominated by Krea2's ~48x hidden-state norm growth across layers -- deep
    bands swamp shallow ones regardless of where a contrast actually acts. This divides each band's
    direction norm by that band's baseline norm, answering "how much did this band move relative to
    its resting magnitude". Pass a robust ``baseline`` (e.g. the mean conditioning over many prompts).
    Bands with zero baseline norm are guarded to 0. ``normalize=True`` -> profile summing to 1.
    """
    direction = np.asarray(direction, dtype=np.float64).ravel()
    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    expected = n_bands * band_dim
    if direction.shape[0] != expected or baseline.shape[0] != expected:
        raise ValueError(f"dims must equal n_bands*band_dim ({expected}); "
                         f"got direction {direction.shape[0]}, baseline {baseline.shape[0]}")
    d = np.linalg.norm(direction.reshape(n_bands, band_dim), axis=1)
    b = np.linalg.norm(baseline.reshape(n_bands, band_dim), axis=1)
    ratio = np.divide(d, b, out=np.zeros_like(d), where=b > 0)
    if normalize:
        total = ratio.sum()
        return ratio / total if total > 0 else ratio
    return ratio


def band_normalize(
    vector: np.ndarray,
    baseline: np.ndarray,
    n_bands: int = 12,
    band_dim: int = 2560,
) -> np.ndarray:
    """Rescale each band of ``vector`` by 1/||baseline_band|| so all bands share a magnitude scale.

    Apply before ``cosine`` / ``direction_consistency`` / ``separability`` to measure them in a
    norm-equalized space. Otherwise L35's ~66% norm share dominates and every cosine is effectively a
    deep-band cosine. Bands with zero baseline norm are zeroed.
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    expected = n_bands * band_dim
    if vector.shape[0] != expected or baseline.shape[0] != expected:
        raise ValueError(f"dims must equal n_bands*band_dim ({expected}); "
                         f"got vector {vector.shape[0]}, baseline {baseline.shape[0]}")
    v = vector.reshape(n_bands, band_dim)
    b = np.linalg.norm(baseline.reshape(n_bands, band_dim), axis=1)
    scale = np.divide(1.0, b, out=np.zeros_like(b), where=b > 0)
    return (v * scale[:, None]).reshape(-1)


def separability(d1: np.ndarray, d2: np.ndarray) -> float:
    """Cosine between two contrast directions. ~0 => separable (orthogonal); ~+-1 => coupled."""
    return cosine(d1, d2)


def project_out(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component along ``direction`` from each row of ``vectors`` (the ablation primitive)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).ravel()
    nrm = np.linalg.norm(direction)
    if nrm == 0.0:
        return vectors.copy()
    unit = direction / nrm
    coeffs = vectors @ unit
    return vectors - np.outer(coeffs, unit)
