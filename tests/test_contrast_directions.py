"""Tests for conditioning-space contrast-direction analysis (pure numpy, no torch/comfy)."""

import numpy as np

from krea2_explorations import contrast_directions as cd


def test_cosine_parallel_orthogonal_antiparallel():
    a = np.array([1.0, 0.0, 0.0])
    assert cd.cosine(a, np.array([2.0, 0.0, 0.0])) == 1.0
    assert abs(cd.cosine(a, np.array([0.0, 1.0, 0.0]))) < 1e-12
    assert cd.cosine(a, np.array([-1.0, 0.0, 0.0])) == -1.0


def test_cosine_zero_vector_is_zero():
    assert cd.cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_difference_of_means():
    a = np.array([[1.0, 0.0], [3.0, 0.0]])  # mean [2,0]
    b = np.array([[0.0, 0.0], [0.0, 0.0]])  # mean [0,0]
    np.testing.assert_allclose(cd.difference_of_means(a, b), [2.0, 0.0])


def test_pair_deltas_row_aligned():
    a = np.array([[2.0, 1.0], [4.0, 1.0]])
    b = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(cd.pair_deltas(a, b), [[1.0, 0.0], [3.0, 0.0]])


def test_pair_deltas_shape_mismatch_raises():
    try:
        cd.pair_deltas(np.zeros((2, 3)), np.zeros((3, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched group shapes")


def test_direction_consistency_aligned_is_one():
    deltas = np.array([[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
    assert cd.direction_consistency(deltas) > 0.999


def test_direction_consistency_orthogonal_is_zero():
    deltas = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(cd.direction_consistency(deltas)) < 1e-9


def test_band_energy_localizes_to_one_band():
    direction = np.zeros(6)
    direction[3:] = [1.0, 1.0, 1.0]  # all energy in band 1
    energy = cd.band_energy(direction, n_bands=2, band_dim=3)
    assert energy.shape == (2,)
    np.testing.assert_allclose(energy, [0.0, 1.0])


def test_band_energy_sums_to_one():
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(12 * 2560)
    energy = cd.band_energy(direction, n_bands=12, band_dim=2560)
    assert abs(energy.sum() - 1.0) < 1e-9


def test_band_energy_wrong_dim_raises():
    try:
        cd.band_energy(np.zeros(7), n_bands=2, band_dim=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when dim != n_bands*band_dim")


def test_separability_orthogonal_directions():
    assert abs(cd.separability(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-12


def test_project_out_removes_component():
    vecs = np.array([[3.0, 4.0], [1.0, 0.0]])
    direction = np.array([1.0, 0.0])
    out = cd.project_out(vecs, direction)
    np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[:, 1], [4.0, 0.0])


def test_relative_band_energy_raw_ratio():
    direction = np.array([3.0, 4.0, 1.0, 0.0])  # band0 norm 5, band1 norm 1
    baseline = np.array([1.0, 0.0, 1.0, 0.0])   # band0 norm 1, band1 norm 1
    rel = cd.relative_band_energy(direction, baseline, n_bands=2, band_dim=2, normalize=False)
    np.testing.assert_allclose(rel, [5.0, 1.0])


def test_relative_band_energy_upweights_quiet_bands():
    # equal movement in both bands, but band1's baseline is tiny -> band1 dominates relatively
    direction = np.array([1.0, 1.0, 1.0, 1.0])
    baseline = np.array([10.0, 10.0, 0.1, 0.1])
    rel = cd.relative_band_energy(direction, baseline, n_bands=2, band_dim=2)
    assert rel[1] > rel[0]


def test_relative_band_energy_normalized_sums_to_one():
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(12 * 2560)
    baseline = np.abs(rng.standard_normal(12 * 2560)) + 0.1
    rel = cd.relative_band_energy(direction, baseline)
    assert abs(rel.sum() - 1.0) < 1e-9


def test_relative_band_energy_zero_baseline_band_is_guarded():
    direction = np.array([1.0, 1.0, 1.0, 1.0])
    baseline = np.array([0.0, 0.0, 2.0, 2.0])  # band0 baseline zero -> no div-by-zero
    rel = cd.relative_band_energy(direction, baseline, n_bands=2, band_dim=2, normalize=False)
    assert rel[0] == 0.0
    assert rel[1] > 0.0


def test_band_normalize_equalizes_band_scale():
    vec = np.array([3.0, 4.0, 3.0, 4.0])   # both bands raw norm 5
    base = np.array([6.0, 8.0, 0.6, 0.8])  # band0 baseline norm 10, band1 norm 1
    out = cd.band_normalize(vec, base, n_bands=2, band_dim=2)
    np.testing.assert_allclose(out, [0.3, 0.4, 3.0, 4.0])  # band0 /10, band1 /1


def test_band_normalize_zero_baseline_band_guarded():
    vec = np.array([1.0, 1.0, 1.0, 1.0])
    base = np.array([0.0, 0.0, 2.0, 2.0])
    out = cd.band_normalize(vec, base, n_bands=2, band_dim=2)
    np.testing.assert_allclose(out[:2], 0.0)


def test_band_normalize_then_cosine_is_norm_equalized():
    # in raw space band0 dominates; after equalizing, the orthogonal band1 disagreement shows up
    base = np.array([10.0, 0.0, 1.0, 0.0])  # band0 norm 10, band1 norm 1
    a = np.array([10.0, 0.0, 1.0, 0.0])     # +band1
    b = np.array([10.0, 0.0, -1.0, 0.0])    # -band1
    raw = cd.cosine(a, b)
    eq = cd.cosine(cd.band_normalize(a, base, 2, 2), cd.band_normalize(b, base, 2, 2))
    assert raw > 0.95          # raw: dominated by the agreeing band0
    assert eq < 0.05           # equalized: band1 disagreement surfaces (~0)
