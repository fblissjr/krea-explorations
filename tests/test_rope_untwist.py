"""Tests for the untwisting-RoPE frequency control, parameterized for Krea2's REAL [32,48,48] axes.

The community fork feeds the builder a flat [64,64] split; sum still == head_dim so its guard never
fires and it silently bands the wrong partition. These tests pin our correct behavior.
"""
import numpy as np

from krea2_explorations.rope_untwist import krea2_rope_axes, krea2_freq_scale_vector


def test_default_axes_are_kreas_real_split():
    assert krea2_rope_axes(128) == [32, 48, 48]
    assert sum(krea2_rope_axes(128)) == 128


def test_vector_length_matches_head_dim():
    assert krea2_freq_scale_vector(128).shape == (128,)


def test_no_untwist_when_high_equals_low():
    v = krea2_freq_scale_vector(128, high_scale=1.0, low_scale=1.0)
    assert np.allclose(v, 1.0)


def test_axis0_default_is_flat_low():
    # axes [32,48,48]: axis0 = first 32 dims, defaults to a flat low_scale band.
    v = krea2_freq_scale_vector(128, high_scale=1.05, low_scale=3.0)
    assert np.allclose(v[:32], 3.0)


def test_curve_axis_endpoints_and_monotonic():
    # last axis = dims 80..127 (48 dims, 24 pairs) on the polynomial curve.
    v = krea2_freq_scale_vector(128, high_scale=1.05, low_scale=3.0, beta=50.0)
    last = v[80:128]
    assert np.isclose(last[0], 1.05, atol=1e-4)   # highest freq pair -> high_scale
    assert np.isclose(last[-1], 3.0, atol=1e-4)    # lowest freq pair -> low_scale
    # per-pair scales are non-decreasing across the axis
    pairs = last[::2]
    assert np.all(np.diff(pairs) >= -1e-6)


def test_real_axes_differ_from_fork_64_64():
    real = krea2_freq_scale_vector(128, axes=[32, 48, 48])
    fork = krea2_freq_scale_vector(128, axes=[64, 64])
    assert not np.allclose(real, fork)
