"""Pure-logic tests for the public concept-direction math (no comfy/torch).

`pooled_direction` turns groups of pooled conditioning vectors into a (n_bands, band_dim) concept direction
for the Krea2ConceptInject node. The comfy encode that produces the pooled vectors lives in
`scripts/krea2_clip.py`; this tests only the math, in the project venv.
"""

import numpy as np

from krea2_explorations.contrast_directions import pooled_direction


def test_pooled_direction_is_diff_of_means_reshaped():
    pos = [np.full(6, 4.0), np.full(6, 2.0)]  # mean 3
    neg = [np.full(6, 1.0), np.full(6, 1.0)]  # mean 1
    d = pooled_direction(pos, neg, n_bands=3)
    assert d.shape == (3, 2)  # 6 features = 3 bands x band_dim 2
    assert np.allclose(d, 2.0)  # (3 - 1) everywhere


def test_pooled_direction_single_pair_preserves_layer_major_layout():
    d = pooled_direction([np.array([1.0, 2.0, 3.0, 4.0])], [np.zeros(4)], n_bands=2)
    assert d.shape == (2, 2)
    assert np.allclose(d.reshape(-1), [1, 2, 3, 4])  # row-major reshape == encoder's layer-major flatten


def test_pooled_direction_default_bands_12():
    # 12*2560 feature vector -> (12, 2560), the Krea2 conditioning layout
    pos = [np.ones(12 * 2560)]
    neg = [np.zeros(12 * 2560)]
    d = pooled_direction(pos, neg)
    assert d.shape == (12, 2560)
