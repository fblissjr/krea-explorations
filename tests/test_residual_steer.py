"""Logic test for the EXP5 Phase-D2 residual-steer helper (numpy-only)."""

import numpy as np

from krea2_explorations.krea2_dit_residual_steer_node import steer_image_residual


def test_steer_adds_proportional_to_token_norm():
    img = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # token0 norm 5, token1 norm 0
    dir_hat = np.array([0.0, 0.0, 1.0])  # unit
    out = steer_image_residual(img, dir_hat, scale=2.0)
    # token0: += 2 * 5 * [0,0,1] = [0,0,10]; token1: += 0 (zero norm)
    assert np.allclose(out[0], [3.0, 4.0, 10.0])
    assert np.allclose(out[1], [0.0, 0.0, 0.0])


def test_steer_scale_zero_is_identity():
    img = np.random.RandomState(0).randn(4, 5)
    out = steer_image_residual(img, np.ones(5) / np.sqrt(5), scale=0.0)
    assert np.allclose(out, img)
