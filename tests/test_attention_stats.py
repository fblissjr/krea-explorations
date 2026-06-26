"""Tests for the pure-numpy attention summarization helpers (no torch/comfy needed)."""

import numpy as np

from krea2_explorations import attention_stats as ast


def test_head_token_average_shape_and_values():
    # attn: [tokens, heads, n, n]
    attn = np.zeros((2, 3, 4, 4), dtype=np.float32)
    attn[..., 0] = 1.0  # everything attends to key 0
    m = ast.head_token_average(attn)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(m[:, 0], 1.0)
    np.testing.assert_allclose(m[:, 1:], 0.0)


def test_per_head_average_shape():
    attn = np.random.rand(5, 6, 4, 4).astype(np.float32)
    ph = ast.per_head_average(attn)
    assert ph.shape == (6, 4, 4)
    np.testing.assert_allclose(ph, attn.mean(axis=0), rtol=1e-6)


def test_hub_strength_identifies_dominant_key():
    m = np.full((4, 4), 0.1, dtype=np.float32)
    m[:, 2] = 0.7  # key layer 2 is the hub
    s = ast.hub_strength(m)
    assert s.shape == (4,)
    assert int(s.argmax()) == 2


def test_hub_ranking_sorted_desc():
    m = np.full((3, 3), 0.1, dtype=np.float32)
    m[:, 1] = 0.5
    m[:, 2] = 0.3
    ranking = ast.hub_ranking(m, labels=["L2", "L5", "L8"])
    assert [lab for lab, _ in ranking] == ["L5", "L8", "L2"]
    assert ranking[0][1] >= ranking[1][1] >= ranking[2][1]
