"""Logic tests for the concept-injection math (add / subtract / project_out), numpy-only (no comfy/torch).

The node passes torch tensors; `apply_direction` uses only ops common to numpy and torch, so the math is
verified here in the project venv. Direction is broadcast over all but the last (feature) axis.
"""

import numpy as np

from krea2_explorations.krea2_concept_inject_node import apply_direction


def test_add_moves_along_direction():
    cond = np.zeros((1, 3, 4), dtype=np.float64)
    d = np.array([0.0, 0.0, 0.0, 2.0])  # not unit; normalize=True -> unit [0,0,0,1]
    out = apply_direction(cond, d, scale=5.0, mode="add", normalize=True)
    assert np.allclose(out[..., 3], 5.0) and np.allclose(out[..., :3], 0.0)


def test_subtract_is_negative_add():
    cond = np.ones((2, 2, 4))
    d = np.array([1.0, 0.0, 0.0, 0.0])
    a = apply_direction(cond, d, 3.0, "add", normalize=True)
    s = apply_direction(cond, d, 3.0, "subtract", normalize=True)
    assert np.allclose((a + s) / 2.0, cond)  # add and subtract symmetric about cond


def test_amplify_scales_only_the_present_component():
    d = np.array([0.0, 1.0, 0.0, 0.0])
    cond = np.array([[[3.0, 7.0, 1.0, 2.0]]])  # component along d is 7
    out = apply_direction(cond, d, scale=1.0, mode="amplify", normalize=True)
    assert np.allclose(out[..., 1], 14.0)  # 7*(1+scale) = 14 (scale 1 == bypass's x2)
    assert np.allclose(out[..., [0, 2, 3]], cond[..., [0, 2, 3]])  # orthogonal untouched


def test_amplify_cannot_conjure_an_absent_component():
    # THE key property: amplify only boosts what's present -> can't create an absent (unpaired) concept.
    d = np.array([0.0, 1.0, 0.0, 0.0])
    cond = np.array([[[3.0, 0.0, 1.0, 2.0]]])  # zero component along d
    out = apply_direction(cond, d, scale=50.0, mode="amplify", normalize=True)
    assert np.allclose(out, cond)  # nothing to amplify -> unchanged at any scale


def test_project_out_removes_the_component():
    d = np.array([0.0, 1.0, 0.0, 0.0])
    cond = np.array([[[3.0, 7.0, 1.0, 2.0]]])
    out = apply_direction(cond, d, scale=1.0, mode="project_out", normalize=True)
    assert np.allclose(out[..., 1], 0.0, atol=1e-6)  # the d-component is zeroed (1e-8 unit-eps residual ok)
    assert np.allclose(out[..., [0, 2, 3]], cond[..., [0, 2, 3]])  # orthogonal parts untouched


def test_project_out_is_idempotent():
    d = np.array([1.0, 1.0, 0.0, 0.0])
    cond = np.random.default_rng(0).standard_normal((1, 5, 4))
    once = apply_direction(cond, d, 1.0, "project_out", normalize=True)
    twice = apply_direction(once, d, 1.0, "project_out", normalize=True)
    assert np.allclose(once, twice, atol=1e-6)


def test_unknown_mode_raises():
    try:
        apply_direction(np.zeros((1, 1, 2)), np.ones(2), 1.0, "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_load_direction_rejects_empty_path():
    # empty direction_path (the node's default "") must error clearly, not crash on np.load("")
    from krea2_explorations.krea2_concept_inject_node import _load_direction

    try:
        _load_direction("", 30720, 12, None, np)  # torch unused before the guard fires
    except ValueError as e:
        assert "direction_path" in str(e)
        return
    raise AssertionError("expected ValueError for empty direction_path")


# --- in-graph direction builder (Krea2ConceptDirection): the pooling/diff math, numpy-tested ---

def test_pool_cond_vec_means_over_all_but_last_axis():
    from krea2_explorations.krea2_concept_inject_node import _pool_cond_vec

    c = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)  # (B, seq, feat)
    v = _pool_cond_vec(c)
    assert v.shape == (4,)
    assert np.allclose(v, c.reshape(-1, 4).mean(0))  # mean over B and seq


def test_concept_direction_is_pooled_difference_of_means():
    from krea2_explorations.krea2_concept_inject_node import _concept_direction

    pos = [[np.full((1, 2, 4), 3.0), {}]]  # pools to 3
    neg = [[np.full((1, 2, 4), 1.0), {}]]  # pools to 1
    d = _concept_direction(pos, neg)
    assert d.shape == (4,)
    assert np.allclose(d, 2.0)  # 3 - 1


def test_concept_direction_averages_multiple_conditioning_entries():
    from krea2_explorations.krea2_concept_inject_node import _concept_direction

    pos = [[np.full((1, 1, 4), 2.0), {}], [np.full((1, 1, 4), 4.0), {}]]  # entries average to 3
    neg = [[np.zeros((1, 1, 4)), {}]]
    d = _concept_direction(pos, neg)
    assert np.allclose(d, 3.0)
