"""Tests for the random matched-norm LoRA control generator.

A matched-norm random LoRA is the zero-information perturbation control for any
LoRA experiment: same keys/shapes as a real trained adapter, same *magnitude* of
weight delta, but a random direction. If the random control reproduces an effect
the trained LoRA produces, the effect is pure perturbation, not learned content.
"""
import numpy as np

from krea2_explorations.random_lora import match_norm_random


def _fro(a):
    return float(np.sqrt(np.sum(a.astype(np.float64) ** 2)))


def test_matches_effective_delta_norm_per_pair():
    rng = np.random.default_rng(0)
    src = {
        "blk.attn.to_q.lora_A.weight": rng.standard_normal((4, 16)).astype(np.float32),
        "blk.attn.to_q.lora_B.weight": rng.standard_normal((16, 4)).astype(np.float32),
    }
    out = match_norm_random(src, np.random.default_rng(1))

    assert set(out) == set(src)
    for k in src:
        assert out[k].shape == src[k].shape
        assert out[k].dtype == src[k].dtype

    # the physically meaningful quantity — the effective delta B@A — is norm-matched
    od = src["blk.attn.to_q.lora_B.weight"] @ src["blk.attn.to_q.lora_A.weight"]
    nd = out["blk.attn.to_q.lora_B.weight"] @ out["blk.attn.to_q.lora_A.weight"]
    assert np.isclose(_fro(od), _fro(nd), rtol=1e-5)

    # but the direction is genuinely random (not the original)
    assert not np.allclose(out["blk.attn.to_q.lora_A.weight"], src["blk.attn.to_q.lora_A.weight"])


def test_unpaired_tensor_matched_per_tensor():
    # a lone tensor (no A/B partner) is norm-matched on its own
    src = {"alpha": np.array([[3.0, 4.0]], dtype=np.float32)}  # Frobenius norm 5
    out = match_norm_random(src, np.random.default_rng(2))
    assert out["alpha"].shape == (1, 2)
    assert np.isclose(_fro(src["alpha"]), _fro(out["alpha"]), rtol=1e-5)


def test_deterministic_given_rng():
    src = {
        "x.lora_A.weight": np.ones((2, 3), np.float32),
        "x.lora_B.weight": np.ones((4, 2), np.float32),
    }
    a = match_norm_random(src, np.random.default_rng(7))
    b = match_norm_random(src, np.random.default_rng(7))
    for k in src:
        assert np.array_equal(a[k], b[k])


def test_preserves_key_order():
    src = {
        "z.lora_A.weight": np.ones((2, 2), np.float32),
        "a.lora_B.weight": np.ones((2, 2), np.float32),
        "z.lora_B.weight": np.ones((2, 2), np.float32),
        "a.lora_A.weight": np.ones((2, 2), np.float32),
    }
    out = match_norm_random(src, np.random.default_rng(0))
    assert list(out) == list(src)
