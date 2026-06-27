"""CPU tests for the Krea2 untwisting attention patch (piece 2).

Validates the risky index/scale logic — ref-only untwist, target untouched, text tokens excluded, correct
shapes — with synthetic tensors, no model and no GPU. torch lives in the ComfyUI venv, not the torch-free
project venv, so this is skipped under `uv run pytest` and run with:

    <comfyui-venv>/bin/python -m pytest tests/test_krea2_untwist_attn.py
"""
import pytest

torch = pytest.importorskip("torch")

from krea2_explorations.krea2_untwist_attn import (  # noqa: E402
    is_image_attention_name,
    untwist_reference_image_keys,
    build_shared_kv,
)
from krea2_explorations.rope_untwist import krea2_freq_scale_vector  # noqa: E402


def _kv(B, H, S, D):
    g = torch.Generator().manual_seed(0)
    k = torch.randn(2 * B, H, S, D, generator=g)
    v = torch.randn(2 * B, H, S, D, generator=g)
    return k, v


def _scale(D):
    return torch.tensor(krea2_freq_scale_vector(D, axes=[D // 2, D // 2], high_scale=1.0, low_scale=3.0))


def test_image_attention_name_excludes_txtfusion():
    assert is_image_attention_name("blocks.0.attn")
    assert is_image_attention_name("blocks.27.attn")
    assert not is_image_attention_name("txtfusion.layerwise_blocks.0.attn")
    assert not is_image_attention_name("txtfusion.refiner_blocks.1.attn")
    assert not is_image_attention_name("blocks.3.ff")


def test_untwist_scales_only_reference_image_keys_without_mutating():
    B, H, S, D, txt = 1, 2, 7, 8, 3
    k, _ = _kv(B, H, S, D)
    sv = _scale(D)
    k0 = k.clone()
    ref = untwist_reference_image_keys(k, sv, (txt, S), B)
    assert torch.allclose(ref, k[B:2 * B, :, txt:S, :] * sv.view(1, 1, 1, -1))
    assert torch.equal(k, k0)  # pure read, no in-place mutation


def test_build_shared_kv_appends_untwisted_ref_image_tokens():
    B, H, S, D, txt = 1, 2, 7, 8, 3
    k, v = _kv(B, H, S, D)
    sv = _scale(D)
    k_t, v_t = build_shared_kv(k, v, sv, (txt, S), B)
    n_img = S - txt
    assert k_t.shape == (B, H, S + n_img, D)
    assert v_t.shape == (B, H, S + n_img, D)
    # target's own K/V are untouched
    assert torch.allclose(k_t[:, :, :S], k[:B])
    assert torch.allclose(v_t[:, :, :S], v[:B])
    # appended portion = reference IMAGE keys * scale (keys) and raw reference image values
    assert torch.allclose(k_t[:, :, S:], k[B:2 * B, :, txt:S, :] * sv.view(1, 1, 1, -1))
    assert torch.allclose(v_t[:, :, S:], v[B:2 * B, :, txt:S, :])
