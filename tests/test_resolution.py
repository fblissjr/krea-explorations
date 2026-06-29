"""Tests for the Krea2 resolution snap helper + buckets (all workable resolutions must be /16)."""

import pytest

from krea2_explorations.krea2_resolution_node import BUCKETS, Krea2Resolution, snap_resolution


def test_snaps_to_nearest_multiple_of_16():
    assert snap_resolution(1001, 1001) == (1008, 1008)   # 1001/16=62.56 -> 63*16
    assert snap_resolution(990, 990) == (992, 992)        # 990/16=61.9 -> 62*16
    assert snap_resolution(1023, 769) == (1024, 768)
    assert snap_resolution(10, 10) == (16, 16)            # floor at one patch*vae unit


def test_snap_to_1mp_preserves_aspect_and_divisibility():
    w, h = snap_resolution(1920, 1080, 16, target_mp=1.0)
    assert w % 16 == 0 and h % 16 == 0
    assert 0.7e6 <= w * h <= 1.3e6                        # rescaled to ~1 MP
    assert w > h                                          # 16:9 aspect preserved


def test_all_buckets_divisible_by_16_and_near_1mp():
    for label, dims in BUCKETS.items():
        if dims is None:                                  # the 'custom' sentinel
            continue
        w, h = dims
        assert w % 16 == 0 and h % 16 == 0, label
        assert 0.6e6 <= w * h <= 1.2e6, label             # ~1 MP training regime


def test_resolve_emits_latent_matching_resolution():
    torch = pytest.importorskip("torch")  # skipped in the torch-free project venv; runs in the comfy venv
    w, h, latent = Krea2Resolution().resolve(preset="1152x896 (9:7)", batch_size=2)
    assert (w, h) == (1152, 896)
    assert latent["samples"].shape == (2, 4, 896 // 8, 1152 // 8)
    assert latent["samples"].dtype == torch.float32
