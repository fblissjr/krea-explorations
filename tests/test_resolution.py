"""Tests for the Krea2 resolution snap helper + buckets (all workable resolutions must be /16)."""

import re

import pytest

from krea2_explorations.krea2_resolution_node import BUCKETS, Krea2Resolution, snap_resolution

# The common aspect ratios we expect as presets, with their exact /16 ~1 MP dimensions.
COMMON_RATIO_BUCKETS = {
    "1120x896 (5:4)":   (1120, 896),
    "896x1120 (4:5)":   (896, 1120),
    "1152x864 (4:3)":   (1152, 864),
    "864x1152 (3:4)":   (864, 1152),
    "1248x832 (3:2)":   (1248, 832),
    "832x1248 (2:3)":   (832, 1248),
    "1280x720 (16:9)":  (1280, 720),
    "720x1280 (9:16)":  (720, 1280),
    "1568x672 (21:9)":  (1568, 672),
    "672x1568 (9:21)":  (672, 1568),
}


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


@pytest.mark.parametrize("label,dims", COMMON_RATIO_BUCKETS.items())
def test_common_ratio_buckets_present_with_exact_dims(label, dims):
    assert BUCKETS.get(label) == dims, f"missing/incorrect preset: {label}"


def test_bucket_label_ratio_matches_actual_dimensions():
    # every labeled bucket's "(a:b)" must match its real w/h within 1% -- guards against mislabels
    # (e.g. the old "1216x832 (3:2)", which is really 1.46:1, not 1.5:1).
    for label, dims in BUCKETS.items():
        if dims is None:                                  # the 'custom' sentinel has no ratio
            continue
        m = re.search(r"\((\d+):(\d+)\)", label)
        assert m, f"bucket label missing an (a:b) ratio: {label}"
        a, b = int(m.group(1)), int(m.group(2))
        w, h = dims
        labeled, actual = a / b, w / h
        assert abs(actual - labeled) / labeled < 0.01, (
            f"{label}: dims are {actual:.4f}:1 but label claims {labeled:.4f}:1")


def test_resolve_emits_latent_matching_resolution():
    torch = pytest.importorskip("torch")  # skipped in the torch-free project venv; runs in the comfy venv
    w, h, latent = Krea2Resolution().resolve(preset="1152x896 (9:7)", batch_size=2)
    assert (w, h) == (1152, 896)
    assert latent["samples"].shape == (2, 4, 896 // 8, 1152 // 8)
    assert latent["samples"].dtype == torch.float32
