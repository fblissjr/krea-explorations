"""Tests for the Krea 2 Sage override node. The pure routing-policy decision (_should_sage) is the
load-bearing logic and is exercised here without torch/sage/comfy or a GPU; the sage call itself is
validated in the in-pipeline A/B, not in unit tests."""

from krea2_explorations import comfy_nodes as cn
from krea2_explorations.krea2_sage_node import SAGE_MODES, _should_sage

MIN = 1024


def test_mapping_registered():
    assert "Krea2SageAttention" in cn.NODE_CLASS_MAPPINGS
    assert "Krea2SageAttention" in cn.NODE_DISPLAY_NAME_MAPPINGS


def test_input_types_well_formed():
    it = cn.NODE_CLASS_MAPPINGS["Krea2SageAttention"].INPUT_TYPES()
    assert "model" in it["required"]
    assert "sage_mode" in it["required"]
    assert set(it["required"]["sage_mode"][0]) == set(SAGE_MODES)
    for opt in ("min_seq_len", "sage_masked", "verbose", "nvtx"):
        assert opt in it["optional"]


def test_modes_include_auto_disabled_and_sdpa_baseline():
    # 'sdpa' is the true-SDPA baseline arm that overrides a global --use-sage-attention flag.
    assert {"auto", "disabled", "sdpa"} <= set(SAGE_MODES)


# --- routing policy: the one call we want sage on is unmasked + large-seq ---

def test_sages_the_unmasked_large_seq_self_attn():
    # the krea2 DiT image stream: mask=None, 4608 tokens -> sage it
    assert _should_sage(mask=None, seq_len=4608, min_seq_len=MIN, sage_masked=False) is True


def test_skips_short_attention():
    # below the seq guard -> default attention, sage overhead not worth it
    assert _should_sage(mask=None, seq_len=256, min_seq_len=MIN, sage_masked=False) is False


def test_skips_masked_calls_by_default():
    # the masked refiner-text blocks stay on the default path to isolate the win
    assert _should_sage(mask=object(), seq_len=4608, min_seq_len=MIN, sage_masked=False) is False


def test_sage_masked_opt_in_routes_masked_large_seq():
    assert _should_sage(mask=object(), seq_len=4608, min_seq_len=MIN, sage_masked=True) is True


def test_seq_guard_still_applies_under_sage_masked():
    # opting into masked does not bypass the seq-length guard
    assert _should_sage(mask=object(), seq_len=256, min_seq_len=MIN, sage_masked=True) is False
