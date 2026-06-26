"""Structural tests for the ComfyUI node (the comfy runtime itself is not exercised here)."""

import numpy as np

from krea2_explorations import comfy_nodes as cn
from krea2_explorations.projector_lora import effective_weights, resolve_gains


def test_mappings_present():
    assert "Krea2ProjectorRebalance" in cn.NODE_CLASS_MAPPINGS
    assert "Krea2ProjectorRebalance" in cn.NODE_DISPLAY_NAME_MAPPINGS


def test_input_types_well_formed():
    it = cn.NODE_CLASS_MAPPINGS["Krea2ProjectorRebalance"].INPUT_TYPES()
    assert "model" in it["required"]
    assert "preset" in it["required"]
    assert "strength" in it["required"]
    choices = it["required"]["preset"][0]
    assert "uniform" in choices and "custom" in choices
    assert "solo_band" in it["optional"]


def test_object_key_is_diffusion_model_prefixed():
    assert cn.OBJECT_KEY == "diffusion_model.txtfusion.projector.weight"


def test_node_math_matches_lora():
    # the patch the node would compute equals the .diff LoRA effective weights (same semantics)
    orig = np.array([-0.0544, -0.1611, 0.3711, 0.5039, 0.7070, 0.3945,
                     0.3984, -1.4375, -0.5117, -0.8906, -0.6094, 0.1128], dtype=np.float32)
    gains = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0]
    np.testing.assert_allclose(effective_weights(orig, gains, 1.0), orig * np.array(gains, dtype=np.float32), rtol=1e-6)
    assert resolve_gains("uniform") == [1.0] * 12
