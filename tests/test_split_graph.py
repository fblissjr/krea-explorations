"""Tests for the two-sampler split graph builder (`scripts/generate.build_split_graph`).

The split runs a high-noise model for steps [0, boundary) then a low-noise model for [boundary, steps) on
one shared shift-warped schedule, via the Wan-2.2-style leftover-noise handoff (two `KSamplerAdvanced`):
stage 1 adds noise and returns with leftover noise; stage 2 adds none and finishes. These assert the wiring
that makes that handoff correct -- the part that silently produces garbage if a flag is wrong.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import build_split_graph  # noqa: E402


def _by_type(g, ctype):
    return {k: v for k, v in g.items() if v["class_type"] == ctype}


def _trace_unet_name(g, ref):
    """Follow a model ref back through LoraLoaderModelOnly to the UNETLoader name."""
    node = g[ref[0]]
    while node["class_type"] != "UNETLoader":
        node = g[node["inputs"]["model"][0]]
    return node["inputs"]["unet_name"]


def _stages(g):
    """Return (high_key, low_key) KSamplerAdvanced nodes, keyed by the add_noise flag."""
    adv = _by_type(g, "KSamplerAdvanced")
    assert len(adv) == 2, f"expected two KSamplerAdvanced, got {len(adv)}"
    high = next(k for k, v in adv.items() if v["inputs"]["add_noise"] == "enable")
    low = next(k for k, v in adv.items() if v["inputs"]["add_noise"] == "disable")
    return high, low


def _graph(**kw):
    base = dict(unet_high="raw.safetensors", unet_low="turbo.safetensors",
                clip="qwen.safetensors", vae="vae.safetensors", boundary=3,
                steps=8, cfg_high=2.5, cfg_low=1.0)
    base.update(kw)
    return build_split_graph("a prompt", **base)


def test_two_advanced_samplers_with_leftover_noise_handoff():
    g = _graph()
    high, low = _stages(g)
    hi, lo = g[high]["inputs"], g[low]["inputs"]
    # Stage 1 (high noise): add noise, run [0, boundary), and KEEP the leftover noise for stage 2.
    assert hi["start_at_step"] == 0 and hi["end_at_step"] == 3
    assert hi["return_with_leftover_noise"] == "enable"
    # Stage 2 (low noise): add NO noise, continue [boundary, steps), finish clean.
    assert lo["add_noise"] == "disable"
    assert lo["start_at_step"] == 3 and lo["end_at_step"] == 8
    assert lo["return_with_leftover_noise"] == "disable"


def test_stage2_consumes_stage1_latent():
    g = _graph()
    high, low = _stages(g)
    assert g[low]["inputs"]["latent_image"][0] == high, "stage 2 must denoise stage 1's leftover latent"
    # Stage 1 starts from the empty latent, not another sampler.
    assert g[g[high]["inputs"]["latent_image"][0]]["class_type"] == "EmptyLatentImage"


def test_shared_schedule_across_stages():
    g = _graph(steps=8, sampler="euler", scheduler="simple")
    high, low = _stages(g)
    for key in ("steps", "sampler_name", "scheduler"):
        assert g[high]["inputs"][key] == g[low]["inputs"][key], f"{key} must match so the split is one schedule"


def test_models_routed_high_to_low():
    g = _graph(unet_high="raw.safetensors", unet_low="turbo.safetensors")
    high, low = _stages(g)
    assert _trace_unet_name(g, g[high]["inputs"]["model"]) == "raw.safetensors"
    assert _trace_unet_name(g, g[low]["inputs"]["model"]) == "turbo.safetensors"
    # No ModelSamplingFlux: the flow shift comes from Krea2's model config (the node is a no-op at ~1MP).
    assert not _by_type(g, "ModelSamplingFlux")


def test_cfg_per_stage():
    g = _graph(cfg_high=4.0, cfg_low=1.0)
    high, low = _stages(g)
    assert g[high]["inputs"]["cfg"] == 4.0
    assert g[low]["inputs"]["cfg"] == 1.0


def test_high_stage_uses_real_negative_when_cfg_on():
    g = _graph(cfg_high=2.5, negative="blurry, lowres")
    high, _ = _stages(g)
    neg = g[g[high]["inputs"]["negative"][0]]
    assert neg["class_type"] == "CLIPTextEncode"
    assert neg["inputs"]["text"] == "blurry, lowres"


def test_low_stage_zeroes_out_negative_when_cfg_off():
    g = _graph(cfg_low=1.0)
    _, low = _stages(g)
    assert g[g[low]["inputs"]["negative"][0]]["class_type"] == "ConditioningZeroOut"


def test_per_stage_lora_inserted_on_its_branch():
    g = _graph(lora_high="projector.safetensors", lora_high_strength=0.5)
    high, _ = _stages(g)
    loras = _by_type(g, "LoraLoaderModelOnly")
    assert any(v["inputs"]["lora_name"] == "projector.safetensors"
               and v["inputs"]["strength_model"] == 0.5 for v in loras.values())
    # The high model chain still resolves to the high UNET (lora sits between loader and sampler).
    assert _trace_unet_name(g, g[high]["inputs"]["model"]) == "raw.safetensors"


@pytest.mark.parametrize("bad", [0, 8, -1, 9])
def test_boundary_must_be_a_real_interior_split(bad):
    with pytest.raises(ValueError):
        _graph(boundary=bad, steps=8)
