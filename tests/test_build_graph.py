"""Tests for `scripts/generate.build_graph` — single-image txt2img wiring after the ModelSamplingFlux removal.

The flow shift now comes from Krea2's model config (MSF was a proven no-op at ~1MP), so the sampler's model
edge wires straight from the UNETLoader (or the LoRA, or the sage node) with no ModelSamplingFlux in between.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import build_graph  # noqa: E402


def _g(**kw):
    base = dict(unet="u.safetensors", clip="c.safetensors", vae="v.safetensors")
    base.update(kw)
    return build_graph("a prompt", **base)


def _types(g):
    return {v["class_type"] for v in g.values()}


def test_no_model_sampling_flux():
    assert "ModelSamplingFlux" not in _types(_g())


def test_sampler_model_wires_to_unet_without_lora():
    g = _g()
    assert g["sampler"]["inputs"]["model"] == ["ckpt", 0]


def test_sampler_model_wires_through_lora_when_set():
    g = _g(lora="proj.safetensors", lora_strength=0.5)
    assert g["sampler"]["inputs"]["model"] == ["lora", 0]
    assert g["lora"]["inputs"]["model"] == ["ckpt", 0]
    assert g["lora"]["inputs"]["strength_model"] == 0.5


def test_sage_node_inserted_on_model_edge():
    g = _g(sage="auto")
    assert g["sage"]["inputs"]["model"] == ["ckpt", 0]      # sage sits right after the loader (no shift)
    assert g["sampler"]["inputs"]["model"] == ["sage", 0]


def test_real_negative_when_cfg_on():
    g = _g(cfg=2.5, negative="blurry")
    neg = g[g["sampler"]["inputs"]["negative"][0]]
    assert neg["class_type"] == "CLIPTextEncode" and neg["inputs"]["text"] == "blurry"


def test_zeroed_negative_when_cfg_off():
    g = _g(cfg=1.0)
    assert g[g["sampler"]["inputs"]["negative"][0]]["class_type"] == "ConditioningZeroOut"
