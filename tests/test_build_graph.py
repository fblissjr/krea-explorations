"""Tests for `scripts/generate.build_graph` — single-image txt2img wiring after the ModelSamplingFlux removal.

The flow shift now comes from Krea2's model config (MSF was a proven no-op at ~1MP), so the sampler's model
edge wires straight from the UNETLoader (or the LoRA, or the sage node) with no ModelSamplingFlux in between.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import PRESETS, DEFAULT_RAW_UNET, DEFAULT_UNET, TURBO_LORA, build_graph  # noqa: E402


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
    assert g["sampler"]["inputs"]["model"] == ["lora0", 0]
    assert g["lora0"]["inputs"]["model"] == ["ckpt", 0]
    assert g["lora0"]["inputs"]["strength_model"] == 0.5


def test_multiple_loras_chain_each_at_its_own_strength():
    g = _g(loras=[("style.safetensors", 1.0), ("proj.diff", 0.5), ("other.safetensors", 0.8)])
    assert g["lora0"]["inputs"]["model"] == ["ckpt", 0]      # first off the loader
    assert g["lora1"]["inputs"]["model"] == ["lora0", 0]     # chained
    assert g["lora2"]["inputs"]["model"] == ["lora1", 0]
    assert g["sampler"]["inputs"]["model"] == ["lora2", 0]   # sampler reads the last in the chain
    assert [g[f"lora{i}"]["inputs"]["strength_model"] for i in range(3)] == [1.0, 0.5, 0.8]


def test_sage_node_inserted_on_model_edge():
    g = _g(sage="auto")
    assert g["sage"]["inputs"]["model"] == ["ckpt", 0]      # sage sits right after the loader (no shift)
    assert g["sampler"]["inputs"]["model"] == ["sage", 0]


def test_real_negative_when_cfg_on():
    g = _g(cfg=2.5, negative="blurry")
    neg = g[g["sampler"]["inputs"]["negative"][0]]
    assert neg["class_type"] == "CLIPTextEncode" and neg["inputs"]["text"] == "blurry"


def test_real_negative_even_when_cfg_off():
    # always a real CLIPTextEncode negative (never ConditioningZeroOut): safe at cfg1, required for _cfg_pp/RAW
    g = _g(cfg=1.0, negative="grainy")
    neg = g[g["sampler"]["inputs"]["negative"][0]]
    assert neg["class_type"] == "CLIPTextEncode" and neg["inputs"]["text"] == "grainy"
    assert "ConditioningZeroOut" not in {v["class_type"] for v in g.values()}


def test_presets_specify_unet_and_loras():
    # every preset fully specifies the recipe, so --preset raw never runs on the Turbo checkpoint
    for name, p in PRESETS.items():
        assert {"steps", "cfg", "unet", "loras"} <= set(p), name
    assert PRESETS["raw"]["unet"] == DEFAULT_RAW_UNET and PRESETS["raw"]["loras"] == ()
    assert PRESETS["turbo"]["unet"] == DEFAULT_UNET


def test_turbo_lora_preset_is_raw_plus_turbo_lora():
    p = PRESETS["turbo_lora"]
    assert p["unet"] == DEFAULT_RAW_UNET                    # RAW checkpoint
    assert p["loras"] == ((TURBO_LORA, 1.0),)              # plus the Turbo LoRA -- the de-distillation dial
    assert (p["steps"], p["cfg"]) == (8, 1.0)


def test_pick_vae_falls_back_to_stock_when_absent():
    from generate import _pick_vae, DEFAULT_VAE, STOCK_VAE
    assert _pick_vae([DEFAULT_VAE, STOCK_VAE]) == DEFAULT_VAE   # preferred present -> use it
    assert _pick_vae([STOCK_VAE]) == STOCK_VAE                  # preferred absent -> fall back to stock
    assert _pick_vae([]) == STOCK_VAE                          # nothing available -> stock
