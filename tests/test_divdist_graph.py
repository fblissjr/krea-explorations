"""Tests for the diversity-distillation ComfyUI API-graph builders.

These build the multi-sampler graphs (k=1 model swap, sigma-rescale) that scripts/generate.build_graph
can't express. Pure dict construction -> testable without a ComfyUI server. The most valuable check is
structural: every node-reference points at a node that exists (catches wiring typos before a GPU run).
"""
import pytest

from krea2_explorations.divdist_graph import (
    api_to_ui,
    build_rab_combined,
    build_rescale,
    build_single,
    build_split,
)

KW = dict(unet="raw.safetensors", clip="clip.safetensors", vae="vae.safetensors",
          lora="turbo.safetensors")


def _refs(graph):
    """Every input that looks like a [node_key, slot] reference must point to an existing node."""
    keys = set(graph)
    for name, node in graph.items():
        for val in node["inputs"].values():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in keys, f"{name} references missing node {val[0]!r}"


def _types(graph):
    return [n["class_type"] for n in graph.values()]


def test_single_is_one_sampler_with_save_and_valid_refs():
    g = build_single("a cat", lora_strength=1.0, steps=8, **KW)
    assert "save" in g and g["save"]["class_type"] == "SaveImage"
    assert _types(g).count("SamplerCustomAdvanced") == 1
    _refs(g)


def test_single_wires_lora_strength_and_step_count():
    g = build_single("a cat", lora_strength=0.0, steps=28, **KW)  # arm R: base, 28 steps
    lora = next(n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["strength_model"] == 0.0
    sched = next(n for n in g.values() if n["class_type"] == "BasicScheduler")
    assert sched["inputs"]["steps"] == 28


def test_split_has_two_samplers_disable_noise_and_handoff():
    g = build_split("a cat", first_strength=0.0, rest_strength=1.0, k=1, steps=8, **KW)
    scas = {k: v for k, v in g.items() if v["class_type"] == "SamplerCustomAdvanced"}
    assert len(scas) == 2
    # exactly one DisableNoise, feeding the second (rest) sampler's noise slot
    assert _types(g).count("DisableNoise") == 1
    first_key = next(k for k, v in scas.items() if v["inputs"]["noise"][0] != _disable_key(g))
    rest_key = next(k for k in scas if k != first_key)
    assert scas[rest_key]["inputs"]["noise"][0] == _disable_key(g)
    # the rest sampler consumes the first sampler's latent output (the handoff)
    assert scas[rest_key]["inputs"]["latent_image"][0] == first_key
    _refs(g)


def test_split_step_matches_k_and_strengths():
    g = build_split("a cat", first_strength=0.5, rest_strength=1.0, k=2, steps=8, **KW)
    split = next(n for n in g.values() if n["class_type"] == "SplitSigmas")
    assert split["inputs"]["step"] == 2
    strengths = sorted(n["inputs"]["strength_model"] for n in g.values()
                       if n["class_type"] == "LoraLoaderModelOnly")
    assert strengths == [0.5, 1.0]


def test_rescale_uses_ksampler_denoise_and_optional_cleanup():
    g1 = build_rescale("a cat", denoise=0.8, cleanup_denoise=0.0, steps=8, **KW)
    ks = [n for n in g1.values() if n["class_type"] == "KSampler"]
    assert len(ks) == 1 and ks[0]["inputs"]["denoise"] == pytest.approx(0.8)
    _refs(g1)
    g2 = build_rescale("a cat", denoise=0.8, cleanup_denoise=0.5, steps=8, **KW)
    ks2 = [n for n in g2.values() if n["class_type"] == "KSampler"]
    assert len(ks2) == 2  # first pass + cleanup
    _refs(g2)


def test_default_encode_is_clip_text_encode():
    g = build_single("a cat", lora_strength=1.0, **KW)
    assert g["pos"]["class_type"] == "CLIPTextEncode"


def test_keep_system_routes_pos_through_te_node_with_template_end():
    # the <think> axis needs the Krea2EncodeKeepSystem node, not stock CLIPTextEncode
    full = "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n..\n</think>\n\n"
    g = build_single(full, lora_strength=1.0, keep_system=True, template_end=3, **KW)
    assert g["pos"]["class_type"] == "Krea2EncodeKeepSystem"
    assert g["pos"]["inputs"]["template_end"] == 3
    assert g["pos"]["inputs"]["text"] == full
    _refs(g)


def test_keep_system_works_for_split_too():
    g = build_split("a cat", first_strength=0.0, k=1, keep_system=True, **KW)
    assert g["pos"]["class_type"] == "Krea2EncodeKeepSystem"
    _refs(g)


def test_combined_rab_shares_loaders_and_has_three_saves():
    g = build_rab_combined("a cat", unet="raw.s", clip="c.s", vae="v.s", lora="t.s")
    _refs(g)
    assert _types(g).count("UNETLoader") == 1  # one RAW load shared by all arms
    assert _types(g).count("SaveImage") == 3
    assert _types(g).count("SamplerCustomAdvanced") == 4  # R=1, A=1, B=2
    prefixes = sorted(n["inputs"]["filename_prefix"] for n in g.values() if n["class_type"] == "SaveImage")
    assert prefixes == ["divdist/A/A", "divdist/B/B", "divdist/R/R"]


def _ui_link_ids(ui):
    return {link[0] for link in ui["links"]}


def test_api_to_ui_preserves_nodes_and_wires_every_ref():
    api = build_single("a cat", lora_strength=1.0, **KW)
    ui = api_to_ui(api)
    assert len(ui["nodes"]) == len(api)
    # one UI link per API reference
    n_refs = sum(1 for node in api.values() for v in node["inputs"].values()
                 if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str))
    assert len(ui["links"]) == n_refs
    # no dangling: every input link id resolves in the links table
    ids = _ui_link_ids(ui)
    for node in ui["nodes"]:
        for inp in node["inputs"]:
            if inp.get("link") is not None:
                assert inp["link"] in ids


def test_api_to_ui_orders_widgets_and_injects_seed_control():
    ui = api_to_ui(build_single("a cat", lora_strength=1.0, seed=7, **KW))
    unet = next(n for n in ui["nodes"] if n["type"] == "UNETLoader")
    assert unet["widgets_values"] == ["raw.safetensors", "default"]
    noise = next(n for n in ui["nodes"] if n["type"] == "RandomNoise")
    assert noise["widgets_values"] == [7, "fixed"]  # control_after_generate injected


def test_api_to_ui_roundtrips_combined_workflow():
    ui = api_to_ui(build_rab_combined("a cat", unet="raw.s", clip="c.s", vae="v.s", lora="t.s"))
    assert ui["version"] == 0.4 and ui["nodes"] and ui["links"]
    assert [n for n in ui["nodes"] if n["type"] == "SaveImage"]


def _disable_key(graph):
    return next(k for k, v in graph.items() if v["class_type"] == "DisableNoise")
