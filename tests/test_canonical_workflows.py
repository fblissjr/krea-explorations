"""Tests for the canonical A/B/C/D recipes (`scripts/canonical_workflows.py`).

These lock the BLESSED recipe so it can't silently drift again: the modular SamplerCustomAdvanced stack, the
`beta57` scheduler, the krea2RealVae decode + bf16 encoder (sourced from `generate.py`), no `ModelSamplingFlux`,
and a `SplitSigmas` two-stage split for C. Pure dict construction -> no torch needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from canonical_workflows import A, B, C, D, build_split  # noqa: E402
from generate import DEFAULT_VAE, DEFAULT_CLIP  # noqa: E402


def _types(g):
    return {v["class_type"] for v in g.values()}


def test_no_recipe_uses_model_sampling_flux():
    # the 1.15 flow-shift is in Krea2's model config; the node is a pixel-identical no-op and must be absent
    for fn in (A, B, C, D):
        assert "ModelSamplingFlux" not in _types(fn(use_res=False)), fn.__name__


def test_single_recipes_use_the_modular_sampler_not_ksampler():
    for fn in (A, B, D):
        t = _types(fn(use_res=False))
        assert "SamplerCustomAdvanced" in t and "KSampler" not in t, fn.__name__


def test_scheduler_is_beta57_everywhere():
    for fn in (A, B, C, D):
        scheds = [n["inputs"]["scheduler"] for n in fn(use_res=False).values()
                  if n["class_type"] == "BasicScheduler"]
        assert scheds and all(s == "beta57" for s in scheds), fn.__name__


def test_vae_and_clip_come_from_generate_defaults():
    g = A(use_res=False)
    assert g["vae"]["inputs"]["vae_name"] == DEFAULT_VAE        # krea2RealVae, single-sourced from generate.py
    assert g["clip"]["inputs"]["clip_name"] == DEFAULT_CLIP     # bf16 qwen3vl encoder


def test_C_is_a_splitsigmas_two_stage_split():
    g = C(use_res=False)
    assert "SplitSigmas" in _types(g)
    assert sum(1 for v in g.values() if v["class_type"] == "SamplerCustomAdvanced") == 2


def test_A_is_cfg_off_and_B_runs_a_cfg_guider():
    assert "BasicGuider" in _types(A(use_res=False)) and "CFGGuider" not in _types(A(use_res=False))
    assert "CFGGuider" in _types(B(use_res=False))


def test_D_is_dpmpp_2m_sde_at_eta_half():
    sde = [v for v in D(use_res=False).values() if v["class_type"] == "SamplerDPMPP_2M_SDE"]
    assert sde and sde[0]["inputs"]["eta"] == 0.5


def test_build_split_rejects_a_non_interior_boundary():
    # parity with generate.build_split_graph: boundary must be a real interior split (0 < boundary < steps),
    # else a stage silently does ~zero work
    for bad in (0, 8, 9):
        with pytest.raises(ValueError):
            build_split("p", boundary=bad, steps=8, use_res=False)
