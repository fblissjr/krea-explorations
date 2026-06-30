#!/usr/bin/env python
"""Canonical Krea2 t2i workflows on the modular SamplerCustomAdvanced stack -- THE source of truth for the
named A/B/C/D recipes. Call A/B/C/D (or build_single/build_split) for ANY canonical render; scripts/generate.py
is the low-level primitive + HTTP layer beneath this.

No ModelSamplingFlux (proven no-op at ~1MP, pixel-identical). RAW + Turbo-LoRA at varying strength is the
de-distillation dial. Modular stack: RandomNoise + guider + sampler + sigmas -> SamplerCustomAdvanced.
  A drop-in        LoRA 0.8, cfg1 (BasicGuider), euler, beta57, 8
  B cfg-headroom   LoRA 0.6, cfg2.5 (CFGGuider+neg), euler, beta57, 8
  C split          RAW(cfg2.5)->RAW+LoRA1.0(cfg1), one beta57 schedule SplitSigmas@3 of 8
  D sde finish     LoRA 0.8, cfg1, SamplerDPMPP_2M_SDE(eta0.5), beta57, 8   (native eta, no RES4LYF)
SaveImage keyed 'save' (generate.run contract). Krea2Resolution->EmptyLatentImage (needs a restart to load;
validation renders use a fixed 1024^2 latent instead).

  uv run python scripts/canonical_workflows.py     # validate-render A/B/C/D + grid
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from generate import DEFAULT_RAW_UNET, TURBO_LORA, DEFAULT_CLIP, DEFAULT_VAE  # noqa: E402  single source for filenames

RAW = DEFAULT_RAW_UNET   # fp8 DiT (12B bf16 won't fit a 24GB card; fp8 leaves headroom)
LORA = TURBO_LORA
CLIP = DEFAULT_CLIP      # bf16 qwen3vl encoder (preferred; main() falls back to fp8 via resolve_clip at run time)
VAE = DEFAULT_VAE        # krea2RealVae detail decoder (preferred; main() falls back to stock via resolve_vae at run time)
BENIGN = "This is a photorealistic photograph of a golden retriever in a sunny park, sharp focus."

KSEL = lambda n: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": n}}
DPMPP_SDE = lambda eta: {"class_type": "SamplerDPMPP_2M_SDE",
                         "inputs": {"solver_type": "midpoint", "eta": eta, "s_noise": 1.0, "noise_device": "gpu"}}


def _common(g, prompt, use_res):
    g["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    if use_res:
        g["res"] = {"class_type": "Krea2Resolution", "inputs": {"preset": "1024x1024 (1:1)"}}
        g["lat"] = {"class_type": "EmptyLatentImage",
                    "inputs": {"width": ["res", 0], "height": ["res", 1], "batch_size": 1}}
    else:
        g["lat"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}


def _tail(g, sca):
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": [sca, 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "canon"}}
    return g


def _base(strength):
    """RAW UNETLoader + Turbo-LoRA preamble shared by build_single and build_split (strength = the dial)."""
    return {"ckpt": {"class_type": "UNETLoader", "inputs": {"unet_name": RAW, "weight_dtype": "default"}},
            "lora": {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["ckpt", 0], "lora_name": LORA, "strength_model": strength}}}


def build_single(prompt, *, strength, cfg, sampler, seed=42, steps=8, scheduler="beta57", use_res=True):
    g = _base(strength)
    _common(g, prompt, use_res)
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sigmas"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": ["lora", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    if cfg > 1.0:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}}
        g["guider"] = {"class_type": "CFGGuider",
                       "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": cfg}}
    else:
        g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["pos", 0]}}
    g["sampler"] = sampler
    g["sca"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                           "sigmas": ["sigmas", 0], "latent_image": ["lat", 0]}}
    return _tail(g, "sca")


def build_split(prompt, *, boundary=3, steps=8, cfg_high=2.5, seed=42, scheduler="beta57", use_res=True):
    if not 0 < boundary < steps:  # match build_split_graph's guard: a real interior split, no empty stage
        raise ValueError(f"boundary must satisfy 0 < boundary < steps; got boundary={boundary}, steps={steps}")
    g = _base(1.0)  # RAW high-noise -> Turbo (LoRA 1.0) finish
    _common(g, prompt, use_res)
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}}
    g["sampler"] = KSEL("euler")
    g["sig_full"] = {"class_type": "BasicScheduler",
                     "inputs": {"model": ["ckpt", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    g["split"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sig_full", 0], "step": boundary}}
    g["gh"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["ckpt", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": cfg_high}}
    g["gl"] = {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["pos", 0]}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["dis"] = {"class_type": "DisableNoise", "inputs": {}}
    g["s1"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["noise", 0], "guider": ["gh", 0], "sampler": ["sampler", 0],
                          "sigmas": ["split", 0], "latent_image": ["lat", 0]}}
    g["s2"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["dis", 0], "guider": ["gl", 0], "sampler": ["sampler", 0],
                          "sigmas": ["split", 1], "latent_image": ["s1", 0]}}
    return _tail(g, "s2")


def A(prompt=BENIGN, use_res=True, seed=42): return build_single(prompt, strength=0.8, cfg=1.0, sampler=KSEL("euler"), seed=seed, use_res=use_res)
def B(prompt=BENIGN, use_res=True, seed=42): return build_single(prompt, strength=0.6, cfg=2.5, sampler=KSEL("euler"), seed=seed, use_res=use_res)
def C(prompt=BENIGN, use_res=True, seed=42): return build_split(prompt, seed=seed, use_res=use_res)
def D(prompt=BENIGN, use_res=True, seed=42): return build_single(prompt, strength=0.8, cfg=1.0, sampler=DPMPP_SDE(0.5), seed=seed, use_res=use_res)


def reference_workflows():
    return {"80_canonical_A_turbolora08": A(), "81_canonical_B_cfg_headroom": B(),
            "82_canonical_C_split": C(), "83_canonical_D_sde_finish": D()}


def main():
    from generate import run, resolve_vae, resolve_clip
    from krea2_explorations.image_grid import build_contact_sheet
    server = "http://127.0.0.1:8188"
    vae, clip = resolve_vae(server), resolve_clip(server)  # fall back to stock/fp8 if the preferred aren't installed
    out = REPO / "data" / "canonical"; out.mkdir(parents=True, exist_ok=True)
    arms = {"A_turbolora08": A, "B_cfg_headroom": B, "C_split": C, "D_sde_finish": D}
    cells = []
    for lab, fn in arms.items():
        g = fn(BENIGN, use_res=False, seed=42)  # fixed latent: Krea2Resolution needs a restart to load
        g["clip"]["inputs"]["clip_name"] = clip  # run-time fallback if bf16/krea2RealVae aren't installed
        g["vae"]["inputs"]["vae_name"] = vae
        ok = run(g, str(out / f"{lab}.png"), harness="canonical", arm=lab, seed=42, prompt=BENIGN)
        print(f"  {lab:16}: {'ok' if ok else 'FAIL'}", flush=True)
        cells.append(out / f"{lab}.png" if ok else None)
    build_contact_sheet([cells], out / "grid.png", col_labels=list(arms),
                        row_labels=["RAW+TurboLoRA modular"],
                        title="Canonical A/B/C/D (modular SamplerCustomAdvanced, no ModelSamplingFlux)")
    print(f"grid -> {out / 'grid.png'}")


if __name__ == "__main__":
    main()
