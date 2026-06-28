"""ComfyUI API-graph builders for diversity-distillation arms (sibling to scripts/generate.build_graph).

generate.build_graph is a single KSampler; the arms here need multi-sampler graphs the stock builder
can't express. These are pure ``dict`` builders (no torch/comfy) so they're unit-testable without a
server; drive them with ``scripts.generate.run`` (graphs name the SaveImage node ``"save"``, which
``run`` expects). All arms realize base-vs-distilled as ONE RAW load + the Turbo LoRA (the distillation
delta): strength 0 = base, 1 = distilled. cfg defaults to 1 (this install's RAW is clean at cfg 1).

Arms (and the named builders that make them):
  build_single   RAW+LoRA@s, full schedule        s=1 Turbo (A) | s=0 RAW/28 (R) | 0<s<1 partial (D)
  build_split    base/partial for first k steps,  first=0,rest=1,k=1 = diversity distillation (B);
                 distilled for the rest           first=s = first-step strength sweep (B')
  build_rescale  Turbo started at a lower sigma   denoise=0.8 = "sigma rescaled to 0.8" (+ cleanup) (C)
  build_rab_combined  one shared-loader graph with R+A+B side by side (for the hand-run UI workflow)

``api_to_ui`` converts any of these API graphs to an importable UI workflow, so the hand-run workflow is
DERIVED from the same builders the sweep uses -- the two can't drift.

Shift is pinned (base==max==1.15) so every arm shares one sigma schedule and only the swapped model
varies. mu=1.15 is Turbo's reference value (see internal/scripts/think_emotion_grid.py).
"""
from __future__ import annotations

PINNED_SHIFT = 1.15


# --------------------------------------------------------------------------------------------------
# shared node helpers (single source of wiring truth for every arm)
# --------------------------------------------------------------------------------------------------

def _loaders(g, *, prompt, clip, vae, width, height, keep_system=False, template_end=0):
    """Add the shared CLIP/VAE/conditioning/latent nodes (cfg-1 negative).

    keep_system=True routes the positive prompt through the Krea2EncodeKeepSystem node (the <think>/
    system-turn TE node) instead of stock CLIPTextEncode; pass the full <|im_start|>... string as prompt.
    """
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    if keep_system:
        g["pos"] = {"class_type": "Krea2EncodeKeepSystem",
                    "inputs": {"clip": ["clip", 0], "text": prompt, "template_end": template_end}}
    else:
        g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    g["latent"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}


def _shifted_model(g, key, *, unet, lora, strength, width, height):
    """RAW -> LoraLoaderModelOnly@strength -> ModelSamplingFlux; returns the shifted model ref."""
    g.setdefault("ckpt", {"class_type": "UNETLoader",
                          "inputs": {"unet_name": unet, "weight_dtype": "default"}})
    g[f"lora_{key}"] = {"class_type": "LoraLoaderModelOnly",
                        "inputs": {"model": ["ckpt", 0], "lora_name": lora, "strength_model": strength}}
    g[f"shift_{key}"] = {"class_type": "ModelSamplingFlux",
                         "inputs": {"model": [f"lora_{key}", 0], "max_shift": PINNED_SHIFT,
                                    "base_shift": PINNED_SHIFT, "width": width, "height": height}}
    return [f"shift_{key}", 0]


def _sca(noise, guider, sigmas, latent):
    """A SamplerCustomAdvanced node dict (sampler is always the shared 'ssel' KSamplerSelect)."""
    return {"class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": noise, "guider": guider, "sampler": ["ssel", 0],
                       "sigmas": sigmas, "latent_image": latent}}


def _cfg_guider(model, cfg):
    return {"class_type": "CFGGuider",
            "inputs": {"model": model, "positive": ["pos", 0], "negative": ["neg", 0], "cfg": cfg}}


def _decode_save(g, sampler_ref, prefix, ns=""):
    """Add VAEDecode + SaveImage under an optional key namespace (default keeps 'save' for run())."""
    g[f"{ns}vaedec"] = {"class_type": "VAEDecode",
                        "inputs": {"samples": sampler_ref, "vae": ["vae", 0]}}
    g[f"{ns}save"] = {"class_type": "SaveImage",
                      "inputs": {"images": [f"{ns}vaedec", 0], "filename_prefix": prefix}}


# --------------------------------------------------------------------------------------------------
# arm builders (standalone graphs for the API sweep; each names its SaveImage node "save")
# --------------------------------------------------------------------------------------------------

def build_single(prompt, *, unet, clip, vae, lora, lora_strength=1.0, steps=8, seed=42,
                 width=1024, height=1024, cfg=1.0, sampler="euler", scheduler="simple",
                 keep_system=False, template_end=0, filename_prefix="divdist"):
    """One model (RAW + LoRA@lora_strength), full schedule, one SamplerCustomAdvanced."""
    g: dict = {}
    _loaders(g, prompt=prompt, clip=clip, vae=vae, width=width, height=height,
             keep_system=keep_system, template_end=template_end)
    model = _shifted_model(g, "m", unet=unet, lora=lora, strength=lora_strength, width=width, height=height)
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["ssel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    g["sig"] = {"class_type": "BasicScheduler",
                "inputs": {"model": model, "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    g["guider"] = _cfg_guider(model, cfg)
    g["sca"] = _sca(["noise", 0], ["guider", 0], ["sig", 0], ["latent", 0])
    _decode_save(g, ["sca", 0], filename_prefix)
    return g


def build_split(prompt, *, unet, clip, vae, lora, first_strength=0.0, rest_strength=1.0, k=1, steps=8,
                seed=42, width=1024, height=1024, cfg=1.0, sampler="euler", scheduler="simple",
                keep_system=False, template_end=0, filename_prefix="divdist"):
    """LoRA@first_strength for the first k steps, LoRA@rest_strength for the rest (one shared schedule)."""
    g: dict = {}
    _loaders(g, prompt=prompt, clip=clip, vae=vae, width=width, height=height,
             keep_system=keep_system, template_end=template_end)
    m_first = _shifted_model(g, "first", unet=unet, lora=lora, strength=first_strength,
                             width=width, height=height)
    m_rest = _shifted_model(g, "rest", unet=unet, lora=lora, strength=rest_strength,
                            width=width, height=height)
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["disable"] = {"class_type": "DisableNoise", "inputs": {}}
    g["ssel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    g["sig"] = {"class_type": "BasicScheduler",
                "inputs": {"model": m_rest, "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    g["split"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sig", 0], "step": k}}
    g["g_first"] = _cfg_guider(m_first, cfg)
    g["g_rest"] = _cfg_guider(m_rest, cfg)
    g["sca_first"] = _sca(["noise", 0], ["g_first", 0], ["split", 0], ["latent", 0])
    g["sca_rest"] = _sca(["disable", 0], ["g_rest", 0], ["split", 1], ["sca_first", 0])
    _decode_save(g, ["sca_rest", 0], filename_prefix)
    return g


def build_rescale(prompt, *, unet, clip, vae, lora, lora_strength=1.0, denoise=0.8, cleanup_denoise=0.0,
                  steps=8, seed=42, width=1024, height=1024, cfg=1.0, sampler="euler",
                  scheduler="simple", keep_system=False, template_end=0, filename_prefix="divdist"):
    """Turbo-only: start at a lower sigma via KSampler denoise<1 (the 'rescale to 0.8'), optional 2nd pass.

    denoise<1 on an empty latent starts the schedule partway down -- the message's diversity trick.
    cleanup_denoise>0 chains a second Turbo pass to finish denoising.
    """
    g: dict = {}
    _loaders(g, prompt=prompt, clip=clip, vae=vae, width=width, height=height,
             keep_system=keep_system, template_end=template_end)
    model = _shifted_model(g, "m", unet=unet, lora=lora, strength=lora_strength, width=width, height=height)
    common = dict(positive=["pos", 0], negative=["neg", 0], steps=steps, cfg=cfg,
                  sampler_name=sampler, scheduler=scheduler)
    g["ks1"] = {"class_type": "KSampler",
                "inputs": {"model": model, "latent_image": ["latent", 0], "denoise": denoise,
                           "seed": seed, **common}}
    final = ["ks1", 0]
    if cleanup_denoise > 0:
        g["ks2"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "latent_image": ["ks1", 0], "denoise": cleanup_denoise,
                               "seed": seed + 1, **common}}
        final = ["ks2", 0]
    _decode_save(g, final, filename_prefix)
    return g


def build_rab_combined(prompt, *, unet, clip, vae, lora, seed=42, steps_turbo=8, steps_raw=28,
                       width=1024, height=1024, cfg=1.0, sampler="euler", scheduler="simple",
                       keep_system=False, template_end=0):
    """One graph with R (raw/28) + A (turbo/8) + B (k=1) sharing loaders, seed and schedule.

    For the hand-run UI workflow: same wiring helpers as the sweep builders, so it can't drift from them.
    """
    g: dict = {}
    _loaders(g, prompt=prompt, clip=clip, vae=vae, width=width, height=height,
             keep_system=keep_system, template_end=template_end)
    base = _shifted_model(g, "base", unet=unet, lora=lora, strength=0.0, width=width, height=height)
    turbo = _shifted_model(g, "turbo", unet=unet, lora=lora, strength=1.0, width=width, height=height)
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["disable"] = {"class_type": "DisableNoise", "inputs": {}}
    g["ssel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    g["sig_turbo"] = {"class_type": "BasicScheduler",
                      "inputs": {"model": turbo, "scheduler": scheduler, "steps": steps_turbo, "denoise": 1.0}}
    g["sig_raw"] = {"class_type": "BasicScheduler",
                    "inputs": {"model": base, "scheduler": scheduler, "steps": steps_raw, "denoise": 1.0}}
    g["split"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sig_turbo", 0], "step": 1}}
    g["g_base"] = _cfg_guider(base, cfg)
    g["g_turbo"] = _cfg_guider(turbo, cfg)
    # R: base, 28 steps (diversity ceiling)
    g["R_sca"] = _sca(["noise", 0], ["g_base", 0], ["sig_raw", 0], ["latent", 0])
    _decode_save(g, ["R_sca", 0], "divdist/R/R", ns="R_")
    # A: distilled, 8 steps (diversity floor == Turbo)
    g["A_sca"] = _sca(["noise", 0], ["g_turbo", 0], ["sig_turbo", 0], ["latent", 0])
    _decode_save(g, ["A_sca", 0], "divdist/A/A", ns="A_")
    # B: base first step, distilled rest (k=1 diversity distillation)
    g["B_sca0"] = _sca(["noise", 0], ["g_base", 0], ["split", 0], ["latent", 0])
    g["B_sca1"] = _sca(["disable", 0], ["g_turbo", 0], ["split", 1], ["B_sca0", 0])
    _decode_save(g, ["B_sca1", 0], "divdist/B/B", ns="B_")
    return g


# --------------------------------------------------------------------------------------------------
# API -> UI workflow converter (so the hand-run workflow is derived, not hand-maintained)
# --------------------------------------------------------------------------------------------------

# Per-type UI metadata for the node set these builders use. ``widgets`` lists the literal inputs in UI
# order ("__control__" -> the control_after_generate widget Comfy injects after a seed). ``outputs`` is
# the output slots. Link inputs are detected structurally (a [key, slot] value), so they need no schema.
_SCHEMA = {
    "UNETLoader": {"widgets": ["unet_name", "weight_dtype"], "outputs": [("MODEL", "MODEL")]},
    "LoraLoaderModelOnly": {"widgets": ["lora_name", "strength_model"], "outputs": [("MODEL", "MODEL")]},
    "ModelSamplingFlux": {"widgets": ["max_shift", "base_shift", "width", "height"],
                          "outputs": [("MODEL", "MODEL")]},
    "CLIPLoader": {"widgets": ["clip_name", "type", "device"], "outputs": [("CLIP", "CLIP")]},
    "VAELoader": {"widgets": ["vae_name"], "outputs": [("VAE", "VAE")]},
    "CLIPTextEncode": {"widgets": ["text"], "outputs": [("CONDITIONING", "CONDITIONING")]},
    "Krea2EncodeKeepSystem": {"widgets": ["text", "template_end"],
                              "outputs": [("CONDITIONING", "CONDITIONING")]},
    "ConditioningZeroOut": {"widgets": [], "outputs": [("CONDITIONING", "CONDITIONING")]},
    "EmptyLatentImage": {"widgets": ["width", "height", "batch_size"], "outputs": [("LATENT", "LATENT")]},
    "RandomNoise": {"widgets": ["noise_seed", "__control__"], "outputs": [("NOISE", "NOISE")]},
    "KSamplerSelect": {"widgets": ["sampler_name"], "outputs": [("SAMPLER", "SAMPLER")]},
    "DisableNoise": {"widgets": [], "outputs": [("NOISE", "NOISE")]},
    "BasicScheduler": {"widgets": ["scheduler", "steps", "denoise"], "outputs": [("SIGMAS", "SIGMAS")]},
    "SplitSigmas": {"widgets": ["step"], "outputs": [("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")]},
    "CFGGuider": {"widgets": ["cfg"], "outputs": [("GUIDER", "GUIDER")]},
    "SamplerCustomAdvanced": {"widgets": [], "outputs": [("output", "LATENT"), ("denoised_output", "LATENT")]},
    "KSampler": {"widgets": ["seed", "__control__", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
                 "outputs": [("LATENT", "LATENT")]},
    "VAEDecode": {"widgets": [], "outputs": [("IMAGE", "IMAGE")]},
    "SaveImage": {"widgets": ["filename_prefix"], "outputs": []},
}


def _is_ref(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def _depths(graph):
    """Longest-path depth per node key (sources = 0), for a readable left-to-right layout."""
    memo: dict = {}

    def depth(key, stack=()):
        if key in memo:
            return memo[key]
        if key in stack:  # cycle guard (graphs here are DAGs; be safe anyway)
            return 0
        refs = [v[0] for v in graph[key]["inputs"].values() if _is_ref(v)]
        d = 0 if not refs else 1 + max(depth(r, stack + (key,)) for r in refs)
        memo[key] = d
        return d

    return {k: depth(k) for k in graph}


def api_to_ui(graph, *, col_px=340, row_px=200):
    """Convert an API graph ({key: {class_type, inputs}}) to an importable UI workflow dict.

    Pure transform: node count is preserved, every [key, slot] reference becomes one UI link, widgets are
    emitted in canonical order (with the seed control widget injected). Layout is a simple depth layering.
    """
    keys = list(graph)
    ids = {k: i + 1 for i, k in enumerate(keys)}
    depth = _depths(graph)
    col_count: dict = {}

    nodes, links, lid = [], [], 0
    for key in keys:
        node = graph[key]
        ct = node["class_type"]
        if ct not in _SCHEMA:
            raise KeyError(f"api_to_ui: no UI schema for node type {ct!r} (add it to _SCHEMA)")
        sch = _SCHEMA[ct]
        d = depth[key]
        row = col_count.get(d, 0)
        col_count[d] = row + 1

        widgets = []
        for w in sch["widgets"]:
            widgets.append("fixed" if w == "__control__" else node["inputs"].get(w))

        inputs = [(name, v) for name, v in node["inputs"].items() if _is_ref(v)]
        ui_inputs = [{"name": name, "type": _SCHEMA[graph[v[0]]["class_type"]]["outputs"][v[1]][1],
                      "link": None} for name, v in inputs]
        ui_outputs = [{"name": n, "type": t, "links": []} for n, t in sch["outputs"]]

        nodes.append({
            "id": ids[key], "type": ct, "pos": [d * col_px, row * row_px], "size": [300, 130],
            "flags": {}, "order": ids[key], "mode": 0,
            "inputs": ui_inputs, "outputs": ui_outputs,
            "properties": {"Node name for S&R": ct}, "widgets_values": widgets, "title": key,
        })

    node_by_id = {n["id"]: n for n in nodes}
    for key in keys:
        to_id = ids[key]
        ref_inputs = [v for v in graph[key]["inputs"].values() if _is_ref(v)]
        for slot, v in enumerate(ref_inputs):
            lid += 1
            from_id, from_slot = ids[v[0]], v[1]
            links.append([lid, from_id, from_slot, to_id, slot, _SCHEMA[graph[v[0]]["class_type"]]["outputs"][from_slot][1]])
            node_by_id[from_id]["outputs"][from_slot]["links"].append(lid)
            node_by_id[to_id]["inputs"][slot]["link"] = lid

    return {"last_node_id": len(nodes), "last_link_id": lid, "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4}
