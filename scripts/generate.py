#!/usr/bin/env python
"""Generate a Krea 2 image through the ComfyUI API.

Talks to a running ComfyUI server over HTTP, so it needs no ComfyUI path — just the
model filenames as they appear in your ComfyUI models dirs.

Key points baked in:
- The flow-match shift (1.15) comes from Krea2's own model config; ModelSamplingFlux was removed because
  it only re-derives that resolution-aware `mu`, which is a proven pixel-identical no-op at the ~1MP
  trained area (it only differs far off-1MP, where it's a weak lever anyway).
- Deterministic euler/simple + fixed seed by default, so A/B comparisons isolate your change.
- Presets: `turbo` (8 steps, CFG off) and `raw` (28 steps, CFG 5.5). `--preset raw` auto-selects the
  RAW checkpoint; pass `--unet` to override.

Examples:
    uv run python scripts/generate.py "a red fox in snow" --out out.png
    uv run python scripts/generate.py --prompt-file p.txt --preset raw \
        --unet krea2_raw_fp8_scaled.safetensors --out raw.png
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Default model filenames (as they appear in the ComfyUI models dirs). Single source so harnesses can
# import these instead of re-hardcoding the strings.
DEFAULT_UNET = "krea2_turbo_fp8_scaled.safetensors"
DEFAULT_RAW_UNET = "krea2_raw_fp8_scaled.safetensors"
DEFAULT_CLIP = "qwen3vl_4b_bf16.safetensors"  # bf16 encoder: faithful conditioning (this repo measures it), ~free VRAM (it's offloaded before the DiT sampling pass). The DiT stays fp8 -- 12B bf16 won't fit a 24GB card.
STOCK_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"  # fp8 fallback for resolve_clip (the prior default encoder)
# krea2RealVae = the community detail VAE (spacepxl's upscale2x decoder, sub-pixel head averaged to 3ch so it
# drops in via the stock VAELoader). Crisper skin/texture than the stock qwen_image_vae; STOCK_VAE is the fallback.
DEFAULT_VAE = "krea2RealVae_v10.safetensors"
STOCK_VAE = "qwen_image_vae.safetensors"
TURBO_LORA = "krea2_turbo_lora_rank_64_bf16.safetensors"

# ComfyUI cfg = reference guidance + 1 (Krea: v = cond + g*(cond-uncond); ComfyUI: uncond + cfg*(...)).
# Reference inference.py default guidance=4.5 -> ComfyUI cfg=5.5. Turbo runs CFG off (cfg=1.0).
# Each preset fully specifies a recipe: steps, cfg, the checkpoint, and (lora_name, strength) pairs. So
# `--preset raw` never silently runs on the Turbo checkpoint, and `turbo_lora` is RAW + the Turbo LoRA
# (the de-distillation dial) turnkey.
PRESETS = {
    "turbo":      dict(steps=8,  cfg=1.0, unet=DEFAULT_UNET,     loras=()),
    "raw":        dict(steps=28, cfg=5.5, unet=DEFAULT_RAW_UNET, loras=()),
    "turbo_lora": dict(steps=8,  cfg=1.0, unet=DEFAULT_RAW_UNET, loras=((TURBO_LORA, 1.0),)),
}


def model_node(class_type, **inputs):
    """A ``model_patches`` entry: insert ``class_type`` on the model edge, wiring the current model ref into
    its ``model`` input, and return the new ref. The single seam that levers (a LoRA, an attention bias, a
    residual steer) compose onto, so the graph skeleton lives here instead of being re-inlined per harness."""
    def patch(g, model_ref):
        nid = f"patch_{sum(1 for k in g if k.startswith('patch_'))}"
        g[nid] = {"class_type": class_type, "inputs": {"model": model_ref, **inputs}}
        return [nid, 0]
    return patch


def build_graph(prompt, *, unet, clip, vae, negative="", steps=8, cfg=1.0, seed=42,
                width=1024, height=1024, sampler="euler", scheduler="simple",
                loras=None, lora=None, lora_strength=1.0,
                sage=None, model_patches=None, filename_prefix="krea2_gen"):
    """Build a ComfyUI API graph for a single Krea 2 txt2img.

    Pass ``loras=[(name, strength), ...]`` to chain several LoRAs on the model edge (e.g. a style LoRA + a
    projector ``.diff`` + others), each at its own strength. ``lora=<filename>`` / ``lora_strength`` is
    the single-LoRA shorthand.

    Pass ``sage=<mode>`` (e.g. "auto") to insert the ``Krea2SageAttention`` node (our own
    optimized_attention_override, no KJNodes) on the model edge before the sampler — the Phase-0
    sage-vs-SDPA A/B. Same seed both arms; diff the outputs and compare wall-clock.

    Pass ``model_patches=[model_node(...), ...]`` to hang extra levers on the model edge, in order, after
    the LoRA and sage — a harness composes its interpretability nodes here instead of re-inlining the skeleton.
    """
    g = {"ckpt": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": unet, "weight_dtype": "default"}}}
    sampler_model = ["ckpt", 0]
    # one LoraLoaderModelOnly per (name, strength), chained on the model edge: `loras` stacks several
    # (e.g. a style LoRA + a projector .diff + others), `lora`/`lora_strength` is the single-LoRA shorthand.
    chain = list(loras) if loras else ([(lora, lora_strength)] if lora else [])
    for i, (lora_name, strength) in enumerate(chain):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": sampler_model, "lora_name": lora_name, "strength_model": strength}}
        sampler_model = [f"lora{i}", 0]
    if sage and sage != "disabled":
        g["sage"] = {"class_type": "Krea2SageAttention",
                     "inputs": {"model": sampler_model, "sage_mode": sage}}
        sampler_model = ["sage", 0]
    # composable levers on the model edge (attention bias, residual steer, ...); each is a model_node(...)
    for patch in model_patches or []:
        sampler_model = patch(g, sampler_model)
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    g["latent"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}
    # Always a real (possibly empty) negative, never ConditioningZeroOut. ZeroOut only behaves on the pure
    # distilled Turbo checkpoint; on RAW (incl. RAW+Turbo-LoRA) and with the CFG++ (_cfg_pp) samplers -- which
    # use the uncond even at cfg 1 -- it produces grainy output. At cfg 1 with a normal sampler ComfyUI skips
    # the uncond pass, so this is byte-identical to ZeroOut there (verified) and costs nothing.
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": sampler_model, "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0], "seed": seed, "steps": steps,
                               "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                               "denoise": 1.0}}
    g["vaedec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["vaedec", 0], "filename_prefix": filename_prefix}}
    return g


def build_split_graph(prompt, *, unet_high, unet_low, clip, vae, boundary,
                      negative="", steps=8, cfg_high=2.5, cfg_low=1.0, seed=42,
                      width=1024, height=1024, sampler="euler", scheduler="simple",
                      lora_high=None, lora_high_strength=1.0,
                      lora_low=None, lora_low_strength=1.0,
                      model_patches_high=None, model_patches_low=None,
                      filename_prefix="krea2_split"):
    """Two-sampler split: a high-noise model denoises steps ``[0, boundary)``, a low-noise model finishes
    ``[boundary, steps)``, on one shared flow-shifted schedule (Wan-2.2-style leftover-noise handoff).

    The high-noise stage carries the guidance — real CFG + ``negative`` — because composition and seed
    diversity are decided in the high-noise steps; the low-noise stage finishes Turbo-style (``cfg_low`` 1,
    no negative). ``unet_high`` / ``unet_low`` are the two checkpoints (e.g. RAW then Turbo); pass
    ``lora_high`` / ``lora_low`` to insert a per-stage ``LoraLoaderModelOnly`` (e.g. the Turbo LoRA on the
    low stage instead of a separate Turbo checkpoint). ``boundary`` is the handoff step index and must be a
    real interior split (``0 < boundary < steps``); both stages share ``steps``/``sampler``/``scheduler`` so
    the schedule is continuous across the handoff.
    """
    if not 0 < boundary < steps:
        raise ValueError(f"boundary must satisfy 0 < boundary < steps; got boundary={boundary}, steps={steps}")

    g = {}

    def model_branch(tag, unet, lora, lora_strength, patches):
        g[f"{tag}_unet"] = {"class_type": "UNETLoader",
                            "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        src = [f"{tag}_unet", 0]
        if lora:
            g[f"{tag}_lora"] = {"class_type": "LoraLoaderModelOnly",
                                "inputs": {"model": src, "lora_name": lora, "strength_model": lora_strength}}
            src = [f"{tag}_lora", 0]
        for patch in patches or []:
            src = patch(g, src)
        return src

    high_model = model_branch("high", unet_high, lora_high, lora_high_strength, model_patches_high)
    low_model = model_branch("low", unet_low, lora_low, lora_low_strength, model_patches_low)

    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    g["latent"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}}

    def neg_node(name):
        # Always a real (empty) negative, never ConditioningZeroOut (see build_graph) -- both stages run RAW.
        g[name] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
        return [name, 0]

    neg_high = neg_node("neg_high")
    neg_low = neg_node("neg_low")

    # Stage 1: high-noise model, real CFG, [0, boundary), KEEP the leftover noise for the handoff.
    g["s1"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg_high,
                          "sampler_name": sampler, "scheduler": scheduler, "positive": ["pos", 0],
                          "negative": neg_high, "latent_image": ["latent", 0], "start_at_step": 0,
                          "end_at_step": boundary, "return_with_leftover_noise": "enable", "model": high_model}}
    # Stage 2: low-noise model, no added noise, continue [boundary, steps), finish clean.
    g["s2"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": cfg_low,
                          "sampler_name": sampler, "scheduler": scheduler, "positive": ["pos", 0],
                          "negative": neg_low, "latent_image": ["s1", 0], "start_at_step": boundary,
                          "end_at_step": steps, "return_with_leftover_noise": "disable", "model": low_model}}
    g["vaedec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["s2", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["vaedec", 0], "filename_prefix": filename_prefix}}
    return g


def run(graph, out_path, server="http://127.0.0.1:8188", timeout=600,
        harness=None, arm="run", seed=0, prompt=None, dump_dir=None):
    """Submit a graph and save the first SaveImage output to out_path. Returns True on success.

    Pass ``harness=...`` to auto-dump the API graph + provenance sidecar (the workflow convention, see
    CLAUDE.md "Workflows") *before* the POST, so every render leaves a loadable artifact even if it fails.
    Opt-in: with no ``harness`` nothing is dumped (harnesses that call ``dump_workflow`` themselves are
    unaffected and won't double-dump). ``dump_dir`` overrides the default ``internal/workflows/``.
    """
    if harness is not None:
        from workflow_dump import dump_workflow
        kw = {"out_dir": dump_dir} if dump_dir is not None else {}
        try:
            dump_workflow(graph, harness=harness, arm=arm, seed=seed, prompt=prompt, **kw)
        except OSError as e:  # best-effort: a disk/permission write failure must not abort a valid render
            print(f"[generate.run] workflow dump skipped ({e})", flush=True)
        # NB: ValueError (a bad/empty graph) is NOT caught -> fail fast on a programmer error.
    req = urllib.request.Request(server + "/prompt",
                                 data=json.dumps({"prompt": graph}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    poll = 0.25  # finer than 1s so the A/B wall-clock isn't quantized to whole seconds
    for _ in range(int(timeout / poll)):
        time.sleep(poll)
        h = json.loads(urllib.request.urlopen(server + f"/history/{pid}", timeout=30).read())
        # wait for COMPLETED, not just present: ComfyUI lists the prompt in /history while it's still running
        # (esp. during the first-render model load), so reading outputs on mere presence returns empty -> a
        # spurious False. This was the real cause of the "first render after a restart fails" flakiness.
        if pid in h and h[pid].get("status", {}).get("completed"):
            # find the first output node carrying images, rather than assuming a SaveImage keyed "save"
            outs = h[pid].get("outputs", {})
            imgs = next((o["images"] for o in outs.values() if o.get("images")), [])
            if not imgs:
                return False  # completed with no image output = errored, or a cached re-run (no fresh output)
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            Path(out_path).write_bytes(urllib.request.urlopen(server + "/view?" + q, timeout=120).read())
            return True
    return False


def _parse_lora(spec):
    """Parse a CLI LoRA spec 'name[:strength]' into (name, strength); strength defaults to 1.0."""
    name, _, strength = spec.partition(":")
    return (name, float(strength) if strength else 1.0)


def _pick(available, preferred, fallback):
    """Return `preferred` if it's in `available`, else `fallback`."""
    return preferred if preferred in available else fallback


def _resolve(server, loader, field, preferred, fallback):
    """Pick `preferred` if the server's `loader` lists it under `field`, else `fallback`. One probe so a
    bf16/krea2RealVae default degrades to the prior default instead of hard-failing the loader on a bare install."""
    try:
        info = json.loads(urllib.request.urlopen(server + f"/object_info/{loader}", timeout=10).read())
        return _pick(info[loader]["input"]["required"][field][0], preferred, fallback)
    except Exception:
        return fallback


def resolve_vae(server, preferred=DEFAULT_VAE, fallback=STOCK_VAE):
    """krea2RealVae if the server has it, else stock qwen_image_vae."""
    return _resolve(server, "VAELoader", "vae_name", preferred, fallback)


def resolve_clip(server, preferred=DEFAULT_CLIP, fallback=STOCK_CLIP):
    """bf16 qwen3vl encoder if the server has it, else the fp8 encoder."""
    return _resolve(server, "CLIPLoader", "clip_name", preferred, fallback)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="prompt text (or use --prompt-file)")
    ap.add_argument("--prompt-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="turbo")
    ap.add_argument("--unet", default=None, help="checkpoint filename; defaults to the checkpoint for --preset")
    ap.add_argument("--clip", default=None, help="CLIP/encoder filename; default = bf16 if present, else fp8")
    ap.add_argument("--vae", default=None, help="VAE filename; default = krea2RealVae if present, else stock")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, help="override preset steps")
    ap.add_argument("--cfg", type=float, help="override preset cfg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]",
                    help="add a LoRA on the model edge (repeatable; stacks on the preset's LoRAs). Strength "
                         "after a colon, e.g. style.safetensors:0.8 (default 1.0).")
    ap.add_argument("--sage", nargs="?", const="auto", default=None,
                    help="insert the Krea2SageAttention node (our override, no KJNodes). Bare --sage = "
                         "'auto' (sm89 -> fp8_cuda++); or pass a mode (fp8_cuda++, fp16_triton, ...). "
                         "Omit for the plain-SDPA baseline. Run both arms at the same seed for the A/B.")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    prompt = a.prompt or Path(a.prompt_file).read_text().strip()
    p = PRESETS[a.preset]
    steps = int(a.steps or p["steps"])
    cfg = float(a.cfg if a.cfg is not None else p["cfg"])
    unet = a.unet or p["unet"]                                   # --preset picks its checkpoint unless overridden
    vae = a.vae or resolve_vae(a.server)                         # krea2RealVae if the server has it, else stock
    clip = a.clip or resolve_clip(a.server)                      # bf16 encoder if the server has it, else fp8
    loras = list(p["loras"]) + [_parse_lora(x) for x in a.lora]  # preset LoRAs first, then any --lora, in order
    g = build_graph(prompt, unet=unet, clip=clip, vae=vae, negative=a.negative,
                    steps=steps, cfg=cfg, seed=a.seed, width=a.width, height=a.height,
                    sampler=a.sampler, scheduler=a.scheduler,
                    loras=loras, sage=a.sage)
    t0 = time.perf_counter()
    ok = run(g, a.out, server=a.server)
    elapsed = time.perf_counter() - t0
    print(f"{'saved' if ok else 'FAILED'} {a.out}  (preset={a.preset} steps={steps} cfg={cfg} "
          f"{a.width}x{a.height} seed={a.seed} loras={loras or 'none'} sage={a.sage or 'off'} wall={elapsed:.2f}s)")


if __name__ == "__main__":
    main()
