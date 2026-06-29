# krea2-explorations

Last updated: 2026-06-29

A small, dependency-light toolkit to measure and surgically edit how Krea 2 builds its text conditioning — and
the findings that came out of using it. It reads and rewrites one tensor inside a 26 GB checkpoint in seconds,
on CPU, without loading the whole model. The generation-side levers run in ComfyUI.

The repo is measure-first: it leads with what the open weights actually do, not with claims. The most useful
single result so far is that you can steer the conditioning from the prompt side — a `<think>` block acts as a
steering vector. See [What we've found](#what-weve-found).

With the toolkit you can:

- read and per-layer-scale the learned projector that fuses the 12 layers;
- emit tiny projector LoRAs for any per-layer gains (load with the stock LoRA loader — no custom node, about
  300 bytes each);
- isolate a single encoder layer to see what it adds to the image;
- measure and dial a named concept axis — build a difference-of-means direction from an A/B prompt pair, then
  amplify, inject or remove it (the *Krea 2 Concept Inject* node and `concept_direction.py`; see
  [Concept directions](#concept-directions-measure-and-dial-any-concept-axis));
- recompute the layer-fusion attention maps from the open weights;
- split the denoise across two models — RAW for the high-noise steps, Turbo for the finish — for more seed
  diversity at near-Turbo speed and clean CFG headroom (see [Two-sampler split](docs/two_sampler_split.md)).

## How Krea 2's text conditioning works

Krea 2's text encoder is a frozen Qwen3-VL-4B. The DiT does not consume a single hidden state — Krea's
`select_layers` picks **12 encoder hidden-state layers** `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]`,
mixes them with **cross-layer attention**, and combines them through a learned `Linear(12 → 1)` projector
(`txtfusion.projector`, weight `[1, 12]`) before the refiner blocks. Krea calls this **multilayer feature
aggregation** (ComfyUI exposes the selected layers as "taps"). This repo turns the knobs on that aggregation
and measures what it learned.

Editing weights (not activations) keeps the conditioning in distribution: the model's own downstream RMSNorm
holds magnitude, so reweighting changes only the *direction* of the combined text vector.

## What we've found

Full write-up, figures and confidence levels are in [`docs/findings.md`](docs/findings.md). The short version:

- **Prompt-side steering works.** A short `<think>` block in the assistant turn acts as a steering vector and
  restores the intense expression Turbo's distillation flattens — in-distribution, with prompt adherence
  intact. Our most useful practical result, at low–medium confidence (one subject, a few seeds).
  → [prompt-side steering](docs/findings.md#prompt-side-think-steering)
- **The 12 layers fuse through an L20 attention hub and a contrastive projector.** Nearly every selected layer
  attends to L20, and the learned `Linear(12→1)` projector is "mid minus deep" — positive on the mid layers,
  negative on the deep ones. Measured off the open weights; it holds across styles and across both the RAW and
  Turbo checkpoints. → [layer-fusion attention](docs/findings.md#layer-fusion)
- **Deep layers carry the content; shallow layers are scaffolding.** Each deep layer (L23/26/29/32) renders a
  coherent image alone; shallow layers alone are noise; L14 carries structure; the final layer alone is
  unusable — which is Krea's stated reason for aggregating layers.
  → [what each layer carries](docs/findings.md#layer-probes)
- **Benign attributes come through the aggregation.** Whether an attribute such as expression appears does not
  hinge on the projector stage; the rebalance lever acts as a detail and intensity knob, not an on/off gate.
  → [attributes vs the rebalance lever](docs/findings.md#attributes)
- **You can dial a named concept axis, within limits.** A difference-of-means direction steers, but how well
  an axis survives the fusion does not predict how well it steers, and `amplify` is a magnitude lever that can
  conjure a concept at high scale — not a presence test. → [labeled-axis steering](docs/findings.md#labeled-axis)
- **Two de-distillation levers.** Krea 2's Turbo LoRA is the distillation delta, so its strength is a
  continuous RAW↔Turbo dial (sweet spot stays at cfg 1, `s0.6–0.8`); and a per-stage split — low-strength
  high-noise → Turbo finish — adds seed diversity at single-pass cost.
  → [Turbo-LoRA dial](docs/turbo_lora_strength.md) · [two-sampler split](docs/two_sampler_split.md)

These findings characterize the trained model's behavior, read off the open weights — they are not architecture
we discovered (the architecture is public; see [Prior work](#prior-work)). Most are low-effort to reproduce:
the projector is 12 numbers in the checkpoint, and the hub is one forward pass.

## Usage

```bash
uv run pytest

# inspect the 12 learned projector weights
uv run krea2-proj inspect <comfyui_models>/diffusion_models/krea2_turbo_bf16.safetensors

# emit a projector .diff LoRA (uniform identity, or custom per-layer gains)
uv run krea2-proj lora <ckpt> <comfyui_models>/loras/krea2_custom.safetensors --gains "1,1,1,1,1,1,1,2,1,1,1,1"

# emit 12 single-layer isolation LoRAs (one per selected layer)
uv run krea2-proj solo <ckpt> <comfyui_models>/loras/krea2_explorer/solo

# recompute the layer-fusion attention maps
uv run python scripts/extract_attention.py
```

In ComfyUI, either load an emitted `.diff` file with the stock **LoraLoaderModelOnly**, or add the
**Krea 2 Projector Rebalance** node between your model loader and sampler. Strength 1 = the chosen gains
exactly; `solo_band` (or a `solo/` LoRA at strength 1) isolates a single selected layer so you can see what
it contributes.

## Concept directions (measure and dial any concept axis)

A targeted generalization of the projector-rebalance lever: instead of scaling whole layers, **measure one
concept direction** from an A/B prompt pair and **amplify / inject / project-out** it in the conditioning.
`amplify` intensifies the component along the measured axis; at high scale it can *conjure* the concept from a
near-zero residual (it's a magnitude lever, not a presence gate — see [findings](docs/findings.md)). Works on
any axis the encoder represents (expression, style, lighting, pose, an attribute), and drags whatever co-varied
in the A/B pair — tighter matched prompts reduce that.

**Turnkey, all in ComfyUI:** wire two `CLIPTextEncode` nodes (concept present / absent) into **Krea 2 Concept
Direction**, feed its output into **Krea 2 Concept Inject** with your prompt, and dial a slider — no terminal,
no files. The bundled [`example_workflows/krea2_concept_inject.json`](example_workflows/krea2_concept_inject.json)
is that wiring with a benign `smile` example; open it and run.

To batch-build a reusable library of directions offline:

```bash
# measure directions from A/B prompt pairs -> one <name>.npy each (CPU)
uv run --active python scripts/concept_direction.py examples/concept_directions.json --out <comfyui_models>/concept_dirs
```

Full guide, modes, and the math: [`docs/concept_directions.md`](docs/concept_directions.md).

## Prior work

Per-layer reweighting of Krea 2's conditioning was introduced by
[`nova452/ComfyUI-ConditioningKrea2Rebalance`](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance)
and refined by [`huwhitememes/comfyui-krea2-conditioning`](https://github.com/huwhitememes/comfyui-krea2-conditioning),
which documented that the **deeper layers carry fine detail and can end up under-represented**, and that a
naive global multiplier inflates the conditioning magnitude (their fix: **RMS-renormalize** to hold
magnitude). Those points — and the magnitude/direction reasoning above — are theirs; the "Confirmatory"
findings overlap with that work.

What this adds is **characterization of the combination** — reading the model's internals rather than only
scaling the conditioning tensor: the **L20 attention hub** and the **contrastive (mid-minus-deep) projector**
(the "Measured" findings), plus the attribute / rebalance-lever measurements above. None of this is hidden
architecture; it's measurement of an open model.

The experimental **Untwist Style Reference** node implements *Untwisting RoPE: Frequency Control for Shared
Attention in DiTs* ([arXiv 2602.05013](https://arxiv.org/abs/2602.05013)). Community ComfyUI nodes exist
([`BigStationW/ComfyUi-Untwisting-RoPE`](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE) and a Krea 2
fork); ours is an independent rebuild on Krea 2's real `[32,48,48]` RoPE axes rather than a `[64,64]`
approximation.

## Components

| Module | What it does |
|--------|--------------|
| `safetensors_patch` | Header parse, single-tensor read, in-place byte patch, and a small-file writer — edits one tensor in a 26 GB checkpoint in seconds without loading it. |
| `projector` | `read_projector` / `scale_projector` — read and per-layer-scale the learned `txtfusion.projector`. |
| `projector_lora` | Emit tiny `txtfusion.projector.diff` LoRAs for arbitrary per-layer gains (`diff = orig*(gain-1)`, so strength 1 = exact gains). `make_band_isolation_loras` emits 12 single-layer probes. Loads via the stock `LoraLoaderModelOnly` — **no custom node required**, ~300 bytes each. |
| `comfy_nodes` | A ComfyUI node, **Krea 2 Projector Rebalance** (`conditioning/Krea2`), that reweights the projector live via the ModelPatcher (`preset` `uniform`/`custom`, `strength`, `per_layer_weights`, `solo_band` to isolate one layer). Restart ComfyUI to load it. |
| `krea2_concept_inject_node` + `scripts/concept_direction.py` | A ComfyUI node, **Krea 2 Concept Inject** (`conditioning/Krea2`), that amplifies / injects / project-outs a single measured concept direction on the conditioning (a targeted generalization of the rebalance lever — a deep-band magnitude boost, aimed), plus a CLI + `contrast_directions.pooled_direction` that measure the direction from an A/B prompt pair. See [`docs/concept_directions.md`](docs/concept_directions.md). |
| `cli` | `krea2-proj inspect \| lora \| solo`. |
| `attention_stats` + `scripts/extract_attention.py` | Pure-numpy summarization helpers, plus a script that loads the Krea 2 CLIP and the DiT's `txtfusion` weights (CPU) and recomputes the 12×12 layer-fusion attention maps (head-averaged, per-head, cross-prompt). |
| `image_grid` | Reusable labeled contact-sheet builder (`build_contact_sheet`) for comparison figures — rows × cols of image paths / `PIL.Image` / `None` (missing cells become placeholders). |
| `scripts/generate` | Build and run a Krea 2 graph over the ComfyUI HTTP API. `build_graph` (one-sampler) and `build_split_graph` (two-stage split); `run` waits for `completed` and finds the image output under any node. See [`docs/krea2_inference.md`](docs/krea2_inference.md). |
| `scripts/generate.build_split_graph` | Two-sampler split graph: a high-noise model for steps `[0, boundary)` then a low-noise model for the finish, on one shared schedule (leftover-noise handoff). RAW→Turbo gives seed/compositional diversity at near-Turbo speed + clean CFG headroom (not negative control). See [`docs/two_sampler_split.md`](docs/two_sampler_split.md). |
| `krea2_resolution_node` | A ComfyUI node, **Krea 2 Resolution** (`conditioning/Krea2`), for Krea 2's workable resolutions: ~1MP buckets, or snap any width/height to the nearest /16 (VAE 8× × DiT patch 2×). Outputs width/height for `EmptyLatentImage`. Restart ComfyUI to load it. See [`docs/krea2_inference.md`](docs/krea2_inference.md). |
| `krea2_untwist_node` (+ `rope_untwist`, `krea2_untwist_attn`) | **Experimental.** A ComfyUI node, **Krea 2 Untwist Style Reference** (`conditioning/Krea2`), for **training-free reference-image style transfer** via untwisting-RoPE shared attention — a separate *positional-axis* lever (the rest of this toolkit edits the *feature/conditioning axis*). Built for Krea 2's real `[32,48,48]` RoPE. v1 uses renoise-to-sigma reference injection (no RF-inversion); style transfer needs a **low `high_scale`** (the high default copies the reference). Restart ComfyUI to load it. |

## License

This tooling is released by the author. The Krea 2 models it operates on are covered by the **Krea 2
Community License** (see [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw)); use of the models is
subject to that license.
