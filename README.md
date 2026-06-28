# krea2-explorations

Last updated: 2026-06-28

A study of **how Krea 2 combines its text-encoder layers** (its "multilayer feature aggregation") and how that
conditioning can be **steered** — plus a small, dependency-light toolkit to reproduce the measurements. The
most practically useful result: the conditioning is steerable *from the prompt side* (a `<think>` block acts
as a steering vector) — see [What we've found](#what-weve-found).

With the toolkit you can:

- read and **per-layer-scale the learned projector** that fuses the layers,
- emit **tiny projector LoRAs** for arbitrary per-layer gains (load with the stock LoRA loader, no custom
  node, ~300 bytes each),
- **isolate a single encoder layer** to see what it contributes to the image, and
- **recompute the layer-fusion attention maps** from the open weights.

Everything runs on CPU and edits the checkpoint in place — one tensor in a 26 GB model in seconds, without
loading the full thing.

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

**Prompt-side steering (a `<think>` block in the assistant turn) — the most useful result.** Beyond editing
weights, the conditioning can be steered from the *prompt side*. Turbo's distillation flattens intense
expression; appending a short `<think>` reasoning span to the assistant turn — injected via the tokenizer's
skip-template route (a full `<|im_start|>…` string passed as the prompt) — restores it **in-distribution**, as
well as or better than the deep-band rebalance lever, with adherence intact. A CPU probe shows the span shifts
the selected hidden states ~17–24% along their own dominant axis (0.86 direction consistency, energy at the
L20/L23 hub): it behaves like a **steering vector**.

![Krea 2 Turbo, same seed, four expressions across three columns — stock prompt, + a `<think>` block, and + the deep-band rebalance lever. The in-distribution `<think>` block restores the distillation-flattened intense expressions (furious, terrified) as well as or better than the weight-space rebalance lever, with prompt adherence intact.](docs/figures/think_steering_grid.png)

*Same seed (#123); only the column lever changes. The `<think>` block (middle) restores Turbo's
distillation-flattened expression in-distribution, matching or beating the deep-band rebalance lever (right)
without leaving the data manifold. `joyful` (bottom) is a control — it isn't a flattened expression, so it
renders in every column.*

Two implementation details, **verified by running Comfy's actual Krea 2 tokenizer** (not assumed): (1) the
special tokens are tokenized per the model's config — `<think>`/`</think>`/`<|im_start|>` each map to a single
token id (151667 / 151668 / 151644), not to literal angle-bracket text; (2) **Krea 2's text encoder strips
the system turn**: `Krea2TEModel.encode_token_weights` slices the conditioning from the *second* `<|im_start|>`
(the user turn) onward, so the entire `<|im_start|>system … <|im_end|>` block is discarded before the DiT sees
it. So the surviving, directly-steerable write-points are the **user turn and the assistant `<think>` turn**;
a *system*-turn prompt only influences conditioning **indirectly** (the surviving tokens attend back over it
during the encoder pass), not as injected conditioning. Low–medium confidence on the visual result (one
subject, a few seeds). Details in [`docs/findings.md`](docs/findings.md).

**Sampler-side: "diversity distillation" does not transfer to Krea 2's flow schedule (a falsification).**
Distilled (Turbo) models lose same-seed diversity; [Gandikota & Bau](https://arxiv.org/abs/2503.10637)
attribute this to over-commitment at the *first* denoising step and fix it by running the **base** model for
just that one step (k=1). Krea 2 is a flow model — they tested none — so we measured it: the collapse
**reproduces** (Turbo gives near-identical compositions across seeds where RAW gives varied ones), but the
**k=1 fix is a null** — running RAW for the first step, or sweeping the first-step strength / handoff step,
changes essentially nothing. Diversity is instead recovered by lowering the **global** Turbo-LoRA strength: a
clean diversity↔quality dial (crossover ~0.5), i.e. the bottleneck is **distributed across all steps**, not
localized at the first. Tooling realizes base↔distilled as one RAW load + Turbo-LoRA strength
(`divdist_graph`; the importable `example_workflows/krea2_diversity_distillation_rab.json` is *derived* from
the same builders via `api_to_ui`). Two prompts, 4 seeds, visual read.

The rest of the findings **characterize the trained aggregation** by reading the open weights:

> Framing: the *architecture* (cross-layer attention over the 12 layers → `Linear(12→1)` projector →
> refiners) is public — see Prior work. The items below **characterize the trained model's behavior**, read
> off the open weights; they are not architecture we uncovered. Most are low-effort to reproduce (the
> projector is 12 numbers in the checkpoint; the hub is one forward pass). The value is the characterization
> plus a few measurements that clarify what the aggregation does, not a hidden-structure reveal. Full write-up with confidence levels in
> [`docs/findings.md`](docs/findings.md); attention-map figures in `docs/figures/`, numeric arrays in
> `docs/data/`.

**Confirmatory (expected from LLM-layer priors).** Isolating one selected layer at a time (and the
complementary leave-one-out): the **deep** selected layers (L23/26/29/32) carry the renderable content and
each can render a coherent image alone; **shallow** layers alone are noise; **L14** carries structure/layout;
the **final layer (L35) alone is unusable for image generation** — consistent with Krea's stated reason for
aggregating multiple layers rather than using the last hidden state.

**Measured (empirical; read off the open model, and — as far as we know — unpublished).**
- **L20 acts as a mid-layer attention hub.** In the layer-fusion attention, nearly every selected layer
  attends to L20. Validated across **5 prompts** (photo / anime / illustration + two long dense prompts)
  and, on the dense ones, **content-token-masked**: L20 is the top key-layer for **~91–95% of content
  tokens** — so it's content-driven, not a padding/template artifact — and it holds across most attention
  heads. The concentration is a **learned *directional* hub, not a magnitude sink**: L20's hidden-state norm
  is mid-pack (rank 6/12) and its pre-norm key norm is among the *lowest* (rank 10/12), and the block's
  `qknorm` equalizes every layer's key magnitude — so the routing is decided by learned query/key
  *direction* (the trained `wq`/`wk` send most queries toward L20's key direction), not by magnitude, and
  not by any hardcoded index (the layerwise blocks carry no positional encoding).
- **The projector combines contrastively, not as an average.** Its learned weights are positive on the mid
  layers (peak at L14) and strongly negative on the deep layers (L23/29/32) — roughly "mid minus deep",
  applied to the attention-mixed slots.

**What the projector-rebalance lever does.** Benign attributes (expression, "wet", blush) come through the
learned aggregation and render whether or not the projector is rebalanced: a difference-of-means probe
shows they survive *more* strongly than ordinary content controls, a with/without causal test renders them
clearly, and a stock-vs-rebalanced test renders them either way. Boosting the deep layers (the rebalance
lever) mainly shifts **detail, contrast, and intensity** — consistent with the deep layers carrying fine
detail (see Prior work) — rather than changing whether an attribute appears.

Confidence is bounded by probes done at a handful of prompts/seeds and an image-level distance metric for the
leave-one-out ranking. The layer-fusion findings hold on **both RAW and Turbo** — the `txtfusion` weights are
near-identical across checkpoints (projector cosine 1.0; L20 hub 92–95% on RAW) — and the **refiner/token
blocks do local (diagonal) token attention** with no dominant sink, so the striking structure lives in the
layer-fusion, not the token-refinement.

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

## Reverse-caption probing

We also use a reverse-caption loop (image → dense caption from a vision LLM → regenerate) to derive dense
test prompts from reference images. Example captions are in
[`examples/test_prompts/`](examples/test_prompts/). The image-to-text system prompt used to produce them:

```
Describe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects and the background. Write a single cohesive paragraph (no lists or markdown), as a dense text-to-image caption. Open by naming the actual medium/style (e.g. photograph, painting, illustration, 3D render — but identify what it truly is). Cover composition and framing, the main subject(s) and their attributes, and the lighting. Put any visible text in quotes, exactly. Be specific and grounded — describe only what is actually visible; do not invent details, intent, or backstory.
```

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
| `cli` | `krea2-proj inspect \| lora \| solo`. |
| `attention_stats` + `scripts/extract_attention.py` | Pure-numpy summarization helpers, plus a script that loads the Krea 2 CLIP and the DiT's `txtfusion` weights (CPU) and recomputes the 12×12 layer-fusion attention maps (head-averaged, per-head, cross-prompt). |
| `image_grid` | Reusable labeled contact-sheet builder (`build_contact_sheet`) for comparison figures — rows × cols of image paths / `PIL.Image` / `None` (missing cells become placeholders). |
| `divdist_graph` | Pure, tested ComfyUI **API-graph builders** for diversity-distillation arms (`build_single`, `build_split` k=1 base→distilled handoff, `build_rescale` denoise sigma-rescale), realizing base↔distilled as one RAW load + Turbo-LoRA strength. `api_to_ui` derives the importable R/A/B workflow from the same builders (no drift); the `<think>` axis routes through `Krea2EncodeKeepSystem`. Driven by `scripts/generate.run`. |
| `diversity` | Dependency-light pairwise perceptual-diversity aggregation (`pairwise_diversity` / `diversity_table`, metric injected) for the average-pairwise-DreamSim protocol. |
| `krea2_untwist_node` (+ `rope_untwist`, `krea2_untwist_attn`) | **Experimental.** A ComfyUI node, **Krea 2 Untwist Style Reference** (`conditioning/Krea2`), for **training-free reference-image style transfer** via untwisting-RoPE shared attention — a separate *positional-axis* lever (the rest of this toolkit edits the *feature/conditioning axis*). Built for Krea 2's real `[32,48,48]` RoPE. v1 uses renoise-to-sigma reference injection (no RF-inversion); style transfer needs a **low `high_scale`** (the high default copies the reference). Restart ComfyUI to load it. |

## License

This tooling is released by the author. The Krea 2 models it operates on are covered by the **Krea 2
Community License** (see [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw)); use of the models is
subject to that license.
