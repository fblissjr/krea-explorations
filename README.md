# krea2-explorations

Last updated: 2026-06-26

Tools to **inspect, edit, and explore how Krea 2 combines its text-encoder layers**.

Krea 2's text encoder is Qwen3-VL-4B. The DiT does not use a single hidden state — it takes **12 selected
encoder hidden-state layers** `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]` (Krea's `select_layers`) and
combines them with cross-layer attention followed by a learned `Linear(12 → 1)` projector
(`txtfusion.projector`, weight `[1, 12]`). Krea calls this **"multilayer feature aggregation"**. (ComfyUI
calls the selected layers "taps".) This repo is a small, dependency-light toolkit for looking at — and
turning the knobs on — that aggregation.

## Components

| Module | What it does |
|--------|--------------|
| `safetensors_patch` | Header parse, single-tensor read, in-place byte patch, and a small-file writer — edits one tensor in a 26 GB checkpoint in seconds without loading it. |
| `projector` | `read_projector` / `scale_projector` — read and per-layer-scale the learned `txtfusion.projector`. |
| `projector_lora` | Emit tiny `txtfusion.projector.diff` LoRAs for arbitrary per-layer gains (`diff = orig*(gain-1)`, so strength 1 = exact gains). `make_band_isolation_loras` emits 12 single-layer probes. Loads via the stock `LoraLoaderModelOnly` — **no custom node required**, ~300 bytes each. |
| `comfy_nodes` | A ComfyUI node, **Krea 2 Projector Rebalance** (`conditioning/Krea2`), that reweights the projector live via the ModelPatcher (`preset` `uniform`/`custom`, `strength`, `per_layer_weights`, `solo_band` to isolate one layer). Restart ComfyUI to load it. |
| `cli` | `krea2-proj inspect | lora | solo`. |
| `attention_stats` + `scripts/extract_attention.py` | Pure-numpy summarization helpers, plus a script that loads the Krea 2 CLIP and the DiT's `txtfusion` weights (CPU) and recomputes the 12×12 layer-fusion attention maps (head-averaged, per-head, cross-prompt). |

Editing weights (not activations) keeps the conditioning in distribution: the model's own downstream
RMSNorm holds magnitude, so reweighting changes only the *direction* of the combined text vector.

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

## What we've found

**Confirmatory (expected from LLM-layer priors).** Isolating one selected layer at a time (and the
complementary leave-one-out): the **deep** selected layers (L23/26/29/32) carry the renderable content and
each can render a coherent image alone; **shallow** layers alone are noise; **L14** carries structure/layout;
the **final layer (L35) alone is unusable for image generation** — consistent with Krea's stated reason for
aggregating multiple layers rather than using the last hidden state.

**Model-specific (High confidence; not what you'd expect from a generic DiT).**
- **L20 is a universal mid-layer attention hub.** In the layer-fusion attention, nearly every selected
  layer attends to L20. Validated across **5 prompts** (photo / anime / illustration + two long dense
  prompts) and, on the dense ones, **content-token-masked**: L20 is the top key-layer for **~91–95% of
  content tokens** — so it's content-driven, not a padding/template artifact — and it holds across most
  attention heads. The concentration is a **learned *directional* hub, not a magnitude sink**: L20's
  hidden-state norm is mid-pack (rank 6/12) and its pre-norm key norm is among the *lowest* (rank 10/12),
  and the block's `qknorm` equalizes every layer's key magnitude — so the routing is decided by learned
  query/key *direction* (the trained `wq`/`wk` send most queries toward L20's key direction), not by
  magnitude, and not by any hardcoded index (the layerwise blocks carry no positional encoding).
- **The projector combines contrastively, not as an average.** Its learned weights are positive on the mid
  layers (peak at L14) and strongly negative on the deep layers (L23/29/32) — roughly "mid minus deep",
  applied to the attention-mixed slots.

Confidence is bounded by: probes done at a handful of prompts/seeds, an image-level distance metric for the
leave-one-out ranking, and the attention maps covering the layer-fusion (pre-projector) blocks only.

## LLM System Prompt
Here's the system prompt I'm testing with to generate from image to text:

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

What's new here is **interpretability of the combination itself** — reading the model's internals rather
than only scaling the conditioning tensor: the **L20 attention hub** and the **contrastive
(mid-minus-deep) projector** (the "Model-specific" findings above), which require running the encoder +
`txtfusion`, not just reshaping the conditioning.

## License

This tooling is released by the author. The Krea 2 models it operates on are covered by the **Krea 2
Community License** (see [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw)); use of the models is
subject to that license.
