# Findings TLDR — Krea2 selected-layer probes

Last updated: 2026-06-26

Basis: solo (keep-one) + leave-one-out (drop-one) sweeps on **one prompt** (portrait), seed 42,
Krea2 **Turbo fp8**, 8 steps euler/simple, image-level **RGB-RMS** distance. Cross-style sweeps
(anime, illustration) are done — see the Cross-style update below.

| # | Finding | Confidence | Why / caveat |
|---|---------|-----------|--------------|
| 1 | **Deep selected layers carry the renderable content; shallow are scaffolding.** Solo L23/L26/L29/L32 each render a coherent portrait alone; L2–L11, L17, L20 alone → noise. | **High** (this prompt) / Medium (general) | Agrees with the learned projector (deep layers have the largest \|weights\|) and the tech report ("final layer optimized for next-token prediction, not image gen"). Generality pending cross-style. |
| 2 | **L14 uniquely carries text/typography.** Solo L14 → text glyphs, despite a prompt with no text. | **Medium-High** | Striking, clean single-layer signal, but one prompt and the prompt had no text. Needs a text-containing prompt + cross-style to confirm. The most surprising solo probe, but modest. |
| 3 | **The final layer (L35) alone is unusable for image gen** (noise). | **High** | Directly confirms the tech report's rationale for multilayer aggregation. |
| 4 | **Necessity ≠ sufficiency: deep layers are partly redundant.** Drop-one keeps everything coherent (11 layers remain). L29 most necessary (Δ=0.35), then L32 (0.25), L23 (0.20); **L26 is sufficient-alone but low drop-importance (0.15) → redundant.** | **Medium** | Ranking from a coarse RGB-RMS metric, one prompt/seed. Direction (deep > shallow) trustworthy; exact order (e.g. L29 vs L32) low confidence. |
| 5 | **The model's learned aggregation agrees with the ablations.** Largest-\|weight\| layers (L23, L29, L32, L26) are the ones that render alone and/or matter most when dropped. | **Medium-High** | Two independent signals cross-check (learned projector vs causal ablation). |

## Confidence summary

- **High:** deep-carry-content / shallow-scaffold / final-layer-unusable — architecturally grounded + visually unambiguous.
- **Medium:** L14 = text; redundancy among deep layers.
- **Low:** exact importance ordering (coarse metric, n=1 prompt/seed).
- **Pending:** cross-style generality (running); precise numbers (conditioning-space leave-one-out + gain-Jacobian via the `txtfusion` extractor, no generation needed).

## Caveats bounding all of the above

Single prompt, single seed, Turbo **fp8** (quantized), **image-level** RGB-RMS distance (not perceptual,
not conditioning-space). These bound confidence; the extractor + cross-style sweeps are how we tighten it.

## Cross-style update (2026-06-26)

Ran solo + leave-one-out on **anime** and **illustration** too. The pattern **held across all 3 styles**:
deep layers (L23/26/29/32) render alone, shallow = noise, and the leave-one-out ranking is consistent
(L29 top everywhere; cluster {L29, L23, L32} then L26/L14; shallow lowest). L14 **refined**: it carries
**structure/layout** (text glyphs in the portrait, road/scene structure in the illustration), not purely
"text". Cross-style consistency moves findings 1 & 3 toward **High**.

## Is any of this novel? (honest calibration)

Mostly **expected**, and worth saying plainly:
- "deep = semantic, shallow = lexical/scaffold" is standard LLM-layer interpretability.
- "final layer unusable for image-gen" is *literally Krea's stated reason* for multilayer aggregation —
  confirmatory, not a discovery.
- redundancy across adjacent layers is normal.

So the solo/LOO work is **verification + reproducible artifacts + the (mild) L14/structure observation** —
not a discovery. The genuinely model-specific questions it raised — now **addressed** below:
1. **The combination mechanism** — answered: the layer-fusion routes through an **L20 directional hub**
   (see "Layer-fusion attention"). ✓
2. **Attribute-level** — answered: benign attributes (expression / "wet" / blush) **survive the aggregation**,
   and the projector/fusion is *not* where they're gated (see "Attributes vs the projector-rebalance lever"). ✓
3. **Where the bias lives** — answered: the encoder is **frozen stock `Qwen/Qwen3-VL-4B-Instruct`** (Krea's
   loading code does `from_pretrained` + `.eval().requires_grad_(False)`; its config is field-for-field
   identical to stock), so all learned aggregation is **DiT-side**. ✓

## Layer-fusion attention — measured behavior (2026-06-26)

Built the `txtfusion` extractor (CPU; loads the Krea2 CLIP for the 12 hidden states + the DiT's txtfusion
weights, recomputes the layerwise attention).

**What's documented vs measured (honest framing):** the *architecture* — cross-layer attention over the 12
layers, then a `Linear(12→1)` projector — is public (diffusers' `transformer_krea2.py` docstring: "the
layerwise_blocks attend across the num_text_layers axis"). We did **not** discover that. What's measured
below is the *learned behavior* of that attention (which layer it concentrates on, the sign pattern of the
learned projector) — emergent properties no source states. This is characterization of an open model, not
an architecture reveal:

1. **L20 is a universal attention hub.** In layerwise block 1, nearly every selected layer attends to L20;
   column-strength L20 = **0.24 / 0.27 / 0.25** across portrait / anime / illustration (~2x the next, L23),
   with a near-identical ranking. Cross-style consistent → **High confidence**, a prompt-independent
   architectural property. Per-head: the hub is **broad** (most of the 20 heads route to L20), with a
   minority specializing on L14 / L23-29 / L35 / L8.
   - **Validated (2026-06-26):** also held on 2 long dense prompts (mushroom, geisha). **Content-token-masked**
     (dropping the 34-token template prefix + suffix), L20 is the top key-layer for **91–95% of content
     tokens** → content-driven, **not** a padding/template artifact. 5 prompts total now.
   - **Mechanism: a learned *directional* hub, not a magnitude sink.** L20's hidden-state norm is mid-pack
     (rank 6/12) and its raw (pre-norm) key norm is near-*lowest* (rank 10/12); the block's `qknorm`
     equalizes every layer's key magnitude. So routing is decided by learned query/key **direction** (trained
     `wq`/`wk` point most queries at L20's key direction), not magnitude — and not a hardcoded index (the
     layerwise blocks have no positional encoding). Encoder hidden-state norm grows ~48x L2→L35 (11→555),
     which may be *why* the projector down-weights deep layers.
2. **The projector is contrastive, not an average.** Learned 12->1 weights are positive on mid layers
   (L8-L20, peak L14 +0.71) and strongly negative on deep layers (L23 -1.44, L29 -0.89, L32 -0.61). Fixed
   weights → prompt-independent. The final text vector ≈ "mid minus deep". **High confidence.**

Caveat: the projector's signed weights act on the attention-**mixed** slots (post block 0/1), not raw layers.

**Refiner blocks (mapped) + RAW check (2026-06-26):** the 2 refiner blocks (token attention, post-projector)
are **diffuse** — normalized entropy ~0.95, peak ~6× uniform, no sink — so the striking structure is in the
layer-fusion, not the token-refinement. And the whole `txtfusion` is **checkpoint-agnostic**: RAW vs Turbo
projector weights are identical (cosine 1.0, 12/12 signs) and the L20 hub holds on RAW (92–95% of content
tokens). So the L20 hub + contrastive projector are "Krea 2" findings, not "Turbo"-specific.
Data: `data/raw_validation/raw_vs_turbo.json`.

## Attributes vs the projector-rebalance lever (2026-06-26)

Difference-of-means + causal tests on benign attributes (expression / "wet" / blush):
- **Conditioning (net-effect through `txtfusion`):** these attributes come through the learned aggregation
  *more* strongly than ordinary content controls (mean relative footprint 0.14 vs 0.04; A=out/in 1.2 vs 0.46).
- **Causal (with/without in the prompt):** each attribute renders clearly on stock Turbo.
- **Stock vs rebalanced (deep-boost projector LoRA, ×1 and ×4):** the attributes render either way. Boosting
  the deep layers mainly shifts **detail / contrast / intensity** — consistent with the deep layers carrying
  fine detail (see Prior work) — rather than changing whether an attribute appears.

So for these benign attributes, presence doesn't hinge on the projector/fusion stage; the rebalance lever
behaves as a detail/intensity knob. Caveats: a few prompts/seeds, controls not magnitude-matched, benign
attributes only (not near-safety-boundary cases), visual + correlational rather than a hard metric.
Data in `data/attribute_directions/` (probe + causal + stock-vs-rebalanced grids).

## Projector-LoRA A/B — result (2026-06-26)

The community hand-rebalances the projector; the trainers can *learn* it (diffusers + musubi target
`text_fusion.projector` by default; ai-toolkit excludes it). Tested whether **training** the 12→1 layer-mix
changes a LoRA: two identical Krea-2-Raw DreamBooth LoRAs (QLoRA/NF4, 5 imgs, 300 steps, rank 16) differing
*only* in whether `text_fusion.projector` is a LoRA target; compared on held-out prompts (RAW, fresh seed).

**Result: at this scale, training the projector made no meaningful difference.** Both arms learned the
subject comparably and neither distorted an unrelated control prompt; differences were within one-seed noise
(a faint hint the with-projector arm kept the scene slightly more faithful on the "beach" prompt, but weak).
So for a small *subject* LoRA, including the projector (diffusers/musubi default) vs freezing it (ai-toolkit)
doesn't much matter — consistent with the projector being a modest lever and the diffusers README's advice to
narrow to attention layers for long runs.

Caveats: quick proof — tiny dataset, 300 steps, rank 16, one validation seed, a *subject* (DreamBooth) LoRA
not a *style* LoRA (where the semantic-depth layer-mix might matter more), NF4-quantized. Doesn't rule out the
projector mattering in a longer or style-focused run. Data: `data/projector_ab/`.
