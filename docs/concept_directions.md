# Concept directions — measure and dial any concept axis

Last updated: 2026-06-28

A targeted generalization of the [projector-rebalance lever](findings.md). The rebalance lever scales whole
*layers* of the conditioning; this aims at a **single measured concept direction** and amplifies, injects, or
removes it. It works on any axis the text encoder already represents — expression, style, lighting, pose, a
material attribute.

Two pieces:

1. **`scripts/concept_direction.py`** — measure a direction from an A/B prompt pair (offline, CPU).
2. **The *Krea 2 Concept Inject* node** (`conditioning/Krea2`) — apply that direction to the conditioning at
   generation time.

## How it works

Krea 2's conditioning is the 12 selected encoder layers fused to `(B, seq, 12*2560)`. A *concept direction* is
the **difference of means** of that conditioning between a group of "positive" prompts (concept present) and
"negative" prompts (concept absent):

```
d = mean(encode(positives)) - mean(encode(negatives))     # shape (12, 2560)
```

The node applies `d` (unit-normalized as `d̂`) per token, in one of four modes:

| mode | effect |
|------|--------|
| `amplify` | `cond += scale · (cond·d̂) d̂` — boost the component **already present**. `scale` 1.0 ≈ the bypass LoRA's ×2, but aimed at one axis. |
| `add` | `cond += scale · d̂` — inject the direction regardless of what's present. |
| `subtract` | `cond -= scale · d̂` |
| `project_out` | `cond -= (cond·d̂) d̂` — remove the concept entirely. |

**Key property (and a built-in sanity check):** `amplify` only scales what is *already* in the conditioning, so
it mathematically **cannot conjure an absent concept** — if the axis isn't represented for a prompt, amplify at
any scale leaves it unchanged. That makes the node double as a test of whether a concept is present in the
aggregation at all.

## Usage

1. Write an A/B prompt set. [`examples/concept_directions.json`](../examples/concept_directions.json) is a
   benign starter (`smile`, `ornate`, `golden_hour`):

   ```json
   {
     "smile": {
       "pos": ["a close-up portrait photograph of a person with a wide, joyful, open-mouthed smile"],
       "neg": ["a close-up portrait photograph of a person with a flat, neutral expression"]
     }
   }
   ```

   Multiple prompts per side average out incidental content and sharpen the axis.

2. Measure the directions (CPU; loads the Krea 2 CLIP via `scripts/krea2_clip.py`):

   ```bash
   uv run --active python scripts/concept_direction.py examples/concept_directions.json --out <comfyui_models>/concept_dirs
   ```

   This writes one `<name>.npy` (shape `12×2560`) per concept. `--text-encoder` overrides the CLIP file.

3. In ComfyUI, drop the **Krea 2 Concept Inject** node between your text-encode and the sampler, set
   `direction_path` to the `.npy`, pick a `mode`, and dial `scale` (`amplify` 1.0 ≈ bypass ×2; negative pushes
   the other way). Restart ComfyUI once to load the node.

The math (`apply_direction`, `pooled_direction`) is numpy/torch-agnostic and unit-tested
(`tests/test_concept_inject.py`, `tests/test_concept_direction.py`); only the encode step needs comfy.

## Relationship to the other levers

- **Projector rebalance** (weights) scales whole layers; this aims at one measured direction in the
  activations. `amplify` is the same magnitude move the bypass LoRA makes, but targeted to a single axis.
- **Prompt-side `<think>` steering** (see [findings.md](findings.md)) is the cheapest steering vector when the
  axis is reachable from wording; a measured concept direction is the activation-space version for when it
  isn't.

The direction is a difference-of-means probe, so quality is bounded by how cleanly your A/B prompts isolate the
concept and by the handful of prompts you average. Treat it as a measured lever, not a guarantee.
