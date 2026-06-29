# Concept directions — measure and dial any concept axis

Last updated: 2026-06-29

A targeted generalization of the [projector-rebalance lever](findings.md). The rebalance lever scales whole
*layers* of the conditioning; this aims at a **single measured concept direction** and amplifies, injects, or
removes it. It works on any axis the text encoder already represents — expression, style, lighting, pose, a
material attribute.

Three pieces (the nodes are under `conditioning/Krea2`):

1. **Krea 2 Concept Direction** node — build a direction **in-graph** from two prompts (the turnkey path: no
   terminal, no file). Outputs a `KREA2_CONCEPT_DIR`.
2. **Krea 2 Concept Inject** node — apply a direction to the conditioning. Takes the builder's `direction`
   output, *or* a saved `.npy` path.
3. **`scripts/concept_direction.py`** — the CLI version of the builder, for batch-building a reusable library
   of `.npy` directions offline.

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
| `amplify` | `cond += scale · (cond·d̂) d̂` — boost the component **already present**. `scale` 1.0 doubles it (×2), aimed at one axis. |
| `add` | `cond += scale · d̂` — inject the direction regardless of what's present. |
| `subtract` | `cond -= scale · d̂` |
| `project_out` | `cond -= (cond·d̂) d̂` — remove the concept entirely. |

**`amplify` scales the *present* component — but it is NOT a reliable "is the concept present?" test.** It
adds `(1+scale)·(cond·d̂)·d̂`, so at low scale it mostly intensifies what's already there. But a real prompt's
projection `cond·d̂` is essentially never exactly zero, and a measured direction carries whatever co-varied in
its A/B pair — so at higher scale `amplify` blows a small residual up into the concept. Measured: amplifying a
"smile" direction on a *no-face landscape* grows a full laughing face by scale ~4 (see
[`findings.md`](findings.md), "Labeled-axis steering"). Treat `amplify` as a magnitude lever that *can*
conjure, not a presence gate; and expect it to drag the axis's co-varied attributes (e.g. expression → close-
up framing) — tighter, attribute-only A/B prompts reduce that drag but cost some steering strength.

## Usage

### In ComfyUI (turnkey — no terminal)

Encode the concept's present/absent prompts with two `CLIPTextEncode` nodes, build the direction, and inject it
into your prompt's conditioning:

```
CLIPTextEncode("...a wide joyful smile") ─┐
                                          ├─► Krea 2 Concept Direction ──┐
CLIPTextEncode("...a neutral expression")─┘                            ▼ (direction)
CLIPTextEncode(your prompt) ───────────────────► Krea 2 Concept Inject(mode, scale) ─► KSampler
```

Pick a `mode` and dial `scale` (`amplify` 1.0 doubles the present component (×2); negative pushes the other way). Restart ComfyUI once
to load the nodes. The bundled
[`example_workflows/krea2_concept_inject.json`](../example_workflows/krea2_concept_inject.json) is exactly this
wiring with the benign `smile` example — open it and run. Set the builder's optional `save_path` to also write
the `.npy` and keep the axis for reuse.

### Offline (build a library of `.npy` directions)

For many concepts at once, the CLI is the batch version of the builder. Write an A/B prompt set
([`examples/concept_directions.json`](../examples/concept_directions.json) is a benign starter — `smile`,
`ornate`, `golden_hour`):

```json
{
  "smile": {
    "pos": ["a close-up portrait photograph of a person with a wide, joyful, open-mouthed smile"],
    "neg": ["a close-up portrait photograph of a person with a flat, neutral expression"]
  }
}
```

Multiple prompts per side average out incidental content and sharpen the axis. Then measure (CPU; loads the
Krea 2 CLIP via `scripts/krea2_clip.py`):

```bash
uv run --active python scripts/concept_direction.py examples/concept_directions.json --out <comfyui_models>/concept_dirs
```

This writes one `<name>.npy` (shape `12×2560`) per concept (`--text-encoder` overrides the CLIP file). In the
**Krea 2 Concept Inject** node, leave `direction` unconnected and set `direction_path` to the `.npy`.

The math (`apply_direction`, `pooled_direction`, `_concept_direction`) is numpy/torch-agnostic and unit-tested
(`tests/test_concept_inject.py`, `tests/test_concept_direction.py`); only the encode step needs comfy.

## Relationship to the other levers

- **Projector rebalance** (weights) scales whole layers; this aims at one measured direction in the
  activations. `amplify` is the same deep-band magnitude move, but targeted to a single axis.
- **Prompt-side `<think>` steering** (see [findings.md](findings.md)) is the cheapest steering vector when the
  axis is reachable from wording; a measured concept direction is the activation-space version for when it
  isn't.

The direction is a difference-of-means probe, so quality is bounded by how cleanly your A/B prompts isolate the
concept and by the handful of prompts you average. Treat it as a measured lever, not a guarantee.
