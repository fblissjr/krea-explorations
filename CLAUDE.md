# CLAUDE.md — krea2-explorations

Last updated: 2026-06-28

Project memory for this repo. Global conventions (uv, TDD, path-privacy, docs, no emojis) are in the
user-level CLAUDE.md and still apply; this file holds only what's specific to this repo.

## What this is

Tools for measuring and surgically editing Krea 2's text conditioning (per-layer projector edits,
checkpoint analysis, attention extraction) plus the experiment harnesses that test them. Measure-first:
lead with falsifications, not "discoveries".

## Two lever classes (keep both in mind)

Two ways to steer Krea 2's conditioning, both in scope here:

1. **Weight/activation edits** — the projector rebalance + single-layer isolation tooling (the core package:
   `projector`, `projector_lora`, `comfy_nodes`), plus the **concept-direction** nodes
   (`krea2_concept_inject_node`): *Krea 2 Concept Direction* measures a difference-of-means axis from an A/B
   prompt pair in-graph, *Krea 2 Concept Inject* amplifies/injects/project-outs it on the conditioning
   (`scripts/concept_direction.py` is the offline CLI that writes the same `.npy`). Edits weights or
   activations, stays in-distribution via downstream RMSNorm. Guide: `docs/concept_directions.md`;
   workflow: `example_workflows/krea2_concept_inject.json`.
2. **Prompt-side steering** — a `<think>` block / system prompt / prefix written into the text the encoder
   sees. Inject custom spans via the **tokenizer skip-template route**: pass a full `<|im_start|>…` string as
   the prompt and the qwen3vl tokenizer emits it verbatim, so no ComfyUI/pipeline edit is needed. It behaves
   like a steering vector — push within-distribution and prompt adherence holds. See `docs/findings.md`
   ("Prompt-side steering").

Public/tracked files stay benign in name and content; sensitive prompts, data, and any
sensitive-referencing filenames live only in gitignored `internal/` and `data/`.

## Public-facing docs — keep in sync (these are the front door)

When you add or change a **public tool** (a ComfyUI node, a `scripts/` CLI, or a package capability), update
the user-facing docs in the SAME change — they're what people actually use:
- `README.md`: the "With the toolkit you can" bullets, a short TL;DR section, and the **Components** table.
- `docs/`: the relevant guide (e.g. `docs/concept_directions.md`) — add one if none fits; cross-link
  `docs/findings.md`.
- Bump the `Last updated:` date on every doc you touch.

Keep all public examples benign (generic prompts/axes — expression, style, pose); the explicit applications
stay in gitignored `internal/`. Don't let README/docs drift behind the code.

## Comparison grids / figures — use the shared util

Experiment validators produce comparison contact sheets (rows = prompts/variants, cols = arms/methods).
There is ONE tested implementation — **do not re-inline PIL grid code**:

```python
from krea2_explorations.image_grid import build_contact_sheet
build_contact_sheet(grid_rows, out_path, col_labels=[...], row_labels=[...])
```

`grid_rows` is a 2D list (rows x cols) of image path / `PIL.Image` / `None`; missing cells render as a
placeholder instead of crashing. It depends only on Pillow + stdlib. Tests: `tests/test_image_grid.py`.

## Two virtualenvs (this bites)

- **Project `.venv`** (uv, `uv run ...`): the importable `krea2_explorations` package + its tests.
- **A separate, isolated training venv** (CUDA torch + editable diffusers w/ `Krea2Pipeline`, peft,
  bitsandbytes): used only by the LoRA training/validation harnesses in `internal/training/`. It does NOT
  have the project package installed — those scripts import shared code (e.g. the grid util) by adding
  `<repo>/src` to `sys.path`. Keep shared helpers Pillow/stdlib-only so they import there.

## Experiment harnesses & logs

- `internal/` is gitignored — experiment scripts, pre-registrations, training notes, session logs, and any
  local/home paths live there (never in committed files).
- LoRA training is GPU-gated: a 12B NF4 QLoRA run needs most of a 24GB card, so ComfyUI must be stopped
  first. Training-free inference (untwisting-RoPE) runs *inside* ComfyUI instead — opposite GPU needs.
- Pre-register predictions before a run (see `internal/training/*_prereg.md`) so grids are interpretable.
