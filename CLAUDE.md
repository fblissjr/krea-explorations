# CLAUDE.md — krea2-explorations

Last updated: 2026-06-27

Project memory for this repo. Global conventions (uv, TDD, path-privacy, docs, no emojis) are in the
user-level CLAUDE.md and still apply; this file holds only what's specific to this repo.

## What this is

Tools for measuring and surgically editing Krea 2's text conditioning (per-layer projector edits,
checkpoint analysis, attention extraction) plus the experiment harnesses that test them. Measure-first:
lead with falsifications, not "discoveries".

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
