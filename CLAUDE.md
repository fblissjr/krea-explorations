# CLAUDE.md — krea2-explorations

Last updated: 2026-06-28

Project memory for this repo. Global conventions (uv, TDD, path-privacy, docs, no emojis) are in the
user-level CLAUDE.md and still apply; this file holds only what's specific to this repo.

## What this is

Tools for measuring and surgically editing Krea 2's text conditioning (per-layer projector edits,
checkpoint analysis, attention extraction) — and its *sampling* (diversity-distillation arms) — plus the
experiment harnesses that test them. Measure-first: lead with falsifications, not "discoveries".

## Three lever classes (keep all in mind)

Three ways to steer Krea 2's output, all in scope here:

1. **Weight/activation edits** — the projector rebalance + single-layer isolation tooling (the core package:
   `projector`, `projector_lora`, `comfy_nodes`). Edits weights, stays in-distribution via downstream RMSNorm.
2. **Prompt-side steering** — a `<think>` block / system prompt / prefix written into the text the encoder
   sees. Inject custom spans via the **tokenizer skip-template route**: pass a full `<|im_start|>…` string as
   the prompt and the qwen3vl tokenizer emits it verbatim, so no ComfyUI/pipeline edit is needed. It behaves
   like a steering vector — push within-distribution and prompt adherence holds. See `docs/findings.md`
   ("Prompt-side steering").
3. **Sampler-side (schedule / LoRA-strength)** — the diversity-distillation arms in `divdist_graph`
   (`build_single` / `build_split` k=1 base→distilled handoff / `build_rescale` denoise sigma-rescale).
   Realize base↔distilled as ONE RAW load + Turbo-LoRA strength (0 = base, 1 = distilled); pure dict builders,
   tested, driven by `scripts/generate.run`. **Workflows are derived from the builders** via `api_to_ui` —
   don't hand-author UI JSON. Finding (`internal/training/diversity_distillation_prereg.md`): the paper's
   (arXiv:2503.10637) k=1 first-step fix is a *null* on Krea 2's flow schedule; the diversity↔quality lever
   that works is *global* LoRA strength (crossover ~0.5).

Public/tracked files stay benign in name and content; sensitive prompts, data, and any
sensitive-referencing filenames live only in gitignored `internal/` and `data/`.

## Comparison grids / figures — use the shared util

Experiment validators produce comparison contact sheets (rows = prompts/variants, cols = arms/methods).
There is ONE tested implementation — **do not re-inline PIL grid code**:

```python
from krea2_explorations.image_grid import build_contact_sheet
build_contact_sheet(grid_rows, out_path, col_labels=[...], row_labels=[...])
```

`grid_rows` is a 2D list (rows x cols) of image path / `PIL.Image` / `None`; missing cells render as a
placeholder instead of crashing. It depends only on Pillow + stdlib. Tests: `tests/test_image_grid.py`.

Likewise for **pairwise diversity** (the average-pairwise-DreamSim protocol): one tested impl —
`krea2_explorations.diversity.pairwise_diversity` / `diversity_table` (metric injected, stdlib-only). Don't
re-inline the pairwise loop. Tests: `tests/test_diversity.py`.

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
