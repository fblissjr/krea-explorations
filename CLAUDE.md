# CLAUDE.md — krea2-explorations

Last updated: 2026-06-30

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

Keep all public examples benign (generic prompts/axes — expression, style, pose); anything sensitive stays in
gitignored `internal/`. Don't let README/docs drift behind the code.

## Comparison grids / figures — use the shared util

Experiment validators produce comparison contact sheets (rows = prompts/variants, cols = arms/methods).
There is ONE tested implementation — **do not re-inline PIL grid code**:

```python
from krea2_explorations.image_grid import build_contact_sheet
build_contact_sheet(grid_rows, out_path, col_labels=[...], row_labels=[...])
```

`grid_rows` is a 2D list (rows x cols) of image path / `PIL.Image` / `None`; missing cells render as a
placeholder instead of crashing. It depends only on Pillow + stdlib. Tests: `tests/test_image_grid.py`.

## Building generation graphs — reuse the built workflows, don't re-inline

NEVER hand-inline a graph dict, and NEVER copy an older harness's `graph()` as a template — a re-inlined copy
silently drifts from the *measured* recipe (this has bitten us: soft renders, a stale `ModelSamplingFlux`, the
wrong scheduler). Two already-built sources, both no-`ModelSamplingFlux`:

- **The A/B/C/D recipes are THE source of truth**, built in `scripts/canonical_workflows.py` (now TRACKED):
  `A` (drop-in), `B` (cfg-headroom), `C` (the two-sampler Turbo-LoRA split), `D` (SDE finish), plus
  `build_single` / `build_split`. They encode the canonical decision — the **modular `SamplerCustomAdvanced`
  stack, `beta57` scheduler, bf16 qwen3vl encoder, fp8 RAW DiT, krea2RealVae** (filenames single-sourced from
  `generate.py`). Call A/B/C/D for ANY canonical render (internal or public; `use_res=False` for a fixed 1024²
  latent — `Krea2Resolution` needs a ComfyUI restart to load).
- **`scripts/generate.py` is the low-level layer beneath** the recipes: `build_graph` (single-`KSampler`
  primitive), `run`, `resolve_vae`, `model_node`, the model-filename constants — and `build_split_graph`, a
  low-level two-stage split primitive (two `KSamplerAdvanced`, `scheduler=simple`) for harnesses that wire their
  own conditioning, NOT the canonical workflow C. Documented in `docs/krea2_inference.md`.
- **A custom node composes onto a build_graph seam, it doesn't re-inline.** A harness adding its own node hangs
  it on one of two seams instead of copying the skeleton: `model_patches=[model_node("Krea2AttnBias", ...)]` for
  a model-edge lever (attention bias, residual steer, DiT capture), or `cond_patches=[cond_node("Krea2ConceptInject", ...)]`
  for a positive-conditioning lever (concept inject / project-out). Both keep the canonical skeleton and only add
  your node; the negative is left untouched.

Traps a re-inline (or copying an older harness's `graph()`) gets wrong:
- **No `ModelSamplingFlux`** — the 1.15 flow-shift is in Krea2's model config, so the node is a pixel-identical
  no-op; it must NOT be in the graph.
- **VAE = krea2RealVae** (crisper skin/texture), never hardcode stock `qwen_image_vae` (`resolve_vae` falls back
  automatically when it's absent).
- **Scheduler `beta57`** in the canonical stack (not `simple`); a real (empty) negative, never
  `ConditioningZeroOut`.

These rules are enforced, not just documented: `tests/test_no_reinlined_graphs.py` fails if any tracked
`scripts/` builder re-inlines a `ModelSamplingFlux` / `ConditioningZeroOut` node or hardcodes the stock VAE.

For training data or any quality-sensitive render use the split (`C`) or the Turbo-LoRA dial — not bare RAW-28 +
stock VAE.

## Workflows — every run saves its graph; promote to public deliberately

ComfyUI graphs here are built in code (`internal/scripts/*.py` harnesses) and POSTed to the server — the
harness is the source of truth, but a graph that's only ever POSTed leaves no reproducible artifact. Convention
so every run is recoverable and loadable:

There is ONE serializer (`scripts/workflow_dump.py` → `dump_workflow`) and TWO ways to feed it — use the right
one, don't mix:
- **Run-instance dumps: pass `harness=/arm=/seed=/prompt=` to `generate.run()`** — it auto-dumps the graph
  *before* the POST (so even a failed render leaves an artifact). This is the single chokepoint; do NOT also
  call `dump_workflow` yourself in the harness (that's the redundant path — one or the other, not both).
- **Reference/canonical workflows: each harness owns them** via a `reference_workflows() -> {stable_name: graph}`
  function; `internal/scripts/export_workflows.py` is a thin discovery loop that calls it on each harness and
  dumps with `stable_name=` (no datetime). Do NOT hardcode a central per-experiment list in `export_workflows`
  reaching into each harness's internals — that's a monolith that breaks on any `graph()` signature change.
- Dumps land in gitignored `internal/workflows/` (graphs can embed sensitive prompts — never a tracked dir).
- **Per-run dumps carry a sortable UTC datetime** in the name:
  `<harness>_<arm>_s<seed>_<YYYYMMDDTHHMMSSZ>.json`. **Do NOT rely on mtime** — it resets on copy/git/rsync,
  exactly when you need provenance. Provenance (`ts_utc, harness, arm, seed, git_sha, prompt`) goes in the
  `.meta.json` sidecar, **not** a top-level key inside the API JSON (ComfyUI treats an extra top-level key as a
  malformed node and fails to load it).
- **Reference/canonical workflows** (one definitive graph per experiment type) keep STABLE names, no datetime
  (e.g. `10_dit_capture_stock.json`) — overwrite on harness change; git is the version history.
- **Promotion path:** `internal/workflows/` (raw, may be sensitive) → validate + make benign (generic
  prompts/axes) → `example_workflows/` (public, UI-native — the front door). ONLY benign, validated workflows
  graduate to `example_workflows/`.
- Loadable via ComfyUI **"Load (API format)"** or re-POST as `{"prompt": <json>}`. A dumped JSON is a single
  arm; re-running the harness reproduces the full multi-arm grid.

## Environment — venvs, hardware, tooling (this bites)

- **Project `.venv`** (uv, `uv run ...`): the importable `krea2_explorations` package + its tests.
- **A separate, isolated training venv** (CUDA torch + editable diffusers w/ `Krea2Pipeline`, peft,
  bitsandbytes): used only by the LoRA training/validation harnesses in `internal/training/`. It does NOT
  have the project package installed — those scripts import shared code (e.g. the grid util) by adding
  `<repo>/src` to `sys.path`. Keep shared helpers Pillow/stdlib-only so they import there.
- **`uv run` (project `.venv`) vs `uv run --active` (the active ComfyUI `.venv`).** Plain `uv run` uses the
  project venv — right for the package + tests, but it lacks torch/transformers/comfy, so the **inference +
  model-loading harnesses in `internal/scripts/`** (render harnesses, `score_caption_register.py`,
  `exp_ood_register.py`) need **`uv run --active`** to pick up ComfyUI's `.venv`. The `VIRTUAL_ENV=… does not
  match … will be ignored` warning is harmless under plain `uv run` (you wanted the project venv) but is the
  tell you forgot `--active` on a model-loading run. Pipe through `grep -v VIRTUAL_ENV` to quieten.
- **Hardware: one 24 GB RTX 4090.** A 12B bf16 DiT (~24 GB) won't fit with headroom → the canonical DiT is
  **fp8**; the bf16 qwen3vl encoder is ~free (offloaded before the DiT sampling pass). Don't propose a bf16 DiT.
  What's installed: `curl <server>/object_info/{UNETLoader,VAELoader,CLIPLoader}` lists the models/encoders/VAEs
  (`resolve_vae`/`resolve_clip` do this at run time); model files are symlinks into a storage dir, so
  `ls models/...` shows link targets, not byte sizes.
- **`grep -r --include='*.py' .` has returned empty when it shouldn't** (hit twice this session, once leading to
  a wrong "function moved" conclusion) — prefer naming the dirs (`grep -rn X scripts internal/scripts`) or
  `git grep` (tracked-only; also the right tool for path-leak / privacy scans before committing).

## Knowledge management — the `internal/` wiki (READ `internal/findings/INDEX.md` first)

`internal/` is a **two-layer knowledge wiki**; keep the layers separate:
- **Living synthesis** (`internal/reference/krea2_master_synthesis.md`) — the *current understanding*. REWRITE it
  as understanding changes. The single entry point for "what we believe now."
- **Evidence ledger** (`internal/findings/`, mapped by `findings/INDEX.md`) — dated, **append-only** records of
  the *measurements*. A measurement never goes stale; never rewrite or delete one.

When you run an experiment / add knowledge / overturn an assumption:
1. **Record** the measurement in the right `findings/` page (numbers/method/date verbatim, under its `Status:` line).
2. **Rewrite** the interpretation in the master synthesis if a conclusion changed.
3. If a finding is superseded, **add a forward-pointer** to its record — keep the original measurement beside it.
4. **Update `findings/INDEX.md`** (one row per page).
5. **Archive** (`findings/archive/`) ONLY disposable scaffolding (pre-run plans/specs) — NEVER a measured finding.

Rule of thumb: **interpretation is rewritten · evidence is appended · measurements are never deleted.** Full
protocol + the page index: `internal/findings/INDEX.md`.

## Experiment harnesses & logs

- `internal/` is gitignored — experiment scripts, pre-registrations, training notes, session logs, and any
  local/home paths live there (never in committed files).
- **The whole training pipeline lives in `internal/training/`, together** — data generation
  (`gen_*_train_set.py`), the train runner (`run_*.sh`), the validate/eval harness, and the `*_prereg.md`.
  Training code/config always goes here; don't scatter it into `internal/scripts/`. The image *set* goes in
  `data/train_data/<name>/` and stays **images-only** (the DreamBooth trainer globs it) — document the set in
  `data/train_data/README.md`, never inside the set dir.
- LoRA training is GPU-gated: a 12B NF4 QLoRA run needs most of a 24GB card, so ComfyUI must be stopped
  first. Training-free inference (untwisting-RoPE) runs *inside* ComfyUI instead — opposite GPU needs.
- Pre-register predictions before a run (see `internal/training/*_prereg.md`) so grids are interpretable.
