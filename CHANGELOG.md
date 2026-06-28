# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `safetensors_patch`: header parse, single-tensor read, in-place byte patch, and a small-file writer —
  edits one tensor in a multi-GB checkpoint without loading it.
- `projector`: `read_projector` / `scale_projector` for the learned `txtfusion.projector` (the
  `Linear(12 -> 1)` combiner over the 12 selected Qwen3-VL layers / "multilayer feature aggregation").
- `projector_lora`: emit tiny `txtfusion.projector.diff` LoRAs for arbitrary per-layer gains
  (`diff = orig*(gain-1)`, exact at strength 1); `make_band_isolation_loras` for 12 single-layer probes;
  presets `uniform` + `custom`. Loads via the stock `LoraLoaderModelOnly` (no custom node).
- `comfy_nodes`: a ComfyUI node ("Krea 2 Projector Rebalance") that reweights the projector live via the
  ModelPatcher (preset / strength / custom gains / `solo_band`).
- `cli` (`krea2-proj`): `inspect | lora | solo`.
- `attention_stats` (pure-numpy head/token averaging + hub-strength ranking) and
  `scripts/extract_attention.py`: load the Krea2 CLIP + the DiT's `txtfusion` weights (CPU) and recompute
  the 12x12 layer-fusion attention maps (head-averaged, per-head, cross-prompt).
- `image_grid`: reusable labeled contact-sheet builder (`build_contact_sheet`) for comparison figures
  (rows x cols of image paths / `PIL.Image` / `None`; missing cells render as placeholders). The
  experiment validators share it instead of re-inlining PIL.
- `rope_untwist` / `krea2_untwist_attn` / `krea2_untwist_node`: a ComfyUI node
  ("Krea 2 Untwist Style Reference") for training-free reference-image style transfer via untwisting-RoPE
  shared attention, built for Krea2's real `[32,48,48]` RoPE (not the community fork's `[64,64]`). Patches
  the image DiT blocks only (txtfusion untouched); renoise-to-sigma reference injection (no RF-inversion).
- `divdist_graph`: pure ComfyUI API-graph builders for diversity-distillation arms (`build_single` /
  `build_split` k=1 base->distilled handoff / `build_rescale` denoise sigma-rescale), realizing base-vs-
  distilled as one RAW load + Turbo-LoRA strength; plus `api_to_ui` so the importable R/A/B workflow is
  derived from the same builders (no drift). `keep_system` routes the `<think>` axis through
  `Krea2EncodeKeepSystem`. Example workflow: `example_workflows/krea2_diversity_distillation_rab.json`.
- `diversity`: dependency-light pairwise perceptual-diversity aggregation (`pairwise_diversity` /
  `diversity_table`, metric injected) following the average-pairwise-DreamSim protocol.

### Findings
- Interpretability of the layer aggregation: a universal mid-layer attention hub (L20) and a contrastive
  projector (positive on mid layers, negative on deep layers). See README / `docs/findings.md`.
- L20 hub validated content-token-masked across 5 prompts (~91–95% of content tokens) — content-driven,
  not a padding artifact — and shown to be a learned *directional* hub, not a magnitude sink.
- Confirmed the text encoder is stock, frozen `Qwen/Qwen3-VL-4B-Instruct` (loading code + config identity);
  all learned aggregation is DiT-side.
- Difference-of-means probe: benign attributes (expression / wet / blush) survive the learned aggregation
  *better* than ordinary content controls — i.e. the projector/fusion is not where such attributes are
  suppressed.
- Prompt-side steering: an in-distribution `<think>` reasoning span (appended to the assistant turn via the
  tokenizer's skip-template route) restores Turbo's distillation-flattened expression as well as or better
  than the deep-band rebalance lever, with prompt adherence intact — i.e. it acts as a steering vector
  (~17–24% conditioning shift, 0.86 direction consistency, energy at the L20/L23 hub). Low–medium confidence
  (one subject, few seeds, visual read).
- Verified (runtime, against Comfy's actual Krea 2 tokenizer): special tokens are tokenized per the model
  config (`<think>`/`</think>`/`<|im_start|>` → single ids 151667/151668/151644, not literal text), and
  `Krea2TEModel.encode_token_weights` **strips the system turn** (slices conditioning from the user turn
  onward). Consequence: the directly-steerable write-points are the user turn and the assistant `<think>`
  turn; a system-turn prompt only influences conditioning indirectly (via attention), not as injected tokens.
- Diversity distillation (arXiv:2503.10637) does **not** transfer to Krea2's flow schedule. Turbo's same-seed
  diversity collapse reproduces, but the paper's k=1 base-first-step fix is a null (the `firststep` and `k`
  sweeps change nothing); diversity is instead recovered by lowering the *global* Turbo-LoRA strength (a clean
  diversity↔quality dial, crossover ~0.5), i.e. the bottleneck is distributed across all steps, not localized
  at the first. Two prompts, 4 seeds, visual read. Measure-first: the per-step-schedule node was not built.

### Documentation
- Published `docs/findings.md`, `docs/figures/` (attention maps), `docs/data/` (numeric arrays), and
  `examples/test_prompts/` (reverse-caption prompts). Raw renders + internal notes stay local/gitignored.

## [0.1.0]

### Added
- Initial uv project scaffold.
