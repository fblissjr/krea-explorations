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

### Documentation
- Published `docs/findings.md`, `docs/figures/` (attention maps), `docs/data/` (numeric arrays), and
  `examples/test_prompts/` (reverse-caption prompts). Raw renders + internal notes stay local/gitignored.

## [0.1.0]

### Added
- Initial uv project scaffold.
