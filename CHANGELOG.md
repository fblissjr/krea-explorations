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

### Findings
- Interpretability of the layer aggregation: a universal mid-layer attention hub (L20, cross-prompt and
  cross-head) and a contrastive projector (positive on mid layers, negative on deep layers). See README.

## [0.1.0]

### Added
- Initial uv project scaffold.
