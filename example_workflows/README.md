# Example workflows — exploring Krea 2's layer aggregation

Last updated: 2026-06-26

Minimal, deterministic Krea 2 Turbo text-to-image graphs for *exploring* how the model combines its 12
selected encoder layers. Prompt-enhancement is intentionally stripped and the seed is fixed, so any change
in output is attributable to the projector reweighting you apply.

## Files

- `krea2_turbo_t2i.json` — a plain Krea 2 Turbo t2i graph (UNETLoader → CLIPLoader[krea2] → KSampler →
  VAEDecode), fixed seed, no prompt-enhancement. A clean base to build on.
- `krea2_turbo_solo_explorer.json` — the base graph plus a **LoraLoaderModelOnly** loading one single-layer
  probe (`projector_solo_b08_L26.safetensors`) at strength 1, so the image is driven by one selected layer.
  Swap the `lora_name` to sweep the 12. (Generate the probes first: `krea2-proj solo` — see the repo README.)

## How to explore per-layer contribution

Add a **LoraLoaderModelOnly** between the model loader and the sampler and load one of the emitted
`.diff` files (see the repo README — `krea2-proj lora` / `solo`):

- a **`solo/` LoRA at strength 1** isolates one selected layer (zeros the other 11), so the image is
  driven by a single layer's contribution — sweep the 12 to compare;
- a **custom-gain LoRA** reweights all 12 layers at once.

Or add the **Krea 2 Projector Rebalance** node (from this repo) and set `solo_band` / `per_layer_weights`
to do the same live, without a file.

Keep the seed fixed and the prompt constant across runs so differences reflect the layer reweighting, not
sampling noise. For a stronger read, repeat at a couple of fixed seeds.
