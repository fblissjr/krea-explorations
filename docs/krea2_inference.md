Last updated: 2026-06-29

# How Krea 2 inference works

This explains how to run Krea 2 through ComfyUI with this toolkit: how the sampling graph is built, which sampler and resolution to choose, and how the helper code in `scripts/generate.py` fits together. Each design choice below was measured, not assumed.

Krea 2 is a flow-matching DiT. The text encoder is a frozen Qwen3-VL-4B. The image decoder is a Qwen-Image VAE with an 8x spatial factor.

## Two ways to build a graph

You build the graph in code and POST it to a running ComfyUI server. There are two shapes, and you pick by what you need to vary.

The simple shape uses one `KSampler` node. `build_graph()` in `scripts/generate.py` produces it. Use it for a single image or an A/B test where you change one thing and hold the rest fixed.

The modular shape splits sampling into separate nodes:

- `RandomNoise` provides the noise
- a guider provides the conditioning and CFG: `BasicGuider` when CFG is off, `CFGGuider` when CFG is on
- `KSamplerSelect` or `SamplerDPMPP_2M_SDE` provides the sampler
- `BasicScheduler` provides the sigmas
- `SamplerCustomAdvanced` runs them together

Use the modular shape when you want to swap the sampler, scheduler, sigmas or guider without rebuilding the graph. The canonical workflows use it.

The flow-match shift of 1.15 comes from Krea 2's model config, so the graph needs no separate shift node.

## Picking a resolution

The image width and height must both divide by 16. This comes from the VAE 8x spatial factor times the DiT patch size of 2. The token count is the image area divided by 16 in each dimension.

The Qwen3-VL model here is the text encoder. It does not see the image, so it does not constrain the image size. Only the VAE and the patch size do.

The `Krea2Resolution` node gives you the workable resolutions. You either:

- pick a preset bucket, all near 1 megapixel and all divisible by 16, from 1024 x 1024 through to 1536 x 640 and its rotations
- pass a custom width and height to snap to the nearest multiple of 16, with an option to first rescale to about 1 megapixel

The node outputs width and height to feed into `EmptyLatentImage`.

## Choosing a sampler

For a deterministic result, use `euler`. The same seed gives the same image every time, which is what you want for an A/B test.

For a sharper, more textured finish at 8 steps, use `dpmpp_2m_sde` with eta. It is an exponential integrator: it solves the stiff part of the flow exactly each step, so it holds more detail at low step counts.

You do not need RES4LYF for this. We compared the native `dpmpp_2m_sde` against RES4LYF `res_2s` and `res_3s` across a face, a macro detail shot and a busy scene. The RES4LYF samplers gave different compositions but no better quality, even on the macro shot where a higher-order method should win. So the toolkit uses native samplers and takes no dependency on RES4LYF.

A note on a wrong turn we corrected: an early comparison put `dpmpp_2m_sde` against ComfyUI's `ddim`, which is not an exponential integrator. That made the exponential method look unique to RES4LYF when it is not. Match the method class, not the sampler name, when you compare.

## What eta does

Eta controls how much noise the sampler re-injects each step before denoising it away. It is the stochastic, or SDE, part of the sampler.

The effect is close to a switch rather than a dial:

- at eta 0 the sampler is deterministic, the same as the plain ODE path
- any eta above 0 changes about 96% of the pixels and can re-roll the composition
- the effect then saturates: 0.25, 0.5, 0.75 and 1.0 look much the same as each other

At 8 steps, eta 1.0 leaves grain the steps cannot resolve. Eta around 0.5 is a clean default. Drop to 0 when you want the result to be reproducible.

## Choosing a VAE

Krea 2's stock VAE (`qwen_image_vae`) is tuned for legible text, not photoreal texture, so it looks soft on skin and fine detail. The toolkit defaults to krea2RealVae, a community decoder finetune that drops in through the stock VAE loader and is crisper on texture, and it falls back to the stock VAE automatically when krea2RealVae is not installed (`resolve_vae` in `scripts/generate.py`). For a 2x-resolution decode, use spacepxl's Wan2.1-VAE-upscale2x through the `ComfyUI-VAE-Utils` node. Full analysis and build provenance: [krea2_vae.md](krea2_vae.md).

## The canonical workflows

Four reference recipes, named A to D, cover the common cases. All use the modular graph and the Turbo LoRA on a RAW checkpoint rather than a separate Turbo checkpoint.

Turbo is RAW plus the Turbo LoRA at strength 1.0. So the LoRA strength is a continuous dial from RAW behaviour up to full Turbo distillation.

### Which workflow for which job

Pick by what the generation needs, then read the recipe row below for the wiring.

| If you need | Use | What it costs |
| --- | --- | --- |
| The everyday default: sharp, repeatable, cheapest. Best for A/B tests, where you change one thing and hold the rest fixed | A | 8 evals, deterministic |
| The same look but with texture and fine detail held at 8 steps | D | 8 evals, stochastic SDE finish |
| Seed and compositional variety, with a negative prompt that bites | B | about 2x A, because CFG runs the uncond pass too |
| The most seed diversity at near-Turbo speed, with CFG headroom to set composition | C, the split | a little above A, CFG runs only on the high-noise steps |
| Full RAW fidelity, no distillation | raw preset | 56 evals, the slowest |

Within any of these the Turbo-LoRA strength is a continuous RAW-to-Turbo dial: lower strength buys CFG headroom and seed variety but needs more steps. The quality-for-compute sweet spot stays at cfg 1, strength 0.6 to 0.8. See [turbo_lora_strength.md](turbo_lora_strength.md) for the dial and [two_sampler_split.md](two_sampler_split.md) for when the split earns its second pass.

| Workflow | Model | Guider | Sampler | Notes |
| --- | --- | --- | --- | --- |
| A drop-in | RAW plus Turbo LoRA 0.8 | BasicGuider, CFG off | euler | sharp, deterministic, 8 steps |
| B headroom | RAW plus Turbo LoRA 0.6 | CFGGuider 2.5 | euler | more seed diversity, CFG room |
| C split | RAW then RAW plus Turbo LoRA 1.0 | CFG 2.5 then off | euler | RAW sets composition, LoRA finishes |
| D SDE finish | RAW plus Turbo LoRA 0.8 | BasicGuider, CFG off | dpmpp_2m_sde eta 0.5 | textured, exponential finish |

Workflow C splits the 8-step schedule at step 3: the first stage runs RAW with real CFG to set the composition, the second runs the Turbo LoRA with CFG off to finish, taking the leftover noise from the first stage. `build_split_graph` in `scripts/generate.py` does this handoff with two `KSamplerAdvanced` nodes and start and end step ranges, rather than a `SplitSigmas` node. The other three recipes follow the modular graph described above.

## The generate.py helpers

`scripts/generate.py` talks to ComfyUI over HTTP, so it needs the model filenames but no ComfyUI path. It builds the simple and split shapes; you wire the modular shape from the nodes yourself, the way the reference recipes do.

`build_graph()` builds the one-sampler graph. Pass `loras=[(name, strength), ...]` to chain several LoRAs on the model edge, each at its own strength — for example a bypass LoRA, a projector `.diff` and others in one run. `lora=<filename>` is the single-LoRA shorthand.

The presets fix steps, CFG, the checkpoint and any LoRAs together: `turbo` (Turbo checkpoint, 8 steps, CFG off), `raw` (RAW checkpoint, 28 steps, CFG 5.5) and `turbo_lora` (RAW plus the Turbo LoRA, 8 steps, CFG off — the de-distillation dial). On the command line, `--lora name:strength` adds more LoRAs on top of the preset, so you can stack a bypass and a projector edit in one run.

`build_split_graph()` builds the 2-stage split that workflow C uses. The high-noise stage carries the CFG and the negative prompt, because the seed and composition are decided in the high-noise steps. The low-noise stage finishes with CFG off.

`run()` submits a graph and saves the first image output. It waits for the prompt to report `completed` before it reads the output. This matters because ComfyUI lists a prompt in its history while it is still running, most often during the first render after a restart while the model loads. Reading the output too early returns nothing and looks like a failure. Waiting for `completed` fixed the long-standing cold-start flakiness, so you no longer need a throwaway warm-up render.

`run()` also finds the image output under any node, rather than assuming the `SaveImage` node has a fixed name. So your graph can name that node anything.

## Related guides

- [krea2_vae.md](krea2_vae.md) compares the VAE decoders and explains the krea2RealVae default
- [findings.md](findings.md) records the measurements behind these choices
- [concept_directions.md](concept_directions.md) covers steering the conditioning with a measured direction
