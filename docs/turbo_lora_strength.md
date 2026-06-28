Last updated: 2026-06-28

# The Turbo-LoRA strength dial (de-distillation lever)

Krea 2 ships a RAW (pre-distillation) checkpoint and a Turbo (8-step distilled) checkpoint, plus a
**Turbo LoRA** that *is* the distillation delta: `Turbo ≈ RAW + 1.0·delta` (the LoRA was merged at -1.0 to
cancel distillation during training). So the LoRA strength is a continuous **de-distillation dial**, and the
two checkpoints are the same model at two points on it. The strength can be read from either end:

> **`RAW + s·LoRA  ≡  Turbo + (s−1)·LoRA`**

| RAW frame (`s`) | Turbo frame (`s−1`) | what it is |
|---|---|---|
| 0.0 | −1.0 | pure RAW (distillation fully removed) |
| 0.5 | −0.5 | half-distilled |
| 0.8 / 0.9 | −0.2 / −0.1 | lightly de-distilled (community "works nice" values) |
| 1.0 | 0.0 | Turbo |
| 1.2 | +0.2 | slightly over-distilled |

This doc characterizes what moving along that dial does. **Confidence: low–medium** — one prompt
(`examples/test_prompts/nightclub.txt`), a few seeds, fp8 checkpoints, visual read on a 24 GB card. Treat as
orientation, not calibrated numbers. Probes are throwaway scripts over `scripts/generate.build_graph` (no
committed tooling); grids are gitignored under `data/`, the figures here are the saved copies.

## 1. At Turbo's native config (8 steps, cfg 1), the dial is a quality ramp

![Turbo-LoRA strength dial: rows = strength (RAW frame, with Turbo frame in parens), cols = seed, at 8 steps / cfg 1.](figures/turbo_lora_strength_dial.png)

*Rows = strength; cols = seed (42 / 123 / 7); 8 steps, cfg 1.*

- **Below ~0.8 the image is soft/washed** — RAW (0.0) at 8 steps is unusable (under-denoised), 0.5 is hazy.
  Low strength needs *more steps and/or cfg>1* to recover (see §2).
- **~0.8–1.2 (Turbo −0.2 … +0.2) is the usable band** and matches the community's "nice low values":
  - **0.8 / 0.9** (Turbo −0.2 / −0.1): sharp, slightly **softer / warmer / more open** than Turbo.
  - **1.0**: baseline Turbo — sharpest contrast, but same-seed compositions are near-identical (the
    distillation diversity collapse).
  - **1.2** (Turbo +0.2): **punchier / more saturated**, more table detail — over-distillation *adds pop*
    here rather than breaking (it only breaks at high cfg, §2).
- **Seed diversity rises as strength falls** (RAW seeds differ a lot; Turbo seeds barely) — but in the usable
  0.8–1.0 band the diversity difference is marginal at 8 steps. (A dedicated diversity-distillation test —
  base-model-for-the-first-step, arXiv:2503.10637 — was a **null** on Krea 2's flow schedule; only this
  global-strength axis moves diversity. Recorded internally, not pursued.)

## 2. Lower strength restores cfg>1 / negative-prompt headroom

![CFG headroom: rows = strength, cols = cfg 1.0 / 2.5 / 4.0, with a lamp-suppression negative.](figures/turbo_lora_cfg_headroom.png)

*Rows = strength; cols = cfg (1.0 / 2.5 / 4.0); 16 steps; negative = "table lamp, lampshade, glowing light".*

Turbo is distilled to run CFG-disabled (cfg 1, no negative). Backing the strength off restores guidance:

- **Turbo (1.0) burns out above cfg ~2.5** — blown highlights, oversaturated faces at cfg 4 (the classic
  distilled-model-breaks-at-high-cfg failure).
- **0.7 / 0.5 / RAW tolerate cfg 4** (more contrast/guidance, no burn) — i.e. **CFG headroom grows as
  strength drops**.
- **RAW (0.0) *requires* cfg>1** — washed out at cfg 1, only looks right at cfg ≳ 2.5.
- Practical window for a de-distilled run: **strength ~0.5, cfg ~2.5–4, ~16 steps**.

## 3. The negative branch is active but did not strongly steer

![Negative isolation: rows = strength, cols = empty / anti-blur / anti-lamp negative, at fixed cfg 3.](figures/turbo_lora_negative_branch.png)

*Rows = strength; cols = negative text (control "" / anti-blur / anti-lamp); cfg 3, 16 steps, fixed seed —
so any column-wise difference is the negative branch alone.*

- The negative **is wired and does perturb** the output at cfg>1 (columns differ at every strength), so
  cfg>1 makes the uncond path live.
- But **targeted suppression failed** — the anti-lamp negative never removed the lamp, at any strength, and
  anti-blur showed little (cfg 3 is already sharp). The effect is mild perturbation, not decisive content
  removal, and not obviously stronger at low strength.
- **Caveat / inconclusive:** the test target (the lamp) is *over-described in the positive prompt* (~30
  words), so a short negative can't win. Whether the negative steers cleanly at partial-distill is still
  open — needs a target that is **not** in the positive (see follow-ups).

## Practical recipe

| Goal | Strength (RAW / Turbo frame) | steps / cfg |
|---|---|---|
| Default fast + sharp | 1.0 / Turbo | 8 / 1 |
| Softer, warmer, a little more open | 0.8–0.9 / Turbo −0.2…−0.1 | 8 / 1 |
| Punchier, more saturated | 1.2 / Turbo +0.2 | 8 / 1 |
| Need cfg>1 or negative prompts | ~0.5 / Turbo −0.5 | ~16 / 2.5–4 |
| Max seed diversity (quality cost) | 0.0–0.5 / Turbo −1.0…−0.5 | needs cfg>1 + more steps |

## Open follow-ups

- **Strength × steps/sigmas** is unmapped: each strength below ~0.8 likely has a different
  steps-to-recover-quality cost. The §1 grid fixes 8 steps; a strength × {8,12,16,20,28} grid would find,
  e.g., "0.5 at 20 steps ≈ Turbo quality with more flexibility?".
- **Clean negative test**: repeat §3 with a target absent from the positive (or an empty-vs-strong negative
  on a washed-out low-strength/low-cfg image where there's headroom to see).
- **bf16 / other prompts**: fp8 RAW grained less here than an earlier note warned; re-check on bf16 and on
  non-"lamp-dominated" prompts.
- **Quantify** instead of eyeballing (sharpness/contrast metric across the dial; the per-cell folders are
  already laid out for it).
