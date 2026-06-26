"""Command-line interface: inspect the Krea2 projector and emit projector .diff LoRAs.

    krea2-proj inspect <checkpoint>
    krea2-proj lora <checkpoint> <out.safetensors> (--preset balanced | --gains "1,1,...,1")
    krea2-proj solo <checkpoint> <out_dir> [--gain 1.0]
"""

from __future__ import annotations

import argparse

from . import projector
from . import projector_lora as pl


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="krea2-proj",
                                 description="Inspect and edit the Krea2 txtfusion projector (per-layer combiner).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_i = sub.add_parser("inspect", help="print the 12 learned projector weights")
    p_i.add_argument("checkpoint")

    p_l = sub.add_parser("lora", help="emit a projector .diff LoRA (loads via the stock LoRA loader)")
    p_l.add_argument("checkpoint")
    p_l.add_argument("out")
    g = p_l.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(pl.PRESETS))
    g.add_argument("--gains", help="comma/semicolon-separated 12 per-band gains")

    p_s = sub.add_parser("solo", help="emit 12 band-isolation LoRAs (one per selected Qwen3-VL layer)")
    p_s.add_argument("checkpoint")
    p_s.add_argument("out_dir")
    p_s.add_argument("--gain", type=float, default=1.0, help="boost applied to the kept band")

    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "inspect":
        weights = projector.read_projector(args.checkpoint)
        for i, (layer, val) in enumerate(zip(projector.SELECT_LAYERS, weights)):
            print(f"band {i:2d} L{layer:2d}: {val:+.4f}")
        return 0

    if args.cmd == "lora":
        if args.preset:
            pl.make_preset_lora(args.checkpoint, args.preset, args.out)
        else:
            gains = [float(x) for x in args.gains.replace(";", ",").split(",") if x.strip()]
            pl.make_projector_lora(args.checkpoint, gains, args.out)
        print(f"wrote {args.out}")
        return 0

    if args.cmd == "solo":
        paths = pl.make_band_isolation_loras(args.checkpoint, args.out_dir, solo_gain=args.gain)
        print(f"wrote {len(paths)} isolation LoRAs to {args.out_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
