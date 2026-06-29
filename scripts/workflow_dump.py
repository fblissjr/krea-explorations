"""Save a ComfyUI API-format graph + a provenance sidecar (krea2-explorations workflow convention).

The harnesses build ComfyUI graphs in code and POST them; a POST-only graph leaves no reproducible artifact.
This is the one place that serializes a run's graph. See CLAUDE.md "Workflows".

Convention:
- Per-run dumps -> gitignored ``internal/workflows/`` (graphs can embed sensitive prompts), named with a
  SORTABLE UTC datetime: ``<harness>_<arm>_s<seed>_<YYYYMMDDTHHMMSSZ>.json``. Do NOT rely on mtime (it resets
  on copy/git/rsync). Provenance goes in a ``.meta.json`` sidecar -- NOT a top-level key inside the API JSON
  (ComfyUI treats an extra top-level key as a malformed node and fails to load it).
- Reference/canonical workflows: pass ``stable_name`` (no datetime); git is their version history.
- Promote a benign, validated workflow to public ``example_workflows/`` by hand.

stdlib-only (json, no orjson) so it imports cleanly in the project venv, the ComfyUI venv, and the isolated
training venv -- same portability rule as ``image_grid``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / "internal" / "workflows"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        return "unknown"


def dump_workflow(graph, *, harness, arm="run", seed=0, out_dir=WORKFLOWS,
                  stable_name=None, prompt=None):
    """Write ``graph`` (a ComfyUI API-format dict) + a ``.meta.json`` sidecar; return the JSON Path.

    Per-run name: ``<harness>_<arm>_s<seed>_<UTC>.json`` (sortable). ``stable_name`` -> a fixed name for a
    reference/canonical workflow. ``prompt`` (optional) is stored in the sidecar for quick eyeballing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = stable_name or f"{harness}_{arm}_s{seed}_{ts}"
    jpath = out / f"{name}.json"
    jpath.write_text(json.dumps(graph, indent=2))
    meta = {"ts_utc": ts, "harness": harness, "arm": arm, "seed": seed, "git_sha": _git_sha()}
    if prompt is not None:
        meta["prompt"] = prompt
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return jpath
