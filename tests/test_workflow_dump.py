"""Tests for the workflow-dump convention: the dump helper + its auto-dump wiring into generate.run().

Every render should leave a loadable API-graph artifact (CLAUDE.md "Workflows"). These pin the sidecar
contract and that run() dumps BEFORE it POSTs (so the artifact survives even if the render fails).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate  # noqa: E402
from workflow_dump import dump_workflow  # noqa: E402


def test_dump_writes_json_and_meta_sidecar(tmp_path):
    g = {"save": {"class_type": "SaveImage", "inputs": {}}}
    p = dump_workflow(g, harness="h", arm="a", seed=3, prompt="hello", out_dir=tmp_path)
    assert p.suffix == ".json" and p.name.startswith("h_a_s3_")
    assert json.loads(p.read_text()) == g                       # graph JSON is the bare API dict (loadable)
    meta = json.loads(p.with_name(p.stem + ".meta.json").read_text())  # match dump_workflow's "{name}.meta.json"
    assert meta["harness"] == "h" and meta["arm"] == "a" and meta["seed"] == 3 and meta["prompt"] == "hello"
    assert "ts_utc" in meta and "git_sha" in meta               # provenance lives in the sidecar, not the graph


def test_provenance_is_not_a_top_level_graph_key(tmp_path):
    # an extra top-level key makes ComfyUI treat the graph as a malformed node -> must stay out of the JSON.
    p = dump_workflow({"a": {"class_type": "X", "inputs": {}}}, harness="h", out_dir=tmp_path)
    graph = json.loads(p.read_text())
    assert set(graph) == {"a"}


def test_stable_name_has_no_timestamp(tmp_path):
    p = dump_workflow({"x": {}}, harness="h", stable_name="10_canonical", out_dir=tmp_path)
    assert p.name == "10_canonical.json"


def test_run_auto_dumps_before_posting_when_harness_given(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no server in test")
    monkeypatch.setattr(generate.urllib.request, "urlopen", boom)
    g = {"save": {"class_type": "SaveImage", "inputs": {}}}
    with pytest.raises(ConnectionError):
        generate.run(g, str(tmp_path / "o.png"), harness="probe", arm="x", seed=7, prompt="p", dump_dir=tmp_path)
    dumps = [p for p in tmp_path.glob("probe_x_s7_*.json") if not p.name.endswith(".meta.json")]
    assert len(dumps) == 1                                       # graph dumped before the (failing) POST


def test_run_without_harness_dumps_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(generate.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
    with pytest.raises(ConnectionError):
        generate.run({"save": {}}, str(tmp_path / "o.png"), dump_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))                    # opt-in: no harness -> no dump


def test_sidecar_name_survives_dots_in_stable_name(tmp_path):
    # a dotted stable_name (e.g. a turbo-lora strength) must NOT get mangled: sidecar is "<name>.meta.json",
    # not a .with_suffix replacement that would eat the trailing ".4". Guards a future dump_workflow refactor.
    p = dump_workflow({"x": {"class_type": "Y", "inputs": {}}}, harness="h",
                      stable_name="61_turbo0.4", out_dir=tmp_path)
    assert p.name == "61_turbo0.4.json"
    assert (tmp_path / "61_turbo0.4.meta.json").exists()        # un-mangled sidecar alongside the .json


def test_rejects_non_dict_or_empty_graph(tmp_path):
    for bad in (None, [], {}, "x"):                             # silently writing null/[]/"" is not "loadable"
        with pytest.raises(ValueError):
            dump_workflow(bad, harness="h", out_dir=tmp_path)


def test_per_run_names_dont_collide_but_stable_names_overwrite(tmp_path):
    a = dump_workflow({"x": {"class_type": "A", "inputs": {}}}, harness="h", arm="a", seed=1, out_dir=tmp_path)
    b = dump_workflow({"x": {"class_type": "B", "inputs": {}}}, harness="h", arm="a", seed=1, out_dir=tmp_path)
    assert a != b and a.exists() and b.exists()                # same harness/arm/seed/second -> distinct files
    c = dump_workflow({"x": {"inputs": {}}}, harness="h", stable_name="10_ref", out_dir=tmp_path)
    d = dump_workflow({"y": {"inputs": {}}}, harness="h", stable_name="10_ref", out_dir=tmp_path)
    assert c == d                                              # stable name is canonical -> overwrite, not bump
