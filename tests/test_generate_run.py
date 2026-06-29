"""Tests for generate.run()'s render-completion polling (no live server -- urlopen is mocked).

ComfyUI lists a prompt in /history *while it is still executing* (notably during the first-render model load
after a restart), so checking mere presence and reading outputs returns empty -> a spurious False. run() must
wait for status.completed. These pin that, plus the genuinely-empty (errored) case.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate  # noqa: E402

_IMG = {"filename": "x.png", "subfolder": "", "type": "output"}
_RUNNING = {"abc": {"status": {"completed": False}, "outputs": {}}}                       # present, still running
_DONE = {"abc": {"status": {"completed": True}, "outputs": {"save": {"images": [_IMG]}}}}  # completed w/ output
_DONE_EMPTY = {"abc": {"status": {"completed": True}, "outputs": {}}}                      # completed, errored


class _Resp:
    def __init__(self, data):
        self._d = data

    def read(self):
        return self._d


def _fake_urlopen(history_seq):
    """A urlopen stand-in: POST -> prompt_id, /history -> next entry in history_seq, /view -> image bytes."""
    state = {"hist": 0}

    def fake(arg, timeout=None):
        url = arg.full_url if hasattr(arg, "full_url") else arg
        if url.endswith("/prompt"):
            return _Resp(json.dumps({"prompt_id": "abc"}).encode())
        if "/history/" in url:
            entry = history_seq[min(state["hist"], len(history_seq) - 1)]
            state["hist"] += 1
            return _Resp(json.dumps(entry).encode())
        if "/view?" in url:
            return _Resp(b"PNGBYTES")
        raise AssertionError(f"unexpected url {url}")

    return fake, state


def test_run_waits_for_completed_not_mere_presence(tmp_path, monkeypatch):
    fake, state = _fake_urlopen([_RUNNING, _RUNNING, _DONE])
    monkeypatch.setattr(generate.urllib.request, "urlopen", fake)
    monkeypatch.setattr(generate.time, "sleep", lambda *_a, **_k: None)
    out = tmp_path / "o.png"
    assert generate.run({"save": {"class_type": "SaveImage", "inputs": {}}}, str(out)) is True
    assert out.read_bytes() == b"PNGBYTES"
    assert state["hist"] >= 3            # polled past the two "running" entries instead of False-on-presence


def test_run_returns_false_when_completed_with_no_output(tmp_path, monkeypatch):
    fake, _ = _fake_urlopen([_DONE_EMPTY])
    monkeypatch.setattr(generate.urllib.request, "urlopen", fake)
    monkeypatch.setattr(generate.time, "sleep", lambda *_a, **_k: None)
    assert generate.run({"save": {"class_type": "SaveImage", "inputs": {}}}, str(tmp_path / "o.png")) is False
