"""Tests for emitting tiny projector ``.diff`` LoRAs.

A projector ``.diff`` LoRA has one tensor, ``diffusion_model.txtfusion.projector.diff`` [1,12], built as
``diff = orig*(gain-1)`` so ComfyUI's ``weight + strength*diff`` reproduces ``orig*gain`` at strength 1,
with the stock loader's strength as a live knob.
"""

import struct

import ml_dtypes
import numpy as np
import orjson
import pytest

from krea2_explorations import projector, projector_lora as pl
from krea2_explorations import safetensors_patch as sp

# real Turbo projector weights
ORIG = [-0.0544, -0.1611, 0.3711, 0.5039, 0.7070, 0.3945,
        0.3984, -1.4375, -0.5117, -0.8906, -0.6094, 0.1128]


def _ckpt(path, w=ORIG):
    """Minimal checkpoint exposing only txtfusion.projector.weight (bf16, [1,12])."""
    arr = np.array([w], dtype=ml_dtypes.bfloat16)
    raw = arr.tobytes()
    header = {projector.PROJECTOR_KEY: {"dtype": "BF16", "shape": [1, 12], "data_offsets": [0, len(raw)]}}
    hjson = orjson.dumps(header)
    hjson += b" " * ((-len(hjson)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(raw)


def test_diff_is_orig_times_gain_minus_one(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    out = tmp_path / "lora.safetensors"
    gains = [1, 1, 1, 1, 1, 1, 1, 2.5, 5.0, 1.1, 4.0, 1.0]
    diff = pl.make_projector_lora(ck, gains, out)

    orig = projector.read_projector(ck)
    np.testing.assert_allclose(diff, orig * (np.array(gains, dtype=np.float32) - 1.0), rtol=1e-6, atol=1e-7)

    # the file carries the expected key/shape and applies (at strength 1) to orig*gain
    t = sp.read_tensor(out, pl.LORA_KEY)
    assert t.shape == (1, 12)
    eff = orig + 1.0 * diff
    np.testing.assert_allclose(eff, orig * np.array(gains, dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_band_isolation_loras(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    paths = pl.make_band_isolation_loras(ck, tmp_path / "solo")
    assert len(paths) == 12
    orig = projector.read_projector(ck)

    diff8 = np.asarray(sp.read_tensor(paths[8], pl.LORA_KEY), dtype=np.float32).reshape(-1)
    eff = orig + diff8  # strength 1
    for i in range(12):
        if i == 8:
            assert abs(eff[i] - orig[8]) < 1e-4
        else:
            assert abs(eff[i]) < 1e-4


def test_effective_weights_semantics():
    orig = np.array(ORIG, dtype=np.float32)
    gains = [1.0] * 12
    gains[8] = 2.0
    g = np.array(gains, dtype=np.float32)
    np.testing.assert_allclose(pl.effective_weights(orig, gains, 1.0), orig * g, rtol=1e-6)
    np.testing.assert_allclose(pl.effective_weights(orig, gains, 0.0), orig, rtol=1e-6)
    np.testing.assert_allclose(pl.effective_weights(orig, gains, 2.0), orig * (1 + 2 * (g - 1)), rtol=1e-6)


def test_resolve_gains():
    assert pl.resolve_gains("uniform") == pl.PRESETS["uniform"]
    assert pl.resolve_gains("custom", "1,1,1,1,1,1,1,1,1,1,1,1") == [1.0] * 12
    solo = pl.resolve_gains("uniform", solo_band=8, solo_gain=3.0)
    assert solo[8] == 3.0 and sum(solo) == 3.0
    with pytest.raises(ValueError):
        pl.resolve_gains("custom", "1,2,3")
    with pytest.raises(ValueError):
        pl.resolve_gains("nope")


def test_rejects_bad_gain_length(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    with pytest.raises(ValueError):
        pl.make_projector_lora(ck, [1, 2, 3], tmp_path / "x.safetensors")


def test_emitted_lora_loads_with_real_safetensors(tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    out = tmp_path / "lora.safetensors"
    pl.make_preset_lora(ck, "uniform", out)
    loaded = safetensors.load_file(str(out))
    assert pl.LORA_KEY in loaded
    assert loaded[pl.LORA_KEY].shape == (1, 12)
