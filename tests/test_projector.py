"""Tests for the Krea2 txtfusion.projector read/edit helpers.

The projector is a learned Linear(12 -> 1); its weight has shape [1, 12] and is the
model's own per-layer combiner over the 12 selected Qwen3-VL layers. Scaling band i of the
conditioning input is equivalent to scaling weight column i, so the surgical edit is
a per-column multiply along the in-features axis.
"""

import struct

import ml_dtypes
import numpy as np
import orjson
import pytest

from krea2_explorations import projector


def _build_st(path, tensors):
    header = {}
    blob = bytearray()
    st_dtype = {np.dtype(ml_dtypes.bfloat16): "BF16", np.dtype(np.float32): "F32"}
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": st_dtype[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob += raw
    hjson = orjson.dumps(header)
    hjson += b" " * ((-len(hjson)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _fake_checkpoint(path):
    w = np.array([[-0.0544, -0.1611, 0.3711, 0.5039, 0.7070, 0.3945,
                   0.3984, -1.4375, -0.5117, -0.8906, -0.6094, 0.1128]],
                 dtype=ml_dtypes.bfloat16)
    _build_st(path, {
        projector.PROJECTOR_KEY: w,
        "blocks.0.mlp.up.weight": np.ones((4, 4), dtype=ml_dtypes.bfloat16),
    })
    return w


def test_read_projector_returns_12_floats(tmp_path):
    p = tmp_path / "k.safetensors"
    w = _fake_checkpoint(p)
    got = projector.read_projector(p)
    assert got.shape == (12,)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, w.astype(np.float32).reshape(-1), rtol=0, atol=0)


def test_scale_projector_multiplies_per_band(tmp_path):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    w = _fake_checkpoint(src).astype(np.float32).reshape(-1)

    # node default gains: boost bands 7,8,9,10
    gains = [1, 1, 1, 1, 1, 1, 1, 2.5, 5.0, 1.1, 4.0, 1.0]
    new = projector.scale_projector(src, dst, gains)

    expected = (w * np.array(gains, dtype=np.float32)).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(new, expected)
    # and it is what got written
    np.testing.assert_array_equal(projector.read_projector(dst), expected)
    # untouched tensor preserved
    from krea2_explorations import safetensors_patch as sp
    np.testing.assert_array_equal(
        sp.read_tensor(dst, "blocks.0.mlp.up.weight"),
        sp.read_tensor(src, "blocks.0.mlp.up.weight"),
    )


def test_scale_projector_rejects_wrong_length(tmp_path):
    src = tmp_path / "src.safetensors"
    _fake_checkpoint(src)
    with pytest.raises(ValueError):
        projector.scale_projector(src, tmp_path / "d.safetensors", [1.0, 2.0, 3.0])
