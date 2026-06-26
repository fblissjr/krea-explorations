"""Tests for surgical single-tensor read/patch of safetensors files.

Fixtures are hand-built here (independent of the module under test) so the tests
verify the format handling rather than just round-tripping our own writer.
"""

import struct

import ml_dtypes
import numpy as np
import orjson
import pytest

from krea2_explorations import safetensors_patch as sp

# numpy dtype -> safetensors dtype string (only what the fixtures need)
_NP_TO_ST = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(ml_dtypes.bfloat16): "BF16",
    np.dtype(np.int32): "I32",
}


def _build_st(path, tensors):
    """Hand-build a minimal safetensors file from {name: np.ndarray}."""
    header = {}
    blob = bytearray()
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": _NP_TO_ST[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob += raw
    hjson = orjson.dumps(header)
    hjson += b" " * ((-len(hjson)) % 8)  # 8-byte align, safetensors convention
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_read_header_and_tensor(tmp_path):
    proj = np.array([[-1.4375, 0.5, 0.0, 2.0]], dtype=ml_dtypes.bfloat16)
    other = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = tmp_path / "m.safetensors"
    _build_st(p, {"txtfusion.projector.weight": proj, "other": other})

    header, _ = sp.read_header(p)
    assert header["txtfusion.projector.weight"]["dtype"] == "BF16"
    assert header["txtfusion.projector.weight"]["shape"] == [1, 4]

    got = sp.read_tensor(p, "txtfusion.projector.weight")
    assert got.shape == (1, 4)
    np.testing.assert_array_equal(got.astype(np.float32), proj.astype(np.float32))


def test_patch_tensor_isolates_change(tmp_path):
    proj = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=ml_dtypes.bfloat16)
    other = np.arange(6, dtype=np.float32).reshape(2, 3)
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    _build_st(src, {"txtfusion.projector.weight": proj, "other": other})

    # values chosen to be exactly bf16-representable so the round-trip is exact
    new = (proj.astype(np.float32) * np.array([2, 1, 0.5, 1], dtype=np.float32)).astype(
        ml_dtypes.bfloat16
    )
    sp.patch_tensor(src, dst, "txtfusion.projector.weight", new)

    got = sp.read_tensor(dst, "txtfusion.projector.weight")
    np.testing.assert_array_equal(got.astype(np.float32), new.astype(np.float32))

    # the untouched tensor is byte-identical
    np.testing.assert_array_equal(sp.read_tensor(dst, "other"), sp.read_tensor(src, "other"))

    # the header is unchanged (same length, same content) -> offsets stay valid
    h_src, n_src = sp.read_header(src)
    h_dst, n_dst = sp.read_header(dst)
    assert n_src == n_dst
    assert h_src == h_dst

    # the source file is left untouched
    assert sp.read_tensor(src, "txtfusion.projector.weight").astype(np.float32).tolist() == [
        [1.0, 2.0, 4.0, 8.0]
    ]


def test_patch_in_place(tmp_path):
    proj = np.array([[1.0, 2.0]], dtype=ml_dtypes.bfloat16)
    p = tmp_path / "m.safetensors"
    _build_st(p, {"txtfusion.projector.weight": proj})
    sp.patch_tensor(p, p, "txtfusion.projector.weight",
                    np.array([[4.0, 8.0]], dtype=ml_dtypes.bfloat16))
    np.testing.assert_array_equal(
        sp.read_tensor(p, "txtfusion.projector.weight").astype(np.float32),
        np.array([[4.0, 8.0]], dtype=np.float32),
    )


def test_patch_rejects_shape_mismatch(tmp_path):
    proj = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=ml_dtypes.bfloat16)
    p = tmp_path / "m.safetensors"
    _build_st(p, {"txtfusion.projector.weight": proj})
    with pytest.raises(ValueError):
        sp.patch_tensor(p, tmp_path / "o.safetensors", "txtfusion.projector.weight",
                        np.array([1.0, 2.0], dtype=ml_dtypes.bfloat16))


def test_read_tensor_missing_name(tmp_path):
    p = tmp_path / "m.safetensors"
    _build_st(p, {"a": np.zeros((2,), dtype=np.float32)})
    with pytest.raises(KeyError):
        sp.read_tensor(p, "does.not.exist")


def test_write_safetensors_roundtrip(tmp_path):
    p = tmp_path / "w.safetensors"
    a = np.array([[1.5, -2.0, 3.0]], dtype=np.float32)
    b = np.arange(4, dtype=ml_dtypes.bfloat16)
    sp.write_safetensors(p, {"a": a, "b": b}, metadata={"foo": "bar"})

    np.testing.assert_array_equal(sp.read_tensor(p, "a"), a)
    np.testing.assert_array_equal(sp.read_tensor(p, "b").astype(np.float32), b.astype(np.float32))
    header, _ = sp.read_header(p)
    assert header["__metadata__"]["foo"] == "bar"
