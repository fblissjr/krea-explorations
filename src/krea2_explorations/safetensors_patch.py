"""Surgical single-tensor read/patch for safetensors files, without loading the whole file.

The safetensors format is::

    [8-byte little-endian header length N][N-byte JSON header][raw tensor data]

The header maps ``name -> {"dtype", "shape", "data_offsets": [start, end]}`` where the offsets
are relative to the start of the data section (byte ``8 + N``). Because we only ever overwrite an
existing tensor with the *same* dtype and shape, its byte length and every offset are unchanged, so
we can patch a copied file in place -- never materializing the (multi-GB) full state dict.

This is the foundation for editing a Krea2 checkpoint's ``txtfusion.projector.weight`` (12 floats)
without touching the other ~26 GB of weights.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import orjson

# safetensors dtype string -> numpy dtype
DTYPE_TO_NP: dict[str, np.dtype] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}

_HEADER_LEN = struct.Struct("<Q")

# numpy dtype -> safetensors dtype string (inverse of DTYPE_TO_NP)
NP_TO_DTYPE: dict[np.dtype, str] = {v: k for k, v in DTYPE_TO_NP.items()}


def read_header(path) -> tuple[dict, int]:
    """Return ``(header_dict, header_byte_length)``.

    The header dict is returned verbatim, including any ``__metadata__`` entry.
    """
    with open(path, "rb") as f:
        (n,) = _HEADER_LEN.unpack(f.read(8))
        header = orjson.loads(f.read(n))
    return header, n


def read_tensor(path, name) -> np.ndarray:
    """Read a single tensor by name as a numpy array (no full-file load)."""
    header, n = read_header(path)
    if name not in header:
        raise KeyError(f"tensor {name!r} not found in {path}")
    meta = header[name]
    dtype = DTYPE_TO_NP[meta["dtype"]]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + start)
        buf = f.read(end - start)
    return np.frombuffer(buf, dtype=dtype).reshape(meta["shape"])


def patch_tensor(src, dst, name, new_array) -> None:
    """Write ``new_array`` into tensor ``name``, copying ``src`` to ``dst`` first.

    ``new_array`` must match the existing tensor's shape; it is cast to the existing dtype.
    If ``src == dst`` the file is patched in place. All other tensors and the header are untouched.
    """
    src, dst = Path(src), Path(dst)
    header, n = read_header(src)
    if name not in header:
        raise KeyError(f"tensor {name!r} not found in {src}")
    meta = header[name]
    dtype = DTYPE_TO_NP[meta["dtype"]]
    start, end = meta["data_offsets"]

    arr = np.asarray(new_array, dtype=dtype)
    if list(arr.shape) != list(meta["shape"]):
        raise ValueError(
            f"new array shape {tuple(arr.shape)} != existing shape {tuple(meta['shape'])} for {name!r}"
        )
    raw = arr.tobytes()
    if len(raw) != end - start:
        raise ValueError(
            f"new byte length {len(raw)} != existing {end - start} for {name!r}"
        )

    if src != dst:
        shutil.copy2(src, dst)
    with open(dst, "r+b") as f:
        f.seek(8 + n + start)
        f.write(raw)


def write_safetensors(path, tensors, metadata=None) -> None:
    """Write a small safetensors file from ``{name: np.ndarray}``.

    ``metadata`` (optional) is stored under ``__metadata__`` as string->string. Intended for
    emitting tiny artifacts (e.g. a 12-float projector ``.diff`` LoRA), not for large state dicts.
    """
    header: dict = {}
    blob = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in NP_TO_DTYPE:
            raise ValueError(f"unsupported dtype {arr.dtype} for tensor {name!r}")
        raw = arr.tobytes()
        header[name] = {
            "dtype": NP_TO_DTYPE[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob += raw
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hjson = orjson.dumps(header)
    hjson += b" " * ((-len(hjson)) % 8)  # 8-byte align (safetensors convention)
    with open(path, "wb") as f:
        f.write(_HEADER_LEN.pack(len(hjson)))
        f.write(hjson)
        f.write(blob)
