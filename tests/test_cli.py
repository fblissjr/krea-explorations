"""Smoke tests for the krea2-proj CLI (thin glue over the tested library functions)."""

import struct

import ml_dtypes
import numpy as np
import orjson

from krea2_explorations import cli, projector
from krea2_explorations import projector_lora as pl
from krea2_explorations.safetensors_patch import read_tensor

ORIG = [-0.0544, -0.1611, 0.3711, 0.5039, 0.7070, 0.3945,
        0.3984, -1.4375, -0.5117, -0.8906, -0.6094, 0.1128]


def _ckpt(path):
    arr = np.array([ORIG], dtype=ml_dtypes.bfloat16)
    raw = arr.tobytes()
    header = {projector.PROJECTOR_KEY: {"dtype": "BF16", "shape": [1, 12], "data_offsets": [0, len(raw)]}}
    hjson = orjson.dumps(header)
    hjson += b" " * ((-len(hjson)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(raw)


def test_cli_inspect(tmp_path, capsys):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    assert cli.main(["inspect", str(ck)]) == 0
    assert "L23" in capsys.readouterr().out


def test_cli_lora_preset(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    out = tmp_path / "l.safetensors"
    assert cli.main(["lora", str(ck), str(out), "--preset", "uniform"]) == 0
    assert read_tensor(out, pl.LORA_KEY).shape == (1, 12)


def test_cli_lora_custom_gains(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    out = tmp_path / "l.safetensors"
    assert cli.main(["lora", str(ck), str(out), "--gains", "1,1,1,1,1,1,1,1,2,2,2,1"]) == 0
    assert read_tensor(out, pl.LORA_KEY).shape == (1, 12)


def test_cli_solo(tmp_path):
    ck = tmp_path / "k.safetensors"
    _ckpt(ck)
    assert cli.main(["solo", str(ck), str(tmp_path / "solo")]) == 0
    assert len(list((tmp_path / "solo").glob("*.safetensors"))) == 12
