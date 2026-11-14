# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import io
import math
import struct
from typing import TYPE_CHECKING

import _distcp_shim
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, scalars: dict, objects: tuple = (), *, native: bool = True, tensors: tuple = ()) -> None:
    entries = [(key, "float32", list(value.shape), value.numpy().tobytes()) for key, value in tensors]
    _distcp_shim.write_rank_shard(str(path), 0, entries, list(objects), scalars=scalars, native=native)
    layouts = [(key, "float32", list(value.shape), [(0, [(0, size) for size in value.shape])]) for key, value in tensors]
    _distcp_shim.write_metadata(str(path), layouts, list(objects), scalars=scalars, native=native)


def _records(path: Path) -> dict[str, bytes]:
    metadata = dcp.FileSystemReader(path).read_metadata()
    result = {}
    for index, storage in metadata.storage_data.items():
        with (path / storage.relative_path).open("rb") as stream:
            stream.seek(storage.offset)
            result[index.fqn] = stream.read(storage.length)
    return result


def test_native_scalars_preserve_exact_eight_byte_encodings(tmp_path: Path) -> None:
    scalars = {
        "minimum": -(1 << 63),
        "maximum": (1 << 64) - 1,
        "iteration": 50,
        "negative_zero": -0.0,
        "lr": 0.00015,
        "positive_infinity": math.inf,
        "negative_infinity": -math.inf,
        "enabled": True,
    }
    _write(tmp_path, scalars)

    records = _records(tmp_path)
    expected = {
        key: struct.pack("<d", value) if isinstance(value, float) else struct.pack("<q" if value < 0 else "<Q", value)
        for key, value in scalars.items()
    }
    assert records == expected
    assert (tmp_path / "__0_0.distcp").stat().st_size == 8 * len(scalars)
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    assert all(isinstance(value, BytesStorageMetadata) for value in metadata.state_dict_metadata.values())
    assert not (tmp_path / "metadata.json").exists()


def test_native_scalar_nan_is_preserved(tmp_path: Path) -> None:
    _write(tmp_path, {"value": math.nan})

    assert math.isnan(struct.unpack("<d", _records(tmp_path)["value"])[0])


def test_native_scalars_share_a_contiguous_file_with_tensors_and_blobs(tmp_path: Path) -> None:
    weight = torch.arange(4, dtype=torch.float32)
    _write(tmp_path, {"iteration": 7}, (("args", b"native arguments"),), tensors=(("model.weight", weight),))

    records = _records(tmp_path)
    assert records["iteration"] == struct.pack("<Q", 7)
    assert records["args"] == b"native arguments"
    torch.testing.assert_close(torch.load(io.BytesIO(records["model.weight"]), weights_only=False), weight)
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    intervals = sorted((record.offset, record.length) for record in metadata.storage_data.values())
    cursor = 0
    for offset, length in intervals:
        assert offset == cursor
        cursor += length
    assert cursor == (tmp_path / "__0_0.distcp").stat().st_size


def test_native_scalar_and_blob_records_refresh_on_every_save(tmp_path: Path) -> None:
    marker = tmp_path / "metadata.json"
    marker.write_text('{"sharded_backend": "torch_dist"}')
    lengths = []
    for iteration, payload in ((7, b"short"), (123456, bytes(range(256)) * 1024)):
        _write(tmp_path, {"iteration": iteration}, (("payload", payload),))

        assert _records(tmp_path) == {"iteration": struct.pack("<Q", iteration), "payload": payload}
        assert not marker.exists()
        lengths.append((tmp_path / "__0_0.distcp").stat().st_size)
    assert lengths == [8 + len(b"short"), 8 + 256 * 1024]


def test_standard_mode_retains_its_scalar_and_format_behavior(tmp_path: Path) -> None:
    _write(tmp_path, {"iteration": 7}, (("payload", b"opaque"),), native=False)

    assert _records(tmp_path) == {"payload": b"opaque"}
    assert (tmp_path / "metadata.json").exists()


def test_empty_native_state_has_no_orphan_shard(tmp_path: Path) -> None:
    _write(tmp_path, {})

    assert _records(tmp_path) == {}
    assert not (tmp_path / "__0_0.distcp").exists()
    assert not (tmp_path / "metadata.json").exists()
