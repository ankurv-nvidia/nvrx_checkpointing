# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import TYPE_CHECKING

import _distcp_shim
import pytest
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.scalar import decode_scalar, scalar_storage_type
from torch.distributed.checkpoint import FileSystemReader

if TYPE_CHECKING:
    from pathlib import Path


def _write_scalar(path: Path, payload: bytes, kind: object = None) -> None:
    objects = [("value", payload)]
    _distcp_shim.write_rank_shard(str(path), 0, [], objects)
    _distcp_shim.write_metadata(str(path), [], objects)
    if kind is not None:
        metadata = FileSystemReader(path).read_metadata()
        metadata.state_dict_metadata["value"].nvrx_scalar_type = kind
        (path / ".metadata").write_bytes(pickle.dumps(metadata))


@pytest.mark.parametrize(
    ("kind", "payload", "expected"),
    [
        ("int64", "ffffffffffffffff", -1),
        ("uint64", "ffffffffffffffff", (1 << 64) - 1),
        ("float64", "000000000000f83f", 1.5),
        ("bool", "0100000000000000", True),
        ("bool", "0000000000000000", False),
    ],
)
def test_scalar_metadata_decodes_exact_known_bytes(tmp_path: Path, kind: str, payload: str, expected: object) -> None:
    actual = decode_scalar(SimpleNamespace(nvrx_scalar_type=kind), bytes.fromhex(payload))
    assert type(actual) is type(expected)
    assert actual == expected
    _write_scalar(tmp_path, bytes.fromhex(payload), kind)
    target = {"value": None}

    checkpoint.load_checkpoint(tmp_path, target)

    assert type(target["value"]) is type(expected)
    assert target["value"] == expected


def test_eight_byte_blob_has_no_implicit_scalar_type(tmp_path: Path) -> None:
    assert scalar_storage_type(SimpleNamespace()) is None
    with pytest.raises(ValueError, match="no scalar datatype"):
        decode_scalar(SimpleNamespace(), b"12345678")
    _write_scalar(tmp_path, b"12345678")
    target = {"value": b""}

    checkpoint.load_checkpoint(tmp_path, target)

    assert type(target["value"]) is bytes
    assert target["value"] == b"12345678"


@pytest.mark.parametrize("kind", ["unknown", [], 17])
def test_unknown_scalar_type_is_rejected(tmp_path: Path, kind: object) -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint scalar datatype"):
        scalar_storage_type(SimpleNamespace(nvrx_scalar_type=kind))
    _write_scalar(tmp_path, b"\0" * 8, kind)

    with pytest.raises((RuntimeError, ValueError), match=r"scalar|string"):
        checkpoint.load_checkpoint(tmp_path, {"value": None})


@pytest.mark.parametrize("payload", [b"", b"1234567", b"123456789"])
def test_scalar_payload_requires_exact_length(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(ValueError, match="exactly eight bytes"):
        decode_scalar(SimpleNamespace(nvrx_scalar_type="int64"), payload)
    _write_scalar(tmp_path, payload, "int64")

    with pytest.raises((RuntimeError, ValueError), match="eight bytes"):
        checkpoint.load_checkpoint(tmp_path, {"value": None})


def test_invalid_boolean_payload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint boolean"):
        decode_scalar(SimpleNamespace(nvrx_scalar_type="bool"), b"\x02" + b"\0" * 7)
    _write_scalar(tmp_path, b"\x02" + b"\0" * 7, "bool")

    with pytest.raises((RuntimeError, ValueError), match=r"scalar|boolean"):
        checkpoint.load_checkpoint(tmp_path, {"value": False})
