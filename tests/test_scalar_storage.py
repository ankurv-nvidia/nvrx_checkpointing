# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from types import SimpleNamespace

import pytest
from nvrx_checkpointing.scalar import decode_scalar, scalar_storage_type


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
def test_scalar_metadata_decodes_exact_known_bytes(kind: str, payload: str, expected: object) -> None:
    metadata = SimpleNamespace(nvrx_scalar_type=kind)

    actual = decode_scalar(metadata, bytes.fromhex(payload))

    assert type(actual) is type(expected)
    assert actual == expected


def test_eight_byte_blob_has_no_implicit_scalar_type() -> None:
    metadata = SimpleNamespace()

    assert scalar_storage_type(metadata) is None
    with pytest.raises(ValueError, match="no scalar datatype"):
        decode_scalar(metadata, b"12345678")


@pytest.mark.parametrize("kind", ["unknown", [], 17])
def test_unknown_scalar_type_is_rejected(kind: object) -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint scalar datatype"):
        scalar_storage_type(SimpleNamespace(nvrx_scalar_type=kind))


@pytest.mark.parametrize("payload", [b"", b"1234567", b"123456789"])
def test_scalar_payload_requires_exact_length(payload: bytes) -> None:
    with pytest.raises(ValueError, match="exactly eight bytes"):
        decode_scalar(SimpleNamespace(nvrx_scalar_type="int64"), payload)


def test_invalid_boolean_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint boolean"):
        decode_scalar(SimpleNamespace(nvrx_scalar_type="bool"), b"\x02" + b"\0" * 7)
