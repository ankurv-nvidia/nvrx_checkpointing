# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decode independently stored scalar checkpoint entries."""

from __future__ import annotations

import struct

_FORMATS = {"int64": "<q", "uint64": "<Q", "float64": "<d", "bool": "<Q"}
_SCALAR_BYTES = 8


def scalar_storage_type(metadata: object) -> str | None:
    """Return the primitive datatype attached to a scalar storage entry."""
    kind = getattr(metadata, "nvrx_scalar_type", None)
    if kind is not None and (not isinstance(kind, str) or kind not in _FORMATS):
        raise ValueError(f"Unsupported checkpoint scalar datatype: {kind!r}")
    return kind


def decode_scalar(metadata: object, data: bytes) -> bool | int | float:
    """Decode a marked eight-byte primitive value."""
    kind = scalar_storage_type(metadata)
    if kind is None:
        raise ValueError("Checkpoint entry has no scalar datatype")
    if len(data) != _SCALAR_BYTES:
        raise ValueError("Checkpoint scalar payloads contain exactly eight bytes")
    value = struct.unpack(_FORMATS[kind], data)[0]
    if kind == "bool":
        if value not in (0, 1):
            raise ValueError("Invalid checkpoint boolean value")
        return bool(value)
    return value
