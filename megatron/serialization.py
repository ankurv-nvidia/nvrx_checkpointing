# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stable padded serialization for checkpoint sharded objects."""

from __future__ import annotations

import io

import torch


def snapshot_blob_capacity(serialized_size: int) -> int:
    """Return an aligned object slot with serialization growth padding."""
    return ((serialized_size + 512 + 1023) // 1024) * 1024


def prepare_sharded_object_blobs(value: object, capacities: dict[str, int]) -> None:
    """Refresh main-replica object bytes inside their registered slot capacities."""
    if isinstance(value, dict):
        for child in value.values():
            prepare_sharded_object_blobs(child, capacities)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            prepare_sharded_object_blobs(child, capacities)
        return
    if type(value).__name__ != "ShardedObject" or not hasattr(value, "data") or not hasattr(value, "unique_key"):
        return
    replica_id = getattr(value, "replica_id", 0)
    if replica_id is not None and (
        (isinstance(replica_id, int) and replica_id != 0)
        or (not isinstance(replica_id, int) and any(component != 0 for component in replica_id))
    ):
        return
    buffer = io.BytesIO()
    torch.save([value.data], buffer)
    serialized = buffer.getvalue()
    key = str(value.unique_key)
    capacity = capacities.setdefault(key, snapshot_blob_capacity(len(serialized)))
    if len(serialized) > capacity:
        raise RuntimeError(
            f"sharded object '{key}' serialized to {len(serialized)} bytes, exceeding its registered {capacity}-byte capacity",
        )
    value._nvrx_serialized_blob = serialized.ljust(capacity, b"\0")  # noqa: SLF001
