# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint load contexts for native shared-memory registration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def load_context(sharded_state_dict: dict, sharded_strategy: object) -> Iterator[None]:
    """Reads metadata directly and restores replica identities after a checkpoint load."""
    from megatron.core.dist_checkpointing.dict_utils import nested_values  # noqa: PLC0415

    strategy = sharded_strategy
    while strategy is not None:
        if hasattr(strategy, "cache_metadata"):
            strategy.cache_metadata = False
        strategy = getattr(strategy, "base_strategy", None)
    replica_ids = [(value, value.replica_id) for value in nested_values(sharded_state_dict) if hasattr(value, "replica_id")]
    try:
        yield
    finally:
        for value, replica_id in replica_ids:
            value.replica_id = replica_id
