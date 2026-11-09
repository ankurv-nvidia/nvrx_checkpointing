# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core save strategies for the shared native checkpoint backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from nvrx_checkpointing.backend import NativeAsyncCallsQueue, NativeCheckpointSession, NativeSaveRequest

from .serialization import prepare_sharded_object_blobs

if TYPE_CHECKING:
    from pathlib import Path


class NativeSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Prepare sharded objects and submit snapshots through a native session."""

    def __init__(self, *, process_group: object | None = None) -> None:
        super().__init__("torch_dist", 1)
        self._session = NativeCheckpointSession(process_group=process_group)
        self._object_capacities: dict[str, int] = {}

    def _prepare_objects(self, value: Any) -> None:  # noqa: ANN401
        if isinstance(value, dict):
            for child in value.values():
                self._prepare_objects(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._prepare_objects(child)
        elif getattr(value, "flattened_range", None) is not None:
            raise ValueError(f"Native checkpoint cannot represent flattened tensor shard {value.key!r}")
        else:
            prepare_sharded_object_blobs(value, self._object_capacities)

    def _prepare_state_dict(self, state: dict) -> dict:
        self._prepare_objects(state)
        return state

    def async_save(self, sharded_state_dict: dict, checkpoint_dir: Path, async_strategy: str = "nvrx") -> NativeSaveRequest:
        del async_strategy
        self._session.register_state_dict(sharded_state_dict, prepare=self._prepare_state_dict)
        return self._session.submit(checkpoint_dir)

    def save(self, sharded_state_dict: dict, checkpoint_dir: Path) -> None:
        self.async_save(sharded_state_dict, checkpoint_dir).execute_sync()


__all__ = ["NativeAsyncCallsQueue", "NativeSaveShardedStrategy"]
