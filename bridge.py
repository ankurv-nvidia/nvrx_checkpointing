# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lazy checkpoint backend factories for Megatron Bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from .backend import NativeAsyncCallsQueue
    from .megatron.bridge import NativeSaveShardedStrategy


def create_save_strategy(*, process_group: object | None = None) -> NativeSaveShardedStrategy:
    """Create a native save strategy for the complete training process group."""
    from .megatron.bridge import NativeSaveShardedStrategy  # noqa: PLC0415

    return NativeSaveShardedStrategy(process_group=process_group)


def create_async_calls_queue(*, process_group: object | None = None) -> NativeAsyncCallsQueue:
    """Create a native durability queue for the complete training process group."""
    from .backend import NativeAsyncCallsQueue  # noqa: PLC0415

    return NativeAsyncCallsQueue(process_group=process_group)


def load_context(sharded_state_dict: dict, sharded_strategy: object) -> AbstractContextManager[None]:
    """Preserve replica identities and disable metadata caching during a load."""
    from .megatron.loading import load_context as context  # noqa: PLC0415

    return context(sharded_state_dict, sharded_strategy)


__all__ = ["create_async_calls_queue", "create_save_strategy", "load_context"]
