# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401, PLC0415

"""Native persistence strategies and durability queues for Megatron Core."""

from __future__ import annotations

import logging
import secrets
import socket
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

from .serialization import prepare_sharded_object_blobs

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _rank(process_group: object | None) -> int:
    return torch.distributed.get_rank(group=process_group) if torch.distributed.is_initialized() else 0


def _gather(value: Any, process_group: object | None) -> list[Any]:
    if not torch.distributed.is_initialized():
        return [value]
    result = [None] * torch.distributed.get_world_size(group=process_group)
    torch.distributed.all_gather_object(result, value, group=process_group)
    return result


def _trainer_peers(hosts: list[str]) -> tuple[list[str], int]:
    peers = list(dict.fromkeys(hosts))
    if not peers or any(not host for host in peers):
        raise RuntimeError("Native checkpoint topology has an empty training hostname")
    counts = [hosts.count(host) for host in peers]
    if len(set(counts)) != 1:
        raise RuntimeError(f"Native checkpoint topology requires equal training ranks per host: {hosts}")
    ranks_per_peer = counts[0]
    if hosts != [host for host in peers for _ in range(ranks_per_peer)]:
        raise RuntimeError(f"Native checkpoint topology requires contiguous training ranks per host: {hosts}")
    return peers, ranks_per_peer


@dataclass
class _NativeRequest:
    wait: Callable[[], None]
    finalize_fns: list[Callable[[], None]] = field(default_factory=list)
    session_id: int = 0
    sequence: int = 0
    checkpoint_dir: str = ""
    process_group: object | None = None

    def add_finalize_fn(self, fn: Callable[[], None]) -> None:
        self.finalize_fns.append(fn)

    def execute_sync(self) -> None:
        error = None
        try:
            self.wait()
        except Exception as exc:  # noqa: BLE001
            error = exc
        errors = _gather(repr(error) if error is not None else None, self.process_group)
        if any(errors):
            raise RuntimeError(f"Native checkpoint durability failed on training ranks: {errors}") from error
        for fn in self.finalize_fns:
            fn()


@dataclass
class _QueuedRequest:
    index: int
    request: _NativeRequest
    future: Future[None]


class NativeAsyncCallsQueue:
    """Keep native requests pending until every training rank reports durability."""

    def __init__(self, *, process_group: object | None = None) -> None:
        self._process_group = process_group
        self._requests: deque[_QueuedRequest] = deque()
        self._executor: ThreadPoolExecutor | None = None
        self._next_index = 0
        self.closed = False

    def schedule_async_request(self, request: _NativeRequest) -> int:
        if self.closed:
            raise RuntimeError("Native checkpoint queue is closed")
        if not isinstance(request, _NativeRequest):
            raise TypeError("Native checkpoint queue received a non-native save request")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-checkpoint")
        index = self._next_index
        self._next_index += 1
        future = self._executor.submit(request.wait)
        self._requests.append(_QueuedRequest(index, request, future))
        return index

    def get_num_unfinalized_calls(self) -> int:
        return len(self._requests)

    def maybe_finalize_async_calls(self, blocking: bool = False) -> list[int]:
        finalized = []
        while self._requests:
            pending = self._requests[0]
            done = pending.future.done()
            error = pending.future.exception() if done else None
            statuses = _gather((pending.index, done, repr(error) if error is not None else None), self._process_group)
            if any(status[0] != pending.index for status in statuses):
                raise RuntimeError(f"Native checkpoint queues disagree on pending save order: {statuses}")
            failures = [(rank, status[2]) for rank, status in enumerate(statuses) if status[2] is not None]
            if failures:
                raise RuntimeError(f"Native checkpoint durability failed on training ranks: {failures}") from error
            if not all(status[1] for status in statuses):
                if not blocking:
                    break
                time.sleep(0.05)
                continue
            pending.future.result()
            for fn in pending.request.finalize_fns:
                fn()
            self._requests.popleft()
            finalized.append(pending.index)
            logger.info(
                "Native checkpoint finalized: session=%s sequence=%s path=%s",
                pending.request.session_id,
                pending.request.sequence,
                pending.request.checkpoint_dir,
            )
        return finalized

    def close(self) -> None:
        self.maybe_finalize_async_calls(blocking=True)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.closed = True


class NativeSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Persist padded sharded state through the native checkpoint daemon."""

    def __init__(self, *, process_group: object | None = None) -> None:
        super().__init__("torch_dist", 1)
        self._process_group = process_group
        self._root = None
        self._session_id: int | None = None
        self._sequence = 0
        self._object_capacities: dict[str, int] = {}

    def _initialize(self) -> None:
        from nvrx_checkpointing import checkpoint

        hosts = _gather(socket.gethostname(), self._process_group)
        peers, ranks_per_peer = _trainer_peers(hosts)
        rank = _rank(self._process_group)
        sessions = _gather(secrets.randbelow(2**32 - 1) + 1 if rank == 0 else None, self._process_group)
        self._session_id = sessions[0]
        self._root = checkpoint.model_state.create_root(
            self._session_id,
            rank,
            len(hosts),
            peers=peers,
            ranks_per_peer=ranks_per_peer,
        )
        logger.info(
            "Native checkpoint initialized: session=%s peers=%s ranks_per_peer=%s world_size=%s",
            self._session_id,
            peers,
            ranks_per_peer,
            len(hosts),
        )

    def _prepare_objects(self, value: Any) -> None:
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

    def async_save(self, sharded_state_dict: dict, checkpoint_dir: Path, async_strategy: str = "nvrx") -> _NativeRequest:
        from nvrx_checkpointing import checkpoint

        del async_strategy
        if self._root is None:
            self._initialize()
        error = None
        try:
            self._prepare_objects(sharded_state_dict)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        errors = _gather(error, self._process_group)
        if any(errors):
            raise RuntimeError(f"Native checkpoint state preparation failed: {errors}")
        error = None
        try:
            self._root.register_state_dict(sharded_state_dict)
            self._root.wait_for_async_completion()
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        errors = _gather(error, self._process_group)
        if any(errors):
            raise RuntimeError(f"Native checkpoint state registration failed: {errors}")
        self._sequence += 1
        sequence = self._sequence
        self._root.startedAt = time.time_ns()
        error = None
        try:
            self._root.save_async(sequence, Path(checkpoint_dir))
            self._root.wait_for_async_completion()
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        errors = _gather(error, self._process_group)
        if any(errors):
            raise RuntimeError(f"Native checkpoint snapshot submission failed: {errors}")
        logger.info("Native checkpoint submitted: session=%s sequence=%s path=%s", self._session_id, sequence, checkpoint_dir)

        def wait() -> None:
            checkpoint.wait_for_ckpt_finalize(self._session_id, sequence)
            logger.info("Native checkpoint durable: session=%s sequence=%s path=%s", self._session_id, sequence, checkpoint_dir)

        return _NativeRequest(
            wait, session_id=self._session_id, sequence=sequence, checkpoint_dir=str(checkpoint_dir), process_group=self._process_group,
        )

    def save(self, sharded_state_dict: dict, checkpoint_dir: Path) -> None:
        self.async_save(sharded_state_dict, checkpoint_dir).execute_sync()
