# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401, PLC0415

"""Native checkpoint sessions and asynchronous durability completion."""

from __future__ import annotations

import logging
import secrets
import socket
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


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
class NativeSaveRequest:
    """Wait for native durability before invoking checkpoint finalizers."""

    wait: Callable[[], None]
    finalize_fns: list[Callable[[], None]] = field(default_factory=list)
    session_id: int = 0
    sequence: int = 0
    checkpoint_dir: str = ""
    process_group: object | None = None
    collective_checks: bool = True

    def add_finalize_fn(self, fn: Callable[[], None]) -> None:
        self.finalize_fns.append(fn)

    def execute_sync(self) -> None:
        error = None
        try:
            self.wait()
        except Exception as exc:  # noqa: BLE001
            error = exc
        status = repr(error) if error is not None else None
        errors = _gather(status, self.process_group) if self.collective_checks else [status]
        if any(errors):
            raise RuntimeError(f"Native checkpoint durability failed on training ranks: {errors}") from error
        for fn in self.finalize_fns:
            fn()


@dataclass
class _QueuedRequest:
    index: int
    request: NativeSaveRequest
    future: Future[None]


class NativeAsyncCallsQueue:
    """Keep native requests pending until their durability checks succeed."""

    def __init__(self, *, process_group: object | None = None, collective_checks: bool = True) -> None:
        self._process_group = process_group
        self._collective_checks = collective_checks
        self._requests: deque[_QueuedRequest] = deque()
        self._executor: ThreadPoolExecutor | None = None
        self._next_index = 0
        self.closed = False

    def schedule_async_request(self, request: NativeSaveRequest) -> int:
        if self.closed:
            raise RuntimeError("Native checkpoint queue is closed")
        if not isinstance(request, NativeSaveRequest):
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

    def is_current_async_call_done(self) -> bool:
        """Return local readiness for the oldest pending durability wait."""
        return not self._requests or self._requests[0].future.done()

    def maybe_finalize_async_calls(self, blocking: bool = False) -> list[int]:
        finalized = []
        while self._requests:
            pending = self._requests[0]
            done = pending.future.done()
            error = pending.future.exception() if done else None
            status = (pending.index, done, repr(error) if error is not None else None)
            statuses = _gather(status, self._process_group) if self._collective_checks else [status]
            if any(value[0] != pending.index for value in statuses):
                raise RuntimeError(f"Native checkpoint queues disagree on pending save order: {statuses}")
            failures = [(rank, value[2]) for rank, value in enumerate(statuses) if value[2] is not None]
            if failures:
                raise RuntimeError(f"Native checkpoint durability failed on training ranks: {failures}") from error
            if not all(value[1] for value in statuses):
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


class NativeCheckpointSession:
    """Register prepared state and submit snapshots to one native session."""

    def __init__(
        self,
        *,
        process_group: object | None = None,
        session_id: int | None = None,
        peers: list[str] | None = None,
        ranks_per_peer: int | None = None,
        collective_checks: bool = True,
    ) -> None:
        if (peers is None) != (ranks_per_peer is None):
            raise ValueError("Native checkpoint peers and ranks per peer must be supplied together")
        self._process_group = process_group
        self._session_id = session_id
        self._peers = peers
        self._ranks_per_peer = ranks_per_peer
        self._collective_checks = collective_checks
        self._root: Any = None
        self._sequence = 0

    @property
    def root(self) -> Any:
        """Return the lazily initialized native state tree."""
        if self._root is None:
            self._initialize()
        return self._root

    @property
    def session_id(self) -> int | None:
        """Return the configured or initialized native session identifier."""
        return self._session_id

    @property
    def process_group(self) -> object | None:
        return self._process_group

    def _initialize(self) -> None:
        from nvrx_checkpointing import checkpoint

        rank = _rank(self._process_group)
        world_size = torch.distributed.get_world_size(group=self._process_group) if torch.distributed.is_initialized() else 1
        if self._peers is None:
            self._peers, self._ranks_per_peer = _trainer_peers(_gather(socket.gethostname(), self._process_group))
        if self._session_id is None:
            sessions = _gather(secrets.randbelow(2**32 - 1) + 1 if rank == 0 else None, self._process_group)
            self._session_id = sessions[0]
        self._root = checkpoint.model_state.create_root(
            self._session_id,
            rank,
            world_size,
            peers=self._peers,
            ranks_per_peer=self._ranks_per_peer,
        )
        logger.info(
            "Native checkpoint initialized: session=%s peers=%s ranks_per_peer=%s world_size=%s",
            self._session_id,
            self._peers,
            self._ranks_per_peer,
            world_size,
        )

    def _run_stage(self, action: Callable[[], _T], label: str) -> _T:
        result: Any = None
        error = None
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001
            error = exc
        status = repr(error) if error is not None else None
        errors = _gather(status, self._process_group) if self._collective_checks else [status]
        if any(errors):
            raise RuntimeError(f"Native checkpoint {label} failed: {errors}") from error
        return result

    def register_state_dict(self, state: dict, *, prepare: Callable[[dict], dict] | None = None) -> Any:
        """Prepare and register current tensor storage with a local completion fence."""
        root = self.root
        if prepare is not None:
            state = self._run_stage(lambda: prepare(state), "state preparation")

        def register() -> None:
            root.register_state_dict(state)
            root.wait_for_async_completion()

        self._run_stage(register, "state registration")
        return root

    def submit(self, checkpoint_dir: Path, *, snapshot_id: int | None = None, started_at_ns: int | None = None) -> NativeSaveRequest:
        """Submit registered storage and return its deferred durability request."""
        from nvrx_checkpointing import checkpoint

        root = self.root
        self._sequence = self._sequence + 1 if snapshot_id is None else snapshot_id
        sequence = self._sequence
        root.startedAt = time.time_ns() if started_at_ns is None else started_at_ns

        def snapshot() -> None:
            root.save_async(sequence, Path(checkpoint_dir))
            root.wait_for_async_completion()

        self._run_stage(snapshot, "snapshot submission")
        session_id = self._session_id
        logger.info("Native checkpoint submitted: session=%s sequence=%s path=%s", session_id, sequence, checkpoint_dir)

        def wait() -> None:
            checkpoint.wait_for_ckpt_finalize(session_id, sequence)
            logger.info("Native checkpoint durable: session=%s sequence=%s path=%s", session_id, sequence, checkpoint_dir)

        return NativeSaveRequest(
            wait,
            session_id=session_id,
            sequence=sequence,
            checkpoint_dir=str(checkpoint_dir),
            process_group=self._process_group,
            collective_checks=self._collective_checks,
        )
