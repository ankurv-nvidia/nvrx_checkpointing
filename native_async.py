# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Native-provider async lifecycle with no NVRx or MCore checkpoint dependency.

Megatron still owns the training-loop call sites, but the integration redirects
those calls here whenever ``--ckpt-provider native`` is active.  The configured
``async_strategy`` is deliberately ignored: it remains available to the
reference provider and is not used to select an implementation for native
checkpoints.
"""

from __future__ import annotations

import contextlib
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.distributed.checkpoint import FileSystemReader, Metadata

try:
    from nemo.lens.helpers import safe_set_span_attributes
    from opentelemetry import trace
except (ImportError, ModuleNotFoundError):
    safe_set_span_attributes = None
    trace = None

if TYPE_CHECKING:
    import os
    from collections.abc import Callable


@dataclass
class AsyncRequest:
    """A native async operation followed by main-thread finalizers."""

    async_fn: Callable[..., Any] | None
    async_fn_args: tuple[Any, ...] = ()
    finalize_fns: list[Callable[[], Any]] = field(default_factory=list)
    async_fn_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingRequest:
    call_idx: int
    future: Future[Any]
    finalize_fns: list[Callable[[], Any]]


_pending: deque[_PendingRequest] = deque()
_executor: ThreadPoolExecutor | None = None
_next_call_idx = 0


def _tag_current_span_call_idx(call_idx: int | list[int]) -> None:
    if safe_set_span_attributes is None or trace is None:
        return
    safe_set_span_attributes(trace.get_current_span(), {"nvrx.call_idx": call_idx})


class NativeCachedMetadataFileSystemReader(FileSystemReader):
    """PyTorch DCP reader with a repo-owned, path-keyed metadata cache."""

    _metadata_cache: ClassVar[dict[str, Metadata]] = {}

    def __init__(self, path: str | os.PathLike[str], *, cache_metadata: bool = True) -> None:
        super().__init__(path=path)
        self._cache_key = str(Path(path).resolve()) if cache_metadata else None

    def read_metadata(self) -> Metadata:
        if self._cache_key is None:
            return super().read_metadata()
        if self._cache_key not in self._metadata_cache:
            self._metadata_cache[self._cache_key] = super().read_metadata()
        return self._metadata_cache[self._cache_key]


def get_filesystem_reader(
    checkpoint_dir: str | os.PathLike[str],
    *,
    cache_metadata: bool,
) -> FileSystemReader:
    """Return a PyTorch reader without resolving an external async strategy."""
    if cache_metadata:
        return NativeCachedMetadataFileSystemReader(checkpoint_dir)
    return FileSystemReader(checkpoint_dir)


def _get_executor() -> ThreadPoolExecutor:
    global _executor  # noqa: PLW0603
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-ckpt-finalize")
    return _executor


def _run_request(request: AsyncRequest) -> None:
    if request.async_fn is not None:
        request.async_fn(*request.async_fn_args, **request.async_fn_kwargs)


def _collective_device() -> torch.device:
    if torch.distributed.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_ranks(value: bool, *, require_all: bool) -> bool:
    """Reduce a boolean consistently when distributed training is initialized."""
    if not torch.distributed.is_initialized():
        return value
    reduced = torch.tensor(int(value), dtype=torch.int32, device=_collective_device())
    op = torch.distributed.ReduceOp.MIN if require_all else torch.distributed.ReduceOp.MAX
    torch.distributed.all_reduce(reduced, op=op)
    return bool(reduced.item())


def _raise_collective_failure(call_idx: int, error: Exception | None, phase: str) -> None:
    if not _all_ranks(error is not None, require_all=False):
        return
    message = f"native async checkpoint request {call_idx} failed during {phase}"
    if error is not None:
        raise RuntimeError(message) from error
    raise RuntimeError(f"{message} on another rank")


def init_persistent_async_worker(rank: int, mp_mode: str = "spawn") -> None:
    """Warm the repo-owned worker used by native checkpoint finalization."""
    del mp_mode
    _get_executor()
    if rank == 0:
        sys.stderr.write("init_native_async_worker: repo-owned async caller started\n")


def schedule_async_save(request: AsyncRequest) -> int:
    """Schedule a native durability wait without resolving Megatron strategies."""
    global _next_call_idx  # noqa: PLW0603
    if not isinstance(request, AsyncRequest):
        raise TypeError(f"native async queue rejected external request type {type(request).__qualname__}")
    call_idx = _next_call_idx
    _next_call_idx += 1
    future = _get_executor().submit(_run_request, request)
    _pending.append(_PendingRequest(call_idx, future, list(request.finalize_fns)))
    _tag_current_span_call_idx(call_idx)
    return call_idx


def maybe_finalize_async_save(blocking: bool = False, terminate: bool = False) -> None:
    """Finalize completed native requests in the same order on every rank."""
    global _executor  # noqa: PLW0603
    finalized = []
    while _pending:
        request = _pending[0]
        if blocking:
            with contextlib.suppress(Exception):  # synchronized and re-raised below
                request.future.result()
        if not _all_ranks(request.future.done(), require_all=True):
            break

        async_error = request.future.exception()
        _raise_collective_failure(request.call_idx, async_error, "background execution")
        _pending.popleft()

        finalize_error: Exception | None = None
        try:
            for finalize_fn in request.finalize_fns:
                finalize_fn()
        except Exception as error:  # noqa: BLE001 - keep ranks in lockstep before surfacing it
            finalize_error = error
        _raise_collective_failure(request.call_idx, finalize_error, "finalization")
        finalized.append(request.call_idx)

    if finalized:
        _tag_current_span_call_idx(finalized)

    if terminate and _executor is not None:
        if _pending:
            raise RuntimeError("cannot terminate native async worker with pending requests")
        _executor.shutdown(wait=True)
        _executor = None


def wait_for_previous_save() -> None:
    """Finish prior native saves before the caller starts another snapshot."""
    maybe_finalize_async_save(blocking=True)


def is_previous_save_in_progress() -> bool:
    """Return consistently whether any rank's previous save is still running."""
    return bool(_pending) and _all_ranks(not _pending[0].future.done(), require_all=False)


def is_empty_async_queue() -> bool:
    """Return whether this rank has no native requests awaiting finalization."""
    return not _pending


def reset_persistent_async_worker(async_strategy: str | None = None) -> None:
    """Close the native worker; ``async_strategy`` is intentionally unused."""
    del async_strategy
    maybe_finalize_async_save(blocking=True, terminate=True)
