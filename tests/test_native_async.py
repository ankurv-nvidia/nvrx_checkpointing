# ruff: noqa: INP001, S101, SLF001

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from nvrx_checkpointing import native_async


def test_native_async_request_runs_and_finalizes_without_external_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_async.torch.distributed, "is_initialized", lambda: False)
    events: list[str] = []
    background_thread: list[int] = []

    def async_fn(value: str) -> None:
        background_thread.append(threading.get_ident())
        events.append(value)

    request = native_async.AsyncRequest(
        async_fn=async_fn,
        async_fn_args=("background",),
        finalize_fns=[lambda: events.append("finalize")],
    )
    native_async.schedule_async_save(request)
    native_async.maybe_finalize_async_save(blocking=True, terminate=True)

    assert events == ["background", "finalize"]
    assert background_thread != [threading.get_ident()]
    assert native_async.is_empty_async_queue()


def test_native_async_request_propagates_background_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_async.torch.distributed, "is_initialized", lambda: False)

    def fail() -> None:
        raise ValueError("write failed")

    native_async.schedule_async_save(native_async.AsyncRequest(async_fn=fail))
    with pytest.raises(RuntimeError, match="failed during background execution") as error:
        native_async.maybe_finalize_async_save(blocking=True)

    assert isinstance(error.value.__cause__, ValueError)
    # The failed request remains at the head by design; remove it so this
    # isolated test does not leak global queue state into the next test.
    native_async._pending.popleft()
    native_async.maybe_finalize_async_save(terminate=True)


def test_native_async_queue_rejects_external_request_types() -> None:
    with pytest.raises(TypeError, match="rejected external request type"):
        native_async.schedule_async_save(object())  # type: ignore[arg-type]


def test_native_metadata_reader_caches_by_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reads: list[Path] = []
    native_async.NativeCachedMetadataFileSystemReader._metadata_cache.clear()

    def read_metadata(reader: native_async.FileSystemReader) -> object:
        reads.append(Path(reader.path))
        return object()

    monkeypatch.setattr(native_async.FileSystemReader, "read_metadata", read_metadata)
    first = native_async.get_filesystem_reader(tmp_path, cache_metadata=True)
    second = native_async.get_filesystem_reader(tmp_path, cache_metadata=True)

    assert first.read_metadata() is second.read_metadata()
    assert reads == [tmp_path]
