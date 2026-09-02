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
    tagged: list[int | list[int]] = []
    monkeypatch.setattr(native_async, "_tag_current_span_call_idx", tagged.append)

    def async_fn(value: str) -> None:
        background_thread.append(threading.get_ident())
        events.append(value)

    request = native_async.AsyncRequest(
        async_fn=async_fn,
        async_fn_args=("background",),
        finalize_fns=[lambda: events.append("finalize")],
    )
    call_idx = native_async.schedule_async_save(request)
    native_async.maybe_finalize_async_save(blocking=True, terminate=True)

    assert events == ["background", "finalize"]
    assert background_thread != [threading.get_ident()]
    assert native_async.is_empty_async_queue()
    assert tagged == [call_idx, [call_idx]]


def test_next_save_waits_for_previous_save_to_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_async.torch.distributed, "is_initialized", lambda: False)
    previous_started = threading.Event()
    release_previous = threading.Event()
    next_started = threading.Event()
    events: list[str] = []

    def previous_save() -> None:
        previous_started.set()
        if not release_previous.wait(timeout=5):
            raise TimeoutError("test did not release previous save")
        events.append("previous durable")

    native_async.schedule_async_save(
        native_async.AsyncRequest(
            async_fn=previous_save,
            finalize_fns=[lambda: events.append("previous finalized")],
        ),
    )
    assert previous_started.wait(timeout=1)
    assert native_async.is_previous_save_in_progress()

    def start_next_save() -> None:
        native_async.wait_for_previous_save()
        events.append("next started")
        next_started.set()

    next_thread = threading.Thread(target=start_next_save)
    next_thread.start()
    try:
        assert not next_started.wait(timeout=0.05)
        release_previous.set()
        assert next_started.wait(timeout=1)
        assert events == ["previous durable", "previous finalized", "next started"]
        assert native_async.is_empty_async_queue()
        assert not native_async.is_previous_save_in_progress()
    finally:
        release_previous.set()
        next_thread.join(timeout=1)
        native_async.maybe_finalize_async_save(blocking=True, terminate=True)


def test_completed_previous_save_is_not_reported_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_async.torch.distributed, "is_initialized", lambda: False)
    native_async.schedule_async_save(
        native_async.AsyncRequest(async_fn=lambda: None),
    )
    native_async._pending[0].future.result(timeout=1)

    assert not native_async.is_previous_save_in_progress()
    native_async.maybe_finalize_async_save(blocking=True, terminate=True)


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
