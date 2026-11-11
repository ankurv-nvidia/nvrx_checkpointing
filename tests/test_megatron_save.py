# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101, SLF001

from __future__ import annotations

import importlib.util
import shutil
import sys
import threading
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import nvrx_checkpointing.megatron as nvrx_megatron
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def save(monkeypatch: pytest.MonkeyPatch) -> Generator[ModuleType]:
    state_dict_utils = _module("megatron.core.dist_checkpointing.state_dict_utils", save_preprocess=lambda state, **_kwargs: (state, {}))
    msc_utils = _module("megatron.core.msc_utils", maybe_msc=SimpleNamespace(open=open))
    rerun = _module("megatron.core.rerun_state_machine", get_rerun_state_machine=SimpleNamespace)
    core_utils = _module("megatron.core.utils", unwrap_model=lambda model: model)
    checkpointing = _module(
        "megatron.training.checkpointing",
        _build_sharded_state_dict_metadata=lambda *_args, **_kwargs: {},
        generate_state_dict=lambda *_args, **_kwargs: {},
        get_checkpoint_name=lambda *_args, **_kwargs: "checkpoint",
        get_checkpoint_tracker_filename=lambda *_args, **_kwargs: "tracker",
        get_rng_state=lambda *_args, **_kwargs: {},
        maybe_save_dataloader_state=lambda *_args, **_kwargs: None,
    )
    global_vars = _module("megatron.training.global_vars", get_args=SimpleNamespace)
    training_utils = _module("megatron.training.utils", print_rank_0=lambda _message: None)
    core = _module(
        "megatron.core",
        mpu=SimpleNamespace(
            get_tensor_model_parallel_group=lambda: None,
            get_pipeline_model_parallel_group=lambda: None,
            get_data_parallel_group=lambda **_kwargs: None,
        ),
    )
    core.__path__ = []
    training = _module("megatron.training")
    training.__path__ = []
    training.checkpointing = checkpointing
    megatron = _module("megatron", core=core, training=training)
    megatron.__path__ = []
    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.dist_checkpointing.state_dict_utils": state_dict_utils,
        "megatron.core.msc_utils": msc_utils,
        "megatron.core.rerun_state_machine": rerun,
        "megatron.core.utils": core_utils,
        "megatron.training": training,
        "megatron.training.checkpointing": checkpointing,
        "megatron.training.global_vars": global_vars,
        "megatron.training.utils": training_utils,
        "nemo": None,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    package_dir = Path(nvrx_megatron.__path__[0])
    spec = importlib.util.spec_from_file_location("nvrx_checkpointing.megatron._save_test", package_dir / "save.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    yield module
    module._root = None
    module._pending = None
    if module._executor is not None:
        module._executor.shutdown(wait=True)


def _queue(
    save: ModuleType,
    async_fn: Callable[[], None],
    finalize_fn: Callable[[], None] = lambda: None,
) -> None:
    future = save._get_executor().submit(async_fn)
    save._pending = save._PendingSave(future, finalize_fn)


@pytest.fixture
def retention(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Generator[SimpleNamespace]:
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("100")
    args = SimpleNamespace(save=tmp_path, save_retain_interval=10000, log_progress=True)
    deleted: list[tuple[int, str, bool]] = []
    threads: list[threading.Thread] = []
    thread_type = threading.Thread

    def checkpoint_name(directory: Path, iteration: int, **_kwargs: object) -> Path:
        return directory / f"iter_{iteration:07d}"

    def delete(directory: Path, iteration: int, log_progress: bool) -> None:
        deleted.append((iteration, tracker.read_text(), log_progress))
        shutil.rmtree(checkpoint_name(directory, iteration))

    def start_thread(**kwargs: object) -> threading.Thread:
        thread = thread_type(**kwargs)
        threads.append(thread)
        return thread

    def wait() -> None:
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    monkeypatch.setattr(save, "get_checkpoint_name", checkpoint_name)
    monkeypatch.setattr(save, "get_checkpoint_tracker_filename", lambda _directory: tracker)
    monkeypatch.setattr(save.checkpointing_module, "_async_delete_checkpoint_impl", delete, raising=False)
    monkeypatch.setattr(save.threading, "Thread", start_thread)
    monkeypatch.setattr(save.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 0)
    for iteration in (20, 80, 100, 120):
        checkpoint_name(tmp_path, iteration).mkdir()
    yield SimpleNamespace(args=args, tracker=tracker, deleted=deleted, threads=threads, wait=wait, checkpoint=checkpoint_name)
    wait()


@pytest.mark.parametrize("initialized", [True, False])
def test_retention_deletes_only_previous_tracked_checkpoint_after_publication(
    save: ModuleType,
    retention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    initialized: bool,
) -> None:
    monkeypatch.setattr(save.torch.distributed, "is_initialized", lambda: initialized)
    if not initialized:
        monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: pytest.fail("uninitialized process has no rank"))

    save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "120"
    assert retention.deleted == [(100, "120", True)]
    assert not retention.checkpoint(retention.args.save, 100).exists()
    assert all(retention.checkpoint(retention.args.save, iteration).is_dir() for iteration in (20, 80, 120))
    assert all(not thread.daemon for thread in retention.threads)


@pytest.mark.parametrize(
    ("previous", "current"),
    [(0, 20), (10000, 10020), (120, 120), (140, 120)],
)
def test_retention_preserves_milestones_and_nonolder_checkpoints(
    save: ModuleType,
    retention: SimpleNamespace,
    previous: int,
    current: int,
) -> None:
    retention.tracker.write_text(str(previous))

    save._write_checkpoint_tracker(retention.args, current)
    retention.wait()

    assert retention.tracker.read_text() == str(current)
    assert retention.deleted == []
    assert retention.threads == []


def test_retention_first_checkpoint_has_no_previous_checkpoint(
    save: ModuleType,
    retention: SimpleNamespace,
) -> None:
    retention.tracker.unlink()

    save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "120"
    assert retention.deleted == []


@pytest.mark.parametrize("explicit", [True, False])
def test_disabled_retention_does_not_read_previous_tracker(
    save: ModuleType,
    retention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    if explicit:
        retention.args.save_retain_interval = None
    else:
        del retention.args.save_retain_interval
    retention.tracker.write_text("release")

    def write_only(path: Path, mode: str):  # noqa: ANN202
        assert mode == "w"
        return path.open(mode)

    monkeypatch.setattr(save.maybe_msc, "open", write_only)

    save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "120"
    assert retention.deleted == []
    assert retention.checkpoint(retention.args.save, 100).is_dir()


def test_retention_preserves_symlinked_previous_checkpoint(
    save: ModuleType,
    retention: SimpleNamespace,
    tmp_path: Path,
) -> None:
    previous = retention.checkpoint(retention.args.save, 100)
    previous.rmdir()
    target = tmp_path / "retained-external-checkpoint"
    target.mkdir()
    previous.symlink_to(target, target_is_directory=True)

    save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "120"
    assert previous.is_symlink()
    assert target.is_dir()
    assert retention.deleted == []


def test_nonzero_global_rank_does_not_publish_or_delete_checkpoints(
    save: ModuleType,
    retention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 4)

    save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "100"
    assert retention.deleted == []


def test_retention_does_not_delete_when_tracker_publication_fails(
    save: ModuleType,
    retention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(path: Path, mode: str):  # noqa: ANN202
        if mode == "w":
            raise OSError("tracker publication failed")
        return path.open(mode)

    monkeypatch.setattr(save.maybe_msc, "open", fail_write)

    with pytest.raises(OSError, match="tracker publication failed"):
        save._write_checkpoint_tracker(retention.args, 120)
    retention.wait()

    assert retention.tracker.read_text() == "100"
    assert retention.checkpoint(retention.args.save, 100).is_dir()
    assert retention.deleted == []


def _complete_async_retention_save(
    save: ModuleType,
    retention: SimpleNamespace,
    future: Future[None],
    wait: tuple[Callable[..., None], tuple[object, ...]],
    failure: bool,
) -> None:
    save.maybe_finalize_async_save()
    assert retention.tracker.read_text() == "100"
    assert retention.deleted == []
    fn, args = wait
    try:
        future.set_result(fn(*args))
    except RuntimeError as exc:
        future.set_exception(exc)
    if failure:
        with pytest.raises(RuntimeError, match="failed during background execution"):
            save.maybe_finalize_async_save()
    else:
        save.maybe_finalize_async_save()


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_retention_waits_for_successful_durable_completion(
    save: ModuleType,
    retention: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_save: bool,
    failure: bool,
) -> None:
    future: Future[None] = Future()
    waits: list[tuple[Callable[..., None], tuple[object, ...]]] = []
    retention.args.async_save = async_save
    retention.args.use_distributed_optimizer = False
    save_args = save.SaveCheckpointArgs(120, object(), None, None, 0)

    class Root:
        startedAt = 0  # noqa: N815

        @staticmethod
        def save_async(_iteration: int, _checkpoint_dir: Path) -> None:
            assert retention.tracker.read_text() == "100"

        @staticmethod
        def wait_for_async_completion() -> None:
            assert retention.tracker.read_text() == "100"

    class ManualExecutor:
        @staticmethod
        def submit(fn: Callable[..., None], *args: object) -> Future[None]:
            waits.append((fn, args))
            return future

    def wait_for_finalize(_session: int, _iteration: int) -> None:
        assert retention.tracker.read_text() == "100"
        assert retention.checkpoint(retention.args.save, 100).is_dir()
        if failure:
            raise RuntimeError("checkpoint durability failed")

    monkeypatch.setattr(save, "get_args", lambda: retention.args)
    monkeypatch.setattr(save, "get_rng_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "get_rerun_state_machine", lambda: SimpleNamespace(state_dict=lambda **_kwargs: {}))
    monkeypatch.setattr(save, "generate_state_dict", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: Root())
    monkeypatch.setattr(save, "_get_executor", ManualExecutor)
    monkeypatch.setattr(save.checkpoint, "wait_for_ckpt_finalize", wait_for_finalize)

    if not async_save and failure:
        with pytest.raises(RuntimeError, match="checkpoint durability failed"):
            save._save_checkpoint_impl(save_args, 123)
    else:
        save._save_checkpoint_impl(save_args, 123)
    if async_save:
        _complete_async_retention_save(save, retention, future, waits.pop(), failure)
    retention.wait()

    if failure:
        assert retention.tracker.read_text() == "100"
        assert retention.checkpoint(retention.args.save, 100).is_dir()
        assert retention.deleted == []
    else:
        assert retention.tracker.read_text() == "120"
        assert not retention.checkpoint(retention.args.save, 100).exists()
        assert retention.deleted == [(100, "120", True)]


def test_checkpoint_span_preserves_names_and_attributes(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def managed_span(scope: str, name: str, **attributes: object) -> nullcontext[None]:
        calls.append((scope, name, attributes))
        return nullcontext()

    monkeypatch.setattr(save, "managed_span", managed_span)

    with save._checkpoint_span("nvrx.checkpoint.save", 42):
        pass

    assert calls == [
        (
            "checkpoint",
            "nvrx.checkpoint.save",
            {"is_goodput_span": True, "megatron.iteration": 42, "nvrx.snapshot_id": 42},
        ),
    ]


def test_native_async_save_runs_and_finalizes_off_thread(save: ModuleType) -> None:
    events: list[str] = []
    background_thread: list[int] = []

    def async_fn() -> None:
        background_thread.append(threading.get_ident())
        events.append("background")

    _queue(save, async_fn, lambda: events.append("finalize"))
    save.maybe_finalize_async_save(blocking=True, terminate=True)

    assert events == ["background", "finalize"]
    assert background_thread != [threading.get_ident()]


def test_next_save_waits_for_previous_save_to_finalize(save: ModuleType) -> None:
    previous_started = threading.Event()
    release_previous = threading.Event()
    events: list[str] = []

    def previous_save() -> None:
        previous_started.set()
        if not release_previous.wait(timeout=5):
            raise TimeoutError("test did not release previous save")
        events.append("previous durable")

    _queue(save, previous_save, lambda: events.append("previous finalized"))
    assert previous_started.wait(timeout=1)

    waiter = threading.Thread(target=save.maybe_finalize_async_save, kwargs={"blocking": True})
    waiter.start()
    try:
        waiter.join(timeout=0.05)
        assert waiter.is_alive()
        release_previous.set()
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        assert events == ["previous durable", "previous finalized"]
    finally:
        release_previous.set()
        waiter.join(timeout=1)


def test_native_async_save_propagates_background_failure(save: ModuleType) -> None:
    def fail() -> None:
        raise ValueError("write failed")

    _queue(save, fail)
    with pytest.raises(RuntimeError, match="failed during background execution") as error:
        save.maybe_finalize_async_save(blocking=True)

    assert isinstance(error.value.__cause__, ValueError)


def test_async_finalize_wait_starts_after_client_send(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Root:
        startedAt = 0  # noqa: N815

        def save_async(self, _iteration: int, _checkpoint_dir: Path) -> None:
            events.append("save_async")

        def wait_for_async_completion(self) -> None:
            events.append("client_send_complete")

    class ImmediateExecutor:
        @staticmethod
        def submit(fn: Callable[..., None], *args: object) -> Future[None]:
            events.append("submit_finalize_wait")
            future: Future[None] = Future()
            future.set_result(fn(*args))
            return future

    args = SimpleNamespace(use_distributed_optimizer=False, async_save=True, save=tmp_path)
    save_args = save.SaveCheckpointArgs(
        iteration=40,
        model=object(),
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
    )
    monkeypatch.setattr(save, "get_args", lambda: args)
    monkeypatch.setattr(save, "get_rng_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "get_rerun_state_machine", lambda: SimpleNamespace(state_dict=lambda **_kwargs: {}))
    monkeypatch.setattr(save, "generate_state_dict", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "get_checkpoint_name", lambda *_args, **_kwargs: tmp_path / "iter_0000040")
    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: Root())
    monkeypatch.setattr(save, "_get_executor", ImmediateExecutor)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        save.checkpoint,
        "wait_for_ckpt_finalize",
        lambda _session, _iteration: events.append("wait_for_finalize"),
    )

    save._save_checkpoint_impl(save_args, 123)

    assert events == ["save_async", "client_send_complete", "submit_finalize_wait", "wait_for_finalize"]


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("failure", ["checkpoint shard write failed: Input/output error", "checkpoint finalize timed out"])
def test_finalize_failure_preserves_tracker_and_prevents_success(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    async_save: bool,
    failure: str,
) -> None:
    messages: list[str] = []
    deleted: list[tuple[object, ...]] = []
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("30")
    previous = tmp_path / "iter_0000030"
    previous.mkdir()
    previous_payload = previous / "checkpoint"
    previous_payload.write_text("last successful checkpoint")
    finalize_error = RuntimeError(failure)

    class Root:
        startedAt = 0  # noqa: N815

        @staticmethod
        def save_async(_iteration: int, _checkpoint_dir: Path) -> None:
            return

        @staticmethod
        def wait_for_async_completion() -> None:
            return

    def fail_finalize(session: int, iteration: int) -> None:
        assert (session, iteration) == (0, 40)
        raise finalize_error

    args = SimpleNamespace(use_distributed_optimizer=False, async_save=async_save, save=tmp_path, save_retain_interval=10000)
    save_args = save.SaveCheckpointArgs(
        iteration=40,
        model=object(),
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
    )
    monkeypatch.setattr(save, "get_args", lambda: args)
    monkeypatch.setattr(save, "get_rerun_state_machine", lambda: SimpleNamespace(state_dict=lambda **_kwargs: {}))
    monkeypatch.setattr(save, "get_checkpoint_name", lambda *_args, **_kwargs: tmp_path / "iter_0000040")
    monkeypatch.setattr(save, "get_checkpoint_tracker_filename", lambda _path: tracker)
    monkeypatch.setattr(save, "maybe_msc", SimpleNamespace(open=Path.open))
    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: Root())
    monkeypatch.setattr(save, "print_rank_0", messages.append)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(save.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(save.checkpoint, "wait_for_ckpt_finalize", fail_finalize)
    monkeypatch.setattr(save.checkpointing_module, "_async_delete_checkpoint_impl", lambda *args: deleted.append(args), raising=False)

    if async_save:
        save._save_checkpoint_impl(save_args, 123)
        with pytest.raises(RuntimeError) as error:
            save.maybe_finalize_async_save(blocking=True)
        assert error.value.__cause__ is finalize_error
    else:
        with pytest.raises(RuntimeError) as error:
            save._save_checkpoint_impl(save_args, 123)
        assert error.value is finalize_error
    assert tracker.read_text() == "30"
    assert previous_payload.read_text() == "last successful checkpoint"
    assert deleted == []
    assert not any("successfully saved checkpoint" in message for message in messages)


@pytest.mark.parametrize(
    ("dp_gtp_remat_group", "expected_dp_group"),
    [("dp-gtp-remat", "dp-gtp-remat"), (None, "dp")],
)
@pytest.mark.parametrize("supports_cp_group", [True, False])
def test_dataloader_state_is_saved_before_snapshot_submission(  # noqa: PLR0913, PLR0917
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dp_gtp_remat_group: str | None,
    expected_dp_group: str,
    supports_cp_group: bool,
) -> None:
    events: list[object] = []

    class Root:
        startedAt = 0  # noqa: N815

        def save_async(self, _iteration: int, _checkpoint_dir: Path) -> None:
            events.append("save_async")

        def wait_for_async_completion(self) -> None:
            events.append("client_send_complete")

    def current_dataloader_state(  # noqa: PLR0913
        iterator: object,
        iteration: int,
        save_path: object,
        *,
        tp_group: object = None,
        pp_group: object = None,
        dp_group: object = None,
        cp_group: object = None,
    ) -> None:
        groups = {"tp_group": tp_group, "pp_group": pp_group, "dp_group": dp_group, "cp_group": cp_group}
        events.append(("dataloader_state", iterator, iteration, save_path, groups))

    def legacy_dataloader_state(  # noqa: PLR0913
        iterator: object,
        iteration: int,
        save_path: object,
        *,
        tp_group: object = None,
        pp_group: object = None,
        dp_group: object = None,
    ) -> None:
        groups = {"tp_group": tp_group, "pp_group": pp_group, "dp_group": dp_group}
        events.append(("dataloader_state", iterator, iteration, save_path, groups))

    args = SimpleNamespace(use_distributed_optimizer=False, async_save=False, save=tmp_path, dataloader_save=tmp_path)
    save_args = save.SaveCheckpointArgs(
        iteration=20,
        model=object(),
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        train_data_iterator="iterator",
        tp_group="tp",
        pp_group="pp",
        dp_group="dp",
        dp_gtp_remat_group=dp_gtp_remat_group,
        cp_group="cp",
    )
    monkeypatch.setattr(save, "get_args", lambda: args)
    monkeypatch.setattr(save, "get_rng_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "get_rerun_state_machine", lambda: SimpleNamespace(state_dict=lambda **_kwargs: {}))
    monkeypatch.setattr(save, "generate_state_dict", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(save, "get_checkpoint_name", lambda *_args, **_kwargs: tmp_path / "iter_0000020")
    dataloader_state = current_dataloader_state if supports_cp_group else legacy_dataloader_state
    monkeypatch.setattr(save, "maybe_save_dataloader_state", dataloader_state)
    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: events.append("register") or Root())
    monkeypatch.setattr(save, "_write_checkpoint_tracker", lambda _args, _iteration: events.append("tracker"))
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        save.checkpoint,
        "wait_for_ckpt_finalize",
        lambda _session, _iteration: events.append("wait_for_finalize"),
    )

    save._save_checkpoint_impl(save_args, 123)

    expected_groups = {"tp_group": "tp", "pp_group": "pp", "dp_group": expected_dp_group}
    if supports_cp_group:
        expected_groups["cp_group"] = "cp"
    assert events == [
        (
            "dataloader_state",
            "iterator",
            20,
            tmp_path,
            expected_groups,
        ),
        "register",
        "save_async",
        "client_send_complete",
        "wait_for_finalize",
        "tracker",
    ]


def test_register_state_dict_reuses_the_process_global_root(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    class Root:
        def __init__(self) -> None:
            self.registered = []

        def register_state_dict(self, state_dict) -> None:  # noqa: ANN001
            self.registered.append(state_dict)

    root = Root()
    save._root = root
    monkeypatch.setattr(save, "_prepare_registration_state_dict", lambda state, iteration: {"prepared": state, "iteration": iteration})
    loaded_state = {"model.weight": "loaded tensor"}
    saved_state = {"model.weight": "saved tensor"}

    assert save.register_state_dict(loaded_state, 0) is root
    assert save.register_state_dict(saved_state, 12) is root

    assert root.registered == [
        {"prepared": loaded_state, "iteration": 0},
        {"prepared": saved_state, "iteration": 12},
    ]
    assert save._root is root


def test_load_registration_synthesizes_rng_with_role_specific_groups(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    prepared = []
    rng_state = {"sharded": "placeholder rng handler"}
    monkeypatch.setattr(
        save,
        "get_rng_state",
        lambda *args, **kwargs: calls.append((args, kwargs)) or rng_state,
    )
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    state = {
        "args": SimpleNamespace(no_load_rng=True, no_save_rng=False),
        "model": "model handler",
        "rerun_state_machine": "rerun handler",
    }

    class Root:
        @staticmethod
        def wait_for_async_completion() -> None:
            return

        @staticmethod
        def finalize_registry() -> None:
            return

    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: Root())

    def load_checkpoint(  # noqa: PLR0913
        ddp_model,  # noqa: ANN001
        optimizer,  # noqa: ANN001
        opt_param_scheduler,  # noqa: ANN001
        *,
        tp_group=None,  # noqa: ANN001
        pp_group=None,  # noqa: ANN001
        dp_cp_group=None,  # noqa: ANN001
        dp_group=None,  # noqa: ANN001
        rng_state_key_prefix="",  # noqa: ANN001
    ) -> tuple[int, int]:
        del ddp_model, optimizer, opt_param_scheduler, tp_group, pp_group, dp_cp_group, dp_group, rng_state_key_prefix
        prepared.append(save._prepare_registration_state_dict(state, 0))
        return 0, 0

    result = save._wrap_load_checkpoint(load_checkpoint)(
        "model",
        "optimizer",
        "scheduler",
        tp_group="tp",
        pp_group="pp",
        dp_cp_group="dp-cp",
        dp_group="dp",
        rng_state_key_prefix="language.",
    )

    assert calls == [
        (
            ("torch_dist", "tp", "pp"),
            {"dp_cp_group": "dp-cp", "dp_group": "dp", "key_prefix": "language."},
        ),
    ]
    assert result == (0, 0)
    assert prepared == [{**state, "rng_state": rng_state}]
    assert "rng_state" not in state
    assert save._LOAD_CONTEXT is None


@pytest.mark.parametrize(
    ("fp4_param_gather", "fp8_param_gather", "fp8_recipe"),
    [
        (False, False, None),
        (True, False, None),
        (False, True, "mxfp8"),
        (False, True, "tensorwise"),
    ],
)
def test_load_registers_live_state_after_loading(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fp4_param_gather: bool,
    fp8_param_gather: bool,
    fp8_recipe: str | None,
) -> None:
    events = []
    args = SimpleNamespace(
        fp4_param_gather=fp4_param_gather,
        fp8_param_gather=fp8_param_gather,
        fp8_recipe=fp8_recipe,
    )
    metadata = {"dp_cp_group": "dp-cp", "checkpoint_version": 3.1}
    monkeypatch.setattr(save, "get_args", lambda: args)
    monkeypatch.setattr(save, "_build_sharded_state_dict_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)

    @contextmanager
    def preserve_quantized_tensors() -> Generator[None]:
        events.append("preserve begin")
        yield
        events.append("preserve end")

    def generate_load_state(*args: object, **kwargs: object) -> dict[str, object]:
        events.append(("generate load", args, kwargs, save._LOAD_CONTEXT))
        return {"model": "load state"}

    def generate_live_state(*args: object, **kwargs: object) -> dict[str, object]:
        events.append(("generate live", args, kwargs, save._LOAD_CONTEXT))
        return {"model": "live state"}

    class Root:
        @staticmethod
        def wait_for_async_completion() -> None:
            events.append("wait")

        @staticmethod
        def finalize_registry() -> None:
            events.append("finalize")

    monkeypatch.setattr(save, "preserve_quantized_tensors", preserve_quantized_tensors)
    monkeypatch.setattr(save, "generate_state_dict", generate_live_state)
    monkeypatch.setattr(
        save,
        "register_state_dict",
        lambda state, iteration: events.append(("register", state, iteration)) or Root(),
    )
    save.checkpointing_module.generate_state_dict = generate_load_state

    def load_checkpoint(
        ddp_model: object,
        optimizer: object,
        opt_param_scheduler: object,
        tp_group: object = None,
        pp_group: object = None,
        dp_cp_group: object = None,
        dp_group: object = None,
        rng_state_key_prefix: str = "",
    ) -> tuple[int, int]:
        del ddp_model, optimizer, opt_param_scheduler, tp_group, pp_group, dp_cp_group, dp_group, rng_state_key_prefix
        assert save.checkpointing_module.generate_state_dict("template") == {"model": "load state"}
        return 12, 34

    result = save._wrap_load_checkpoint(load_checkpoint)(
        "model",
        "optimizer",
        "scheduler",
        "tp",
        "pp",
        "dp-cp",
        "dp",
        "language.",
    )

    assert result == (12, 34)
    load_event = (
        "generate load",
        ("template",),
        {},
        {
            "tp_group": "tp",
            "pp_group": "pp",
            "dp_cp_group": "dp-cp",
            "dp_group": "dp",
            "rng_state_key_prefix": "language.",
        },
    )
    expected = [
        "preserve begin",
        load_event,
        "preserve end",
    ]
    expected.extend(
        [
            "preserve begin",
            (
                "generate live",
                (args, "model", "optimizer", "scheduler", None),
                {
                    "iteration": 12,
                    "optim_sd_kwargs": {"metadata": metadata},
                    "model_sd_kwargs": {"metadata": metadata},
                    "rerun_state": None,
                },
                {
                    "tp_group": "tp",
                    "pp_group": "pp",
                    "dp_cp_group": "dp-cp",
                    "dp_group": "dp",
                    "rng_state_key_prefix": "language.",
                },
            ),
            "preserve end",
            ("register", {"model": "live state", "num_floating_point_operations_so_far": 34}, 0),
            "wait",
            "finalize",
        ],
    )
    assert events == expected
    assert save.checkpointing_module.generate_state_dict is generate_load_state
    assert save._LOAD_CONTEXT is None


def test_load_without_checkpoint_leaves_registration_to_first_save(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(save, "get_args", SimpleNamespace)
    monkeypatch.setattr(save, "generate_state_dict", lambda *_args, **_kwargs: events.append("generate live"))
    monkeypatch.setattr(save, "register_state_dict", lambda *_args: events.append("register"))
    load_generate_state_dict = save.checkpointing_module.generate_state_dict

    def load_checkpoint(ddp_model: object, optimizer: object, opt_param_scheduler: object) -> tuple[int, int]:
        del ddp_model, optimizer, opt_param_scheduler
        return 0, 0

    assert save._wrap_load_checkpoint(load_checkpoint)("model", "optimizer", "scheduler") == (0, 0)
    assert events == []
    assert save.checkpointing_module.generate_state_dict is load_generate_state_dict
    assert save._LOAD_CONTEXT is None


def test_load_registration_warns_and_uses_singleton_dp_groups_when_rng_is_unexpectedly_missing(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_rng_state(_format: str, tp_group: object, pp_group: object, **kwargs: object) -> dict[str, str]:
        assert tp_group is None
        assert pp_group is None
        groups = (kwargs["dp_cp_group"], kwargs["dp_group"])
        assert all(group.rank() == 0 and group.size() == 1 for group in groups)
        return {"sharded": "placeholder rng handler"}

    monkeypatch.setattr(save, "get_rng_state", get_rng_state)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    state = {
        "args": SimpleNamespace(no_load_rng=False, no_save_rng=False),
        "rerun_state_machine": "rerun handler",
    }

    with pytest.warns(RuntimeWarning, match="omitted rng_state without no_load_rng"):
        save._prepare_registration_state_dict(state, 0)


def test_registration_adds_the_stable_rerun_state_when_load_omits_it(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    rerun_state = {"sharded": "rerun handler"}
    rerun = SimpleNamespace(
        state_dict=lambda **kwargs: calls.append(kwargs) or rerun_state,
    )
    monkeypatch.setattr(save, "get_rerun_state_machine", lambda: rerun)
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    state = {"model": "model handler"}

    prepared = save._prepare_registration_state_dict(state, 0)
    prepared_save = save._prepare_registration_state_dict(
        {"model": "fresh model handler", "rerun_state_machine": rerun_state},
        12,
    )

    assert prepared == {"model": "model handler", "rerun_state_machine": rerun_state}
    assert prepared_save == {"model": "fresh model handler", "rerun_state_machine": rerun_state}
    assert state == {"model": "model handler"}
    assert calls == [{"data_iterator": None, "ckpt_format": "torch_dist", "force": True}]


def test_registration_adds_rng_state_with_default_groups_during_save(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    rng_state = {"sharded": "rng handler"}
    monkeypatch.setattr(save.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(save.mpu, "get_pipeline_model_parallel_group", lambda: "pp")
    monkeypatch.setattr(
        save,
        "get_rng_state",
        lambda *args, **kwargs: calls.append((args, kwargs)) or rng_state,
    )
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    state = {
        "args": SimpleNamespace(no_save_rng=False),
        "model": "model handler",
        "rerun_state_machine": "rerun handler",
    }

    prepared = save._prepare_registration_state_dict(state, 12)

    assert prepared == {**state, "rng_state": rng_state}
    assert state == {
        "args": SimpleNamespace(no_save_rng=False),
        "model": "model handler",
        "rerun_state_machine": "rerun handler",
    }
    assert calls == [(("torch_dist", "tp", "pp"), {})]


def test_registration_respects_no_save_rng(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        save,
        "get_rng_state",
        lambda *_args, **_kwargs: pytest.fail("RNG state should not be generated"),
    )
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    state = {
        "args": SimpleNamespace(no_save_rng=True),
        "model": "model handler",
        "rerun_state_machine": "rerun handler",
    }

    assert save._prepare_registration_state_dict(state, 0) == state


def test_common_state_serialization_accepts_dynamic_sizes(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = iter((b"a" * 100, b"b" * 200, b"c" * 2048))

    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(save.torch, "save", lambda _value, buffer: buffer.write(next(payloads)))
    state = {"rerun_state_machine": "rerun handler"}

    first = save._prepare_registration_state_dict(state, 0)[save._DCP_COMMON_STATE_KEY]
    second = save._prepare_registration_state_dict(state, 1)[save._DCP_COMMON_STATE_KEY]
    third = save._prepare_registration_state_dict(state, 2)[save._DCP_COMMON_STATE_KEY]

    assert first == b"a" * 100
    assert second == b"b" * 200
    assert third == b"c" * 2048


def test_rng_sharded_object_padding_is_local_to_megatron_serialization(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShardedObject:
        replica_id = 0

        def __init__(self, data: object) -> None:
            self.data = data
            self.unique_key = "rng_state/shard_0_1"

    payloads = iter((b"a" * 100, b"b" * 200, b"c" * 2048))
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(save.torch, "save", lambda _value, buffer: buffer.write(next(payloads)))

    first = ShardedObject({"iteration": 0})
    second = ShardedObject({"iteration": 1})
    save._prepare_registration_state_dict({"rng_state": first, "rerun_state_machine": {}}, 0)
    save._prepare_registration_state_dict({"rng_state": second, "rerun_state_machine": {}}, 1)
    expected_capacity = save._COMMON_STATE_ALIGNMENT

    first_blob = getattr(first, save._SERIALIZED_SHARDED_OBJECT_ATTR)
    second_blob = getattr(second, save._SERIALIZED_SHARDED_OBJECT_ATTR)
    assert len(first_blob) == len(second_blob) == expected_capacity
    assert first_blob[:100] == b"a" * 100
    assert second_blob[:200] == b"b" * 200

    with pytest.raises(
        RuntimeError,
        match=rf"sharded object 'rng_state/shard_0_1' serialized to 2048 bytes, exceeding its registered {expected_capacity}-byte capacity",
    ):
        save._prepare_registration_state_dict(
            {"rng_state": ShardedObject({"iteration": 2}), "rerun_state_machine": {}},
            2,
        )


@pytest.mark.parametrize("replica_id", [1, (0, 1, 0)])
def test_rng_sharded_object_serialization_skips_non_main_replicas(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    replica_id: int | tuple[int, ...],
) -> None:
    class ShardedObject:
        def __init__(self) -> None:
            self.data = {"iteration": 1}
            self.unique_key = "rng_state/shard_0_1"
            self.replica_id = replica_id

    value = ShardedObject()
    serialized = []
    monkeypatch.setattr(save.torch, "save", lambda *_args: serialized.append(True))

    save._prepare_rng_sharded_object_blobs(value)

    assert serialized == []
    assert not hasattr(value, save._SERIALIZED_SHARDED_OBJECT_ATTR)
