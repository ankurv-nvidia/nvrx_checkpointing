# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101, SLF001

from __future__ import annotations

import importlib.util
import sys
import threading
from contextlib import nullcontext
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
    msc_utils = _module("megatron.core.msc_utils", maybe_msc=SimpleNamespace())
    rerun = _module("megatron.core.rerun_state_machine", get_rerun_state_machine=SimpleNamespace)
    core_utils = _module("megatron.core.utils", unwrap_model=lambda model: model)
    checkpointing = _module(
        "megatron.training.checkpointing",
        _build_sharded_state_dict_metadata=lambda *_args, **_kwargs: {},
        generate_state_dict=lambda *_args, **_kwargs: {},
        get_checkpoint_name=lambda *_args, **_kwargs: "checkpoint",
        get_checkpoint_tracker_filename=lambda *_args, **_kwargs: "tracker",
        get_rng_state=lambda *_args, **_kwargs: {},
    )
    global_vars = _module("megatron.training.global_vars", get_args=SimpleNamespace)
    training_utils = _module("megatron.training.utils", print_rank_0=lambda _message: None)
    core = _module(
        "megatron.core",
        mpu=SimpleNamespace(
            get_tensor_model_parallel_group=lambda: None,
            get_pipeline_model_parallel_group=lambda: None,
        ),
    )
    core.__path__ = []
    training = _module("megatron.training")
    training.__path__ = []
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


def test_warmup_initializes_finalize_executor(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []
    peers = SimpleNamespace(checkpoint_args=lambda: {"local_world_size": 2})
    monkeypatch.setattr(save, "_get_executor", lambda: events.append("executor"))
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(save.torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(save, "detect_peers", lambda _rank, _world_size: peers)
    monkeypatch.setattr(
        save.checkpoint.model_state,
        "warmup",
        lambda rank, world_size, **kwargs: events.append((rank, world_size, kwargs)),
    )

    save.init_persistent_async_worker(3)
    assert events == []

    save.warmup()
    assert events == ["executor", (3, 8, {"local_world_size": 2})]
