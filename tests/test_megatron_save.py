# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101, SLF001

from __future__ import annotations

import importlib.util
import sys
import threading
from concurrent.futures import Future
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


@pytest.mark.parametrize(
    ("dp_gtp_remat_group", "expected_dp_group"),
    [("dp-gtp-remat", "dp-gtp-remat"), (None, "dp")],
)
def test_dataloader_state_is_saved_before_snapshot_submission(
    save: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dp_gtp_remat_group: str | None,
    expected_dp_group: str,
) -> None:
    events: list[object] = []

    class Root:
        startedAt = 0  # noqa: N815

        def save_async(self, _iteration: int, _checkpoint_dir: Path) -> None:
            events.append("save_async")

        def wait_for_async_completion(self) -> None:
            events.append("client_send_complete")

    def maybe_save_dataloader_state(iterator: object, iteration: int, save_path: object, **groups: object) -> None:
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
    monkeypatch.setattr(save, "maybe_save_dataloader_state", maybe_save_dataloader_state)
    monkeypatch.setattr(save, "register_state_dict", lambda _state, _iteration: events.append("register") or Root())
    monkeypatch.setattr(save, "_write_checkpoint_tracker", lambda _args, _iteration: events.append("tracker"))
    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        save.checkpoint,
        "wait_for_ckpt_finalize",
        lambda _session, _iteration: events.append("wait_for_finalize"),
    )

    save._save_checkpoint_impl(save_args, 123)

    assert events == [
        (
            "dataloader_state",
            "iterator",
            20,
            tmp_path,
            {"tp_group": "tp", "pp_group": "pp", "dp_group": expected_dp_group, "cp_group": "cp"},
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

    def load_checkpoint(
        *,
        tp_group=None,  # noqa: ANN001
        pp_group=None,  # noqa: ANN001
        dp_cp_group=None,  # noqa: ANN001
        dp_group=None,  # noqa: ANN001
        rng_state_key_prefix="",  # noqa: ANN001
    ) -> dict:
        del tp_group, pp_group, dp_cp_group, dp_group, rng_state_key_prefix
        return save._prepare_registration_state_dict(state, 0)

    prepared = save._wrap_load_checkpoint(load_checkpoint)(
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
    assert prepared == {**state, "rng_state": rng_state}
    assert "rng_state" not in state
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


def test_common_state_padding_is_local_to_megatron_serialization(save: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = iter((b"a" * 100, b"b" * 200, b"c" * 2048))

    monkeypatch.setattr(save.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(save.torch, "save", lambda _value, buffer: buffer.write(next(payloads)))
    state = {"rerun_state_machine": "rerun handler"}

    first = save._prepare_registration_state_dict(state, 0)[save._DCP_COMMON_STATE_KEY]
    second = save._prepare_registration_state_dict(state, 1)[save._DCP_COMMON_STATE_KEY]
    expected_capacity = save._COMMON_STATE_ALIGNMENT

    assert len(first) == len(second) == expected_capacity
    assert first[:100] == b"a" * 100
    assert second[:200] == b"b" * 200
    with pytest.raises(
        RuntimeError,
        match=rf"common state serialized to 2048 bytes, exceeding its registered {expected_capacity}-byte capacity",
    ):
        save._prepare_registration_state_dict(state, 2)


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
