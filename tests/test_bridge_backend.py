# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401, ARG001, ARG005, INP001, S101, S102, SLF001

"""Checkpoint backend contracts with isolated Megatron Core and native bindings."""

from __future__ import annotations

import ast
import importlib.util
import pickle
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path | None = None) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: C901
    native = package(monkeypatch, "nvrx_checkpointing", ROOT)
    package(monkeypatch, "nvrx_checkpointing.megatron", ROOT / "megatron")
    for name in ("nemo_rl", "megatron.bridge", "megatron.training"):
        monkeypatch.setitem(sys.modules, name, None)
    torch = package(monkeypatch, "torch")
    torch.distributed = SimpleNamespace(is_initialized=lambda: False)
    torch.save = pickle.dump
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.strategies",
        "megatron.core.dist_checkpointing.strategies.torch",
        "megatron.core.dist_checkpointing.dict_utils",
    ):
        package(monkeypatch, name)

    class TorchDistSaveShardedStrategy:
        def __init__(self, backend: str, version: int) -> None:
            self.backend, self.version = backend, version

    def nested_values(value: object) -> Any:
        children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else None
        if children is None:
            yield value
        else:
            for child in children:
                yield from nested_values(child)

    sys.modules["megatron.core.dist_checkpointing.strategies.torch"].TorchDistSaveShardedStrategy = TorchDistSaveShardedStrategy
    sys.modules["megatron.core.dist_checkpointing.dict_utils"].nested_values = nested_values

    class ShardedObject:
        def __init__(self, key: str, data: object, replica_id: object = 0) -> None:
            self.unique_key, self.data, self.replica_id = key, data, replica_id

    calls = SimpleNamespace(events=[], snapshots=[], blobs=[], created=[], durable=[])

    class NativeRoot:
        def register_state_dict(self, value: dict) -> None:
            calls.events.append("register")
            calls.blobs.append({key: getattr(obj, "_nvrx_serialized_blob", None) for key, obj in value.items()})

        def wait_for_async_completion(self) -> None:
            calls.events.append("local_fence")

        def save_async(self, sequence: int, path: Path) -> None:
            calls.events.append("snapshot")
            calls.snapshots.append((sequence, path))

    root = NativeRoot()

    def create_root(*args: object, **kwargs: object) -> NativeRoot:
        calls.created.append((args, kwargs))
        return root

    native.checkpoint = SimpleNamespace(
        model_state=SimpleNamespace(create_root=create_root),
        wait_for_ckpt_finalize=lambda session, sequence: calls.durable.append((session, sequence)),
    )
    serialization = load_module(monkeypatch, "nvrx_checkpointing.megatron.serialization", ROOT / "megatron/serialization.py")
    loading = load_module(monkeypatch, "nvrx_checkpointing.megatron.loading", ROOT / "megatron/loading.py")
    runtime = load_module(monkeypatch, "nvrx_checkpointing.backend", ROOT / "backend.py")
    implementation = load_module(monkeypatch, "nvrx_checkpointing.megatron.bridge", ROOT / "megatron/bridge.py")
    facade = load_module(monkeypatch, "nvrx_checkpointing.bridge", ROOT / "bridge.py")
    monkeypatch.setattr(runtime.secrets, "randbelow", lambda _limit: 40)
    return SimpleNamespace(
        module=implementation,
        runtime=runtime,
        facade=facade,
        loading=loading,
        serialization=serialization,
        calls=calls,
        root=root,
        torch=torch,
        ShardedObject=ShardedObject,
    )


def test_facade_import_is_lazy_without_megatron_or_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    package(monkeypatch, "nvrx_checkpointing", ROOT)
    for name in ("torch", "megatron", "nemo_rl", "nvrx_checkpointing.megatron.bridge"):
        monkeypatch.setitem(sys.modules, name, None)
    facade = load_module(monkeypatch, "nvrx_checkpointing.bridge", ROOT / "bridge.py")
    assert callable(facade.create_save_strategy)
    assert callable(facade.create_async_calls_queue)
    assert callable(facade.load_context)
    assert sys.modules["torch"] is None
    assert sys.modules["megatron"] is None


def test_queue_factory_works_without_megatron(monkeypatch: pytest.MonkeyPatch) -> None:
    package(monkeypatch, "nvrx_checkpointing", ROOT)
    torch = package(monkeypatch, "torch")
    torch.distributed = SimpleNamespace(is_initialized=lambda: False)
    for name in ("megatron", "megatron.bridge", "nemo_rl"):
        monkeypatch.setitem(sys.modules, name, None)
    runtime = load_module(monkeypatch, "nvrx_checkpointing.backend", ROOT / "backend.py")
    facade = load_module(monkeypatch, "nvrx_checkpointing.bridge", ROOT / "bridge.py")

    queue = facade.create_async_calls_queue()

    assert type(queue) is runtime.NativeAsyncCallsQueue
    queue.close()
    assert sys.modules["megatron"] is None


def test_explicit_snapshot_ids_and_local_completion_share_the_backend(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_collective(*_args: object) -> None:
        pytest.fail("Explicit local completion invokes no training-rank collective")

    monkeypatch.setattr(backend.runtime, "_gather", unexpected_collective)
    session = backend.runtime.NativeCheckpointSession(
        session_id=0,
        peers=["localhost"],
        ranks_per_peer=1,
        collective_checks=False,
    )
    queue = backend.runtime.NativeAsyncCallsQueue(collective_checks=False)
    published = []
    for iteration in (12, 24):
        session.register_state_dict({})
        request = session.submit(Path(f"iter_{iteration:07d}"), snapshot_id=iteration)
        request.add_finalize_fn(lambda iteration=iteration: published.append(iteration))
        queue.schedule_async_request(request)
        queue.maybe_finalize_async_calls(blocking=True)
    queue.close()

    assert backend.calls.created == [((0, 0, 1), {"peers": ["localhost"], "ranks_per_peer": 1})]
    assert backend.calls.snapshots == [(12, Path("iter_0000012")), (24, Path("iter_0000024"))]
    assert backend.calls.durable == [(0, 12), (0, 24)]
    assert published == [12, 24]


def test_factories_keep_the_training_process_group(backend: Any) -> None:
    group = object()
    strategy = backend.facade.create_save_strategy(process_group=group)
    queue = backend.facade.create_async_calls_queue(process_group=group)
    assert strategy._session.process_group is group
    assert queue._process_group is group
    assert (strategy.backend, strategy.version) == ("torch_dist", 1)
    queue.close()


def test_pending_durability_remains_visible_without_publication(backend: Any) -> None:
    ready = threading.Event()
    published = []
    queue = backend.facade.create_async_calls_queue()
    request = backend.runtime.NativeSaveRequest(lambda: ready.wait(2), [lambda: published.append(True)])
    queue.schedule_async_request(request)
    try:
        assert queue.maybe_finalize_async_calls() == []
        assert queue.get_num_unfinalized_calls() == 1
        assert published == []
    finally:
        ready.set()
        queue.close()
    assert published == [True]


def test_global_readiness_precedes_callbacks(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    group, published = object(), []
    queue = backend.facade.create_async_calls_queue(process_group=group)
    queue.schedule_async_request(backend.runtime.NativeSaveRequest(lambda: None, [lambda: published.append(True)]))
    queue._requests[0].future.result(timeout=2)
    groups = []

    def gather(value: object, process_group: object) -> list:
        groups.append(process_group)
        return [(0, True, None), (0, False, None)]

    monkeypatch.setattr(backend.runtime, "_gather", gather)
    assert queue.maybe_finalize_async_calls() == []
    assert queue.get_num_unfinalized_calls() == 1
    assert published == []
    monkeypatch.setattr(backend.runtime, "_gather", lambda _value, process_group: [(0, True, None)] * 2)
    assert queue.maybe_finalize_async_calls() == [0]
    assert groups == [group]
    assert published == [True]
    queue.close()


@pytest.mark.parametrize("remote_error", [False, True])
def test_durability_errors_keep_work_pending(backend: Any, monkeypatch: pytest.MonkeyPatch, remote_error: bool) -> None:
    published = []

    def wait() -> None:
        if not remote_error:
            raise OSError("disk unavailable")

    queue = backend.facade.create_async_calls_queue()
    queue.schedule_async_request(backend.runtime.NativeSaveRequest(wait, [lambda: published.append(True)]))
    pending = queue._requests[0]
    if remote_error:
        pending.future.result(timeout=2)
        monkeypatch.setattr(backend.runtime, "_gather", lambda *_args: [(0, True, None), (0, True, "disk unavailable")])
    try:
        with pytest.raises(RuntimeError, match="disk unavailable"):
            queue.maybe_finalize_async_calls(blocking=True)
        assert queue.get_num_unfinalized_calls() == 1
        assert published == []
    finally:
        queue._executor.shutdown(wait=True)


def test_queue_order_disagreement_prevents_publication(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    published = []
    queue = backend.facade.create_async_calls_queue()
    queue.schedule_async_request(backend.runtime.NativeSaveRequest(lambda: None, [lambda: published.append(True)]))
    queue._requests[0].future.result(timeout=2)
    monkeypatch.setattr(backend.runtime, "_gather", lambda *_args: [(0, True, None), (1, True, None)])
    try:
        with pytest.raises(RuntimeError, match="pending save order"):
            queue.maybe_finalize_async_calls()
        assert published == []
        assert queue.get_num_unfinalized_calls() == 1
    finally:
        queue._executor.shutdown(wait=True)


def test_queue_close_drains_and_rejects_new_work(backend: Any) -> None:
    queue = backend.facade.create_async_calls_queue()
    with pytest.raises(TypeError, match="non-native"):
        queue.schedule_async_request(SimpleNamespace())
    assert queue._executor is None
    published = []
    request = backend.runtime.NativeSaveRequest(lambda: None)
    request.add_finalize_fn(lambda: published.append(True))
    assert queue.schedule_async_request(request) == 0
    queue.close()
    assert queue.closed
    assert queue.get_num_unfinalized_calls() == 0
    assert published == [True]
    with pytest.raises(RuntimeError, match="closed"):
        queue.schedule_async_request(request)


def test_remote_failure_escapes_close_before_local_durability(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    local_done, close_returned = threading.Event(), threading.Event()
    errors = []
    queue = backend.facade.create_async_calls_queue()
    queue.schedule_async_request(backend.runtime.NativeSaveRequest(lambda: local_done.wait(5)))
    monkeypatch.setattr(backend.runtime, "_gather", lambda *_args: [(0, False, None), (0, True, "remote failure")])

    def close() -> None:
        try:
            queue.close()
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            close_returned.set()

    caller = threading.Thread(target=close)
    caller.start()
    try:
        assert close_returned.wait(2)
        assert errors
        assert "remote failure" in errors[0]
        assert not queue.closed
    finally:
        local_done.set()
        caller.join(timeout=2)
        queue._executor.shutdown(wait=True)
    assert not caller.is_alive()


def test_sync_request_checks_global_durability(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    group, published = object(), []
    request = backend.runtime.NativeSaveRequest(lambda: None, [lambda: published.append(True)], process_group=group)
    groups = []

    def gather(value: object, process_group: object) -> list:
        groups.append(process_group)
        return [None, "remote failure"]

    monkeypatch.setattr(backend.runtime, "_gather", gather)
    with pytest.raises(RuntimeError, match="remote failure"):
        request.execute_sync()
    assert groups == [group]
    assert published == []


def test_sequences_and_serialized_objects_refresh_across_saves(backend: Any) -> None:
    strategy = backend.facade.create_save_strategy()
    obj = backend.ShardedObject("common_state/shard_0_1", {"outer_step": 2})
    first = strategy.async_save({"state": obj}, Path("step_2/iter_0000000"))
    obj.data = {"outer_step": 3}
    second = strategy.async_save({"state": obj}, Path("step_3/iter_0000000"))
    first.execute_sync()
    second.execute_sync()
    assert [sequence for sequence, _path in backend.calls.snapshots] == [1, 2]
    assert backend.calls.durable == [(41, 1), (41, 2)]
    first_blob, second_blob = (entry["state"] for entry in backend.calls.blobs)
    assert len(first_blob) == len(second_blob)
    assert len(first_blob) % 1024 == 0
    assert pickle.loads(first_blob) == [{"outer_step": 2}]  # noqa: S301
    assert pickle.loads(second_blob) == [{"outer_step": 3}]  # noqa: S301
    assert backend.calls.events == ["register", "local_fence", "snapshot", "local_fence"] * 2


def test_object_overflow_precedes_submission_and_preserves_sequence(backend: Any) -> None:
    strategy = backend.facade.create_save_strategy()
    obj = backend.ShardedObject("common_state/shard_0_1", "small")
    strategy.async_save({"state": obj}, Path("first"))
    obj.data = "x" * 10000
    with pytest.raises(RuntimeError, match="exceeding its registered"):
        strategy.async_save({"state": obj}, Path("second"))
    assert len(backend.calls.snapshots) == 1
    assert strategy._session._sequence == 1


def test_object_capacities_are_per_strategy_and_nonmain_replicas_are_skipped(backend: Any) -> None:
    first, second = backend.facade.create_save_strategy(), backend.facade.create_save_strategy()
    first._prepare_objects({"objects": [backend.ShardedObject("key", "small")]})
    second._prepare_objects((backend.ShardedObject("key", "x" * 5000),))
    assert first._object_capacities["key"] < second._object_capacities["key"]
    replica = backend.ShardedObject("replica", "data", (0, 1))
    first._prepare_objects(replica)
    assert "replica" not in first._object_capacities
    assert not hasattr(replica, "_nvrx_serialized_blob")


def test_flattened_tensor_is_rejected_before_registration(backend: Any) -> None:
    strategy = backend.facade.create_save_strategy()
    shard = SimpleNamespace(key="flattened", flattened_range=slice(0, 10))
    with pytest.raises(RuntimeError, match="flattened tensor"):
        strategy.async_save({"weight": shard}, Path("checkpoint"))
    assert backend.calls.events == []
    assert strategy._session._sequence == 0


def test_registration_fence_failure_precedes_snapshot(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = backend.facade.create_save_strategy()

    def fail() -> None:
        raise OSError("registration unavailable")

    monkeypatch.setattr(backend.root, "wait_for_async_completion", fail)
    with pytest.raises(RuntimeError, match="state registration failed"):
        strategy.async_save({}, Path("checkpoint"))
    assert backend.calls.events == ["register"]
    assert backend.calls.snapshots == []
    assert strategy._session._sequence == 0


def test_submission_fence_failure_is_shared_before_request_return(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = backend.facade.create_save_strategy()
    strategy._session._root = backend.root
    fences = []

    def fence() -> None:
        fences.append(True)
        if len(fences) == 2:  # noqa: PLR2004
            raise OSError("snapshot unavailable")

    monkeypatch.setattr(backend.root, "wait_for_async_completion", fence)
    monkeypatch.setattr(backend.runtime, "_gather", lambda value, _group: [value, None])
    with pytest.raises(RuntimeError, match="snapshot submission failed"):
        strategy.async_save({}, Path("checkpoint"))
    assert strategy._session._sequence == 1
    assert backend.calls.durable == []


@pytest.mark.parametrize("phase", ["preparation", "registration"])
def test_remote_state_failure_blocks_snapshot(backend: Any, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    strategy = backend.facade.create_save_strategy()
    strategy._session._root = backend.root
    results = iter([[None, "remote failure"]] if phase == "preparation" else [[None, None], [None, "remote failure"]])
    monkeypatch.setattr(backend.runtime, "_gather", lambda *_args: next(results))
    with pytest.raises(RuntimeError, match=f"state {phase} failed"):
        strategy.async_save({}, Path("checkpoint"))
    assert backend.calls.snapshots == []
    assert strategy._session._sequence == 0


def test_explicit_group_controls_training_topology_and_consensus(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    group, seen = object(), []
    dist = backend.torch.distributed
    dist.is_initialized = lambda: True
    dist.get_rank = lambda *, group: 1
    dist.get_world_size = lambda *, group: 4

    def gather(output: list, value: object, *, group: object) -> None:
        seen.append(group)
        if isinstance(value, str):
            output[:] = ["train-a", "train-a", "train-b", "train-b"]
        elif value is None and not backend.calls.created:
            output[:] = [41, None, None, None]
        else:
            output[:] = [value] * 4

    dist.all_gather_object = gather
    strategy = backend.facade.create_save_strategy(process_group=group)
    request = strategy.async_save({}, Path("checkpoint"))
    request.execute_sync()
    assert backend.calls.created == [((41, 1, 4), {"peers": ["train-a", "train-b"], "ranks_per_peer": 2})]
    assert seen
    assert all(value is group for value in seen)


@pytest.mark.parametrize(
    ("hosts", "message"),
    [([], "empty"), ([""], "empty"), (["a", "a", "b"], "equal"), (["a", "b", "a", "b"], "contiguous")],
)
def test_invalid_training_topology_is_rejected(backend: Any, hosts: list, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        backend.runtime._trainer_peers(hosts)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("cached", [True, None])
def test_load_context_restores_nested_replica_ids_and_disables_cache(backend: Any, fail: bool, cached: bool | None) -> None:
    tensor, obj = SimpleNamespace(replica_id=(0, 0)), backend.ShardedObject("rng", {}, 0)
    base = SimpleNamespace(cache_metadata=cached)
    wrapper = SimpleNamespace(cache_metadata=cached, base_strategy=base)

    def load() -> None:
        with backend.facade.load_context({"nested": [tensor, (obj,)]}, wrapper):
            assert not wrapper.cache_metadata
            assert not base.cache_metadata
            tensor.replica_id, obj.replica_id = (1, 1), 2
            if fail:
                raise OSError("bad checkpoint")

    if fail:
        with pytest.raises(OSError, match="bad checkpoint"):
            load()
    else:
        load()
    assert tensor.replica_id == (0, 0)
    assert obj.replica_id == 0
    assert not wrapper.cache_metadata
    assert not base.cache_metadata


def test_ordinary_provider_delegates_object_slots_without_resetting_capacity(backend: Any) -> None:
    tree = ast.parse((ROOT / "megatron/save.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_snapshot_blob_capacity", "_prepare_rng_sharded_object_blobs")
    ]
    capacities = {}
    namespace = {
        "snapshot_blob_capacity": backend.serialization.snapshot_blob_capacity,
        "prepare_sharded_object_blobs": backend.serialization.prepare_sharded_object_blobs,
        "_sharded_object_capacities": capacities,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "save.py", "exec"), namespace)
    obj = backend.ShardedObject("rng_state/shard_0", {"step": 2})
    namespace["_prepare_rng_sharded_object_blobs"]([obj])
    capacity = capacities[obj.unique_key]
    obj.data = {"step": 3}
    namespace["_prepare_rng_sharded_object_blobs"]([obj])
    assert capacities[obj.unique_key] == capacity
    assert pickle.loads(obj._nvrx_serialized_blob) == [{"step": 3}]  # noqa: S301
    assert namespace["_snapshot_blob_capacity"](capacity) > capacity
