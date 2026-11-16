# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101, SLF001

from __future__ import annotations

import argparse
import enum
import importlib.util
import io
import math
import struct
from pathlib import Path
from types import ModuleType

import _distcp_shim
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner


@pytest.fixture
def dictionary_load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).parents[1] / "megatron" / "load.py"
    spec = importlib.util.spec_from_file_location("nvrx_checkpointing.megatron.dictionary_load_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load", "load_state_dict"):
        monkeypatch.setattr(dcp, name, getattr(dcp, name))
    monkeypatch.setattr(dcp, module._INSTALL_MARKER, False, raising=False)
    module.install(ModuleType("megatron.training.checkpointing"))
    return module


def _archive(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _save_independent_scalars(path: Path, scalars: dict, objects: tuple = ()) -> None:
    _distcp_shim.write_rank_shard(str(path), 0, [], list(objects), scalars=scalars)
    _distcp_shim.write_metadata(str(path), [], list(objects), scalars=scalars)


def _mark_torch_dist(path: Path) -> None:
    (path / "metadata.json").write_text('{"sharded_backend": "torch_dist", "sharded_backend_version": 1}')


def _save_optimizer_metadata(path: Path, groups: dict, *, native: bool) -> None:
    if not native:
        dcp.save({"optimizer": {"param_to_group_meta": groups}}, checkpoint_id=path)
        return

    def text(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<I", len(encoded)) + encoded

    def encode(value: object) -> bytes:
        if isinstance(value, dict):
            return b"\x09" + struct.pack("<I", len(value)) + b"".join(encode(key) + encode(child) for key, child in value.items())
        if isinstance(value, str):
            return b"\x05" + text(value)
        if isinstance(value, bool):
            return b"\x04" + bytes([value])
        if isinstance(value, float):
            return b"\x03" + struct.pack("<d", value)
        raise TypeError(type(value))

    key = "__nvrx_native_dict_rank_0__.optimizer.param_to_group_meta"
    objects = [
        (key, b"NVRXOBJ1" + encode(groups)),
        ("__nvrx_native_state_rank_0__", b"NVRXNST1" + struct.pack("<I", 1) + text(key) + b"\x14"),
    ]
    _distcp_shim.write_rank_shard(str(path), 0, [], objects)
    _distcp_shim.write_metadata(str(path), [], objects)


@pytest.mark.parametrize("saved", [None, False, True])
def test_missing_defaults_preserve_saved_values(dictionary_load: ModuleType, tmp_path: Path, saved: bool | None) -> None:
    source = {"step": 7}
    if saved is not None:
        source["optional"] = saved
    dcp.save(source, checkpoint_id=tmp_path)
    target = {"step": 0, "optional": True, "container": {}}

    dictionary_load.checkpoint.load_checkpoint(
        tmp_path,
        target,
        missing_defaults={"step": 99, "optional": False, "container": False, "unrequested": False},
    )

    assert target == {"step": 7, "optional": saved if saved is not None else False, "container": {}}


@pytest.mark.parametrize("allow_partial_load", [False, True])
def test_missing_defaults_preserve_tensor_requirements(dictionary_load: ModuleType, tmp_path: Path, allow_partial_load: bool) -> None:
    dcp.save({"iteration": 7}, checkpoint_id=tmp_path)
    tensor = torch.zeros(2)
    target = {"iteration": 0, "weight": tensor, "optional": None}
    defaults = {"weight": False, "optional": False}

    if allow_partial_load:
        dictionary_load.checkpoint.load_checkpoint(tmp_path, target, allow_partial_load=True, missing_defaults=defaults)
        assert target["weight"] is tensor
        torch.testing.assert_close(tensor, torch.zeros(2))
        assert target["optional"] is False
    else:
        with pytest.raises((RuntimeError, ValueError), match=r"missing checkpoint tensor 'weight'"):
            dictionary_load.checkpoint.load_checkpoint(tmp_path, target, missing_defaults=defaults)


@pytest.mark.parametrize("native", [False, True], ids=["stock-leaves", "native-parent"])
@pytest.mark.parametrize("saved", [None, False, True])
@pytest.mark.parametrize("runtime", [False, True])
def test_optimizer_indexer_metadata_uses_checkpoint_or_legacy_default(
    dictionary_load: ModuleType,
    tmp_path: Path,
    native: bool,
    saved: bool | None,
    runtime: bool,
) -> None:
    del dictionary_load
    group = {"lr": 0.002}
    if saved is not None:
        group["is_dsa_indexer"] = saved
    _save_optimizer_metadata(tmp_path, {"weight": group}, native=native)
    destination = {"lr": 1.0, "is_dsa_indexer": runtime}
    target = {"optimizer": {"param_to_group_meta": {"weight": destination}}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["optimizer"]["param_to_group_meta"]["weight"] is destination
    assert destination == {"lr": 0.002, "is_dsa_indexer": saved if saved is not None else runtime}


@pytest.mark.parametrize("native", [False, True], ids=["stock-leaves", "native-parent"])
def test_optimizer_indexer_default_preserves_other_required_metadata(dictionary_load: ModuleType, tmp_path: Path, native: bool) -> None:
    del dictionary_load
    _save_optimizer_metadata(tmp_path, {"weight": {"lr": 0.002}}, native=native)
    target = {"optimizer": {"param_to_group_meta": {"weight": {"lr": 1.0, "is_dsa_indexer": True, "required": 0}}}}

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)missing.*optimizer.param_to_group_meta.weight.required"):
        dcp.load(target, checkpoint_id=tmp_path)


@pytest.mark.parametrize("flatten", [True, False])
def test_independent_scalars_restore_without_object_archives(dictionary_load: ModuleType, tmp_path: Path, flatten: bool) -> None:
    del dictionary_load
    captured = {"step": -(1 << 63), "maximum": (1 << 64) - 1, "enabled": True, "negative_zero": -0.0}
    _save_independent_scalars(tmp_path, captured)
    target = {"step": 0, "maximum": 0, "enabled": False, "negative_zero": 1.0}

    dcp.load(target, checkpoint_id=tmp_path, planner=DefaultLoadPlanner(flatten_state_dict=flatten))

    assert target == captured
    assert type(target["enabled"]) is bool
    assert math.copysign(1.0, target["negative_zero"]) == -1.0


def test_independent_scalars_load_into_nested_destinations(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    _save_independent_scalars(tmp_path, {"optimizer.step": 17, "settings.enabled": True})
    optimizer = {"step": 0}
    target = {"optimizer": optimizer, "settings": {"enabled": False}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target == {"optimizer": {"step": 17}, "settings": {"enabled": True}}
    assert target["optimizer"] is optimizer


def test_independent_scalar_restores_destination_enum(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load

    class Mode(enum.IntEnum):
        DISABLED = 0
        ENABLED = 1

    _save_independent_scalars(tmp_path, {"mode": 1})
    target = {"mode": Mode.DISABLED}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["mode"] is Mode.ENABLED


def test_independent_scalars_coexist_with_untouched_common_archive(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    common = {"iteration": 17, "enabled": True, "payload": b"12345678"}
    key = "common_state/shard_0_1"
    _save_independent_scalars(tmp_path, {"iteration": 17, "enabled": True}, ((key, _archive([common])),))
    target = {key: io.BytesIO(), "iteration": 0, "enabled": False}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target[key] == [common]
    assert target["iteration"] == 17
    assert target["enabled"] is True
    assert (tmp_path / "metadata.json").is_file()


def test_unmarked_eight_byte_blob_is_not_read_as_a_scalar(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    _save_independent_scalars(tmp_path, {"iteration": 17}, (("opaque", b"12345678"),))
    target = {"iteration": 0}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target == {"iteration": 17}
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    assert not hasattr(metadata.state_dict_metadata["opaque"], "nvrx_scalar_type")


def test_bootstrap_restores_arguments_and_scalars(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    arguments = argparse.Namespace(num_layers=12, consumed_train_samples=384)
    dcp.save({"args": arguments, "iteration": 50, "checkpoint_version": 3.0}, checkpoint_id=tmp_path)
    target = {"args": None, "iteration": None, "checkpoint_version": None}

    dcp.load(target, checkpoint_id=tmp_path)

    assert vars(target["args"]) == vars(arguments)
    assert target["iteration"] == 50
    assert target["checkpoint_version"] == 3.0


def test_full_load_preserves_tensor_and_optimizer_aliases(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_average = torch.ones(2, 3)
    dcp.save(
        {
            "model": {"weight": expected_weight},
            "optimizer": {
                "state": {"weight": {"exp_avg": expected_average}},
                "param_to_group_meta": {"weight": {"lr": 0.002, "step": 7, "betas": (0.8, 0.95)}},
            },
            "iteration": 7,
            "rng_state": {"(0, 0)": [{"random_rng_state": (3, (1, 2, 3), None)}]},
        },
        checkpoint_id=tmp_path,
    )
    weight = torch.zeros_like(expected_weight)
    average = torch.zeros_like(expected_average)
    optimizer = {"state": {"weight": {"exp_avg": average}}, "param_to_group_meta": {"weight": {"lr": 1.0, "step": 0, "betas": (0.1, 0.2)}}}
    raw_optimizer = optimizer.copy()
    target = {
        "model": {"weight": weight},
        "optimizer": optimizer,
        "iteration": 0,
        "rng_state": {"(0, 0)": [{"random_rng_state": (0, (), None)}]},
    }

    dcp.load_state_dict(target, storage_reader=dcp.FileSystemReader(tmp_path), no_dist=True, planner=DefaultLoadPlanner())

    assert target["model"]["weight"] is weight
    assert target["optimizer"]["state"]["weight"]["exp_avg"] is average
    torch.testing.assert_close(weight, expected_weight)
    torch.testing.assert_close(average, expected_average)
    assert raw_optimizer["param_to_group_meta"]["weight"] == {"lr": 0.002, "step": 7, "betas": (0.8, 0.95)}
    assert target["rng_state"]["(0, 0)"][0]["random_rng_state"] == (3, (1, 2, 3), None)
    assert target["iteration"] == 7


def test_missing_checkpoint_value_is_rejected(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    _save_independent_scalars(tmp_path, {"iteration": 7})

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)missing.*optimizer.step"):
        dcp.load_state_dict({"optimizer": {"step": 0}}, storage_reader=dcp.FileSystemReader(tmp_path), no_dist=True)


def test_empty_tensor_preserves_destination_identity(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    empty = torch.empty(0, 3)
    dcp.save({"model": {"empty": empty, "weight": torch.ones(2)}}, checkpoint_id=tmp_path)
    destination = torch.empty_like(empty, requires_grad=True)
    target = {"model": {"empty": destination, "weight": torch.zeros(2)}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["model"]["empty"] is destination
    torch.testing.assert_close(destination, empty)
    torch.testing.assert_close(target["model"]["weight"], torch.ones(2))


def test_torch_dist_checkpoint_retains_custom_planner_callbacks(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    dcp.save({"iteration": 7}, checkpoint_id=tmp_path)
    _mark_torch_dist(tmp_path)
    calls = []

    class CustomPlanner(DefaultLoadPlanner):
        def load_bytes(self, read_item: object, value: io.BytesIO) -> None:
            calls.append("load_bytes")
            super().load_bytes(read_item, value)
            self.original_state_dict["iteration"] += 100

    target = {"iteration": 0}
    dcp.load(target, checkpoint_id=tmp_path, planner=CustomPlanner())

    assert calls == ["load_bytes"]
    assert target["iteration"] == 107


def test_torch_dist_common_state_remains_opaque(dictionary_load: ModuleType, tmp_path: Path) -> None:
    del dictionary_load
    common = {"iteration": 7, "args": argparse.Namespace(num_layers=12)}
    dcp.save(
        {"common_state/shard_0_1": [common]},
        checkpoint_id=tmp_path,
        planner=DefaultSavePlanner(flatten_state_dict=False),
    )
    _mark_torch_dist(tmp_path)
    target = {"common_state/shard_0_1": io.BytesIO()}

    dcp.load(target, checkpoint_id=tmp_path)

    loaded = target["common_state/shard_0_1"]
    assert loaded[0]["iteration"] == 7
    assert vars(loaded[0]["args"]) == {"num_layers": 12}


def test_hook_passes_rank_and_partial_load_to_native_binding(
    dictionary_load: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".metadata").write_bytes(b"native")
    target = {"iteration": 0}
    calls = []

    def native_load(path: Path, state: dict, *, rank: int, world_size: int, allow_partial_load: bool, **kwargs: object) -> None:
        assert kwargs == {"missing_defaults": {}}
        calls.append((path, state is target, rank, world_size, allow_partial_load))
        state["iteration"] = 7

    monkeypatch.setattr(dictionary_load.checkpoint, "load_checkpoint", native_load)
    monkeypatch.setattr(dictionary_load.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dictionary_load.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dictionary_load.dist, "get_world_size", lambda: 4)

    dcp.load(target, checkpoint_id=tmp_path, planner=DefaultLoadPlanner(allow_partial_load=True))

    assert calls == [(tmp_path, True, 2, 4, True)]
    assert target == {"iteration": 7}


@pytest.mark.parametrize("implicit_args", [False, True])
def test_bootstrap_forwards_runtime_template(
    dictionary_load: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    implicit_args: bool,
) -> None:
    class Mode(enum.Enum):
        SLOW = "slow"
        FAST = "fast"

    (tmp_path / ".metadata").write_bytes(b"native")
    checkpointing = ModuleType("native_checkpoint_bootstrap_test")
    runtime = argparse.Namespace(mode=Mode.SLOW, layers=4)

    def native_load(_path: Path, state: dict, **_kwargs: object) -> None:
        assert state["args"] is runtime
        state["args"] = argparse.Namespace(mode=Mode.FAST, layers=12)

    def load_base_checkpoint(_directory: Path, args: argparse.Namespace | None = None, rank0: bool = False) -> dict:
        assert args is (None if implicit_args else runtime)
        assert rank0
        target = {"args": None}
        dcp.load(target, checkpoint_id=tmp_path)
        return target

    monkeypatch.setattr(dictionary_load.checkpoint, "load_checkpoint", native_load)
    checkpointing.get_args = lambda: runtime
    checkpointing._load_base_checkpoint = load_base_checkpoint
    dictionary_load.install(checkpointing)

    target = (
        checkpointing._load_base_checkpoint(tmp_path, rank0=True)
        if implicit_args
        else checkpointing._load_base_checkpoint(tmp_path, runtime, rank0=True)
    )

    assert target["args"].mode is Mode.FAST
    assert target["args"].layers == 12
    assert runtime.mode is Mode.SLOW
    assert runtime.layers == 4
    assert dictionary_load._BOOTSTRAP_TEMPLATE.get() is None


@pytest.mark.parametrize("saved_flops", [None, 3_180_960_153_600])
def test_full_megatron_load_requests_optional_flops_counter(dictionary_load: ModuleType, tmp_path: Path, saved_flops: int | None) -> None:
    source = {"model": {"weight": torch.ones(2)}, "iteration": 50}
    if saved_flops is not None:
        source["num_floating_point_operations_so_far"] = saved_flops
    dcp.save(source, checkpoint_id=tmp_path)
    checkpointing = ModuleType("native_checkpoint_full_load_test")

    def load_base_checkpoint(_directory: Path, rank0: bool = False, sharded_state_dict: dict | None = None) -> dict:
        assert not rank0
        assert sharded_state_dict is not None
        state = sharded_state_dict.copy()
        state.pop("_model")
        dcp.load(state, checkpoint_id=tmp_path)
        return state

    checkpointing._load_base_checkpoint = load_base_checkpoint
    dictionary_load.install(checkpointing)
    weight = torch.zeros(2)
    template = {"model": {"weight": weight}, "iteration": 0, "_model": [object()]}

    restored = checkpointing._load_base_checkpoint(tmp_path, sharded_state_dict=template)

    assert restored["num_floating_point_operations_so_far"] == (saved_flops or 0)
    assert restored["iteration"] == 50
    assert restored["model"]["weight"] is weight
    torch.testing.assert_close(weight, torch.ones(2))
    assert dictionary_load._LOAD_DEFAULTS.get() is None
    if saved_flops is None:
        with pytest.raises((RuntimeError, ValueError), match=r"Missing checkpoint value: num_floating_point_operations_so_far"):
            dcp.load({"num_floating_point_operations_so_far": 0}, checkpoint_id=tmp_path)


def test_full_non_fsdp_load_preserves_requested_state(dictionary_load: ModuleType, tmp_path: Path) -> None:
    dcp.save({"iteration": 50, "num_floating_point_operations_so_far": 1000}, checkpoint_id=tmp_path)
    checkpointing = ModuleType("native_checkpoint_non_fsdp_load_test")

    def load_base_checkpoint(_directory: Path, rank0: bool = False, sharded_state_dict: dict | None = None) -> dict:
        assert not rank0
        assert sharded_state_dict is not None
        assert "num_floating_point_operations_so_far" not in sharded_state_dict
        assert dictionary_load._LOAD_DEFAULTS.get() == {}
        state = sharded_state_dict.copy()
        dcp.load(state, checkpoint_id=tmp_path)
        return state

    checkpointing._load_base_checkpoint = load_base_checkpoint
    dictionary_load.install(checkpointing)
    template = {"iteration": 0}

    restored = checkpointing._load_base_checkpoint(tmp_path, sharded_state_dict=template)

    assert restored == {"iteration": 50}
    assert template == {"iteration": 0}
    assert dictionary_load._LOAD_DEFAULTS.get() is None


def test_install_preserves_filesystem_reader_and_is_idempotent(dictionary_load: ModuleType) -> None:
    original_reader = dcp.FileSystemReader
    installed_load = dcp.load
    installed_load_state_dict = dcp.load_state_dict

    dictionary_load.install(ModuleType("megatron.training.checkpointing"))

    assert dcp.FileSystemReader is original_reader
    assert dcp.load is installed_load
    assert dcp.load_state_dict is installed_load_state_dict
