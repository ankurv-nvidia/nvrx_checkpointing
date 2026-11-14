# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101, SLF001

from __future__ import annotations

import argparse
import enum
import importlib.util
import io
import math
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

    def native_load(path: Path, state: dict, *, rank: int, world_size: int, allow_partial_load: bool) -> None:
        calls.append((path, state is target, rank, world_size, allow_partial_load))
        state["iteration"] = 7

    monkeypatch.setattr(dictionary_load.checkpoint, "load_checkpoint", native_load)
    monkeypatch.setattr(dictionary_load.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dictionary_load.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dictionary_load.dist, "get_world_size", lambda: 4)

    dcp.load(target, checkpoint_id=tmp_path, planner=DefaultLoadPlanner(allow_partial_load=True))

    assert calls == [(tmp_path, True, 2, 4, True)]
    assert target == {"iteration": 7}


def test_bootstrap_supplies_runtime_template_without_mutating_it(
    dictionary_load: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Mode(enum.Enum):
        SLOW = "slow"
        FAST = "fast"

    (tmp_path / ".metadata").write_bytes(b"native")
    checkpointing = ModuleType("native_checkpoint_bootstrap_test")
    runtime = argparse.Namespace(mode=Mode.SLOW, layers=4)

    def native_load(_path: Path, state: dict, **_kwargs: object) -> None:
        assert state["args"] is not runtime
        assert state["args"].mode is Mode.SLOW
        state["args"].mode = Mode.FAST
        state["args"].layers = 12

    def load_base_checkpoint(_directory: Path, args: argparse.Namespace, rank0: bool = False) -> dict:
        assert args is runtime
        assert rank0
        target = {"args": None}
        dcp.load(target, checkpoint_id=tmp_path)
        return target

    monkeypatch.setattr(dictionary_load.checkpoint, "load_checkpoint", native_load)
    checkpointing._load_base_checkpoint = load_base_checkpoint
    dictionary_load.install(checkpointing)

    target = checkpointing._load_base_checkpoint(tmp_path, runtime, rank0=True)

    assert target["args"].mode is Mode.FAST
    assert target["args"].layers == 12
    assert runtime.mode is Mode.SLOW
    assert runtime.layers == 4
    assert dictionary_load._BOOTSTRAP_TEMPLATE.get() is None


def test_install_preserves_filesystem_reader_and_is_idempotent(dictionary_load: ModuleType) -> None:
    original_reader = dcp.FileSystemReader
    installed_load = dcp.load
    installed_load_state_dict = dcp.load_state_dict

    dictionary_load.install(ModuleType("megatron.training.checkpointing"))

    assert dcp.FileSystemReader is original_reader
    assert dcp.load is installed_load
    assert dcp.load_state_dict is installed_load_state_dict
