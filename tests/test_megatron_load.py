# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101, SLF001

from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.api import CheckpointException


@pytest.fixture
def bundle_load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("native_bundle_load_test", Path(__file__).parents[1] / "megatron/load.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load", "load_state_dict", "FileSystemReader"):
        monkeypatch.setattr(dcp, name, getattr(dcp, name))
    monkeypatch.setattr(dcp, module._INSTALL_MARKER, False, raising=False)
    module.install(ModuleType("megatron.training.checkpointing"))
    return module


def _archive(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _save_bundle(path: Path, objects: dict[str, object], tensors: dict[str, object] | None = None) -> None:
    state, _ = flatten_state_dict(tensors or {})
    state["common_state/shard_0_1"] = [objects]
    dcp.save(state, checkpoint_id=path, planner=DefaultSavePlanner(flatten_state_dict=False))


def test_bootstrap_restores_arguments_and_scalars(bundle_load: ModuleType, tmp_path: Path) -> None:
    arguments = argparse.Namespace(num_layers=12, consumed_train_samples=384)
    _save_bundle(
        tmp_path,
        {"args": _archive(arguments), "iteration": 50, "checkpoint_version": 3.0},
    )
    target = {"args": None, "iteration": None, "checkpoint_version": None}

    dcp.load(target, checkpoint_id=tmp_path)

    assert vars(target["args"]) == vars(arguments)
    assert target["iteration"] == 50
    assert target["checkpoint_version"] == 3.0
    assert "iteration" in bundle_load.BundleFileSystemReader(tmp_path).read_metadata().state_dict_metadata


def test_full_load_preserves_tensor_and_optimizer_aliases(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_average = torch.ones(2, 3)
    _save_bundle(
        tmp_path,
        {
            "iteration": 7,
            "optimizer.param_to_group_meta.weight.lr": 0.002,
            "optimizer.param_to_group_meta.weight.step": 7,
            "optimizer.param_to_group_meta.weight.betas": _archive((0.8, 0.95)),
            "rng_state.(0, 0).0.random_rng_state": _archive((3, (1, 2, 3), None)),
        },
        {"model": {"weight": expected_weight}, "optimizer": {"state": {"weight": {"exp_avg": expected_average}}}},
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


def test_missing_bundled_object_is_rejected(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    _save_bundle(tmp_path, {"iteration": 7})

    with pytest.raises(CheckpointException, match=r"Missing bundled checkpoint object: optimizer\.step"):
        dcp.load_state_dict({"optimizer": {"step": 0}}, storage_reader=dcp.FileSystemReader(tmp_path), no_dist=True)


def test_bundled_empty_tensor_preserves_destination_identity(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    empty = torch.empty(0, 3)
    _save_bundle(tmp_path, {"model.empty": _archive(empty)}, {"model": {"weight": torch.ones(2)}})
    destination = torch.empty_like(empty, requires_grad=True)
    target = {"model": {"empty": destination, "weight": torch.zeros(2)}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["model"]["empty"] is destination
    torch.testing.assert_close(destination, empty)
    torch.testing.assert_close(target["model"]["weight"], torch.ones(2))


def test_existing_dcp_checkpoint_uses_default_decoding(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    dcp.save({"iteration": 7, "model": {"weight": torch.ones(2)}}, checkpoint_id=tmp_path)
    target = {"iteration": 0, "model": {"weight": torch.zeros(2)}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["iteration"] == 7
    torch.testing.assert_close(target["model"]["weight"], torch.ones(2))


def test_standard_checkpoint_retains_custom_planner_callbacks(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    dcp.save({"iteration": 7}, checkpoint_id=tmp_path)
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


def test_bundle_decode_failure_uses_dcp_error_handling(bundle_load: ModuleType, tmp_path: Path) -> None:
    _save_bundle(tmp_path, {"args": b"invalid object archive"})
    reader = bundle_load.BundleFileSystemReader(tmp_path)
    metadata = reader.read_metadata()

    assert "args" in metadata.state_dict_metadata
    assert reader.objects == {"args": b"invalid object archive"}
    with pytest.raises(CheckpointException, match="UnpicklingError"):
        dcp.load({"args": None}, storage_reader=reader)


def test_peer_bundle_restores_rank_specific_objects(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    dcp.save(
        {
            "common_state/shard_0_1": [{"iteration": 7}],
            "common_state/shard_0_1.rank_1": io.BytesIO(_archive([{"optimizer.expert4.lr": 0.002}])),
        },
        checkpoint_id=tmp_path,
        planner=DefaultSavePlanner(flatten_state_dict=False),
    )
    target = {"iteration": 0, "optimizer": {"expert4": {"lr": 1.0}}}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target == {"iteration": 7, "optimizer": {"expert4": {"lr": 0.002}}}


def test_duplicate_peer_bundle_objects_are_rejected(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    dcp.save(
        {
            "common_state/shard_0_1": [{"iteration": 7}],
            "common_state/shard_0_1.rank_1": [{"iteration": 8}],
        },
        checkpoint_id=tmp_path,
        planner=DefaultSavePlanner(flatten_state_dict=False),
    )

    with pytest.raises(CheckpointException, match="Duplicate bundled checkpoint objects"):
        dcp.load({"iteration": 0}, checkpoint_id=tmp_path)


def test_torch_dist_common_state_remains_opaque(bundle_load: ModuleType, tmp_path: Path) -> None:
    del bundle_load
    common = {"iteration": 7, "args": argparse.Namespace(num_layers=12)}
    _save_bundle(tmp_path, common)
    (tmp_path / "metadata.json").write_text('{"sharded_backend": "torch_dist", "sharded_backend_version": 1}')
    target = {"common_state/shard_0_1": io.BytesIO()}

    dcp.load(target, checkpoint_id=tmp_path)

    loaded = target["common_state/shard_0_1"]
    assert loaded[0]["iteration"] == 7
    assert vars(loaded[0]["args"]) == {"num_layers": 12}
