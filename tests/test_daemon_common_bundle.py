# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import os
import random
import struct
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.debug import distcp_trace
from nvrx_checkpointing.megatron import load as native_load
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1", reason="requires checkpoint-server IPC support"),
]
_WEIGHT = "decoder.layers.0.weight"
_TENSOR_SIZES = {_WEIGHT: 4, "decoder.layers.1.weight": 17, "decoder.final_layernorm.weight": 7}


def _save(rank: int, world_size: int, session: int, directory: str) -> None:
    path = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{path / 'store'}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        root = checkpoint.model_state.create_root(session, rank, world_size, ["localhost"], world_size)
        for iteration, length in enumerate((1, 8192), start=1):
            state = {
                "model": {
                    name: DTensor.from_local(torch.full((size,), float(100 * rank + iteration)), mesh, [Replicate()], run_check=False)
                    for name, size in _TENSOR_SIZES.items()
                },
                "iteration": iteration,
                "objects": Namespace(message="x" * length, values=[None, (), {}, [1, 2]], rng=np.random.RandomState(17).get_state()),
                "optimizer": {
                    "metadata": {f"expert_{rank}": {"lr": 0.001, "step": iteration}},
                    "shared": True,
                    "momentum": torch.full((11,), float(100 * rank + iteration)),
                },
                "enabled": True,
                "empty": torch.empty(0),
            }
            original_save = torch.save

            def forbidden_save(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("native state capture called torch.save")

            torch.save = forbidden_save
            try:
                root.register_state_dict(state, native=True)
                if iteration == 1:
                    root.finalize_registry()
                root.save_async(iteration, path / f"native-{iteration}")
                root.wait_for_async_completion()
                checkpoint.wait_for_ckpt_finalize(session, iteration)
            finally:
                torch.save = original_save
            dist.barrier()
            dcp.save(state, checkpoint_id=path / f"reference-{iteration}")
    finally:
        dist.destroy_process_group()


def _assert_same(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, Namespace):
        _assert_same(vars(actual), vars(expected))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_same(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("world_size", [2, 4])
def test_native_nested_state_matches_dcp_and_preserves_peer_entries(
    tmp_path: Path,
    world_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = random.randrange(1, 1 << 31)  # noqa: S311
    context = mp.get_context("spawn")
    workers = [context.Process(target=_save, args=(rank, world_size, session, str(tmp_path))) for rank in range(world_size)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        assert [worker.exitcode for worker in workers] == [0] * world_size

        for name in ("load", "load_state_dict", "FileSystemReader"):
            monkeypatch.setattr(dcp, name, getattr(dcp, name))
        monkeypatch.setattr(dcp, native_load._INSTALL_MARKER, False, raising=False)  # noqa: SLF001
        native_load.install(ModuleType("megatron.training.checkpointing"))
        for iteration in (1, 2):

            def destination() -> dict:
                return {
                    "model": {name: torch.zeros(size) for name, size in _TENSOR_SIZES.items()},
                    "iteration": 0,
                    "objects": Namespace(message="", values=[], rng=np.random.RandomState(2).get_state()),
                    "optimizer": {
                        "metadata": {f"expert_{rank}": {"lr": 1.0, "step": 0} for rank in range(world_size)},
                        "shared": False,
                        "momentum": torch.zeros(11),
                    },
                    "enabled": False,
                    "empty": torch.empty(0),
                }

            reference, target = destination(), destination()
            dcp.load(reference, checkpoint_id=tmp_path / f"reference-{iteration}")
            selected_ranks = {int(value[0].item()) // 100 for value in reference["model"].values()}
            assert len(selected_ranks) > 1
            optimizer = target["optimizer"]
            weight = target["model"][_WEIGHT]
            directory = tmp_path / f"native-{iteration}"
            dcp.load(target, checkpoint_id=directory)

            _assert_same(target, reference)
            assert target["optimizer"] is optimizer
            assert target["model"][_WEIGHT] is weight
            metadata = dcp.FileSystemReader(directory).read_metadata()
            assert f"model.{_WEIGHT}" in metadata.state_dict_metadata
            assert "common_state/shard_0_1" not in metadata.state_dict_metadata
            scalar = metadata.storage_data[MetadataIndex("iteration")]
            assert scalar.length == struct.calcsize("<Q")
            with (directory / scalar.relative_path).open("rb") as stream:
                stream.seek(scalar.offset)
                assert int.from_bytes(stream.read(8), "little") == iteration
            assert not (directory / "metadata.json").exists()
            distcp_trace(directory, extract_tensor_data=False, verify_native_server=False)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
        checkpoint.svc_spindown()
