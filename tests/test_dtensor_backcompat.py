# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from nvrx_checkpointing import checkpoint
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

_FC1 = "module.decoder.layers.0.mlp.experts.linear_fc1.weight0_w"
_FC2 = "module.decoder.layers.0.mlp.experts.linear_fc2.weight0"
_PROJECTION = "module.decoder.layers.0.self_attention.linear_proj.weight"
_OPTIMIZER_FQN = f"module.module.{_FC1}"


def _uneven_destination(mesh: DeviceMesh, rank: int) -> DTensor:
    rows = (5, 2)[rank]
    local = torch.full((3, rows), -1.0).t()
    tensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False, shape=torch.Size((7, 3)), stride=(3, 1))
    chunk = ChunkStorageMetadata(offsets=torch.Size((0 if rank == 0 else 5, 0)), sizes=local.shape)
    tensor._local_tensor.__create_chunk_list__ = lambda: [chunk]  # noqa: SLF001
    return tensor


def _load_dtensors(rank: int, store: str, directory: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        fc1 = torch.arange(35, dtype=torch.bfloat16).reshape(7, 5)
        fc2 = torch.arange(4, dtype=torch.float32).reshape(1, 4) + 100
        projection = torch.arange(4, dtype=torch.float32).reshape(1, 4) + 200
        moment = torch.arange(21, dtype=torch.float32).reshape(7, 3) + 300
        dcp.save(
            {
                "model": {
                    _FC1: distribute_tensor(fc1, mesh, [Shard(0)]),
                    _FC2: distribute_tensor(fc2, mesh, [Shard(0)]),
                    _PROJECTION: distribute_tensor(projection, mesh, [Replicate()]),
                },
                "optimizer": {"state": {_OPTIMIZER_FQN: {"exp_avg": distribute_tensor(moment, mesh, [Shard(0)])}}},
                "iteration": 50,
            },
            checkpoint_id=directory,
        )

        def destination() -> dict:
            return {
                "model": {
                    _FC1: distribute_tensor(torch.zeros_like(fc1, dtype=torch.float32), mesh, [Shard(1)]),
                    _FC2: distribute_tensor(torch.zeros_like(fc2), mesh, [Replicate()]),
                    _PROJECTION: distribute_tensor(torch.zeros_like(projection), mesh, [Shard(0)]),
                },
                "optimizer": {"state": {_OPTIMIZER_FQN: {"exp_avg": _uneven_destination(mesh, rank)}}},
                "iteration": 0,
            }

        native = destination()
        stock = destination()
        leaves = [*list(native["model"].values()), native["optimizer"]["state"][_OPTIMIZER_FQN]["exp_avg"]]
        local_storage = [tensor._local_tensor for tensor in leaves]  # noqa: SLF001

        dcp.load(stock, checkpoint_id=directory)
        checkpoint.load_checkpoint(directory, native, rank=rank, world_size=2)

        for key in native["model"]:
            torch.testing.assert_close(native["model"][key].to_local(), stock["model"][key].to_local())
        torch.testing.assert_close(native["model"][_FC1].to_local(), fc1[:, rank * 3 : (rank + 1) * 3].float())
        torch.testing.assert_close(native["model"][_FC2].to_local(), fc2)
        torch.testing.assert_close(native["model"][_PROJECTION].to_local(), projection if rank == 0 else projection[:0])
        native_moment = native["optimizer"]["state"][_OPTIMIZER_FQN]["exp_avg"]
        stock_moment = stock["optimizer"]["state"][_OPTIMIZER_FQN]["exp_avg"]
        torch.testing.assert_close(native_moment.to_local(), stock_moment.to_local())
        offset, rows = (0, 5) if rank == 0 else (5, 2)
        torch.testing.assert_close(native_moment.to_local(), moment[offset : offset + rows])
        assert native["iteration"] == stock["iteration"] == 50
        restored = [*list(native["model"].values()), native_moment]
        assert all(value is original for value, original in zip(restored, leaves, strict=True))
        assert all(value._local_tensor is original for value, original in zip(restored, local_storage, strict=True))  # noqa: SLF001
        if rank == 1:
            assert native["model"][_PROJECTION].to_local().numel() == 0
    finally:
        dist.destroy_process_group()


def test_stock_dtensors_load_into_different_local_shards(tmp_path: Path) -> None:
    mp.spawn(_load_dtensors, args=(str(tmp_path / "store"), str(tmp_path / "checkpoint")), nprocs=2, join=True)
