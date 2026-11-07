# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, PLR2004, S101
"""Registration of torch DTensor leaves."""

from __future__ import annotations

import io
import os
import random
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from nvrx_checkpointing import checkpoint
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

_REQUIRES_CHECKPOINT_SERVER = pytest.mark.skipif(
    os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
    reason="requires checkpoint-server IPC support",
)
_WORLD_SIZE = 2
_FSDP_ROWS = (5, 2)


def _expected_tensors() -> dict[str, torch.Tensor]:
    def values(*shape: int) -> torch.Tensor:
        return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)

    return {
        "even": values(8, 4),
        "uneven": values(7, 3) + 100,
        "replicated": values(5) + 200,
        "hsdp": values(6, 2) + 300,
        "fsdp_uneven": values(sum(_FSDP_ROWS), 2) + 400,
    }


def _uneven_chunk(full: torch.Tensor, mesh: DeviceMesh, offset: int, rows: int) -> DTensor:
    """Holds rows [offset, offset + rows) with the chunk override Megatron-FSDP applies to uneven shards."""
    local = full[offset : offset + rows].clone()
    dtensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False, shape=full.shape, stride=full.stride())
    chunk = ChunkStorageMetadata(offsets=torch.Size([offset, 0]), sizes=local.shape)
    dtensor._local_tensor.__create_chunk_list__ = lambda: [chunk]  # noqa: SLF001
    return dtensor


@pytest.fixture
def single_rank_mesh(tmp_path: Path) -> Iterator[DeviceMesh]:
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _create_root() -> object:
    return checkpoint.model_state.create_root(random.randrange(1, 1 << 31), 0, 1, ["localhost"], 1)  # noqa: S311


def test_registers_the_local_chunk_of_main_replicas(single_rank_mesh: DeviceMesh) -> None:
    full = _expected_tensors()["fsdp_uneven"]
    replica = distribute_tensor(torch.ones(5), init_device_mesh("cpu", (1,)), [Replicate()])
    replica.device_mesh.get_coordinate = lambda: [1]
    state = {
        "even": distribute_tensor(_expected_tensors()["even"], single_rank_mesh, [Shard(0)]),
        "fsdp_uneven": _uneven_chunk(full, single_rank_mesh, offset=_FSDP_ROWS[0], rows=_FSDP_ROWS[1]),
        "empty": _uneven_chunk(full, single_rank_mesh, offset=full.shape[0], rows=0),
        "replica": replica,
    }

    root = _create_root()
    root.register_state_dict(state)
    assert root.snapshot_size == (8 * 4 + _FSDP_ROWS[1] * 2) * 4


def test_registers_empty_tensors_as_objects(single_rank_mesh: DeviceMesh) -> None:
    del single_rank_mesh
    root = _create_root()
    empty = torch.empty(0, dtype=torch.uint8)
    root.register_state_dict({"full": torch.ones(4), "empty": empty})
    root.wait_for_async_completion()
    serialized = io.BytesIO()
    torch.save(empty, serialized)
    assert root.snapshot_size == 4 * 4 + len(serialized.getvalue())


def test_rejects_partial_placements(single_rank_mesh: DeviceMesh) -> None:
    root = _create_root()
    root.register_state_dict({"partial": DTensor.from_local(torch.ones(4), single_rank_mesh, [Partial()])})
    with pytest.raises(RuntimeError, match="partial placement"):
        root.wait_for_async_completion()


def _save_and_load(rank: int, store: str, session_id: int, checkpoint_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=_WORLD_SIZE)
    try:
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,))
        replicate_mesh = init_device_mesh("cpu", (_WORLD_SIZE, 1))
        expected = _expected_tensors()
        offset = sum(_FSDP_ROWS[:rank])
        state = {
            "even": distribute_tensor(expected["even"], mesh, [Shard(0)]),
            "uneven": distribute_tensor(expected["uneven"], mesh, [Shard(0)]),
            "replicated": distribute_tensor(expected["replicated"], mesh, [Replicate()]),
            "hsdp": distribute_tensor(expected["hsdp"], replicate_mesh, [Replicate(), Shard(0)]),
            "fsdp_uneven": _uneven_chunk(expected["fsdp_uneven"], mesh, offset=offset, rows=_FSDP_ROWS[rank]),
            "empty": torch.empty(0, dtype=torch.uint8),
        }
        root = checkpoint.model_state.create_root(session_id, rank, _WORLD_SIZE, ["localhost"], _WORLD_SIZE)
        root.register_state_dict({"model": state})
        root.save_async(1, checkpoint_dir)
        root.wait_for_async_completion()
        checkpoint.wait_for_ckpt_finalize(session_id, 1)
        dist.barrier()

        loaded = {"model": {name: distribute_tensor(torch.zeros_like(full), mesh, [Shard(0)]) for name, full in expected.items()}}
        loaded["model"]["empty"] = torch.empty(0, dtype=torch.uint8)
        dcp.load(loaded, checkpoint_id=checkpoint_dir)
        for name, full in expected.items():
            torch.testing.assert_close(loaded["model"][name].full_tensor(), full)
        torch.testing.assert_close(loaded["model"]["empty"], torch.empty(0, dtype=torch.uint8))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@_REQUIRES_CHECKPOINT_SERVER
def test_saved_chunks_load_through_dcp(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "iter_0000001"
    session_id = random.randrange(1, 1 << 31)  # noqa: S311
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=_save_and_load, args=(rank, str(tmp_path / "store"), session_id, str(checkpoint_dir)))
        for rank in range(_WORLD_SIZE)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=300)
        assert [worker.exitcode for worker in workers] == [0] * _WORLD_SIZE
        assert not (checkpoint_dir / "metadata.json").exists()

        expected = _expected_tensors()
        loaded = {"model": {name: torch.zeros_like(full) for name, full in expected.items()}}
        loaded["model"]["empty"] = torch.empty(0, dtype=torch.uint8)
        dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
        for name, full in expected.items():
            torch.testing.assert_close(loaded["model"][name], full)
        torch.testing.assert_close(loaded["model"]["empty"], torch.empty(0, dtype=torch.uint8))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        checkpoint.svc_spindown()
