# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101
"""Registration of torch DTensor leaves."""

from __future__ import annotations

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


def test_registers_every_participating_local_replica(single_rank_mesh: DeviceMesh) -> None:
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
    assert root.snapshot_size == (8 * 4 + _FSDP_ROWS[1] * 2 + 5) * 4


def test_ignores_ranks_outside_the_tensor_mesh(single_rank_mesh: DeviceMesh) -> None:
    outside = distribute_tensor(torch.ones(5), single_rank_mesh, [Replicate()])
    outside.device_mesh.get_coordinate = lambda: None
    root = _create_root()

    root.register_state_dict({"outside": outside})

    assert root.snapshot_size == 0


def test_registers_empty_tensors_as_native_values(single_rank_mesh: DeviceMesh, monkeypatch: pytest.MonkeyPatch) -> None:
    del single_rank_mesh
    root = _create_root()
    empty = torch.empty(0, dtype=torch.uint8)

    def reject_archive(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("native registration called torch.save")

    monkeypatch.setattr(torch, "save", reject_archive)
    root.register_state_dict({"full": torch.ones(4), "empty": empty}, native=True)
    root.wait_for_async_completion()

    assert root.snapshot_size > 4 * 4


def test_rejects_partial_placements(single_rank_mesh: DeviceMesh) -> None:
    root = _create_root()
    root.register_state_dict({"partial": DTensor.from_local(torch.ones(4), single_rank_mesh, [Partial()])})
    with pytest.raises(RuntimeError, match="partial placement"):
        root.wait_for_async_completion()


@pytest.mark.integration
@_REQUIRES_CHECKPOINT_SERVER
def test_scalar_dtensor_preserves_its_zero_dimensional_shape(tmp_path: Path, single_rank_mesh: DeviceMesh) -> None:
    session_id = random.randrange(1, 1 << 31)  # noqa: S311
    root = checkpoint.model_state.create_root(session_id, 0, 1, ["localhost"], 1)
    source = distribute_tensor(torch.tensor(7.0), single_rank_mesh, [Replicate()])
    destination = distribute_tensor(torch.tensor(0.0), single_rank_mesh, [Replicate()])
    plain_source = torch.tensor(11, dtype=torch.int64)
    plain_destination = torch.tensor(0, dtype=torch.int64)
    try:
        root.register_state_dict({"model": {"scalar": source, "plain_scalar": plain_source}}, native=True)
        root.finalize_registry()
        root.save(1, tmp_path)
        checkpoint.wait_for_ckpt_finalize(session_id, 1)
        metadata = dcp.FileSystemReader(tmp_path).read_metadata()
        assert metadata.state_dict_metadata["model.scalar"].size == torch.Size([])
        assert metadata.state_dict_metadata["model.plain_scalar"].size == torch.Size([])
        target = {"model.scalar": destination, "model.plain_scalar": plain_destination}
        dcp.load(target, checkpoint_id=tmp_path, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
        assert destination.shape == torch.Size([])
        assert plain_destination.shape == torch.Size([])
        torch.testing.assert_close(destination.to_local(), source.to_local())
        torch.testing.assert_close(plain_destination, plain_source)
    finally:
        checkpoint.svc_spindown()


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
        }
        root = checkpoint.model_state.create_root(session_id, rank, _WORLD_SIZE, ["localhost"], _WORLD_SIZE)
        root.register_state_dict({"model": state})
        root.finalize_registry()
        root.save_async(1, checkpoint_dir)
        root.wait_for_async_completion()
        checkpoint.wait_for_ckpt_finalize(session_id, 1)
        dist.barrier()

        loaded = {"model": {name: distribute_tensor(torch.zeros_like(full), mesh, [Shard(0)]) for name, full in expected.items()}}
        dcp.load(loaded, checkpoint_id=checkpoint_dir)
        for name, full in expected.items():
            torch.testing.assert_close(loaded["model"][name].full_tensor(), full)
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
        assert (checkpoint_dir / "metadata.json").exists()

        expected = _expected_tensors()
        loaded = {"model": {name: torch.zeros_like(full) for name, full in expected.items()}}
        dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
        for name, full in expected.items():
            torch.testing.assert_close(loaded["model"][name], full)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        checkpoint.svc_spindown()
