# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import io
import os
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.debug import distcp_trace
from nvrx_checkpointing.megatron.load import decode_common_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1", reason="requires checkpoint-server IPC support"),
]
_KEY = "common_state/shard_0_1"
_WEIGHTS = ("z.weight", "a.weight")


def _archive(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _save(rank: int, world_size: int, session: int, directory: str) -> None:
    path = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{path / 'store'}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        root = checkpoint.model_state.create_root(session, rank, world_size, ["localhost"], world_size)
        for iteration, length in enumerate((1, 8192), start=1):
            state = {
                key: DTensor.from_local(torch.full((4,), float(rank * 100 + iteration + index)), mesh, [Replicate()], run_check=False)
                for index, key in enumerate(_WEIGHTS)
            }
            state.update(
                {
                    "iteration": iteration + rank,
                    "objects.args": Namespace(rank=rank, message="x" * length),
                    "enabled": rank % 2 == 0,
                    "empty": torch.empty(0),
                    f"peer.{rank}": rank + iteration * 10,
                },
            )
            planner = dcp.DefaultSavePlanner()
            planner.set_up_planner(state, is_coordinator=rank == 0)
            descriptions = [
                (item.index.fqn, list(item.index.offset) if item.index.offset is not None else None, item.tensor_storage_size() or 1)
                for item in planner.create_local_plan().items
            ]
            captured = {
                key: value if isinstance(value, torch.Tensor) or type(value) in (int, float) else _archive(value)
                for key, value in state.items()
            }
            root.register_state_dict(captured, write_items=descriptions, common_state_key=_KEY)
            if iteration == 1:
                root.finalize_registry()
            root.save_async(iteration, path / f"native-{iteration}")
            root.wait_for_async_completion()
            checkpoint.wait_for_ckpt_finalize(session, iteration)
            dist.barrier()
            dcp.save(state, checkpoint_id=path / f"reference-{iteration}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_daemon_source_selection_matches_dcp_for_differing_replicas(tmp_path: Path, world_size: int) -> None:
    session = random.randrange(1, 1 << 31)  # noqa: S311
    context = mp.get_context("spawn")
    workers = [context.Process(target=_save, args=(rank, world_size, session, str(tmp_path))) for rank in range(world_size)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        assert [worker.exitcode for worker in workers] == [0] * world_size

        bundle_keys = [_KEY, *(f"{_KEY}.rank_{rank}" for rank in range(1, world_size))]
        for iteration in (1, 2):
            directory = tmp_path / f"native-{iteration}"
            reference = {key: torch.zeros(4) for key in _WEIGHTS}
            reference.update({"iteration": None, "objects.args": None, "enabled": None, "empty": torch.empty(0)})
            reference.update({f"peer.{rank}": None for rank in range(world_size)})
            dcp.load(reference, checkpoint_id=tmp_path / f"reference-{iteration}")
            target = {key: torch.zeros(4) for key in _WEIGHTS}
            target.update(dict.fromkeys(bundle_keys))
            dcp.load(target, checkpoint_id=directory)

            objects = {}
            for key in bundle_keys:
                restored = decode_common_state(target[key][0])
                assert not objects.keys() & restored.keys()
                objects.update(restored)
            assert set(objects) == set(reference) - set(_WEIGHTS)
            for key, expected in reference.items():
                if key in _WEIGHTS:
                    torch.testing.assert_close(target[key], expected, rtol=0, atol=0)
                elif isinstance(expected, torch.Tensor):
                    torch.testing.assert_close(objects[key], expected, rtol=0, atol=0)
                else:
                    assert objects[key] == expected
            assert not (directory / "metadata.json").exists()
            distcp_trace(directory, extract_tensor_data=False, verify_native_server=False)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
        checkpoint.svc_spindown()
