# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101

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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1", reason="requires checkpoint-server IPC support"),
]
_KEY = "common_state/shard_0_1"


def _archive(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue().ljust(4096, b"\0")


def _save(rank: int, session: int, store: str, directory: str, planned: bool) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        root = checkpoint.model_state.create_root(session, rank, 2, ["localhost"], 2)
        weight = torch.arange(8.0)
        for iteration, value in enumerate((7, 123456), start=1):
            state = {"weight": weight, "iteration": value + rank, "lr": 0.125, "args": _archive(Namespace(rank=rank))}
            if rank == 1:
                state["peer_only"] = 9
            options = {}
            if planned:
                options["write_items"] = [("weight", [0], weight.numel() * weight.element_size())]
                options["write_items"].extend((key, None, 1) for key in state if key != "weight")
            root.register_state_dict(state, **options)
            root.save_async(iteration, Path(directory) / str(iteration), common_state_key=_KEY)
            root.wait_for_async_completion()
            checkpoint.wait_for_ckpt_finalize(session, iteration)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("planned", [False, True])
def test_daemon_bundles_selected_local_state_for_each_snapshot(tmp_path: Path, planned: bool) -> None:
    session = random.randrange(1, 1 << 31)  # noqa: S311
    context = mp.get_context("spawn")
    workers = [context.Process(target=_save, args=(rank, session, str(tmp_path / "store"), str(tmp_path), planned)) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        assert [worker.exitcode for worker in workers] == [0, 0]

        for iteration, value in enumerate((7, 123456), start=1):
            directory = tmp_path / str(iteration)
            metadata = dcp.FileSystemReader(directory).read_metadata()
            peer_key = f"{_KEY}.rank_1"
            assert set(metadata.state_dict_metadata) == {"weight", _KEY, peer_key}
            storage = next(info for index, info in metadata.storage_data.items() if index.fqn == _KEY)
            assert storage.relative_path == "__0_0.distcp"
            peer_storage = next(info for index, info in metadata.storage_data.items() if index.fqn == peer_key)
            assert peer_storage.relative_path == "__1_0.distcp"
            target = {"weight": torch.zeros(8), _KEY: None, peer_key: None}
            dcp.load(target, checkpoint_id=directory)
            common = decode_common_state(target[_KEY][0])
            peer = decode_common_state(target[peer_key][0])
            assert not common.keys() & peer.keys()
            objects = {**common, **peer}
            owner = int(planned)
            assert objects["iteration"] == value + owner
            assert objects["lr"] == 0.125
            assert objects["args"].rank == owner
            assert "peer_only" not in common
            assert peer["peer_only"] == 9
            torch.testing.assert_close(target["weight"], torch.arange(8.0))
            distcp_trace(directory, extract_tensor_data=False, verify_native_server=False)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
        checkpoint.svc_spindown()
