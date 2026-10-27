# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101
"""Daemon spin-down and relaunch within one client process."""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron.mapping import ShardedTensor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
        reason="requires checkpoint-server IPC support",
    ),
]

_SERVICE_PORT = 4100


def _instance_lock_is_free() -> bool:
    lock_path = Path(tempfile.gettempdir()) / f"nvrx_ckpt_server.{_SERVICE_PORT}.lock"
    with lock_path.open("a") as lock_file:
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.lockf(lock_file, fcntl.LOCK_UN)
    return True


def test_saves_reach_a_relaunched_daemon(tmp_path: Path) -> None:
    for cycle in range(3):
        session_id = 3000 + cycle
        expected = torch.full((8,), float(cycle))
        root = checkpoint.model_state.create_root(session_id, 0, 1, ["localhost"], 1)
        weight = ShardedTensor(
            key="weight",
            data=expected.clone(),
            dtype=torch.float32,
            local_shape=(8,),
            global_shape=(8,),
            global_offset=(0,),
            axis_fragmentations=None,
        )
        root.register_state_dict({"weight": weight})
        checkpoint_dir = tmp_path / f"cycle_{cycle}"
        try:
            root.save_async(1, checkpoint_dir)
            root.wait_for_async_completion()
            checkpoint.wait_for_ckpt_finalize(session_id, 1)
        finally:
            checkpoint.svc_spindown()
        assert _instance_lock_is_free()

        loaded = {"weight": torch.empty(8)}
        dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
        assert torch.equal(loaded["weight"], expected)
