# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101
"""Daemon-level coverage for embedded native common state."""

from __future__ import annotations

import io
import json
import os
import time
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron.mapping import ShardedObject, ShardedTensor

if TYPE_CHECKING:
    from pathlib import Path

_DCP_COMMON_STATE_KEY = "common_state/shard_0_1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
        reason="requires checkpoint-server IPC support",
    ),
]


def _serialize_common_state(iteration: int) -> bytes:
    buffer = io.BytesIO()
    torch.save([{"iteration": iteration, "scheduler": {"num_steps": iteration * 10}}], buffer)
    return buffer.getvalue()


def _load_common_state(checkpoint_dir: Path) -> dict[str, object]:
    key = "common_state/shard_0_1"
    loaded = {key: io.BytesIO()}
    dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
    value = loaded[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False)
    return value[0]


def _load_sharded_object(checkpoint_dir: Path) -> dict[str, object]:
    key = "rng_state/shard_0_1"
    loaded = {key: io.BytesIO()}
    dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
    value = loaded[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False)
    return value[0]


def _assert_common_state_layout(checkpoint_dir: Path, expected_capacity: int) -> None:
    metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
    assert set(metadata.state_dict_metadata) == {"weight", "rng_state/shard_0_1", _DCP_COMMON_STATE_KEY}
    assert type(metadata.state_dict_metadata[_DCP_COMMON_STATE_KEY]).__name__ == "BytesStorageMetadata"

    storage_entries = [(index, info) for index, info in metadata.storage_data.items() if index.fqn == _DCP_COMMON_STATE_KEY]
    assert len(storage_entries) == 1
    index, info = storage_entries[0]
    assert index.offset is None
    assert index.index is None
    assert info.relative_path == "__0_0.distcp"
    assert info.length == expected_capacity


def test_daemon_embeds_fresh_common_state(tmp_path: Path) -> None:
    initial_common = _serialize_common_state(0)
    common_capacity = ((len(initial_common) + 512 + 1023) // 1024) * 1024
    root = checkpoint.model_state.create_root(2000, 0, 1, ["localhost"], 1)
    state_dict = {
        "weight": ShardedTensor(
            key="weight",
            data=torch.arange(8, dtype=torch.float32),
            dtype=torch.float32,
            local_shape=(8,),
            global_shape=(8,),
            global_offset=(0,),
            axis_fragmentations=None,
        ),
        "rng_state": ShardedObject("rng_state", {"iteration": 0}, (1,), (0,)),
        _DCP_COMMON_STATE_KEY: initial_common.ljust(common_capacity, b"\0"),
    }
    root.register_state_dict(state_dict)
    root.finalize_registry()

    try:
        for iteration in (1, 2):
            checkpoint_dir = tmp_path / f"iter_{iteration:07d}"
            assert not checkpoint_dir.exists()
            common_bytes = _serialize_common_state(iteration).ljust(common_capacity, b"\0")
            root.register_state_dict(
                {
                    "weight": state_dict["weight"],
                    _DCP_COMMON_STATE_KEY: common_bytes,
                    "rng_state": ShardedObject("rng_state", {"iteration": iteration}, (1,), (0,)),
                },
            )
            root.save_async(iteration, checkpoint_dir)
            root.wait_for_async_completion()

            deadline = time.monotonic() + 30
            while not (checkpoint_dir / ".metadata").is_file() or not (checkpoint_dir / "metadata.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"checkpoint daemon did not publish {checkpoint_dir}")
                time.sleep(0.05)

            assert _load_common_state(checkpoint_dir) == {
                "iteration": iteration,
                "scheduler": {"num_steps": iteration * 10},
            }
            assert _load_sharded_object(checkpoint_dir) == {"iteration": iteration}
            _assert_common_state_layout(checkpoint_dir, common_capacity)
            assert not (checkpoint_dir / "common.pt").exists()
            assert json.loads((checkpoint_dir / "metadata.json").read_text()) == {
                "sharded_backend": "torch_dist",
                "sharded_backend_version": 1,
            }
    finally:
        checkpoint.svc_spindown()


def test_daemon_embeds_changing_common_state_lengths(tmp_path: Path) -> None:
    session_id = 2001
    root = checkpoint.model_state.create_root(session_id, 0, 1, ["localhost"], 1)
    root.register_state_dict({_DCP_COMMON_STATE_KEY: b""})
    root.finalize_registry()

    try:
        for iteration, length in enumerate((10, 4096, 0), 1):
            value = {"payload": "x" * length, "iteration": iteration}
            buffer = io.BytesIO()
            torch.save([value], buffer)
            serialized = buffer.getvalue()
            checkpoint_dir = tmp_path / f"iter_{iteration:07d}"

            root.register_state_dict({_DCP_COMMON_STATE_KEY: serialized})
            root.save(iteration, checkpoint_dir)
            checkpoint.wait_for_ckpt_finalize(session_id, iteration)

            assert _load_common_state(checkpoint_dir) == value
            metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
            entry = next(info for index, info in metadata.storage_data.items() if index.fqn == _DCP_COMMON_STATE_KEY)
            assert entry.length == len(serialized)
    finally:
        checkpoint.svc_spindown()


def test_daemon_cold_save_preserves_mixed_state(tmp_path: Path) -> None:
    session_id = 2002
    root = checkpoint.model_state.create_root(session_id, 0, 1, ["localhost"], 1)

    def state(iteration: int) -> dict[str, object]:
        return {
            "weight": torch.arange(8, dtype=torch.float32) + iteration,
            "step": iteration,
            "rng_state": ShardedObject("rng_state", {"iteration": iteration}, (1,), (0,)),
            "empty_blob": b"",
            _DCP_COMMON_STATE_KEY: _serialize_common_state(iteration),
        }

    try:
        for iteration in (1, 2):
            directory = tmp_path / f"iter_{iteration:07d}"
            root.register_state_dict(state(iteration))
            root.save(iteration, directory)
            checkpoint.wait_for_ckpt_finalize(session_id, iteration)

            loaded = {"weight": torch.zeros(8, dtype=torch.float32)}
            dcp.load(loaded, checkpoint_id=str(directory))
            torch.testing.assert_close(loaded["weight"], torch.arange(8, dtype=torch.float32) + iteration)
            assert _load_common_state(directory)["iteration"] == iteration
            assert _load_sharded_object(directory) == {"iteration": iteration}
            metadata = dcp.FileSystemReader(directory).read_metadata()
            entries = [info for index, info in metadata.storage_data.items() if index.fqn == "empty_blob"]
            assert len(entries) == 1
            assert entries[0].length == 0

        incomplete = state(3)
        del incomplete[_DCP_COMMON_STATE_KEY]
        root.register_state_dict(incomplete)
        root.save_async(3, tmp_path / "iter_0000003")
        with pytest.raises(RuntimeError, match=r"registered blob 'common_state/shard_0_1' has no snapshot data"):
            root.wait_for_async_completion()
        assert not (tmp_path / "iter_0000003").exists()
    finally:
        checkpoint.svc_spindown()
