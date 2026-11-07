# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401, INP001, PLR2004, S101

"""Daemon-owned plans select capture data before and across native snapshots."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron.mapping import ShardedObject, ShardedTensor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
        reason="requires checkpoint-server IPC support",
    ),
]

_SESSION_IDS = count(90000)
_COMMON_STATE_KEY = "common_state/shard_0_1"
_SCALAR, _DYNAMIC_BLOB, _DTENSOR, _FIXED_BLOB = 2, 3, 5, 6
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "registry_tree.py"
_SPEC = importlib.util.spec_from_file_location("snapshot_registry_tree", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
registry_tree = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = registry_tree
_SPEC.loader.exec_module(registry_tree)


def _roots(session_id: int) -> list[checkpoint.model_state]:
    return [checkpoint.model_state.create_root(session_id, rank, 2, ["localhost"], 2) for rank in range(2)]


def _read_plan(session_id: int, rank: int, directory: Path) -> Any:
    url = f"http://localhost:4100/api/snapshot-plan/{session_id}/{rank}"
    with urllib.request.urlopen(url, timeout=20) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "application/octet-stream"
        data = response.read()
    plan = registry_tree.read_registry(directory / f"plan-{rank}.bin", data)
    assert plan.session_id == session_id
    assert plan.rank == rank
    assert plan.trailing == 0
    return plan


def _entries(plan: Any) -> dict[str, tuple[int, Any]]:
    def name(handle: int) -> str:
        parts = []
        while handle != 0:
            parts.append(plan.names[handle])
            handle = plan.parents[handle]
        return ".".join(reversed(parts))

    return {name(handle): (kind, value) for handle, kind, value in plan.entries}


def _serialize(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save([value], buffer)
    return buffer.getvalue()


def _state(rank: int, iteration: int, length: int) -> dict[str, object]:
    return {
        "weight": ShardedTensor.from_rank_offsets(
            "weight",
            torch.arange(8, dtype=torch.float32) + iteration * 10,
            replica_id=(rank + 1, 7),
        ),
        "rng_state": ShardedObject("rng_state", {"iteration": iteration}, (1,), (0,), replica_id=rank + 1),
        "step": iteration,
        "path": b"run/" + b"x" * length,
        _COMMON_STATE_KEY: _serialize({"iteration": iteration, "message": "x" * length}),
    }


def _load_object(directory: Path, key: str) -> object:
    state = {key: io.BytesIO()}
    dcp.load(state, checkpoint_id=str(directory))
    value = state[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False)
    return value[0]


def _assert_file_coverage(directory: Path, metadata: Any) -> None:
    covered = Counter()
    for entry in metadata.storage_data.values():
        covered[entry.relative_path] += entry.length
    files = {path.name: path.stat().st_size for path in directory.glob("__*_0.distcp")}
    assert files == dict(covered)


def _assert_plan_ownership(plans: list[Any]) -> None:
    entries = [_entries(plan) for plan in plans]
    counts = Counter(key for rank_entries in entries for key in rank_entries)
    assert counts == dict.fromkeys(("weight", "rng_state", "step", "path", _COMMON_STATE_KEY), 1)
    for rank, rank_entries in enumerate(entries):
        for kind, value in rank_entries.values():
            if kind in {_DTENSOR, _FIXED_BLOB}:
                assert {shard.rank for shard in value.shards} == {rank}
            else:
                assert kind in {_SCALAR, _DYNAMIC_BLOB}
                assert value.replicas == (rank,)
                if kind == _SCALAR:
                    assert value.value[1] == 0
                else:
                    assert value.storage_size == 0
    fixed_bytes = sum(
        shard.storage_size
        for rank_entries in entries
        for kind, value in rank_entries.values()
        if kind in {_DTENSOR, _FIXED_BLOB}
        for shard in value.shards
    )
    assert fixed_bytes == 8 * 4 + len(_serialize({"iteration": 0}))


def test_plan_waits_for_registration_and_selects_each_value_once(tmp_path: Path) -> None:
    checkpoint.svc_spindown()
    session_id = next(_SESSION_IDS)
    roots = _roots(session_id)
    checkpoint.ping()
    roots[0].register_state_dict(_state(0, 0, 1))
    roots[0].wait_for_async_completion()
    roots[0].finalize_registry()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(_read_plan, session_id, 0, tmp_path)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.2)
            with urllib.request.urlopen("http://localhost:4100/api/ping", timeout=5) as response:
                assert response.status == 200

            roots[1].register_state_dict(_state(1, 0, 1))
            roots[1].finalize_registry()
            for root in roots:
                root.wait_for_async_completion()
            plans = [pending.result(timeout=20), _read_plan(session_id, 1, tmp_path)]

        _assert_plan_ownership(plans)
        assert not list(tmp_path.glob("iter_*"))

        for iteration, length in enumerate((10, 4096), 1):
            directory = tmp_path / f"iter_{iteration:07d}"
            for rank, root in enumerate(roots):
                root.register_state_dict(_state(rank, iteration, length))
                root.save_async(iteration, directory)
            for root in roots:
                root.wait_for_async_completion()
            checkpoint.wait_for_ckpt_finalize(session_id, iteration)

            loaded = {"weight": torch.zeros(8, dtype=torch.float32)}
            dcp.load(loaded, checkpoint_id=str(directory))
            torch.testing.assert_close(loaded["weight"], torch.arange(8, dtype=torch.float32) + iteration * 10)
            assert _load_object(directory, "rng_state/shard_0_1") == {"iteration": iteration}
            assert _load_object(directory, _COMMON_STATE_KEY) == {"iteration": iteration, "message": "x" * length}
            metadata = dcp.FileSystemReader(directory).read_metadata()
            path_entries = [entry for index, entry in metadata.storage_data.items() if index.fqn == "path"]
            assert len(path_entries) == 1
            entry = path_entries[0]
            data = (directory / entry.relative_path).read_bytes()
            assert data[entry.offset : entry.offset + entry.length] == b"run/" + b"x" * length
            _assert_file_coverage(directory, metadata)
    finally:
        checkpoint.svc_spindown()


def test_rank_with_empty_capture_plan_still_finalizes(tmp_path: Path) -> None:
    checkpoint.svc_spindown()
    session_id = next(_SESSION_IDS)
    roots = _roots(session_id)
    source = torch.arange(4, dtype=torch.float32)
    try:
        for root in roots:
            root.register_state_dict({"weight": source})
            root.finalize_registry()
        for root in roots:
            root.wait_for_async_completion()
        plans = [_entries(_read_plan(session_id, rank, tmp_path)) for rank in range(2)]
        assert sorted(map(len, plans)) == [0, 1]

        for root in roots:
            root.register_state_dict({"weight": source + 20})
            root.save_async(1, tmp_path)
        for root in roots:
            root.wait_for_async_completion()
        checkpoint.wait_for_ckpt_finalize(session_id, 1)

        owner = next(rank for rank, entries in enumerate(plans) if entries)
        assert {path.name for path in tmp_path.glob("__*_0.distcp")} == {f"__{owner}_0.distcp"}
        loaded = {"weight": torch.zeros_like(source)}
        dcp.load(loaded, checkpoint_id=str(tmp_path))
        torch.testing.assert_close(loaded["weight"], source + 20)
    finally:
        checkpoint.svc_spindown()
