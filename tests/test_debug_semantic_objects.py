# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101
"""Semantic comparison coverage for DCP BytesStorageMetadata values."""

from __future__ import annotations

import hashlib
import io
import pickle
from typing import TYPE_CHECKING

import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import debug
from nvrx_checkpointing.debug import compare_reference_metadata, distcp_trace
from torch.distributed.checkpoint import BytesStorageMetadata

if TYPE_CHECKING:
    from pathlib import Path

_COMMON_KEY = "common_state/shard_0_1"
_RNG_KEY = "rng_state/shard_0_1"
_OPTIMIZER_KEY = "optimizer/shard_0_1"
_RERUN_KEY = "rerun_state_machine_state/shard_0_1"


def _archive(value: object, *, legacy: bool = False) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.save([value], buffer, _use_new_zipfile_serialization=not legacy)
    buffer.seek(0)
    return buffer


def _write_checkpoint(
    path: Path,
    *,
    step: int,
    legacy_archives: bool,
    rng_seed: int = 3,
    rerun_iteration: int = 5,
) -> None:
    dcp.save(
        {
            _COMMON_KEY: _archive({"iteration": 1}, legacy=legacy_archives),
            _RNG_KEY: _archive(
                {
                    "random_rng_state": (rng_seed, (1, 2, 3), None),
                    "torch_rng_state": torch.arange(8, dtype=torch.uint8),
                },
                legacy=legacy_archives,
            ),
            _RERUN_KEY: _archive(
                {"current_iteration": rerun_iteration, "mode": "enabled"},
                legacy=legacy_archives,
            ),
            _OPTIMIZER_KEY: _archive(
                {"state": {0: {"step": step, "exp_avg": torch.arange(4, dtype=torch.float32)}}},
                legacy=legacy_archives,
            ),
        },
        checkpoint_id=str(path),
    )
    distcp_trace(path, extract_tensor_data=False, verify_native_server=False)


def test_semantic_object_compare_ignores_archive_encoding(tmp_path: Path) -> None:
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    _write_checkpoint(current, step=7, legacy_archives=False)
    _write_checkpoint(legacy, step=7, legacy_archives=True)

    assert compare_reference_metadata(current, legacy) == 0


def test_semantic_object_compare_detects_dynamic_state_changes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    changed = tmp_path / "changed"
    _write_checkpoint(reference, step=7, legacy_archives=False)
    _write_checkpoint(
        changed,
        step=8,
        rng_seed=4,
        rerun_iteration=6,
        legacy_archives=True,
    )

    reference_entries = (reference / "reference.metadata.info").read_text()
    changed_entries = (changed / "reference.metadata.info").read_text()
    for semantic_key in (
        f"{_OPTIMIZER_KEY}.object.state.0.step",
        f"{_RNG_KEY}.object.random_rng_state.0",
        f"{_RERUN_KEY}.object.current_iteration",
    ):
        reference_line = next(line for line in reference_entries.splitlines() if line.startswith(f"{semantic_key}="))
        changed_line = next(line for line in changed_entries.splitlines() if line.startswith(f"{semantic_key}="))
        assert reference_line != changed_line

    assert compare_reference_metadata(changed, reference) > 0


def test_native_server_blob_claim_uses_exact_archive_bytes(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, step=7, legacy_archives=False)

    with (checkpoint / ".metadata").open("rb") as fd:
        metadata = pickle.load(fd)  # noqa: S301

    claims = []
    for index, storage in metadata.storage_data.items():
        if index.fqn == _COMMON_KEY or not isinstance(metadata.state_dict_metadata[index.fqn], BytesStorageMetadata):
            continue
        with (checkpoint / storage.relative_path).open("rb") as fd:
            fd.seek(storage.offset)
            raw = fd.read(storage.length)
        claims.append(f"{index.fqn}.bytes.md5sum={hashlib.md5(raw).hexdigest()}")  # noqa: S324
    (checkpoint / "0.metadata.info").write_text("\n".join(sorted(claims)))

    monkeypatch.setattr(debug, "try_ping", lambda: None)
    debug.distcp_trace(checkpoint, extract_tensor_data=False)
