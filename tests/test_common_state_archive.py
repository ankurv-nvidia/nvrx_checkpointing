# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import _distcp_shim
import pytest
import torch
import torch.distributed.checkpoint as dcp

if TYPE_CHECKING:
    from pathlib import Path


def test_common_archive_preserves_supported_values() -> None:
    values = {
        "iteration": 123456,
        "minimum": -(1 << 63),
        "maximum": (1 << 63) - 1,
        "negative": -1,
        "lr": 0.00015,
        "negative_zero": -0.0,
        "positive_infinity": math.inf,
        "negative_infinity": -math.inf,
        "empty": b"",
        "binary": bytes(range(256)) * 1024,
        "nested.λ": b"\x00\xff",
    }

    archive = _distcp_shim.common_state_archive(values)
    restored = torch.load(io.BytesIO(archive), weights_only=False)

    assert restored == [values]
    assert math.copysign(1, restored[0]["negative_zero"]) == -1
    assert isinstance(restored[0]["iteration"], int)
    assert isinstance(restored[0]["lr"], float)
    assert isinstance(restored[0]["binary"], bytes)


def test_common_archive_preserves_nan() -> None:
    archive = _distcp_shim.common_state_archive({"value": math.nan})

    assert math.isnan(torch.load(io.BytesIO(archive), weights_only=False)[0]["value"])


def test_common_archive_supports_empty_state() -> None:
    archive = _distcp_shim.common_state_archive({})

    assert torch.load(io.BytesIO(archive), weights_only=False) == [{}]


def test_common_archive_rejects_unsupported_values() -> None:
    with pytest.raises(RuntimeError, match="supports integers, floats, and bytes"):
        _distcp_shim.common_state_archive({"value": object()})


def test_bundled_checkpoint_tracks_current_lengths_and_format(tmp_path: Path) -> None:
    key = "checkpoint.objects"
    marker = tmp_path / "metadata.json"
    marker.write_text('{"sharded_backend": "torch_dist"}')
    lengths = []
    for iteration, payload in ((7, b"short"), (123456, bytes(range(256)) * 1024)):
        values = {"iteration": iteration, "nested.payload": payload}
        _distcp_shim.write_common_state(str(tmp_path), key, values)

        assert not marker.exists()
        metadata = dcp.FileSystemReader(tmp_path).read_metadata()
        assert set(metadata.state_dict_metadata) == {key}
        storage = next(iter(metadata.storage_data.values()))
        assert storage.offset == 0
        assert storage.length == (tmp_path / storage.relative_path).stat().st_size
        lengths.append(storage.length)
        restored = {key: None}
        dcp.load(restored, checkpoint_id=tmp_path)
        assert restored[key] == [values]
    assert lengths[1] > lengths[0]


def test_empty_bundle_is_a_loadable_rank_zero_checkpoint(tmp_path: Path) -> None:
    key = "checkpoint.objects"
    _distcp_shim.write_common_state(str(tmp_path), key, {})

    restored = {key: None}
    dcp.load(restored, checkpoint_id=tmp_path)

    assert restored[key] == [{}]
    assert (tmp_path / "__0_0.distcp").is_file()
