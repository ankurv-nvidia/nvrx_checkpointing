# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import io
import math

import _distcp_shim
import pytest
import torch


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
