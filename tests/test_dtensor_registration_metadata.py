# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import random

import pytest
import torch
from nvrx_checkpointing import checkpoint

_COMMON_STATE_KEY = "common_state/shard_0_1"


def _create_root() -> object:
    return checkpoint.model_state.create_root(random.randrange(1, 1 << 31), 0, 1, ["localhost"], 1)  # noqa: S311


def test_registration_accepts_ordered_tensor_and_object_descriptors() -> None:
    root = _create_root()
    root.register_state_dict(
        {"model.weight": torch.ones(2), "iteration": 7, "optimizer.settings": b"objects"},
        write_items=(("model.weight", (0,), 8), ("iteration", None, 1), ("optimizer.settings", None, 1)),
        common_state_key=_COMMON_STATE_KEY,
    )

    assert root.snapshot_size == 8 + len(b"objects")


@pytest.mark.parametrize(
    ("later_items", "later_key"),
    [
        ((("weight", (0,), 16), ("iteration", None, 1)), _COMMON_STATE_KEY),
        ((("iteration", None, 1), ("weight", (0,), 8)), _COMMON_STATE_KEY),
        ((("weight", (0,), 8), ("iteration", None, 1)), "other_common_state"),
    ],
)
def test_registration_rejects_changed_planning_metadata(later_items: tuple, later_key: str) -> None:
    root = _create_root()
    state = {"weight": torch.ones(2), "iteration": 7}
    root.register_state_dict(
        state,
        write_items=(("weight", (0,), 8), ("iteration", None, 1)),
        common_state_key=_COMMON_STATE_KEY,
    )
    root.wait_for_async_completion()

    root.register_state_dict(state, write_items=later_items, common_state_key=later_key)

    with pytest.raises(RuntimeError, match="planning metadata changed"):
        root.wait_for_async_completion()


def test_empty_registrations_retain_the_same_metadata() -> None:
    root = _create_root()
    root.register_state_dict({}, common_state_key=_COMMON_STATE_KEY)
    root.wait_for_async_completion()
    root.register_state_dict({}, common_state_key=_COMMON_STATE_KEY)

    assert root.snapshot_size == 0


@pytest.mark.parametrize("write_items", [(("weight", (0,)),), (("weight", (0,), -1),), (("weight", (-1,), 8),)])
def test_registration_rejects_invalid_write_descriptors(write_items: tuple) -> None:
    root = _create_root()

    with pytest.raises((TypeError, ValueError)):
        root.register_state_dict({}, write_items=write_items, common_state_key=_COMMON_STATE_KEY)


def test_registration_metadata_is_keyword_only() -> None:
    root = _create_root()

    with pytest.raises(TypeError):
        root.register_state_dict({}, (), _COMMON_STATE_KEY)
