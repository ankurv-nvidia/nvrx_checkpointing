# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# ruff: noqa: INP001

from __future__ import annotations

import torch
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.mapping import ShardedTensor


def test_register_state_dict_accepts_multiple_two_axis_shards() -> None:
    tensors = [
        ShardedTensor.from_rank_offsets(
            "decoder.layers.mlp.experts.experts.linear_fc1.weight",
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            (0, layer, 16),
            (1, expert, 8),
            (2, 1, 4),
            prepend_axis_num=2,
        )
        for layer, expert in ((3, 5), (3, 6), (4, 5), (4, 6))
    ]
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)

    root.register_state_dict({"weight_shards": tensors})
