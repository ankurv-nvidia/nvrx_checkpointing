# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

import pytest
import torch
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron.mapping import ShardedObject, ShardedTensor, ShardedTensorFactory


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


def test_register_state_dict_repoints_sharded_tensor_data() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedTensor.from_rank_offsets("weight", torch.arange(4, dtype=torch.float32), (0, 0, 1))
    fresh = ShardedTensor.from_rank_offsets("weight", torch.arange(4, dtype=torch.float32) + 10, (0, 0, 1))
    state = {"weight": initial}
    root.register_state_dict(state)

    root.register_state_dict({"weight": fresh})

    assert state["weight"] is fresh


def test_register_state_dict_refreshes_when_the_same_dictionary_is_reused() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"rng_state": ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))}
    root.register_state_dict(state)
    state["rng_state"] = ShardedObject("rng_state", {"iteration": 2, "payload": "x" * 4096}, (1,), (0,))

    with pytest.raises(RuntimeError, match=r"exceeding its registered .* snapshot capacity"):
        root.register_state_dict(state)


def test_register_state_dict_accepts_dotted_sharded_object_key() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"rng_state": ShardedObject("nested.rng_state", {"iteration": 1}, (1,), (0,))}
    root.register_state_dict(state)

    root.register_state_dict({"rng_state": ShardedObject("nested.rng_state", {"iteration": 2}, (1,), (0,))})

    assert state["rng_state"].data == {"iteration": 2}


def test_register_state_dict_is_the_only_state_registration_method() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)

    assert not hasattr(root, "capture_state")
    assert not hasattr(root, "register")
    assert not hasattr(root, "register_state_dict_async")
    assert not hasattr(root, "register_scalars_in_dict")
    assert not hasattr(root, "register_weight_tensor")
    assert not hasattr(root, "register_immediate_tensor")


def test_register_state_dict_refreshes_sharded_tensor_factory_outputs() -> None:
    def factory(dtype: torch.dtype) -> ShardedTensorFactory:
        tensor = torch.arange(4, dtype=dtype)
        return ShardedTensorFactory(
            key="weight",
            data=tensor,
            build_fn=lambda key, data, *_args: [ShardedTensor.from_rank_offsets(key, data, (0, 0, 1))],
            merge_fn=None,
        )

    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict({"weight": factory(torch.float32)})

    with pytest.raises(RuntimeError, match=r"changed tensor metadata"):
        root.register_state_dict({"weight": factory(torch.float64)})


def test_register_state_dict_accepts_nested_and_dotted_partial_state() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))
    state = {
        "rng_state": initial,
        "nested": {"step": 1, "untouched": 2, "none": None, "empty": []},
    }
    root.register_state_dict(state)

    root.register_state_dict(
        {
            "rng_state": ShardedObject("rng_state", {"iteration": 2}, (1,), (0,)),
            "nested.step": 3,
            "nested": {"none": None, "empty": []},
        },
    )

    assert state["rng_state"].data == {"iteration": 2}
    assert state["nested"]["step"] == 3
    assert state["nested"]["untouched"] == 2


def test_register_state_dict_matches_optimizer_tuple_keys() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    dtype_pair = (torch.bfloat16, torch.float32)
    state = {"optimizer": {dtype_pair: {"step": 1}}}
    root.register_state_dict(state)

    root.register_state_dict({"optimizer": {dtype_pair: {"step": 2}}})
    assert state["optimizer"][dtype_pair]["step"] == 2


def test_register_state_dict_matches_nested_optimizer_bucket_lists() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    dtype_pair = (torch.bfloat16, torch.float32)
    initial = {"optimizer": {"param_state": {0: {dtype_pair: [[{"step": 1}]]}}}}
    root.register_state_dict(initial)

    refreshed = {"optimizer": {"param_state": {0: {dtype_pair: [[{"step": 2}]]}}}}
    root.register_state_dict(refreshed)
    assert initial["optimizer"]["param_state"][0][dtype_pair][0][0]["step"] == 2


@pytest.mark.parametrize("rank", [0, 1])
def test_register_state_dict_rejects_snapshot_capacity_overflow(rank: int) -> None:
    root = checkpoint.model_state.create_root(rank, 2, ["coordinator", "remote"], 1)
    initial = ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))
    state = {"rng_state": initial}
    root.register_state_dict(state)
    fresh = ShardedObject("rng_state", {"iteration": 2, "payload": "x" * 4096}, (1,), (0,))

    with pytest.raises(RuntimeError, match=r"exceeding its registered .* snapshot capacity"):
        root.register_state_dict({"rng_state": fresh})


def test_register_state_dict_rejects_duplicate_normalized_paths() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"a": {"b": 1}}
    root.register_state_dict(state)

    with pytest.raises(RuntimeError, match=r"duplicate normalized state 'a.b'"):
        root.register_state_dict({"a.b": 2, "a": {"b": 3}})


@pytest.mark.parametrize(
    ("fresh", "path"),
    [
        ({"unknown": 2}, "unknown"),
        ({"unknown": None}, "unknown"),
        ({"unknown": []}, "unknown"),
        ({"unknown": ()}, "unknown"),
        ({"known": {"unknown": None}}, "known.unknown"),
    ],
)
def test_register_state_dict_rejects_unregistered_path(fresh: dict, path: str) -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"known": {"value": 1}}
    root.register_state_dict(state)

    with pytest.raises(RuntimeError, match=rf"supplied state '{path}' is not registered"):
        root.register_state_dict(fresh)
