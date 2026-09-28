# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

import gc
import threading
import weakref
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, Self

import pytest
import torch
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron.mapping import ShardedObject, ShardedTensor, ShardedTensorFactory

if TYPE_CHECKING:
    from pathlib import Path


class _CudaGraphModule(Enum):
    MAMBA = auto()
    ATTN = auto()
    MOE_ROUTER = auto()


class _StorageLessQuantizedTensor(torch.Tensor):
    dequantized: ClassVar[list[weakref.ReferenceType[torch.Tensor]]] = []

    @staticmethod
    def __new__(cls: type[Self], source: torch.Tensor) -> Self:
        tensor = torch.Tensor._make_wrapper_subclass(  # noqa: SLF001
            cls,
            source.shape,
            dtype=source.dtype,
            device=source.device,
            requires_grad=False,
        )
        tensor.source = source
        return tensor

    @classmethod
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):  # noqa: ANN001, ANN206
        del kwargs
        if func == torch.ops.aten.detach.default:
            return args[0]
        if func == torch.ops.aten.is_pinned.default:
            return False
        raise NotImplementedError(func)

    def dequantize(self) -> torch.Tensor:
        result = self.source.clone()
        self.dequantized.append(weakref.ref(result))
        return result


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
    assert root.snapshot_size == sum(tensor.data.numel() * tensor.data.element_size() for tensor in tensors)


def test_register_state_dict_repoints_sharded_tensor_data() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedTensor.from_rank_offsets("weight", torch.arange(4, dtype=torch.float32), (0, 0, 1))
    fresh = ShardedTensor.from_rank_offsets("weight", torch.arange(4, dtype=torch.float32) + 10, (0, 0, 1))
    state = {"weight": initial}
    root.register_state_dict(state)
    root.finalize_registry()

    refreshed = {"weight": fresh}
    root.register_state_dict(refreshed)
    assert root.snapshot_size == fresh.data.numel() * fresh.data.element_size()

    assert state["weight"] is initial
    assert refreshed["weight"] is fresh


def test_register_state_dict_releases_superseded_python_objects() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    tensor = torch.arange(4, dtype=torch.float32)
    tensor_ref = weakref.ref(tensor)
    state = {"weight": ShardedTensor.from_rank_offsets("weight", tensor, (0, 0, 1))}

    root.register_state_dict(state)
    root.finalize_registry()
    root.register_state_dict(
        {"weight": ShardedTensor.from_rank_offsets("weight", torch.arange(4, dtype=torch.float32), (0, 0, 1))},
    )
    assert root.snapshot_size == 4 * 4
    del state, tensor
    gc.collect()

    assert tensor_ref() is None


def test_register_state_dict_materializes_storage_less_quantized_tensor() -> None:
    _StorageLessQuantizedTensor.dequantized.clear()
    data = _StorageLessQuantizedTensor(torch.arange(8, dtype=torch.bfloat16))
    assert data.data_ptr() == 0
    state = {
        "weight": ShardedTensor.from_rank_offsets("weight", data, (0, 0, 1)),
    }

    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict(state)
    assert root.snapshot_size == 16

    assert len(_StorageLessQuantizedTensor.dequantized) == 1
    gc.collect()
    assert _StorageLessQuantizedTensor.dequantized[0]() is not None

    root.finalize_registry()
    fresh_data = _StorageLessQuantizedTensor(torch.arange(8, dtype=torch.bfloat16) + 1)
    root.register_state_dict(
        {"weight": ShardedTensor.from_rank_offsets("weight", fresh_data, (0, 0, 1))},
    )
    assert root.snapshot_size == 16

    assert len(_StorageLessQuantizedTensor.dequantized) == 2
    gc.collect()
    assert _StorageLessQuantizedTensor.dequantized[0]() is None
    assert _StorageLessQuantizedTensor.dequantized[1]() is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_save_materializes_storage_less_cuda_tensor(tmp_path: Path) -> None:
    _StorageLessQuantizedTensor.dequantized.clear()
    source = torch.arange(8, dtype=torch.bfloat16, device="cuda")
    data = _StorageLessQuantizedTensor(source)
    assert data.data_ptr() == 0

    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict(
        {"weight": ShardedTensor.from_rank_offsets("weight", data, (0, 0, 1))},
    )
    assert root.snapshot_size == 16

    assert len(_StorageLessQuantizedTensor.dequantized) == 1
    materialized = _StorageLessQuantizedTensor.dequantized[0]()
    assert materialized is not None
    assert materialized.is_cuda

    try:
        root.save(1, tmp_path / "iter_0000001")
    finally:
        checkpoint.svc_spindown()

    torch.testing.assert_close(materialized, source)
    del materialized
    gc.collect()
    assert _StorageLessQuantizedTensor.dequantized[0]() is None


def test_register_state_dict_refreshes_when_the_same_dictionary_is_reused() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"rng_state": ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))}
    root.register_state_dict(state)
    assert root.snapshot_size > 0
    root.finalize_registry()
    state["rng_state"] = ShardedObject("rng_state", {"iteration": 2, "payload": "x" * 4096}, (1,), (0,))

    root.register_state_dict(state)
    with pytest.raises(RuntimeError, match=r"changed serialized size after registry synchronization"):
        _ = root.snapshot_size


def test_register_state_dict_accepts_dotted_sharded_object_key() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"rng_state": ShardedObject("nested.rng_state", {"iteration": 1}, (1,), (0,))}
    root.register_state_dict(state)
    root.finalize_registry()

    refreshed = {"rng_state": ShardedObject("nested.rng_state", {"iteration": 2}, (1,), (0,))}
    root.register_state_dict(refreshed)
    assert root.snapshot_size > 0

    assert state["rng_state"].data == {"iteration": 1}
    assert refreshed["rng_state"].data == {"iteration": 2}


def test_register_state_dict_uses_prepared_sharded_object_blob() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))
    setattr(initial, "_nvrx_serialized_blob", b"a" * 2048)
    root.register_state_dict({"rng_state": initial})
    root.finalize_registry()

    refreshed = ShardedObject("rng_state", {"iteration": 2}, (1,), (0,))
    setattr(refreshed, "_nvrx_serialized_blob", b"b" * 2048)
    root.register_state_dict({"rng_state": refreshed})

    assert root.snapshot_size == 2048


def test_register_state_dict_is_the_only_state_registration_method() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)

    assert not hasattr(root, "capture_state")
    assert not hasattr(root, "register")
    assert not hasattr(root, "register_state_dict_async")
    assert not hasattr(root, "register_scalars_in_dict")
    assert not hasattr(root, "register_weight_tensor")
    assert not hasattr(root, "register_immediate_tensor")
    assert not hasattr(root, "add_child")
    assert not hasattr(root, "warmup_for_snapshot")
    assert not hasattr(checkpoint.model_state, "warmup")


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
    assert root.snapshot_size == 4 * 4
    root.finalize_registry()

    root.register_state_dict({"weight": factory(torch.float64)})
    with pytest.raises(RuntimeError, match=r"changed tensor metadata"):
        _ = root.snapshot_size


def test_state_registration_runs_asynchronously() -> None:
    calls: list[tuple[int, float]] = []

    def factory(value: float) -> ShardedTensorFactory:
        tensor = torch.full((4,), value, dtype=torch.float32)

        def build(key: str, data: torch.Tensor, *_args: object) -> list[ShardedTensor]:
            calls.append((threading.get_ident(), value))
            return [ShardedTensor.from_rank_offsets(key, data, (0, 0, 1))]

        return ShardedTensorFactory(key="weight", data=tensor, build_fn=build, merge_fn=None)

    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict({"weight": factory(1)})
    assert root.snapshot_size == 4 * 4
    root.finalize_registry()
    root.register_state_dict({"weight": factory(2)})
    assert root.snapshot_size == 4 * 4

    assert [value for _, value in calls] == [1, 2]
    assert all(thread != threading.get_ident() for thread, _ in calls)


def test_register_state_dict_rebinds_a_complete_nested_state() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))
    state = {
        "rng_state": initial,
        "nested": {"step": 1, "untouched": 2, "none": None, "empty": []},
    }
    root.register_state_dict(state)
    root.finalize_registry()

    refreshed = {
        "rng_state": ShardedObject("rng_state", {"iteration": 2}, (1,), (0,)),
        "nested": {"step": 3, "untouched": 2, "none": None, "empty": []},
    }
    root.register_state_dict(refreshed)
    assert root.snapshot_size > 0

    assert state["rng_state"].data == {"iteration": 1}
    assert state["nested"]["step"] == 1
    assert state["nested"]["untouched"] == 2
    assert refreshed["rng_state"].data == {"iteration": 2}
    assert refreshed["nested"]["step"] == 3


def test_register_state_dict_matches_optimizer_tuple_keys() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    dtype_pair = (torch.bfloat16, torch.float32)
    state = {"optimizer": {dtype_pair: {"step": 1}}}
    root.register_state_dict(state)
    root.finalize_registry()

    refreshed = {"optimizer": {dtype_pair: {"step": 2}}}
    root.register_state_dict(refreshed)
    assert root.snapshot_size == 0
    assert state["optimizer"][dtype_pair]["step"] == 1
    assert refreshed["optimizer"][dtype_pair]["step"] == 2


def test_register_state_dict_matches_nested_optimizer_bucket_lists() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    dtype_pair = (torch.bfloat16, torch.float32)
    initial = {"optimizer": {"param_state": {0: {dtype_pair: [[{"step": 1}]]}}}}
    root.register_state_dict(initial)
    root.finalize_registry()

    refreshed = {"optimizer": {"param_state": {0: {dtype_pair: [[{"step": 2}]]}}}}
    root.register_state_dict(refreshed)
    assert root.snapshot_size == 0
    assert initial["optimizer"]["param_state"][0][dtype_pair][0][0]["step"] == 1
    assert refreshed["optimizer"]["param_state"][0][dtype_pair][0][0]["step"] == 2


def test_register_load_factory_uses_dotted_checkpoint_key() -> None:
    """Accept Megatron's Mamba conv1d state-name compatibility mapping."""
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    checkpoint_key = "decoder.layers.0.mixer.conv1d.weight"
    factory = ShardedTensorFactory(
        key=checkpoint_key,
        data=tensor,
        build_fn=lambda key, data, replica_id, _flattened_range: [
            ShardedTensor.from_rank_offsets(key, data, replica_id=replica_id),
        ],
        merge_fn=lambda values: values,
    )
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)

    root.register_state_dict({"decoder.layers.0.mixer.conv1d_weight": factory})

    assert root.snapshot_size == tensor.numel() * tensor.element_size()


def test_state_registration_accepts_integer_enum_list() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)

    root.register_state_dict(
        {
            "args": {
                "cuda_graph_modules": [
                    _CudaGraphModule.MAMBA,
                    _CudaGraphModule.ATTN,
                    _CudaGraphModule.MOE_ROUTER,
                ],
            },
        },
    )

    assert root.snapshot_size == 3 * 8
    with pytest.raises(AttributeError):
        root.snapshot_size = 0


@pytest.mark.parametrize("rank", [0, 1])
def test_register_state_dict_rejects_blob_size_changes(rank: int) -> None:
    root = checkpoint.model_state.create_root(rank, 2, ["coordinator", "remote"], 1)
    initial = ShardedObject("rng_state", {"iteration": 1}, (1,), (0,))
    state = {"rng_state": initial}
    root.register_state_dict(state)
    assert root.snapshot_size > 0
    root.finalize_registry()
    fresh = ShardedObject("rng_state", {"iteration": 2, "payload": "x" * 4096}, (1,), (0,))

    root.register_state_dict({"rng_state": fresh})
    with pytest.raises(RuntimeError, match=r"changed serialized size after registry synchronization"):
        _ = root.snapshot_size


def test_register_state_dict_accepts_repeated_normalized_paths() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    initial = ShardedTensor.from_rank_offsets("a.b", torch.arange(4, dtype=torch.float32), (0, 0, 1))
    root.register_state_dict({"a": {"b": initial}})
    root.finalize_registry()
    direct = ShardedTensor.from_rank_offsets("a.b", torch.arange(4, dtype=torch.float32), (0, 0, 1))
    nested = ShardedTensor.from_rank_offsets("a.b", torch.arange(4, dtype=torch.float32), (0, 0, 1))

    root.register_state_dict({"direct": direct, "a": {"b": nested}})
    assert root.snapshot_size == 4 * 4


@pytest.mark.parametrize(
    ("fresh", "path"),
    [
        ({"unknown": 2}, "unknown"),
        ({"known": {"value": 1}, "unknown": 2}, "unknown"),
        ({"known": {"value": 1, "unknown": 2}}, "known.unknown"),
    ],
)
def test_register_state_dict_rejects_unregistered_path(fresh: dict, path: str) -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    state = {"known": {"value": 1}}
    root.register_state_dict(state)
    assert root.snapshot_size == 0
    root.finalize_registry()

    root.register_state_dict(fresh)
    with pytest.raises(RuntimeError, match=rf"supplied state '{path}' is not registered"):
        _ = root.snapshot_size


def test_register_state_dict_allows_missing_registered_path() -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict({"known": {"value": 1}})
    assert root.snapshot_size == 0
    root.finalize_registry()

    root.register_state_dict({"known": {}})
    assert root.snapshot_size == 0


def test_save_rejects_a_missing_registered_tensor(tmp_path: Path) -> None:
    root = checkpoint.model_state.create_root(0, 1, ["localhost"], 1)
    root.register_state_dict({"weight": torch.arange(4, dtype=torch.float32), "step": 1})
    assert root.snapshot_size == 4 * 4
    root.finalize_registry()

    root.register_state_dict({"step": 2})
    root.save_async(1, tmp_path / "iter_0000001")
    with pytest.raises(RuntimeError, match=r"registered tensor 'weight' was not supplied for this snapshot"):
        root.wait_for_deferred_save()
