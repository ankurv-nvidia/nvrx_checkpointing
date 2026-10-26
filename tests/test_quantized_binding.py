# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001
"""Registration coverage for quantized tensor snapshot payloads."""

from __future__ import annotations

import gc
import os
import sys
import time
import types
import weakref
from itertools import count
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from typing_extensions import Self

try:
    from nvrx_checkpointing import checkpoint
    from nvrx_checkpointing.megatron import mapping as mapping_module
    from nvrx_checkpointing.megatron.mapping import ShardedTensor, ShardedTensorFactory
except ModuleNotFoundError:
    import checkpoint
    from megatron import mapping as mapping_module
    from megatron.mapping import ShardedTensor, ShardedTensorFactory

package = types.ModuleType("nvrx_checkpointing")
package.__path__ = []
megatron_package = types.ModuleType("nvrx_checkpointing.megatron")
megatron_package.__path__ = []
megatron_package.mapping = mapping_module
package.megatron = megatron_package
sys.modules.setdefault("nvrx_checkpointing", package)
sys.modules.setdefault("nvrx_checkpointing.megatron", megatron_package)
sys.modules.setdefault("nvrx_checkpointing.megatron.mapping", mapping_module)

_SESSION_IDS = count(3000)
_QUANTIZED_HEADER_BYTES = 144
_REQUIRES_CHECKPOINT_SERVER = pytest.mark.skipif(
    os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
    reason="requires checkpoint-server IPC support",
)


def _create_root():
    return checkpoint.model_state.create_root(next(_SESSION_IDS), 0, 1, ["localhost"], 1)


class _FP4DType:
    def __str__(self) -> str:
        return "DType.kFloat4E2M1"


class _NumericFP4DType:
    def __str__(self) -> str:
        return "10"


class _FP8DType:
    def __str__(self) -> str:
        return "DType.kFloat8E4M3"


class _NumericFP8DType:
    def __str__(self) -> str:
        return "7"


class _FakeFP8Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls: type[Self], shape: tuple[int, int]) -> Self:
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=torch.bfloat16),
            False,  # noqa: FBT003
        )

    def __dlpack__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("logical FP8 tensor must not be exported or GPU-dequantized")


class _FakeMXFP8Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls: type[Self], shape: tuple[int, int]) -> Self:
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=torch.bfloat16),
            False,  # noqa: FBT003
        )

    def __dlpack__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("logical MXFP8 tensor must not be exported or GPU-dequantized")


class _FakeNVFP4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls: type[Self], shape: tuple[int, int]) -> Self:
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=torch.bfloat16),
            False,  # noqa: FBT003
        )

    def __dlpack__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("logical NVFP4 tensor must not be exported or GPU-dequantized")


@pytest.mark.parametrize("fp8_dtype", [_FP8DType(), _NumericFP8DType()])
def test_fp8_registration_reads_raw_components(fp8_dtype: object) -> None:
    tensor = _FakeFP8Tensor((128, 512))
    tensor._data = torch.arange(128 * 512, dtype=torch.int64).to(torch.uint8).reshape(128, 512)
    tensor._scale_inv = torch.tensor([0.125], dtype=torch.float32)
    tensor._fp8_dtype = fp8_dtype

    root = _create_root()
    root.register_state_dict({"weight": tensor})
    assert root.snapshot_size == (
        _QUANTIZED_HEADER_BYTES + tensor._data.numel() + tensor._scale_inv.numel() * tensor._scale_inv.element_size()
    )


@pytest.mark.parametrize("fp8_dtype", [_FP8DType(), _NumericFP8DType()])
def test_mxfp8_registration_reads_raw_components_without_gtp_marker(fp8_dtype: object) -> None:
    tensor = _FakeMXFP8Tensor((128, 512))
    tensor._rowwise_data = torch.arange(128 * 512, dtype=torch.int64).to(torch.uint8).reshape(128, 512)
    tensor._rowwise_scale_inv = torch.full((128, 16), 120, dtype=torch.uint8)
    tensor._columnwise_data = torch.arange(128 * 512, dtype=torch.int64).to(torch.uint8).reshape(128, 512)
    tensor._columnwise_scale_inv = torch.full((4, 512), 121, dtype=torch.uint8)
    tensor._fp8_dtype = fp8_dtype

    state = {
        "weight": ShardedTensor(
            key="weight",
            data=tensor,
            dtype=torch.bfloat16,
            local_shape=(128, 512),
            global_shape=(128, 512),
            global_offset=(0, 0),
            axis_fragmentations=None,
        ),
    }
    root = _create_root()
    root.register_state_dict(state)
    assert root.snapshot_size == (_QUANTIZED_HEADER_BYTES + tensor._rowwise_data.numel() + tensor._rowwise_scale_inv.numel())


@pytest.mark.parametrize("fp8_dtype", [_FP8DType(), _NumericFP8DType()])
def test_bare_mxfp8_registration_reads_raw_components(fp8_dtype: object) -> None:
    tensor = _FakeMXFP8Tensor((128, 512))
    tensor._rowwise_data = torch.arange(128 * 512, dtype=torch.int64).to(torch.uint8).reshape(128, 512)
    tensor._rowwise_scale_inv = torch.full((128, 16), 120, dtype=torch.uint8)
    tensor._columnwise_data = None
    tensor._columnwise_scale_inv = None
    tensor._fp8_dtype = fp8_dtype
    tensor._with_gemm_swizzled_scales = False

    root = _create_root()
    root.register_state_dict({"weight": tensor})


@pytest.mark.parametrize("fp4_dtype", [_FP4DType(), _NumericFP4DType()])
def test_nvfp4_registration_reads_raw_components(fp4_dtype: object) -> None:
    tensor = _FakeNVFP4Tensor((128, 512))
    tensor._rowwise_data = torch.arange(128 * 256, dtype=torch.int64).to(torch.uint8).reshape(128, 256)
    tensor._rowwise_scale_inv = torch.full((128, 32), 120, dtype=torch.uint8)
    tensor._amax_rowwise = torch.tensor([123.25], dtype=torch.float32)
    tensor._columnwise_data = torch.arange(512 * 64, dtype=torch.int64).to(torch.uint8).reshape(512, 64)
    tensor._columnwise_scale_inv = torch.full((512, 8), 121, dtype=torch.uint8)
    tensor._amax_columnwise = torch.tensor([91.5], dtype=torch.float32)
    tensor._fp4_dtype = fp4_dtype
    tensor._with_gemm_swizzled_scales = True
    tensor._row_scaled_nvfp4 = False
    tensor._nvfp4_e4m3_max = 448

    state = {
        "weight": ShardedTensor(
            key="weight",
            data=tensor,
            dtype=torch.bfloat16,
            local_shape=(128, 512),
            global_shape=(128, 512),
            global_offset=(0, 0),
            axis_fragmentations=None,
        ),
    }
    root = _create_root()
    root.register_state_dict(state)
    assert root.snapshot_size == (
        _QUANTIZED_HEADER_BYTES
        + tensor._rowwise_data.numel()
        + tensor._rowwise_scale_inv.numel()
        + tensor._amax_rowwise.numel() * tensor._amax_rowwise.element_size()
    )


@pytest.mark.parametrize("fp4_dtype", [_FP4DType(), _NumericFP4DType()])
def test_bare_nvfp4_registration_reads_raw_components(fp4_dtype: object) -> None:
    tensor = _FakeNVFP4Tensor((128, 512))
    tensor._rowwise_data = torch.arange(128 * 256, dtype=torch.int64).to(torch.uint8).reshape(128, 256)
    tensor._rowwise_scale_inv = torch.full((128, 32), 120, dtype=torch.uint8)
    tensor._amax_rowwise = torch.tensor([123.25], dtype=torch.float32)
    tensor._columnwise_data = None
    tensor._columnwise_scale_inv = None
    tensor._amax_columnwise = None
    tensor._fp4_dtype = fp4_dtype
    tensor._with_gemm_swizzled_scales = True
    tensor._row_scaled_nvfp4 = False
    tensor._nvfp4_e4m3_max = 448

    root = _create_root()
    root.register_state_dict({"weight": tensor})


@pytest.mark.integration
@_REQUIRES_CHECKPOINT_SERVER
@pytest.mark.parametrize("storage_type", ["fp8", "mxfp8", "nvfp4"])
def test_cpu_quantized_snapshot_dequantizes_during_dcp_write(tmp_path: Path, storage_type: str) -> None:
    if storage_type == "fp8":
        tensor = _FakeFP8Tensor((128, 512))
        tensor._data = torch.full((128, 512), 0x38, dtype=torch.uint8)
        tensor._scale_inv = torch.tensor([1.0], dtype=torch.float32)
        tensor._fp8_dtype = _FP8DType()
    elif storage_type == "mxfp8":
        tensor = _FakeMXFP8Tensor((128, 512))
        tensor._rowwise_data = torch.full((128, 512), 0x38, dtype=torch.uint8)
        tensor._rowwise_scale_inv = torch.full((128, 16), 0x7F, dtype=torch.uint8)
        tensor._fp8_dtype = _FP8DType()
    else:
        tensor = _FakeNVFP4Tensor((128, 512))
        tensor._rowwise_data = torch.full((128, 256), 0x22, dtype=torch.uint8)
        tensor._rowwise_scale_inv = torch.full((128, 32), 0x38, dtype=torch.uint8)
        tensor._amax_rowwise = torch.tensor([2688.0], dtype=torch.float32)
        tensor._fp4_dtype = _FP4DType()
        tensor._row_scaled_nvfp4 = False
        tensor._nvfp4_e4m3_max = 448
    if storage_type != "fp8":
        tensor._with_gemm_swizzled_scales = False

    root = _create_root()
    root.register_state_dict({"weight": tensor})
    checkpoint_dir = tmp_path / storage_type
    try:
        root.save(1, checkpoint_dir)
        deadline = time.monotonic() + 30
        while not (checkpoint_dir / ".metadata").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"checkpoint daemon did not publish {checkpoint_dir}")
            time.sleep(0.05)

        loaded = {"weight": torch.zeros(tensor.shape, dtype=torch.bfloat16)}
        dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
        torch.testing.assert_close(loaded["weight"], torch.ones_like(loaded["weight"]))
    finally:
        checkpoint.svc_spindown()


@pytest.mark.integration
@_REQUIRES_CHECKPOINT_SERVER
def test_nvfp4_factory_shards_keep_distinct_storage(tmp_path: Path) -> None:
    def tensor(packed_value: int) -> _FakeNVFP4Tensor:
        value = _FakeNVFP4Tensor((128, 512))
        value._rowwise_data = torch.full((128, 256), packed_value, dtype=torch.uint8)
        value._rowwise_scale_inv = torch.full((128, 32), 0x38, dtype=torch.uint8)
        value._amax_rowwise = torch.tensor([2688.0], dtype=torch.float32)
        value._fp4_dtype = _FP4DType()
        value._with_gemm_swizzled_scales = False
        value._row_scaled_nvfp4 = False
        value._nvfp4_e4m3_max = 448
        return value

    shards = [tensor(0x22), tensor(0x44)]

    def build(key: str, *_args: object) -> list[ShardedTensor]:
        return [
            ShardedTensor(
                key=key,
                data=value,
                dtype=torch.bfloat16,
                local_shape=(128, 512),
                global_shape=(2, 128, 512),
                global_offset=(index, 0, 0),
                axis_fragmentations=None,
                prepend_axis_num=1,
            )
            for index, value in enumerate(shards)
        ]

    state = {
        "weight": ShardedTensorFactory(
            key="weight",
            data=torch.empty(0),
            build_fn=build,
            merge_fn=None,
        ),
    }
    root = _create_root()
    root.register_state_dict(state)
    checkpoint_dir = tmp_path / "nvfp4-shards"
    try:
        root.save(1, checkpoint_dir)
        deadline = time.monotonic() + 30
        while not (checkpoint_dir / ".metadata").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"checkpoint daemon did not publish {checkpoint_dir}")
            time.sleep(0.05)

        loaded = {"weight": torch.zeros((2, 128, 512), dtype=torch.bfloat16)}
        dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
        torch.testing.assert_close(loaded["weight"][0], torch.ones((128, 512), dtype=torch.bfloat16))
        torch.testing.assert_close(loaded["weight"][1], torch.full((128, 512), 2.0, dtype=torch.bfloat16))
    finally:
        checkpoint.svc_spindown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantized_registration_rejects_components_on_different_devices() -> None:
    tensor = _FakeMXFP8Tensor((128, 512))
    tensor._rowwise_data = torch.zeros((128, 512), dtype=torch.uint8, device="cuda")
    tensor._rowwise_scale_inv = torch.ones((128, 16), dtype=torch.uint8)
    tensor._fp8_dtype = _FP8DType()
    tensor._with_gemm_swizzled_scales = False

    root = _create_root()
    with pytest.raises(RuntimeError, match="components must be on one device"):
        root.register_state_dict({"weight": tensor})


def test_quantized_sharded_tensor_data_follows_registration_lifecycle() -> None:
    def make_state() -> tuple[dict[str, ShardedTensor], weakref.ReferenceType[torch.Tensor]]:
        tensor = _FakeMXFP8Tensor((128, 512))
        tensor._rowwise_data = torch.zeros((128, 512), dtype=torch.uint8)
        tensor._rowwise_scale_inv = torch.full((128, 16), 120, dtype=torch.uint8)
        tensor._columnwise_data = None
        tensor._columnwise_scale_inv = None
        tensor._fp8_dtype = _FP8DType()
        tensor._with_gemm_swizzled_scales = False
        return {
            "weight": ShardedTensor(
                key="weight",
                data=tensor,
                dtype=torch.bfloat16,
                local_shape=(128, 512),
                global_shape=(128, 512),
                global_offset=(0, 0),
                axis_fragmentations=None,
            ),
        }, weakref.ref(tensor)

    state, first_ref = make_state()
    root = _create_root()
    root.register_state_dict(state)
    del state
    gc.collect()
    assert first_ref() is not None

    assert root.snapshot_size > 0

    refreshed, second_ref = make_state()
    root.register_state_dict(refreshed)
    del refreshed
    assert root.snapshot_size > 0
    gc.collect()
    assert first_ref() is None
    assert second_ref() is not None

    del root
    gc.collect()
    assert second_ref() is None
