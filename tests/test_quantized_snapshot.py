# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeMXFP8Tensor:
    # Mirrors Transformer Engine's storage constructor.
    def __init__(  # noqa: PLR0913
        self,
        *,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor | None,
        columnwise_scale_inv: torch.Tensor | None,
        fp8_dtype: object,
        quantizer: object | None,
        with_gemm_swizzled_scales: bool,
        shape: tuple[int, int],
        dtype: torch.dtype,
        **_kwargs: object,
    ) -> None:
        self._rowwise_data = rowwise_data
        self._rowwise_scale_inv = rowwise_scale_inv
        self._columnwise_data = columnwise_data
        self._columnwise_scale_inv = columnwise_scale_inv
        self._fp8_dtype = fp8_dtype
        self._quantizer = quantizer
        self._with_gemm_swizzled_scales = with_gemm_swizzled_scales
        self.shape = shape
        self.dtype = dtype

    def narrow(self, _dim: int, _start: int, _length: int) -> None:
        raise AssertionError("MXFP8 narrow dispatched to the high-precision fallback")

    def get_metadata(self) -> dict[str, object]:
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
            "with_gemm_swizzled_scales": self._with_gemm_swizzled_scales,
        }


class _FakeNVFP4Tensor:
    # Mirrors Transformer Engine's storage constructor.
    def __init__(  # noqa: PLR0913
        self,
        *,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor | None,
        columnwise_scale_inv: torch.Tensor | None,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor | None,
        fp4_dtype: object,
        quantizer: object | None,
        with_gemm_swizzled_scales: bool,
        row_scaled_nvfp4: bool,
        shape: tuple[int, int],
        dtype: torch.dtype,
        **_kwargs: object,
    ) -> None:
        self._rowwise_data = rowwise_data
        self._rowwise_scale_inv = rowwise_scale_inv
        self._columnwise_data = columnwise_data
        self._columnwise_scale_inv = columnwise_scale_inv
        self._amax_rowwise = amax_rowwise
        self._amax_columnwise = amax_columnwise
        self._fp4_dtype = fp4_dtype
        self._quantizer = quantizer
        self._with_gemm_swizzled_scales = with_gemm_swizzled_scales
        self._row_scaled_nvfp4 = row_scaled_nvfp4
        self.shape = shape
        self.dtype = dtype

    def narrow(self, _dim: int, _start: int, _length: int) -> None:
        raise AssertionError("NVFP4 narrow dispatched to the high-precision fallback")

    def get_metadata(self) -> dict[str, object]:
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "amax_rowwise": self._amax_rowwise,
            "amax_columnwise": self._amax_columnwise,
            "fp4_dtype": self._fp4_dtype,
            "quantizer": self._quantizer,
            "with_gemm_swizzled_scales": self._with_gemm_swizzled_scales,
            "row_scaled_nvfp4": self._row_scaled_nvfp4,
        }


@dataclass
class _FakeShardedTensor:
    data: object
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    prepend_axis_num: int = 0

    def narrow(self, dim: int, start: int, length: int) -> list[_FakeShardedTensor]:
        data = self.data.narrow(dim, start, length)
        return [
            _FakeShardedTensor(
                data=data,
                local_shape=tuple(data.shape),
                global_shape=self.global_shape,
                global_offset=self.global_offset,
                prepend_axis_num=self.prepend_axis_num,
            ),
        ]


def _load_module(monkeypatch: pytest.MonkeyPatch, gtp_api: object) -> ModuleType:
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.__path__ = []
    tensor_parallel = ModuleType("megatron.core.tensor_parallel")
    tensor_parallel.__path__ = []
    tensor_parallel.gtp_api = gtp_api
    dist_checkpointing = ModuleType("megatron.core.dist_checkpointing")
    dist_checkpointing.__path__ = []
    mapping = ModuleType("megatron.core.dist_checkpointing.mapping")
    mapping.ShardedTensor = _FakeShardedTensor
    dist_checkpointing.mapping = mapping
    core.tensor_parallel = tensor_parallel
    core.dist_checkpointing = dist_checkpointing
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(sys.modules, "megatron.core.dist_checkpointing", dist_checkpointing)
    monkeypatch.setitem(sys.modules, "megatron.core.dist_checkpointing.mapping", mapping)

    spec = importlib.util.spec_from_file_location("quantized_snapshot_under_test", REPO_ROOT / "quantized_snapshot.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_quantized_gtp_param(encoding: str) -> SimpleNamespace:
    if encoding == "fp8":
        return SimpleNamespace(_data=object(), _scale_inv=object(), _fp8_dtype=object())
    attributes = {
        "_rowwise_data": object(),
        "_rowwise_scale_inv": object(),
        "_with_gemm_swizzled_scales": False,
        "get_metadata": dict,
    }
    if encoding == "mxfp8":
        attributes.update(
            _columnwise_data=None,
            _columnwise_scale_inv=None,
            _fp8_dtype=object(),
        )
    else:
        attributes.update(_amax_rowwise=object(), _fp4_dtype=object())
    return SimpleNamespace(**attributes)


@pytest.mark.parametrize("encoding", ["fp8", "mxfp8", "nvfp4"])
def test_context_keeps_gtp_parameter_quantized_and_restores_backlink(
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    original = lambda param: f"dequantized-{param}"  # noqa: E731
    gtp_api = SimpleNamespace(HAVE_GTP=True, dequantize_gtp_native_fp8=original)
    module = _load_module(monkeypatch, gtp_api)
    param = _fake_quantized_gtp_param(encoding)

    with module.preserve_quantized_tensors():
        assert gtp_api.dequantize_gtp_native_fp8(param) is param
        assert gtp_api.dequantize_gtp_native_fp8(param) is param
        param._gtp_dequant_src = param

    assert gtp_api.dequantize_gtp_native_fp8 is original
    assert not hasattr(param, "_gtp_dequant_src")


def test_context_dequantizes_unsupported_gtp_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    dequantized = object()
    calls = []

    def original(param):  # noqa: ANN001, ANN202
        calls.append(param)
        return dequantized

    gtp_api = SimpleNamespace(HAVE_GTP=True, dequantize_gtp_native_fp8=original)
    module = _load_module(monkeypatch, gtp_api)
    param = SimpleNamespace()

    with module.preserve_quantized_tensors():
        assert gtp_api.dequantize_gtp_native_fp8(param) is dequantized

    assert calls == [param]


def test_context_restores_existing_backlink_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lambda param: f"dequantized-{param}"  # noqa: E731
    gtp_api = SimpleNamespace(HAVE_GTP=True, dequantize_gtp_native_fp8=original)
    module = _load_module(monkeypatch, gtp_api)
    previous = object()
    param = _fake_quantized_gtp_param("mxfp8")
    param._gtp_dequant_src = previous
    original_narrow = _FakeShardedTensor.narrow

    def fail_while_patched() -> None:
        with module.preserve_quantized_tensors():
            assert gtp_api.dequantize_gtp_native_fp8(param) is param
            param._gtp_dequant_src = param
            raise RuntimeError("state dict failed")

    with pytest.raises(RuntimeError, match="state dict failed"):
        fail_while_patched()

    assert gtp_api.dequantize_gtp_native_fp8 is original
    assert param._gtp_dequant_src is previous
    assert _FakeShardedTensor.narrow is original_narrow


def test_context_is_noop_without_gtp(monkeypatch: pytest.MonkeyPatch) -> None:
    gtp_api = SimpleNamespace(HAVE_GTP=False)
    module = _load_module(monkeypatch, gtp_api)

    with module.preserve_quantized_tensors():
        pass


def test_context_is_noop_without_tensor_parallel_package(monkeypatch: pytest.MonkeyPatch) -> None:
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.__path__ = []
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.delitem(sys.modules, "megatron.core.tensor_parallel", raising=False)

    spec = importlib.util.spec_from_file_location("quantized_snapshot_without_gtp", REPO_ROOT / "quantized_snapshot.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with module.preserve_quantized_tensors():
        pass


def test_context_chunks_mxfp8_as_packed_views(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, SimpleNamespace(HAVE_GTP=False))
    rowwise_data = torch.arange(256 * 64, dtype=torch.uint8).reshape(256, 64)
    rowwise_scale_inv = torch.arange(256 * 2, dtype=torch.uint8).reshape(256, 2)
    columnwise_data = rowwise_data.clone()
    columnwise_scale_inv = torch.arange(8 * 64, dtype=torch.uint8).reshape(8, 64)
    tensor = _FakeMXFP8Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=object(),
        quantizer=object(),
        with_gemm_swizzled_scales=False,
        shape=(256, 64),
        dtype=torch.bfloat16,
    )
    original_chunk = torch.chunk

    with module.preserve_quantized_tensors():
        first, second = torch.chunk(tensor, 2, dim=0)

    assert torch.chunk is original_chunk
    assert first.shape == second.shape == (128, 64)
    assert first._rowwise_data.data_ptr() == rowwise_data.data_ptr()
    assert second._rowwise_data.data_ptr() == rowwise_data[128:].data_ptr()
    assert first._rowwise_scale_inv.data_ptr() == rowwise_scale_inv.data_ptr()
    assert second._rowwise_scale_inv.data_ptr() == rowwise_scale_inv[128:].data_ptr()
    assert first._columnwise_data is None
    assert second._columnwise_data is None
    assert first._columnwise_scale_inv is None
    assert second._columnwise_scale_inv is None
    assert first._quantizer is None
    assert second._quantizer is None


def test_context_narrows_mxfp8_sharded_tensor_as_packed_views(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, SimpleNamespace(HAVE_GTP=False))
    rowwise_data = torch.arange(256 * 64, dtype=torch.uint8).reshape(256, 64)
    rowwise_scale_inv = torch.arange(256 * 2, dtype=torch.uint8).reshape(256, 2)
    columnwise_data = rowwise_data.clone()
    columnwise_scale_inv = torch.arange(8 * 64, dtype=torch.uint8).reshape(8, 64)
    tensor = _FakeMXFP8Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=object(),
        quantizer=object(),
        with_gemm_swizzled_scales=False,
        shape=(256, 64),
        dtype=torch.bfloat16,
    )
    shard = _FakeShardedTensor(
        data=tensor,
        local_shape=(256, 64),
        global_shape=(512, 64),
        global_offset=(256, 0),
    )
    original_narrow = _FakeShardedTensor.narrow

    with module.preserve_quantized_tensors():
        (narrowed,) = shard.narrow(0, 128, 128)

    assert _FakeShardedTensor.narrow is original_narrow
    assert narrowed.local_shape == (128, 64)
    assert narrowed.global_shape == (256, 64)
    assert narrowed.global_offset == (128, 0)
    assert narrowed.data._rowwise_data.data_ptr() == rowwise_data[128:].data_ptr()
    assert narrowed.data._rowwise_scale_inv.data_ptr() == rowwise_scale_inv[128:].data_ptr()
    assert narrowed.data._columnwise_data is None
    assert narrowed.data._columnwise_scale_inv is None
    assert narrowed.data._quantizer is None


def test_context_narrows_nvfp4_sharded_tensor_as_packed_views(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, SimpleNamespace(HAVE_GTP=False))
    rowwise_data = torch.zeros((4640, 256), dtype=torch.uint8)
    rowwise_scale_inv = torch.zeros((4736, 32), dtype=torch.uint8)
    tensor = _FakeNVFP4Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=torch.zeros((512, 2320), dtype=torch.uint8),
        columnwise_scale_inv=torch.zeros((512, 292), dtype=torch.uint8),
        amax_rowwise=torch.ones(1),
        amax_columnwise=torch.ones(1),
        fp4_dtype=object(),
        quantizer=object(),
        with_gemm_swizzled_scales=True,
        row_scaled_nvfp4=False,
        shape=(4640, 512),
        dtype=torch.bfloat16,
    )
    shard = _FakeShardedTensor(
        data=tensor,
        local_shape=(4640, 512),
        global_shape=(9280, 512),
        global_offset=(4640, 0),
    )
    original_narrow = _FakeShardedTensor.narrow

    with module.preserve_quantized_tensors():
        (narrowed,) = shard.narrow(0, 4352, 288)

    assert _FakeShardedTensor.narrow is original_narrow
    assert narrowed.local_shape == (288, 512)
    assert narrowed.global_shape == (576, 512)
    assert narrowed.global_offset == (288, 0)
    assert narrowed.data._rowwise_data.shape == (288, 256)
    assert narrowed.data._rowwise_scale_inv.shape == (384, 32)
    assert narrowed.data._rowwise_data.data_ptr() == rowwise_data[4352:].data_ptr()
    assert narrowed.data._rowwise_scale_inv.data_ptr() == rowwise_scale_inv[4352:].data_ptr()
    assert narrowed.data._columnwise_data is None
    assert narrowed.data._columnwise_scale_inv is None
    assert narrowed.data._amax_columnwise is None
    assert narrowed.data._quantizer is None


def test_installed_sharded_narrow_handles_reference_provider_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, SimpleNamespace(HAVE_GTP=False))
    rowwise_data = torch.zeros((4640, 256), dtype=torch.uint8)
    rowwise_scale_inv = torch.zeros((4736, 32), dtype=torch.uint8)
    tensor = _FakeNVFP4Tensor(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=torch.zeros((512, 2320), dtype=torch.uint8),
        columnwise_scale_inv=torch.zeros((512, 292), dtype=torch.uint8),
        amax_rowwise=torch.ones(1),
        amax_columnwise=torch.ones(1),
        fp4_dtype=object(),
        quantizer=object(),
        with_gemm_swizzled_scales=True,
        row_scaled_nvfp4=False,
        shape=(4640, 512),
        dtype=torch.bfloat16,
    )
    shard = _FakeShardedTensor(
        data=tensor,
        local_shape=(4640, 512),
        global_shape=(9280, 512),
        global_offset=(4640, 0),
    )
    original_narrow = _FakeShardedTensor.narrow
    marker = module._SHARDED_NARROW_MARKER
    try:
        module.install_quantized_sharded_narrow()
        installed_narrow = _FakeShardedTensor.narrow
        module.install_quantized_sharded_narrow()

        assert _FakeShardedTensor.narrow is installed_narrow
        (narrowed,) = shard.narrow(0, 0, 2048)
        assert narrowed.local_shape == (2048, 512)
        assert narrowed.data._rowwise_data.data_ptr() == rowwise_data.data_ptr()
        assert narrowed.data._rowwise_scale_inv.shape == (2048, 32)
        assert narrowed.data._columnwise_data is None
    finally:
        _FakeShardedTensor.narrow = original_narrow
        if hasattr(_FakeShardedTensor, marker):
            delattr(_FakeShardedTensor, marker)
