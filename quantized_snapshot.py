# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Preserve packed Transformer Engine tensors while constructing a snapshot."""

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import replace
from functools import wraps
from threading import RLock

import torch

_PATCH_LOCK = RLock()
_SHARDED_NARROW_MARKER = "_nvrx_checkpointing_quantized_narrow_installed"
_MXFP8_BLOCK_SIZE = 32
_NVFP4_BLOCK_SIZE = 16
_SCALE_ROW_ALIGNMENT = 128
_WEIGHT_DIMENSIONS = 2


def _has_attributes(value: object, attributes: tuple[str, ...]) -> bool:
    return all(hasattr(value, attribute) for attribute in attributes)


def _is_mxfp8_tensor(tensor: object) -> bool:
    """Return whether a tensor exposes Transformer Engine MXFP8 storage."""
    return _has_attributes(
        tensor,
        (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_columnwise_data",
            "_columnwise_scale_inv",
            "_fp8_dtype",
            "_with_gemm_swizzled_scales",
            "get_metadata",
        ),
    )


def _is_fp8_tensor(tensor: object) -> bool:
    """Return whether a tensor exposes Transformer Engine FP8 storage."""
    return _has_attributes(tensor, ("_data", "_scale_inv", "_fp8_dtype"))


def _is_nvfp4_tensor(tensor: object) -> bool:
    """Return whether a tensor exposes Transformer Engine NVFP4 storage."""
    return _has_attributes(
        tensor,
        (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_amax_rowwise",
            "_fp4_dtype",
            "_with_gemm_swizzled_scales",
            "get_metadata",
        ),
    )


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _validate_row_slice(tensor: object, start: int, length: int, block_size: int, encoding: str) -> tuple[int, int]:
    shape = tuple(tensor.shape)
    if len(shape) != _WEIGHT_DIMENSIONS or start < 0 or length <= 0 or start + length > shape[0]:
        raise ValueError(f"Unsupported {encoding} row slice start={start}, length={length}, shape={shape}")
    if shape[1] % block_size != 0:
        raise ValueError(f"{encoding} inner dimension must be divisible by {block_size}, got {shape[1]}")
    return shape


def _slice_mxfp8_rows(tensor: object, start: int, length: int) -> object:
    """Construct an MXFP8 row slice backed by views of the packed storage."""
    shape = _validate_row_slice(tensor, start, length, _MXFP8_BLOCK_SIZE, "MXFP8")
    if tensor._with_gemm_swizzled_scales:  # noqa: SLF001
        raise ValueError("MXFP8 checkpoint row slices require compact scales")
    if start % _SCALE_ROW_ALIGNMENT != 0:
        raise ValueError(f"MXFP8 row slices must start on a {_SCALE_ROW_ALIGNMENT}-row boundary, got {start}")

    metadata = tensor.get_metadata()
    rowwise_data = tensor._rowwise_data  # noqa: SLF001
    rowwise_scale_inv = tensor._rowwise_scale_inv  # noqa: SLF001
    if rowwise_data is None or rowwise_scale_inv is None:
        raise ValueError("MXFP8 rowwise data and scales are required for checkpointing")
    metadata["rowwise_data"] = rowwise_data.narrow(0, start, length)
    metadata["rowwise_scale_inv"] = rowwise_scale_inv.narrow(0, start, _round_up(length, _SCALE_ROW_ALIGNMENT))
    metadata["columnwise_data"] = None
    metadata["columnwise_scale_inv"] = None
    metadata["quantizer"] = None
    return type(tensor)(**metadata, shape=(length, shape[1]), dtype=tensor.dtype)


def _slice_nvfp4_rows(tensor: object, start: int, length: int) -> object:
    """Construct an NVFP4 row slice backed by views of the packed storage."""
    shape = _validate_row_slice(tensor, start, length, _NVFP4_BLOCK_SIZE, "NVFP4")
    if tensor._with_gemm_swizzled_scales and start % _SCALE_ROW_ALIGNMENT != 0:  # noqa: SLF001
        raise ValueError(f"GEMM-swizzled NVFP4 row slices must start on a {_SCALE_ROW_ALIGNMENT}-row boundary, got {start}")

    rowwise_data = tensor._rowwise_data  # noqa: SLF001
    rowwise_scale_inv = tensor._rowwise_scale_inv  # noqa: SLF001
    if rowwise_data is None or rowwise_scale_inv is None:
        raise ValueError("NVFP4 rowwise data and scales are required for checkpointing")

    metadata = tensor.get_metadata()
    metadata["rowwise_data"] = rowwise_data.narrow(0, start, length)
    metadata["rowwise_scale_inv"] = rowwise_scale_inv.narrow(0, start, _round_up(length, _SCALE_ROW_ALIGNMENT))
    if metadata.get("row_scaled_nvfp4", False):
        rowwise_amax = tensor._amax_rowwise  # noqa: SLF001
        if rowwise_amax is None:
            raise ValueError("Row-scaled NVFP4 checkpoint tensors require rowwise amax values")
        metadata["amax_rowwise"] = rowwise_amax.narrow(0, start, length)
    metadata["columnwise_data"] = None
    metadata["columnwise_scale_inv"] = None
    metadata["amax_columnwise"] = None
    metadata["quantizer"] = None
    return type(tensor)(**metadata, shape=(length, shape[1]), dtype=tensor.dtype)


def _narrow_sharded_tensor(
    sharded_tensor: object,
    start: int,
    length: int,
    slice_rows: Callable[[object, int, int], object],
) -> list[object]:
    """Narrow a checkpoint shard without dispatching an operation on its packed tensor."""
    local_length = sharded_tensor.local_shape[0]
    prepended_dim = sharded_tensor.prepend_axis_num
    global_shape = list(sharded_tensor.global_shape)
    global_offset = list(sharded_tensor.global_offset)
    if global_shape[prepended_dim] % local_length != 0:
        raise ValueError(f"Quantized checkpoint shard has an irregular global shape: {sharded_tensor}")
    if global_offset[prepended_dim] % local_length != 0:
        raise ValueError(f"Quantized checkpoint shard has an irregular global offset: {sharded_tensor}")

    data = slice_rows(sharded_tensor.data, start, length)
    global_shape[prepended_dim] = global_shape[prepended_dim] * length // local_length
    global_offset[prepended_dim] = global_offset[prepended_dim] * length // local_length
    return [
        replace(
            sharded_tensor,
            data=data,
            local_shape=tuple(data.shape),
            global_shape=tuple(global_shape),
            global_offset=tuple(global_offset),
        ),
    ]


def _quantized_sharded_narrow(original_narrow: Callable) -> Callable:
    """Wrap Megatron shard narrowing without dispatching on packed tensors."""

    @wraps(original_narrow)
    def keep_quantized_narrow(self, dim, start, length):  # noqa: ANN001, ANN202
        normalized_dim = dim + len(self.local_shape) if dim < 0 else dim
        if normalized_dim == 0:
            if _is_mxfp8_tensor(self.data):
                return _narrow_sharded_tensor(self, start, length, _slice_mxfp8_rows)
            if _is_nvfp4_tensor(self.data):
                return _narrow_sharded_tensor(self, start, length, _slice_nvfp4_rows)
        return original_narrow(self, dim, start, length)

    return keep_quantized_narrow


def install_quantized_sharded_narrow() -> None:
    """Install provider-neutral packed slicing for Megatron checkpoint factories."""
    from megatron.core.dist_checkpointing import mapping as checkpoint_mapping  # type: ignore[import-not-found] # noqa: PLC0415

    sharded_tensor_class = checkpoint_mapping.ShardedTensor
    if getattr(sharded_tensor_class, _SHARDED_NARROW_MARKER, False):
        return
    sharded_tensor_class.narrow = _quantized_sharded_narrow(sharded_tensor_class.narrow)
    setattr(sharded_tensor_class, _SHARDED_NARROW_MARKER, True)


def _chunk_mxfp8_rows(tensor: object, chunks: int) -> tuple[object, ...]:
    """Implement ``torch.chunk(..., dim=0)`` without dequantizing MXFP8."""
    rows = int(tensor.shape[0])
    if chunks <= 0:
        raise ValueError(f"chunks must be greater than 0, got {chunks}")
    chunk_rows = math.ceil(rows / chunks)
    return tuple(_slice_mxfp8_rows(tensor, start, min(chunk_rows, rows - start)) for start in range(0, rows, chunk_rows))


@contextmanager
def _preserve_sharded_narrow() -> Generator[None]:
    """Patch Megatron's checkpoint-only shard narrowing for one snapshot."""
    checkpoint_mapping = None
    with suppress(ImportError):
        from megatron.core.dist_checkpointing import mapping as checkpoint_mapping  # noqa: PLC0415
    if checkpoint_mapping is None:
        yield
        return

    sharded_tensor_class = checkpoint_mapping.ShardedTensor
    if getattr(sharded_tensor_class, _SHARDED_NARROW_MARKER, False):
        yield
        return
    original_narrow = sharded_tensor_class.narrow
    sharded_tensor_class.narrow = _quantized_sharded_narrow(original_narrow)
    try:
        yield
    finally:
        sharded_tensor_class.narrow = original_narrow


@contextmanager
def preserve_quantized_tensors() -> Generator[None]:
    """Keep FP4 and FP8 parameters packed while building a native snapshot."""
    with _PATCH_LOCK:
        original_chunk = torch.chunk

        def keep_quantized_chunk(tensor, chunks, dim=0):  # noqa: ANN001, ANN202
            normalized_dim = dim + len(tensor.shape) if dim < 0 else dim
            if _is_mxfp8_tensor(tensor) and normalized_dim == 0:
                return _chunk_mxfp8_rows(tensor, chunks)
            return original_chunk(tensor, chunks, dim)

        torch.chunk = keep_quantized_chunk
        gtp_api = None
        original_dequantize = None
        captured: dict[int, tuple[object, bool, object | None]] = {}
        with suppress(ImportError):
            from megatron.core.tensor_parallel import gtp_api  # noqa: PLC0415
        if gtp_api is not None and getattr(gtp_api, "HAVE_GTP", False) and hasattr(gtp_api, "dequantize_gtp_native_fp8"):
            original_dequantize = gtp_api.dequantize_gtp_native_fp8

            def keep_quantized(param):  # noqa: ANN001, ANN202
                if not (_is_fp8_tensor(param) or _is_mxfp8_tensor(param) or _is_nvfp4_tensor(param)):
                    return original_dequantize(param)
                captured.setdefault(
                    id(param),
                    (param, hasattr(param, "_gtp_dequant_src"), getattr(param, "_gtp_dequant_src", None)),
                )
                return param

            gtp_api.dequantize_gtp_native_fp8 = keep_quantized
        try:
            with _preserve_sharded_narrow():
                yield
        finally:
            torch.chunk = original_chunk
            if gtp_api is not None and original_dequantize is not None:
                gtp_api.dequantize_gtp_native_fp8 = original_dequantize
            for param, had_backlink, backlink in captured.values():
                if had_backlink:
                    param._gtp_dequant_src = backlink  # noqa: SLF001
                elif hasattr(param, "_gtp_dequant_src"):
                    delattr(param, "_gtp_dequant_src")


__all__ = ["install_quantized_sharded_narrow", "preserve_quantized_tensors"]
