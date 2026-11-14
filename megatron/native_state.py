# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401, PLR2004

"""Decode native checkpoint values and restore destination objects."""

from __future__ import annotations

import argparse
import enum
import math
import re
import struct
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, MetadataIndex

if TYPE_CHECKING:
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.metadata import Metadata

_SCHEMA_VERSIONS = {b"NVRXNST1": 1, b"NVRXNST2": 2}
_OBJECT_MAGIC = b"NVRXOBJ1"
_SCHEMA_KEY = re.compile(r"__nvrx_native_state_rank_(\d+)__\Z")
_DICT_KEY = re.compile(r"__nvrx_native_dict_rank_\d+__\.(.+)\Z")
_SCALAR_FORMATS = {1: "<q", 2: "<Q", 3: "<d", 4: "<Q"}
_BOOTSTRAP_ENUM_TYPES = {
    "args.attention_backend": ("megatron.core.transformer.enums", "AttnBackend"),
    "args.inference_cuda_graph_scope": ("megatron.core.transformer.enums", "InferenceCudaGraphScope"),
    "args.exit_signal": ("signal", "Signals"),
}


class NativeEnum:
    """Hold an enum value whose class comes from the destination."""

    def __init__(self, value: Any) -> None:
        self.value = value


class NativeGenerator:
    """Hold the device and state of a captured generator."""

    def __init__(self, device: str, state: bytes) -> None:
        self.device = device
        self.state = state


class NativeSchema(dict[str, int]):
    """Hold storage kinds and ordered logical checkpoint descriptions."""

    def __init__(self, version: int) -> None:
        super().__init__()
        self.version = version
        self.write_items: list[tuple[str, tuple[int, ...] | None, int]] = []


class NativeRecords(dict[str, Any]):
    """Hold captured object records and per-rank logical descriptions."""

    def __init__(self) -> None:
        super().__init__()
        self.write_items: dict[int, list[tuple[str, tuple[int, ...] | None, int]]] = {}


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        self.offset = 0

    def take(self, count: int) -> memoryview:
        if count < 0 or count > len(self.data) - self.offset:
            raise ValueError("Truncated native checkpoint value")
        result = self.data[self.offset : self.offset + count]
        self.offset += count
        return result

    def number(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.number("<I")).tobytes().decode("utf-8")

    def shape(self) -> tuple[int, ...]:
        dimensions = self.number("<I")
        if dimensions > 64:
            raise ValueError("Native checkpoint array has too many dimensions")
        return tuple(self.number("<Q") for _ in range(dimensions))

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise ValueError("Trailing bytes in native checkpoint value")

    def value(self, depth: int = 0) -> Any:  # noqa: C901, PLR0911, PLR0912
        if depth > 100:
            raise ValueError("Native checkpoint nesting exceeds the supported depth")
        kind = self.number("<B")
        if kind == 0:
            return None
        if kind in _SCALAR_FORMATS:
            if kind == 4:
                value = self.number("<B")
                if value not in (0, 1):
                    raise ValueError("Invalid native boolean value")
                return bool(value)
            return self.number(_SCALAR_FORMATS[kind])
        if kind == 5:
            return self.string()
        if kind == 6:
            return self.take(self.number("<Q")).tobytes()
        if kind in (7, 8):
            values = [self.value(depth + 1) for _ in range(self.number("<I"))]
            return tuple(values) if kind == 8 else values
        if kind in (9, 10):
            values = {}
            for _ in range(self.number("<I")):
                key = self.value(depth + 1)
                if key in values:
                    raise ValueError("Duplicate native dictionary key")
                values[key] = self.value(depth + 1)
            if kind == 10:
                if any(not isinstance(key, str) for key in values):
                    raise ValueError("Native namespace attributes require string names")
                return argparse.Namespace(**values)
            return values
        if kind == 11:
            dtype = np.dtype(self.string())
            if dtype.hasobject:
                raise ValueError("Native checkpoint arrays require a non-object dtype")
            shape = self.shape()
            count = self.number("<Q")
            if count != math.prod(shape) * dtype.itemsize:
                raise ValueError("Native array byte length does not match its shape")
            return np.frombuffer(self.take(count), dtype=dtype).copy().reshape(shape)
        if kind == 12:
            return _torch_dtype(self.string())
        if kind == 13:
            return NativeEnum(self.value(depth + 1))
        if kind == 14:
            return torch.device(self.string())
        if kind == 15:
            dtype = _torch_dtype(self.string())
            shape = self.shape()
            if math.prod(shape) != 0:
                raise ValueError("Native empty tensor has a nonempty shape")
            return torch.empty(shape, dtype=dtype)
        if kind == 16:
            return NativeGenerator(self.string(), self.take(self.number("<Q")).tobytes())
        raise ValueError(f"Unsupported native checkpoint value kind: {kind}")


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise TypeError(f"Unknown native tensor dtype: {name}")
    return value


def is_native_state_key(key: str) -> bool:
    """Identify the rank-local native state schemas."""
    return _SCHEMA_KEY.fullmatch(key) is not None


def is_native_state(metadata: Metadata) -> bool:
    """Identify checkpoint metadata containing native value schemas."""
    return any(map(is_native_state_key, metadata.state_dict_metadata))


def native_object_key(key: str, kind: int) -> str:
    """Return the logical address of an object record."""
    if kind == 20:
        match = _DICT_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Invalid rank-local dictionary address: {key}")
        return match.group(1)
    return key


def decode_native_schema(data: bytes) -> NativeSchema:
    """Read the native storage kinds for one rank's object entries."""
    reader = _Reader(data)
    version = _SCHEMA_VERSIONS.get(reader.take(8).tobytes())
    if version is None:
        raise ValueError("Unsupported native state schema")
    entries = NativeSchema(version)
    for _ in range(reader.number("<I")):
        key = reader.string()
        kind = reader.number("<B")
        if not key or key in entries or kind not in (*_SCALAR_FORMATS, 10, 20):
            raise ValueError("Invalid native state schema entry")
        entries[key] = kind
    if version == 2:
        for _ in range(reader.number("<I")):
            key = reader.string()
            has_offsets = reader.number("<B")
            if not key or has_offsets not in (0, 1):
                raise ValueError("Invalid native logical checkpoint description")
            offsets = reader.shape() if has_offsets else None
            cost = reader.number("<Q")
            if cost == 0:
                raise ValueError("Native logical checkpoint costs are positive")
            entries.write_items.append((key, offsets, cost))
    reader.finish()
    return entries


def decode_native_value(kind: int, data: bytes) -> Any:
    """Read a raw scalar or a native object payload."""
    if kind in _SCALAR_FORMATS:
        if len(data) != 8:
            raise ValueError("Native scalar payloads contain exactly eight bytes")
        value = struct.unpack(_SCALAR_FORMATS[kind], data)[0]
        if kind == 4:
            if value not in (0, 1):
                raise ValueError("Invalid native boolean value")
            return bool(value)
        return value
    if kind not in (10, 20):
        raise ValueError(f"Unsupported native storage kind: {kind}")
    reader = _Reader(data)
    if reader.take(len(_OBJECT_MAGIC)) != _OBJECT_MAGIC:
        raise ValueError("Unsupported native object payload")
    result = reader.value()
    reader.finish()
    return result


def _read_entry(reader: FileSystemReader, metadata: Metadata, key: str) -> bytes:
    storage = metadata.storage_data[MetadataIndex(key)]
    with (Path(reader.path) / storage.relative_path).open("rb") as stream:
        stream.seek(storage.offset)
        data = stream.read(storage.length)
    if len(data) != storage.length:
        raise ValueError(f"Truncated native checkpoint entry: {key}")
    return data


def read_native_state(
    reader: FileSystemReader,
    metadata: Metadata,
    *,
    decode_objects: bool = True,
) -> dict[str, Any]:
    """Read native non-tensor records using the rank-local schemas."""
    kinds = {}
    schemas = {}
    for key in metadata.state_dict_metadata:
        if not is_native_state_key(key):
            continue
        rank = int(_SCHEMA_KEY.fullmatch(key).group(1))
        schema = decode_native_schema(_read_entry(reader, metadata, key))
        schemas[rank] = schema
        for name, kind in schema.items():
            if name in kinds and kinds[name] != kind:
                raise ValueError(f"Conflicting native storage kinds for: {name}")
            kinds[name] = kind
    if len({schema.version for schema in schemas.values()}) > 1:
        raise ValueError("Native checkpoint schemas use different versions")
    state = NativeRecords()
    state.write_items = {rank: schema.write_items for rank, schema in schemas.items() if schema.version == 2}
    for key, storage in metadata.state_dict_metadata.items():
        if not isinstance(storage, BytesStorageMetadata) or is_native_state_key(key):
            continue
        if MetadataIndex(key) not in metadata.storage_data:
            continue
        if key not in kinds:
            raise ValueError(f"Missing native storage schema for: {key}")
        kind, data = kinds[key], _read_entry(reader, metadata, key)
        state[key] = (kind, data)
    return decode_native_records(state) if decode_objects else state


def _same_native_value(left: Any, right: Any) -> bool:  # noqa: PLR0911
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        equal = np.array_equal(left, right, equal_nan=True) if left.dtype.kind in "fc" else np.array_equal(left, right)
        return left.dtype == right.dtype and left.shape == right.shape and equal
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return True
    if isinstance(left, (argparse.Namespace, SimpleNamespace)):
        return _same_native_value(vars(left), vars(right))
    if isinstance(left, NativeEnum):
        return _same_native_value(left.value, right.value)
    if isinstance(left, NativeGenerator):
        return left.device == right.device and left.state == right.state
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_native_value(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_native_value(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _merge_native_dictionary(target: dict[Any, Any], incoming: dict[Any, Any], path: str) -> None:
    for key, value in incoming.items():
        child = f"{path}.{key}" if path else str(key)
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_native_dictionary(target[key], value, child)
        elif not _same_native_value(target[key], value):
            raise ValueError(f"Conflicting replicated dictionary value: {child}")


def native_source_ranks(
    plans: dict[int, list[tuple[str, tuple[int, ...] | None, int]]],
) -> dict[tuple[str, tuple[int, ...] | None], int]:
    """Select logical sources using DCP's ordered storage-cost policy."""
    holders = {}
    costs = {}
    assigned = dict.fromkeys(plans, 0)
    for rank in sorted(plans):
        for key, offsets, cost in plans[rank]:
            index = (key, offsets)
            holders.setdefault(index, set()).add(rank)
            costs[index] = cost
    selected = {}
    for index, ranks in holders.items():
        rank = min(ranks, key=assigned.__getitem__)
        selected[index] = rank
        assigned[rank] += costs[index]
    return selected


def _terminal_native_value(value: Any) -> bool:
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return all(_terminal_native_value(item) and not isinstance(item, torch.Tensor) for item in value)
    return True


def _merge_dictionary_sources(
    candidates: dict[int, Any],
    path: str,
    sources: dict[tuple[str, tuple[int, ...] | None], int],
) -> Any:
    first = next(iter(candidates.values()))
    index = (path, tuple(0 for _ in first.shape) if isinstance(first, torch.Tensor) else None)
    if index in sources:
        rank = sources[index]
        if rank not in candidates:
            raise ValueError(f"Missing selected native dictionary source for: {path}")
        return candidates[rank]
    if all(isinstance(value, dict) for value in candidates.values()):
        keys = dict.fromkeys(key for value in candidates.values() for key in value)
        return {
            key: _merge_dictionary_sources(
                {rank: value[key] for rank, value in candidates.items() if key in value},
                f"{path}.{key}",
                sources,
            )
            for key in keys
        }
    if all(isinstance(value, list) and not _terminal_native_value(value) for value in candidates.values()):
        return [
            _merge_dictionary_sources(
                {rank: value[index] for rank, value in candidates.items() if index < len(value)},
                f"{path}.{index}",
                sources,
            )
            for index in range(max(map(len, candidates.values())))
        ]
    if all(_same_native_value(first, value) for value in candidates.values()):
        return first
    raise ValueError(f"Missing logical source description for conflicting dictionary value: {path}")


def decode_native_records(records: dict[str, Any]) -> dict[str, Any]:
    """Decode records and merge dictionaries captured by their local owners."""
    objects = {}
    dictionaries = {}
    for key, (kind, data) in records.items():
        value = decode_native_value(kind, data)
        logical_key = native_object_key(key, kind)
        if kind == 20:
            if not isinstance(value, dict):
                raise TypeError("Rank-local native dictionary records contain dictionaries")
            rank = int(key.partition("__.")[0].removeprefix("__nvrx_native_dict_rank_"))
            dictionaries.setdefault(logical_key, {})[rank] = value
        elif logical_key in objects:
            raise ValueError(f"Duplicate native checkpoint object: {logical_key}")
        else:
            objects[logical_key] = value
    plans = getattr(records, "write_items", {})
    sources = native_source_ranks(plans)
    for key, candidates in dictionaries.items():
        if key in objects:
            raise ValueError(f"Conflicting native object records: {key}")
        if plans:
            objects[key] = _merge_dictionary_sources(candidates, key, sources)
        else:
            merged = {}
            for value in candidates.values():
                _merge_native_dictionary(merged, value, key)
            objects[key] = merged
    return objects


def materialize_native_value(value: Any) -> Any:  # noqa: PLR0911
    """Construct built-in Python values for checkpoint inspection."""
    if isinstance(value, NativeEnum):
        return materialize_native_value(value.value)
    if isinstance(value, NativeGenerator):
        return restore_native_value(value, None)
    if isinstance(value, (argparse.Namespace, SimpleNamespace)):
        return argparse.Namespace(**{key: materialize_native_value(item) for key, item in vars(value).items()})
    if isinstance(value, dict):
        return {key: materialize_native_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(map(materialize_native_value, value))
    if isinstance(value, tuple):
        return tuple(map(materialize_native_value, value))
    return value


def restore_native_value(value: Any, target: Any, path: str = "") -> Any:  # noqa: C901, PLR0911, PLR0912
    """Apply decoded values while retaining mutable destination objects."""
    if isinstance(value, NativeEnum):
        enum_type = type(target) if isinstance(target, enum.Enum) else None
        if enum_type is None and (schema := _BOOTSTRAP_ENUM_TYPES.get(path)) is not None:
            module, name = schema
            enum_type = getattr(import_module(module), name)
        if enum_type is None:
            raise TypeError(f"Missing destination enum type for: {path}")
        return enum_type(value.value)
    if isinstance(value, NativeGenerator):
        generator = target if isinstance(target, torch.Generator) else torch.Generator(device=value.device)
        generator.set_state(torch.frombuffer(bytearray(value.state), dtype=torch.uint8))
        return generator
    if isinstance(target, np.generic) and isinstance(value, (bool, int, float, str, bytes, tuple)):
        if target.dtype.kind in "SU":
            return type(target)(value)
        scalar = np.asarray(value, dtype=target.dtype)
        if scalar.ndim != 0:
            raise ValueError(f"Native value does not match the NumPy scalar destination at: {path}")
        return scalar[()]
    if isinstance(value, (argparse.Namespace, SimpleNamespace)):
        result = target if isinstance(target, (argparse.Namespace, SimpleNamespace)) else argparse.Namespace()
        _restore_mapping(vars(value), vars(result), path)
        return result
    if isinstance(value, dict):
        result = target if isinstance(target, dict) else {}
        _restore_mapping(value, result, path)
        return result
    if isinstance(value, list):
        result = target if isinstance(target, list) else []
        original = list(result)
        result[:] = [
            restore_native_value(item, original[index] if index < len(original) else None, f"{path}.{index}")
            for index, item in enumerate(value)
        ]
        return result
    if isinstance(value, tuple):
        old = target if isinstance(target, tuple) else ()
        values = [
            restore_native_value(item, old[index] if index < len(old) else None, f"{path}.{index}") for index, item in enumerate(value)
        ]
        if isinstance(target, tuple) and hasattr(target, "_fields"):
            return type(target)(*values)
        return type(target)(values) if isinstance(target, tuple) else tuple(values)
    if isinstance(value, np.ndarray):
        if isinstance(target, np.ndarray) and target.shape == value.shape and target.dtype == value.dtype:
            np.copyto(target, value)
            return target
        return value
    if isinstance(value, torch.Tensor):
        if isinstance(target, torch.Tensor):
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"Native tensor does not match the destination at: {path}")
            target.detach().copy_(value)
            return target
        return value
    return value


def _restore_mapping(values: dict[Any, Any], target: dict[Any, Any], path: str) -> None:
    for key in tuple(target):
        if key not in values:
            del target[key]
    for key, value in values.items():
        target[key] = restore_native_value(value, target.get(key), f"{path}.{key}" if path else str(key))
