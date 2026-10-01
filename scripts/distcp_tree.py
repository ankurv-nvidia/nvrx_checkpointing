#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Render the state-dict tree of a distcp checkpoint as an interactive HTML page.

The tree comes from ``<checkpoint>/.metadata``: ``.`` and ``/`` in a state-dict
key separate tree levels, and a ShardedObject ``shard_<offsets>_<shape>``
suffix stays a single level. Leaves are tensors, scalars (0-dim tensors) and
blobs (non-tensor values stored as pickled bytes).

All sizes are logical byte counts. A tensor chunk counts ``numel * itemsize``
and a blob counts its stored length. Each chunk and blob belongs to the rank
in the name of the ``__<rank>_<n>.distcp`` file that holds it, so a rank's
size is what that rank's files hold; a replicated entry is stored, and
counted, once.

``--expand-blobs`` also reads each blob from its ``.distcp`` file and expands
dicts, lists and objects into subtrees whose leaves show scalar types and
values and the dtype and shape of nested tensors and arrays; a decoded blob
counts its archive length without slot padding. Decoding never imports torch,
numpy or Megatron and never executes code from the checkpoint: a restricted
unpickler substitutes an inert placeholder for every class it meets.

The output is one self-contained HTML file: a collapsible tree table with
sortable and filterable columns, all-ranks and per-rank sizes for two
selectable ranks, a per-rank size chart for the selected subtree, a treemap,
and a view that nests optimizer states under their parameters.

Usage::

    python scripts/distcp_tree.py <checkpoint_dir> [-o tree.html] [--expand-blobs]
"""

from __future__ import annotations

import argparse
import base64
import datetime
import functools
import gzip
import html
import io
import json
import math
import pickle
import re
import struct
import sys
import zipfile
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

DTYPE_ITEMSIZE: dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "complex128": 16,
    "complex64": 8,
    "complex32": 4,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint64": 8,
    "uint32": 4,
    "uint16": 2,
    "uint8": 1,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
    "qint8": 1,
    "quint8": 1,
    "qint32": 4,
    "quint4x2": 1,
    "quint2x4": 1,
    "bits8": 1,
    "bits16": 2,
    "bits1x8": 1,
    "bits2x4": 1,
    "bits4x2": 1,
}

_STORAGE_DTYPES = {
    "DoubleStorage": "float64",
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
    "ComplexDoubleStorage": "complex128",
    "ComplexFloatStorage": "complex64",
    "QInt8Storage": "qint8",
    "QUInt8Storage": "quint8",
    "QInt32Storage": "qint32",
    "QUInt4x2Storage": "quint4x2",
    "QUInt2x4Storage": "quint2x4",
}

UNKNOWN_RANK = 0xFFFFFFFF
DEFAULT_CHUNK_DETAILS_LIMIT = 500_000

_U32_BYTES = 4
_U32 = "I" if array("I").itemsize == _U32_BYTES else "L"
_RANK_FILE = re.compile(r"__(\d+)_\d+\.distcp$")
_SHARD_SUFFIX = re.compile(r"shard_[\d.]+_[\d.]+")
_OPTIMIZER_STATE = re.compile(r"(?P<prefix>(?:[^./]+\.)*?)optimizer\.state\.(?P<state>[^./]+)\.(?P<param>.+)")
_NUMPY_DTYPE = re.compile(r"[<>|=]?([a-zA-Z])(\d*)")
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_END = b"PK\x05\x06"
_ZIP_END_SIZE = 22
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_SAFE_ENCODINGS = frozenset({"latin1", "latin-1", "iso-8859-1", "iso8859-1", "utf-8", "utf8", "ascii"})
_NUMPY_KINDS = {"b": "bool", "i": "int", "u": "uint", "f": "float", "c": "complex", "O": "object", "U": "str", "S": "bytes"}
_NUMPY_FIXED_ITEMSIZE = {"b": 1, "O": 8}
_NUMPY_SCALAR_FORMATS = {
    "float64": "<d",
    "float32": "<f",
    "float16": "<e",
    "int64": "<q",
    "int32": "<i",
    "int16": "<h",
    "int8": "<b",
    "uint64": "<Q",
    "uint32": "<I",
    "uint16": "<H",
    "uint8": "<B",
    "bool": "<?",
}
_INT64_MIN = -(2**63)
_INT64_LIMIT = 2**63
_FIXED_SCALAR_SIZES = {type(None): 0, bool: 1, float: 8, complex: 16}
_SLOTTED_STATE_LEN = 2
_NDARRAY_STATE_LEN = 4
_NDARRAY_VERSIONED_STATE_LEN = 5
_MAX_CALL_ARGS = 3
_MAX_JSON_DEPTH = 8
_MAX_DETAIL_DEPTH = 12
_MAX_DETAIL_CHILDREN = 1000
_MAX_DETAIL_NODES = 5000
_MAX_VALUE_CHARS = 200


# Restricted unpickling: every global resolves to an inert placeholder.


class _Opaque:
    """Inert placeholder for an instance of a class that is never imported."""

    module = "builtins"
    qualname = "object"

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        obj = super().__new__(cls)
        obj.args = args
        obj.kwargs = kwargs
        obj.state = None
        obj.items = []
        obj.mapping = {}
        return obj

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        self.state = state

    def append(self, value: object) -> None:
        self.items.append(value)

    def extend(self, values: Iterable[object]) -> None:
        self.items.extend(values)

    def __setitem__(self, key: object, value: object) -> None:
        self.mapping[key] = value


@functools.cache
def _opaque_class(module: str, qualname: str) -> type[_Opaque]:
    return type(qualname.rsplit(".", 1)[-1] or "object", (_Opaque,), {"module": module, "qualname": qualname})


def _state_fields(state: object) -> dict:
    """Return the attribute mapping held by a pickled instance state."""
    if isinstance(state, dict):
        return state
    if (
        isinstance(state, tuple)
        and len(state) == _SLOTTED_STATE_LEN
        and isinstance(state[1], dict)
        and (state[0] is None or isinstance(state[0], dict))
    ):
        return {**(state[0] or {}), **state[1]}
    return {}


class _Record:
    """Slotted placeholder for a checkpoint metadata dataclass."""

    __slots__ = ()

    def __new__(cls, *_args: object, **_kwargs: object) -> Self:
        obj = super().__new__(cls)
        for name in cls.__slots__:
            setattr(obj, name, None)
        return obj

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        values = _state_fields(state)
        for name in self.__slots__:
            setattr(self, name, values.get(name))


class _MetadataIndex(_Record):
    __slots__ = ("fqn", "offset")


class _StorageInfo(_Record):
    __slots__ = ("length", "offset", "relative_path", "transform_descriptors")

    def __setstate__(self, state: object) -> None:
        super().__setstate__(state)
        if isinstance(self.relative_path, str):
            self.relative_path = sys.intern(self.relative_path)


class _ChunkStorageMetadata(_Record):
    __slots__ = ("offsets", "sizes")


class _TensorStorageMetadata(_Record):
    __slots__ = ("chunks", "properties", "size")


class _BytesStorageMetadata(_Record):
    __slots__ = ()


class _DType:
    """Placeholder for a torch dtype."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _TensorProperties(_Record):
    __slots__ = ("dtype",)

    def __setstate__(self, state: object) -> None:
        # The tuple form starts with the dtype.
        if isinstance(state, tuple) and state and isinstance(state[0], _DType):
            self.dtype = state[0]
        else:
            super().__setstate__(state)


_DCP_RECORDS: dict[str, type[_Record]] = {
    "MetadataIndex": _MetadataIndex,
    "_StorageInfo": _StorageInfo,
    "ChunkStorageMetadata": _ChunkStorageMetadata,
    "TensorStorageMetadata": _TensorStorageMetadata,
    "BytesStorageMetadata": _BytesStorageMetadata,
    "TensorProperties": _TensorProperties,
}


class _StorageType:
    __slots__ = ("dtype",)

    def __init__(self, dtype: _DType | None) -> None:
        self.dtype = dtype


class _StorageRef:
    __slots__ = ("dtype",)

    def __init__(self, dtype: object) -> None:
        self.dtype = dtype


class _Tensor:
    """Placeholder for a tensor serialized inside a blob."""

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype: object, shape: object) -> None:
        self.dtype = dtype.name if isinstance(dtype, _DType) else None
        dims = shape if isinstance(shape, (tuple, list)) else ()
        self.shape = tuple(dims) if all(isinstance(dim, int) for dim in dims) else None

    @property
    def nbytes(self) -> int:
        if self.shape is None:
            return 0
        return math.prod(self.shape) * DTYPE_ITEMSIZE.get(self.dtype or "", 0)


def _arg(args: tuple[object, ...], index: int, default: object = None) -> object:
    return args[index] if len(args) > index else default


def _rebuild_tensor_v2(*args: object) -> _Tensor:
    return _Tensor(getattr(_arg(args, 0), "dtype", None), _arg(args, 2, ()))


def _rebuild_tensor_v3(*args: object) -> _Tensor:
    return _Tensor(_arg(args, 6), _arg(args, 2, ()))


def _rebuild_meta_tensor(*args: object) -> _Tensor:
    return _Tensor(_arg(args, 0), _arg(args, 1, ()))


def _rebuild_wrapper_subclass(*args: object) -> _Tensor:
    return _Tensor(_arg(args, 1), _arg(args, 2, ()))


def _first_arg(*args: object) -> object:
    return _arg(args, 0)


def _rebuild_from_type(*args: object) -> object:
    func, call_args = _arg(args, 0), _arg(args, 2, ())
    if callable(func) and isinstance(call_args, tuple):
        return func(*call_args)
    return _opaque_class("torch._tensor", "_rebuild_from_type_v2")(*args)


def _numpy_dtype(code: object) -> tuple[str, int]:
    match = _NUMPY_DTYPE.fullmatch(code) if isinstance(code, str) else None
    if match is None or match.group(1) not in _NUMPY_KINDS:
        return str(code), 0
    kind, digits = match.group(1), match.group(2)
    width = _NUMPY_FIXED_ITEMSIZE.get(kind, int(digits) if digits else 0)
    if kind in "iufc":
        return f"{_NUMPY_KINDS[kind]}{width * 8}", width
    return _NUMPY_KINDS[kind], (4 * width if kind == "U" else width)


class _NumpyDType:
    __slots__ = ("itemsize", "name")

    def __init__(self, code: object = "V", *_args: object) -> None:
        self.name, self.itemsize = _numpy_dtype(code)

    def __setstate__(self, _state: object) -> None:
        pass


class _NumpyArray:
    __slots__ = ("dtype", "nbytes", "shape")

    def __init__(self, *_args: object) -> None:
        self.dtype = None
        self.shape = None
        self.nbytes = 0

    def __setstate__(self, state: object) -> None:
        if not isinstance(state, tuple) or len(state) < _NDARRAY_STATE_LEN:
            return
        versioned = len(state) >= _NDARRAY_VERSIONED_STATE_LEN
        shape, dtype, raw = (state[1], state[2], state[4]) if versioned else (state[0], state[1], state[3])
        self._describe(shape, dtype, raw)

    def _describe(self, shape: object, dtype: object, raw: object) -> None:
        self.shape = tuple(shape) if isinstance(shape, tuple) and all(isinstance(dim, int) for dim in shape) else None
        self.dtype = dtype.name if isinstance(dtype, _NumpyDType) else None
        itemsize = dtype.itemsize if isinstance(dtype, _NumpyDType) else 0
        self.nbytes = len(raw) if isinstance(raw, (bytes, bytearray)) else math.prod(self.shape or ()) * itemsize


def _numpy_frombuffer(*args: object) -> _NumpyArray:
    array_ = _NumpyArray()
    array_._describe(_arg(args, 2), _arg(args, 1), _arg(args, 0))  # noqa: SLF001
    return array_


class _NumpyScalar:
    __slots__ = ("dtype", "itemsize", "value")

    def __init__(self, dtype: object = None, raw: object = b"", *_args: object) -> None:
        self.dtype = dtype.name if isinstance(dtype, _NumpyDType) else "?"
        self.itemsize = dtype.itemsize if isinstance(dtype, _NumpyDType) else 0
        layout = _NUMPY_SCALAR_FORMATS.get(self.dtype)
        if layout is not None and isinstance(raw, bytes) and len(raw) == struct.calcsize(layout):
            self.value = struct.unpack(layout, raw)[0]
        else:
            self.value = raw


class _BytesIO:
    __slots__ = ("data",)

    def __new__(cls, *args: object) -> Self:
        obj = super().__new__(cls)
        first = _arg(args, 0)
        obj.data = bytes(first) if isinstance(first, (bytes, bytearray)) else b""
        return obj

    def __init__(self, *_args: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple) and state and isinstance(state[0], (bytes, bytearray)):
            self.data = bytes(state[0])


class _RawPayload:
    __slots__ = ("length",)

    def __init__(self, length: int) -> None:
        self.length = length


def _reconstructor(*args: object) -> object:
    cls = _arg(args, 0)
    if isinstance(cls, type) and issubclass(cls, (_Opaque, _Record, _BytesIO)):
        return cls()
    return _opaque_class("copyreg", "_reconstructor")(*args)


def _codecs_encode(*args: object) -> object:
    text, encoding = _arg(args, 0, ""), _arg(args, 1, "utf-8")
    if isinstance(text, str) and isinstance(encoding, str) and encoding.lower() in _SAFE_ENCODINGS:
        return text.encode(encoding)
    return _opaque_class("_codecs", "encode")(*args)


def _empty_dict(*_args: object) -> dict:
    return {}


def _as_list(*args: object) -> list:
    first = _arg(args, 0, ())
    return list(first) if isinstance(first, (list, tuple)) else []


_BUILTINS: dict[str, object] = {
    "set": set,
    "frozenset": frozenset,
    "complex": complex,
    "slice": slice,
    "bytearray": bytearray,
    "bytes": bytes,
    "range": range,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

_GLOBALS: dict[tuple[str, str], object] = {
    ("collections", "OrderedDict"): dict,
    ("collections", "Counter"): dict,
    ("collections", "defaultdict"): _empty_dict,
    ("collections", "deque"): _as_list,
    ("copyreg", "_reconstructor"): _reconstructor,
    ("_codecs", "encode"): _codecs_encode,
    ("_io", "BytesIO"): _BytesIO,
    ("io", "BytesIO"): _BytesIO,
    ("torch", "Size"): tuple,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_tensor_v3"): _rebuild_tensor_v3,
    ("torch._utils", "_rebuild_qtensor"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _first_arg,
    ("torch._utils", "_rebuild_parameter_with_state"): _first_arg,
    ("torch._utils", "_rebuild_meta_tensor_no_storage"): _rebuild_meta_tensor,
    ("torch._utils", "_rebuild_wrapper_subclass"): _rebuild_wrapper_subclass,
    ("torch._tensor", "_rebuild_from_type"): _rebuild_from_type,
    ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type,
    ("torch.serialization", "_get_layout"): _first_arg,
    ("numpy", "dtype"): _NumpyDType,
}
for _numpy_module in ("numpy.core.multiarray", "numpy._core.multiarray"):
    _GLOBALS[(_numpy_module, "_reconstruct")] = _NumpyArray
    _GLOBALS[(_numpy_module, "scalar")] = _NumpyScalar
for _numpy_module in ("numpy.core.numeric", "numpy._core.numeric"):
    _GLOBALS[(_numpy_module, "_frombuffer")] = _numpy_frombuffer


@functools.cache
def _resolve_global(module: str, name: str) -> object:
    known = _BUILTINS.get(name) if module == "builtins" else _GLOBALS.get((module, name))
    if known is not None:
        return known
    if module == "torch" and name in DTYPE_ITEMSIZE:
        return _DType(name)
    if module in ("torch", "torch.storage") and (name in _STORAGE_DTYPES or name in ("UntypedStorage", "TypedStorage")):
        dtype = _STORAGE_DTYPES.get(name)
        return _StorageType(_resolve_global("torch", dtype) if dtype else None)
    if "checkpoint" in module and name in _DCP_RECORDS:
        return _DCP_RECORDS[name]
    return _opaque_class(module, name)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that maps every global to a placeholder and never imports a module."""

    def find_class(self, module: str, name: str) -> object:
        return _resolve_global(module, name)

    def persistent_load(self, pid: object) -> object:
        if isinstance(pid, tuple) and len(pid) > 1 and pid[0] == "storage":
            storage_type = pid[1]
            return _StorageRef(storage_type.dtype if isinstance(storage_type, _StorageType) else None)
        return _opaque_class("pickle", "PersistentId")(pid)


def _unpickle(stream: IO[bytes]) -> object:
    return _RestrictedUnpickler(stream).load()


# Checkpoint metadata.


def split_fqn(fqn: str) -> list[str]:
    """Split a state-dict key into tree levels."""
    segments: list[str] = []
    for part in fqn.split("/"):
        if _SHARD_SUFFIX.fullmatch(part):
            segments.append(part)
        elif part:
            segments.extend(_split_dotted(part))
    return segments or [fqn]


def _split_dotted(text: str) -> list[str]:
    tokens: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "." and depth == 0:
            tokens.append(text[start:index])
            start = index + 1
    tokens.append(text[start:])
    merged: list[str] = []
    for token in tokens:
        # A stringified torch dtype stays inside its segment.
        if merged and token in DTYPE_ITEMSIZE and (merged[-1] == "torch" or merged[-1].endswith("_torch")):
            merged[-1] = f"{merged[-1]}.{token}"
        elif token:
            merged.append(token)
    return merged


def optimizer_paths(fqns: list[str]) -> dict[int, list[str]]:
    """Map keys to tree levels that nest each optimizer state under its parameter."""
    index = {fqn: position for position, fqn in enumerate(fqns)}
    paths: dict[int, list[str]] = {}
    params: set[int] = set()
    for position, fqn in enumerate(fqns):
        match = _OPTIMIZER_STATE.fullmatch(fqn)
        if match is None:
            continue
        paths[position] = [*split_fqn(match["param"]), f"{match['prefix']}{match['state']}"]
        param = index.get(match["param"])
        if param is not None:
            params.add(param)
    for param in params:
        paths[param] = [*split_fqn(fqns[param]), "(param)"]
    return paths


@functools.cache
def _file_rank(relative_path: str) -> int:
    match = _RANK_FILE.search(relative_path)
    return int(match.group(1)) if match else UNKNOWN_RANK


def _shape_text(dims: tuple[int, ...]) -> str:
    return f"[{', '.join(str(dim) for dim in dims)}]"


def _layout_text(count: int, sizes: tuple[int, ...] | None, uniform: bool) -> str:
    if sizes is None or count == 0:
        return ""
    if count == 1:
        return _shape_text(sizes)
    return f"{count} \u00d7 {_shape_text(sizes)}" if uniform else f"{count} chunks"


@dataclass
class Leaf:
    """One state-dict entry with its per-rank logical sizes."""

    fqn: str
    kind: str
    nbytes: int = 0
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    ranks: dict[int, list] = field(default_factory=dict)
    chunks: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    location: tuple[str, int, int] | None = None
    transform: object = None
    blob: dict[str, object] | None = None
    extra: object = None
    unique: int | None = None
    storage: str | None = None
    rank_labels: dict[int, str] = field(default_factory=dict)

    def add(self, rank: int, nbytes: int, sizes: tuple[int, ...] | None) -> None:
        self.nbytes += nbytes
        slot = self.ranks.get(rank)
        if slot is None:
            self.ranks[rank] = [nbytes, 1, sizes, True]
        else:
            slot[0] += nbytes
            slot[1] += 1
            slot[3] = slot[3] and slot[2] == sizes

    def chunk_count(self) -> int:
        return sum(slot[1] for slot in self.ranks.values())

    def layout(self) -> str:
        slots = [slot for slot in self.ranks.values() if slot[2] is not None]
        if not slots:
            return ""
        first = slots[0][2]
        return _layout_text(sum(slot[1] for slot in slots), first, all(slot[3] and slot[2] == first for slot in slots))


@dataclass
class Report:
    """Everything the page shows about one checkpoint."""

    source: Path
    leaves: list[Leaf]
    ranks: int
    files: int
    metadata_bytes: int
    expanded: bool
    warnings: list[str]
    input_label: str = ".metadata"
    notes: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    ranks_per_host: int = 0


def _items(mapping: object) -> Iterator[tuple[object, object]]:
    if isinstance(mapping, dict):
        yield from mapping.items()
    elif isinstance(mapping, _Opaque):
        yield from mapping.mapping.items()


def _object_fields(obj: object) -> dict:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, _Opaque):
        return {**_state_fields(obj.state), **{key: value for key, value in obj.mapping.items() if isinstance(key, str)}}
    return {}


def _jsonable(value: object, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth < _MAX_JSON_DEPTH and isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if depth < _MAX_JSON_DEPTH and isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    return repr(value) if isinstance(value, _DType) else type(value).__name__


def _take(located: dict, fqn: str, offsets: tuple[int, ...] | None) -> _StorageInfo | None:
    info = located.pop((fqn, offsets), None)
    if info is None and not offsets:
        info = located.pop((fqn, () if offsets is None else None), None)
    return info


def _tensor_leaf(fqn: str, meta: _TensorStorageMetadata, located: dict, problems: Counter) -> Leaf:
    dtype = getattr(meta.properties, "dtype", None)
    name = dtype.name if isinstance(dtype, _DType) else str(dtype)
    itemsize = DTYPE_ITEMSIZE.get(name)
    if itemsize is None:
        problems[f"tensors with unknown dtype {name} (counted as 0 bytes)"] += 1
        itemsize = 0
    shape = tuple(meta.size) if isinstance(meta.size, (tuple, list)) else ()
    leaf = Leaf(fqn, "tensor" if shape else "scalar", dtype=name, shape=shape)
    for chunk in meta.chunks or ():
        offsets = tuple(chunk.offsets or ())
        sizes = tuple(chunk.sizes or ())
        info = _take(located, fqn, offsets)
        if info is None:
            problems["tensor chunks without a storage entry"] += 1
            rank = UNKNOWN_RANK
        else:
            rank = _file_rank(info.relative_path)
        leaf.add(rank, math.prod(sizes) * itemsize, sizes)
        leaf.chunks.append((rank, offsets, sizes))
    return leaf


def _blob_leaf(fqn: str, located: dict, problems: Counter) -> Leaf:
    leaf = Leaf(fqn, "blob")
    info = _take(located, fqn, None)
    if info is None:
        problems["blobs without a storage entry"] += 1
        leaf.add(UNKNOWN_RANK, 0, None)
        return leaf
    length = int(info.length or 0)
    leaf.location = (info.relative_path, int(info.offset or 0), length)
    leaf.transform = info.transform_descriptors
    leaf.add(_file_rank(info.relative_path), length, None)
    return leaf


def _collect_leaves(fields: dict, problems: Counter) -> tuple[list[Leaf], int]:
    located: dict[tuple[str, tuple[int, ...] | None], _StorageInfo] = {}
    for index, info in _items(fields.get("storage_data")):
        if isinstance(index, _MetadataIndex) and isinstance(info, _StorageInfo) and isinstance(index.fqn, str):
            offset = None if index.offset is None else tuple(index.offset)
            located[(index.fqn, offset)] = info
        else:
            problems["storage entries of an unknown type"] += 1
    files = {info.relative_path for info in located.values() if isinstance(info.relative_path, str)}
    problems["storage entries in files without a __<rank>_<n>.distcp name"] += sum(
        1 for info in located.values() if not isinstance(info.relative_path, str) or _file_rank(info.relative_path) == UNKNOWN_RANK
    )
    extras = fields.get("mcore_data") if isinstance(fields.get("mcore_data"), dict) else {}
    leaves: list[Leaf] = []
    for fqn, meta in _items(fields.get("state_dict_metadata")):
        if isinstance(meta, _TensorStorageMetadata):
            leaf = _tensor_leaf(str(fqn), meta, located, problems)
        elif isinstance(meta, _BytesStorageMetadata):
            leaf = _blob_leaf(str(fqn), located, problems)
        else:
            problems["state-dict entries of an unknown metadata type"] += 1
            continue
        if fqn in extras:
            leaf.extra = _jsonable(extras[fqn])
        leaves.append(leaf)
    problems["storage entries that match no state-dict key"] += len(located)
    return leaves, len(files)


# Blob decoding.


def _zip_length(raw: bytes) -> int:
    end = raw.rfind(_ZIP_END)
    if end < 0 or end + _ZIP_END_SIZE > len(raw):
        return len(raw)
    comment = struct.unpack_from("<H", raw, end + _ZIP_END_SIZE - 2)[0]
    return min(len(raw), end + _ZIP_END_SIZE + comment)


def _load_zip_archive(raw: bytes) -> object:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name == "data.pkl" or name.endswith("/data.pkl")]
        if not names:
            raise ValueError("archive holds no data.pkl")
        payload = archive.read(min(names, key=len))
    return _unpickle(io.BytesIO(payload))


def _load_pickle_stream(raw: bytes) -> object:
    stream = io.BytesIO(raw)
    first = _unpickle(stream)
    if first != _LEGACY_MAGIC:
        return first
    _unpickle(stream)
    _unpickle(stream)
    return _unpickle(stream)


def _decode_payload(raw: bytes) -> tuple[int, object]:
    """Return the archive length and the decoded value of stored blob bytes."""
    if raw.startswith(_ZIP_MAGIC):
        length = _zip_length(raw)
        return length, _load_zip_archive(raw[:length])
    if raw.startswith(b"\x80"):
        return len(raw), _load_pickle_stream(raw)
    return len(raw), _RawPayload(len(raw))


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_VALUE_CHARS else f"{text[: _MAX_VALUE_CHARS - 1]}…"


def _int_size(value: int) -> int:
    return 8 if _INT64_MIN <= value < _INT64_LIMIT else (value.bit_length() + 8) // 8


def _scalar_node(value: object) -> dict[str, object] | None:
    node = _python_scalar(value)
    return node if node is not None else _marker_scalar(value)


def _python_scalar(value: object) -> dict[str, object] | None:
    size = _FIXED_SCALAR_SIZES.get(type(value))
    if size is not None:
        return {"k": "scalar", "t": type(value).__name__, "b": size, "v": repr(value)}
    if isinstance(value, int):
        return {"k": "scalar", "t": "int", "b": _int_size(value), "v": _clip(repr(value))}
    if isinstance(value, str):
        return {"k": "scalar", "t": "str", "b": len(value.encode("utf-8", "surrogatepass")), "v": _clip(repr(value))}
    if isinstance(value, (bytes, bytearray)):
        return {"k": "scalar", "t": type(value).__name__, "b": len(value), "v": _clip(repr(bytes(value[:64])))}
    return None


def _marker_scalar(value: object) -> dict[str, object] | None:
    if isinstance(value, _DType):
        return {"k": "scalar", "t": "dtype", "b": 0, "v": repr(value)}
    if isinstance(value, _NumpyScalar):
        return {"k": "scalar", "t": f"numpy.{value.dtype}", "b": value.itemsize, "v": _clip(repr(value.value))}
    if isinstance(value, _NumpyDType):
        return {"k": "scalar", "t": "numpy.dtype", "b": 0, "v": value.name}
    if isinstance(value, type):
        name = f"{value.module}.{value.qualname}" if issubclass(value, _Opaque) else value.__name__
        return {"k": "scalar", "t": "type", "b": 0, "v": name}
    if isinstance(value, (slice, range)):
        return {"k": "scalar", "t": type(value).__name__, "b": 0, "v": _clip(repr(value))}
    return None


def _call_text(obj: _Opaque) -> str | None:
    """Render an argument-only object such as an enum member as a call."""
    if obj.state is not None or obj.items or obj.mapping or obj.kwargs or len(obj.args) > _MAX_CALL_ARGS:
        return None
    if not all(isinstance(arg, (bool, int, float, str, type(None))) for arg in obj.args):
        return None
    return _clip(f"{obj.qualname.rsplit('.', 1)[-1]}({', '.join(repr(arg) for arg in obj.args)})")


def _key_text(key: object) -> str:
    if isinstance(key, str):
        return key
    node = _scalar_node(key)
    return str(node["v"]) if node is not None else type(key).__name__


class _Describer:
    """Convert a decoded blob value into nested detail nodes for the page."""

    def __init__(self) -> None:
        self.remaining = _MAX_DETAIL_NODES
        self.active: set[int] = set()

    def node(self, name: str, value: object, depth: int) -> dict[str, object]:
        self.remaining -= 1
        leaf = _scalar_node(value) or _payload_node(value)
        if leaf is not None:
            return {"n": name, **leaf}
        if isinstance(value, _BytesIO):
            return self._bytes_io(name, value, depth)
        container = _container(value)
        if container is None:
            return {"n": name, "k": "object", "t": type(value).__name__, "b": 0}
        return self._container_node(name, value, container, depth)

    def _container_node(
        self,
        name: str,
        value: object,
        container: tuple[str, str, list[tuple[str, object]]],
        depth: int,
    ) -> dict[str, object]:
        kind, label, entries = container
        if id(value) in self.active:
            return {"n": name, "k": "ref", "t": label, "b": 0, "v": "(cycle)"}
        node: dict[str, object] = {"n": name, "k": kind, "t": label, "l": len(entries), "b": 0}
        if isinstance(value, _Opaque) and value.qualname != label:
            node["q"] = f"{value.module}.{value.qualname}"
        if depth >= _MAX_DETAIL_DEPTH:
            node["m"] = len(entries)
            return node
        self.active.add(id(value))
        children = []
        for child_name, child in entries:
            if self.remaining <= 0 or len(children) >= _MAX_DETAIL_CHILDREN:
                break
            children.append(self.node(child_name, child, depth + 1))
        self.active.discard(id(value))
        node["b"] = sum(int(child.get("b") or 0) for child in children)
        if children:
            node["c"] = children
        if len(children) < len(entries):
            node["m"] = len(entries) - len(children)
        return node

    def _bytes_io(self, name: str, value: _BytesIO, depth: int) -> dict[str, object]:
        node: dict[str, object] = {"n": name, "k": "bytes", "t": "BytesIO", "b": len(value.data)}
        if value.data.startswith((_ZIP_MAGIC, b"\x80")) and depth < _MAX_DETAIL_DEPTH:
            try:
                _, inner = _decode_payload(value.data)
            except Exception as error:  # noqa: BLE001 - any decoding failure is reported on the page
                node["v"] = _clip(f"undecodable: {type(error).__name__}: {error}")
            else:
                node["c"] = [self.node("(archive)", inner, depth + 1)]
        return node


def _payload_node(value: object) -> dict[str, object] | None:
    if isinstance(value, _Tensor):
        return {"k": "tensor", "t": "Tensor", "d": value.dtype or "?", "s": value.shape, "b": value.nbytes}
    if isinstance(value, _NumpyArray):
        return {"k": "ndarray", "t": "ndarray", "d": value.dtype or "?", "s": value.shape, "b": value.nbytes}
    if isinstance(value, _StorageRef):
        return {"k": "tensor", "t": "Storage", "d": getattr(value.dtype, "name", "?"), "b": 0}
    if isinstance(value, _RawPayload):
        return {"k": "bytes", "t": "raw bytes", "b": value.length}
    text = _call_text(value) if isinstance(value, _Opaque) else None
    if text is not None:
        return {"k": "scalar", "t": value.qualname.rsplit(".", 1)[-1], "b": 0, "v": text}
    return None


def _container(value: object) -> tuple[str, str, list[tuple[str, object]]] | None:
    if isinstance(value, dict):
        return "dict", "dict", [(_key_text(key), item) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return type(value).__name__, type(value).__name__, [(f"[{index}]", item) for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return "set", type(value).__name__, [(f"{{{index}}}", item) for index, item in enumerate(value)]
    if isinstance(value, _Opaque):
        entries: list[tuple[str, object]] = []
        fields = _state_fields(value.state)
        if fields:
            entries.extend((_key_text(key), item) for key, item in fields.items())
        elif isinstance(value.state, (list, tuple)):
            entries.extend((f"state[{index}]", item) for index, item in enumerate(value.state))
        elif value.state is not None:
            entries.append(("state", value.state))
        if value.args and not fields:
            entries.extend((f"args[{index}]", item) for index, item in enumerate(value.args))
        entries.extend((f"[{index}]", item) for index, item in enumerate(value.items))
        entries.extend((_key_text(key), item) for key, item in value.mapping.items())
        return "object", value.qualname.rsplit(".", 1)[-1], entries
    return None


def describe_blob(raw: bytes) -> dict[str, object]:
    """Decode stored blob bytes into the page's blob description."""
    info: dict[str, object] = {"len": len(raw)}
    try:
        length, value = _decode_payload(raw)
        # A byte-stream state-dict value is stored pickled, with its archive inside.
        while isinstance(value, _BytesIO) and value.data.startswith((_ZIP_MAGIC, b"\x80")):
            _, value = _decode_payload(value.data)
            info["io"] = 1
    except Exception as error:  # noqa: BLE001 - any decoding failure is reported on the page
        info["err"] = _clip(f"{type(error).__name__}: {error}")
        return info
    info["arch"] = length
    # Megatron serializes each blob as a one-element list.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
        info["w"] = 1
    root = _Describer().node("", value, 0)
    root.pop("n", None)
    info.update(root)
    return info


def _read_blob(stream: IO[bytes], leaf: Leaf) -> dict[str, object]:
    _, offset, length = leaf.location
    if leaf.transform:
        return {"len": length, "err": f"stored with transforms {_jsonable(leaf.transform)}"}
    stream.seek(offset)
    raw = stream.read(length)
    if len(raw) != length:
        return {"len": length, "err": f"short read: {len(raw)} of {length} bytes"}
    return describe_blob(raw)


def _read_blob_file(source: Path, relative_path: str, group: list[Leaf]) -> None:
    group.sort(key=lambda leaf: leaf.location[1])
    try:
        with (source / relative_path).open("rb") as stream:
            for leaf in group:
                leaf.blob = _read_blob(stream, leaf)
    except OSError as error:
        for leaf in group:
            if leaf.blob is None:
                leaf.blob = {"len": leaf.location[2], "err": _clip(f"{type(error).__name__}: {error}")}


def _apply_blob(leaf: Leaf, problems: Counter) -> None:
    if "err" in leaf.blob:
        problems["blobs that could not be decoded"] += 1
        return
    logical = int(leaf.blob["arch"])
    leaf.nbytes = logical
    for slot in leaf.ranks.values():
        slot[0] = logical
    if leaf.blob.get("k") == "scalar":
        leaf.kind = "scalar"


def _expand_blobs(source: Path, leaves: list[Leaf], threads: int, problems: Counter) -> None:
    by_file: dict[str, list[Leaf]] = defaultdict(list)
    for leaf in leaves:
        if leaf.location is not None:
            by_file[leaf.location[0]].append(leaf)
    if by_file:
        with ThreadPoolExecutor(max_workers=max(1, min(threads, len(by_file)))) as pool:
            list(pool.map(lambda item: _read_blob_file(source, *item), by_file.items()))
    for leaf in leaves:
        if leaf.blob is not None:
            _apply_blob(leaf, problems)


def analyze(checkpoint_dir: Path | str, *, expand_blobs: bool = False, io_threads: int = 16) -> Report:
    """Read a checkpoint's metadata, and optionally its blobs, into a report."""
    source = Path(checkpoint_dir)
    metadata_path = source / ".metadata"
    with metadata_path.open("rb") as stream:
        fields = _object_fields(_unpickle(stream))
    if not isinstance(fields.get("state_dict_metadata"), dict):
        raise TypeError(f"{metadata_path} holds no distcp state_dict_metadata")
    problems: Counter = Counter()
    leaves, files = _collect_leaves(fields, problems)
    if expand_blobs:
        _expand_blobs(source, leaves, io_threads, problems)
    ranks = 1 + max((rank for leaf in leaves for rank in leaf.ranks if rank != UNKNOWN_RANK), default=-1)
    return Report(
        source=source.resolve(),
        leaves=leaves,
        ranks=ranks,
        files=files,
        metadata_bytes=metadata_path.stat().st_size,
        expanded=expand_blobs,
        warnings=[f"{count} {problem}" for problem, count in sorted(problems.items()) if count],
    )


# Page encoding.


class _Interner:
    def __init__(self) -> None:
        self.values: list[str] = []
        self._index: dict[str, int] = {}

    def add(self, value: str) -> int:
        position = self._index.get(value)
        if position is None:
            position = self._index[value] = len(self.values)
            self.values.append(value)
        return position


def _pack(arrays: dict[str, array]) -> tuple[bytes, dict[str, list]]:
    buffer = bytearray()
    index: dict[str, list] = {}
    for name, values in arrays.items():
        buffer.extend(b"\0" * (-len(buffer) % 8))
        data = values
        if sys.byteorder != "little":
            data = array(values.typecode, values)
            data.byteswap()
        index[name] = [len(buffer), len(values), "f64" if values.typecode == "d" else "u32"]
        buffer.extend(data.tobytes())
    return bytes(buffer), index


def _fit_dims(dims: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return dims[:ndim] + (0,) * max(0, ndim - len(dims))


def _leaf_item(leaf: Leaf, alt_path: list[str] | None, dtypes: _Interner) -> dict[str, object]:
    item: dict[str, object] = {"f": leaf.fqn, "p": split_fqn(leaf.fqn), "k": leaf.kind, "b": leaf.nbytes, "c": leaf.chunk_count()}
    if leaf.dtype is not None:
        item["d"] = dtypes.add(leaf.dtype)
        item["s"] = list(leaf.shape or ())
        item["cs"] = leaf.layout()
    else:
        item["bs"] = 1
    optional = {"u": leaf.unique, "st": leaf.storage, "a": alt_path, "x": leaf.blob, "m": leaf.extra}
    item.update((key, value) for key, value in optional.items() if value is not None)
    return item


def _encode_leaves(leaves: list[Leaf], dtypes: _Interner, shapes: _Interner) -> tuple[list[dict], dict[str, array]]:
    alt_paths = optimizer_paths([leaf.fqn for leaf in leaves])
    arrays = {
        "leafPtr": array(_U32, [0]),
        "entRank": array(_U32),
        "entBytes": array("d"),
        "entChunks": array(_U32),
        "entShape": array(_U32),
    }
    items = []
    for position, leaf in enumerate(leaves):
        items.append(_leaf_item(leaf, alt_paths.get(position), dtypes))
        for rank in sorted(leaf.ranks):
            nbytes, count, sizes, uniform = leaf.ranks[rank]
            label = leaf.rank_labels.get(rank)
            arrays["entRank"].append(rank)
            arrays["entBytes"].append(float(nbytes))
            arrays["entChunks"].append(count)
            arrays["entShape"].append(shapes.add(_layout_text(count, sizes, uniform) if label is None else label))
        arrays["leafPtr"].append(len(arrays["entRank"]))
    return items, arrays


def _encode_chunks(leaves: list[Leaf]) -> dict[str, array]:
    chunk_ptr = array(_U32, [0])
    chunk_rank = array(_U32)
    chunk_dims = array("d")
    for leaf in leaves:
        ndim = len(leaf.shape or ()) if leaf.dtype is not None else 0
        for rank, offsets, sizes in leaf.chunks:
            chunk_rank.append(rank)
            chunk_dims.extend(_fit_dims(offsets, ndim))
            chunk_dims.extend(_fit_dims(sizes, ndim))
        chunk_ptr.append(len(chunk_rank))
    return {"chunkPtr": chunk_ptr, "chunkRank": chunk_rank, "chunkDims": chunk_dims}


def encode_report(report: Report, *, title: str, chunk_details_limit: int = DEFAULT_CHUNK_DETAILS_LIMIT) -> tuple[dict, bytes]:
    """Build the page's JSON document and packed per-rank arrays."""
    dtypes = _Interner()
    shapes = _Interner()
    shapes.add("")
    items, arrays = _encode_leaves(report.leaves, dtypes, shapes)
    tensors = [leaf for leaf in report.leaves if leaf.dtype is not None]
    total_chunks = sum(leaf.chunk_count() for leaf in tensors)
    chunk_details = 0 < total_chunks <= chunk_details_limit and all(len(leaf.chunks) == leaf.chunk_count() for leaf in tensors)
    if chunk_details:
        arrays.update(_encode_chunks(report.leaves))
    binary, index = _pack(arrays)
    kinds = Counter(leaf.kind for leaf in report.leaves)
    doc = {
        "title": title,
        "source": str(report.source),
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "options": {
            "expandBlobs": report.expanded,
            "chunkDetails": chunk_details,
            "chunkDetailsLimit": chunk_details_limit,
            "unique": any(leaf.unique is not None for leaf in report.leaves),
            "values": any(leaf.blob is not None and "v" in leaf.blob for leaf in report.leaves),
        },
        "summary": {
            "leaves": len(report.leaves),
            "tensors": kinds["tensor"],
            "scalars": kinds["scalar"],
            "blobs": kinds["blob"],
            "chunks": total_chunks,
            "ranks": report.ranks,
            "files": report.files,
            "totalBytes": sum(leaf.nbytes for leaf in report.leaves),
            "uniqueBytes": sum(leaf.nbytes if leaf.unique is None else leaf.unique for leaf in report.leaves),
            "tensorBytes": sum(leaf.nbytes for leaf in tensors),
            "blobBytes": sum(leaf.nbytes for leaf in report.leaves if leaf.dtype is None),
            "unknownRankBytes": sum(
                slot[0] for leaf in report.leaves for rank, slot in leaf.ranks.items() if rank == UNKNOWN_RANK
            ),
            "metadataBytes": report.metadata_bytes,
        },
        "warnings": report.warnings,
        "inputLabel": report.input_label,
        "notes": report.notes,
        "hosts": report.hosts,
        "ranksPerHost": report.ranks_per_host,
        "dtypes": dtypes.values,
        "dtypeSizes": [DTYPE_ITEMSIZE.get(name, 0) for name in dtypes.values],
        "shapes": shapes.values,
        "leaves": items,
        "bin": index,
    }
    return doc, binary


def render_html(doc: dict, binary: bytes) -> str:
    """Embed the encoded report into the self-contained page."""
    doc_bytes = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    doc_text = base64.b64encode(gzip.compress(doc_bytes, mtime=0)).decode("ascii")
    bin_text = base64.b64encode(gzip.compress(binary, mtime=0)).decode("ascii")
    return _PAGE.replace("@@DOC@@", doc_text).replace("@@BIN@@", bin_text).replace("@@TITLE@@", html.escape(doc["title"]))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("checkpoint_dir", type=Path, help="checkpoint folder holding .metadata and the __<rank>_<n>.distcp files")
    parser.add_argument("-o", "--output", type=Path, help="HTML file to write (default: ./<checkpoint folder name>.distcp_tree.html)")
    parser.add_argument(
        "--expand-blobs",
        action="store_true",
        help="read and decode every blob from its .distcp file and expand its contents",
    )
    parser.add_argument(
        "--chunk-details-limit",
        type=int,
        default=DEFAULT_CHUNK_DETAILS_LIMIT,
        help="embed per-chunk offsets and sizes when the checkpoint has at most this many tensor chunks (0 disables)",
    )
    parser.add_argument("--io-threads", type=int, default=16, help="parallel .distcp file readers for --expand-blobs")
    parser.add_argument("--title", help="page title (default: the last two components of the checkpoint path)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source: Path = args.checkpoint_dir
    if not (source / ".metadata").is_file():
        sys.stderr.write(f"error: {source / '.metadata'} does not exist\n")
        return 2
    report = analyze(source, expand_blobs=args.expand_blobs, io_threads=args.io_threads)
    title = args.title or "/".join(report.source.parts[-2:])
    doc, binary = encode_report(report, title=title, chunk_details_limit=args.chunk_details_limit)
    output: Path = args.output or Path(f"{report.source.name}.distcp_tree.html")
    output.write_text(render_html(doc, binary), encoding="utf-8")
    summary = doc["summary"]
    sys.stdout.write(
        f"{output}: {summary['leaves']} entries ({summary['tensors']} tensors, {summary['scalars']} scalars, "
        f"{summary['blobs']} blobs), {summary['chunks']} chunks, {summary['ranks']} ranks, "
        f"{summary['totalBytes']} logical bytes\n",
    )
    for warning in report.warnings:
        sys.stderr.write(f"warning: {warning}\n")
    return 0


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --other: #c3c2b7;
  --critical: #d03b3b;
  --wash: rgba(42, 120, 214, 0.16);
  --hover: rgba(11, 11, 11, 0.04);
  --selected: rgba(42, 120, 214, 0.13);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --other: #383835;
    --wash: rgba(57, 135, 229, 0.26);
    --hover: rgba(255, 255, 255, 0.05);
    --selected: rgba(57, 135, 229, 0.22);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --other: #383835;
  --wash: rgba(57, 135, 229, 0.26);
  --hover: rgba(255, 255, 255, 0.05);
  --selected: rgba(57, 135, 229, 0.22);
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body { height: 100%; margin: 0; }
body { background: var(--page); color: var(--ink); font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.app { display: flex; flex-direction: column; height: 100vh; padding: 12px 16px; gap: 10px; }
.top { display: flex; flex-direction: column; gap: 8px; }
.title { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; }
h1 { margin: 0; font-size: 18px; font-weight: 600; }
h2 { margin: 0; font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
h3 { margin: 0 0 4px; font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
h4 { margin: 12px 0 4px; font-size: 12px; font-weight: 600; color: var(--ink-2); }
.source { color: var(--ink-2); font-size: 12px; overflow-wrap: anywhere; }
.kpis { display: flex; flex-wrap: wrap; gap: 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; min-width: 100px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 17px; font-weight: 600; }
.warnings { font-size: 12px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
.controls label { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); }
.grow { flex: 1; }
select, input, button {
  font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px;
}
button { cursor: pointer; }
button:hover { background: var(--hover); }
button[aria-pressed="true"], button[aria-selected="true"] { border-color: var(--ink-2); font-weight: 600; }
:focus-visible { outline: 2px solid var(--series-1); outline-offset: 1px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; flex: none; }
.card-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; }
.stats { display: flex; flex-wrap: wrap; gap: 2px 14px; color: var(--ink-2); font-size: 12px; }
.stats b { color: var(--ink); font-weight: 600; }
.help { color: var(--ink-2); font-size: 12px; }
.chart { position: relative; height: 172px; margin-top: 6px; }
.chart canvas, .map canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.tooltip {
  position: absolute; z-index: 5; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14); padding: 6px 9px; font-size: 12px; max-width: 460px;
}
.tooltip .tt-value { font-weight: 600; font-size: 13px; }
.tooltip .tt-label { color: var(--ink-2); overflow-wrap: anywhere; }
.rank-table { max-height: 240px; overflow: auto; margin-top: 6px; }
table.simple { border-collapse: collapse; font-size: 12px; width: 100%; }
table.simple th, table.simple td { padding: 2px 8px; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; }
table.simple th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
table.simple .num { text-align: right; font-variant-numeric: tabular-nums; }
table.simple tr.mark td { font-weight: 600; }
table.simple tr.pick { cursor: pointer; }
table.simple tr.pick:hover td { background: var(--hover); }
.tabs { display: flex; gap: 6px; flex: none; }
.menu { position: relative; }
.menu summary {
  list-style: none; cursor: pointer; border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; background: var(--surface);
}
.menu summary::-webkit-details-marker { display: none; }
.menu-body {
  position: absolute; z-index: 10; top: calc(100% + 4px); left: 0; display: flex; flex-direction: column; gap: 2px;
  min-width: 170px; padding: 6px 10px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14);
}
.menu-body label { color: var(--ink); white-space: nowrap; }
.main { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 10px; }
.main.hide-details { grid-template-columns: minmax(0, 1fr); }
.main.hide-details .details { display: none; }
.pane {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  display: flex; flex-direction: column; min-height: 0; min-width: 0; overflow: hidden;
}
.pane[hidden] { display: none; }
.head-wrap { overflow: hidden; border-bottom: 1px solid var(--grid); flex: none; }
.grid-row { display: grid; grid-template-columns: var(--cols); width: var(--grid-width); }
.head .cell { font-weight: 600; color: var(--ink-2); padding: 6px 8px; border: 0; border-radius: 0; background: none; text-align: left; }
.head .cell.num { text-align: right; }
.head .cell:hover { background: var(--hover); }
.filters .cell { padding: 0 4px 6px; }
.filters input, .filters select { width: 100%; padding: 2px 6px; font-size: 12px; }
.filters .bad { outline: 2px solid var(--critical); }
.viewport { flex: 1; min-height: 0; overflow: auto; position: relative; }
.sizer { position: relative; width: var(--grid-width); }
.row { position: absolute; top: 0; left: 0; height: 24px; align-items: center; cursor: default; }
.row:hover { background: var(--hover); }
.row.selected { background: var(--selected); }
.cell { padding: 0 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.cell.num { text-align: right; font-variant-numeric: tabular-nums; }
.cell.size { position: relative; }
.cell.size .bar { position: absolute; left: 0; top: 4px; bottom: 4px; background: var(--wash); border-radius: 0 4px 4px 0; }
.cell.size .txt { position: relative; }
.cell.info { color: var(--ink-2); font-style: italic; }
.cell.name { display: flex; align-items: center; gap: 4px; padding-right: 8px; }
.cell.name .label { overflow: hidden; text-overflow: ellipsis; }
.twisty { width: 14px; flex: none; color: var(--ink-2); text-align: center; cursor: pointer; }
.count { color: var(--muted); font-size: 11px; flex: none; }
.status { border-top: 1px solid var(--grid); padding: 4px 10px; color: var(--ink-2); font-size: 12px; flex: none; }
.details { overflow: auto; padding: 10px 12px; display: block; }
.details .path {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--ink-2); overflow-wrap: anywhere;
}
.details dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 10px; margin: 8px 0; font-size: 12px; }
.details dt { color: var(--ink-2); }
.details dd { margin: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.details .scroll { max-height: 320px; overflow: auto; }
.details .note { color: var(--ink-2); font-size: 12px; margin: 6px 0; }
.map-head {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; padding: 6px 10px;
  border-bottom: 1px solid var(--grid); flex: none;
}
.crumbs { display: flex; flex-wrap: wrap; align-items: center; gap: 2px; }
.crumbs button { border: 0; background: none; padding: 0 2px; text-decoration: underline; }
.crumbs .sep { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; color: var(--ink-2); font-size: 12px; }
.legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.map { position: relative; flex: 1; min-height: 0; }
.empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--ink-2); }
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="title"><h1 id="title">Loading…</h1><span class="source" id="source"></span></div>
    <div class="kpis" id="kpis"></div>
    <div class="warnings" id="warnings"></div>
  </header>
  <div class="controls">
    <label>View <select id="view">
      <option value="keys">Key hierarchy</option>
      <option value="optim">Optimizer under params</option>
    </select></label>
    <label>Rank A <select id="rankA"></select></label>
    <label>Rank B <select id="rankB"></select></label>
    <label><input type="checkbox" id="onlyA"> Only entries on rank A</label>
    <button type="button" id="expandAll" title="Expand every folder; blob contents stay collapsed">Expand all</button>
    <button type="button" id="collapseAll">Collapse all</button>
    <label>Expand to level <select id="level"></select></label>
    <button type="button" id="clearFilters">Clear filters</button>
    <details class="menu" id="columnsMenu"><summary>Columns</summary><div class="menu-body" id="columnsBody"></div></details>
    <span class="grow"></span>
    <button type="button" id="toggleChart" aria-pressed="true">Rank chart</button>
    <button type="button" id="toggleDetails" aria-pressed="true">Details</button>
    <label>Theme <select id="theme">
      <option value="">Auto</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select></label>
  </div>
  <section class="card" id="rankCard">
    <div class="card-head">
      <h2 id="rankTitle">Bytes per rank</h2>
      <div class="stats" id="rankStats"></div>
      <span class="grow"></span>
      <span class="help">Click: rank A · Shift+click: rank B</span>
      <button type="button" id="rankTableToggle" aria-pressed="false">Table view</button>
    </div>
    <div class="chart" id="rankChart">
      <canvas id="rankCanvas" role="img" aria-label="Bytes per rank"></canvas>
      <div class="tooltip" id="rankTip" hidden></div>
    </div>
    <div class="rank-table" id="rankTable" hidden></div>
  </section>
  <div class="tabs" role="tablist">
    <button type="button" role="tab" id="tabTree" aria-selected="true" aria-controls="treePane">Tree</button>
    <button type="button" role="tab" id="tabMap" aria-selected="false" aria-controls="mapPane">Treemap</button>
  </div>
  <div class="main" id="main">
    <section class="pane" id="treePane" role="tabpanel">
      <div class="head-wrap" id="headWrap">
        <div class="grid-row head" id="head" role="row"></div>
        <div class="grid-row filters" id="filters"></div>
      </div>
      <div class="viewport" id="viewport" tabindex="0" role="tree" aria-label="State-dict tree">
        <div class="sizer" id="sizer"></div>
      </div>
      <div class="status" id="status">Loading…</div>
    </section>
    <section class="pane" id="mapPane" role="tabpanel" hidden>
      <div class="map-head">
        <nav class="crumbs" id="crumbs" aria-label="Treemap focus"></nav>
        <span class="grow"></span>
        <label>Area <select id="mapMetric"></select></label>
        <div class="legend" id="legend"></div>
      </div>
      <div class="map" id="map">
        <canvas id="mapCanvas" role="img" aria-label="Treemap of the focused subtree"></canvas>
        <div class="empty" id="mapEmpty" hidden></div>
        <div class="tooltip" id="mapTip" hidden></div>
      </div>
    </section>
    <aside class="pane details" id="details" aria-label="Details"></aside>
  </div>
</div>
<script id="doc-data" type="application/octet-stream">@@DOC@@</script>
<script id="bin-data" type="application/octet-stream">@@BIN@@</script>
<script>
"use strict";
const DistcpTree = (() => {
  const UNKNOWN_RANK = 0xffffffff;
  const UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
  const collator = new Intl.Collator("en", {numeric: true, sensitivity: "base"});

  function fmtBytes(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "";
    let v = Math.abs(n);
    if (v < 1024) return `${n} B`;
    let unit = 0;
    while (v >= 1024 && unit < UNITS.length - 1) {
      v /= 1024;
      unit += 1;
    }
    const text = v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    return `${n < 0 ? "-" : ""}${text} ${UNITS[unit]}`;
  }

  function fmtCount(n) {
    return Number(n).toLocaleString("en-US");
  }

  function fmtPct(x) {
    if (!Number.isFinite(x)) return "";
    if (x === 0) return "0%";
    if (x < 0.0001) return "<0.01%";
    return `${(x * 100).toFixed(x < 0.1 ? 2 : 1)}%`;
  }

  function shapeText(shape) {
    return Array.isArray(shape) ? `[${shape.join(", ")}]` : "";
  }

  function b64ToBytes(text) {
    const raw = atob(text.trim());
    const out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
    return out;
  }

  async function gunzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function decodePayload(docText, binText) {
    const doc = JSON.parse(new TextDecoder().decode(await gunzip(b64ToBytes(docText))));
    const raw = await gunzip(b64ToBytes(binText));
    const arrays = {};
    for (const [name, spec] of Object.entries(doc.bin)) {
      const [offset, length, type] = spec;
      const Kind = type === "f64" ? Float64Array : Uint32Array;
      arrays[name] = new Kind(raw.buffer, raw.byteOffset + offset, length);
    }
    return {doc, arrays};
  }

  function buildModel(doc, arrays) {
    const leaves = doc.leaves;
    const nLeaves = leaves.length;
    const nRanks = doc.summary.ranks;
    const {leafPtr, entRank, entBytes, entChunks, entShape} = arrays;
    const nEnt = entRank.length;
    const entLeaf = new Uint32Array(nEnt);
    const leafRanks = new Uint32Array(nLeaves);
    const rankPtr = new Uint32Array(nRanks + 1);
    const rankBytes = new Float64Array(nRanks);
    for (let l = 0; l < nLeaves; l += 1) {
      for (let e = leafPtr[l]; e < leafPtr[l + 1]; e += 1) {
        entLeaf[e] = l;
        const r = entRank[e];
        if (r < nRanks) {
          rankPtr[r + 1] += 1;
          rankBytes[r] += entBytes[e];
          leafRanks[l] += 1;
        }
      }
    }
    for (let r = 0; r < nRanks; r += 1) rankPtr[r + 1] += rankPtr[r];
    const rankEnt = new Uint32Array(rankPtr[nRanks]);
    const cursor = rankPtr.slice(0, nRanks);
    for (let e = 0; e < nEnt; e += 1) {
      const r = entRank[e];
      if (r < nRanks) {
        rankEnt[cursor[r]] = e;
        cursor[r] += 1;
      }
    }
    const dtypeBytes = new Float64Array(doc.dtypes.length);
    for (const leaf of leaves) if (leaf.d !== undefined) dtypeBytes[leaf.d] += leaf.b;
    const ranked = Array.from(dtypeBytes.keys())
      .filter((d) => dtypeBytes[d] > 0)
      .sort((p, q) => dtypeBytes[q] - dtypeBytes[p]);
    const dtypeSlot = new Int8Array(doc.dtypes.length).fill(-1);
    ranked.slice(0, 3).forEach((d, i) => {
      dtypeSlot[d] = i;
    });
    const leafSlot = Int8Array.from(leaves, (leaf) => (leaf.d !== undefined ? dtypeSlot[leaf.d] : -1));
    let dimStart = null;
    if (arrays.chunkPtr) {
      dimStart = new Float64Array(nLeaves);
      let acc = 0;
      for (let l = 0; l < nLeaves; l += 1) {
        dimStart[l] = acc;
        const ndim = leaves[l].d !== undefined ? leaves[l].s.length : 0;
        acc += (arrays.chunkPtr[l + 1] - arrays.chunkPtr[l]) * 2 * ndim;
      }
    }
    return {
      doc, leaves, nLeaves, nRanks, leafPtr, entRank, entBytes, entChunks, entShape, entLeaf, leafRanks,
      rankPtr, rankEnt, rankBytes, leafSlot,
      legend: ranked.slice(0, 3).map((d) => doc.dtypes[d]),
      otherDtypes: ranked.slice(3).map((d) => doc.dtypes[d]),
      dtypes: doc.dtypes,
      dtypeSizes: doc.dtypeSizes,
      shapes: doc.shapes,
      chunkPtr: arrays.chunkPtr || null,
      chunkRank: arrays.chunkRank || null,
      chunkDims: arrays.chunkDims || null,
      dimStart,
    };
  }

  function rankEntries(model, rank) {
    if (!(rank >= 0 && rank < model.nRanks)) return null;
    const ent = new Int32Array(model.nLeaves).fill(-1);
    for (let i = model.rankPtr[rank]; i < model.rankPtr[rank + 1]; i += 1) {
      const e = model.rankEnt[i];
      ent[model.entLeaf[e]] = e;
    }
    return ent;
  }

  function buildTree(model, view) {
    const name = [];
    const parent = [];
    const children = [];
    const leaf = [];
    const info = [];
    const owner = [];
    const lookup = [];
    const leafNode = new Int32Array(model.nLeaves).fill(-1);
    const add = (label, par, item, own) => {
      const id = name.length;
      name.push(label);
      parent.push(par);
      children.push([]);
      leaf.push(-1);
      info.push(item);
      owner.push(own);
      lookup.push(null);
      if (par >= 0) children[par].push(id);
      return id;
    };
    const root = add("", -1, null, -1);
    const place = (node, l) => {
      leaf[node] = l;
      owner[node] = l;
      leafNode[l] = node;
    };
    const uniqueChild = (par, label) => {
      const map = lookup[par] || (lookup[par] = new Map());
      let text = label;
      for (let k = 2; map.has(text); k += 1) text = `${label} ${k}`;
      const id = add(text, par, null, -1);
      map.set(text, id);
      return id;
    };
    const vacate = (node) => {
      const l = leaf[node];
      if (l < 0) return;
      leaf[node] = -1;
      owner[node] = -1;
      place(uniqueChild(node, "(self)"), l);
    };
    const child = (par, label) => {
      const map = lookup[par];
      const hit = map ? map.get(label) : undefined;
      if (hit !== undefined) return hit;
      vacate(par);
      return uniqueChild(par, label);
    };
    model.leaves.forEach((entry, l) => {
      const path = view === "optim" && entry.a ? entry.a : entry.p;
      let node = root;
      for (const segment of path) node = child(node, segment);
      vacate(node);
      if (node === root || children[node].length > 0) node = uniqueChild(node, "(self)");
      place(node, l);
    });
    const attach = (par, items, l) => {
      for (const item of items) {
        const id = add(item.n === "" ? "(value)" : String(item.n), par, item, l);
        if (item.c) attach(id, item.c, l);
      }
    };
    model.leaves.forEach((entry, l) => {
      if (entry.x && entry.x.c) attach(leafNode[l], entry.x.c, l);
    });
    const n = name.length;
    const order = new Uint32Array(n);
    const lo = new Uint32Array(n);
    const hi = new Uint32Array(n);
    const depth = new Int16Array(n);
    const dfsLeaves = new Uint32Array(model.nLeaves);
    let k = 0;
    let m = 0;
    depth[root] = -1;
    const stack = [root];
    while (stack.length) {
      const v = stack.pop();
      if (v < 0) {
        hi[~v] = m;
        continue;
      }
      order[k] = v;
      k += 1;
      lo[v] = m;
      if (leaf[v] >= 0) {
        dfsLeaves[m] = leaf[v];
        m += 1;
      }
      stack.push(~v);
      const kids = children[v];
      for (let j = kids.length - 1; j >= 0; j -= 1) {
        depth[kids[j]] = depth[v] + 1;
        stack.push(kids[j]);
      }
    }
    const path = new Array(n);
    path[root] = "";
    for (let i = 1; i < n; i += 1) {
      const v = order[i];
      const p = parent[v];
      if (leaf[v] >= 0) path[v] = model.leaves[leaf[v]].f;
      else if (info[v]) path[v] = info[p] ? `${path[p]}.${name[v]}` : `${path[p]} \u203a ${name[v]}`;
      else path[v] = p === root ? name[v] : `${path[p]}.${name[v]}`;
    }
    return {n, root, name, parent, children, leaf, info, owner, order, lo, hi, depth, dfsLeaves, leafNode, path};
  }

  function leafShape(entry) {
    if (entry.d !== undefined) return shapeText(entry.s);
    const x = entry.x;
    if (!x || x.err) return "";
    if (x.s) return shapeText(x.s);
    return x.l !== undefined ? `(${fmtCount(x.l)})` : "";
  }

  function infoShape(item) {
    if (item.s) return shapeText(item.s);
    return item.l !== undefined ? `(${fmtCount(item.l)})` : "";
  }

  function leafDtype(model, entry) {
    if (entry.d !== undefined) return model.dtypes[entry.d];
    const x = entry.x;
    return x && !x.err ? x.d || x.t || "" : "";
  }

  function leafValue(entry) {
    const x = entry.x;
    if (!x) return "";
    if (x.err) return `decode failed: ${x.err}`;
    return x.v !== undefined ? x.v : "";
  }

  function rankValue(ctx, v, ent, counts, sums) {
    if (!ent) return null;
    const {model, tree} = ctx;
    const l = tree.leaf[v];
    if (l >= 0) return ent[l] >= 0 ? model.entBytes[ent[l]] : null;
    const item = tree.info[v];
    if (item) return ent[tree.owner[v]] >= 0 ? item.b ?? 0 : null;
    return counts[v] > 0 ? sums[v] : null;
  }

  function rankShape(ctx, v, ent, counts) {
    if (!ent) return null;
    const {model, tree, ev} = ctx;
    const l = tree.leaf[v];
    if (l >= 0) return ent[l] >= 0 ? model.shapes[model.entShape[ent[l]]] : null;
    if (tree.info[v]) return null;
    return counts[v] > 0 ? `${fmtCount(counts[v])}/${fmtCount(ev.leafCount[v])} entries` : null;
  }

  function cellValue(ctx, col, v) {
    const {model, tree, ev} = ctx;
    const l = tree.leaf[v];
    const item = tree.info[v];
    const entry = l >= 0 ? model.leaves[l] : null;
    switch (col) {
      case "name":
        return tree.name[v];
      case "path":
        return tree.path[v];
      case "kind":
        return item ? item.k : entry ? entry.k : "dir";
      case "dtype":
        return item ? item.d || item.t || "" : entry ? leafDtype(model, entry) : "";
      case "shape":
        return item ? infoShape(item) : entry ? leafShape(entry) : "";
      case "chunks":
        return item ? null : entry ? entry.c : ev.chunks[v];
      case "ranks":
        return entry ? model.leafRanks[l] : null;
      case "all":
        return item ? item.b ?? null : entry ? entry.b : ev.all[v];
      case "unique":
        return item ? null : entry ? entry.u ?? entry.b : ev.unique[v];
      case "a":
        return rankValue(ctx, v, ctx.entA, ev.countA, ev.a);
      case "b":
        return rankValue(ctx, v, ctx.entB, ev.countB, ev.b);
      case "ashape":
        return rankShape(ctx, v, ctx.entA, ev.countA);
      case "bshape":
        return rankShape(ctx, v, ctx.entB, ev.countB);
      case "value":
        return item ? item.v ?? "" : entry ? leafValue(entry) : "";
      default:
        return null;
    }
  }

  function evaluate(model, tree, entA, entB, filters, onlyA) {
    const n = tree.n;
    const ev = {
      matched: new Uint8Array(n),
      visible: new Uint8Array(n),
      all: new Float64Array(n),
      unique: new Float64Array(n),
      a: new Float64Array(n),
      b: new Float64Array(n),
      chunks: new Float64Array(n),
      leafCount: new Uint32Array(n),
      countA: new Uint32Array(n),
      countB: new Uint32Array(n),
      matchedLeaves: 0,
      active: filters.length > 0 || onlyA,
    };
    const ctx = {model, tree, ev, entA, entB};
    for (let v = 0; v < n; v += 1) {
      if (tree.leaf[v] < 0 && !tree.info[v]) continue;
      let ok = onlyA ? entA !== null && entA[tree.owner[v]] >= 0 : true;
      for (let i = 0; ok && i < filters.length; i += 1) ok = filters[i].test(cellValue(ctx, filters[i].col, v));
      ev.matched[v] = ok ? 1 : 0;
    }
    for (let k = n - 1; k >= 0; k -= 1) {
      const v = tree.order[k];
      const l = tree.leaf[v];
      if (ev.matched[v]) {
        ev.visible[v] = 1;
        if (l >= 0) {
          const entry = model.leaves[l];
          ev.all[v] = entry.b;
          ev.unique[v] = entry.u ?? entry.b;
          ev.chunks[v] = entry.c;
          ev.leafCount[v] = 1;
          ev.matchedLeaves += 1;
          if (entA && entA[l] >= 0) {
            ev.a[v] = model.entBytes[entA[l]];
            ev.countA[v] = 1;
          }
          if (entB && entB[l] >= 0) {
            ev.b[v] = model.entBytes[entB[l]];
            ev.countB[v] = 1;
          }
        }
      }
      const p = tree.parent[v];
      if (p < 0) continue;
      if (ev.visible[v]) ev.visible[p] = 1;
      if (tree.info[v]) continue;
      ev.all[p] += ev.all[v];
      ev.unique[p] += ev.unique[v];
      ev.a[p] += ev.a[v];
      ev.b[p] += ev.b[v];
      ev.chunks[p] += ev.chunks[v];
      ev.leafCount[p] += ev.leafCount[v];
      ev.countA[p] += ev.countA[v];
      ev.countB[p] += ev.countB[v];
    }
    return ev;
  }

  function isBlank(x) {
    return x === null || x === undefined || x === "";
  }

  function compareValues(p, q, dir) {
    if (isBlank(p) || isBlank(q)) return isBlank(p) === isBlank(q) ? 0 : isBlank(p) ? 1 : -1;
    const c = typeof p === "number" && typeof q === "number" ? p - q : collator.compare(String(p), String(q));
    return dir === "desc" ? -c : c;
  }

  function buildRows(ctx, expanded, sort) {
    const {tree, ev} = ctx;
    const rows = [];
    const kids = (v) => {
      const list = tree.children[v].filter((c) => ev.visible[c]);
      if (!sort) return list;
      const keyed = list.map((c) => [c, cellValue(ctx, sort.col, c)]);
      keyed.sort((p, q) => compareValues(p[1], q[1], sort.dir));
      return keyed.map((pair) => pair[0]);
    };
    const stack = kids(tree.root).reverse();
    while (stack.length) {
      const v = stack.pop();
      rows.push(v);
      if (expanded.has(v)) {
        const list = kids(v);
        for (let j = list.length - 1; j >= 0; j -= 1) stack.push(list[j]);
      }
    }
    return rows;
  }

  const SIZE_UNITS = {
    b: 1, k: 1024, kib: 1024, kb: 1e3, m: 1024 ** 2, mib: 1024 ** 2, mb: 1e6, g: 1024 ** 3, gib: 1024 ** 3, gb: 1e9,
    t: 1024 ** 4, tib: 1024 ** 4, tb: 1e12, p: 1024 ** 5, pib: 1024 ** 5, pb: 1e15,
  };

  function parseQuantity(text, sized) {
    const m = /^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*([a-z]*)\s*$/i.exec(text);
    if (!m) return NaN;
    const unit = m[2].toLowerCase();
    if (!unit) return Number(m[1]);
    return sized && unit in SIZE_UNITS ? Number(m[1]) * SIZE_UNITS[unit] : NaN;
  }

  function makeFilter(col, type, spec) {
    const text = String(spec || "").trim();
    if (!text) return null;
    if (type === "select") return {col, test: (v) => String(v ?? "") === text};
    if (type === "text") {
      const negate = text.startsWith("!");
      const body = negate ? text.slice(1) : text;
      let match;
      const re = /^\/(.+)\/([a-z]*)$/i.exec(body);
      if (re) {
        let rx;
        try {
          rx = new RegExp(re[1], `${re[2].replace(/[gyi]/g, "")}i`);
        } catch (error) {
          return {col, error: `invalid regular expression: ${error.message}`};
        }
        match = (s) => rx.test(s);
      } else {
        const needle = body.toLowerCase();
        match = (s) => s.toLowerCase().includes(needle);
      }
      return {col, test: (v) => match(isBlank(v) ? "" : String(v)) !== negate};
    }
    const sized = type === "size";
    const usage = sized ? "use >1MiB, <=4096, =0 or 1KiB..2MiB" : "use >1, <=4, =0 or 2..8";
    const range = /^(.*)\.\.(.*)$/.exec(text);
    if (range) {
      const low = range[1].trim() ? parseQuantity(range[1], sized) : -Infinity;
      const high = range[2].trim() ? parseQuantity(range[2], sized) : Infinity;
      if (Number.isNaN(low) || Number.isNaN(high)) return {col, error: usage};
      return {col, test: (v) => typeof v === "number" && v >= low && v <= high};
    }
    const m = /^(>=|<=|!=|>|<|=)?(.*)$/.exec(text);
    const op = m[1] || (sized ? ">=" : "=");
    const q = parseQuantity(m[2], sized);
    if (Number.isNaN(q)) return {col, error: usage};
    const ops = {
      ">=": (v) => v >= q, "<=": (v) => v <= q, ">": (v) => v > q, "<": (v) => v < q, "=": (v) => v === q, "!=": (v) => v !== q,
    };
    const compare = ops[op];
    return {col, test: (v) => typeof v === "number" && compare(v)};
  }

  function rankTotals(model, tree, ev, node) {
    const acc = new Float64Array(model.nRanks);
    const v = tree.info[node] ? tree.leafNode[tree.owner[node]] : node;
    for (let k = tree.lo[v]; k < tree.hi[v]; k += 1) {
      const l = tree.dfsLeaves[k];
      if (!ev.matched[tree.leafNode[l]]) continue;
      for (let e = model.leafPtr[l]; e < model.leafPtr[l + 1]; e += 1) {
        const r = model.entRank[e];
        if (r < model.nRanks) acc[r] += model.entBytes[e];
      }
    }
    return acc;
  }

  function squarify(values, x0, y0, w0, h0) {
    const out = values.map(() => ({x: x0, y: y0, w: 0, h: 0}));
    let total = 0;
    for (const value of values) total += value;
    if (!(total > 0) || w0 <= 0 || h0 <= 0) return out;
    const scale = (w0 * h0) / total;
    let x = x0;
    let y = y0;
    let w = w0;
    let h = h0;
    let i = 0;
    while (i < values.length) {
      const side = Math.min(w, h);
      let j = i;
      let sum = 0;
      let best = Infinity;
      while (j < values.length) {
        const area = values[j] * scale;
        const next = sum + area;
        const worst = Math.max((side * side * values[i] * scale) / (next * next), (next * next) / (side * side * area));
        if (j > i && worst > best) break;
        best = worst;
        sum = next;
        j += 1;
      }
      if (w >= h) {
        const width = sum / h;
        let yy = y;
        for (let k = i; k < j; k += 1) {
          const height = (values[k] * scale) / width;
          out[k] = {x, y: yy, w: width, h: height};
          yy += height;
        }
        x += width;
        w -= width;
      } else {
        const height = sum / w;
        let xx = x;
        for (let k = i; k < j; k += 1) {
          const width = (values[k] * scale) / height;
          out[k] = {x: xx, y, w: width, h: height};
          xx += width;
        }
        y += height;
        h -= height;
      }
      i = j;
    }
    return out;
  }

  function treemapLayout(tree, ev, valueOf, focus, width, height, maxDepth = 16, maxRects = 20000) {
    const rects = [];
    const kidsOf = (v) => (tree.leaf[v] >= 0 ? [] : tree.children[v].filter((c) => !tree.info[c] && ev.visible[c] && valueOf(c) > 0));
    const visit = (top, x, y, w, h, depth) => {
      const chain = [top];
      let v = top;
      let kids = kidsOf(v);
      while (depth > 0 && kids.length === 1) {
        v = kids[0];
        chain.push(v);
        kids = kidsOf(v);
      }
      const label = chain.map((node) => tree.name[node]).join(" \u203a ");
      const rect = {v, label, x, y, w, h, depth, header: 0, folder: false};
      rects.push(rect);
      if (depth >= maxDepth || rects.length >= maxRects) return;
      if (!kids.length) return;
      const header = depth > 0 && h >= 40 && w >= 60 ? 17 : 0;
      const pad = depth > 0 ? 3 : 0;
      const ix = x + pad;
      const iy = y + (header || pad);
      const iw = w - 2 * pad;
      const ih = h - (header || pad) - pad;
      if (iw < 6 || ih < 6) return;
      rect.folder = true;
      rect.header = header;
      kids.sort((p, q) => valueOf(q) - valueOf(p));
      const boxes = squarify(kids.map(valueOf), ix, iy, iw, ih);
      for (let i = 0; i < kids.length; i += 1) {
        const box = boxes[i];
        if (box.w * box.h >= 2 && rects.length < maxRects) visit(kids[i], box.x, box.y, box.w, box.h, depth + 1);
      }
    };
    visit(focus, 0, 0, width, height, 0);
    return rects;
  }

  function niceTicks(max, target = 4) {
    if (!(max > 0)) return {top: 1, ticks: [0], unit: 1, unitName: "B"};
    let unit = 1;
    let u = 0;
    while (max / unit >= 1024 && u < UNITS.length - 1) {
      unit *= 1024;
      u += 1;
    }
    const span = max / unit;
    const raw = span / target;
    const pow = 10 ** Math.floor(Math.log10(raw));
    let step = 10 * pow;
    for (const f of [1, 2, 2.5, 5, 10]) {
      if (f * pow >= raw) {
        step = f * pow;
        break;
      }
    }
    if (u === 0) step = Math.max(1, Math.ceil(step));
    const top = Math.ceil(span / step - 1e-9) * step;
    const ticks = [];
    for (let k = 0; k * step <= top + step * 1e-9; k += 1) ticks.push(k * step * unit);
    return {top: top * unit, ticks, unit, unitName: UNITS[u]};
  }

  return {
    UNKNOWN_RANK, fmtBytes, fmtCount, fmtPct, shapeText, decodePayload, buildModel, rankEntries, buildTree, cellValue,
    evaluate, buildRows, makeFilter, parseQuantity, rankTotals, squarify, treemapLayout, niceTicks,
  };
})();

function startApp(DT) {
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs, ...kids) => {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? "" : String(value));
      }
    }
    for (const kid of kids) if (kid !== null && kid !== undefined && kid !== false) node.append(kid);
    return node;
  };
  const ROW_H = 24;
  const INDENT = 14;
  const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  const COLUMNS = [
    {id: "name", title: "Name", width: 280, filter: "text", hint: "key text, /regex/, !negate"},
    {id: "kind", title: "Kind", width: 64, filter: "select"},
    {id: "dtype", title: "DType", width: 92, filter: "select"},
    {id: "shape", title: "Shape", width: 128, filter: "text", hint: "text"},
    {id: "chunks", title: "Chunks", width: 72, filter: "num", num: true, hint: ">1"},
    {id: "ranks", title: "Ranks", width: 64, filter: "num", num: true, hint: ">1"},
    {id: "all", title: "All ranks", width: 112, filter: "size", num: true, size: true, hint: ">1MiB"},
    {id: "unique", title: "Unique", width: 112, filter: "size", num: true, size: true, hint: ">1MiB", needsUnique: true},
    {id: "a", title: "Rank A", width: 124, filter: "size", num: true, size: true, hint: ">0"},
    {id: "ashape", title: "Rank A shape", width: 128, filter: "text", hint: "text"},
    {id: "b", title: "Rank B", width: 124, filter: "size", num: true, size: true, hint: ">0", needsB: true},
    {id: "bshape", title: "Rank B shape", width: 128, filter: "text", hint: "text", needsB: true},
    {id: "value", title: "Value", width: 180, filter: "text", hint: "text", needsValues: true},
  ];
  const S = {
    doc: null, model: null, tree: null, ev: null, ctx: null, view: "keys",
    rankA: -1, rankB: -1, entA: null, entB: null, onlyA: false,
    filterText: {}, filters: [], filterInputs: {}, sort: null, hiddenColumns: new Set(),
    expanded: new Set(), rows: [], rowIndex: new Map(), selected: 0, focus: 0,
    tab: "tree", columns: [], pool: [], rankValues: new Float64Array(0), hoverRank: -1,
    rankTable: false, metric: "all", mapRects: [], hoverRect: null, filterTimer: 0,
  };
  const viewport = $("viewport");
  const sizer = $("sizer");
  const statusLine = $("status");

  statusLine.textContent = "Decoding embedded data…";
  DT.decodePayload($("doc-data").textContent, $("bin-data").textContent)
    .then(({doc, arrays}) => init(doc, arrays))
    .catch((error) => {
      statusLine.textContent = `Could not load the embedded data: ${error && error.message ? error.message : error}`;
      throw error;
    });

  function init(doc, arrays) {
    S.doc = doc;
    S.model = DT.buildModel(doc, arrays);
    document.title = doc.title;
    renderHeader();
    setupControls();
    setupTree();
    setupChart();
    setupMap();
    setView("keys");
  }

  function colors() {
    const style = getComputedStyle(document.documentElement);
    const get = (name) => style.getPropertyValue(name).trim();
    return {
      ink: get("--ink"), ink2: get("--ink-2"), muted: get("--muted"), grid: get("--grid"), axis: get("--axis"),
      surface: get("--surface"), other: get("--other"), series: [get("--series-1"), get("--series-2"), get("--series-3")],
    };
  }

  function luminance(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return 0.5;
    const n = parseInt(m[1], 16);
    const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
      const s = c / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  function textOn(fill) {
    const lum = luminance(fill);
    return 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.05 ? "#ffffff" : "#0b0b0b";
  }

  function hostOf(rank) {
    const {hosts, ranksPerHost} = S.doc;
    return hosts && hosts.length && ranksPerHost ? hosts[Math.floor(rank / ranksPerHost)] || "" : "";
  }

  function setupCanvas(canvas, box) {
    const width = box.clientWidth;
    const height = box.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    g.font = FONT;
    return {g, width, height};
  }

  function fitText(g, text, maxWidth) {
    if (maxWidth <= 0) return "";
    if (g.measureText(text).width <= maxWidth) return text;
    let lo = 0;
    let hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (g.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) lo = mid;
      else hi = mid - 1;
    }
    return lo >= 3 ? `${text.slice(0, lo)}…` : "";
  }

  function placeTip(tip, box, x, y) {
    tip.hidden = false;
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    const left = x + 14 + width > box.clientWidth ? Math.max(0, x - 14 - width) : x + 14;
    const top = y + 14 + height > box.clientHeight ? Math.max(0, y - 14 - height) : y + 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function renderHeader() {
    const {doc} = S;
    const sum = doc.summary;
    $("title").textContent = doc.title;
    $("source").textContent = doc.source;
    const tiles = [
      ["Total size", DT.fmtBytes(sum.totalBytes), `${DT.fmtCount(sum.totalBytes)} logical bytes`],
      ["Tensors", DT.fmtCount(sum.tensors)],
      ["Scalars", DT.fmtCount(sum.scalars)],
      ["Blobs", DT.fmtCount(sum.blobs)],
      ["Chunks", DT.fmtCount(sum.chunks)],
      ["Ranks", DT.fmtCount(sum.ranks)],
      [doc.inputLabel || ".metadata", DT.fmtBytes(sum.metadataBytes), `${DT.fmtCount(sum.metadataBytes)} bytes`],
    ];
    if (doc.options.unique) {
      tiles.splice(1, 0, ["Unique", DT.fmtBytes(sum.uniqueBytes), "each distinct shard range, blob and scalar counted once"]);
    }
    if (sum.unknownRankBytes > 0) {
      tiles.push(["Unattributed", DT.fmtBytes(sum.unknownRankBytes), "bytes in entries without a known rank"]);
    }
    $("kpis").replaceChildren(
      ...tiles.map(([label, value, tip]) =>
        el("div", {class: "tile", title: tip}, el("div", {class: "label", text: label}), el("div", {class: "value", text: value}))),
    );
    $("warnings").replaceChildren(
      ...(doc.warnings || []).map((warning) => el("div", null, el("span", {"aria-hidden": "true", text: "⚠ "}), `Warning: ${warning}`)),
    );
  }

  function rankOptions(select, withNone) {
    const fragment = document.createDocumentFragment();
    if (withNone) fragment.append(el("option", {value: "-1", text: "none"}));
    for (let r = 0; r < S.model.nRanks; r += 1) {
      fragment.append(el("option", {value: String(r), text: `${r} · ${DT.fmtBytes(S.model.rankBytes[r])}`}));
    }
    select.replaceChildren(fragment);
  }

  function setupControls() {
    const {model} = S;
    rankOptions($("rankA"), model.nRanks === 0);
    rankOptions($("rankB"), true);
    $("rankA").addEventListener("change", (event) => setRankA(Number(event.target.value)));
    $("rankB").addEventListener("change", (event) => setRankB(Number(event.target.value)));
    S.rankA = model.nRanks ? 0 : -1;
    S.entA = DT.rankEntries(model, S.rankA);
    $("rankA").value = String(S.rankA);
    $("rankB").value = "-1";
    $("onlyA").addEventListener("change", (event) => {
      S.onlyA = event.target.checked;
      recompute({expandMatches: S.onlyA});
    });
    const view = $("view");
    if (!model.leaves.some((entry) => entry.a)) {
      const option = view.querySelector('option[value="optim"]');
      option.disabled = true;
      option.title = "This checkpoint has no optimizer.state.<state>.<param> keys";
      view.title = option.title;
    }
    setupColumnsMenu();
    view.addEventListener("change", () => setView(view.value));
    $("expandAll").addEventListener("click", () => {
      expandWhere((v) => !S.tree.info[v] && S.tree.leaf[v] < 0);
    });
    $("collapseAll").addEventListener("click", () => {
      S.expanded.clear();
      refreshRows();
    });
    $("level").addEventListener("change", (event) => {
      const level = Number(event.target.value);
      if (level > 0) expandWhere((v) => !S.tree.info[v] && S.tree.leaf[v] < 0 && S.tree.depth[v] < level, true);
    });
    $("clearFilters").addEventListener("click", clearFilters);
    $("toggleChart").addEventListener("click", (event) => {
      const show = $("rankCard").hidden;
      $("rankCard").hidden = !show;
      event.target.setAttribute("aria-pressed", String(show));
      if (show) renderChart();
      afterLayoutChange();
    });
    $("toggleDetails").addEventListener("click", (event) => {
      const main = $("main");
      const show = main.classList.contains("hide-details");
      main.classList.toggle("hide-details", !show);
      event.target.setAttribute("aria-pressed", String(show));
      afterLayoutChange();
    });
    $("theme").addEventListener("change", (event) => {
      if (event.target.value) document.documentElement.dataset.theme = event.target.value;
      else delete document.documentElement.dataset.theme;
      renderChart();
      if (S.tab === "map") renderMap();
    });
    $("tabTree").addEventListener("click", () => setTab("tree"));
    $("tabMap").addEventListener("click", () => setTab("map"));
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      renderChart();
      if (S.tab === "map") renderMap();
    });
  }

  function afterLayoutChange() {
    layoutColumns();
    paintRows();
    if (S.tab === "map") renderMap();
  }

  function setTab(tab) {
    S.tab = tab;
    $("tabTree").setAttribute("aria-selected", String(tab === "tree"));
    $("tabMap").setAttribute("aria-selected", String(tab === "map"));
    $("treePane").hidden = tab !== "tree";
    $("mapPane").hidden = tab !== "map";
    if (tab === "map") renderMap();
    else {
      layoutColumns();
      paintRows();
    }
  }

  function setView(view) {
    S.view = view;
    S.tree = DT.buildTree(S.model, view);
    S.selected = S.tree.root;
    S.focus = S.tree.root;
    let maxDepth = 0;
    for (let v = 0; v < S.tree.n; v += 1) if (!S.tree.info[v]) maxDepth = Math.max(maxDepth, S.tree.depth[v] + 1);
    const level = $("level");
    level.replaceChildren(el("option", {value: "0", text: "—"}));
    for (let d = 1; d <= Math.min(maxDepth, 16); d += 1) level.append(el("option", {value: String(d), text: String(d)}));
    setupGrid();
    S.ev = DT.evaluate(S.model, S.tree, S.entA, S.entB, S.filters, S.onlyA);
    S.ctx = {model: S.model, tree: S.tree, ev: S.ev, entA: S.entA, entB: S.entB};
    S.expanded = defaultExpansion();
    recompute({expandMatches: S.filters.length > 0 || S.onlyA});
  }

  function defaultExpansion() {
    const {tree} = S;
    const set = new Set();
    const structural = (v) => !tree.info[v] && tree.leaf[v] < 0;
    for (const v of tree.children[tree.root]) {
      let node = v;
      if (structural(node)) set.add(node);
      while (structural(node) && tree.children[node].length === 1 && tree.depth[node] < 8) {
        node = tree.children[node][0];
        if (structural(node)) set.add(node);
      }
    }
    return set;
  }

  function expandWhere(predicate, replace) {
    const {tree, ev} = S;
    const set = replace ? new Set() : S.expanded;
    for (let v = 0; v < tree.n; v += 1) {
      if (v !== tree.root && ev.visible[v] && tree.children[v].length && predicate(v)) set.add(v);
    }
    S.expanded = set;
    refreshRows();
  }

  function expandVisible(limit) {
    const {tree, ev} = S;
    let rows = 0;
    for (let k = 0; k < tree.n && rows < limit; k += 1) {
      const v = tree.order[k];
      if (v === tree.root || !ev.visible[v]) continue;
      rows += 1;
      if (tree.children[v].some((c) => ev.visible[c])) S.expanded.add(v);
    }
  }

  function setRankA(rank) {
    S.rankA = rank;
    S.entA = DT.rankEntries(S.model, rank);
    $("rankA").value = String(rank);
    renderHead();
    metricOptions();
    recompute();
  }

  function setRankB(rank) {
    const toggled = (rank >= 0) !== (S.rankB >= 0);
    S.rankB = rank;
    S.entB = DT.rankEntries(S.model, rank);
    $("rankB").value = String(rank);
    if (toggled) setupGrid();
    else renderHead();
    metricOptions();
    recompute();
  }

  function offered(col) {
    const {options} = S.doc;
    return (!col.needsValues || options.expandBlobs || options.values) && (!col.needsUnique || options.unique);
  }

  function activeColumns() {
    return COLUMNS.filter((col) => !S.hiddenColumns.has(col.id) && (!col.needsB || S.rankB >= 0) && offered(col));
  }

  function setupColumnsMenu() {
    const menu = $("columnsMenu");
    const choices = COLUMNS.slice(1).filter(offered);
    $("columnsBody").replaceChildren(
      ...choices.map((col) => {
        const box = el("input", {type: "checkbox", checked: true});
        box.addEventListener("change", () => {
          if (box.checked) S.hiddenColumns.delete(col.id);
          else S.hiddenColumns.add(col.id);
          setupGrid();
          recompute();
        });
        return el("label", {title: col.needsB ? "Shown while rank B is selected" : null}, box, ` ${col.title}`);
      }),
    );
    document.addEventListener("click", (event) => {
      if (menu.open && !menu.contains(event.target)) menu.open = false;
    });
  }

  function setupGrid() {
    S.columns = activeColumns();
    for (const entry of S.pool) entry.el.remove();
    S.pool = [];
    layoutColumns();
    renderHead();
    renderFilterRow();
  }

  function layoutColumns() {
    const pane = $("treePane");
    const others = S.columns.slice(1).reduce((total, col) => total + col.width, 0);
    const available = pane.clientWidth - 18;
    const widths = [Math.max(COLUMNS[0].width, available - others), ...S.columns.slice(1).map((col) => col.width)];
    pane.style.setProperty("--cols", widths.map((w) => `${w}px`).join(" "));
    pane.style.setProperty("--grid-width", `${widths.reduce((total, w) => total + w, 0)}px`);
  }

  function columnTitle(col) {
    if (col.id === "a") return S.rankA >= 0 ? `${col.title} (r${S.rankA})` : col.title;
    if (col.id === "b") return S.rankB >= 0 ? `${col.title} (r${S.rankB})` : col.title;
    return col.title;
  }

  function renderHead() {
    $("head").replaceChildren(
      ...S.columns.map((col) => {
        const sorted = S.sort && S.sort.col === col.id;
        const ascending = sorted && S.sort.dir === "asc";
        return el(
          "button",
          {
            type: "button",
            class: `cell${col.num ? " num" : ""}`,
            role: "columnheader",
            "aria-sort": sorted ? (ascending ? "ascending" : "descending") : "none",
            title: `Sort by ${columnTitle(col)}`,
            onclick: () => toggleSort(col),
          },
          columnTitle(col),
          sorted ? (ascending ? " ▲" : " ▼") : "",
        );
      }),
    );
  }

  function toggleSort(col) {
    const first = col.num ? "desc" : "asc";
    const second = col.num ? "asc" : "desc";
    if (!S.sort || S.sort.col !== col.id) S.sort = {col: col.id, dir: first};
    else if (S.sort.dir === first) S.sort = {col: col.id, dir: second};
    else S.sort = null;
    renderHead();
    refreshRows();
  }

  function filterOptions(colId) {
    const values = new Set();
    const ctx = {model: S.model, tree: S.tree, ev: S.ev, entA: null, entB: null};
    for (let v = 0; v < S.tree.n; v += 1) {
      if (S.tree.leaf[v] < 0 && !S.tree.info[v]) continue;
      const value = DT.cellValue(ctx, colId, v);
      if (value) values.add(String(value));
    }
    return Array.from(values).sort((p, q) => p.localeCompare(q, "en", {numeric: true}));
  }

  function renderFilterRow() {
    S.filterInputs = {};
    $("filters").replaceChildren(
      ...S.columns.map((col) => {
        let input;
        if (col.filter === "select") {
          input = el("select", {"aria-label": `Filter ${col.title}`}, el("option", {value: "", text: "all"}));
          for (const value of filterOptions(col.id)) input.append(el("option", {value, text: value}));
        } else {
          input = el("input", {type: "search", placeholder: col.hint || "", "aria-label": `Filter ${col.title}`, spellcheck: "false"});
        }
        input.value = S.filterText[col.id] || "";
        input.addEventListener(col.filter === "select" ? "change" : "input", () => {
          S.filterText[col.id] = input.value;
          clearTimeout(S.filterTimer);
          S.filterTimer = setTimeout(applyFilters, col.filter === "select" ? 0 : 200);
        });
        S.filterInputs[col.id] = input;
        return el("div", {class: "cell"}, input);
      }),
    );
  }

  function buildFilters() {
    const filters = [];
    for (const col of S.columns) {
      const made = DT.makeFilter(col.id === "name" ? "path" : col.id, col.filter, S.filterText[col.id]);
      const input = S.filterInputs[col.id];
      if (input) {
        input.classList.toggle("bad", Boolean(made && made.error));
        input.title = made && made.error ? made.error : col.hint ? `Filter: ${col.hint}` : "";
      }
      if (made && !made.error) filters.push(made);
    }
    return filters;
  }

  function applyFilters() {
    S.filters = buildFilters();
    recompute({expandMatches: S.filters.length > 0});
  }

  function clearFilters() {
    S.filterText = {};
    S.onlyA = false;
    $("onlyA").checked = false;
    for (const input of Object.values(S.filterInputs)) {
      input.value = "";
      input.classList.remove("bad");
    }
    S.filters = [];
    recompute();
  }

  function recompute(options = {}) {
    S.filters = buildFilters();
    S.ev = DT.evaluate(S.model, S.tree, S.entA, S.entB, S.filters, S.onlyA);
    S.ctx = {model: S.model, tree: S.tree, ev: S.ev, entA: S.entA, entB: S.entB};
    if (!S.ev.visible[S.selected]) {
      S.selected = S.tree.root;
      S.focus = S.tree.root;
    }
    if (options.expandMatches) expandVisible(3000);
    refreshRows();
    renderChart();
    if (S.tab === "map") renderMap();
    renderDetails();
  }

  // Tree table.

  function setupTree() {
    viewport.addEventListener("scroll", () => {
      $("headWrap").scrollLeft = viewport.scrollLeft;
      paintRows();
    });
    viewport.addEventListener("keydown", onTreeKey);
    new ResizeObserver(() => {
      layoutColumns();
      paintRows();
    }).observe($("treePane"));
  }

  function refreshRows() {
    S.rows = DT.buildRows(S.ctx, S.expanded, S.sort);
    S.rowIndex = new Map();
    S.rows.forEach((v, i) => S.rowIndex.set(v, i));
    sizer.style.height = `${S.rows.length * ROW_H}px`;
    paintRows();
    renderStatus();
  }

  function makeRow() {
    const row = el("div", {class: "grid-row row", role: "treeitem"});
    const cells = {};
    for (const col of S.columns) {
      const cell = el("div", {class: `cell${col.num ? " num" : ""}${col.size ? " size" : ""}`});
      if (col.id === "name") {
        const twisty = el("span", {class: "twisty", "aria-hidden": "true"});
        const label = el("span", {class: "label"});
        const count = el("span", {class: "count"});
        cell.classList.add("name");
        cell.append(twisty, label, count);
        cells.name = {cell, twisty, label, count};
      } else if (col.id === "all") {
        const bar = el("span", {class: "bar"});
        const txt = el("span", {class: "txt"});
        cell.append(bar, txt);
        cells.all = {cell, bar, txt};
      } else {
        cells[col.id] = {cell};
      }
      row.append(cell);
    }
    const entry = {el: row, cells, v: -1};
    row.addEventListener("click", (event) => {
      if (event.target.classList.contains("twisty")) toggle(entry.v);
      else select(entry.v);
    });
    row.addEventListener("dblclick", (event) => {
      if (!event.target.classList.contains("twisty")) toggle(entry.v);
    });
    return entry;
  }

  function hasVisibleChildren(v) {
    return S.tree.children[v].some((c) => S.ev.visible[c]);
  }

  function paintRows() {
    if (!S.tree || S.tab !== "tree") return;
    const top = viewport.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW_H) - 6);
    const last = Math.min(S.rows.length, Math.ceil((top + viewport.clientHeight) / ROW_H) + 6);
    const needed = Math.max(0, last - first);
    while (S.pool.length < needed) {
      const entry = makeRow();
      S.pool.push(entry);
      sizer.append(entry.el);
    }
    S.pool.forEach((entry, i) => {
      if (i < needed) {
        entry.el.hidden = false;
        fillRow(entry, S.rows[first + i], first + i);
      } else {
        entry.el.hidden = true;
      }
    });
  }

  function fillRow(entry, v, index) {
    const {tree, ev} = S;
    entry.v = v;
    const row = entry.el;
    row.style.transform = `translateY(${index * ROW_H}px)`;
    row.classList.toggle("selected", v === S.selected);
    row.setAttribute("aria-selected", String(v === S.selected));
    row.setAttribute("aria-level", String(tree.depth[v] + 1));
    const expandable = hasVisibleChildren(v);
    if (expandable) row.setAttribute("aria-expanded", String(S.expanded.has(v)));
    else row.removeAttribute("aria-expanded");
    const isInfo = Boolean(tree.info[v]);
    const isFolder = tree.leaf[v] < 0 && !isInfo;
    const total = ev.all[tree.root];
    for (const col of S.columns) {
      const slot = entry.cells[col.id];
      if (col.id === "name") {
        slot.cell.style.paddingLeft = `${4 + tree.depth[v] * INDENT}px`;
        slot.twisty.textContent = expandable ? (S.expanded.has(v) ? "▾" : "▸") : "";
        slot.label.textContent = tree.name[v];
        slot.count.textContent = isFolder ? `(${DT.fmtCount(ev.leafCount[v])})` : "";
        slot.cell.title = tree.path[v];
        slot.cell.classList.toggle("info", isInfo);
        continue;
      }
      const value = DT.cellValue(S.ctx, col.id, v);
      const blank = value === null || value === undefined || value === "";
      const text = blank ? "" : col.size ? DT.fmtBytes(value) : col.num ? DT.fmtCount(value) : String(value);
      slot.cell.classList.toggle("info", isInfo && Boolean(col.size));
      if (col.id === "all") {
        slot.txt.textContent = text;
        const p = tree.parent[v];
        const base = !isInfo && p >= 0 ? ev.all[p] : 0;
        slot.bar.style.width = base > 0 && value > 0 ? `${Math.min(100, (value / base) * 100)}%` : "0";
        let tip = blank ? "" : `${DT.fmtCount(value)} bytes`;
        if (!blank && base > 0) tip += ` · ${DT.fmtPct(value / base)} of parent`;
        if (!blank && !isInfo && total > 0) tip += ` · ${DT.fmtPct(value / total)} of total`;
        if (isInfo) tip += " · decoded payload estimate, not part of folder totals";
        slot.cell.title = tip;
      } else {
        slot.cell.textContent = text;
        slot.cell.title = col.size && !blank ? `${DT.fmtCount(value)} bytes` : text;
      }
    }
  }

  function renderStatus() {
    const {ev, model} = S;
    let text = `${DT.fmtCount(S.rows.length)} rows · ${DT.fmtCount(ev.matchedLeaves)} of ${DT.fmtCount(model.nLeaves)} entries`;
    if (ev.active) text += " match the filters · folder sizes count matching entries only";
    statusLine.textContent = text;
  }

  function toggle(v) {
    if (!hasVisibleChildren(v)) return;
    if (S.expanded.has(v)) S.expanded.delete(v);
    else S.expanded.add(v);
    refreshRows();
  }

  function select(v, options = {}) {
    const {tree} = S;
    S.selected = v;
    if (!options.keepFocus) {
      const structural = !tree.info[v] && tree.leaf[v] < 0;
      let focus = structural ? v : tree.parent[v];
      while (focus > 0 && (tree.info[focus] || tree.leaf[focus] >= 0)) focus = tree.parent[focus];
      S.focus = Math.max(focus, tree.root);
    }
    paintRows();
    renderChart();
    renderDetails();
    if (S.tab === "map") renderMap();
  }

  function reveal(v) {
    const {tree} = S;
    let changed = false;
    for (let p = tree.parent[v]; p > tree.root; p = tree.parent[p]) {
      if (!S.expanded.has(p)) {
        S.expanded.add(p);
        changed = true;
      }
    }
    if (changed) refreshRows();
    const index = S.rowIndex.get(v);
    if (index === undefined) return;
    const top = index * ROW_H;
    if (top < viewport.scrollTop) viewport.scrollTop = top;
    else if (top + ROW_H > viewport.scrollTop + viewport.clientHeight) viewport.scrollTop = top + ROW_H - viewport.clientHeight;
    paintRows();
  }

  function onTreeKey(event) {
    const {tree} = S;
    const rows = S.rows;
    if (!rows.length) return;
    const index = S.rowIndex.has(S.selected) ? S.rowIndex.get(S.selected) : -1;
    const page = Math.max(1, Math.floor(viewport.clientHeight / ROW_H) - 1);
    let target = null;
    switch (event.key) {
      case "ArrowDown":
        target = rows[Math.min(rows.length - 1, index + 1)];
        break;
      case "ArrowUp":
        target = rows[Math.max(0, index - 1)];
        break;
      case "PageDown":
        target = rows[Math.min(rows.length - 1, index + page)];
        break;
      case "PageUp":
        target = rows[Math.max(0, index - page)];
        break;
      case "Home":
        target = rows[0];
        break;
      case "End":
        target = rows[rows.length - 1];
        break;
      case "ArrowRight":
        if (index < 0) return;
        if (hasVisibleChildren(S.selected) && !S.expanded.has(S.selected)) toggle(S.selected);
        else if (S.expanded.has(S.selected) && index + 1 < rows.length) target = rows[index + 1];
        break;
      case "ArrowLeft":
        if (index < 0) return;
        if (S.expanded.has(S.selected)) toggle(S.selected);
        else if (tree.parent[S.selected] > tree.root) target = tree.parent[S.selected];
        break;
      case "Enter":
      case " ":
        if (index >= 0) toggle(S.selected);
        break;
      case "Escape":
        target = tree.root;
        break;
      default:
        return;
    }
    event.preventDefault();
    if (target !== null && target !== undefined) {
      select(target);
      if (target !== tree.root) reveal(target);
    }
  }

  // Per-rank chart.

  function setupChart() {
    const canvas = $("rankCanvas");
    const box = $("rankChart");
    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const rank = rankAt(x);
      if (rank !== S.hoverRank) {
        S.hoverRank = rank;
        drawRankChart();
      }
      showRankTip(rank, x, y);
    });
    canvas.addEventListener("mouseleave", () => {
      S.hoverRank = -1;
      $("rankTip").hidden = true;
      drawRankChart();
    });
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const rank = rankAt(event.clientX - rect.left);
      if (rank < 0) return;
      if (event.shiftKey) setRankB(rank);
      else setRankA(rank);
    });
    $("rankTableToggle").addEventListener("click", (event) => {
      S.rankTable = !S.rankTable;
      event.target.setAttribute("aria-pressed", String(S.rankTable));
      renderChart();
    });
    new ResizeObserver(() => drawRankChart()).observe(box);
  }

  function chartGeometry() {
    const box = $("rankChart");
    const width = box.clientWidth;
    const height = box.clientHeight;
    const padL = 72;
    const padR = 12;
    const padT = 10;
    const padB = 44;
    return {width, height, padL, padT, plotW: Math.max(10, width - padL - padR), plotH: Math.max(10, height - padT - padB)};
  }

  function rankAt(x) {
    const n = S.rankValues.length;
    if (!n) return -1;
    const {padL, plotW} = chartGeometry();
    if (x < padL || x > padL + plotW) return -1;
    return Math.min(n - 1, Math.max(0, Math.floor(((x - padL) / plotW) * n)));
  }

  function tickLabel(value, ticks) {
    if (value === 0) return "0";
    return `${Number((value / ticks.unit).toFixed(2))} ${ticks.unitName}`;
  }

  function renderChart() {
    if ($("rankCard").hidden || !S.tree) return;
    const {tree, model, ev} = S;
    const node = tree.info[S.selected] ? tree.leafNode[tree.owner[S.selected]] : S.selected;
    S.rankValues = model.nRanks ? DT.rankTotals(model, tree, ev, node) : new Float64Array(0);
    const subject = node === tree.root ? "whole checkpoint" : tree.path[node];
    $("rankTitle").textContent = `Bytes per rank · ${subject}${ev.active ? " (matching entries)" : ""}`;
    const values = S.rankValues;
    let total = 0;
    let max = 0;
    let min = Infinity;
    let withData = 0;
    for (const x of values) {
      total += x;
      if (x > max) max = x;
      if (x < min) min = x;
      if (x > 0) withData += 1;
    }
    const mean = values.length ? total / values.length : 0;
    const stat = (label, value, tip) => el("span", {title: tip || null}, `${label} `, el("b", {text: value}));
    $("rankStats").replaceChildren(
      stat("Total", DT.fmtBytes(total), `${DT.fmtCount(total)} bytes`),
      stat("Ranks with data", `${DT.fmtCount(withData)} / ${DT.fmtCount(values.length)}`),
      stat("Min", values.length ? DT.fmtBytes(min) : "\u2013"),
      stat("Mean", values.length ? DT.fmtBytes(mean) : "\u2013"),
      stat("Max", values.length ? DT.fmtBytes(max) : "\u2013"),
      stat("Max ÷ mean", mean > 0 ? `${(max / mean).toFixed(2)}\u00d7` : "\u2013"),
    );
    $("rankChart").hidden = S.rankTable;
    $("rankTable").hidden = !S.rankTable;
    if (S.rankTable) renderRankTable(max);
    else drawRankChart();
  }

  function drawRankChart() {
    const box = $("rankChart");
    if (box.hidden || $("rankCard").hidden) return;
    const {g, width, height} = setupCanvas($("rankCanvas"), box);
    const c = colors();
    const values = S.rankValues;
    const n = values.length;
    if (!n || !width) {
      g.fillStyle = c.ink2;
      g.textBaseline = "middle";
      g.fillText("No per-rank data", 12, height / 2);
      return;
    }
    const {padL, padT, plotW, plotH} = chartGeometry();
    let max = 0;
    for (const x of values) if (x > max) max = x;
    const ticks = DT.niceTicks(max);
    const yOf = (value) => padT + plotH - (value / ticks.top) * plotH;
    const base = padT + plotH;
    g.lineWidth = 1;
    g.textBaseline = "middle";
    g.textAlign = "right";
    for (const t of ticks.ticks) {
      const y = Math.round(yOf(t)) + 0.5;
      g.strokeStyle = t === 0 ? c.axis : c.grid;
      g.beginPath();
      g.moveTo(padL, y);
      g.lineTo(padL + plotW, y);
      g.stroke();
      g.fillStyle = c.muted;
      g.fillText(tickLabel(t, ticks), padL - 8, y);
    }
    const slot = plotW / n;
    const gap = slot >= 6 ? 2 : 0;
    const barW = slot >= 1 ? Math.max(1, Math.min(24, slot - gap)) : 1;
    const xOf = (r) => padL + (r + 0.5) * slot;
    const drawBar = (r, fill) => {
      const value = values[r];
      if (!(value > 0)) return;
      const y = yOf(value);
      const h = Math.max(1, base - y);
      const x = xOf(r) - barW / 2;
      g.fillStyle = fill;
      g.beginPath();
      if (barW >= 8 && g.roundRect) g.roundRect(x, base - h, barW, h, [4, 4, 0, 0]);
      else g.rect(x, base - h, barW, h);
      g.fill();
    };
    if (slot >= 1) {
      for (let r = 0; r < n; r += 1) drawBar(r, c.series[0]);
    } else {
      g.fillStyle = c.series[0];
      for (let px = 0; px < plotW; px += 1) {
        const r0 = Math.floor((px / plotW) * n);
        const r1 = Math.max(r0 + 1, Math.floor(((px + 1) / plotW) * n));
        let value = 0;
        for (let r = r0; r < r1 && r < n; r += 1) value = Math.max(value, values[r]);
        if (value > 0) g.fillRect(padL + px, yOf(value), 1, base - yOf(value));
      }
    }
    if (S.hoverRank >= 0) {
      const x = Math.round(xOf(S.hoverRank)) + 0.5;
      if (slot >= 4) {
        g.globalAlpha = 0.45;
        drawBar(S.hoverRank, c.surface);
        g.globalAlpha = 1;
      } else {
        g.strokeStyle = c.ink2;
        g.beginPath();
        g.moveTo(x, padT);
        g.lineTo(x, base);
        g.stroke();
      }
    }
    g.textAlign = "center";
    g.textBaseline = "alphabetic";
    const markers = [];
    if (S.rankA >= 0 && S.rankA < n) markers.push([S.rankA, "A"]);
    if (S.rankB >= 0 && S.rankB < n) markers.push([S.rankB, "B"]);
    const placed = [];
    for (const [r, label] of markers) {
      const x = xOf(r);
      g.fillStyle = c.ink;
      g.beginPath();
      g.moveTo(x, base + 2);
      g.lineTo(x - 4, base + 8);
      g.lineTo(x + 4, base + 8);
      g.closePath();
      g.fill();
      const shared = placed.find((p) => Math.abs(p - x) < 12);
      g.fillText(label, shared !== undefined ? x + 10 : x, base + 20);
      placed.push(x);
    }
    const minGap = 56;
    const steps = [1, 2, 5];
    let step = 1;
    for (let k = 0; step * slot < minGap; k += 1) step = steps[k % 3] * 10 ** Math.floor(k / 3);
    g.fillStyle = c.muted;
    for (let r = 0; r < n; r += step) g.fillText(String(r), xOf(r), base + 36);
  }

  function showRankTip(rank, x, y) {
    const tip = $("rankTip");
    if (rank < 0) {
      tip.hidden = true;
      return;
    }
    const values = S.rankValues;
    let total = 0;
    for (const value of values) total += value;
    const mean = values.length ? total / values.length : 0;
    const value = values[rank];
    const marks = [rank === S.rankA ? "rank A" : "", rank === S.rankB ? "rank B" : ""].filter(Boolean).join(", ");
    tip.replaceChildren(
      el("div", {class: "tt-value", text: DT.fmtBytes(value)}),
      el("div", {class: "tt-label", text: `rank ${rank}${marks ? ` (${marks})` : ""}${hostOf(rank) ? ` · ${hostOf(rank)}` : ""}`}),
      el("div", {class: "tt-label", text: `${DT.fmtCount(value)} bytes${mean > 0 ? ` · ${(value / mean).toFixed(2)}\u00d7 mean` : ""}`}),
    );
    placeTip(tip, $("rankChart"), x, y);
  }

  function renderRankTable(max) {
    const values = S.rankValues;
    const withHosts = Boolean(hostOf(0));
    const body = el("tbody");
    for (let r = 0; r < values.length; r += 1) {
      const marks = `${r === S.rankA ? " (A)" : ""}${r === S.rankB ? " (B)" : ""}`;
      body.append(
        el(
          "tr",
          {
            class: `pick${marks ? " mark" : ""}`,
            title: "Click to set rank A, Shift+click to set rank B",
            onclick: (event) => (event.shiftKey ? setRankB(r) : setRankA(r)),
          },
          el("td", {text: `${r}${marks}`}),
          withHosts ? el("td", {text: hostOf(r)}) : null,
          el("td", {class: "num", text: DT.fmtBytes(values[r]), title: `${DT.fmtCount(values[r])} bytes`}),
          el("td", {class: "num", text: max > 0 ? DT.fmtPct(values[r] / max) : "\u2013"}),
        ),
      );
    }
    const titles = el(
      "tr",
      null,
      el("th", {text: "Rank"}),
      withHosts ? el("th", {text: "Host"}) : null,
      el("th", {class: "num", text: "Bytes"}),
      el("th", {class: "num", text: "% of max"}),
    );
    $("rankTable").replaceChildren(el("table", {class: "simple"}, el("thead", null, titles), body));
  }

  // Treemap.

  function setupMap() {
    const canvas = $("mapCanvas");
    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = mapHit(x, y);
      if (hit !== S.hoverRect) {
        S.hoverRect = hit;
        drawMap();
      }
      showMapTip(hit, x, y);
    });
    canvas.addEventListener("mouseleave", () => {
      S.hoverRect = null;
      $("mapTip").hidden = true;
      drawMap();
    });
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const hit = mapHit(event.clientX - rect.left, event.clientY - rect.top);
      if (!hit) return;
      if (hit.folder) {
        S.focus = hit.v;
        select(hit.v, {keepFocus: true});
      } else {
        select(hit.v, {keepFocus: true});
      }
      reveal(hit.v);
    });
    $("mapMetric").addEventListener("change", (event) => {
      S.metric = event.target.value;
      renderMap();
    });
    metricOptions();
    new ResizeObserver(() => {
      if (S.tab === "map") renderMap();
    }).observe($("map"));
  }

  function metricOptions() {
    const select = $("mapMetric");
    const options = [el("option", {value: "all", text: "All ranks"})];
    if (S.doc.options.unique) options.push(el("option", {value: "unique", text: "Unique"}));
    if (S.rankA >= 0) options.push(el("option", {value: "a", text: `Rank A (r${S.rankA})`}));
    if (S.rankB >= 0) options.push(el("option", {value: "b", text: `Rank B (r${S.rankB})`}));
    select.replaceChildren(...options);
    if ((S.metric === "a" && S.rankA < 0) || (S.metric === "b" && S.rankB < 0)) S.metric = "all";
    select.value = S.metric;
  }

  function metricOf() {
    const {ev} = S;
    if (S.metric === "a" && S.entA) return (v) => ev.a[v];
    if (S.metric === "b" && S.entB) return (v) => ev.b[v];
    if (S.metric === "unique") return (v) => ev.unique[v];
    return (v) => ev.all[v];
  }

  function renderMap() {
    if (!S.tree || S.tab !== "map") return;
    const {tree, ev} = S;
    if (S.focus !== tree.root && !ev.visible[S.focus]) S.focus = tree.root;
    const box = $("map");
    const valueOf = metricOf();
    S.mapRects = valueOf(S.focus) > 0 ? DT.treemapLayout(tree, ev, valueOf, S.focus, box.clientWidth, box.clientHeight) : [];
    const empty = $("mapEmpty");
    empty.hidden = S.mapRects.length > 0;
    empty.textContent = S.mapRects.length ? "" : "No bytes under this node for the selected area measure and filters";
    S.hoverRect = null;
    $("mapTip").hidden = true;
    renderCrumbs();
    renderLegend();
    drawMap();
  }

  function leafColor(l, c) {
    const slot = S.model.leafSlot[l];
    return slot >= 0 ? c.series[slot] : c.other;
  }

  function drawMap() {
    const box = $("map");
    const {g} = setupCanvas($("mapCanvas"), box);
    const {tree} = S;
    const c = colors();
    const valueOf = metricOf();
    g.textBaseline = "middle";
    g.textAlign = "left";
    for (const rect of S.mapRects) {
      if (rect.depth === 0) continue;
      const v = rect.v;
      if (rect.folder) {
        if (rect.header) {
          g.fillStyle = c.grid;
          g.fillRect(rect.x + 1, rect.y + 1, Math.max(0, rect.w - 2), rect.header - 2);
          const label = fitText(g, `${rect.label} · ${DT.fmtBytes(valueOf(v))}`, rect.w - 10);
          g.fillStyle = c.ink;
          if (label) g.fillText(label, rect.x + 5, rect.y + rect.header / 2);
        }
        continue;
      }
      const l = tree.leaf[v];
      const fill = l >= 0 ? leafColor(l, c) : c.grid;
      g.fillStyle = fill;
      g.fillRect(rect.x + 1, rect.y + 1, Math.max(0, rect.w - 2), Math.max(0, rect.h - 2));
      if (rect.w >= 30 && rect.h >= 16) {
        const ink = l >= 0 ? textOn(fill) : c.ink;
        g.fillStyle = ink;
        const name = fitText(g, rect.label, rect.w - 10);
        if (name) g.fillText(name, rect.x + 5, rect.y + 10);
        if (rect.h >= 32) {
          const size = fitText(g, DT.fmtBytes(valueOf(v)), rect.w - 10);
          if (size) g.fillText(size, rect.x + 5, rect.y + 24);
        }
      }
    }
    const outline = (rect) => {
      g.strokeStyle = c.ink;
      g.lineWidth = 2;
      g.strokeRect(rect.x + 2, rect.y + 2, Math.max(0, rect.w - 4), Math.max(0, rect.h - 4));
    };
    const selectedRect = S.mapRects.find((rect) => rect.v === S.selected && rect.depth > 0);
    if (selectedRect) outline(selectedRect);
    if (S.hoverRect && S.hoverRect !== selectedRect) outline(S.hoverRect);
  }

  function mapHit(x, y) {
    for (let i = S.mapRects.length - 1; i >= 0; i -= 1) {
      const rect = S.mapRects[i];
      if (rect.depth > 0 && x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h) return rect;
    }
    return null;
  }

  function showMapTip(hit, x, y) {
    const tip = $("mapTip");
    if (!hit) {
      tip.hidden = true;
      return;
    }
    const {tree, model} = S;
    const valueOf = metricOf();
    const value = valueOf(hit.v);
    const whole = valueOf(S.focus);
    const l = tree.leaf[hit.v];
    const rows = [
      el("div", {class: "tt-value", text: DT.fmtBytes(value)}),
      el("div", {class: "tt-label", text: tree.path[hit.v]}),
    ];
    if (l >= 0) {
      const entry = model.leaves[l];
      const facts = [entry.k, DT.cellValue(S.ctx, "dtype", hit.v), DT.cellValue(S.ctx, "shape", hit.v)].filter(Boolean).join(" · ");
      rows.push(el("div", {class: "tt-label", text: facts}));
    } else {
      rows.push(el("div", {class: "tt-label", text: `${DT.fmtCount(S.ev.leafCount[hit.v])} entries · click to zoom in`}));
    }
    rows.push(el("div", {class: "tt-label", text: `${DT.fmtCount(value)} bytes · ${DT.fmtPct(whole > 0 ? value / whole : 0)} of view`}));
    tip.replaceChildren(...rows);
    placeTip(tip, $("map"), x, y);
  }

  function renderCrumbs() {
    const {tree} = S;
    const chain = [];
    for (let v = S.focus; v >= 0; v = tree.parent[v]) chain.unshift(v);
    const parts = [];
    chain.forEach((v, i) => {
      if (i > 0) parts.push(el("span", {class: "sep", "aria-hidden": "true", text: "\u203a"}));
      const label = v === tree.root ? "All" : tree.name[v];
      if (i === chain.length - 1) parts.push(el("span", {"aria-current": "location", text: label}));
      else {
        parts.push(
          el("button", {
            type: "button",
            text: label,
            onclick: () => {
              S.focus = v;
              renderMap();
            },
          }),
        );
      }
    });
    $("crumbs").replaceChildren(...parts);
  }

  function renderLegend() {
    const {model} = S;
    const c = colors();
    const items = model.legend.map((name, i) => [c.series[i], name]);
    const others = [...model.otherDtypes, ...(model.leaves.some((entry) => entry.d === undefined) ? ["blobs"] : [])];
    if (others.length) items.push([c.other, `other (${others.join(", ")})`]);
    $("legend").replaceChildren(
      ...items.map(([color, label]) => el("span", null, el("span", {class: "swatch", style: `background:${color}`}), label)),
    );
  }

  // Details panel.

  function dl(rows) {
    const list = el("dl");
    for (const [key, value, tip] of rows) {
      if (value === null || value === undefined || value === "") continue;
      list.append(el("dt", {text: key}), el("dd", {text: String(value), title: tip || null}));
    }
    return list;
  }

  function note(text) {
    return el("div", {class: "note", text});
  }

  function bytesText(n) {
    return n === null || n === undefined ? "" : `${DT.fmtBytes(n)} (${DT.fmtCount(n)} B)`;
  }

  function simpleTable(headers, rows, numeric) {
    const head = el("tr", null, ...headers.map((h, i) => el("th", {class: numeric[i] ? "num" : null, text: h})));
    const body = el("tbody");
    for (const row of rows) {
      const cells = row.cells.map((cell, i) => el("td", {class: numeric[i] ? "num" : null, text: cell}));
      body.append(el("tr", {class: row.mark ? "mark" : null}, ...cells));
    }
    return el("div", {class: "scroll"}, el("table", {class: "simple"}, el("thead", null, head), body));
  }

  function dtypeBreakdown(v) {
    const {tree, model, ev} = S;
    const groups = new Map();
    for (let k = tree.lo[v]; k < tree.hi[v]; k += 1) {
      const l = tree.dfsLeaves[k];
      if (!ev.matched[tree.leafNode[l]]) continue;
      const entry = model.leaves[l];
      const key = entry.d !== undefined ? model.dtypes[entry.d] : entry.st ? entry.k : `${entry.k} (pickled)`;
      const group = groups.get(key) || {bytes: 0, count: 0};
      group.bytes += entry.b;
      group.count += 1;
      groups.set(key, group);
    }
    const total = ev.all[v];
    const rows = Array.from(groups.entries())
      .sort((p, q) => q[1].bytes - p[1].bytes)
      .map(([key, group]) => ({
        cells: [key, DT.fmtCount(group.count), DT.fmtBytes(group.bytes), DT.fmtPct(total > 0 ? group.bytes / total : 0)],
      }));
    return simpleTable(["Type", "Entries", "Bytes", "Share"], rows, [false, true, true, true]);
  }

  function renderDetails() {
    const {tree} = S;
    const v = S.selected;
    let parts;
    if (v === tree.root) parts = rootDetails();
    else if (tree.info[v]) parts = infoDetails(v);
    else if (tree.leaf[v] >= 0) parts = leafDetails(v);
    else parts = folderDetails(v);
    $("details").replaceChildren(...parts.filter(Boolean));
  }

  function rootDetails() {
    const {doc, ev, tree} = S;
    const sum = doc.summary;
    return [
      el("h3", {text: "Checkpoint"}),
      el("div", {class: "path", text: doc.source}),
      dl([
        ["Generated", doc.generated],
        ["Blob contents", doc.options.expandBlobs ? "decoded (--expand-blobs)" : "not decoded"],
        [
          "Chunk details",
          doc.options.chunkDetails ? "embedded" : `omitted (more than ${DT.fmtCount(doc.options.chunkDetailsLimit)} chunks)`,
        ],
        ["Entries", DT.fmtCount(sum.leaves)],
        ["Tensors", DT.fmtCount(sum.tensors)],
        ["Scalars", DT.fmtCount(sum.scalars)],
        ["Blobs", DT.fmtCount(sum.blobs)],
        ["Chunks", DT.fmtCount(sum.chunks)],
        ["Ranks", DT.fmtCount(sum.ranks)],
        ["Files", DT.fmtCount(sum.files)],
        ["Total", bytesText(sum.totalBytes)],
        ["Unique", doc.options.unique ? bytesText(sum.uniqueBytes) : ""],
        ["Tensor data", bytesText(sum.tensorBytes)],
        ["Non-tensor data", bytesText(sum.blobBytes)],
        ["Unattributed", sum.unknownRankBytes ? bytesText(sum.unknownRankBytes) : ""],
        ["Matching", ev.active ? bytesText(ev.all[tree.root]) : ""],
      ]),
      el("h4", {text: "Bytes by type"}),
      dtypeBreakdown(tree.root),
      el("h4", {text: "Reading the page"}),
      ...(doc.notes || []).map(note),
      doc.notes && doc.notes.length ? null : note(
        "Sizes are logical bytes: numel \u00d7 dtype size for tensor chunks and the stored length for pickled entries. " +
          "A rank owns what its __<rank>_<n>.distcp files hold; replicated entries are stored and counted once.",
      ),
      note(
        "Filters: text columns take a substring, /regex/ or a leading ! to negate; number columns take >x, <=x, =x or a..b, " +
          "with KiB/MiB/GiB units on sizes. Folder sizes then count matching entries only.",
      ),
      note("Keys in the tree: ↑/↓ move, →/← expand or collapse, Enter toggles, Esc selects the whole checkpoint."),
    ];
  }

  function folderDetails(v) {
    const {tree, ev} = S;
    const total = ev.all[tree.root];
    return [
      el("h3", {text: tree.name[v]}),
      el("div", {class: "path", text: tree.path[v]}),
      dl([
        ["Entries", DT.fmtCount(ev.leafCount[v])],
        ["All ranks", bytesText(ev.all[v])],
        ["Unique", S.doc.options.unique ? bytesText(ev.unique[v]) : ""],
        ["Share of total", total > 0 ? DT.fmtPct(ev.all[v] / total) : ""],
        ["Chunks", DT.fmtCount(ev.chunks[v])],
        [`Rank A (r${S.rankA})`, S.entA ? `${bytesText(ev.a[v])} · ${DT.fmtCount(ev.countA[v])} entries` : ""],
        [`Rank B (r${S.rankB})`, S.entB ? `${bytesText(ev.b[v])} · ${DT.fmtCount(ev.countB[v])} entries` : ""],
      ]),
      el("h4", {text: "Bytes by type"}),
      dtypeBreakdown(v),
    ];
  }

  function leafDetails(v) {
    const {tree, model} = S;
    const l = tree.leaf[v];
    const entry = model.leaves[l];
    const parts = [el("h3", {text: tree.name[v]}), el("div", {class: "path", text: entry.f})];
    const rows = [
      ["Kind", entry.k],
      ["Storage", entry.st || (entry.bs ? "pickled bytes (BytesStorageMetadata)" : "tensor chunks (TensorStorageMetadata)")],
    ];
    if (entry.d !== undefined) {
      const numel = entry.s.reduce((p, d) => p * d, 1);
      rows.push(
        ["DType", `${model.dtypes[entry.d]} (${model.dtypeSizes[entry.d]} B)`],
        ["Global shape", DT.shapeText(entry.s)],
        ["Elements", DT.fmtCount(numel)],
        ["Chunks", `${DT.fmtCount(entry.c)}${entry.cs ? ` · ${entry.cs}` : ""}`],
      );
    }
    rows.push(["Size", bytesText(entry.b)], ["Unique", bytesText(entry.u)], ["Ranks", DT.fmtCount(model.leafRanks[l])]);
    const onRank = (bytes, shape) => (bytes === null ? "not on this rank" : [bytesText(bytes), shape].filter(Boolean).join(" · "));
    if (S.entA) rows.push([`Rank A (r${S.rankA})`, onRank(DT.cellValue(S.ctx, "a", v), DT.cellValue(S.ctx, "ashape", v))]);
    if (S.entB) rows.push([`Rank B (r${S.rankB})`, onRank(DT.cellValue(S.ctx, "b", v), DT.cellValue(S.ctx, "bshape", v))]);
    parts.push(dl(rows));
    if (entry.x) parts.push(...blobDetails(entry.x));
    if (entry.m) parts.push(el("h4", {text: "Megatron metadata"}), el("div", {class: "path", text: JSON.stringify(entry.m)}));
    parts.push(el("h4", {text: "Ranks"}), rankBreakdown(l));
    if (model.chunkPtr && entry.d !== undefined) parts.push(el("h4", {text: "Chunks"}), chunkBreakdown(l));
    return parts;
  }

  function blobDetails(x) {
    const rows = [["Stored length", bytesText(x.len)]];
    if (x.err) rows.push(["Decoding", `failed: ${x.err}`]);
    else {
      rows.push(
        ["Archive length", bytesText(x.arch)],
        ["Slot padding", x.len > x.arch ? bytesText(x.len - x.arch) : ""],
        ["Decoded type", x.q || x.t],
        ["Value", x.v],
        ["Wrapper", [x.io ? "serialized io.BytesIO" : "", x.w ? "one-element list" : ""].filter(Boolean).join(" holding a ")],
        ["Omitted", x.m ? `${DT.fmtCount(x.m)} items beyond the display limits` : ""],
      );
    }
    return [el("h4", {text: x.len === undefined ? "Value" : "Blob"}), dl(rows)];
  }

  function rankBreakdown(l) {
    const {model} = S;
    const rows = [];
    const lo = model.leafPtr[l];
    const hi = model.leafPtr[l + 1];
    for (let e = lo; e < Math.min(hi, lo + 2000); e += 1) {
      const r = model.entRank[e];
      const label = r === DT.UNKNOWN_RANK ? "unattributed" : `${r}${r === S.rankA ? " (A)" : ""}${r === S.rankB ? " (B)" : ""}`;
      rows.push({
        mark: r === S.rankA || r === S.rankB,
        cells: [label, DT.fmtBytes(model.entBytes[e]), DT.fmtCount(model.entChunks[e]), model.shapes[model.entShape[e]]],
      });
    }
    const table = simpleTable(["Rank", "Bytes", "Chunks", "Shape"], rows, [false, true, true, false]);
    if (hi - lo <= 2000) return table;
    return el("div", null, table, el("div", {class: "note", text: `First 2,000 of ${DT.fmtCount(hi - lo)} ranks shown`}));
  }

  function chunkBreakdown(l) {
    const {model} = S;
    const entry = model.leaves[l];
    const ndim = entry.s.length;
    const itemsize = model.dtypeSizes[entry.d];
    const lo = model.chunkPtr[l];
    const hi = model.chunkPtr[l + 1];
    const rows = [];
    for (let k = lo; k < Math.min(hi, lo + 2000); k += 1) {
      const at = model.dimStart[l] + (k - lo) * 2 * ndim;
      const offsets = Array.from(model.chunkDims.subarray(at, at + ndim));
      const sizes = Array.from(model.chunkDims.subarray(at + ndim, at + 2 * ndim));
      const r = model.chunkRank[k];
      rows.push({
        mark: r === S.rankA || r === S.rankB,
        cells: [
          String(k - lo),
          r === DT.UNKNOWN_RANK ? "?" : String(r),
          DT.shapeText(offsets),
          DT.shapeText(sizes),
          DT.fmtBytes(sizes.reduce((p, d) => p * d, 1) * itemsize),
        ],
      });
    }
    const table = simpleTable(["#", "Rank", "Offsets", "Sizes", "Bytes"], rows, [true, true, false, false, true]);
    if (hi - lo <= 2000) return table;
    return el("div", null, table, el("div", {class: "note", text: `First 2,000 of ${DT.fmtCount(hi - lo)} chunks shown`}));
  }

  function infoDetails(v) {
    const {tree, model} = S;
    const item = tree.info[v];
    const entry = model.leaves[tree.owner[v]];
    return [
      el("h3", {text: tree.name[v]}),
      el("div", {class: "path", text: tree.path[v]}),
      dl([
        ["Kind", item.k],
        ["Type", item.q || item.t],
        ["DType", item.d],
        ["Shape", item.s ? DT.shapeText(item.s) : item.l !== undefined ? `${DT.fmtCount(item.l)} items` : ""],
        ["Payload estimate", bytesText(item.b)],
        ["Value", item.v],
        ["Omitted", item.m ? `${DT.fmtCount(item.m)} items beyond the display limits` : ""],
        ["Inside blob", entry.f],
      ]),
      note(
        "Sizes inside a blob estimate the decoded payload and do not add to folder totals; " +
          "the blob's own size is its archive length.",
      ),
    ];
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = DistcpTree;
} else {
  startApp(DistcpTree);
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
