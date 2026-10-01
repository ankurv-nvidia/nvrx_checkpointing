#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Render checkpoint daemon state registries as the interactive page of distcp_tree.py.

Reads the daemon's binary state registries: per-rank registries named
``registry.<rank>.bin`` or ``registry.rank<rank>.bin``, or the merged global
registry ``registry.bin`` that the daemon writes beside each checkpoint. Both
hold a rank topology followed by the flattened state tree, whose leaves are
tensor shards, blobs and 64-bit scalars.

Sizes are logical bytes as registered: ``numel * itemsize`` for each tensor
shard, the inline length of each blob and the value width of each scalar. A
rank's size is what that rank registered, replicas included. "All ranks" sums
every rank and "Unique" counts each distinct shard range, blob and scalar once.

Per-rank registries hold every entry. The global registry holds every rank's
tensor shards, but its blob and scalar nodes come from the first rank that
registered, carry no rank, and its blobs hold no bytes.

``--expand-blobs`` decodes the blob bytes held in per-rank registries with the
restricted unpickler of distcp_tree.py, which must sit next to this script.
Only the Python standard library is required.

Usage::

    python scripts/registry_tree.py <dir-or-registry-file> [...] [-o tree.html] [--expand-blobs]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import struct
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType


def _page_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("distcp_tree.py")
    loaded = sys.modules.get("distcp_tree")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location("distcp_tree", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass creation looks the defining module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


page = _page_module()

_INVALID, _DICT, _SCALAR, _BLOB, _TENSOR_SHARD, _SHARDED_TENSOR = range(6)
_DTYPES = {
    1: "float16",
    2: "bfloat16",
    3: "float32",
    4: "float64",
    5: "uint8",
    6: "uint16",
    7: "uint32",
    8: "uint64",
    9: "int8",
    10: "int16",
    11: "int32",
    12: "int64",
    13: "bool",
}
_MAX_DIMS = 4
_INVALID_RANK = 0xFFFFFFFF
_ROOT = 0
_UNSIGNED, _SIGNED, _FLOAT = 1, 2, 3
_SCALAR_WIDTHS = {1: 1, 2: 2, 3: 4, 4: 8}
_TOPOLOGY = struct.Struct("<IIII")
_U32 = struct.Struct("<I")
_NODE = struct.Struct("<BBHII")
_SCALAR_VALUE = struct.Struct("<B7xQ")
_TENSOR_SHARD_VALUE = struct.Struct("<9Q5QB7xQ4sI")
_SHARDED_HEADER = struct.Struct("<IB5Q")
_RANKED_SHARD = struct.Struct("<9Q16xI4x")
_DOUBLE = struct.Struct("<d")
_LOCAL_NAME = re.compile(r"registry\.(?:rank)?(\d+)\.bin")
_GLOBAL_NAMES = frozenset({"registry.bin", "registry.global.bin"})
_PREFETCH = 16
_SIZES_NOTE = (
    "Sizes are logical bytes as registered with the checkpoint daemon: numel times dtype size for each tensor shard, "
    "the inline length of each blob and the value width of each scalar. A rank's size is what that rank registered, "
    "replicas included; All ranks sums every rank and Unique counts each distinct shard range, blob and scalar once."
)


class _Reader:
    def __init__(self, path: Path, view: memoryview) -> None:
        self.path = path
        self.view = view
        self.pos = 0

    def take(self, size: int) -> memoryview:
        if self.pos + size > len(self.view):
            raise ValueError(f"{self.path}: truncated at byte {self.pos} (needs {size} more bytes)")
        chunk = self.view[self.pos : self.pos + size]
        self.pos += size
        return chunk

    def unpack(self, layout: struct.Struct) -> tuple:
        return layout.unpack(self.take(layout.size))

    def text(self, size: int) -> str:
        return bytes(self.take(size)).decode("utf-8", "replace")


@dataclass
class Registry:
    """One parsed registry file."""

    path: Path
    size: int
    rank: int
    world_size: int
    ranks_per_peer: int
    peers: list[str]
    names: dict[int, str] = field(default_factory=dict)
    parents: dict[int, int] = field(default_factory=dict)
    entries: list[tuple[int, int, object]] = field(default_factory=list)
    trailing: int = 0


def read_registry(path: Path, data: bytes | None = None) -> Registry:
    """Parse a registry file into its topology and tree entries."""
    raw = path.read_bytes() if data is None else data
    reader = _Reader(path, memoryview(raw))
    rank, world_size, ranks_per_peer, _ = reader.unpack(_TOPOLOGY)
    peer_count = -(-world_size // ranks_per_peer) if ranks_per_peer else 0
    peers = [reader.text(reader.unpack(_U32)[0]) for _ in range(peer_count)]
    registry = Registry(path, len(raw), rank, world_size, ranks_per_peer, peers)
    (count,) = reader.unpack(_U32)
    for _ in range(count):
        start = reader.pos
        kind, name_length, _, handle, parent = reader.unpack(_NODE)
        name = reader.text(name_length)
        if kind == _INVALID:
            continue
        if kind == _DICT:
            reader.take(_U32.size * reader.unpack(_U32)[0])
        elif kind == _SCALAR:
            registry.entries.append((handle, kind, reader.unpack(_SCALAR_VALUE)))
        elif kind == _BLOB:
            registry.entries.append((handle, kind, reader.take(reader.unpack(_U32)[0])))
        elif kind == _TENSOR_SHARD:
            registry.entries.append((handle, kind, reader.unpack(_TENSOR_SHARD_VALUE)))
        elif kind == _SHARDED_TENSOR:
            header = reader.unpack(_SHARDED_HEADER)
            registry.entries.append((handle, kind, (header, reader.take(_RANKED_SHARD.size * header[0]))))
        else:
            raise ValueError(f"{path}: node at byte {start} has unknown type {kind}")
        registry.names[handle] = name
        registry.parents[handle] = parent
    registry.trailing = len(raw) - reader.pos
    return registry


def _resolver(registry: Registry, problems: Counter) -> Callable[[int], str]:
    cache = {_ROOT: ""}

    def resolve(handle: int) -> str:
        chain: list[int] = []
        node = handle
        while node not in cache:
            if node not in registry.names or node in chain:
                problems["tree nodes with a missing or cyclic parent"] += 1
                cache[node] = "?"
                break
            chain.append(node)
            node = registry.parents[node]
        prefix = cache[node]
        for item in reversed(chain):
            prefix = f"{prefix}.{registry.names[item]}" if prefix else registry.names[item]
            cache[item] = prefix
        return cache[handle]

    return resolve


def _ndim(ndim: int) -> int:
    if ndim > _MAX_DIMS:
        raise ValueError(f"dimension count {ndim} exceeds {_MAX_DIMS}")
    return ndim


def _dims(fields: tuple, start: int, ndim: int) -> tuple[int, ...]:
    return fields[start : start + _ndim(ndim)]


def _pairs(fields: tuple, ndim: int) -> tuple[int, ...]:
    return fields[: 2 * _ndim(ndim)]


def _decode_scalar(bits: int, raw: int) -> tuple[str, int, str]:
    width = _SCALAR_WIDTHS.get(bits & 0b111, 8)
    category = (bits >> 3) & 0b11
    if category == _FLOAT:
        return f"float{8 * width}", width, repr(_DOUBLE.unpack(raw.to_bytes(8, "little"))[0])
    if category == _SIGNED:
        return f"int{8 * width}", width, str(raw - (1 << 64) if raw >> 63 else raw)
    if category == _UNSIGNED:
        return f"uint{8 * width}", width, str(raw)
    return "unknown", width, f"0x{raw:016x}"


class _Builder:
    """Aggregate registry entries into one leaf per state-dict key."""

    def __init__(self, *, expand_blobs: bool, chunk_limit: int, problems: Counter) -> None:
        self.expand_blobs = expand_blobs
        self.chunk_limit = chunk_limit
        self.keep_chunks = chunk_limit > 0
        self.kept = 0
        self.problems = problems
        self.leaves: dict[str, page.Leaf] = {}
        self.distinct: dict[str, set[int]] = {}
        self.values: dict[str, dict[str, None]] = {}

    def _leaf(self, fqn: str, kind: str, storage: str, dtype: str | None = None, shape: tuple[int, ...] | None = None) -> page.Leaf:
        leaf = self.leaves.get(fqn)
        if leaf is None:
            leaf = self.leaves[fqn] = page.Leaf(fqn, kind, dtype=dtype, shape=shape, unique=0, storage=storage)
        elif leaf.dtype != dtype or leaf.shape != shape:
            self.problems["entries whose type, dtype or shape differs between ranks"] += 1
        return leaf

    def _first(self, fqn: str, key: int) -> bool:
        seen = self.distinct.setdefault(fqn, set())
        if key in seen:
            return False
        seen.add(key)
        return True

    def _keep(self, leaf: page.Leaf, rank: int, offsets: tuple[int, ...], sizes: tuple[int, ...]) -> None:
        leaf.chunks.append((rank, offsets, sizes))
        self.kept += 1
        if self.kept > self.chunk_limit:
            self.keep_chunks = False
            for other in self.leaves.values():
                other.chunks.clear()

    def _dtype(self, code: int) -> tuple[str, int]:
        name = _DTYPES.get(code)
        if name is None:
            self.problems[f"tensors with unknown dtype code {code} (counted as 0 bytes)"] += 1
            return f"dtype{code}", 0
        return name, page.DTYPE_ITEMSIZE[name]

    def _tensor(self, fqn: str, dtype_code: int, fulldim: tuple[int, ...], storage: str) -> tuple[page.Leaf, int]:
        dtype, itemsize = self._dtype(dtype_code)
        return self._leaf(fqn, "tensor" if fulldim else "scalar", storage, dtype, fulldim), itemsize

    def _shard(self, leaf: page.Leaf, itemsize: int, pairs: tuple[int, ...], rank: int) -> None:
        offsets = pairs[0::2]
        sizes = tuple(end - begin for begin, end in zip(offsets, pairs[1::2]))
        nbytes = math.prod(sizes) * itemsize if sizes else 0
        leaf.add(rank, nbytes, sizes)
        if self._first(leaf.fqn, hash(pairs)):
            leaf.unique += nbytes
        if self.keep_chunks:
            self._keep(leaf, rank, offsets, sizes)

    def tensor_shard(self, fqn: str, fields: tuple, rank: int) -> None:
        leaf, itemsize = self._tensor(fqn, fields[14], _dims(fields, 9, fields[13]), "tensor shards from per-rank registries")
        self._shard(leaf, itemsize, _pairs(fields, fields[8]), rank)

    def sharded_tensor(self, fqn: str, header: tuple, shards: memoryview) -> None:
        leaf, itemsize = self._tensor(fqn, header[1], _dims(header, 2, header[6]), "sharded tensor in the global registry")
        for record in _RANKED_SHARD.iter_unpack(shards):
            rank = page.UNKNOWN_RANK if record[9] == _INVALID_RANK else record[9]
            self._shard(leaf, itemsize, _pairs(record, record[8]), rank)

    def blob(self, fqn: str, data: memoryview, rank: int, storage: str) -> None:
        leaf = self._leaf(fqn, "blob", storage)
        leaf.add(rank, len(data), None)
        if self._first(fqn, 0):
            leaf.unique += len(data)
        if self.expand_blobs and leaf.blob is None and len(data):
            leaf.blob = page.describe_blob(bytes(data))
            if leaf.blob.get("k") == "scalar" and "err" not in leaf.blob:
                leaf.kind = "scalar"

    def scalar(self, fqn: str, value: tuple[int, int], rank: int, storage: str) -> None:
        label, width, text = _decode_scalar(*value)
        leaf = self._leaf(fqn, "scalar", storage)
        leaf.add(rank, width, None)
        leaf.rank_labels[rank] = f"= {text}"
        if self._first(fqn, 0):
            leaf.unique += width
        values = self.values.setdefault(fqn, {})
        values[text] = None
        shown = text if len(values) == 1 else f"{len(values)} distinct values across ranks"
        leaf.blob = {"k": "scalar", "t": label, "b": width, "v": shown}

    def add(self, registry: Registry, *, per_rank: bool) -> None:
        resolve = _resolver(registry, self.problems)
        rank = registry.rank if per_rank else page.UNKNOWN_RANK
        where = "per-rank registries" if per_rank else "the global registry"
        for handle, kind, payload in registry.entries:
            fqn = resolve(handle)
            if kind == _TENSOR_SHARD:
                self.tensor_shard(fqn, payload, registry.rank)
            elif kind == _SHARDED_TENSOR:
                self.sharded_tensor(fqn, *payload)
            elif kind == _BLOB:
                self.blob(fqn, payload, rank, f"inline blob bytes in {where}")
            else:
                self.scalar(fqn, payload, rank, f"64-bit scalar in {where}")


def _discover(inputs: list[Path]) -> tuple[list[Path], list[Path]]:
    per_rank: list[Path] = []
    merged: list[Path] = []
    for item in inputs:
        if item.is_dir():
            children = sorted(item.iterdir())
            per_rank.extend(child for child in children if _LOCAL_NAME.fullmatch(child.name))
            merged.extend(child for child in children if child.name in _GLOBAL_NAMES)
        elif item.is_file():
            (per_rank if _LOCAL_NAME.fullmatch(item.name) else merged).append(item)
        else:
            raise FileNotFoundError(f"{item} does not exist")
    per_rank.sort(key=lambda path: (int(_LOCAL_NAME.fullmatch(path.name).group(1)), str(path)))
    return per_rank, merged


def _read_all(paths: list[Path]) -> Iterator[Registry]:
    with ThreadPoolExecutor(max_workers=_PREFETCH) as pool:
        pending: deque = deque()
        for path in paths:
            pending.append((path, pool.submit(path.read_bytes)))
            if len(pending) >= _PREFETCH:
                ready, future = pending.popleft()
                yield read_registry(ready, future.result())
        while pending:
            ready, future = pending.popleft()
            yield read_registry(ready, future.result())


def _common_parent(paths: list[Path]) -> Path:
    parents = {path.resolve().parent for path in paths}
    return parents.pop() if len(parents) == 1 else Path.cwd().resolve()


def _select(per_rank: list[Path], merged: list[Path], warnings: list[str]) -> list[Path]:
    if per_rank:
        if merged:
            warnings.append(f"ignored {len(merged)} global registry file(s) because per-rank registries were given")
        return per_rank
    if len(merged) == 1:
        return merged
    if merged:
        raise ValueError(f"pass one global registry at a time, got {len(merged)}: {', '.join(str(path) for path in merged)}")
    raise ValueError("no registry.<rank>.bin, registry.rank<rank>.bin or registry.bin files found")


def _check_rank(registry: Registry, seen: dict[int, Path], warnings: list[str]) -> None:
    named = int(_LOCAL_NAME.fullmatch(registry.path.name).group(1))
    if named != registry.rank:
        warnings.append(f"{registry.path.name} holds the registry of rank {registry.rank}")
    if registry.rank in seen:
        raise ValueError(f"{registry.path} and {seen[registry.rank]} both hold rank {registry.rank}")
    seen[registry.rank] = registry.path


def _describe_global(registry: Registry, notes: list[str], warnings: list[str]) -> None:
    notes.append(
        f"This is the global registry: it holds every rank's tensor shards, but its blob and scalar nodes come "
        f"from the first rank to register (rank {registry.rank}), carry no rank, and its blobs hold no bytes.",
    )
    kinds = Counter(kind for _, kind, _ in registry.entries)
    if kinds[_BLOB]:
        warnings.append(f"the global registry's {kinds[_BLOB]} blobs hold no bytes and no rank; per-rank registries hold their sizes")
    if kinds[_SCALAR]:
        warnings.append(f"the global registry's {kinds[_SCALAR]} scalars carry rank {registry.rank}'s values and no rank")


def analyze(
    inputs: list[Path | str],
    *,
    expand_blobs: bool = False,
    chunk_details_limit: int = page.DEFAULT_CHUNK_DETAILS_LIMIT,
) -> page.Report:
    """Read per-rank or global registries into a distcp_tree report."""
    per_rank, merged = _discover([Path(item) for item in inputs])
    warnings: list[str] = []
    files = _select(per_rank, merged, warnings)
    notes = [_SIZES_NOTE]
    problems: Counter = Counter()
    builder = _Builder(expand_blobs=expand_blobs, chunk_limit=chunk_details_limit, problems=problems)
    seen: dict[int, Path] = {}
    world_sizes: set[int] = set()
    first: Registry | None = None
    total_bytes = 0
    for registry in _read_all(files):
        first = first or registry
        total_bytes += registry.size
        world_sizes.add(registry.world_size)
        if registry.trailing:
            problems["registry files with trailing bytes after the tree"] += 1
        if per_rank:
            _check_rank(registry, seen, warnings)
        else:
            _describe_global(registry, notes, warnings)
        builder.add(registry, per_rank=bool(per_rank))
    world = max(world_sizes)
    if len(world_sizes) > 1:
        warnings.append(f"registries disagree on the world size: {sorted(world_sizes)}")
    if per_rank and len(seen) < world:
        warnings.append(f"per-rank registries cover {len(seen)} of {world} ranks")
    warnings.extend(f"{count} {problem}" for problem, count in sorted(problems.items()) if count)
    return page.Report(
        source=_common_parent(files),
        leaves=list(builder.leaves.values()),
        ranks=world,
        files=len(files),
        metadata_bytes=total_bytes,
        expanded=expand_blobs,
        warnings=warnings,
        input_label="Registries",
        notes=notes,
        hosts=first.peers,
        ranks_per_host=first.ranks_per_peer,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("paths", nargs="+", type=Path, help="registry files, or directories holding them")
    parser.add_argument("-o", "--output", type=Path, help="HTML file to write (default: ./<input folder name>.registry_tree.html)")
    parser.add_argument("--expand-blobs", action="store_true", help="decode and expand the blob bytes held in per-rank registries")
    parser.add_argument(
        "--chunk-details-limit",
        type=int,
        default=page.DEFAULT_CHUNK_DETAILS_LIMIT,
        help="embed per-shard offsets and sizes when the registries hold at most this many tensor shards (0 disables)",
    )
    parser.add_argument("--title", help="page title (default: the last two components of the input folder)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze(args.paths, expand_blobs=args.expand_blobs, chunk_details_limit=args.chunk_details_limit)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    title = args.title or "/".join(report.source.parts[-2:])
    doc, binary = page.encode_report(report, title=title, chunk_details_limit=args.chunk_details_limit)
    output: Path = args.output or Path(f"{report.source.name}.registry_tree.html")
    output.write_text(page.render_html(doc, binary), encoding="utf-8")
    summary = doc["summary"]
    sys.stdout.write(
        f"{output}: {report.files} registries, {summary['leaves']} entries ({summary['tensors']} tensors, "
        f"{summary['scalars']} scalars, {summary['blobs']} blobs), {summary['chunks']} shards, {summary['ranks']} ranks, "
        f"{summary['totalBytes']} registered bytes ({summary['uniqueBytes']} unique)\n",
    )
    for warning in report.warnings:
        sys.stderr.write(f"warning: {warning}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
