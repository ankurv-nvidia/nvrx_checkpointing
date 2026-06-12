#! /usr/bin/python3
import argparse
import difflib
import enum
import hashlib
import io
import json
import pickle
import re
import sys
import time
import types
import zipfile
from pathlib import Path
from typing import IO

import torch
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex

from . import mapping


def transform_complex_obj(obj: object, visited: set[object] | None = None) -> str | dict[str, object] | list[object]:  # noqa: C901, PLR0911, PLR0912
    if visited is None:
        visited = set()
    if (type(obj) in (int, float, str, bool)) or obj is None:
        return str(obj)
    try:
        if obj in visited:
            return "*"
        visited.add(obj)
    except TypeError:
        pass
    if isinstance(obj, mapping.LocalNonpersistentObject):
        return {"#local_nonpersistent_object#": vars(obj)}  #    return transform_complex_obj(vars(obj), visited)
    if isinstance(obj, mapping.ShardedTensor):
        return f"ShardedTensor(key={obj.key}, tensor={transform_complex_obj(obj.data)})"
    if isinstance(obj, mapping.ShardedTensorFactory):
        return f"ShardedTensorFactory(key={obj.key}, tensor={transform_complex_obj(obj.data)})"
    if isinstance(obj, mapping.ShardedObject):
        return {"#sharded_object#": vars(obj)}
    # check if tuple of ints
    if isinstance(obj, tuple) and all(isinstance(i, int) for i in obj):
        return str(obj)
    if isinstance(obj, tuple) and all(isinstance(i, str) for i in obj):
        return str(obj)
    if isinstance(obj, tuple) and all(isinstance(i, float) for i in obj):
        return str(obj)
    if (
        isinstance(obj, tuple)
        and len(obj) == 3  # noqa: PLR2004
        and isinstance(obj[0], int)
        and isinstance(obj[1], tuple)
        and obj[2] is None
        and all(isinstance(i, int) for i in obj[1])
    ):
        return "#python_rng_state#"
    if isinstance(obj, torch.Generator):
        return f"Generator(device={obj.device})"
    # check if its a class
    # if isinstance(obj, type):
    #    return f"#class#{obj.__name__}"
    if isinstance(obj, slice):
        return f"slice(start={obj.start}, stop={obj.stop}, step={obj.step})"
    # check if torch.float32, torch.int64, etc.
    if isinstance(obj, torch.dtype):
        return str(obj)
    # check if enum
    if isinstance(obj, enum.Enum):
        return f"Enum[{obj.__class__.__name__}]({obj.value})"
    if isinstance(obj, (argparse.Namespace, types.SimpleNamespace)):
        return transform_complex_obj(vars(obj), visited)
        # return "#ShardedObject()#"    #    return transform_complex_obj(vars(obj), visited)
    # Handle lists recursively
    if isinstance(obj, list) and all(isinstance(i, int) for i in obj):
        return str(obj)
    if isinstance(obj, list) and all(isinstance(i, str) for i in obj):
        return str(obj)
    if isinstance(obj, list) and all(isinstance(i, float) for i in obj):
        return str(obj)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, torch.Tensor):
        return f"Tensor(shape={obj.shape}, dtype={obj.dtype})"
    # Handle dicts recursively
    if isinstance(obj, dict):
        return obj
    # Handle custom objects with __dict__
    # if hasattr(obj, "__dict__"):
    #    return {
    #        str(k): transform_complex_obj(v, visited) for k, v in (vars(obj)).items()
    # Fallback: convert to string representation
    raise RuntimeError(f"{type(obj).__name__}::({obj!s})")


def print_sharded_state_dict_recursively(prefix: str, value: object, f: IO[str]) -> None:
    value = transform_complex_obj(value)
    if isinstance(value, dict):
        for key, v in value.items():
            if type(key) is not str and type(key) is not int:
                continue
            if isinstance(v, str):
                f.write(f"{prefix}.{key} = {v}\n")
            else:
                print_sharded_state_dict_recursively(f"{prefix}.{key}", v, f)
    elif isinstance(value, list):
        for key, v in enumerate(value):
            if isinstance(v, str):
                f.write(f"{prefix}.{key} = {v}\n")
            else:
                print_sharded_state_dict_recursively(f"{prefix}.{key}", v, f)
    elif isinstance(value, str):
        f.write(f"{prefix} = {value}\n")
    else:
        raise TypeError(f"Unexpected type {prefix} {type(value)}")


class SaferJSONEncoder(json.JSONEncoder):
    def default(self, obj: object) -> str | dict[str, object] | list[object]:
        return transform_complex_obj(obj)


def dump_state_dir(checkpoint_dir: str, sharded_state_dict: mapping.ShardedStateDict, suffix: str) -> None:
    with (Path(checkpoint_dir) / f"sharded_state_dict.{suffix}.txt").open("w") as _f:
        print_sharded_state_dict_recursively("", sharded_state_dict, _f)


# import torch
try:
    import megatron  # pyright: ignore[reportMissingImports]
    import megatron.core  # pyright: ignore[reportMissingImports]
except ImportError:
    from scripts.import_stubs import megatron
    from scripts.import_stubs.megatron import core as mcore

    # megatron_root_path = Path(__file__).parent.parent.parent.parent
    # if str(megatron_root_path) not in sys.path:
    #    sys.path.append(str(megatron_root_path))
    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = mcore


def flatten_to_entries(prefix: str, obj: object, entries: dict, visited: set[int] | None = None) -> None:  # noqa: C901, PLR0912
    """Recursively flatten an object into dot-separated key=value pairs.

    Dicts and objects with __dict__ are descended into; lists/tuples are
    indexed numerically (.0, .1, …); torch.Size is rendered as a plain list.
    Everything else is converted with str(). `visited` tracks ids along the
    current recursion path so cycles get short-circuited instead of looping.
    """
    if visited is None:
        visited = set()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        entries[prefix] = str(obj)
        return
    obj_id = id(obj)
    if obj_id in visited:
        entries[prefix] = "<cycle>"
        return
    visited.add(obj_id)
    try:
        if isinstance(obj, enum.Enum):
            entries[prefix] = f"Enum[{obj.__class__.__name__}]({obj.value})"
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten_to_entries(f"{prefix}.{k}", v, entries, visited)
        elif isinstance(obj, torch.Size):
            entries[prefix] = str(list(obj))
        elif isinstance(obj, (list, tuple)):
            # Homogeneous scalar sequences collapse to a single string value;
            # heterogeneous or nested sequences get per-element keys.
            if all(isinstance(i, (int, float, str, bool)) for i in obj):
                entries[prefix] = str(list(obj))
            else:
                for i, item in enumerate(obj):
                    flatten_to_entries(f"{prefix}.{i}", item, entries, visited)
        elif hasattr(obj, "__dict__") and vars(obj):
            for k, v in vars(obj).items():
                flatten_to_entries(f"{prefix}.{k}", v, entries, visited)
        else:
            entries[prefix] = str(obj)
    finally:
        visited.discard(obj_id)


def try_ping() -> None:
    try:
        import megatron.native.checkpoint  # noqa: PLC0415

        megatron.native.checkpoint.ping()
    except ImportError:
        pass


def svc_stop() -> None:
    try:
        import megatron.native.checkpoint  # noqa: PLC0415

        megatron.native.checkpoint.svc_stop()
    except ImportError:
        pass


def svc_spindown() -> None:
    """Gracefully spin the checkpoint server down, blocking until it is idle.

    Unlike svc_stop this does not abort in-flight saves on other ranks; the server stops
    only once every session is idle. `timeout` bounds the wait before falling back to a
    hard stop; None lets the server apply its own default bound.
    """
    try:
        import megatron.native.checkpoint  # noqa: PLC0415

        megatron.native.checkpoint.svc_spindown()
    except ImportError:
        pass


def distcp_trace(  # noqa: C901, PLR0912, PLR0915
    checkpoint_dir: Path | str,
    extract_tensor_data: bool,
    *,
    verify_native_server: bool = True,
) -> None:
    # distcp_trace only reads on-disk files (the .distcp chunks for md5s, and the server's
    # 0.metadata.info for the claim check below) — it never talks to a live server. So only
    # ensure one is up when we actually verify against it; the md5 / cross-provider path
    # (verify_native_server=False) must not launch a server it would then strand on the node.
    if verify_native_server:
        try_ping()
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")
    commonpt = checkpoint_dir / "common.pt"
    metadata_file = checkpoint_dir / ".metadata"
    if not commonpt.exists():
        raise FileNotFoundError(f"common.pt file does not exist in checkpoint directory {checkpoint_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist in checkpoint directory {checkpoint_dir}")
    with metadata_file.open("rb") as fd:
        metadata = pickle.load(fd)  # noqa: S301
    common_pt_data = torch.load(f"{checkpoint_dir}/common.pt", map_location="cpu", weights_only=False)

    common_pt_entries: dict[str, str] = {}
    for k, v in common_pt_data.items():
        flatten_to_entries(str(k), v, common_pt_entries)
    with (checkpoint_dir / "common_pt_dump.info").open("w") as _f:
        _f.write("\n".join(f"{k}={common_pt_entries[k]}" for k in sorted(common_pt_entries)))

    file_offsets = {}
    for metadata_index, storage_info in metadata.storage_data.items():
        if not isinstance(metadata_index, MetadataIndex):
            raise TypeError(f"Expected MetadataIndex, got {type(metadata_index)}")
        if not isinstance(storage_info, _StorageInfo):
            raise TypeError(f"Expected _StorageInfo, got {type(storage_info)}")
        file_offsets.setdefault(storage_info.relative_path, {})
        file_offsets[storage_info.relative_path][storage_info.offset] = {"storage_info": storage_info, "metadata_index": metadata_index}

    info_entries: dict[str, str] = {}
    tensors = {}
    # Non-contiguous offsets are expected today: every rank writes the same archive for a
    # replicated shard, but only one rank's _StorageInfo survives torch's MetadataIndex dict
    # on pickle.load, leaving orphan bytes between metadata-tracked entries. With multiple
    # nodes the canonical _StorageInfo for a replicated shard can land in another rank's file,
    # so a file's own first metadata-tracked entry may start past offset 0 — the bytes before
    # it are that same kind of orphan. Both cases flood the log one-per-shard, so accumulate
    # the leading and interior gaps together and emit a single summary after the walk. dcp.load
    # seeks by the recorded offset and never reads the orphan bytes, so tensors round-trip; the
    # md5 / cross-provider check is the gate that fails on real offset-vs-data divergence.
    gap_count = 0
    gap_bytes = 0
    files_with_gaps: set[str] = set()
    for relative_path, offsets in file_offsets.items():
        sorted_offsets = dict(sorted(offsets.items(), key=lambda item: item[0]))
        start_offset = 0
        for offset, info in sorted_offsets.items():
            tnsr = info["metadata_index"].fqn
            if offset != start_offset:
                gap_count += 1
                gap_bytes += offset - start_offset
                files_with_gaps.add(relative_path)
                start_offset = offset
            tensors.setdefault(tnsr, [])
            tensors[tnsr].append(info)
            start_offset += info["storage_info"].length

    if gap_count:
        # TODO(ankurv/megatron-lm#6): these gaps disappear once the native writer dedups
        # replicated tensors at registration time. dcp.load seeks by the recorded offset and
        # ignores the orphan bytes, so tensors still round-trip correctly — warn, don't raise.
        sys.stderr.write(
            f"WARN: {gap_count} non-contiguous tensor offset(s) across {len(files_with_gaps)} file(s) "
            f"(+{gap_bytes} orphan bytes from replicated shards); tensors still round-trip. "
            f"TODO(ankurv/megatron-lm#6): remove once the native writer dedups replicated tensors.\n",
        )

    tensormeta = {}
    chunkmeta = {}
    bytemeta = {}
    for path_str, meta_obj in metadata.state_dict_metadata.items():
        if isinstance(meta_obj, TensorStorageMetadata):
            tensormeta[path_str] = meta_obj
        elif isinstance(meta_obj, ChunkStorageMetadata):
            chunkmeta[path_str] = meta_obj
        elif isinstance(meta_obj, BytesStorageMetadata):
            bytemeta[path_str] = meta_obj
        else:
            raise TypeError(f"relative_path={relative_path}: {path_str}: {meta_obj} : Unsupported metadata type {type(meta_obj)}")
    if len(tensormeta) + len(chunkmeta) + len(bytemeta) != len(tensors):
        sys.stderr.write(
            f"relative_path={relative_path}: "
            f"Total count from storage offsets {len(tensors)} does not match "
            f"len(tensormeta)[{len(tensormeta)}]+ len(chunkmeta)[{len(chunkmeta)}] + len(bytemeta)[{len(bytemeta)}] = "
            f"{len(tensormeta) + len(chunkmeta) + len(bytemeta)}\n",
        )
        # find missing keys
        for tnsr in tensors:
            if tnsr not in tensormeta and tnsr not in chunkmeta and tnsr not in bytemeta:
                raise TypeError(f"\tMissing metadata for tensor {tnsr}\n")
        for tnsr, obj in tensormeta.items():
            if tnsr not in tensors:
                sys.stderr.write(f"\tMetadata tensor {tnsr} : {obj} not found in storage offsets\n")
        for tnsr, obj in chunkmeta.items():
            if tnsr not in tensors:
                sys.stderr.write(f"\tMetadata chunk {tnsr} : {obj} not found in storage offsets\n")
        for tnsr, obj in bytemeta.items():
            if tnsr not in tensors:
                sys.stderr.write(f"\tMetadata byte {tnsr} : {obj} not found in storage offsets\n")

    for relative_path in file_offsets:
        tnsr_data_dcp_file = checkpoint_dir / relative_path

        # Build chunk_storage: (fqn, chunk_offsets_tuple) -> storage_info
        # Used to locate each chunk's raw bytes within the distcp file.
        chunk_storage: dict[tuple, object] = {}
        for midx, si in metadata.storage_data.items():
            if isinstance(midx, MetadataIndex):
                offset_key = tuple(midx.offset) if midx.offset is not None else ()
                chunk_storage[(midx.fqn, offset_key)] = si
        with tnsr_data_dcp_file.open("rb") as dcp_fd:
            # Write per-tensor raw bytes (unchanged behaviour)
            extracted_dir = checkpoint_dir / "reference"
            if extract_tensor_data:
                extracted_dir.mkdir(exist_ok=True)
            # Build flat key=value metadata entries for all TensorStorageMetadata tensors
            if verify_native_server:
                try_ping()
            for fqn, meta in tensormeta.items():
                flatten_to_entries(fqn, meta, info_entries)
                for i, chunk in enumerate(meta.chunks):
                    si = chunk_storage[(fqn, tuple(chunk.offsets))]
                    if si.relative_path != relative_path:
                        continue
                    dcp_fd.seek(si.offset)
                    raw = dcp_fd.read(si.length)
                    if len(raw) != si.length:
                        raise ValueError(
                            f"[native_ckpt:cli] {relative_path}:{fqn}.chunk.{i} offset={si.offset}: Got {len(raw)} of {si.length} bytes\n",
                        )
                    data_size = chunk.sizes.numel() * torch.tensor([], dtype=meta.properties.dtype).element_size()
                    # Each si.length chunk is a complete torch.save() ZIP64 archive.
                    # Raw tensor bytes live in the 'archive/data/0' member; the ZIP
                    # local-file-header + data.pkl + ZIP footer surround them.
                    with zipfile.ZipFile(io.BytesIO(raw)) as _zip, _zip.open("archive/data/0") as _f:
                        raw_data = _f.read()
                    if len(raw_data) != data_size:
                        raise ValueError(
                            f"[native_ckpt:cli] WARNING: Expected {data_size} raw bytes, got {len(raw_data)} bytes for {fqn}[chunk#{i}]\n"
                            f"\tchunk_size={si.length} bytes\n",
                        )
                    md5hex = hashlib.md5(raw_data).hexdigest()  # noqa: S324
                    info_entries[f"{fqn}.chunks.{i}.md5sum"] = md5hex
                    datafile = extracted_dir / f"{fqn}.chunks.{i}.dat"
                    # info_entries[f"{fqn}.chunks.{i}.size"] = str(si.length)
                    if extract_tensor_data:
                        datafile.unlink() if datafile.exists() else None
                        with datafile.open("ab") as fd:
                            fd.write(raw_data)

            # ShardedObject (BytesStorageMetadata) entries: raw torch.save bytes,
            # not a ZIP archive. md5 the content (content-based, so it matches across
            # providers/runs regardless of which file/offset the bytes live at).
            #
            # Restricted to TransformerEngine `_extra_state`: it is computed from the
            # model and is byte-identical across providers. The other objects — the
            # distributed-optimizer state dict and rng_state — are not byte-identical
            # across checkpoint backends (the native server and dist_checkpointing
            # consume rng and serialize the optimizer dict differently), so tracing
            # them would spuriously fail the cross-provider compare.
            for fqn in bytemeta:
                if "_extra_state" not in fqn:
                    continue
                si = chunk_storage.get((fqn, ()))
                if si is None or si.relative_path != relative_path:
                    continue
                dcp_fd.seek(si.offset)
                raw = dcp_fd.read(si.length)
                if len(raw) != si.length:
                    raise ValueError(
                        f"[native_ckpt:cli] {relative_path}:{fqn} offset={si.offset}: Got {len(raw)} of {si.length} bytes\n",
                    )
                info_entries[f"{fqn}.bytes.md5sum"] = hashlib.md5(raw).hexdigest()  # noqa: S324

    ref_info_file = checkpoint_dir / "reference.metadata.info"
    rflines = [f"{k}={info_entries[k]}" for k in sorted(info_entries.keys())]
    ref_info_file.write_text("\n".join(rflines))

    # Internal consistency check: when this dir holds native checkpoint_server
    # output, the server emits {peerIndex}.metadata.info — what it *claims* to
    # have written. Diff that claim against the truth we just derived from the
    # on-disk files. Callers pointed at the nvrx side of a dual-provider run
    # pass verify_native_server=False to skip this; the cross-provider diff
    # then runs via compare_reference_metadata.
    if verify_native_server:
        native_info_file = checkpoint_dir / "0.metadata.info"
        while not native_info_file.exists():
            time.sleep(1)
        nflines = native_info_file.read_text().splitlines()
        diff = list(
            difflib.unified_diff(
                rflines,
                nflines,
                fromfile=str(ref_info_file),
                tofile=str(native_info_file),
                n=1,
            ),
        )
        sys.stderr.write(
            f"[native_ckpt:cli] {'PASS' if len(diff) == 0 else 'FAIL'}: {len(diff)} differences\n\t{ref_info_file}\n\t{native_info_file}\n",
        )
        if len(diff) < len(nflines) / 2:
            sys.stdout.writelines("\n".join(diff))
        if len(diff) > 0:
            raise ValueError(f"[native_ckpt:cli] FAIL: {len(diff)} Differences found between {ref_info_file} and {native_info_file}")
        try_ping()


_CHUNK_KEY_RE = re.compile(r"^(?P<fqn>.+)\.chunks\.(?P<idx>\d+)\.(?P<field>\w+)$")


def _parse_reference_metadata(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], tuple[tuple[str, str], ...]]]:
    """Split a ``reference.metadata.info`` dump into scalar entries and per-chunk fields.

    Chunk lines have the form ``<fqn>.chunks.<n>.<field>`` where ``<n>`` is the
    chunk's position in the writer's chunk list. That position reflects how a
    given writer happens to enumerate a tensor's chunks — and a multinode native
    dump lists a chunk once per rank that holds it, so a replicated chunk appears
    several times — so it is not a stable identity. Re-key each chunk by
    ``(fqn, offsets)`` (its location in the full tensor) and carry the remaining
    fields (sizes, md5sum) as the value; replicas of one chunk collapse to a
    single entry. Every other ``key=value`` line is returned verbatim.
    """
    scalars: dict[str, str] = {}
    chunk_fields: dict[tuple[str, str], dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        m = _CHUNK_KEY_RE.match(key)
        if m is None:
            scalars[key] = value
        else:
            chunk_fields.setdefault((m["fqn"], m["idx"]), {})[m["field"]] = value
    chunks: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for (fqn, _idx), fields in chunk_fields.items():
        offsets = fields.get("offsets", "")
        rest = tuple(sorted((f, v) for f, v in fields.items() if f != "offsets"))
        chunks[(fqn, offsets)] = rest
    return scalars, chunks


def compare_reference_metadata(checkpoint_dir: Path | str, baseline_dir: Path | str) -> int:
    """Diff reference.metadata.info between a checkpoint dir and its baseline snapshot.

    Each dir must already contain a ``reference.metadata.info`` produced by
    :func:`distcp_trace` — independent dumps of what's on disk (tensor
    properties + per-chunk md5sums) for that writer's output.

    Tensor properties are compared by key; per-chunk entries are compared by
    chunk identity ``(fqn, offsets)`` rather than by list position. A divergence
    is reported only when a chunk's recorded data (md5sum/sizes) differs, or when
    one side is missing a chunk the other has — not when two writers merely
    enumerate or replicate the same chunks in a different order. This keeps the
    cross-provider equality contract meaningful across the native and nvrx
    writers, which list chunks in their own orders, and across single- vs
    multinode native runs, where replicated chunks are written once per node.

    Returns the number of differing entries — 0 means equal. Callers decide
    whether to warn or hard-fail; see ``scripts/metadata_compare.py``.

    Raises:
        FileNotFoundError: when either dir is missing reference.metadata.info.
    """
    checkpoint_dir = Path(checkpoint_dir)
    baseline_dir = Path(baseline_dir)
    checkpoint_ref = checkpoint_dir / "reference.metadata.info"
    baseline_ref = baseline_dir / "reference.metadata.info"
    for label, path in (("checkpoint", checkpoint_ref), ("baseline", baseline_ref)):
        if not path.is_file():
            raise FileNotFoundError(f"[native_ckpt:cli] missing {label} reference.metadata.info at {path}")
    base_scalars, base_chunks = _parse_reference_metadata(baseline_ref)
    ckpt_scalars, ckpt_chunks = _parse_reference_metadata(checkpoint_ref)

    findings: list[str] = []
    for key in sorted(set(base_scalars) | set(ckpt_scalars)):
        if base_scalars.get(key) != ckpt_scalars.get(key):
            findings.append(f"  {key}: baseline={base_scalars.get(key)!r} checkpoint={ckpt_scalars.get(key)!r}")
    for ident in sorted(set(base_chunks) | set(ckpt_chunks)):
        if base_chunks.get(ident) != ckpt_chunks.get(ident):
            fqn, offsets = ident
            findings.append(f"  {fqn} chunk offsets={offsets}: baseline={base_chunks.get(ident)} checkpoint={ckpt_chunks.get(ident)}")

    result = "PASS" if not findings else "FAIL"
    sys.stderr.write(
        f"[native_ckpt:cli] {result}: {len(findings)} differences vs baseline\n\t{baseline_ref}\n\t{checkpoint_ref}\n",
    )
    if findings:
        sys.stderr.write("\n".join(findings[:50]) + "\n")
    return len(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse distcp trace and create checkpoint metadata")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to the distcp trace file")
    parser.add_argument("--output-dir", type=Path, required=False, help="Path to save the checkpoint metadata")
    parser.add_argument("--extract-tensor-data", action="store_true", help="Whether to extract raw tensor data from the distcp trace")
    args = parser.parse_args()
    distcp_trace(args.checkpoint_dir, args.extract_tensor_data)


if __name__ == "__main__":
    main()
