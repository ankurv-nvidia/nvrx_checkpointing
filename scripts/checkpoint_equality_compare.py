"""Equality-compare a native checkpoint directory against an nvrx one.

Two policies — tensors are checked strictly; ``common.pt`` keeps the
asymmetric carve-out for the binding-side gap (enum / dtype / Namespace
coverage in ``bindings.cpp``) until that gap closes.

Tensors (every FQN under each side's ``.metadata``):

==========================  =============================================
Situation                   Outcome
==========================  =============================================
present in both, equal      pass
present in both, differ     FAIL  (exit 1)
present in nvrx only        FAIL  (exit 1)
present in native only      FAIL  (exit 1)
==========================  =============================================

``common.pt`` (recursed through dict / ``Namespace`` / list / tuple /
scalar leaves):

==========================  =============================================
Situation                   Outcome
==========================  =============================================
present in both, equal      pass
present in both, differ     FAIL  (exit 1)
present in nvrx only        WARN  (logged to stderr, still exit 0)
present in native only      FAIL  (exit 1)
==========================  =============================================

Exit codes:

* ``0`` — tensors are bit-identical and any ``common.pt`` divergences
  are limited to "missing on native" (binding-gap-expected) warnings.
* ``1`` — any tensor divergence, or any "extra on native" / value
  mismatch in ``common.pt``.

CLI::

    python -m nvrx_checkpointing.scripts.checkpoint_equality_compare \\
        --native <ckpt_dir>/native \\
        --nvrx   <ckpt_dir>/nvrx
"""

from __future__ import annotations

import argparse
import io
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, TensorStorageMetadata


class _Report:
    """Accumulate per-finding counts while walking the two checkpoints."""

    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def warn(self, msg: str) -> None:
        print(f"WARN: {msg}", file=sys.stderr)  # noqa: T201
        self.warnings += 1

    def fail(self, msg: str) -> None:
        print(f"FAIL: {msg}", file=sys.stderr)  # noqa: T201
        self.failures += 1


def _load_tensors(ckpt_dir: Path) -> tuple[dict[str, torch.Tensor], Metadata]:
    """Load every tensor in `ckpt_dir` into a flat dict via `dcp.load`.

    Builds a zero-filled template state_dict from the metadata so that
    `dcp.load` populates every FQN — we have no caller-side state_dict to
    reuse. Single-process invocation is fine: `dcp.load` falls back to local
    I/O when `torch.distributed` is uninitialized and emits one benign
    warning. The raw `Metadata` is returned alongside so callers can map
    each FQN back to its on-disk shard locations.
    """
    reader = FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()
    template: dict[str, torch.Tensor] = {}
    for fqn, meta in metadata.state_dict_metadata.items():
        # Current Megatron stores common state as BytesStorageMetadata while
        # legacy/native checkpoints use common.pt. Common state is loaded in a
        # separate pass below, so only tensor entries belong in this template.
        if isinstance(meta, TensorStorageMetadata):
            template[fqn] = torch.zeros(meta.size, dtype=meta.properties.dtype)
    dcp.load(template, checkpoint_id=str(ckpt_dir))
    return template, metadata


def _shard_locations(metadata: Metadata, fqn: str) -> list[str]:
    """Render one line per shard of `fqn`: seq index, distcp file, file offset."""
    meta = metadata.state_dict_metadata.get(fqn)
    if not isinstance(meta, TensorStorageMetadata):
        return []
    storage_data = metadata.storage_data or {}
    lines: list[str] = []
    for idx, chunk in enumerate(meta.chunks):
        info = storage_data.get(MetadataIndex(fqn, chunk.offsets, idx))
        if info is None:
            lines.append(
                f"seq={idx} offsets={tuple(chunk.offsets)} sizes={tuple(chunk.sizes)} (no storage info)",
            )
        else:
            lines.append(
                f"seq={idx} file={info.relative_path} "
                f"file_offset={info.offset} length={info.length} "
                f"offsets={tuple(chunk.offsets)} sizes={tuple(chunk.sizes)}",
            )
    return lines


def _load_common(ckpt_dir: Path) -> dict:
    commonpt = ckpt_dir / "common.pt"
    if commonpt.is_file():
        return torch.load(commonpt, map_location="cpu", weights_only=False)

    key = "common_state/shard_0_1"
    state_dict = {key: io.BytesIO()}
    dcp.load(
        state_dict,
        storage_reader=FileSystemReader(str(ckpt_dir)),
        no_dist=True,
    )
    loaded = state_dict[key]
    if isinstance(loaded, io.BytesIO):
        loaded.seek(0)
        loaded = torch.load(loaded, map_location="cpu", weights_only=False)
    if not isinstance(loaded, list) or len(loaded) != 1:
        raise TypeError(f"Expected one embedded common-state object, got {type(loaded).__name__}")
    common_state = loaded[0]
    if not isinstance(common_state, dict):
        raise TypeError(f"Expected common state dict, got {type(common_state).__name__}")
    return common_state


def _compare_tensors(
    nvrx: dict[str, torch.Tensor],
    native: dict[str, torch.Tensor],
    nvrx_meta: Metadata,
    native_meta: Metadata,
    report: _Report,
) -> None:
    nvrx_keys = set(nvrx)
    native_keys = set(native)
    # Tensors are not subject to the binding-side gap that makes common.pt
    # asymmetric — both providers consume the same in-memory state_dict, so
    # any divergence in either direction is a real failure.
    for fqn in sorted(nvrx_keys - native_keys):
        report.fail(f"missing on native side: tensor:{fqn}")
    for fqn in sorted(native_keys - nvrx_keys):
        report.fail(f"extra on native side: tensor:{fqn}")
    for fqn in sorted(nvrx_keys & native_keys):
        a, b = nvrx[fqn], native[fqn]
        if a.dtype != b.dtype or a.shape != b.shape:
            report.fail(
                f"tensor metadata mismatch: tensor:{fqn} (nvrx={a.dtype}{tuple(a.shape)}, native={b.dtype}{tuple(b.shape)})",
            )
        elif not torch.equal(a, b):
            nvrx_shards = _shard_locations(nvrx_meta, fqn)
            native_shards = _shard_locations(native_meta, fqn)
            detail = ["  nvrx shards:"] + [f"    {s}" for s in nvrx_shards]
            detail += ["  native shards:"] + [f"    {s}" for s in native_shards]
            report.fail(
                f"tensor value mismatch: tensor:{fqn}\n" + "\n".join(detail),
            )


def _is_namespace(x: Any) -> bool:
    return isinstance(x, (argparse.Namespace, types.SimpleNamespace))


def _compare_namespace(nvrx: Any, native: Any, report: _Report, path: str) -> None:
    if type(nvrx) is not type(native):
        report.fail(
            f"type mismatch at {path}: nvrx={type(nvrx).__name__}, native={type(native).__name__}",
        )
        return
    _compare_common(vars(nvrx), vars(native), report, path)


def _compare_dict(nvrx: dict, native: Any, report: _Report, path: str) -> None:
    if not isinstance(native, dict):
        report.fail(
            f"type mismatch at {path}: nvrx=dict, native={type(native).__name__}",
        )
        return
    nvrx_keys = set(nvrx)
    native_keys = set(native)
    for key in sorted(nvrx_keys - native_keys, key=str):
        report.warn(f"missing on native side: {path}:{key}")
    for key in sorted(native_keys - nvrx_keys, key=str):
        report.fail(f"extra on native side: {path}:{key}")
    for key in sorted(nvrx_keys & native_keys, key=str):
        _compare_common(nvrx[key], native[key], report, f"{path}:{key}")


def _compare_sequence(nvrx: Any, native: Any, report: _Report, path: str) -> None:
    if type(nvrx) is not type(native):
        report.fail(
            f"type mismatch at {path}: nvrx={type(nvrx).__name__}, native={type(native).__name__}",
        )
        return
    if len(nvrx) != len(native):
        report.fail(
            f"length mismatch at {path}: nvrx={len(nvrx)}, native={len(native)}",
        )
        return
    for idx, (a, b) in enumerate(zip(nvrx, native)):  # noqa: B905
        _compare_common(a, b, report, f"{path}[{idx}]")


def _compare_tensor(nvrx: torch.Tensor, native: Any, report: _Report, path: str) -> None:
    if not isinstance(native, torch.Tensor):
        report.fail(
            f"type mismatch at {path}: nvrx=torch.Tensor, native={type(native).__name__}",
        )
        return
    if nvrx.dtype != native.dtype or nvrx.shape != native.shape:
        report.fail(
            f"tensor metadata mismatch at {path} (nvrx={nvrx.dtype}{tuple(nvrx.shape)}, native={native.dtype}{tuple(native.shape)})",
        )
    elif not torch.equal(nvrx, native):
        report.fail(f"tensor value mismatch at {path}")


def _compare_scalar(nvrx: Any, native: Any, report: _Report, path: str) -> None:
    if nvrx != native:
        report.fail(
            f"value mismatch at {path}: nvrx={nvrx!r}, native={native!r}",
        )


def _compare_fallback(nvrx: Any, native: Any, report: _Report, path: str) -> None:
    # Complex / unknown leaf types (enums other than IntEnum, numpy arrays, …).
    # We can't pickle-deep-compare here, so we settle for repr equality and
    # only surface differences — silently passing matched complex leaves keeps
    # stderr usable when the tree has many of them.
    if repr(nvrx) != repr(native):
        report.fail(
            f"complex-type value mismatch at {path} ({type(nvrx).__name__}): repr differs",
        )


def _compare_common(
    nvrx: Any,
    native: Any,
    report: _Report,
    path: str,
) -> None:
    """Recursively apply the asymmetric policy to a `common.pt` subtree."""
    if _is_namespace(nvrx) or _is_namespace(native):
        _compare_namespace(nvrx, native, report, path)
    elif isinstance(nvrx, dict):
        _compare_dict(nvrx, native, report, path)
    elif isinstance(nvrx, (list, tuple)):
        _compare_sequence(nvrx, native, report, path)
    elif isinstance(nvrx, torch.Tensor):
        _compare_tensor(nvrx, native, report, path)
    elif isinstance(nvrx, (int, float, str, bool, type(None))):
        _compare_scalar(nvrx, native, report, path)
    else:
        _compare_fallback(nvrx, native, report, path)


def main(native_dir: Path, nvrx_dir: Path) -> int:
    """Equality-compare two checkpoint directories.

    Tensors must match exactly in both directions; ``common.pt`` is checked
    under the asymmetric policy (missing-on-native warns, everything else
    fails). Returns 0 only when tensors are bit-identical and any
    ``common.pt`` divergences are limited to "missing on native" warnings.
    """
    report = _Report()

    nvrx_tensors, nvrx_meta = _load_tensors(nvrx_dir)
    native_tensors, native_meta = _load_tensors(native_dir)
    _compare_tensors(nvrx_tensors, native_tensors, nvrx_meta, native_meta, report)

    nvrx_common = _load_common(nvrx_dir)
    native_common = _load_common(native_dir)
    _compare_common(nvrx_common, native_common, report, "common.pt")

    parts = []
    if report.warnings:
        parts.append(f"{report.warnings} warning(s)")
    if report.failures:
        parts.append(f"{report.failures} failure(s)")
    summary = ", ".join(parts) if parts else "equal"
    print(  # noqa: T201
        f"checkpoint_equality_compare: {summary} (native={native_dir}, nvrx={nvrx_dir})",
        file=sys.stderr,
    )
    return 1 if report.failures else 0


def _cli() -> int:
    # `__doc__` is None under `python -OO`; fall back to a literal so argparse
    # still has a description.
    description = (__doc__ or "Compare a native checkpoint against an nvrx one").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--native",
        type=Path,
        required=True,
        help="path to the native checkpoint sub-directory",
    )
    parser.add_argument(
        "--nvrx",
        type=Path,
        required=True,
        help="path to the nvrx checkpoint sub-directory",
    )
    args = parser.parse_args()
    return main(args.native, args.nvrx)


if __name__ == "__main__":
    sys.exit(_cli())
