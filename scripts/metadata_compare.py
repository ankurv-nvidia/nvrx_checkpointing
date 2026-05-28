"""Metadata-level cross-provider compare for one native+nvrx iter pair.

Companion to ``checkpoint_equality_compare`` (which does tensor + common.pt
semantic equality via ``dcp.load`` / ``torch.load``). This script runs the
byte-level metadata diff:

1. ``distcp_trace(native, extract_tensor_data=False)`` — walks the native
   checkpoint dir, computes per-chunk md5sums + flattened common.pt entries,
   writes ``reference.metadata.info``, and verifies the native server's own
   ``0.metadata.info`` claims against on-disk truth.
2. ``distcp_trace(nvrx, ..., verify_native_server=False)`` — same on the
   nvrx side, minus the server-claim check (nvrx doesn't write
   ``0.metadata.info``).
3. ``compare_reference_metadata(native, nvrx)`` — diffs the two
   ``reference.metadata.info`` files and raises on any divergence.

Exits 0 on pass, 1 on any failure.

Runs as a subprocess from ``_run_equality_compare`` in ``run_job.py``: the
in-srun parent process doesn't have the ``TORCHINDUCTOR_WORKER_START`` env
overlay that the training subprocess gets, so importing ``megatron.core``
(transitively pulled in by ``megatron.native.debug``) there triggers
``torch._inductor.config.decide_worker_start_method`` with an empty
``TORCHINDUCTOR_WORKER_START`` and crashes. Subprocessing lets us thread
the same env overlay used by the training subprocess via ``env=...``.

CLI::

    python -m megatron.native.scripts.metadata_compare \\
        --native <ckpt_dir>/native \\
        --nvrx   <ckpt_dir>/nvrx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(native_dir: Path, nvrx_dir: Path) -> int:
    import megatron.native.debug  # noqa: PLC0415 — defer until argparse done so --help is fast

    native_info = native_dir / "0.metadata.info"
    if not native_info.is_file():
        print(f"FAIL: native 0.metadata.info missing at {native_info}", file=sys.stderr)  # noqa: T201
        return 1
    try:
        megatron.native.debug.distcp_trace(native_dir, extract_tensor_data=False)
        megatron.native.debug.distcp_trace(nvrx_dir, extract_tensor_data=False, verify_native_server=False)
        megatron.native.debug.compare_reference_metadata(native_dir, nvrx_dir)
    except Exception as ex:  # noqa: BLE001 — best-effort diagnostic, surface every failure mode
        print(f"FAIL: {type(ex).__name__}: {ex}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("--native", type=Path, required=True, help="Native checkpoint subdir (iter_*/native)")
    p.add_argument("--nvrx", type=Path, required=True, help="nvrx checkpoint subdir (iter_*/nvrx)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    sys.exit(main(args.native, args.nvrx))
