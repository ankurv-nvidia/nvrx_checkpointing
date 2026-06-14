"""Generate a per-tensor md5 trace for a checkpoint dir and diff it against a baseline.

For one iter directory of a single-provider pretrain run, this script:

1. Runs ``distcp_trace`` on ``--checkpoint-dir`` — walks the on-disk
   ``.metadata`` / ``*.distcp`` files, computes per-chunk md5sums, and writes
   ``reference.metadata.info`` next to the inputs. Both providers use the same
   path here; ``distcp_trace``'s optional native-server claim check (diffing
   against the server-written ``0.metadata.info``) is disabled — this script
   only cares about the trace-vs-baseline diff.
2. If the baseline holds a ``reference.metadata.info`` (or its xz-compressed
   ``reference.metadata.info.xz`` form), diffs the just-derived file against it
   via ``compare_reference_metadata``.

The baseline is the nvrx-derived reference metadata checked into
``megatron/native/metadata/<cfg>_checkpoints_iter_NNNNNNN/`` as an xz-compressed
``reference.metadata.info.xz`` — it is the cross-provider equality contract
that both pretrain-nvrx (self-consistency) and pretrain-native (cross-provider)
compare against. ``compare_reference_metadata`` decompresses it in memory.

Exit codes (callers decide whether to warn or hard-fail on each):

  0 — equal, or no baseline yet for provider=nvrx (lenient bootstrap path so
      pretrain-nvrx can seed the baseline on a fresh cfg/iter via its
      update-baselines job).
  1 — structural error: missing inputs, distcp_trace crashed, OR no baseline
      for provider=native (silent-pass here would mask a misconfigured run,
      since native compares against the nvrx-derived baseline that should
      already be in metadata/).
  3 — baseline mismatch (reference.metadata.info differs from baseline).

Runs as a subprocess from ``_compare_iters_to_baseline`` in ``run_job.py``:
the in-srun parent process lacks the ``TORCHINDUCTOR_WORKER_START`` env
overlay that the training subprocess gets, so importing ``megatron.core``
(transitively pulled in by ``megatron.native.debug``) there trips
``torch._inductor.config.decide_worker_start_method``. Subprocessing lets us
forward the same overlay used by the training subprocess via ``env=...``.

CLI::

    python -m megatron.native.scripts.metadata_compare \\
        --checkpoint-dir <iter_NNNNNNN> \\
        --baseline-dir   <metadata/<cfg>_checkpoints_iter_NNNNNNN> \\
        --provider       native|nvrx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(checkpoint_dir: Path, baseline_dir: Path, provider: str) -> int:
    import megatron.native.debug  # noqa: PLC0415 — defer until argparse done so --help is fast

    try:
        # verify_native_server=False for both providers — this script's only job
        # is the trace-vs-baseline diff; the server-claim check is a separate
        # diagnostic available via distcp_trace's CLI for offline debugging.
        megatron.native.debug.distcp_trace(
            checkpoint_dir,
            extract_tensor_data=False,
            verify_native_server=False,
        )
    except Exception as ex:  # noqa: BLE001 — best-effort diagnostic, surface every failure mode
        print(f"FAIL (distcp_trace): {type(ex).__name__}: {ex}", file=sys.stderr)  # noqa: T201
        return 1

    baseline_ref = baseline_dir / "reference.metadata.info"
    if not baseline_ref.is_file() and not (baseline_dir / "reference.metadata.info.xz").is_file():
        # Provider-asymmetric bootstrap policy: nvrx is the source of truth
        # that seeds the baseline (via pretrain-nvrx → update-baselines), so
        # let it pass without a baseline. Native compares against that already-
        # seeded snapshot, so a missing baseline means the run isn't actually
        # validating anything — surface as a hard fail rather than silent-pass.
        if provider == "nvrx":
            print(
                f"INFO: no baseline at {baseline_ref} — skipping compare (run pretrain-nvrx:*:update_baselines to seed)",
                file=sys.stderr,
            )
            return 0
        print(
            f"FAIL: no baseline at {baseline_ref} — pretrain-native requires the nvrx baseline to be seeded "
            f"first (run pretrain-nvrx → pretrain-nvrx:*:update_baselines)",
            file=sys.stderr,
        )
        return 1

    try:
        diff_count = megatron.native.debug.compare_reference_metadata(checkpoint_dir, baseline_dir)
    except Exception as ex:  # noqa: BLE001
        print(f"FAIL (compare): {type(ex).__name__}: {ex}", file=sys.stderr)  # noqa: T201
        return 1

    return 3 if diff_count > 0 else 0


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Iter checkpoint dir produced by the just-completed run")
    p.add_argument("--baseline-dir", type=Path, required=True, help="metadata/ baseline dir holding reference.metadata.info")
    p.add_argument("--provider", choices=["native", "nvrx"], required=True, help="Checkpoint provider used by the run")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    sys.exit(main(args.checkpoint_dir, args.baseline_dir, args.provider))
