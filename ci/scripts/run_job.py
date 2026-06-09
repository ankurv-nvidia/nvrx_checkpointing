#!/usr/bin/env python3
"""CI job runner for megatron/native benchmarks and e2e tests.

Jobs:
  pretrain-native  native provider; deterministic adds the cross-provider md5 compare.
  pretrain-nvrx    nvrx provider; deterministic runs twice (cross-run determinism gate),
                   perf is a fast-path smoke.

Flavors (--flavor, repeatable, default perf): perf (flash + fusion, perf-gated),
deterministic (reproducible kernels + md5 checks, excludes perf), valgrind (memcheck the
server, single-node).

SLURM vs inline is inferred from SLURM_PARTITION; --slurm forces SLURM for local runs.
Per-cluster srun/image-build templates live in templates/ (selected by CLUSTER); env knobs:
PLATFORM (tag + baseline namespace), CLUSTER, GPUS_PER_NODE, SLURM_PARTITION, CI_DIR.

Exit codes: 0 = pass (or --skip-perf), 3 = perf regression (soft), 1 = job/setup failure.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MEGATRON_ROOT = (SCRIPT_DIR / "../../../..").resolve()
BASELINES_DIR = (SCRIPT_DIR / "../baselines").resolve()
CONFIGS_DIR = (SCRIPT_DIR / "../configs").resolve()
logger = logging.getLogger("native_ci")

# Sentinel first arg that re-enters this script as the in-process training entry: the launcher
# runs `python3 run_job.py __pretrain_entry__ *opts` (or via torchrun) and __main__ dispatches it
# to run pretrain in-process, keeping the CI markers in megatron/native and pretrain_hybrid.py unmodified.
_PRETRAIN_ENTRY_CMD = "__pretrain_entry__"

sys.path.insert(0, str(MEGATRON_ROOT))

ENROOT_SCRIPTS_DIR = (MEGATRON_ROOT / "megatron/native/scripts").resolve()
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.04-py3")


def wait_for_file(path: Path, timeout_seconds: int) -> None:
    """Wait for a file to exist, polling every 10 seconds, up to timeout_seconds.

    If path's basename contains a glob wildcard (*?[), waits for any match under path.parent
    instead — useful for files whose exact name isn't known ahead of time (e.g. async-written
    checkpoint metadata).
    """
    has_wildcard = any(c in path.name for c in "*?[")
    start_time = time.time()
    while True:
        if has_wildcard:
            if any(path.parent.glob(path.name)):
                return
        elif path.is_file():
            return
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {path} after {timeout_seconds} seconds")
        prefix = "[in-srun] " if "SLURM_PROCID" in os.environ else ""
        logger.info(f"{prefix}Waiting for {path} to be available...")
        time.sleep(10)


def _prepare_sh_hash() -> str:
    """First 10 hex chars of sha256(prepare.sh) — namespaces the built sqsh image so any
    change to prepare.sh forces a rebuild rather than reusing a stale image. Must match
    the hash computed in scripts/enroot.sh."""
    return hashlib.sha256((ENROOT_SCRIPTS_DIR / "prepare.sh").read_bytes()).hexdigest()[:10]


def _default_enroot_image() -> str:
    return f"{Path.home()}/data/native.{_prepare_sh_hash()}.sqsh"


def _image_path() -> Path:
    """Enroot image path: the ENROOT_IMAGE override, else the prepare.sh-hashed default.

    Shared by prepare_image.py (which builds it in each workload job's before_script)
    and the workload itself (which uses it as --container-image), so both agree on the
    same file.
    """
    return Path(os.environ.get("ENROOT_IMAGE", _default_enroot_image()))


def _platform_tag() -> str:
    """Hardware platform identifier used to namespace baseline files (e.g. gb200, h100, h100).

    Comes from PLATFORM (set per runner at setup), which is also the GitLab runner
    tag that routes jobs. Falls back to CLUSTER for runners predating the
    PLATFORM/CLUSTER split, then to the hostname stem. Perf baselines key on this,
    so one GPU platform shares baselines across clusters; CLUSTER (which selects the
    srun templates) does not enter the baseline name.
    """
    tag = os.environ.get("PLATFORM", "").strip()
    return tag or socket.gethostname().split(".")[0]


# Ensure the repo root is on sys.path so `import megatron` works regardless of
# how the script is invoked (e.g. from a GitLab runner without a proper install).
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))
RESULTS_DIR = MEGATRON_ROOT / "megatron" / "native" / "ci" / "ci-results"
PERF_TOLERANCE = 0.05  # 5% tolerance — both regressions and improvements beyond this fail
# Max tolerated slower/faster socket-half mean ratio. Post NUMA-locality fix the two
# rank halves (= the two NUMA sockets) sit at ~1.0-1.13 on every platform; a breach
# means a per-socket bimodality has returned (the bug had a ~2x split). Gated on the
# socket halves, not individual ranks: per-rank D2H noise alone reaches ~1.48 on h100.
MAX_BW_SOCKET_IMBALANCE = 1.30

_BASELINE_CHANGE_THRESHOLD = 1e-6  # minimum relative change to count as a meaningful baseline update
_SOFT_FAIL_RC = 3  # exit code for a tolerated soft fail (perf regression / nvrx baseline drift)


def _ensure_vcpkg_root() -> None:
    """Pin VCPKG_ROOT to a per-checkout location so the CMake build — triggered by
    `import megatron.native`, forwarded into the container — finds it. Call before any
    such import; subprocesses inherit it. Override via the VCPKG_ROOT env var.

    The root sits beside the source checkout (MEGATRON_ROOT/../vcpkg). GitLab clones each
    concurrent build into its own numbered checkout dir, so concurrent pipelines on one
    runner land on distinct roots and never contend on a shared vcpkg installed-tree lock.
    """
    os.environ.setdefault("VCPKG_ROOT", str(MEGATRON_ROOT.parent / "vcpkg"))


def _is_pretrain_job(job: str) -> bool:
    return job.startswith("pretrain-")


def _ckpt_provider(job: str) -> str:
    """The --ckpt-provider a pretrain job exercises (the suffix after 'pretrain-').

    Also selects metadata-compare semantics: native = strict cross-provider,
    nvrx = lenient self-consistency.
    """
    return job.split("-", 1)[1]


# Flavors compose over the core fast-path recipe: perf leaves it as-is; deterministic swaps in
# reproducible kernels + --deterministic-mode; valgrind memchecks the server.


@dataclass
class Recipe:
    """CLI opts, process env, and SLURM knobs that a flavor transforms in place."""

    opts: list[str]
    env: dict[str, str]
    build_env: dict[str, str] = field(default_factory=dict)
    srun_time_min: int = 60  # SLURM --time in minutes; 1h default, raised per flavor (e.g. valgrind)
    srun_comments: list[str] = field(default_factory=list)


@dataclass
class RunCtx:
    """State shared with a flavor's prolog/epilog for one pretrain run."""

    args: argparse.Namespace
    job: str
    cfg: str
    provider: str
    machine: str
    flavors: list[Flavor]
    log_file: Path | None = None  # perf log; set only in the process that owns it
    valgrind_rc: int | None = None  # set by the valgrind prolog when its server exits
    soft_fail: bool = False  # perf regression → exit 3

    @property
    def flavor_tag(self) -> str:
        return "-".join(f.name for f in self.flavors)

    @property
    def stem(self) -> str:
        return _artifact_stem(self.machine, self.job, self.flavor_tag, self.cfg)

    @property
    def is_worker(self) -> bool:
        """In the process that ran training (can read checkpoints, run container tools)."""
        return in_srun() or not self.args.slurm

    @property
    def is_rank0(self) -> bool:
        return os.environ.get("SLURM_PROCID", "0") == "0"

    def has(self, name: str) -> bool:
        return any(f.name == name for f in self.flavors)


class Flavor(ABC):
    """A composable transform over the core pretrain recipe, with optional run hooks.

    The base class is the perf fast path: an identity transform and no-op hooks.
    Subclasses override transform (recipe edits), prolog (a context manager wrapping
    the training launch), and/or epilog (the post-spin-down check). Shared recipe
    scaffolding — the neutral env and option editors — lives here too.
    """

    name: ClassVar[str]

    def transform(self, recipe: Recipe) -> None:  # noqa: B027 — optional hook, no-op by default
        """Edit the core recipe in place. Default: identity (the fast path)."""

    @contextlib.contextmanager
    def prolog(self, ctx: RunCtx) -> Iterator[None]:  # noqa: ARG002 — ctx is for overrides
        """Wrap the training launch. Default: no setup/teardown."""
        yield

    def epilog(self, ctx: RunCtx) -> None:  # noqa: B027 — optional hook, no-op by default
        """Run this flavor's check after the server has spun down. Default: none."""

    @staticmethod
    def base_env(triton_dir: Path) -> dict[str, str]:
        """Process env shared by every pretrain run, independent of flavor."""
        return {
            "NCCL_IB_TIMEOUT": "24",
            "UCX_MEM_MMAP_HOOK_MODE": "none",
            "UCX_MEM_CUDA_HOOK_MODE": "none",
            "UCX_MEM_MALLOC_HOOKS": "n",
            "UCX_ERROR_SIGNALS": "",
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
            "TORCHINDUCTOR_WORKER_START": "fork",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "QUANTIZATION_TYPE_DEBUG": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TRITON_CACHE_DIR": str(triton_dir),
        }

    @staticmethod
    def _set_opt(opts: list[str], flag: str, value: str) -> None:
        """Replace flag's value token, appending `flag value` when flag is absent."""
        if flag in opts:
            opts[opts.index(flag) + 1] = value
        else:
            opts.extend([flag, value])

    @staticmethod
    def _drop_opt(opts: list[str], flag: str, *, has_value: bool = False) -> None:
        """Remove flag (and its following value token when has_value) if present."""
        if flag in opts:
            i = opts.index(flag)
            del opts[i : i + (2 if has_value else 1)]

    @staticmethod
    def _add_flag(opts: list[str], flag: str) -> None:
        if flag not in opts:
            opts.append(flag)


class PerfFlavor(Flavor):
    """Fast path (flash attention + cross-entropy fusion, already in the core recipe).

    Gated on the checkpoint-write perf baseline; runs no md5 compare.
    """

    name = "perf"

    def transform(self, recipe: Recipe) -> None:
        recipe.build_env["PHOENIX_BUILD_FOR_PERF"] = "1"

    def epilog(self, ctx: RunCtx) -> None:
        self.run_perf_epilog(ctx)

    @staticmethod
    def run_perf_epilog(ctx: RunCtx) -> None:
        """Parse perf metrics from the run log, compare to the per-platform baseline, and
        soft-fail (exit 3 — a non-blocking CI warning) on out-of-tolerance.

        Shared with the deterministic flavor so both warn on perf drift against their own
        baseline. No-op under --skip-perf, with no log to parse, or under valgrind.
        """
        if ctx.log_file is None or ctx.args.skip_perf or ctx.has("valgrind"):
            return
        passed = PerfReport.parse_and_report(
            ctx.job,
            ctx.cfg,
            ctx.flavor_tag,
            ctx.log_file,
            BASELINES_DIR / f"{ctx.stem}.yaml",
            RESULTS_DIR / f"{ctx.stem}_report.json",
            ctx.args.update_baseline,
            ctx.args.gpus_per_node,
        )
        if not passed:
            ctx.soft_fail = True


class DeterministicFlavor(Flavor):
    """Reproducible numerics for the metadata md5 checks, plus a non-blocking perf-drift
    warning against this flavor's own baseline.

    Swaps fused/atomic-add kernels for deterministic ones and adds --deterministic-mode.
    """

    name = "deterministic"

    # Must be set before the training process starts: --deterministic-mode consumes the
    # first three, and the Mamba SSD kernel bakes its @triton.autotune choice from
    # MAMBA_DETERMINISTIC at import time (before parse_args runs).
    ENV: ClassVar[dict[str, str]] = {
        "NCCL_ALGO": "Ring",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "MAMBA_DETERMINISTIC": "1",
    }

    def transform(self, recipe: Recipe) -> None:
        self._set_opt(recipe.opts, "--attention-backend", "unfused")
        self._drop_opt(recipe.opts, "--cross-entropy-loss-fusion")
        self._drop_opt(recipe.opts, "--cross-entropy-fusion-impl", has_value=True)
        self._add_flag(recipe.opts, "--deterministic-mode")
        recipe.env.update(self.ENV)

    def epilog(self, ctx: RunCtx) -> None:
        # Skipped under valgrind: that flavor owns the server and tears it down before the
        # epilog, leaving the md5 trace with no server (md5 is covered by the plain run).
        if ctx.is_worker and not ctx.has("valgrind"):
            PerfReport._compare_iters_to_baseline(ctx.cfg, ctx.provider)
        # Warn (non-blocking exit 3) on perf drift against this flavor's own slower baseline;
        # the md5 mismatch above still hard-fails.
        PerfFlavor.run_perf_epilog(ctx)


class ValgrindFlavor(Flavor):
    """Memcheck the checkpoint_server: build with shrunk D2H buffers, pre-launch the
    server under valgrind on rank 0, and fail if valgrind reports leaks/errors.

    Composes with a numerics flavor; single-node only (only rank 0's server is checked).
    """

    name = "valgrind"

    _READY_MARKER = "Checkpoint server started"
    # server.cpp ~L1088 logs this when it begins orderly shutdown; we stop tee'ing to the
    # console once seen (the file keeps the rest) to drop the auto-shutdown wait's repeated
    # per-channel lines and valgrind's exit diagnostics.
    _STOP_MARKER = "Shutting down server thread"

    # Stall guard: a stuck server keeps logging but stops writing checkpoint files, so
    # checkpoint-file activity (not log activity) is the progress signal.
    _STALL_SILENCE_S = 300  # 5 min of no checkpoint-file activity = stalled
    _STALL_POLL_S = 10  # how often to poll for progress / exit
    _STALL_STOP_GRACE_S = 120  # grace for a stalled server to honor a re-issued stop
    _STALL_LOG_EVERY_S = 60  # heartbeat cadence while waiting, so the wait is visible

    # memcheck *error* headers (NOT leak kinds — leaks stay gated by --error-exitcode).
    # valgrind prints each as the error happens, prefixed ==<pid>==, so they remain in
    # the log even when valgrind is later SIGKILLed before it can emit ERROR SUMMARY.
    _MEMCHECK_ERROR_RE = re.compile(
        r"==\d+==\s+("
        r"Invalid read of size \d+|"
        r"Invalid write of size \d+|"
        r"Use of uninitialised value|"
        r"Conditional jump or move depends on uninitialised value|"
        r"Mismatched free\(\)|"
        r"Invalid free\(\)|"
        r"Source and destination overlap|"
        r"Jump to the invalid address|"
        r"Process terminating with default action of signal|"
        r"Syscall param "
        r")",
    )

    @classmethod
    def _scan_valgrind_errors(cls, server_log: Path) -> list[str]:
        """Distinct memcheck error headers in the server log.

        Kept as a robust SUMMARY and backstop — not the primary gate. A real memory
        error normally trips a regular guard first (a crash → non-zero exit, or a
        hang → the stall guard). This scan prints exactly what valgrind flagged, and
        lets us detect the case where an error slips past the regular guards.
        """
        found: list[str] = []
        seen: set[str] = set()
        try:
            with server_log.open(errors="replace") as fp:
                for line in fp:
                    m = cls._MEMCHECK_ERROR_RE.search(line)
                    if m is not None and m.group(1) not in seen:
                        seen.add(m.group(1))
                        found.append(m.group(1))
        except OSError as ex:
            logger.info(f"[run_job] could not scan valgrind log {server_log}: {ex}")
        return found

    @staticmethod
    def _ckpt_mtimes(ckpt_root: Path) -> dict[str, float]:
        """{relative path: mtime} for every file under ckpt_root (recursive).

        The max value is the progress signal for the stall guard; the per-file map
        lets the wait loop report which files changed since the last heartbeat.
        """
        out: dict[str, float] = {}
        try:
            for p in ckpt_root.rglob("*"):
                try:
                    if p.is_file():
                        out[str(p.relative_to(ckpt_root))] = p.stat().st_mtime
                except OSError:
                    continue
        except OSError:
            pass
        return out

    def _await_server_exit(self, proc: subprocess.Popen, ckpt_root: Path) -> tuple[int | None, bool]:
        """Wait for the valgrind'd server to exit, bounding the wait by checkpoint-file
        activity instead of a flat timeout. Returns (rc, stalled).

        While the server keeps writing checkpoint files it is making progress, so keep
        waiting (a healthy 1x8 finalize under memcheck is slow but never silent for
        long, and the final leak check is seconds). _STALL_SILENCE_S with no
        checkpoint-file activity means it is stuck: re-request a stop, then SIGKILL if
        it still will not exit. Either way the caller fails the job. rc is the process
        exit code, or None if it had to be SIGKILLed.
        """
        mtimes = self._ckpt_mtimes(ckpt_root)
        last_max = max(mtimes.values(), default=0.0)
        logged_mtimes = dict(mtimes)
        start = last_progress = last_log = time.monotonic()
        while True:
            try:
                return proc.wait(timeout=self._STALL_POLL_S), False
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            mtimes = self._ckpt_mtimes(ckpt_root)
            cur_max = max(mtimes.values(), default=0.0)
            if cur_max > last_max:
                last_max = cur_max
                last_progress = now
            # Per-minute heartbeat so a slow-but-progressing finalize under valgrind is
            # visible in the CI log instead of looking like a hang. Report the checkpoint
            # files written/modified since the last heartbeat; if none, show how long it
            # has been silent and when the stall guard will trip.
            if now - last_log >= self._STALL_LOG_EVERY_S:
                changed = sorted(p for p, m in mtimes.items() if logged_mtimes.get(p) != m)
                logged_mtimes = dict(mtimes)
                last_log = now
                waited = round(now - start)
                if changed:
                    shown = ", ".join(changed[:6]) + (f" (+{len(changed) - 6} more)" if len(changed) > 6 else "")
                    logger.info(
                        f"[run_job] waiting for checkpoint_server to exit ({waited}s elapsed); "
                        f"{len(changed)} checkpoint file(s) modified in the last {self._STALL_LOG_EVERY_S}s: {shown}",
                    )
                else:
                    silent = round(now - last_progress)
                    logger.info(
                        f"[run_job] waiting for checkpoint_server to exit ({waited}s elapsed); "
                        f"no checkpoint-file activity for {silent}s (stall guard trips at {self._STALL_SILENCE_S}s)",
                    )
            if now - last_progress >= self._STALL_SILENCE_S:
                logger.error(
                    f"[run_job] checkpoint_server made no checkpoint-file progress for "
                    f"{self._STALL_SILENCE_S}s under {ckpt_root} — treating as stalled",
                )
                CheckpointServer._dump_server_threads()
                CheckpointServer._request_server_stop()
                try:
                    rc = proc.wait(timeout=self._STALL_STOP_GRACE_S)
                    logger.error(f"[run_job] stalled checkpoint_server exited after re-issued stop (rc={rc})")
                    return rc, True
                except subprocess.TimeoutExpired:
                    logger.exception("[run_job] stalled checkpoint_server ignored stop — SIGKILL")
                    CheckpointServer._gcore_and_kill_server()
                    return None, True

    def transform(self, recipe: Recipe) -> None:
        # PHOENIX_BUILD_FOR_VALGRIND → CMakeLists.txt sets BUILDENV_DISABLE_MARCH_NATIVE
        # and shrinks D2H buffers so valgrind's shadow memory stays manageable.
        recipe.build_env["PHOENIX_BUILD_FOR_VALGRIND"] = "1"
        # valgrind runs ~10-50x slower; give it double the regular budget so the
        # stall guard (above) — not the SLURM time limit — is what ends a hung run.
        recipe.srun_time_min = recipe.srun_time_min * 2

    @contextlib.contextmanager
    def prolog(self, ctx: RunCtx) -> Iterator[None]:
        # Only global rank 0 hosts the server; other ranks skip their own auto-fork.
        if not ctx.is_rank0:
            os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
            yield
            return
        with self._server(ctx):
            yield

    def epilog(self, ctx: RunCtx) -> None:
        if ctx.is_worker and ctx.valgrind_rc not in (None, 0):
            raise RuntimeError(f"valgrind reported errors in checkpoint_server (exit {ctx.valgrind_rc}); see the server log")

    @staticmethod
    def _binary() -> Path:
        import megatron.native as _native  # noqa: F401, PLC0415 — triggers CMake build

        binary = MEGATRON_ROOT / "megatron/native/checkpoint_server"
        if not binary.is_file():
            raise FileNotFoundError(f"checkpoint_server not found at {binary}")
        return binary

    @contextlib.contextmanager
    def _server(self, ctx: RunCtx) -> Iterator[None]:
        """Pre-launch checkpoint_server under valgrind for the block.

        On enter: start the server under memcheck, set CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH=1
        so the in-process client connects to it, and block until the ready marker. On exit:
        wait for the server's no-activity auto-shutdown (so valgrind flushes its leak report)
        and record the exit code in ctx for epilog to check. valgrind's --log-fd=2 diagnostics
        share the server's stderr pipe; the watcher tees the unified log to the console.
        """
        valgrind = shutil.which("valgrind")
        if valgrind is None:
            raise RuntimeError("valgrind not found in PATH — required for the valgrind flavor")
        server = self._binary()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        server_log = RESULTS_DIR / f"{ctx.stem}_server.log"
        server_log.write_bytes(b"")  # truncate any prior run's log

        cmd = [valgrind, "--tool=memcheck", "--leak-check=full", "--show-leak-kinds=all", "--error-exitcode=42", "--log-fd=2", str(server)]
        logger.info(f"[run_job] Launching checkpoint_server under valgrind; log: {server_log}")
        CheckpointServer.require_no_checkpoint_server()  # a leftover here means a prior run didn't spin down

        os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
        # Server (RedirectStdio in server_main.cpp) appends its stderr here too; O_APPEND
        # plus our line-flushed writes keep entries from interleaving mid-line.
        os.environ["NATIVE_CKPT_SERVER_LOG"] = str(server_log)
        log_fp = server_log.open("a")
        proc, ready, watcher = self._spawn(cmd, server_log, log_fp)
        try:
            if not ready.wait(timeout=600):
                raise RuntimeError(f"checkpoint_server did not print '{self._READY_MARKER}' within 600s under valgrind; see {server_log}")
            if proc.poll() is not None:
                raise RuntimeError(f"checkpoint_server exited before becoming ready (rc={proc.returncode}); see {server_log}")
            logger.info("[run_job] checkpoint_server is ready — proceeding with pretrain")
            yield
        finally:
            # Ask the server to stop now rather than waiting out its idle auto-shutdown
            # — under memcheck that wait plus the slow teardown can blow the budget.
            # /api/stop triggers Server::Stop immediately; valgrind still flushes its
            # leak report when the process exits.
            logger.info("[run_job] Pretrain finished — stopping checkpoint_server so valgrind can flush its report")
            CheckpointServer._request_server_stop()
            # PRIMARY GATE (regular guards): wait for exit, bounded by checkpoint-file
            # activity. A stall (no file progress for _STALL_SILENCE_S) re-stops then
            # SIGKILLs the server and is a hard failure; a clean exit with a non-zero
            # rc (valgrind --error-exitcode=42) is failed by epilog.
            ckpt_root = MEGATRON_ROOT / "megatron/native/ci/ci-runs" / ctx.cfg
            rc, stalled = self._await_server_exit(proc, ckpt_root)
            ctx.valgrind_rc = rc  # record before cleanup so epilog sees it even if cleanup raises
            watcher.join(timeout=5)  # drain the final log lines before closing
            log_fp.close()
            if rc is not None:
                logger.info(f"[run_job] valgrind exited with code {rc}; server log: {server_log}")

            # ROBUST CROSS-CHECK (kept; a summary + backstop, NOT the primary gate):
            # list the memcheck errors valgrind flagged, however it exited.
            errors = self._scan_valgrind_errors(server_log)
            summary = "; ".join(errors[:5]) + (" ..." if len(errors) > 5 else "")
            if errors:
                logger.error(f"[run_job] valgrind memcheck summary: {len(errors)} error type(s): {summary}; see {server_log}")
            else:
                logger.info(f"[run_job] valgrind memcheck summary: no errors found in {server_log}")

            # Regular guard: a stalled server is a hard failure.
            if stalled:
                raise RuntimeError(
                    f"checkpoint_server stalled (no checkpoint-file activity for {self._STALL_SILENCE_S}s) "
                    f"and had to be stopped/killed — see the thread dump above and {server_log}",
                ) from None
            # Backstop: if the regular guards say PASS (clean exit, rc 0) yet the scan
            # still found memcheck errors, one slipped past them — fail loudly and say
            # so, so the regular guards get strengthened rather than leaning on the scan.
            if rc == 0 and errors:
                raise RuntimeError(
                    "valgrind reported memcheck errors that the exit-code/stall guards did NOT catch — "
                    f"the regular guards need strengthening: {summary}; see {server_log}",
                ) from None
            # rc != 0 (e.g. 42) is handled by epilog's exit-code guard.

    def _spawn(self, cmd: list[str], server_log: Path, log_fp: IO[str]) -> tuple[subprocess.Popen, threading.Event, threading.Thread]:
        """Spawn the server; a pump thread copies its pipe into log_fp and a watcher tees the
        unified log to the console. The watcher is returned so the caller can join it after
        proc.wait() (otherwise the final ~0.25s of output is lost)."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=1, text=True)
        ready = threading.Event()

        def _pump() -> None:
            assert proc.stdout is not None  # noqa: S101 — Popen with stdout=PIPE always has it
            for line in proc.stdout:
                log_fp.write(line)
                log_fp.flush()
            ready.set()  # pipe closed (process exited): unblock a stuck waiter

        threading.Thread(target=_pump, daemon=True).start()
        watcher = threading.Thread(target=self._watch_log, args=(server_log, proc, ready), daemon=True)
        watcher.start()
        return proc, ready, watcher

    def _watch_log(self, server_log: Path, proc: subprocess.Popen, ready: threading.Event) -> None:
        """Tail server_log and tee new bytes to stderr (the server writes there directly, not
        through the pipe). Set `ready` at the readiness marker; stop tee'ing at the shutdown
        marker. Always set `ready` on exit so a stuck waiter doesn't deadlock."""
        pos = 0
        ready_b, stop_b = self._READY_MARKER.encode(), self._STOP_MARKER.encode()
        teeing = True

        def drain() -> None:
            nonlocal pos, teeing
            if not server_log.exists():
                return
            with server_log.open("rb") as fp:
                fp.seek(pos)
                chunk = fp.read()
            if not chunk:
                return
            pos += len(chunk)
            if teeing:
                sys.stderr.buffer.write(chunk)
                sys.stderr.flush()
                if stop_b in chunk:
                    teeing = False
            if not ready.is_set() and ready_b in chunk:
                ready.set()

        while True:
            drain()
            if proc.poll() is not None:
                drain()  # final read in case bytes landed between drain() and poll()
                ready.set()
                return
            time.sleep(0.25)


_FLAVOR_ORDER = ("perf", "deterministic", "valgrind")
_FLAVORS: dict[str, type[Flavor]] = {cls.name: cls for cls in (PerfFlavor, DeterministicFlavor, ValgrindFlavor)}


def _resolve_flavors(requested: list[str] | None, nodes: int) -> list[Flavor]:
    """Validate the --flavor selection and instantiate it in canonical order.

    Defaults to ['perf']. perf and deterministic are mutually exclusive. pretrain-nvrx
    runs under either: deterministic gates the cross-run determinism check, while perf
    is a fast-path smoke (no parseable bw metrics — its run prints checkpoint times).
    valgrind memchecks rank 0's server only, so it is single-node.
    """
    names = list(dict.fromkeys(requested or ["perf"]))
    unknown = [n for n in names if n not in _FLAVORS]
    if unknown:
        raise SystemExit(f"--flavor: unknown flavor(s) {unknown}; choose from {sorted(_FLAVORS)}")
    if "perf" in names and "deterministic" in names:
        raise SystemExit("--flavor: 'perf' and 'deterministic' are mutually exclusive")
    if "valgrind" in names and nodes > 1:
        raise SystemExit("--flavor valgrind is single-node only (only rank 0's checkpoint_server is memchecked)")
    return [_FLAVORS[n]() for n in _FLAVOR_ORDER if n in names]


def _apply_flavors(recipe: Recipe, flavors: list[Flavor]) -> None:
    for flavor in flavors:
        flavor.transform(recipe)


@contextlib.contextmanager
def _flavored_run(ctx: RunCtx) -> Iterator[None]:
    """Bracket a pretrain run for hygiene: enter each flavor's prolog (worker only), run
    training in the body, then spin the checkpoint_server down and run the flavor epilogs.

    The checkpoint_server is a per-node detached daemon (127.0.0.1:41000), so the spin-down
    runs on EVERY rank — each stops its own node's daemon (idempotent: extra ranks on a node
    find the port already free). It runs on both paths: cleanly (wait for the async flush,
    then fail with a thread dump if it won't stop) on success, and best-effort if training
    raised, so a crash can't strand a node's daemon and block the next job. The flavor
    epilogs are global checks (md5/perf over the shared-filesystem checkpoint) and run on
    global rank 0 only — running distcp_trace on every rank would collide on the server port.
    A perf soft-fail becomes exit 3.
    """
    stack = contextlib.ExitStack()
    if ctx.is_worker:
        for flavor in ctx.flavors:
            stack.enter_context(flavor.prolog(ctx))
    try:
        yield
    except BaseException:
        stack.close()  # prologs exit (valgrind waits for its server, records the rc)
        CheckpointServer._ensure_server_down(ctx, wait_flush=False, fail_on_timeout=False)
        raise
    stack.close()
    CheckpointServer._ensure_server_down(ctx)
    if not ctx.is_rank0:
        return
    for flavor in ctx.flavors:
        flavor.epilog(ctx)
    if ctx.soft_fail:
        sys.exit(3)


def _artifact_stem(machine: str, job: str, flavor_tag: str, cfg: str) -> str:
    """`{machine}_{job}_{flavor_tag}_{cfg}`, dropping the tag when empty (e.g. ckptbench)."""
    parts = [machine, job, flavor_tag, cfg] if flavor_tag else [machine, job, cfg]
    return "_".join(parts)


# SLURM context detection


def _runner_uses_slurm() -> bool:
    """True when this runner submits via SLURM, inferred from SLURM_PARTITION being set.

    setup_runner.py forwards SLURM_PARTITION from the cluster config (it also feeds the
    srun --partition), so a partitioned cluster runs via SLURM and one without runs
    inline (e.g. devbox) — no separate flag needed. main() folds this into args.slurm.
    """
    return bool(os.environ.get("SLURM_PARTITION", "").strip())


def in_srun() -> bool:
    return "SLURM_PROCID" in os.environ


# Per-cluster SLURM/image-build templates: per-run values fill ${...} placeholders, cluster
# constants live in the file. Repo ships defaults; a runner can override per cluster.

TEMPLATES_DIR = (SCRIPT_DIR / "../templates").resolve()


def _ci_dir() -> Path:
    """Runner state dir holding `config` and any override template files.

    setup_runner.py forwards it as CI_DIR; falls back to the ~/data/ci
    default used by the runner-setup docs.
    """
    raw = os.environ.get("CI_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / "data" / "ci"


def _load_template(name: str) -> str:
    """Return template `name`, searched in priority order:

    1. CI_DIR/<name>                  — per-runner override next to the config
    2. TEMPLATES_DIR/<cluster>.<name> — repo cluster-specific default (e.g. cwdfw.srun.tmpl)
    3. TEMPLATES_DIR/<name>           — repo generic default
    """
    cluster = os.environ.get("CLUSTER", "").strip()
    candidates = [_ci_dir() / name]
    if cluster:
        candidates.append(TEMPLATES_DIR / f"{cluster}.{name}")
    candidates.append(TEMPLATES_DIR / name)
    for path in candidates[:-1]:
        if path.is_file():
            logger.info(f"[run_job] using template {path}")
            return path.read_text()
    default = candidates[-1]
    if not default.is_file():
        raise FileNotFoundError(f"SLURM template {name!r} not found (looked in {', '.join(str(c) for c in candidates)})")
    return default.read_text()


def _render_template(name: str, values: dict[str, object]) -> str:
    """Substitute ${lowercase} placeholders in template `name`.

    Uses safe_substitute so shell ${UPPER}/$VAR references in script templates
    are left untouched — our placeholders are lowercase by convention.
    """
    return string.Template(_load_template(name)).safe_substitute(values)


def _build_srun_cmd(  # noqa: PLR0913
    *,
    nodes: int,
    ntasks: int,
    time_minutes: int,
    job_name: str,
    image: str,
    container_env: str,
    gpus_per_node: int,
    total_gpus: int,
    extra_srun_args: list[str] | None = None,
) -> list[str]:
    """Render srun.tmpl into the launch-command prefix (before the job payload).

    Shared by every launched job; callers append the payload (e.g.
    ``python3 -u run_job.py <job>``). Comment lines (first non-space char '#')
    and blank lines are dropped, the rest is ${...}-substituted and tokenized.
    Every job is exempted from the idle-GPU reaper for its full --time budget, so a job
    whose GPUs go transiently idle (image build/compile, checkpoint flush, multinode
    coordination) is not reaped before its time limit. ``extra_srun_args`` are appended
    after the template flags for any per-job srun options not in the cluster template.
    """
    values: dict[str, object] = {
        "nodes": nodes,
        "ntasks": ntasks,
        "time_minutes": time_minutes,
        "total_gpus": total_gpus,
        "gpus_per_node": gpus_per_node,
        "job_name": job_name,
        "image": image,
        "container_env": container_env,
        "home": Path.home().as_posix(),
        "partition": os.environ.get("SLURM_PARTITION", ""),
    }
    body = "\n".join(ln for ln in _load_template("srun.tmpl").splitlines() if not ln.lstrip().startswith("#"))
    # Exempt the job from the OccupiedIdleGPUsJobReaper for its full time budget so a job whose
    # GPUs go transiently idle (build/compile, checkpoint flush, multinode coordination) is not
    # reaped before its --time limit; exemptIdleTimeMins tracks the job's own time_minutes.
    reaper = "--comment=" + json.dumps(
        {"OccupiedIdleGPUsJobReaper": {"exemptIdleTimeMins": str(time_minutes), "reason": "other", "description": "native checkpoint CI job"}},
        separators=(",", ":"),
    )
    return [*shlex.split(string.Template(body).safe_substitute(values)), reaper, *(extra_srun_args or [])]


def _runner_gpus_per_node() -> int | None:
    """GPUs-per-node configured at runner setup (GPUS_PER_NODE env), or None if unset.

    Used as the default when --gpus-per-node isn't passed on the command line, so
    the count lives with the runner instead of the CI YAML.
    """
    raw = os.environ.get("GPUS_PER_NODE", "").strip()
    return int(raw) if raw else None


# Job: pretrain


_CHECKPOINT_SERVER_PORT = 41000


# Phase 1 trains from random init to this iteration and saves one checkpoint that
# phase 2 loads. Phase 2 then trains exactly as a standalone run does today,
# resuming from this iteration and saving every 250 (iters 250/500/750).
_PHASE1_TRAIN_ITERS = 100


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base; override wins on conflicts."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _flatten_cli(config: dict) -> list[str]:
    """Flatten a {flag: value} mapping into CLI tokens: ``True`` -> ``--flag``, a list ->
    ``--flag v1 v2``, ``False``/``None`` drops the flag, anything else -> ``--flag value``."""
    out: list[str] = []
    for key, val in config.items():
        if val is True:
            out.append(f"--{key}")
        elif val is False or val is None:
            continue
        elif isinstance(val, list):
            out.append(f"--{key}")
            out.extend(str(v) for v in val)
        else:
            out.extend([f"--{key}", str(val)])
    return out


@dataclass
class PretrainRunConfig:
    nodes: int
    gpus_per_node: int
    ckpt_dir: Path  # --save target
    tb_dir: Path
    datacache_dir: Path
    save_interval: int = 250
    num_checkpoints: int = 3
    train_iters: int | None = None  # LR-schedule horizon (lr_decay_iters defaults to it); None -> exit_interval
    ckpt_provider: str = "nvrx,native"
    phase: int = 2
    load_dir: Path | None = None  # --load source; None starts from random init
    synchronous_save: bool = False  # phase 1: stock torch_dist save (no server, durable on exit)
    config_profile: str = "pico"  # model profile (pico/nano/ultra): base.yaml + <profile>.yaml

    @property
    def scale(self) -> str:
        return f"{self.nodes}x{self.gpus_per_node}"


# Inline pretrain (in-process, no torchrun subprocess)


# Launch + perf post-processing


def _tee_run(cmd: list[str], log_file: Path, env: dict | None = None, *, append: bool = False) -> int:
    """Run cmd, tee combined stdout/stderr to log_file, return the exit code.

    append=True keeps prior content (tee -a) so a two-phase run's phase-1 and
    phase-2 output both land in one log. Raises on any code other than 0 or 3
    (3 = soft fail bubbled up by an in-srun epilog, e.g. an nvrx baseline mismatch).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[run_job] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    # srun+enroot doesn't forward container stdout through a PIPE, so tee via shell.
    tee_flag = "-a " if append else ""
    shell_cmd = f"set -o pipefail; {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {tee_flag}{shlex.quote(str(log_file))}"
    rc = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", env=env, check=False).returncode  # noqa: S602
    if rc not in (0, 3):
        raise RuntimeError(f"Job failed with exit code {rc} — full log: {log_file}")
    return rc


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 — integrity check, not security


# Perf log parsers

_RE_NATIVE_BW = re.compile(
    r"\[native_ckpt:cli\]\[(\d+)\]\[INFO\]\[perf\] Finished "
    r"bw_mbps=([0-9.]+).*?wr_tnsr_us=([0-9]+).*?total_written_bytes=([0-9]+)",
    re.DOTALL,
)

# Rank-0 peak GPU / peak CPU RSS, printed once at process exit by the training entry.
_RE_NATIVE_MEM = re.compile(r"\[native_ckpt:mem\] gpu_peak_bytes=(\d+) cpu_rss_bytes=(\d+)")

# Banner pretrain_hybrid.py prints once per phase; the parser scopes metrics to
# the phase-2 (load-from-checkpoint) section so phase-1 warmup samples don't count.
_PHASE2_MARKER = "[native_ckpt:phase] phase=2"


# Baseline comparison + reporting


# Baseline aggregation (update stage — no re-run)


class CheckpointServer:
    @staticmethod
    def _ensure_server_down(ctx: RunCtx, *, wait_flush: bool = True, fail_on_timeout: bool = True) -> None:
        """Spin the native checkpoint_server down via its /api/stop endpoint.

        On the success path (wait_flush + fail_on_timeout): wait for the async writes to flush,
        request the stop, and fail with a thread dump if it doesn't free the port. On a crash
        path both are False — stop best-effort without masking the original error. No-op
        off-worker, for nvrx (no server), and for valgrind (its prolog owns the server).
        """
        if not ctx.is_worker or ctx.provider != "native" or ctx.has("valgrind"):
            return
        if wait_flush:
            try:
                BaselineUpdater._wait_for_native_metadata_files(ctx.cfg, timeout_seconds=1800)
            except TimeoutError as ex:
                logger.info(f"[run_job] wait for native artifacts timed out: {ex}")
        if not CheckpointServer._stop_checkpoint_server() and fail_on_timeout:
            CheckpointServer._dump_server_threads()
            CheckpointServer._gcore_and_kill_server()
            raise RuntimeError(
                "checkpoint_server did not spin down after a clean-stop request — "
                "see the thread dump above and the core file in ci-results",
            )

    @staticmethod
    def _checkpoint_server_pids() -> list[str]:
        """PIDs of any running checkpoint_server on this host (incl. a valgrind-wrapped one).

        Single source of truth for "is a server alive". Matches the binary on the full command
        line (pgrep -f): the daemon's comm is truncated to 15 chars and a valgrind wrapper's comm
        isn't 'checkpoint_server' at all. The process outlives its listening port — Server::Stop
        closes the acceptor (freeing the port) well before it joins its threads and exits — so the
        process, not the port, is the reliable liveness signal.

        The pattern is bracketed (``[c]heckpoint_server``) so it doesn't match other ranks' own
        concurrent ``pgrep`` commands — every rank runs this at once, and a literal pattern would
        appear in each pgrep's cmdline and they'd all match each other.
        """
        try:
            out = subprocess.run(["pgrep", "-f", "[c]heckpoint_server"], capture_output=True, text=True, check=False)  # noqa: S607
        except OSError:
            return []
        return out.stdout.split()

    @staticmethod
    def _dump_server_threads() -> None:
        """Best-effort: dump a stack trace of every thread of any running checkpoint_server.

        Used when a server won't start (already running) or won't spin down, to make the stuck
        process diagnosable rather than force-killing it. A bare `gdb -p` on a valgrind-wrapped
        process shows valgrind's own frames, not the guest, so for those we attach through vgdb;
        otherwise plain gdb, else the kernel per-thread stacks under /proc. Never raises —
        diagnostics must not mask the real error.
        """
        pids = CheckpointServer._checkpoint_server_pids()
        if not pids:
            logger.info("[run_job] no checkpoint_server process found to dump")
            return
        gdb = shutil.which("gdb")
        vgdb = shutil.which("vgdb")
        for pid in pids:
            cmdline = Path(f"/proc/{pid}/cmdline")
            under_valgrind = cmdline.exists() and "valgrind" in cmdline.read_text(errors="replace")
            logger.info(f"[run_job] === checkpoint_server pid {pid} thread dump{' (via vgdb)' if under_valgrind else ''} ===")
            try:
                if gdb and under_valgrind and vgdb:
                    # A bare `gdb -p` on a valgrind process shows valgrind's frames, not the
                    # guest; attach through valgrind's gdbserver to see the real server threads.
                    bt = [gdb, "-batch", "-ex", f"target remote | {vgdb} --pid={pid}", "-ex", "thread apply all bt"]
                    subprocess.run(bt, check=False, timeout=120)
                elif gdb:
                    subprocess.run([gdb, "-p", pid, "-batch", "-ex", "thread apply all bt"], check=False, timeout=120)
                else:
                    for task in sorted(Path(f"/proc/{pid}/task").glob("*")):
                        with contextlib.suppress(OSError):
                            logger.info(f"[run_job] thread {task.name}:\n{(task / 'stack').read_text()}")
            except (OSError, subprocess.TimeoutExpired) as ex:
                logger.info(f"[run_job] thread dump for pid {pid} failed: {ex}")

    @staticmethod
    def _gcore_and_kill_server() -> None:
        """Snapshot a core file for any running checkpoint_server, then SIGKILL it.

        Called after a thread dump when the server won't spin down: the core file lets a
        post-mortem inspect live state (queue depths, in-flight RPCs) that a stack trace
        alone can't show, and the SIGKILL frees the port and shm so the next job isn't
        blocked by the leftover. Under valgrind the SIGKILL costs the leak report —
        accepted as the price of unblocking CI; the thread dump and core file remain.
        Mirrors `_dump_server_threads`' valgrind handling: a bare `gcore` against a
        valgrind-wrapped process captures valgrind's frames, so for those we drive
        gdb through vgdb. Core files land in RESULTS_DIR as
        `checkpoint_server.<pid>.core`. Never raises — diagnostics must not mask the
        original error.
        """
        pids = CheckpointServer._checkpoint_server_pids()
        if not pids:
            return
        gcore = shutil.which("gcore")
        gdb = shutil.which("gdb")
        vgdb = shutil.which("vgdb")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for pid in pids:
            cmdline = Path(f"/proc/{pid}/cmdline")
            under_valgrind = cmdline.exists() and "valgrind" in cmdline.read_text(errors="replace")
            core_path = RESULTS_DIR / f"checkpoint_server.{pid}.core"
            logger.info(f"[run_job] writing core file for pid {pid} -> {core_path}{' (via vgdb)' if under_valgrind else ''}")
            try:
                if gdb and under_valgrind and vgdb:
                    cmd = [gdb, "-batch", "-ex", f"target remote | {vgdb} --pid={pid}", "-ex", f"generate-core-file {core_path}"]
                    subprocess.run(cmd, check=False, timeout=300)
                elif gcore:
                    subprocess.run([gcore, "-o", str(core_path), pid], check=False, timeout=300)
                elif gdb:
                    subprocess.run([gdb, "-batch", "-p", pid, "-ex", f"generate-core-file {core_path}"], check=False, timeout=300)
                else:
                    logger.info(f"[run_job] no gcore/gdb available; skipping core dump for pid {pid}")
            except (OSError, subprocess.TimeoutExpired) as ex:
                logger.info(f"[run_job] core dump for pid {pid} failed: {ex}")
            try:
                os.kill(int(pid), signal.SIGKILL)
                logger.info(f"[run_job] sent SIGKILL to pid {pid}")
            except (OSError, ValueError) as ex:
                logger.info(f"[run_job] kill pid {pid} failed: {ex}")

    @staticmethod
    def require_no_checkpoint_server() -> None:
        """Fail if a checkpoint_server process is already running — a prior job's leftover.

        The daemon (``server_main.cpp``) calls ``setsid()`` and survives parent death. We never
        force-kill it (it may be mid-write or under valgrind, where a SIGKILL loses the leak
        report); instead we dump its threads and fail loudly so it is visible. Check the process,
        not the port: the server frees the port well before it exits, so a port check would miss a
        still-shutting-down process that still holds its shm. With no process running, clear any
        orphaned shm queues (``/dev/shm/checkpoint_request_*``) so the next server starts clean.
        """
        pids = CheckpointServer._checkpoint_server_pids()
        if pids:
            logger.info(f"[run_job] checkpoint_server already running (pids {', '.join(pids)})")
            CheckpointServer._dump_server_threads()
            raise RuntimeError(
                f"checkpoint_server already running (pids {', '.join(pids)}); refusing to start. "
                f"A previous job did not spin it down — see the thread dump above and stop it before retrying.",
            )
        for shm in Path("/dev/shm").glob("checkpoint_request_*"):  # noqa: S108 — named shm, not a temp file
            try:
                shm.unlink()
            except OSError as ex:
                logger.info(f"[run_job] could not remove stale shm {shm}: {ex}")

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _request_server_stop() -> None:
        """Best-effort GET /api/stop — sets the server's shutdown flag (no OS signals)."""
        if not CheckpointServer._is_port_in_use(_CHECKPOINT_SERVER_PORT):
            return
        url = f"http://127.0.0.1:{_CHECKPOINT_SERVER_PORT}/api/stop"
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            logger.info(f"[run_job]   /api/stop request failed: {exc}")

    @staticmethod
    def _stop_checkpoint_server(timeout: float = 30.0) -> bool:
        """Ask the checkpoint_server to stop and wait for the PROCESS to fully exit.

        Sends the server's own HTTP /api/stop (no OS signals, so it can't hit an unrelated
        process), then polls until no checkpoint_server process remains. Waits on the process,
        not the port: Server::Stop closes the listener — freeing the port — well before it joins
        its threads and releases its shm, so a port check reports "stopped" while the process is
        still alive and would strand its shm for the next job. Returns False if a process is still
        alive after `timeout`, so the caller can dump threads and fail.
        """
        if not CheckpointServer._checkpoint_server_pids():
            return True

        logger.info("[run_job] checkpoint_server running; requesting graceful stop via /api/stop")
        CheckpointServer._request_server_stop()

        deadline = time.monotonic() + timeout
        while CheckpointServer._checkpoint_server_pids() and time.monotonic() < deadline:
            time.sleep(0.25)
        if CheckpointServer._checkpoint_server_pids():
            logger.info(f"[run_job]   checkpoint_server process still alive after {timeout:.0f}s")
            return False
        logger.info("[run_job]   checkpoint_server stopped (process exited)")
        return True


class PerfReport:
    @staticmethod
    def _report_checkpoint_times(log_text: str) -> None:
        """Print the save-checkpoint times for each phase plus phase 2's load-checkpoint
        time. The two-phase run concatenates both processes' stdout into one log with a
        phase banner between them, so we split on it: phase 1 saves once (iter 100), phase
        2 loads that checkpoint and saves the perf-measured 250/500/750 checkpoints. With
        no banner (e.g. a single-phase inline run) everything is reported under phase 2.
        """
        split = log_text.rfind(_PHASE2_MARKER)
        phase1_text, phase2_text = (log_text[:split], log_text[split:]) if split != -1 else ("", log_text)

        for header, text, needle in (
            ("phase-1 save-checkpoint times", phase1_text, "save-checkpoint"),
            ("phase-2 save-checkpoint times", phase2_text, "save-checkpoint"),
            ("phase-2 load-checkpoint times", phase2_text, "load-checkpoint"),
        ):
            lines = [ln for ln in text.splitlines() if needle in ln]
            if not lines:
                continue
            logger.info(f"--- {header} ---")
            for ln in lines:
                logger.info(ln)
            logger.info("-" * (len(header) + 8))

    @staticmethod
    def parse_and_report(  # noqa: PLR0913
        job: str,
        config: str,
        flavor_tag: str,
        log_file: Path,
        baseline_file: Path,
        report_file: Path,
        update_baseline: bool,
        gpus_per_node: int,
    ) -> bool:
        """Parse perf metrics from log_file, compare to baseline, write report. Returns True if passed."""
        log_text = log_file.read_text(errors="replace")

        PerfReport._report_checkpoint_times(log_text)

        metrics = PerfReport._parse_metrics(log_text, job)

        if not metrics:
            if job == "pretrain-nvrx":
                # nvrx provider doesn't emit parseable metrics; times printed above.
                return True
            if in_srun():
                raise RuntimeError(f"no perf metrics found in {log_file}")

        if update_baseline:
            BaselineUpdater._update_baseline_file(baseline_file, metrics, job, config, flavor_tag, gpus_per_node)
            return True

        baseline = {}
        if baseline_file.exists():
            with baseline_file.open() as f:
                baseline = yaml.safe_load(f)

        comparison, passed = PerfReport._compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
        PerfReport._print_perf_summary(comparison, passed, metrics)

        report = {
            "job": job,
            "config": config,
            "flavor_tag": flavor_tag,
            "gpus_per_node": gpus_per_node,
            "log": str(log_file),
            "baseline": str(baseline_file),
            "passed": passed,
            "metrics": {k: {kk: vv for kk, vv in v.items() if kk not in ("samples", "ranks")} for k, v in metrics.items()},
            "comparison": comparison,
        }
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        return passed

    @staticmethod
    def _make_metric(samples: list, unit: str, higher_is_better: bool | None, ranks: list | None = None) -> dict:
        return {
            "samples": samples,
            # Per-sample originating rank (aligned with samples), so the summary can
            # surface per-rank / per-socket structure (e.g. NUMA-driven bimodality).
            "ranks": ranks,
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "unit": unit,
            "higher_is_better": higher_is_better,
        }

    @staticmethod
    def _parse_metrics(log_text: str, _job: str) -> dict:
        """Parse all perf metrics for the given job type from log_text."""
        result = {}

        # Scope to phase 2 so phase-1 warmup checkpoint/memory samples never count.
        marker = log_text.rfind(_PHASE2_MARKER)
        if marker != -1:
            log_text = log_text[marker:]

        bw_matches = list(_RE_NATIVE_BW.finditer(log_text))
        if bw_matches:
            ranks = [int(m.group(1)) for m in bw_matches]
            bw_samples = [float(m.group(2)) for m in bw_matches]
            result.update(
                {
                    "bw_mbps": PerfReport._make_metric(bw_samples, "MB/s", higher_is_better=True, ranks=ranks),
                    "wr_tnsr_us": PerfReport._make_metric([int(m.group(3)) for m in bw_matches], "us", higher_is_better=False, ranks=ranks),
                    "total_written_bytes": PerfReport._make_metric([int(m.group(4)) for m in bw_matches], "B", None, ranks=ranks),
                },
            )
            # Per-rank bandwidth imbalance (informational): fastest/slowest rank mean.
            # Reaches ~1.48 on h100 from per-rank D2H noise alone, so it is NOT gated.
            per_rank: dict[int, list[float]] = {}
            for r, v in zip(ranks, bw_samples):
                per_rank.setdefault(r, []).append(v)
            ordered_ranks = sorted(per_rank)
            rank_means = [statistics.mean(per_rank[r]) for r in ordered_ranks]
            if len(rank_means) > 1 and min(rank_means) > 0:
                result["bw_rank_imbalance"] = PerfReport._make_metric([max(rank_means) / min(rank_means)], "ratio", higher_is_better=None)
                # Socket imbalance (gated): ratio of the two rank-halves' means. Ranks
                # are GPU-socket-ordered (0..N/2-1 on NUMA 0, rest on NUMA 1), so the
                # halves are the two sockets; averaging over a socket cancels per-rank
                # noise, leaving the per-socket split the bimodality bug produced.
                half = len(rank_means) // 2
                h0 = statistics.mean(rank_means[:half])
                h1 = statistics.mean(rank_means[half:])
                if min(h0, h1) > 0:
                    result["bw_socket_imbalance"] = PerfReport._make_metric([max(h0, h1) / min(h0, h1)], "ratio", higher_is_better=None)

        mem_matches = list(_RE_NATIVE_MEM.finditer(log_text))
        if mem_matches:
            result.update(
                {
                    "gpu_peak_bytes": PerfReport._make_metric([int(m.group(1)) for m in mem_matches], "B", higher_is_better=False),
                    # "cpu_rss_bytes": PerfReport._make_metric([int(m.group(2)) for m in mem_matches], "B", higher_is_better=False),
                },
            )

        return result

    @staticmethod
    def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
        baseline_metrics = baseline.get("metrics", {})
        results = {}
        all_passed = True

        for metric_name, measured in metrics.items():
            higher_is_better = measured.get("higher_is_better")
            measured_val = measured.get("mean")
            if measured_val is None:
                continue
            if metric_name == "bw_socket_imbalance":
                # One-sided gate: the two sockets must stay balanced. A breach means a
                # NUMA mis-placement has reintroduced the per-socket bimodality. Only an
                # upper bound is meaningful (1.0 is perfect), so no symmetric tolerance.
                passed = measured_val <= MAX_BW_SOCKET_IMBALANCE
                if not passed:
                    all_passed = False
                results[metric_name] = {
                    "status": "PASS" if passed else "FAIL",
                    "measured": measured_val,
                    "threshold": MAX_BW_SOCKET_IMBALANCE,
                    "unit": measured.get("unit", ""),
                }
                continue
            if metric_name in baseline_metrics:
                baseline_val = baseline_metrics[metric_name]
                tolerances = baseline.get("tolerances") or {}
                # Info-only metrics (no direction) must match exactly; directional perf metrics use default_tolerance.
                tol = tolerances.get(metric_name, 0.0 if higher_is_better is None else default_tolerance)
                lo = baseline_val * (1.0 - tol)
                hi = baseline_val * (1.0 + tol)
                passed = lo <= measured_val <= hi
                pct_change = (measured_val - baseline_val) / baseline_val * 100 if baseline_val else 0.0
                if not passed:
                    all_passed = False
                results[metric_name] = {
                    "status": "PASS" if passed else "FAIL",
                    "measured": measured_val,
                    "baseline": baseline_val,
                    "lo": lo,
                    "hi": hi,
                    "pct_change": round(pct_change, 2),
                    "tolerance_pct": round(tol * 100, 1),
                    "unit": measured.get("unit", ""),
                    "higher_is_better": higher_is_better,
                    "min": measured.get("min"),
                    "max": measured.get("max"),
                    "median": measured.get("median"),
                    "n_samples": len(measured.get("samples", [])),
                }
            elif higher_is_better is None:
                results[metric_name] = {"status": "info", "measured": measured_val}
            else:
                results[metric_name] = {
                    "status": "no_baseline",
                    "measured": measured_val,
                    "unit": measured.get("unit", ""),
                    "higher_is_better": higher_is_better,
                    "min": measured.get("min"),
                    "max": measured.get("max"),
                    "median": measured.get("median"),
                    "n_samples": len(measured.get("samples", [])),
                }
        return results, all_passed

    @staticmethod
    def _print_perf_summary(comparison: dict, passed: bool, metrics: dict | None = None) -> None:
        bar = "=" * 70
        logger.info(f"\n{bar}")
        logger.info(" Performance Results")
        logger.info(bar)
        for name, data in comparison.items():
            status = data.get("status", "?")
            measured = data.get("measured")
            baseline = data.get("baseline")
            unit = data.get("unit", "")
            pct = data.get("pct_change")
            if name == "bw_socket_imbalance" and status in ("PASS", "FAIL"):
                thr = data.get("threshold")
                note = "" if status == "PASS" else "  <-- per-socket bimodality regression"
                logger.info(f"  [{status}] {name}: {measured:.3f} (max allowed {thr}; ratio of the two NUMA sockets' mean BW){note}")
            elif status == "info":
                logger.info(f"  {name}: {measured:.4f} {unit}  [informational]")
            elif status == "no_baseline":
                logger.info(f"  {name}: {measured:.4f} {unit}  [no baseline]")
            elif status in ("PASS", "FAIL"):
                sign = "+" if pct and pct >= 0 else ""
                pct_str = f"{sign}{pct:.1f}%" if pct is not None else "N/A"
                tol = data.get("tolerance_pct", "?")
                samples = data.get("n_samples", "?")
                logger.info(
                    f"  [{status}] {name}: {measured:.4f} {unit}  (baseline={baseline:.4f}, Δ={pct_str}, tol=±{tol}%, n={samples})",
                )
                # The PASS/FAIL above is on the mean; print the underlying samples so
                # multi-modal behaviour (e.g. a per-socket NUMA split between ranks)
                # is visible rather than averaged away.
                PerfReport._print_sample_detail(metrics, name)
        logger.info(bar)
        logger.info(f" Overall: {'PASS' if passed else 'FAIL'}")
        logger.info(f"{bar}\n")

    @staticmethod
    def _print_sample_detail(metrics: dict | None, name: str) -> None:
        """Show the individual samples behind a metric: spread, per-rank means, and
        the full sorted list. Reveals run-to-run / per-rank modality the mean hides."""
        metric = (metrics or {}).get(name)
        if not metric:
            return
        # Only directional perf metrics vary meaningfully; exact info metrics
        # (e.g. total_written_bytes, higher_is_better=None) are constant per rank.
        if metric.get("higher_is_better") is None:
            return
        samples = metric.get("samples") or []
        if len(samples) < 2:
            return
        ranks = metric.get("ranks")
        f = lambda v: f"{v:.0f}"  # noqa: E731 — these are MB/s or microseconds; integer precision is plenty
        logger.info(f"         spread: min={f(min(samples))} median={f(statistics.median(samples))} max={f(max(samples))}")
        if ranks and len(ranks) == len(samples):
            by_rank: dict[int, list] = {}
            for r, s in zip(ranks, samples):
                by_rank.setdefault(r, []).append(s)
            per_rank = "  ".join(f"r{r}={f(statistics.mean(by_rank[r]))}" for r in sorted(by_rank))
            logger.info(f"         per-rank mean: {per_rank}")
        logger.info(f"         all values: {' '.join(f(s) for s in samples)}")

    @staticmethod
    def _compare_iters_to_baseline(cfg: str, provider: str) -> None:
        """Re-derive reference.metadata.info per iter and diff against the baseline snapshot.

        For each ``ci-runs/<cfg>/checkpoints/iter_*`` directory:
          1. Run ``distcp_trace`` (via ``scripts/metadata_compare.py``) to compute
             per-tensor md5sums and write ``reference.metadata.info`` next to the
             distcp shards. For provider=native this also verifies the
             checkpoint_server's ``0.metadata.info`` claims against on-disk truth.
          2. Diff that file against
             ``megatron/native/metadata/<cfg>_checkpoints_iter_NNNNNNN/reference.metadata.info``
             — the nvrx-derived baseline checked into the repo.

        Provider semantics on mismatch:
          * ``pretrain-nvrx`` (provider=nvrx, self-consistency): warn and
            ``sys.exit(3)`` so the matching ``update-baselines`` job stays
            triggerable. The gitlab YAML treats exit 3 as ``allow_failure``.
          * ``pretrain-native`` (provider=native, cross-provider): hard-fail.
            The nvrx-derived baseline is the cross-provider equality contract;
            a divergence is a real bug.

        Structural errors (missing inputs, native-server claims diverging from
        on-disk truth, distcp_trace crashed, ...) always hard-fail regardless of
        provider — metadata_compare returns 1 for these.

        Runs each compare as a subprocess so ``megatron.native.debug``'s transitive
        import of ``megatron.core`` → ``torch._inductor`` reads the same
        ``TORCHINDUCTOR_WORKER_START`` overlay we hand the training subprocess;
        importing in-process from the in-srun parent (which doesn't have that env)
        trips an ``Invalid start method`` assertion in newer torch builds.
        """
        strict_on_mismatch = provider == "native"

        ckpt_root = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
        metadata_root = MEGATRON_ROOT / "megatron/native/metadata"
        iter_dirs = sorted(ckpt_root.glob("iter_*"))
        if not iter_dirs:
            raise RuntimeError(f"no iter_* dirs in {ckpt_root}")
        # Thread MEGATRON_ROOT onto PYTHONPATH so `-m megatron.native.scripts.*` resolves
        # regardless of cwd (CI launches from $CI_PROJECT_DIR; manual runs don't).
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{MEGATRON_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(MEGATRON_ROOT)
        # srun --container-env can mangle TORCHINDUCTOR_WORKER_START into a value newer
        # torch._inductor rejects; set it explicitly for the compare subprocesses.
        env["TORCHINDUCTOR_WORKER_START"] = "fork"
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        failures: list[str] = []
        mismatches: list[str] = []
        for iter_dir in iter_dirs:
            iter_str = iter_dir.name.removeprefix("iter_")
            baseline_dir = metadata_root / f"{cfg}_checkpoints_iter_{iter_str}"
            logger.info(f"[run_job] baseline-compare {iter_dir.name} (provider={provider})")
            rc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "megatron.native.scripts.metadata_compare",
                    "--checkpoint-dir",
                    str(iter_dir),
                    "--baseline-dir",
                    str(baseline_dir),
                    "--provider",
                    provider,
                ],
                env=env,
                check=False,
            ).returncode
            if rc == 0:
                continue
            if rc == 3 and not strict_on_mismatch:  # noqa: PLR2004 — 3 = baseline mismatch (see metadata_compare)
                mismatches.append(iter_dir.name)
            else:
                failures.append(f"{iter_dir.name} (rc={rc})")

        if failures:
            raise RuntimeError(
                f"baseline-compare failed for {len(failures)} iter(s): {'; '.join(failures)}; see stderr above for per-finding detail",
            )
        if mismatches:
            logger.info(
                f"[run_job] WARN: {provider} baseline mismatch for {len(mismatches)} iter(s): {', '.join(mismatches)} "
                f"— run the matching update-baselines job to refresh.",
            )
            sys.exit(3)


class BaselineUpdater:
    @staticmethod
    def _baseline_run_cmds(job: str, config: str, flavor_tag: str, gpus_per_node: int) -> tuple[str, str]:
        gflag = f"--gpus-per-node {gpus_per_node}"
        flavor_flags = "".join(f" --flavor {f}" for f in flavor_tag.split("-") if f)
        if job == "ckptbench":
            script = "python3 megatron/native/ci/scripts/ckptbench.py"
            flags = f"--slurm {gflag}"
        else:
            script = "python3 megatron/native/ci/scripts/run_job.py"
            # config is "NxG"; map the node count to the profile that runs at it in CI
            # (pico single-node, nano at 8, ultra at 16) so the reproducer is runnable.
            nodes = int(config.split("x", 1)[0]) if "x" in config else 1
            profile = {8: "nano", 16: "ultra"}.get(nodes, "pico")
            cflag = f" --config {profile}" if profile != "pico" else ""
            scale = f"--nodes {nodes}" if nodes > 1 else "--slurm"
            flags = f"{job} {scale}{cflag} {gflag}{flavor_flags}"
        return f"{script} {flags}", f"{script} {flags} --update-baseline"

    @staticmethod
    def _round_to_sig_figs(value: float, sig_figs: int) -> float:
        """Round so only the leftmost `sig_figs` digits are non-zero (e.g. 5742.485 -> 5700.0)."""
        if value == 0 or not math.isfinite(value):
            return float(value)
        magnitude = math.floor(math.log10(abs(value)))
        return round(value, sig_figs - 1 - magnitude)

    @staticmethod
    def _round_for_baseline(mean: float, tolerance: float, higher_is_better: bool | None) -> float | int:
        """Round a measured mean to the precision implied by its tolerance.

        5% (the default for perf metrics) → 2 sig figs so values are easy to read at a glance.
        Exact match (tolerance==0, e.g. info metrics like total_written_bytes) → preserve as int.
        Custom tolerances → 6 decimals, since the user picked a non-default precision deliberately.
        """
        if tolerance == 0.0:
            return int(mean) if mean == int(mean) else round(mean, 6)
        if tolerance == PERF_TOLERANCE:
            rounded = BaselineUpdater._round_to_sig_figs(mean, 2)
            # Info-only metrics that landed in the perf-default branch (no old baseline) still
            # serialize cleanly as int when the rounded value is integral.
            if higher_is_better is None and rounded == int(rounded):
                return int(rounded)
            return rounded
        return round(mean, 6)

    @staticmethod
    def _update_baseline_file(baseline_file: Path, metrics: dict, job: str, config: str, flavor_tag: str, gpus_per_node: int) -> None:  # noqa: PLR0913
        if baseline_file.exists():
            with baseline_file.open() as f:
                baseline = yaml.safe_load(f)
        else:
            run_cmd, update_cmd = BaselineUpdater._baseline_run_cmds(job, config, flavor_tag, gpus_per_node)
            baseline = {
                "description": "auto-generated by CI",
                "run_cmd": run_cmd,
                "update_cmd": update_cmd,
                "metrics": {},
            }

        existing = baseline.setdefault("metrics", {})
        tolerances = baseline.get("tolerances") or {}
        changed: list[str] = []
        for name, data in metrics.items():
            mean = data.get("mean")
            if mean is None:
                continue
            old_val = existing.get(name)
            higher_is_better = data.get("higher_is_better")
            # Effective tolerance mirrors _compare_to_baseline; for new metrics (no old_val) we
            # fall back to higher_is_better since type-of-baseline isn't available yet.
            if name in tolerances:
                tol = tolerances[name]
            elif old_val is not None:
                tol = 0.0 if isinstance(old_val, int) else PERF_TOLERANCE
            else:
                tol = 0.0 if higher_is_better is None else PERF_TOLERANCE

            new_val = BaselineUpdater._round_for_baseline(mean, tol, higher_is_better)
            if old_val is None or abs(new_val - old_val) / max(abs(old_val), 1e-9) > _BASELINE_CHANGE_THRESHOLD:
                existing[name] = new_val
                changed.append(f"  {name}: {old_val} -> {new_val}")

        if not changed:
            return

        for _line in changed:
            pass
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with baseline_file.open("w") as f:
            yaml.dump(baseline, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _parse_job_filter(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
        """Return (filter_job, filter_config, filter_flavor) from --filter or CI_JOB_NAME.

        --filter format:   <job>[:<config>[:<flavor>]]
        CI_JOB_NAME format: <platform>:<job>[:<config>[:<flavor>]][:update_baselines]
          gb200:ckptbench:update_baselines              → (ckptbench, None, None)
          gb200:native:1x4:[DETR]:update_baselines      → (pretrain-native, 1x4, deterministic)

        Generated job names abbreviate two segments: the provider (native/nvrx, dropping
        the "pretrain-" prefix) and the flavor ([PERF]/[DETR]). Map both back to the real
        values (pretrain-native/pretrain-nvrx, perf/deterministic) so the report-file globs
        and baseline matching below — which key on the real job/flavor — still work.
        """
        labels = {"[PERF]": "perf", "[DETR]": "deterministic"}
        providers = {"native": "pretrain-native", "nvrx": "pretrain-nvrx"}

        def flavor(seg: str | None) -> str | None:
            return labels.get(seg, seg)

        def job_of(seg: str) -> str:
            return providers.get(seg, seg)

        if args.filter:
            job_name, *rest = args.filter.split(":")
            return job_of(job_name), (rest[0] if rest else None), flavor(rest[1] if len(rest) > 1 else None)

        ci_job_name = os.environ.get("CI_JOB_NAME", "")
        if ci_job_name:
            _, job_name, *rest = ci_job_name.removesuffix(":update_baselines").split(":")
            return job_of(job_name), (rest[0] if rest else None), flavor(rest[1] if len(rest) > 1 else None)

        return None, None, None

    @staticmethod
    def run_update_baselines(args: argparse.Namespace) -> None:
        """Update baseline YAMLs from *_report.json files in RESULTS_DIR without re-running jobs.

        Job/config are auto-detected from CI_JOB_NAME; override with --filter.
        """
        machine = _platform_tag()
        filter_job, filter_config, filter_flavor = BaselineUpdater._parse_job_filter(args)

        # Seed/update perf baselines from *_report.json (perf, deterministic, ckptbench all
        # emit one). nvrx emits none: skip to the md5 snapshot refresh below.
        if filter_job != "pretrain-nvrx":
            job_glob = filter_job or "*"
            reports = sorted(RESULTS_DIR.glob(f"{machine}_{job_glob}_*_report.json"))
            if not reports:
                filter_desc = ":".join(p for p in (filter_job, filter_config, filter_flavor) if p) or "all jobs"
                raise RuntimeError(
                    f"no report files found in {RESULTS_DIR} matching '{machine}_{job_glob}_*_report.json' "
                    f"(filter: {filter_desc}) — did the test stage run first?",
                )

            for report_file in reports:
                with report_file.open() as f:
                    report = json.load(f)

                job = report.get("job")
                config = report.get("config")
                flavor_tag = report.get("flavor_tag", "")
                gpus_per_node = report.get("gpus_per_node", args.gpus_per_node)
                metrics = report.get("metrics", {})

                if not job or not config or not metrics:
                    continue
                if filter_config is not None and config != filter_config:
                    continue
                if filter_flavor is not None and flavor_tag != filter_flavor:
                    continue

                baseline_file = BASELINES_DIR / f"{_artifact_stem(machine, job, flavor_tag, config)}.yaml"
                BaselineUpdater._update_baseline_file(baseline_file, metrics, job, config, flavor_tag, gpus_per_node)

        # The metadata md5 snapshot is only produced by deterministic runs. nvrx
        # always runs deterministic; native does so only under the deterministic flavor.
        flavor_is_deterministic = "deterministic" in (filter_flavor or "").split("-")
        if filter_job == "pretrain-native" and flavor_is_deterministic:
            BaselineUpdater._update_native_metadata_files(filter_config)
        elif filter_job == "pretrain-nvrx":
            BaselineUpdater._update_nvrx_baseline_files(filter_config)

    @staticmethod
    def _foreach_native_metadata_files(  # noqa: C901
        filter_config: str | None,
        callback: Callable[[Path, Path], None],
        wait_seconds: int = 0,
    ) -> None:
        """Run callback over each native metadata file to snapshot; with wait_seconds, poll
        for each first (written asynchronously). Some files are waited on but not copied."""
        metadata_root = MEGATRON_ROOT / "megatron/native/metadata"
        ci_runs_root = MEGATRON_ROOT / "megatron/native/ci/ci-runs"
        if not ci_runs_root.is_dir():
            raise FileNotFoundError(f"ci-runs root not found: {ci_runs_root}")

        files_to_copy = ("registry.bin", "*.txt")
        # Gate on .metadata, the last file a finalizing checkpoint writes, so the server is
        # not stopped mid-write; common.pt feeds the equality-compare. Neither is copied.
        wait_only_patterns = (".metadata", "common.pt")
        iter_pattern = re.compile(r"^iter_(\d{7})$")
        cfg_pattern = re.compile(r"^\d+x\d+$")

        cfgs = (
            [filter_config]
            if filter_config is not None
            else sorted(d.name for d in ci_runs_root.iterdir() if d.is_dir() and cfg_pattern.match(d.name))
        )

        for cfg in cfgs:
            ckpt_root = ci_runs_root / cfg / "checkpoints"
            if not ckpt_root.is_dir():
                raise FileNotFoundError(
                    f"checkpoint dir not found for {cfg}: {ckpt_root} — did the pretrain-native test stage run on this runner?",
                )

            iter_dirs = [d for d in sorted(ckpt_root.iterdir()) if d.is_dir() and iter_pattern.match(d.name)]
            if not iter_dirs:
                raise FileNotFoundError(f"no iter_* checkpoints found under {ckpt_root}")

            for src_iter_dir in iter_dirs:
                iter_str = iter_pattern.match(src_iter_dir.name).group(1)
                dst_dir = metadata_root / f"{cfg}_checkpoints_iter_{iter_str}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                for pattern in files_to_copy:
                    if wait_seconds > 0:
                        wait_for_file(src_iter_dir / pattern, wait_seconds)
                    matches = sorted(src_iter_dir.glob(pattern))
                    if not matches:
                        raise FileNotFoundError(f"no files matching {pattern!r} in {src_iter_dir}")
                    for src in matches:
                        dst = dst_dir / src.name
                        callback(src, dst)
                if wait_seconds > 0:
                    for pattern in wait_only_patterns:
                        wait_for_file(src_iter_dir / pattern, wait_seconds)

    @staticmethod
    def _wait_for_native_metadata_files(cfg: str, timeout_seconds: int) -> None:
        """Block until each checkpoint's final artifacts are flushed, so the later update
        stage finds them."""

        def _log(src: Path, _dst: Path) -> None:
            logger.info(f"[run_job] Found native metadata file: {src}")

        BaselineUpdater._foreach_native_metadata_files(cfg, _log, wait_seconds=timeout_seconds)

    @staticmethod
    def _update_native_metadata_files(filter_config: str | None) -> None:
        """Mirror each iter's snapshot files into the metadata/ baseline dirs; with a
        filter, only that scale."""

        def log_and_copy(src: Path, dst: Path) -> None:
            shutil.copy2(src, dst)
            logger.info(f"[run_job] updated {dst} from {src}")

        BaselineUpdater._foreach_native_metadata_files(filter_config, log_and_copy)

    @staticmethod
    def _update_nvrx_baseline_files(filter_config: str | None) -> None:
        """Refresh reference.metadata.info under megatron/native/metadata/ from pretrain-nvrx output.

        The nvrx-derived reference.metadata.info is the per-tensor md5 baseline that
        both pretrain-nvrx (self-consistency check) and pretrain-native (cross-provider
        check) compare against. Mirrors ci-runs/<cfg>/checkpoints/iter_NNNNNNN/reference.metadata.info
        to metadata/<cfg>_checkpoints_iter_NNNNNNN/reference.metadata.info, one file per iter.
        The source file is produced by the in-srun _compare_iters_to_baseline → distcp_trace
        pass that runs after pretrain.
        """
        metadata_root = MEGATRON_ROOT / "megatron/native/metadata"
        ci_runs_root = MEGATRON_ROOT / "megatron/native/ci/ci-runs"
        iter_pattern = re.compile(r"^iter_(\d{7})$")
        cfg_pattern = re.compile(r"^\d+x\d+$")

        cfgs = (
            [filter_config]
            if filter_config is not None
            else sorted(d.name for d in ci_runs_root.iterdir() if d.is_dir() and cfg_pattern.match(d.name))
        )

        for cfg in cfgs:
            ckpt_root = ci_runs_root / cfg / "checkpoints"
            if not ckpt_root.is_dir():
                raise FileNotFoundError(
                    f"checkpoint dir not found for {cfg}: {ckpt_root} — did the pretrain-nvrx test stage run on this runner?",
                )
            iter_dirs = [d for d in sorted(ckpt_root.iterdir()) if d.is_dir() and iter_pattern.match(d.name)]
            if not iter_dirs:
                raise FileNotFoundError(f"no iter_* checkpoints found under {ckpt_root}")
            for src_iter_dir in iter_dirs:
                iter_str = iter_pattern.match(src_iter_dir.name).group(1)
                dst_dir = metadata_root / f"{cfg}_checkpoints_iter_{iter_str}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                src = src_iter_dir / "reference.metadata.info"
                if not src.is_file():
                    raise FileNotFoundError(
                        f"missing reference.metadata.info at {src} — did pretrain-nvrx run its baseline-compare step?",
                    )
                dst = dst_dir / "reference.metadata.info"
                shutil.copy2(src, dst)
                logger.info(f"[run_job] updated {dst} from {src}")


class PretrainJob:
    @staticmethod
    def run_pretrain(args: argparse.Namespace) -> None:
        provider = _ckpt_provider(args.job)
        cfg = f"{args.nodes}x{args.gpus_per_node}"
        extra_args = getattr(args, "extra_args", None)

        if in_srun():
            # GPUs-per-node arrives on argv (round-tripping it through the container env corrupted it).
            nodes = int(os.environ.get("SLURM_NNODES", "1"))
            cfg = f"{nodes}x{args.gpus_per_node}"
            ctx = RunCtx(args, args.job, cfg, provider, _platform_tag(), args.flavors)
            with _flavored_run(ctx):
                # Two sequential processes in one srun task share a single checkpoint_server
                # lifecycle and perf epilog: phase 1 trains from random init and saves; phase 2
                # loads it and saves the perf-measured checkpoints.
                common = {"ckpt_provider": provider, "flavors": args.flavors, "extra_args": extra_args, "config_profile": args.config}
                cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, nodes, args.gpus_per_node, phase=1, **common)
                subprocess.run(cmd, env=env, check=True)
                PretrainJob._wait_for_phase1_checkpoint(cfg)
                cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, nodes, args.gpus_per_node, phase=2, **common)
                subprocess.run(cmd, env=env, check=True)
            return

        # nvrx deterministic via SLURM/multinode runs the two-run cross-run determinism check;
        # perf falls through to a single measured run (the check needs deterministic references).
        is_deterministic = any(f.name == "deterministic" for f in args.flavors)
        if provider == "nvrx" and is_deterministic and (args.nodes > 1 or args.slurm):
            PretrainJob._run_pretrain_determinism_check(args, cfg)
            return

        # Clear stale checkpoints once in the launcher so in-srun tasks don't race on rmtree.
        # Both phase dirs are cleared so phase 1 always starts from random init.
        for stale in (
            MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints",
            PretrainJob._phase1_ckpt_dir(cfg),
        ):
            if stale.exists():
                shutil.rmtree(stale)

        ctx = RunCtx(args, args.job, cfg, provider, _platform_tag(), args.flavors)
        ctx.log_file = RESULTS_DIR / f"{ctx.stem}.log"
        if args.nodes > 1 or args.slurm:
            # Launcher: training + worker epilogs run in the srun task; this launcher tees the log
            # and runs the perf epilog from it.
            cmd = PretrainJob._build_pretrain_srun_cmd(
                cfg, args.nodes, args.gpus_per_node, args.job, args.flavors, extra_args=extra_args, config_profile=args.config,
            )
            with _flavored_run(ctx):
                ctx.soft_fail = _tee_run(cmd, ctx.log_file) == _SOFT_FAIL_RC
        elif args.gpus_per_node > 1:
            # Multi-GPU devbox: torchrun subprocess so worker output is captured in the log.
            # Both phases tee into one log; the perf parser scopes metrics to phase 2.
            with _flavored_run(ctx):
                common = {
                    "use_torchrun": True,
                    "ckpt_provider": provider,
                    "flavors": args.flavors,
                    "extra_args": extra_args,
                    "config_profile": args.config,
                }
                cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, args.nodes, args.gpus_per_node, phase=1, **common)
                _tee_run(cmd, ctx.log_file, env=env)
                PretrainJob._wait_for_phase1_checkpoint(cfg)
                cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, args.nodes, args.gpus_per_node, phase=2, **common)
                ctx.soft_fail = _tee_run(cmd, ctx.log_file, env=env, append=True) == _SOFT_FAIL_RC
        else:
            # Single-GPU inline: call pretrain() in-process (debugger-friendly).
            with _flavored_run(ctx):
                PretrainJob._launch_pretrain_inline(ctx)

    @staticmethod
    def _build_pretrain_srun_cmd(  # noqa: PLR0913 — distinct srun inputs, not worth a struct
        cfg: str,
        nodes: int,
        gpus_per_node: int,
        job: str,
        flavors: list[Flavor],
        extra_args: list[str] | None = None,
        config_profile: str = "pico",
    ) -> list[str]:
        image = _image_path().as_posix()
        triton_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/triton-cache"
        total_tasks = nodes * gpus_per_node

        recipe = Recipe(opts=[], env=Flavor.base_env(triton_dir))
        _apply_flavors(recipe, flavors)
        # KEY=VALUE for the run env; VCPKG_ROOT and any build_env (already exported in
        # main()) are forwarded by name — forwarding them by value has hit KEY=VALUE
        # parsing ambiguity in pyxis/enroot.
        container_env = ",".join(f"{k}={v}" for k, v in recipe.env.items())
        container_env += "," + ",".join(["VCPKG_ROOT", *recipe.build_env])

        return [
            *_build_srun_cmd(
                nodes=nodes,
                ntasks=total_tasks,
                time_minutes=recipe.srun_time_min,
                job_name=f"{job}_{cfg}_{os.environ.get('CI_PIPELINE_ID', 'local')}",
                image=image,
                container_env=container_env,
                gpus_per_node=gpus_per_node,
                total_gpus=total_tasks,
                extra_srun_args=recipe.srun_comments,
            ),
            # SLURM_PROCID will be set on each task, triggering in-srun mode.
            "python3",
            "-u",
            str(Path(__file__).resolve()),
            job,
            # Pass gpus-per-node + config profile + flavors on argv so the in-srun task
            # re-derives the same recipe from argv, not the container env (a KEY=VALUE
            # collision there corrupted forwarded values). --nodes is re-derived from
            # SLURM_NNODES in-srun, so it is not forwarded here.
            "--gpus-per-node",
            str(gpus_per_node),
            "--config",
            config_profile,
            *[tok for f in flavors for tok in ("--flavor", f.name)],
            *(extra_args or []),
        ]

    @staticmethod
    def _phase1_ckpt_dir(cfg: str) -> Path:
        """Phase-1's dedicated checkpoint dir, kept separate from the perf-measured
        checkpoints/ so the latter holds only the 250/500/750 phase-2 checkpoints."""
        return MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/phase1-checkpoints"

    @staticmethod
    def _wait_for_phase1_checkpoint(cfg: str, timeout_seconds: int = 1800) -> None:
        """Block until phase 1's checkpoint is durable, so the fresh phase-2 process can load it.

        The checkpoint_server flushes the DCP ``.metadata`` commit marker last (after the
        shards), so its presence means the checkpoint is complete on disk; metadata.json
        (the torch_dist format marker) is written synchronously by the phase-1 process
        before it exits, hence present by the time .metadata lands.
        """
        marker = PretrainJob._phase1_ckpt_dir(cfg) / f"iter_{_PHASE1_TRAIN_ITERS:07d}" / ".metadata"
        logger.info(f"[run_job] waiting for phase-1 checkpoint to flush: {marker}")
        wait_for_file(marker, timeout_seconds)

    @staticmethod
    def _pretrain_cmd_and_env(  # noqa: PLR0913
        cfg: str,
        nodes: int,
        gpus_per_node: int,
        use_torchrun: bool = False,
        ckpt_provider: str = "nvrx,native",
        flavors: list[Flavor] | None = None,
        extra_args: list[str] | None = None,
        phase: int = 2,
        config_profile: str = "pico",
    ) -> tuple[list[str], dict]:
        """Return (cmd, env) to run the in-process pretrain entry (run_job.py __pretrain_entry__) on a compute node.

        use_torchrun=True  — devbox multi-GPU; torchrun spawns one process per GPU.
        use_torchrun=False — in-srun; srun already placed one process per GPU.

        phase 1 trains from random init to _PHASE1_TRAIN_ITERS, saving into a separate
        phase1-checkpoints dir; phase 2 loads that checkpoint and saves the perf-measured
        250/500/750 checkpoints into the main checkpoints dir.
        """
        run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
        ckpt_dir = run_dir / "checkpoints"
        phase1_dir = PretrainJob._phase1_ckpt_dir(cfg)
        tb_dir = run_dir / "tensorboard"
        datacache = run_dir / "data-cache"
        triton_dir = run_dir / "triton-cache"
        for d in [ckpt_dir, phase1_dir, tb_dir, datacache, triton_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if phase == 1:
            config = PretrainJob._phase1_config(
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                ckpt_dir=phase1_dir,
                tb_dir=tb_dir,
                datacache_dir=datacache,
                ckpt_provider=ckpt_provider,
                config_profile=config_profile,
            )
        else:
            config = PretrainRunConfig(
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                ckpt_dir=ckpt_dir,
                tb_dir=tb_dir,
                datacache_dir=datacache,
                ckpt_provider=ckpt_provider,
                load_dir=phase1_dir,
                config_profile=config_profile,
            )
        opts = PretrainJob._pretrain_cli_args(config)
        recipe = Recipe(opts=opts, env=Flavor.base_env(triton_dir))
        _apply_flavors(recipe, flavors or [])
        opts = recipe.opts

        env = os.environ.copy()
        env.update(recipe.env)

        if use_torchrun:
            launcher = ["torchrun", "--nproc_per_node", str(gpus_per_node)]
        else:
            logger.info(
                f"[run_job] in-srun mode: RANK={env['RANK']} LOCAL_RANK={env['LOCAL_RANK']} WORLD_SIZE={env['WORLD_SIZE']}"
                f"MASTER_ADDR={env['MASTER_ADDR']} MASTER_PORT={env['MASTER_PORT']}",
            )
            launcher = ["python3", "-u"]

        # Re-invoke this script in __pretrain_entry__ mode so the CI markers live under
        # megatron/native and the repo-root training scripts stay unmodified. The phase follows the
        # sentinel; the entry forwards the remaining opts to pretrain() in-process.
        return [*launcher, str(SCRIPT_DIR / "run_job.py"), _PRETRAIN_ENTRY_CMD, str(phase), *opts, *(extra_args or [])], env

    @staticmethod
    def _phase1_config(**kwargs: object) -> PretrainRunConfig:
        """A phase-1 config: random init, train to _PHASE1_TRAIN_ITERS, one synchronous save.

        Phase 1 saves with the job's own provider (so the native job's phase 1 is a native
        save). megatron.native.tools.save writes a metadata.json marker and persists the
        ShardedObjects, so the native checkpoint is read back by the regular torch_dist
        load path in phase 2. Saving synchronously (no async persistent worker) makes it
        durable on process exit.
        """
        # lr_decay_iters defaults to --train-iters, so phase 1 must advertise the full run's
        # train_iters, else the LR schedule and the weights phase 2 resumes from diverge.
        full_train_iters = PretrainRunConfig.save_interval * PretrainRunConfig.num_checkpoints
        return PretrainRunConfig(
            save_interval=_PHASE1_TRAIN_ITERS,
            num_checkpoints=1,
            train_iters=full_train_iters,
            phase=1,
            load_dir=None,
            synchronous_save=True,
            **kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _pretrain_cli_args(config: PretrainRunConfig) -> list[str]:
        save_interval, num_checkpoints = config.save_interval, config.num_checkpoints
        exit_interval = save_interval * num_checkpoints
        # The LR schedule spans train_iters (lr_decay_iters defaults to it). Phase 1 exits
        # early (exit_interval) but keeps the full run's horizon so its trajectory matches a
        # single-phase run; phase 2 / single-phase leave train_iters None -> exit_interval.
        train_iters = config.train_iters if config.train_iters is not None else exit_interval
        seq_len = 1024

        # Phase 1 saves synchronously (no --async-save / persistent worker) so the
        # checkpoint is durable on process exit and phase 2 can --load it.
        if config.synchronous_save:
            checkpoint_opts = [
                f"--ckpt-provider={config.ckpt_provider}",
                "--ckpt-format",
                "torch_dist",
                "--ckpt-assume-constant-structure",
            ]
        else:
            checkpoint_opts = [
                f"--ckpt-provider={config.ckpt_provider}",
                "--ckpt-format",
                "torch_dist",
                "--ckpt-assume-constant-structure",
                "--async-save",
                "--use-persistent-ckpt-worker",
            ]
        checkpoint_opts += ["--save", str(config.ckpt_dir)]
        if config.load_dir is not None:
            checkpoint_opts += ["--load", str(config.load_dir)]
        checkpoint_opts += [
            "--save-interval",
            str(save_interval),
            "--save-retain-interval",
            str(save_interval),
        ]

        # Runtime defaults; a model config overrides shared keys. pico scales parallelism +
        # batch to GPU count, nano/ultra pin them; eval-interval tracks the full horizon.
        dynamic = {
            "tensor-model-parallel-size": config.gpus_per_node,
            "expert-model-parallel-size": config.gpus_per_node,
            "global-batch-size": config.nodes * config.gpus_per_node,
            "seq-length": seq_len,
            "max-position-embeddings": seq_len,
            "data-cache-path": str(config.datacache_dir),
            "tensorboard-dir": str(config.tb_dir),
            "log-interval": save_interval,
            "eval-interval": train_iters,
            "exit-interval": exit_interval,
            "train-iters": train_iters,
        }
        model = PretrainJob._load_model_config(config.config_profile)
        return _flatten_cli({**dynamic, **model}) + checkpoint_opts

    @staticmethod
    def _load_model_config(profile: str) -> dict:
        """Static model + training args from base.yaml, deep-merged with the named
        profile override <profile>.yaml (pico/nano/ultra)."""
        base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text()) or {}
        override = CONFIGS_DIR / f"{profile}.yaml"
        if not override.exists():
            raise FileNotFoundError(
                f"unknown config profile {profile!r}: {override} not found "
                f"(available: {', '.join(_available_profiles())})",
            )
        return _deep_merge(base, yaml.safe_load(override.read_text()) or {})

    @staticmethod
    def _launch_pretrain_inline(ctx: RunCtx) -> None:
        """Run pretrain in-process (no subprocess/torchrun) for single-GPU devbox runs.

        Output is tee'd to ctx.log_file so the perf epilog can parse it; run_pretrain
        runs spin-down + epilogs via _flavored_run afterwards.
        """
        args, cfg = ctx.args, ctx.cfg
        nodes, gpus_per_node = args.nodes, args.gpus_per_node
        run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
        ckpt_dir = run_dir / "checkpoints"
        tb_dir = run_dir / "tensorboard"
        datacache = run_dir / "data-cache"
        triton_dir = run_dir / "triton-cache"
        for d in [ckpt_dir, tb_dir, datacache, triton_dir]:
            d.mkdir(parents=True, exist_ok=True)

        opts = PretrainJob._pretrain_cli_args(
            PretrainRunConfig(
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                ckpt_dir=ckpt_dir,
                tb_dir=tb_dir,
                datacache_dir=datacache,
                save_interval=1,
                num_checkpoints=2,
                ckpt_provider=ctx.provider,
                config_profile=args.config,
            ),
        )
        recipe = Recipe(opts=opts, env=Flavor.base_env(triton_dir))
        _apply_flavors(recipe, args.flavors)
        opts = [*recipe.opts, *(getattr(args, "extra_args", None) or [])]
        # Applied before the in-process pretrain imports the model: MAMBA_DETERMINISTIC
        # (under the deterministic flavor) is read at import time.
        os.environ.update(recipe.env)
        # Single-GPU inline is a debugger smoke test: one short single-phase run with no phase
        # banner, so the perf parser scopes to the whole log (correct for one phase). The two-phase
        # resume handoff is exercised by the torchrun and srun paths.

        if args.skip_perf:
            ctx.log_file = None  # nothing for the perf epilog to parse
            PretrainJob._call_pretrain_inline(opts, nodes, gpus_per_node)
            return

        class _Tee:
            def __init__(self, stream: IO[str], logfp: IO[str]) -> None:
                self._stream, self._logfp = stream, logfp

            def write(self, data: str) -> int:
                self._stream.write(data)
                self._logfp.write(data)
                return len(data)

            def flush(self) -> None:
                self._stream.flush()
                self._logfp.flush()

            def fileno(self) -> int:
                return self._stream.fileno()

            def __getattr__(self, name: str) -> object:
                return getattr(self._stream, name)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with ctx.log_file.open("w") as log_fp:
            saved_stdout, saved_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(saved_stdout, log_fp)
            sys.stderr = _Tee(saved_stderr, log_fp)
            try:
                PretrainJob._call_pretrain_inline(opts, nodes, gpus_per_node)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                if rc != 0:
                    raise RuntimeError(f"pretrain exited with code {rc} — full log: {ctx.log_file}") from None
            finally:
                sys.stdout, sys.stderr = saved_stdout, saved_stderr

    @staticmethod
    def _call_pretrain_inline(opts: list[str], nodes: int, gpus_per_node: int) -> None:
        """Set sys.argv and call pretrain() directly.

        gpus_per_node==1: runs in the calling process (debugger-friendly).
        gpus_per_node>1:  uses torch.multiprocessing.spawn.
        """
        import time  # noqa: PLC0415

        if str(MEGATRON_ROOT) not in sys.path:
            sys.path.insert(0, str(MEGATRON_ROOT))

        program_start = time.time()

        if "MASTER_PORT" not in os.environ:
            import socket as _socket  # noqa: PLC0415

            with _socket.socket() as _s:
                _s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(_s.getsockname()[1])

        if gpus_per_node == 1:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            # Force num-workers=0 so DataLoader runs in-process with no subprocess spawning.
            clean = []
            skip = False
            for tok in opts:
                if skip:
                    skip = False
                    continue
                if tok == "--num-workers":
                    skip = True
                    continue
                clean.append(tok)
            opts = [*clean, "--num-workers", "0"]
            sys.argv = ["pretrain_mamba.py", *opts]
            PretrainJob._pretrain_main(program_start)
        else:
            import torch.multiprocessing as mp  # noqa: PLC0415

            mp.spawn(
                PretrainJob._pretrain_worker,
                args=(gpus_per_node, nodes, opts, program_start),
                nprocs=gpus_per_node,
                join=True,
            )

    @staticmethod
    def _pretrain_worker(local_rank: int, gpus_per_node: int, nodes: int, opts: list[str], program_start: float) -> None:
        """Entry point for each torch.multiprocessing.spawn worker (multi-GPU inline)."""
        os.environ["RANK"] = str(local_rank)  # single-node: global rank == local rank
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(nodes * gpus_per_node)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        sys.argv = ["pretrain_mamba.py", *opts]
        PretrainJob._pretrain_main(program_start)

    @staticmethod
    def _pretrain_main(program_start: float, ci_phase: int | None = None) -> None:
        """Mirror of pretrain_hybrid.py's __main__ block — calls pretrain() directly.

        Brackets pretrain() with the two CI-only stdout markers the perf parser reads, so every
        launch path emits them from one place: a phase banner up front (when ci_phase is set, so
        the parser can split the two-phase log) and, on exit, the rank-0 peak-GPU / peak-CPU-RSS
        marker. Both fire only on global rank 0; inline debug runs pass ci_phase=None (no banner).
        """
        import time  # noqa: PLC0415
        from functools import partial  # noqa: PLC0415

        from megatron.core.enums import ModelType  # noqa: PLC0415
        from megatron.training import inprocess_restart, pretrain, set_startup_timestamps  # noqa: PLC0415
        from megatron.training.argument_utils import (  # noqa: PLC0415
            hybrid_config_from_args,
            pretrain_cfg_container_from_args,
        )
        from megatron.training.arguments import parse_and_validate_args  # noqa: PLC0415

        try:
            from megatron.post_training.arguments import add_modelopt_args  # noqa: PLC0415

            has_modelopt = True
        except ImportError:
            has_modelopt = False

        import pretrain_hybrid as ph  # noqa: PLC0415
        from hybrid_builders import hybrid_builder  # noqa: PLC0415
        from model_provider import model_provider as _mp  # noqa: PLC0415

        rank0 = int(os.environ.get("RANK", "0")) == 0
        # Parser markers print to stdout: this subprocess has no logging handler and
        # the perf parser scans the tee'd stdout. Keep them (T201 suppressed).
        if rank0 and ci_phase is not None:
            print(f"[native_ckpt:phase] phase={ci_phase}", flush=True)  # noqa: T201

        set_startup_timestamps(program_start=program_start, main_entry=time.time())

        ph.train_valid_test_datasets_provider.is_distributed = True

        pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        args = parse_and_validate_args(
            extra_args_provider=add_modelopt_args if has_modelopt else None,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        )
        model_cfg = hybrid_config_from_args(args)
        full_config = pretrain_cfg_container_from_args(args, model_cfg)
        try:
            pretrain_fn(
                full_config,
                ph.train_valid_test_datasets_provider,
                partial(_mp, hybrid_builder),
                ModelType.encoder_or_decoder,
                ph.forward_step,
                store=store,
            )
        finally:
            # Peak GPU/CPU are cumulative, so one reading at exit equals the per-iteration max.
            if rank0:
                try:
                    import resource  # noqa: PLC0415

                    import torch  # noqa: PLC0415

                    gpu_peak_bytes = torch.cuda.max_memory_allocated()
                    cpu_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # ru_maxrss is KiB on Linux
                    print(f"[native_ckpt:mem] gpu_peak_bytes={gpu_peak_bytes} cpu_rss_bytes={cpu_rss_bytes}", flush=True)  # noqa: T201
                except Exception as exc:  # noqa: BLE001 — a metrics hiccup must never mask the training result
                    print(f"[native_ckpt:mem] unavailable: {exc}", flush=True)  # noqa: T201

    @staticmethod
    def _run_pretrain_determinism_check(args: argparse.Namespace, cfg: str) -> None:  # noqa: C901, PLR0912, PLR0915 — sequential orchestration (run1 → save → wipe → run2 → diff)
        """Run pretrain twice via srun and verify the two runs produce bit-identical
        checkpoints — both the per-tensor trace (reference.metadata.info) and the
        non-tensor state (common.pt).

        This is the primary gate for pretrain-nvrx under --deterministic-mode: if
        every kernel honours determinism, two back-to-back runs of identical training
        code on identical hardware must produce identical checkpoint bytes at every
        save iter. Any cross-run diff means residual non-determinism is leaking
        through (custom kernels that bypass torch.use_deterministic_algorithms,
        asynchronous reduction ordering, etc.). The reverse — pass — is a strong
        foundation for the cross-provider check in pretrain-native.

        Each srun reuses the existing in-srun flow (training + distcp_trace +
        metadata/ baseline compare). The in-srun's metadata/-baseline compare may
        soft-fail with exit 3 when the seeded baseline drifts; that's tolerated
        here because the cross-run check is the real gate.

        Layout (run-1 stashed in the sidecar, run-2 left in place):
          ci-runs/<cfg>/checkpoints/iter_NNNNNNN/{reference.metadata.info,common.pt}      (run-2)
          ci-runs/<cfg>/run1-references/iter_NNNNNNN/{reference.metadata.info,common.pt}  (run-1)
        """
        extra_args = getattr(args, "extra_args", None)
        ckpt_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
        phase1_dir = PretrainJob._phase1_ckpt_dir(cfg)
        sidecar_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/run1-references"
        machine = _platform_tag()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = PretrainJob._build_pretrain_srun_cmd(
            cfg, args.nodes, args.gpus_per_node, args.job, args.flavors, extra_args=extra_args, config_profile=args.config,
        )
        stem = _artifact_stem(machine, args.job, args.flavor_tag, cfg)

        # phase1-checkpoints is wiped too so each run's phase 1 starts from random init.
        for label, path in (("checkpoints", ckpt_dir), ("phase-1 checkpoints", phase1_dir), ("run-1 sidecar", sidecar_dir)):
            if path.exists():
                logger.info(f"[run_job] determinism: clearing prior {label} at {path}")
                shutil.rmtree(path)

        def _run(log_file: Path, banner: str) -> None:
            logger.info(f"[run_job] {banner}: {' '.join(shlex.quote(c) for c in cmd)}")
            shell_cmd = f"set -o pipefail; {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {shlex.quote(str(log_file))}"
            rc = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", check=False).returncode  # noqa: S602
            # rc == 3 → in-srun's lenient nvrx metadata-baseline compare flagged a
            # diff vs metadata/. Tolerated; the cross-run compare below gates the job.
            if rc not in (0, 3):
                raise RuntimeError(f"{banner} failed with exit code {rc} — full log: {log_file}")
            if rc == 3:  # noqa: PLR2004
                logger.info(f"[run_job] {banner}: soft-failed vs metadata/ baseline (rc=3); continuing to cross-run compare")

        def _collect_iter_refs(root: Path, label: str) -> list[Path]:
            iter_dirs = sorted(root.glob("iter_*"))
            if not iter_dirs:
                raise RuntimeError(f"determinism: {label} produced no iter_* under {root}")
            return iter_dirs

        # --- Run 1: stash the per-tensor trace and the non-tensor state (common.pt) ---
        _run(RESULTS_DIR / f"{stem}_run1.log", "determinism: run 1/2")
        for iter_dir in _collect_iter_refs(ckpt_dir, "run 1/2"):
            dst_dir = sidecar_dir / iter_dir.name
            dst_dir.mkdir(parents=True)
            for name in ("reference.metadata.info", "common.pt"):
                src = iter_dir / name
                if not src.is_file():
                    raise RuntimeError(f"determinism: run 1/2 produced no {name} at {src}")
                shutil.copy2(src, dst_dir / name)
        logger.info(f"[run_job] determinism: saved run-1 references under {sidecar_dir}")

        # Wipe so run-2 starts from a clean state (otherwise --load picks up run-1's
        # iter_NNNNNNN and shortcuts the training). phase1-checkpoints is wiped too so
        # run-2's phase 1 retrains from random init.
        shutil.rmtree(ckpt_dir)
        if phase1_dir.exists():
            shutil.rmtree(phase1_dir)

        # --- Run 2 ---
        _run(RESULTS_DIR / f"{stem}_run2.log", "determinism: run 2/2")

        # --- Cross-run diff: per-tensor trace (text) + common.pt (binary md5) ---
        failures: list[str] = []
        iter_dirs = _collect_iter_refs(ckpt_dir, "run 2/2")
        for iter_dir in iter_dirs:
            ref1, ref2 = sidecar_dir / iter_dir.name / "reference.metadata.info", iter_dir / "reference.metadata.info"
            if not ref2.is_file():
                raise RuntimeError(f"determinism: missing run-2 reference at {ref2}")
            diff = list(
                difflib.unified_diff(
                    ref1.read_text().splitlines(),
                    ref2.read_text().splitlines(),
                    fromfile=str(ref1),
                    tofile=str(ref2),
                    n=1,
                ),
            )
            if diff:
                logger.info(f"[run_job] determinism FAIL on {iter_dir.name}: reference.metadata.info differs ({len(diff)} diff lines)")
                for line in diff[:30]:
                    logger.info(line.rstrip("\n"))
                if len(diff) > 30:  # noqa: PLR2004
                    logger.info(f"... ({len(diff) - 30} more diff lines suppressed; full files in artifacts)")
                failures.append(f"{iter_dir.name}/reference.metadata.info")

            common2 = iter_dir / "common.pt"
            if not common2.is_file():
                raise RuntimeError(f"determinism: missing run-2 common.pt at {common2}")
            if _md5(sidecar_dir / iter_dir.name / "common.pt") != _md5(common2):
                logger.info(f"[run_job] determinism FAIL on {iter_dir.name}: common.pt md5 differs between the two runs")
                failures.append(f"{iter_dir.name}/common.pt")

        if failures:
            raise RuntimeError(
                f"determinism check FAILED on {len(failures)} artifact(s): {', '.join(failures)}. "
                f"Training is not bit-deterministic across runs even under --deterministic-mode — "
                f"see the diff snippets above and the per-iter artifacts.",
            )
        logger.info(f"[run_job] determinism PASS: {len(iter_dirs)} iter(s) bit-identical (tensors + common.pt) across two nvrx runs")


# Entry point


def _available_profiles() -> list[str]:
    """Config-profile names selectable via --config: every configs/*.yaml except base.yaml
    (the shared base each profile deep-merges over)."""
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml") if p.stem != "base")


def _detect_local_gpu_count() -> int:
    """Return the number of visible GPUs on this host, falling back to 4."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and cuda_visible not in ("NoDevFiles", "none", ""):
        return len(cuda_visible.split(","))
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output([nvidia_smi, "--list-gpus"], text=True, stderr=subprocess.DEVNULL)
            n = sum(1 for ln in out.splitlines() if ln.strip())
            return max(n, 1)
        except (subprocess.CalledProcessError, OSError):
            pass
    return 4


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(shlex.join(sys.argv))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "job",
        choices=["pretrain-native", "pretrain-nvrx", "update_baselines"],
    )
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1 — runs inline; >1 uses SLURM automatically)")
    p.add_argument(
        "--config",
        default="pico",
        choices=_available_profiles(),
        help="Model config profile (default: pico). Selects configs/<profile>.yaml, deep-merged "
        "over configs/base.yaml. pico scales parallelism to the runner's GPUs; nano (8 nodes) and "
        "ultra (16 nodes) pin it.",
    )
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        dest="gpus_per_node",
        help="GPUs per node (default: GPUS_PER_NODE env set at runner setup, else auto-detect)",
    )
    p.add_argument("--slurm", action="store_true", help="Submit via SLURM even for single-node runs")
    p.add_argument(
        "--flavor",
        action="append",
        choices=list(_FLAVORS),
        help="Pretrain run flavor (repeatable; default: perf). Composable, e.g. "
        "--flavor deterministic --flavor valgrind. perf and deterministic are mutually exclusive.",
    )
    p.add_argument("--skip-perf", action="store_true", dest="skip_perf", help="Skip perf parsing; tee full output to console instead")
    p.add_argument(
        "--update-baseline",
        action="store_true",
        dest="update_baseline",
        help="Record current measurements as the new baseline and exit",
    )
    p.add_argument(
        "--filter",
        default=None,
        dest="filter",
        help="update_baselines only: <job>[:<config>[:<flavor>]] to update, e.g. "
        "'pretrain-native:1x4:deterministic'. Auto-detected from CI_JOB_NAME when running in GitLab CI.",
    )
    args, extra_args = p.parse_known_args()
    args.extra_args = extra_args
    _ensure_vcpkg_root()  # before any `import megatron.native` triggers the CMake build
    # Honor the runner's configured scheduler mode (SLURM_PARTITION set => SLURM),
    # so the CI YAML needn't pass --slurm in JOB_ARGS.
    if _runner_uses_slurm():
        args.slurm = True
    if args.gpus_per_node is None:
        # Prefer the count configured at runner setup (GPUS_PER_NODE), so the CI
        # YAML stays cluster-agnostic; fall back to probing the local host.
        args.gpus_per_node = _runner_gpus_per_node() or _detect_local_gpu_count()

    # Flavors apply only to pretrain jobs; other jobs keep an empty tag (e.g.
    # ckptbench baselines stay {machine}_ckptbench_{cfg}).
    args.flavors = _resolve_flavors(args.flavor, args.nodes) if _is_pretrain_job(args.job) else []
    args.flavor_tag = "-".join(f.name for f in args.flavors)

    # build_env (e.g. valgrind's PHOENIX_BUILD_FOR_VALGRIND) is read when
    # `import megatron.native` triggers the CMake build, so export it before any
    # such import; subprocess/srun children inherit it.
    build_env = Recipe(opts=[], env={})
    _apply_flavors(build_env, args.flavors)
    os.environ.update(build_env.build_env)

    if args.job == "update_baselines":
        BaselineUpdater.run_update_baselines(args)
        return

    # Fail fast if a prior job left a server bound to the port (refuse to start, don't kill it).
    # Only on compute nodes — the slurm launcher runs on the head node, where no server runs.
    if in_srun() or not args.slurm:
        CheckpointServer.require_no_checkpoint_server()
    PretrainJob.run_pretrain(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _PRETRAIN_ENTRY_CMD:
        # Subprocess entry for the srun/torchrun paths: drop the sentinel and the phase that
        # follows it, then run pretrain in-process. argv[0] is cosmetic; Megatron parses argv[1:].
        ci_phase = int(sys.argv[2])
        sys.argv = ["pretrain_hybrid.py", *sys.argv[3:]]
        PretrainJob._pretrain_main(time.time(), ci_phase=ci_phase)
    else:
        main()
