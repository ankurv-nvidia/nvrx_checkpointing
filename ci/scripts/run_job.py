#!/usr/bin/env python3
"""CI job runner for megatron/native benchmarks and e2e tests.

Jobs:
  pretrain-native  native provider; deterministic adds the cross-provider md5 compare.
  pretrain-nvrx    nvrx provider; deterministic runs twice (cross-run determinism gate),
                   perf is a fast-path smoke.

Flavors (--flavor, repeatable, default perf): perf (flash + fusion, perf-gated),
deterministic (reproducible kernels + md5 checks, excludes perf), valgrind (memcheck the
server, single-node).

SLURM vs inline is inferred from SLURM_ACCOUNT being set; --slurm forces SLURM for local runs.
Per-cluster srun/image-build templates live in templates/ (selected by CLUSTER); env knobs:
PLATFORM (tag + baseline namespace), CLUSTER, GPUS_PER_NODE, SLURM_ACCOUNT, CI_DIR.

Exit codes: 0 = pass (or --skip-perf), 3 = perf regression (soft), 1 = job/setup failure.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import http.client
import json
import logging
import lzma
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
    from collections.abc import Iterator

import wheel_install  # companion script in ci/scripts
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR / "../..").resolve()
BASELINES_DIR = (SCRIPT_DIR / "../baselines").resolve()
CONFIGS_DIR = (SCRIPT_DIR / "../configs").resolve()
logger = logging.getLogger("native_ci")

# The single state registry checked into metadata/: the fixed input that the
# ckptbench benchmark (ckptbench.py) and the ckpttest round-trip test
# (CMakeLists.txt) replay. Every other config's registry.bin is a runtime-only
# artifact and is not mirrored into the baseline tree.
_CKPTBENCH_REGISTRY_CFG = "pico_1x2"
_CKPTBENCH_REGISTRY_ITER = "0000250"
CKPTBENCH_REGISTRY = REPO_ROOT / "metadata" / f"{_CKPTBENCH_REGISTRY_CFG}_checkpoints_iter_{_CKPTBENCH_REGISTRY_ITER}" / "registry.bin"

# Sentinel first arg that re-enters this script as the in-process training entry, so the
# launched and in-process paths share one code path.
_PRETRAIN_ENTRY_CMD = "__pretrain_entry__"

ENROOT_SCRIPTS_DIR = (REPO_ROOT / "scripts").resolve()
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.04-py3")


def sys_path_insert(path: Path) -> None:
    """Insert a path into sys.path if not already present, using the path's as_posix() form."""
    p = path.as_posix()
    if p not in sys.path:
        sys.path.insert(0, p)
    pythonpath = os.environ.get("PYTHONPATH", "").split(":")
    if p not in pythonpath:
        os.environ["PYTHONPATH"] = ":".join([p, *pythonpath])


def wait_for_file(path: Path, timeout_seconds: int, *, poll_s: int = 10, log_every_s: int = 30) -> None:
    """Poll until the file exists (or any match, when the name is a glob), bounded by a timeout.
    Logs at most every log_every_s so a long many-rank barrier wait doesn't flood the job log."""
    has_wildcard = any(c in path.name for c in "*?[")
    start_time = time.time()
    last_log = 0.0
    while True:
        if has_wildcard:
            if any(path.parent.glob(path.name)):
                return
        elif path.is_file():
            return
        now = time.time()
        if now - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {path} after {timeout_seconds} seconds")
        if now - last_log >= log_every_s:
            prefix = "[in-srun] " if "SLURM_PROCID" in os.environ else ""
            logger.info(f"{prefix}Waiting for {path} to be available...")
            last_log = now
        time.sleep(poll_s)


def _prepare_sh_hash() -> str:
    """Short content hash that namespaces the built image, so a build-recipe change forces a rebuild."""
    return hashlib.sha256((ENROOT_SCRIPTS_DIR / "prepare.sh").read_bytes()).hexdigest()[:10]


def _default_enroot_image() -> str:
    return f"{Path.home()}/data/native.{_prepare_sh_hash()}.sqsh"


def image_path() -> Path:
    """Path to the container image, shared by the build step and the workload so both agree."""
    return Path(os.environ.get("ENROOT_IMAGE", _default_enroot_image()))


def megatron_root() -> Path:
    """Path to the megatron"""
    out = subprocess.check_output(["bash", (ENROOT_SCRIPTS_DIR / "megatron_root.sh").as_posix()], text=True)  # noqa: S607
    return Path(out.strip()).resolve()


def _triton_cache_dir() -> Path:
    """Node-local kernel cache. A shared-filesystem cache races across nodes (a kernel written
    on one node isn't yet visible on another), so each node gets its own under local storage."""
    env = os.environ.get("TRITON_CACHE_DIR")
    d = Path(env) if env else Path("/tmp/triton-cache")  # noqa: S108 — node-local cache, isolation is the point
    d.mkdir(parents=True, exist_ok=True)
    return d


def platform_tag() -> str:
    """Hardware platform identifier used to namespace baseline files, so one GPU platform
    shares baselines across clusters."""
    tag = os.environ.get("PLATFORM", "").strip()
    return tag or socket.gethostname().split(".")[0]


def cluster_tag() -> str:
    """Runner cluster identifier used only for cluster-specific perf centers."""
    return os.environ.get("CLUSTER", "").strip()


RESULTS_DIR = REPO_ROOT / "ci" / "ci-results"
PERF_TOLERANCE = 0.05  # 5% tolerance — both regressions and improvements beyond this fail
# Max tolerated bandwidth ratio between the two socket halves; a breach means a
# per-socket imbalance has returned.
MAX_BW_SOCKET_IMBALANCE = 1.30

_BASELINE_CHANGE_THRESHOLD = 1e-6  # minimum relative change to count as a meaningful baseline update
SOFT_FAIL_RC = 3  # exit code for a tolerated soft fail (perf regression / nvrx baseline drift)


def ensure_vcpkg_root() -> None:
    """Pin the build-dependency root beside the checkout so the in-container build finds it and
    concurrent builds on one runner don't contend on a shared lock."""
    os.environ.setdefault("VCPKG_ROOT", (REPO_ROOT.parent / "vcpkg").as_posix())


def _is_pretrain_job(job: str) -> bool:
    return job.startswith("pretrain-")


def _ckpt_provider(job: str) -> str:
    """The checkpoint provider a pretrain job exercises; also selects strict vs lenient
    metadata-compare semantics."""
    return job.split("-", 1)[1]


# Flavors compose over the core recipe: perf leaves it as-is, deterministic swaps in
# reproducible kernels, valgrind memchecks the server.


@dataclass
class Recipe:
    """CLI opts, process env, and SLURM knobs that a flavor transforms in place."""

    opts: list[str]
    env: dict[str, str]
    build_env: dict[str, str] = field(default_factory=dict)
    srun_time_min: int = 30  # SLURM --time in minutes; 1h default, raised per flavor (e.g. valgrind)
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
        return artifact_stem(self.machine, self.job, self.flavor_tag, self.cfg)

    @property
    def is_worker(self) -> bool:
        """In the process that ran training (can read checkpoints, run container tools)."""
        return in_srun() or not self.args.slurm

    @property
    def is_rank0(self) -> bool:
        return os.environ.get("SLURM_PROCID", "0") == "0"

    @property
    def is_local_rank0(self) -> bool:
        return os.environ.get("SLURM_LOCALID", "0") == "0"

    def has(self, name: str) -> bool:
        return any(f.name == name for f in self.flavors)


class Flavor(ABC):
    """A composable transform over the core pretrain recipe, with optional run hooks.
    The base class is the perf fast path: identity transform and no-op hooks."""

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
            "TRITON_CACHE_PATH": str(triton_dir),
            # Route the server log into the uploaded artifact dir (not node-local
            # storage). %h keeps multinode runs from colliding; %p keeps daemon
            # candidates apart — every rank's launcher spawns a candidate whose
            # startup opens the log and whose losing exit removes it, so a shared
            # per-node name would let a loser unlink the winning daemon's live log.
            "NATIVE_CKPT_SERVER_LOG": str(RESULTS_DIR / "checkpoint_server.%h.%p.log"),
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

    @staticmethod
    def _drop_opt_multi(opts: list[str], flag: str) -> None:
        """Remove flag and all of its value tokens (up to the next --flag or end). Handles both
        single-value (--recompute-modules moe) and multi-value (--cuda-graph-modules a b c) options."""
        if flag not in opts:
            return
        i = opts.index(flag)
        j = i + 1
        while j < len(opts) and not opts[j].startswith("--"):
            j += 1
        del opts[i:j]

    # Acceleration flags some profiles pin for the perf flavor (CUDA graphs, selective
    # activation recompute). Names match configs/*.yaml. The deterministic flavor keeps CUDA
    # graphs (they cut the run time and the cross-run check validates they stay reproducible)
    # but drops recompute; valgrind drops both (graph capture is infeasible under memcheck).
    _CUDA_GRAPH_FLAGS: ClassVar[tuple[str, ...]] = ("--cuda-graph-impl", "--cuda-graph-modules")
    _RECOMPUTE_FLAGS: ClassVar[tuple[str, ...]] = ("--recompute-granularity", "--recompute-modules")

    @classmethod
    def _drop_opts(cls, opts: list[str], flags: tuple[str, ...]) -> None:
        # A bare string would iterate per character and silently drop nothing.
        assert not isinstance(flags, str), "pass a tuple of flags, not a bare string"
        for flag in flags:
            cls._drop_opt_multi(opts, flag)


class PerfFlavor(Flavor):
    """Fast path (flash attention + cross-entropy fusion, already in the core recipe).

    Gated on the checkpoint-write perf baseline; runs no md5 compare.
    """

    name = "perf"

    def transform(self, recipe: Recipe) -> None:
        pass

    def epilog(self, ctx: RunCtx) -> None:
        self.run_perf_epilog(ctx)

    @staticmethod
    def run_perf_epilog(ctx: RunCtx) -> None:
        """Parse perf metrics from the run log, compare to the baseline, and soft-fail on
        out-of-tolerance drift (a non-blocking warning)."""
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
    """Reproducible numerics for the metadata checks (deterministic kernels), plus a non-blocking
    perf-drift warning."""

    name = "deterministic"

    # Must be set before the training process starts: some are consumed at import time,
    # before argument parsing runs.
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
        self._drop_opts(recipe.opts, self._RECOMPUTE_FLAGS)
        self._add_flag(recipe.opts, "--deterministic-mode")
        self._drop_opt(recipe.opts, "--async-ckpt-use-cpu-shm")
        recipe.env.update(self.ENV)

    def epilog(self, ctx: RunCtx) -> None:
        # Skipped under valgrind: that flavor owns the server and tears it down before the
        # epilog, leaving the md5 trace with no server (md5 is covered by the plain run).
        # Skipped past two nodes: training there is bit-reproducible only within one
        # allocation, so no committed baseline can match — the in-allocation reference
        # check (run_pretrain) owns that compare.
        if ctx.is_worker and not ctx.has("valgrind") and int(os.environ.get("SLURM_NNODES", "1")) <= 2:  # noqa: PLR2004
            PerfReport.compare_iters_to_baseline(ctx.cfg, ctx.provider)
        # Warn (non-blocking exit 3) on perf drift against this flavor's own slower baseline;
        # the md5 mismatch above still hard-fails.
        PerfFlavor.run_perf_epilog(ctx)


class ValgrindFlavor(Flavor):
    """Memcheck the server under valgrind on rank 0 and fail on reported leaks/errors.
    Single-node only."""

    name = "valgrind"

    _READY_MARKER = "Checkpoint server started"
    # Logged when the server begins orderly shutdown; once seen we stop tee'ing to the
    # console (the file keeps the rest) to drop the noisy shutdown diagnostics.
    _STOP_MARKER = "Shutting down server thread"

    # Stall guard: a stuck server keeps logging but stops writing checkpoint files, so
    # checkpoint-file activity (not log activity) is the progress signal.
    _STALL_SILENCE_S = 300  # 5 min of no checkpoint-file activity = stalled
    _STALL_POLL_S = 10  # how often to poll for progress / exit
    _STALL_STOP_GRACE_S = 120  # grace for a stalled server to honor a re-issued stop
    _STALL_LOG_EVERY_S = 60  # heartbeat cadence while waiting, so the wait is visible
    _MAX_FILES_SHOWN = 6  # max changed files to list in the heartbeat log line

    # Memcheck error headers (not leak kinds). Each is printed as it happens, so they
    # survive in the log even if the process is later killed before the summary.
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
    _MAX_ERRORS_SUMMARY = 5

    @classmethod
    def _scan_valgrind_errors(cls, server_log: Path) -> list[str]:
        """Distinct memcheck error headers in the server log — a backstop summary, not the
        primary gate (a real error usually trips a crash or stall guard first)."""
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
        """Modification times of every checkpoint file — the freshest one is the stall
        guard's progress signal, the per-file map reports what changed."""
        out: dict[str, float] = {}
        try:
            for p in ckpt_root.rglob("*"):
                try:
                    if p.is_file():
                        out[str(p.relative_to(ckpt_root))] = p.stat().st_mtime
                except OSError:  # noqa: PERF203
                    continue
        except OSError:
            pass
        return out

    def _await_server_exit(self, proc: subprocess.Popen, ckpt_root: Path) -> tuple[int | None, bool]:
        """Wait for the server to exit, bounding the wait by checkpoint-file activity: progress
        means keep waiting, prolonged silence means stalled (re-request stop, then kill)."""
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
            # Periodic heartbeat so a slow-but-progressing run looks alive, not hung:
            # report recently-changed files, or how long it's been silent.
            if now - last_log >= self._STALL_LOG_EVERY_S:
                changed = sorted(p for p, m in mtimes.items() if logged_mtimes.get(p) != m)
                logged_mtimes = dict(mtimes)
                last_log = now
                waited = round(now - start)
                if changed:
                    n = self._MAX_FILES_SHOWN
                    shown = ", ".join(changed[:n]) + (f" (+{len(changed) - n} more)" if len(changed) > n else "")
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
                CheckpointServer.dump_server_threads()
                CheckpointServer.request_server_stop()
                try:
                    rc = proc.wait(timeout=self._STALL_STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.exception("[run_job] stalled checkpoint_server ignored stop — SIGKILL")
                    CheckpointServer.gcore_and_kill_server()
                    return None, True
                else:
                    logger.error(f"[run_job] stalled checkpoint_server exited after re-issued stop (rc={rc})")
                    return rc, True

    def transform(self, recipe: Recipe) -> None:
        # Build flag that shrinks the staging buffers so valgrind's shadow memory stays manageable.
        recipe.build_env["PHOENIX_BUILD_FOR_VALGRIND"] = "1"
        # Strip CUDA graphs AND recompute: graph capture is infeasible under memcheck. valgrind
        # composes with deterministic (which keeps CUDA graphs), so drop them explicitly here.
        self._drop_opts(recipe.opts, self._CUDA_GRAPH_FLAGS + self._RECOMPUTE_FLAGS)

    @contextlib.contextmanager
    def prolog(self, ctx: RunCtx) -> Iterator[None]:
        # Each node's local rank 0 hosts that node's memchecked (valgrind) server; the node's
        # other ranks connect to it (suppress their auto-fork). So every node's server runs under
        # memcheck, not just the coordinator's.
        if ctx.is_local_rank0:
            with self._server(ctx):
                yield
            return
        os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
        yield

    def epilog(self, ctx: RunCtx) -> None:
        if ctx.is_worker and ctx.valgrind_rc not in (None, 0):
            raise RuntimeError(f"valgrind reported errors in checkpoint_server (exit {ctx.valgrind_rc}); see the server log")

    @staticmethod
    def _binary() -> Path:
        import nvrx_checkpointing  # noqa: PLC0415 — triggers CMake build

        binary = nvrx_checkpointing.SERVER_PATH
        if not binary.is_file():
            raise FileNotFoundError(f"checkpoint_server not found at {binary}")
        return binary

    @contextlib.contextmanager
    def _server(self, ctx: RunCtx) -> Iterator[None]:
        """Pre-launch the server under valgrind for the block: start it, route the client to it,
        wait for readiness; on exit wait for its idle auto-shutdown so the leak report flushes."""
        valgrind = shutil.which("valgrind")
        if valgrind is None:
            raise RuntimeError("valgrind not found in PATH — required for the valgrind flavor")
        server = self._binary()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Per-node filename: on multinode each node's local rank 0 runs its own valgrind server,
        # and RESULTS_DIR is shared, so a single name would clobber across nodes.
        server_log = RESULTS_DIR / f"{ctx.stem}_server.{socket.gethostname().split('.')[0]}.log"
        server_log.write_bytes(b"")  # truncate any prior run's log

        cmd = [valgrind, "--tool=memcheck", "--leak-check=full", "--show-leak-kinds=all", "--error-exitcode=42", "--log-fd=2", str(server)]
        logger.info(f"[run_job] Launching checkpoint_server under valgrind; log: {server_log}")
        CheckpointServer.require_no_checkpoint_server()  # a leftover here means a prior run didn't spin down

        os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
        # The server appends its stderr to this same file; append-mode + line-flushed
        # writes keep entries from interleaving mid-line.
        os.environ["NATIVE_CKPT_SERVER_LOG"] = str(server_log)
        log_fp = server_log.open("a")
        proc, ready, watcher, pump = self._spawn(cmd, server_log, log_fp)
        try:
            if not ready.wait(timeout=600):
                raise RuntimeError(f"checkpoint_server did not print '{self._READY_MARKER}' within 600s under valgrind; see {server_log}")
            if proc.poll() is not None:
                raise RuntimeError(f"checkpoint_server exited before becoming ready (rc={proc.returncode}); see {server_log}")
            logger.info("[run_job] checkpoint_server is ready — proceeding with pretrain")
            yield
        finally:
            # Ask the server to stop now rather than waiting out its idle auto-shutdown — under
            # memcheck that wait plus the slow teardown can blow the time budget.
            logger.info("[run_job] Pretrain finished — stopping checkpoint_server so valgrind can flush its report")
            CheckpointServer.request_server_stop()
            # Primary gate: wait for exit, bounded by checkpoint-file activity. A stall re-stops
            # then kills the server (hard failure); a non-zero clean exit is failed by the check below.
            ckpt_root = REPO_ROOT / "ci/ci-runs" / ctx.cfg
            rc, stalled = self._await_server_exit(proc, ckpt_root)
            ctx.valgrind_rc = rc  # record before cleanup so epilog sees it even if cleanup raises
            watcher.join(timeout=5)
            pump.join(timeout=60)  # let the pump drain valgrind's full exit report before the file is closed
            log_fp.close()
            if rc is not None:
                logger.info(f"[run_job] valgrind exited with code {rc}; server log: {server_log}")

            # ROBUST CROSS-CHECK (kept; a summary + backstop, NOT the primary gate):
            # list the memcheck errors valgrind flagged, however it exited.
            errors = self._scan_valgrind_errors(server_log)
            summary = "; ".join(errors[: self._MAX_ERRORS_SUMMARY]) + (" ..." if len(errors) > self._MAX_ERRORS_SUMMARY else "")
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
            # Backstop: a clean exit that nonetheless logged memcheck errors means one
            # slipped past the guards — fail loudly rather than lean on this scan.
            if rc == 0 and errors:
                raise RuntimeError(
                    "valgrind reported memcheck errors that the exit-code/stall guards did NOT catch — "
                    f"the regular guards need strengthening: {summary}; see {server_log}",
                ) from None
            # Fail here (not just in the global-rank-0 epilog) so every node's local rank 0 reports
            # its own server's valgrind errors — required for per-node multinode memcheck.
            if rc not in (None, 0):
                raise RuntimeError(f"valgrind reported errors in checkpoint_server (exit {rc}); see {server_log}") from None

    def _spawn(
        self,
        cmd: list[str],
        server_log: Path,
        log_fp: IO[str],
    ) -> tuple[subprocess.Popen, threading.Event, threading.Thread, threading.Thread]:
        """Spawn the server with threads that capture its output and tee the log to the console.
        Returns the watcher and the pump so the caller can join them after the process exits."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=1, text=True)
        ready = threading.Event()

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fp.write(line)
                log_fp.flush()
            ready.set()  # pipe closed (process exited): unblock a stuck waiter

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        watcher = threading.Thread(target=self._watch_log, args=(server_log, proc, ready), daemon=True)
        watcher.start()
        return proc, ready, watcher, pump

    def _watch_log(self, server_log: Path, proc: subprocess.Popen, ready: threading.Event) -> None:
        """Tail the server log to the console, signalling readiness at its marker and muting at
        the shutdown marker; always signal on exit so a waiter can't deadlock."""
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


def _resolve_flavors(requested: list[str] | None, _nodes: int) -> list[Flavor]:
    """Validate the requested flavors and instantiate them in canonical order (default perf).
    perf and deterministic are mutually exclusive."""
    names = list(dict.fromkeys(requested or ["perf"]))
    unknown = [n for n in names if n not in _FLAVORS]
    if unknown:
        raise SystemExit(f"--flavor: unknown flavor(s) {unknown}; choose from {sorted(_FLAVORS)}")
    if "perf" in names and "deterministic" in names:
        raise SystemExit("--flavor: 'perf' and 'deterministic' are mutually exclusive")
    # valgrind runs at any node count: each node's local rank 0 memchecks its own server.
    return [_FLAVORS[n]() for n in _FLAVOR_ORDER if n in names]


def _apply_flavors(recipe: Recipe, flavors: list[Flavor]) -> None:
    for flavor in flavors:
        flavor.transform(recipe)


@contextlib.contextmanager
def _flavored_run(ctx: RunCtx) -> Iterator[None]:
    """Bracket a pretrain run: set up each flavor, run training, then spin the per-node checkpoint
    daemon down on every rank (on success and failure) and run the rank-0 checks."""
    stack = contextlib.ExitStack()
    if ctx.is_worker:
        for flavor in ctx.flavors:
            stack.enter_context(flavor.prolog(ctx))
    try:
        try:
            yield
        except BaseException:
            stack.close()  # prologs exit (valgrind waits for its server, records the rc)
            # On a failure the daemon may have crashed; echo its log (one rank per node) so its
            # [CRASH]/SIGABRT backtrace reaches the console, not just the uploaded artifact.
            if ctx.is_worker and ctx.provider == "native" and not ctx.has("valgrind") and os.environ.get("SLURM_LOCALID", "0") == "0":
                CheckpointServer.dump_server_log()
            CheckpointServer.ensure_server_down(ctx, graceful=False, verify_files=False, fail_on_timeout=False)
            raise
        stack.close()
        CheckpointServer.ensure_server_down(ctx)
        if not ctx.is_rank0:
            return
        for flavor in ctx.flavors:
            flavor.epilog(ctx)
        # Args baseline gate (launcher owns the tee'd log): warn (non-blocking) if the resolved
        # arguments drifted from the committed baseline. update_baselines accepts the new block.
        if ctx.log_file is not None and not ctx.args.update_baseline and not _arguments_match_baseline(ctx.stem):
            ctx.soft_fail = True
        if ctx.soft_fail:
            sys.exit(3)
    finally:
        # Capture the resolved arguments block from the run log (artifact + baseline), on success
        # and failure alike — args are printed at model init, before a later crash. Only the
        # launcher/torchrun/inline path owns ctx.log_file; in-srun tasks (log_file=None) skip it,
        # so the capture runs once per job from the process that holds the tee'd log.
        if ctx.log_file is not None:
            capture_arguments(ctx.stem)


def artifact_stem(machine: str, job: str, flavor_tag: str, cfg: str) -> str:
    """`{machine}_{job}_{flavor_tag}_{cfg}`, dropping the tag when empty (e.g. ckptbench)."""
    parts = [machine, job, flavor_tag, cfg] if flavor_tag else [machine, job, cfg]
    return "_".join(parts)


# Markers Megatron prints around the fully-resolved argument dump at model init.
_ARGS_BLOCK_START = "------------------------ arguments ------------------------"
_ARGS_BLOCK_END = "-------------------- end of arguments ---------------------"
# Strip a leading per-task/per-rank log prefix (srun --label "N: ", torchrun "[rankN]: ") so the
# captured block matches Megatron's raw output; the value lines themselves start with spaces. srun
# right-justifies the task number to the max-rank width (" 0: " at >= 10 tasks), hence the \s*.
_ARGS_LINE_PREFIX_RE = re.compile(r"^(?:\[rank\d+\]:\s?|\s*\d+:\s)")
# Arguments whose value is a per-runner CI output path (save/load/data_cache_path/
# tensorboard_dir — all under ci-runs/) are pure build-location noise that varies by
# runner. These lines are pruned from the captured/committed arguments block and ignored
# in the compare, so a baseline never carries a checkout-specific path.
_ARGS_VOLATILE_PATH_RE = re.compile(r"/\S*/ci-runs/")
# Megatron prints one "  name ....... value" line per argument. A clean line starts with two
# spaces, the name, then the dotted leader (or padding spaces for a long name).
_ARGS_LINE_RE = re.compile(r"^  [A-Za-z_]\w* [.\s]")
# The same shape with a dotted leader, embedded mid-line: a stderr log line flushed without a
# trailing newline prepends junk to the next argument line, which we recover from here.
_ARGS_EMBEDDED_RE = re.compile(r"  [A-Za-z_]\w* \.{3,} ")
# A stderr log line can also merge onto the END of an argument's value (no separator), e.g.
# "logging_level ... 20WARNING: TensorBoard ..." or "bf16 ... True> setting tensorboard ...";
# drop the appended message from its log marker on.
_ARGS_APPENDED_RE = re.compile(r"(?:(?:WARNING|INFO|ERROR|DEBUG|CRITICAL): |>\s*setting tensorboard\b).*$")


def _clean_arguments_line(line: str) -> str | None:
    """The argument portion of ``line`` — kept when clean, recovered when a stderr log line without a
    trailing newline merged onto the front of (or landed inside) it, with any log message merged onto
    the end stripped — or None if the line carries no argument (blank line, standalone log message,
    progress output that interleaved into the dump)."""
    if _ARGS_LINE_RE.match(line):
        arg = line
    else:
        m = _ARGS_EMBEDDED_RE.search(line)
        if not m:
            return None
        arg = line[m.start() :]
    arg = _ARGS_APPENDED_RE.sub("", arg)
    # Prune volatile per-runner CI output paths (save/load/data_cache_path/tensorboard_dir)
    # so they never reach the captured/committed arguments.txt.
    if _ARGS_VOLATILE_PATH_RE.search(arg):
        return None
    return arg


def _extract_arguments_block(log_text: str) -> str | None:
    """Return the last complete Megatron arguments block (markers inclusive) in ``log_text``, per-line
    log prefixes stripped and non-argument lines that interleaved into the dump dropped, or None if no
    complete block is present. The last block is the measured (phase-2 / run-2) one; earlier blocks are
    the warm-up phase-1 / run-1 runs."""
    end = log_text.rfind(_ARGS_BLOCK_END)
    if end == -1:
        return None
    start = log_text.rfind(_ARGS_BLOCK_START, 0, end)
    if start == -1:
        return None
    block = log_text[start : end + len(_ARGS_BLOCK_END)]
    out = [_ARGS_BLOCK_START]
    for raw in block.splitlines():
        ln = _ARGS_LINE_PREFIX_RE.sub("", raw)
        if ln in (_ARGS_BLOCK_START, _ARGS_BLOCK_END):
            continue
        cleaned = _clean_arguments_line(ln)
        if cleaned is not None:
            out.append(cleaned)
    out.append(_ARGS_BLOCK_END)
    return "\n".join(out) + "\n"


def _read_arguments_block(stem: str) -> str | None:
    """The resolved arguments block from this run's log(s), or None if none was printed. For the
    two-run nvrx determinism check the run-2 log wins (its block is the last one read)."""
    candidates = [RESULTS_DIR / f"{stem}.log", *sorted(RESULTS_DIR.glob(f"{stem}_run*.log"))]
    block: str | None = None
    for log in candidates:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        found = _extract_arguments_block(text)
        if found is not None:
            block = found
    return block


def _normalize_arguments(block: str) -> str:
    """Canonicalize an arguments block for comparison: drop volatile ci-runs/ path arguments, so
    a baseline captured before pruning still matches a freshly-captured block that omits them."""
    return "\n".join(ln for ln in block.splitlines() if not _ARGS_VOLATILE_PATH_RE.search(ln))


def capture_arguments(stem: str) -> None:
    """Write this run's resolved arguments block to ``RESULTS_DIR/{stem}_arguments.txt`` — uploaded
    as an artifact and committed (plaintext) to the baselines by update_baselines, one per
    config+flavor. Best-effort: a job that prints no block (e.g. ckptbench) writes nothing. Never
    raises — capture must not turn a passing run red."""
    block = _read_arguments_block(stem)
    out = RESULTS_DIR / f"{stem}_arguments.txt"
    if block is None:
        logger.info(f"[run_job] no arguments block in {stem} log(s); not writing {out.name}")
        return
    try:
        out.write_text(block)
        logger.info(f"[run_job] captured resolved arguments -> {out}")
    except OSError as ex:
        logger.info(f"[run_job] could not write {out}: {ex}")


def _arguments_match_baseline(stem: str) -> bool:
    """Compare this run's resolved arguments against the committed baseline (volatile ci-runs/ paths
    ignored). True on match, when no baseline is committed yet (bootstrap), or when the run printed
    no block; False (and logs a unified diff) on drift. update_baselines refreshes the baseline."""
    current = _read_arguments_block(stem)
    if current is None:
        return True
    baseline_file = BASELINES_DIR / f"{stem}_arguments.txt"
    if not baseline_file.exists():
        logger.info(f"[run_job] no arguments baseline {baseline_file.name}; run update_baselines to seed it")
        return True
    base_lines = _normalize_arguments(baseline_file.read_text()).splitlines()
    cur_lines = _normalize_arguments(current).splitlines()
    if base_lines == cur_lines:
        logger.info(f"[run_job] resolved arguments match baseline {baseline_file.name}")
        return True
    diff = difflib.unified_diff(
        base_lines,
        cur_lines,
        fromfile=f"{baseline_file.name} (baseline)",
        tofile=f"{stem} (this run)",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    logger.warning(f"[run_job] resolved arguments differ from baseline:\n{diff_text}")
    return False


# SLURM context detection


def runner_uses_slurm() -> bool:
    """True when this runner submits via SLURM, inferred from a configured account; clusters
    without one run inline."""
    return bool(os.environ.get("SLURM_ACCOUNT", "").strip())


def in_srun() -> bool:
    return "SLURM_PROCID" in os.environ


# Per-cluster SLURM/image-build templates: per-run values fill ${...} placeholders, cluster
# constants live in the file. Repo ships defaults; a runner can override per cluster.

TEMPLATES_DIR = (SCRIPT_DIR / "../templates").resolve()


def _ci_dir() -> Path:
    """Runner state dir holding the config and any override templates; falls back to a default."""
    raw = os.environ.get("CI_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / "data" / "ci"


def _load_template(name: str) -> str:
    """Return the named template, preferring a per-runner override, then a cluster-specific
    default, then the generic default."""
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
    """Fill the template's lowercase placeholders, leaving shell-style references untouched."""
    return string.Template(_load_template(name)).safe_substitute(values)


def build_cmd_from_template(name: str, values: dict[str, object], payload: list[str]) -> list[str]:
    """Render the named template into a command list (stripping comments and blank lines,
    tokenizing the rest) and append the payload."""
    body = "\n".join(ln for ln in _load_template(name).splitlines() if not ln.lstrip().startswith("#"))
    return [*shlex.split(string.Template(body).safe_substitute(values)), *payload]


def build_srun_cmd(  # noqa: PLR0913
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
    strip_container: bool = False,
    overlap: bool = False,
    label: bool = True,
) -> list[str]:
    """Render the scheduler template into the launch-command prefix (callers append the payload),
    exempting every job from the idle-GPU reaper for its full time budget.

    ``strip_container`` drops the template's --container-* args, yielding the
    host-level allocation srun for the prep/run dispatch; ``overlap`` marks an
    inner job step that shares that allocation's resources.
    """
    account = os.environ.get("SLURM_ACCOUNT", "").strip()
    if not account:
        raise RuntimeError("SLURM_ACCOUNT is not set — required for SLURM job submission")
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
        "root": Path(image).parent.as_posix(),
        "account": account,
    }
    body = "\n".join(ln for ln in _load_template("srun.tmpl").splitlines() if not ln.lstrip().startswith("#"))
    # Exempt the job from the idle-GPU reaper for its full time budget, so transient idle
    # (build, checkpoint flush, multinode coordination) doesn't get it killed early.
    reaper = "--comment=" + json.dumps(
        {
            "OccupiedIdleGPUsJobReaper": {
                "exemptIdleTimeMins": str(time_minutes),
                "reason": "other",
                "description": "native checkpoint CI job",
            },
        },
        separators=(",", ":"),
    )
    # --label makes srun buffer each task's stdout on line boundaries before merging, so one
    # rank's partial write can't splice mid-line into another's (which clobbers rank 0's argument
    # dump). Appended here, not in the templates, so every cluster's render carries it. The
    # allocation-level dispatch srun passes label=False: only its task 0 emits output, and a
    # second prefix on the inner run srun's already-labeled lines would break the arguments
    # capture (whose regexes strip exactly one prefix).
    tokens = shlex.split(string.Template(body).safe_substitute(values))
    if strip_container:
        tokens = [tok for tok in tokens if not tok.startswith("--container-")]
    if overlap:
        tokens.insert(1, "--overlap")
    return [*tokens, reaper, *(["--label"] if label else []), *(extra_srun_args or [])]


def runner_gpus_per_node() -> int | None:
    """GPUs-per-node configured at runner setup, or None — the default when not passed on the
    command line, so the count lives with the runner."""
    raw = os.environ.get("GPUS_PER_NODE", "").strip()
    return int(raw) if raw else None


# Job: pretrain


_CHECKPOINT_SERVER_PORT = 41000


# Phase 1 trains from scratch to this iteration and saves one checkpoint; phase 2
# resumes from it and runs as a standalone training would.
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
    config_profile: str = "pico"  # model profile (pico/nano3/nano4/ultra3bf16/ultra3fp8): base.yaml + <profile>.yaml

    @property
    def scale(self) -> str:
        return f"{self.nodes}x{self.gpus_per_node}"


# Inline pretrain (in-process, no torchrun subprocess)


# Launch + perf post-processing


def tee_run(cmd: list[str], log_file: Path, env: dict | None = None, *, append: bool = False) -> int:
    """Run the command, tee combined output to the log, and return the exit code; append keeps a
    prior phase's output. Raises on anything but success or the tolerated soft-fail code."""
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
    def ensure_server_down(ctx: RunCtx, *, graceful: bool = True, verify_files: bool = True, fail_on_timeout: bool = True) -> None:
        """Bring the checkpoint server down. The graceful path blocks until it has flushed its
        writes and exited, then confirms the checkpoint files landed. The crash path stops it
        immediately (waiting for idle would stall, since the work never completes), best-effort."""
        if not ctx.is_worker or ctx.provider != "native" or ctx.has("valgrind"):
            return
        if not CheckpointServer._stop_checkpoint_server(graceful=graceful) and fail_on_timeout:
            CheckpointServer.dump_server_log()
            CheckpointServer.dump_server_threads()
            CheckpointServer.gcore_and_kill_server()
            raise RuntimeError(
                "checkpoint_server did not spin down after a clean-stop request — "
                "see the checkpoint_server log dumped above (and the core file in ci-results)",
            )
        if verify_files and ctx.is_rank0:
            CheckpointServer._report_native_checkpoint_files(ctx.cfg)

    @staticmethod
    def _report_native_checkpoint_files(cfg: str) -> None:
        """Confirm each saved checkpoint is non-empty and log every file's size. The graceful
        spin-down returns only once writes are flushed, so the files are final by the time we look."""
        ckpt_root = REPO_ROOT / f"ci/ci-runs/{cfg}/checkpoints"
        iter_dirs = sorted(d for d in ckpt_root.glob("iter_*") if d.is_dir()) if ckpt_root.is_dir() else []
        if not iter_dirs:
            raise FileNotFoundError(f"no checkpoints found under {ckpt_root}")
        for iter_dir in iter_dirs:
            files = sorted(p for p in iter_dir.rglob("*") if p.is_file())
            if not files:
                raise FileNotFoundError(f"checkpoint {iter_dir} is empty")
            logger.info(f"[run_job] checkpoint {iter_dir.name}: {len(files)} file(s)")
            for p in files:
                logger.info(f"[run_job]   {p.relative_to(iter_dir)}: {p.stat().st_size} bytes")

    @staticmethod
    def _checkpoint_server_pids() -> list[str]:
        """PIDs of any running checkpoint server on this host — the liveness signal (the process
        outlives its port). The match pattern is bracketed so concurrent checks don't match each other."""
        try:
            out = subprocess.run(["pgrep", "-f", "[c]heckpoint_server"], capture_output=True, text=True, check=False)  # noqa: S607
        except OSError:
            return []
        return out.stdout.split()

    # Cap the per-log console dump so a runaway log can't flood the job output; the crash
    # diagnostics (asan/heap-corruption report, fatal stderr) sit at the tail.
    _SERVER_LOG_TAIL_BYTES = 256 * 1024

    @staticmethod
    def dump_server_log() -> None:
        """Echo this node's checkpoint_server stderr log to the console.

        On a teardown hang the gdb/gcore dumps routinely can't attach (restricted
        ptrace_scope, a concurrent attach racing the same pid, core-file writes failing on
        the scratch mount), so the server's own log is the only reliable diagnostic — yet it
        only lands in the uploaded artifact dir, never on the console. Surface it. The server
        expands %h->hostname and %p->pid when it opens the file (server_main.cpp); resolve %h
        to this node's hostname and glob %p so the match is pid-independent. Never raises."""
        template = os.environ.get("NATIVE_CKPT_SERVER_LOG")
        if not template:
            logger.info("[run_job] NATIVE_CKPT_SERVER_LOG unset; no server log to dump")
            return
        resolved = template.replace("%h", socket.gethostname()).replace("%p", "*")
        resolved_path = Path(resolved)
        matches = sorted(resolved_path.parent.glob(resolved_path.name))
        if not matches:
            logger.info(f"[run_job] no checkpoint_server log found matching {resolved}")
            return
        for log_path in matches:
            try:
                size = log_path.stat().st_size
                with log_path.open("rb") as fp:
                    if size > CheckpointServer._SERVER_LOG_TAIL_BYTES:
                        fp.seek(size - CheckpointServer._SERVER_LOG_TAIL_BYTES)
                        raw = fp.read()
                        raw = raw[raw.find(b"\n") + 1 :]  # drop the partial leading line
                        span = f"last {CheckpointServer._SERVER_LOG_TAIL_BYTES // 1024} KiB of {size} bytes"
                    else:
                        raw = fp.read()
                        span = f"{size} bytes"
            except OSError as ex:
                logger.info(f"[run_job] could not read checkpoint_server log {log_path}: {ex}")
                continue
            logger.info(f"[run_job] === checkpoint_server log {log_path.name} ({span}) ===")
            for line in raw.decode(errors="replace").splitlines():
                logger.info(f"[ncs:{log_path.name}] {line}")
            logger.info(f"[run_job] === end checkpoint_server log {log_path.name} ===")

    @staticmethod
    def _local_server_log_path() -> Path | None:
        """Resolve this node's checkpoint_server log path from the template the server opened
        (server_main.cpp expands %h->hostname and %p->pid). Returns None if unset or, for a
        %p template, if no server process is running to supply the pid."""
        template = os.environ.get("NATIVE_CKPT_SERVER_LOG")
        if not template:
            return None
        resolved = template.replace("%h", socket.gethostname())
        if "%p" in resolved:
            pids = CheckpointServer._checkpoint_server_pids()
            if not pids:
                return None
            resolved = resolved.replace("%p", pids[0])
        return Path(resolved)

    @staticmethod
    def _append_to_server_log(text: str) -> None:
        """Append run_job-side teardown diagnostics (the thread dump) to this node's
        checkpoint_server log, so the backtrace lands in the uploaded artifact alongside the
        server's own crash output — not just inline in the live job trace. Never raises."""
        path = CheckpointServer._local_server_log_path()
        if path is None:
            return
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fp:
                fp.write(f"\n[run_job] ==== run_job teardown thread dump @ {time.strftime('%Y-%m-%dT%H:%M:%S')} ====\n")
                fp.write(text)
                fp.write("\n")

    @staticmethod
    def dump_server_threads() -> None:
        """Best-effort thread dump of any running checkpoint server, to make a stuck process
        diagnosable instead of just killing it. The dump is emitted to the console and appended
        to the node's checkpoint_server log so the backtrace survives in the uploaded artifact
        too. Always captures /proc kernel stacks (which need no ptrace, so they work even when
        gdb can't attach — restricted ptrace_scope, or a sibling rank already attached on a
        multi-rank node) in addition to the symbolized gdb backtrace. Never raises."""
        pids = CheckpointServer._checkpoint_server_pids()
        if not pids:
            logger.info("[run_job] no checkpoint_server process found to dump")
            return
        gdb = shutil.which("gdb")
        vgdb = shutil.which("vgdb")
        chunks: list[str] = []
        for pid in pids:
            cmdline = Path(f"/proc/{pid}/cmdline")
            under_valgrind = cmdline.exists() and "valgrind" in cmdline.read_text(errors="replace")
            header = f"=== checkpoint_server pid {pid} thread dump{' (via vgdb)' if under_valgrind else ''} ==="
            gdb_out = ""
            try:
                if gdb and under_valgrind and vgdb:
                    # A bare `gdb -p` on a valgrind process shows valgrind's frames, not the
                    # guest; attach through valgrind's gdbserver to see the real server threads.
                    cmd = [gdb, "-batch", "-ex", f"target remote | {vgdb} --pid={pid}", "-ex", "thread apply all bt"]
                    res = subprocess.run(cmd, check=False, timeout=120, capture_output=True, text=True)
                    gdb_out = res.stdout + res.stderr
                elif gdb:
                    res = subprocess.run(
                        [gdb, "-p", pid, "-batch", "-ex", "thread apply all bt"],
                        check=False,
                        timeout=120,
                        capture_output=True,
                        text=True,
                    )
                    gdb_out = res.stdout + res.stderr
            except (OSError, subprocess.TimeoutExpired) as ex:
                gdb_out = f"gdb backtrace for pid {pid} failed: {ex}"
            # Kernel per-thread stacks — no ptrace, so they survive when the gdb attach can't.
            kstacks = []
            for task in sorted(Path(f"/proc/{pid}/task").glob("*")):
                with contextlib.suppress(OSError):
                    kstacks.append(f"thread {task.name}:\n{(task / 'stack').read_text().rstrip()}")
            body = "\n".join(filter(None, [gdb_out.strip(), "\n".join(kstacks)]))
            chunk = f"{header}\n{body}"
            chunks.append(chunk)
            logger.info(f"[run_job] {header}")
            for line in body.splitlines():
                logger.info(f"[run_job] {line}")
        CheckpointServer._append_to_server_log("\n\n".join(chunks))

    @staticmethod
    def gcore_and_kill_server() -> None:
        """Snapshot a core file for any running checkpoint server, then kill it — the core aids
        post-mortem and the kill frees the port and shm for the next job. Never raises."""
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
        """Fail loudly (with a thread dump) if a server from a prior job is still running, rather
        than force-killing it; otherwise clear any orphaned shm so the next server starts clean."""
        pids = CheckpointServer._checkpoint_server_pids()
        if pids:
            logger.info(f"[run_job] checkpoint_server already running (pids {', '.join(pids)})")
            CheckpointServer.dump_server_threads()
            raise RuntimeError(
                f"checkpoint_server already running (pids {', '.join(pids)}); refusing to start. "
                f"A previous job did not spin it down — see the thread dump above and stop it before retrying.",
            )
        for shm in Path("/dev/shm").glob("checkpoint_request_*"):  # noqa: S108 — named shm, not a temp file
            try:
                shm.unlink()
            except OSError as ex:  # noqa: PERF203
                logger.info(f"[run_job] could not remove stale shm {shm}: {ex}")

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def request_server_stop() -> None:
        """Best-effort immediate hard stop for the non-graceful paths (stall, crash, teardown);
        the normal completion path spins down gracefully instead."""
        if not CheckpointServer._is_port_in_use(_CHECKPOINT_SERVER_PORT):
            return
        url = f"http://127.0.0.1:{_CHECKPOINT_SERVER_PORT}/api/stop"
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, http.client.HTTPException) as exc:
            logger.info(f"[run_job]   stop request failed: {exc}")

    # Socket read timeout for the spin-down request. The server sends frequent heartbeats while
    # waiting for idle, so this only fires if it goes silent — then the caller kills the process.
    _SPINDOWN_READ_TIMEOUT_S = 60.0

    @staticmethod
    def _request_server_spindown() -> None:
        """Block until the server is idle and spinning down. Unlike a hard stop it leaves in-flight
        saves running, so one node's stop can't abort work the other nodes still need."""
        if not CheckpointServer._is_port_in_use(_CHECKPOINT_SERVER_PORT):
            return
        url = f"http://127.0.0.1:{_CHECKPOINT_SERVER_PORT}/api/spindown"
        try:
            with urllib.request.urlopen(url, timeout=CheckpointServer._SPINDOWN_READ_TIMEOUT_S) as resp:
                resp.read()  # blocks until the server finishes spinning down
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, http.client.HTTPException) as exc:
            logger.info(f"[run_job]  spindown request failed: {exc}")

    @staticmethod
    def _stop_checkpoint_server(timeout: float = 30.0, *, graceful: bool = True) -> bool:
        """Stop the server, then wait for the process to fully exit — it outlives its port, so a
        port check would strand its shm. Graceful waits for idle; the hard path is for crash
        teardown, where waiting for idle would stall since the work never completes. Returns False
        on timeout so the caller can fail."""
        if not CheckpointServer._checkpoint_server_pids():
            return True

        if graceful:
            logger.info("[run_job] checkpoint_server running; requesting graceful spin-down")
            CheckpointServer._request_server_spindown()
        else:
            logger.info("[run_job] checkpoint_server running; requesting immediate stop")
            CheckpointServer.request_server_stop()

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
    def _select_cluster_baseline(baseline: dict, cluster: str) -> dict:
        """Overlay a cluster's metric centers while retaining the platform-wide gates.

        Platforms can have materially different healthy throughput across clusters. The
        tolerance and fixed socket-imbalance threshold remain shared; only metric centers
        explicitly listed under ``cluster_metrics`` are replaced.
        """
        overrides = (baseline.get("cluster_metrics") or {}).get(cluster)
        if not isinstance(overrides, dict):
            return baseline
        selected = dict(baseline)
        selected["metrics"] = {**(baseline.get("metrics") or {}), **overrides}
        return selected

    @staticmethod
    def _report_checkpoint_times(log_text: str) -> None:
        """Print each phase's save-checkpoint times plus the second phase's load time, splitting
        the combined two-phase log on its phase banner."""
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

        metrics = PerfReport._parse_metrics(log_text, job, gpus_per_node)
        cluster = cluster_tag()

        if not metrics:
            if job == "pretrain-nvrx":
                # nvrx provider doesn't emit parseable metrics; times printed above.
                return True
            if in_srun():
                raise RuntimeError(f"no perf metrics found in {log_file}")

        if update_baseline:
            BaselineUpdater.update_baseline_file(
                baseline_file,
                metrics,
                job,
                config,
                flavor_tag,
                gpus_per_node,
                cluster=cluster,
            )
            return True

        baseline = {}
        if baseline_file.exists():
            with baseline_file.open() as f:
                baseline = yaml.safe_load(f)
        baseline = PerfReport._select_cluster_baseline(baseline, cluster)

        comparison, passed = PerfReport._compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
        PerfReport._print_perf_summary(comparison, passed, metrics)

        report = {
            "job": job,
            "config": config,
            "flavor_tag": flavor_tag,
            "gpus_per_node": gpus_per_node,
            "log": str(log_file),
            "baseline": str(baseline_file),
            "cluster": cluster,
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
    def _parse_metrics(log_text: str, job: str, gpus_per_node: int | None = None) -> dict:
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
            wr_samples = [int(m.group(3)) for m in bw_matches]
            bw_metric = PerfReport._make_metric(bw_samples, "MB/s", higher_is_better=True, ranks=ranks)
            wr_metric = PerfReport._make_metric(wr_samples, "us", higher_is_better=False, ranks=ranks)

            # ckptbench currently runs two ranks. Its rank 0 measures the stable local fast
            # path, while the peer's placement-dependent result can vary by an order of
            # magnitude across allocations. Gate the reproducible reference-rank measurement;
            # keep the full spread and rank-imbalance metric below so the skew stays visible.
            unique_ranks = sorted(set(ranks))
            if job == "ckptbench" and len(unique_ranks) == 2:  # noqa: PLR2004
                reference_rank = unique_ranks[0]
                bw_metric["mean"] = statistics.mean(v for r, v in zip(ranks, bw_samples) if r == reference_rank)  # noqa: B905
                wr_metric["mean"] = statistics.mean(v for r, v in zip(ranks, wr_samples) if r == reference_rank)  # noqa: B905
                bw_metric["aggregation"] = f"rank{reference_rank}"
                wr_metric["aggregation"] = f"rank{reference_rank}"

            result.update(
                {
                    "bw_mbps": bw_metric,
                    "wr_tnsr_us": wr_metric,
                    "total_written_bytes": PerfReport._make_metric([int(m.group(4)) for m in bw_matches], "B", None, ranks=ranks),
                },
            )
            # Per-rank bandwidth imbalance (informational): fastest/slowest rank mean.
            # Reaches ~1.48 on h100 from per-rank D2H noise alone, so it is NOT gated.
            per_rank: dict[int, list[float]] = {}
            for r, v in zip(ranks, bw_samples):  # noqa: B905
                per_rank.setdefault(r, []).append(v)
            ordered_ranks = sorted(per_rank)
            rank_means = [statistics.mean(per_rank[r]) for r in ordered_ranks]
            if len(rank_means) > 1 and min(rank_means) > 0:
                result["bw_rank_imbalance"] = PerfReport._make_metric([max(rank_means) / min(rank_means)], "ratio", higher_is_better=None)
                # Socket imbalance (gated): ratio of the two rank-halves' means. It is valid
                # only for a single node with at least two ranks per half. In a multinode job,
                # splitting global ranks in half compares nodes rather than sockets; with only
                # two ranks, it degenerates into the noisy per-rank ratio above.
                is_single_node = gpus_per_node is not None and len(rank_means) == gpus_per_node
                if is_single_node and len(rank_means) >= 4 and len(rank_means) % 2 == 0:  # noqa: PLR2004
                    half = len(rank_means) // 2
                    h0 = statistics.mean(rank_means[:half])
                    h1 = statistics.mean(rank_means[half:])
                    if min(h0, h1) > 0:
                        result["bw_socket_imbalance"] = PerfReport._make_metric(
                            [max(h0, h1) / min(h0, h1)],
                            "ratio",
                            higher_is_better=None,
                        )

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
                # One-sided gate: the two sockets must stay balanced; a breach means a NUMA
                # mis-placement. Only an upper bound is meaningful, so there's no symmetric tolerance.
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
                    "aggregation": measured.get("aggregation"),
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
                    "aggregation": measured.get("aggregation"),
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
                aggregation = f", aggregate={data['aggregation']}" if data.get("aggregation") else ""
                logger.info(
                    f"  [{status}] {name}: {measured:.4f} {unit}  "
                    f"(baseline={baseline:.4f}, Δ={pct_str}, tol=±{tol}%, n={samples}{aggregation})",
                )
                # The pass/fail above is on the mean; print the underlying samples so multi-modal
                # behaviour is visible rather than averaged away.
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
        if len(samples) < 2:  # noqa: PLR2004 — need at least two samples to compute spread
            return
        ranks = metric.get("ranks")
        f = lambda v: f"{v:.0f}"  # noqa: E731 — these are MB/s or microseconds; integer precision is plenty
        logger.info(f"         spread: min={f(min(samples))} median={f(statistics.median(samples))} max={f(max(samples))}")
        if ranks and len(ranks) == len(samples):
            by_rank: dict[int, list] = {}
            for r, s in zip(ranks, samples):  # noqa: B905
                by_rank.setdefault(r, []).append(s)
            per_rank = "  ".join(f"r{r}={f(statistics.mean(by_rank[r]))}" for r in sorted(by_rank))
            logger.info(f"         per-rank mean: {per_rank}")
        logger.info(f"         all values: {' '.join(f(s) for s in samples)}")

    @staticmethod
    def compare_iters_to_baseline(cfg: str, provider: str) -> None:
        """Re-derive each iteration's metadata and diff it against the checked-in baseline: native
        hard-fails on mismatch, nvrx soft-fails. Each compare runs as a subprocess for env parity."""
        strict_on_mismatch = provider == "native"

        ckpt_root = REPO_ROOT / f"ci/ci-runs/{cfg}/checkpoints"
        metadata_root = REPO_ROOT / "metadata"
        iter_dirs = sorted(ckpt_root.glob("iter_*"))
        if not iter_dirs:
            raise RuntimeError(f"no iter_* dirs in {ckpt_root}")
        # Thread REPO_ROOT onto PYTHONPATH so `-m nvrx_checkpointing.scripts.*` resolves
        # regardless of cwd (CI launches from $CI_PROJECT_DIR; manual runs don't).
        env = os.environ.copy()
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
                    "nvrx_checkpointing.scripts.metadata_compare",
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


def _write_compressed_baseline(src: Path, dst_xz: Path) -> None:
    """Write ``src`` into the metadata/ baseline as an xz-compressed file,
    replacing any stale plaintext form. The xz stream carries no timestamp, so
    identical input yields byte-identical output and an unchanged baseline
    produces no git churn."""
    dst_xz.write_bytes(lzma.compress(src.read_bytes(), preset=9 | lzma.PRESET_EXTREME))
    dst_xz.with_suffix("").unlink(missing_ok=True)


class BaselineUpdater:
    @staticmethod
    def _baseline_run_cmds(job: str, config: str, flavor_tag: str, gpus_per_node: int) -> tuple[str, str]:
        gflag = f"--gpus-per-node {gpus_per_node}"
        flavor_flags = "".join(f" --flavor {f}" for f in flavor_tag.split("-") if f)
        if job == "ckptbench":
            script = "python3 ci/scripts/ckptbench.py"
            flags = f"--slurm {gflag}"
        else:
            script = "python3 ci/scripts/run_job.py"
            # config is the run key "{profile}_NxG" (e.g. pico_1x2, nano3_8x4); recover the
            # profile and node count for a runnable reproducer command.
            m = re.match(r"^(?P<profile>.+)_(?P<nodes>\d+)x(?P<gpus>\d+)$", config)
            profile = m.group("profile") if m else "pico"
            nodes = int(m.group("nodes")) if m else 1
            scale = f"--nodes {nodes}" if nodes > 1 else "--slurm"
            flags = f"{job} {scale} --config {profile} {gflag}{flavor_flags}"
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
        """Round a measured mean to the precision its tolerance implies: a couple of significant
        figures for the default perf tolerance, exact integers for exact-match metrics."""
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
    def update_baseline_file(  # noqa: PLR0913
        baseline_file: Path,
        metrics: dict,
        job: str,
        config: str,
        flavor_tag: str,
        gpus_per_node: int,
        *,
        cluster: str = "",
    ) -> None:
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

        platform_metrics = baseline.setdefault("metrics", {})
        overrides = (baseline.get("cluster_metrics") or {}).get(cluster)
        existing = overrides if isinstance(overrides, dict) else platform_metrics
        effective_metrics = {**platform_metrics, **existing}
        tolerances = baseline.get("tolerances") or {}
        changed: list[str] = []
        for name, data in metrics.items():
            # These ratios are deliberately not baseline-relative: rank imbalance is
            # informational, while socket imbalance has a fixed one-sided safety gate.
            # Recording either one would turn run-to-run noise into a brittle exact-match
            # baseline (or misleadingly imply that the socket threshold is configurable).
            if name in ("bw_rank_imbalance", "bw_socket_imbalance"):
                continue
            mean = data.get("mean")
            if mean is None:
                continue
            old_val = effective_metrics.get(name)
            higher_is_better = data.get("higher_is_better")
            # Effective tolerance mirrors _compare_to_baseline. YAML's int-vs-float spelling
            # does not affect comparison semantics: directional metrics use the perf tolerance,
            # while metrics without a direction (for example total_written_bytes) match exactly.
            tol = tolerances.get(name, 0.0 if higher_is_better is None else PERF_TOLERANCE)

            # A baseline centered on multiple healthy runner modes should remain stable when
            # the latest sample is already inside its accepted band. This also avoids harmless
            # within-tolerance CI runs churning the committed value on every manual update.
            if old_val is not None:
                lo = old_val * (1.0 - tol)
                hi = old_val * (1.0 + tol)
                if lo <= mean <= hi:
                    continue

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
        """Resolve the (job, config, flavor) selection from an explicit filter or the CI job name,
        mapping the abbreviated provider/flavor labels in generated job names back to real values."""
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
        """Update baseline files from the prior run's reports without re-running jobs; the
        job/config are auto-detected and overridable by a filter."""
        machine = platform_tag()
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

                socket_result = report.get("comparison", {}).get("bw_socket_imbalance", {})
                if socket_result.get("status") == "FAIL":
                    logger.warning(
                        f"[run_job] skipping perf baseline update from {report_file}: the run breached the fixed socket-imbalance gate",
                    )
                    continue

                baseline_file = BASELINES_DIR / f"{artifact_stem(machine, job, flavor_tag, config)}.yaml"
                BaselineUpdater.update_baseline_file(
                    baseline_file,
                    metrics,
                    job,
                    config,
                    flavor_tag,
                    gpus_per_node,
                    cluster=report.get("cluster", ""),
                )

        # The metadata md5 snapshot is only produced by deterministic runs. nvrx
        # always runs deterministic; native does so only under the deterministic flavor.
        flavor_is_deterministic = "deterministic" in (filter_flavor or "").split("-")
        if filter_job == "pretrain-native" and flavor_is_deterministic:
            BaselineUpdater._update_native_metadata_files(filter_config)
        elif filter_job == "pretrain-nvrx":
            BaselineUpdater._update_nvrx_baseline_files(filter_config)

        # Mirror the captured resolved-arguments block into the baselines (plaintext, not
        # compressed) so the committed baseline records exactly which args each config+flavor ran
        # with. Runs for every job type (native, nvrx, ckptbench); a job that produced no
        # arguments file is simply skipped.
        BaselineUpdater._update_arguments_baselines(machine, filter_job, filter_config, filter_flavor)

    @staticmethod
    def _update_arguments_baselines(machine: str, filter_job: str | None, filter_config: str | None, filter_flavor: str | None) -> None:
        """Copy each ``{stem}_arguments.txt`` produced by the test run into BASELINES_DIR verbatim
        (plaintext), honoring the same job/config/flavor filter as the perf baselines."""
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        for src in sorted(RESULTS_DIR.glob(f"{machine}_{filter_job or '*'}_*_arguments.txt")):
            stem = src.name[: -len("_arguments.txt")]
            if filter_config is not None and not stem.endswith(f"_{filter_config}"):
                continue
            if filter_flavor is not None and f"_{filter_flavor}_" not in stem:
                continue
            shutil.copy2(src, BASELINES_DIR / src.name)
            logger.info(f"[run_job] updated arguments baseline {BASELINES_DIR / src.name}")

    @staticmethod
    def _update_native_metadata_files(filter_config: str | None) -> None:
        """Refresh the single checked-in state registry (``CKPTBENCH_REGISTRY``) from the
        matching deterministic native run. Only the ckptbench/ckpttest input is mirrored;
        other configs' registry.bin files stay runtime-only."""
        if filter_config not in (None, _CKPTBENCH_REGISTRY_CFG):
            return
        src = REPO_ROOT / "ci/ci-runs" / _CKPTBENCH_REGISTRY_CFG / "checkpoints" / f"iter_{_CKPTBENCH_REGISTRY_ITER}" / "registry.bin"
        if not src.is_file():
            # filter_config=None means "update whatever ran"; the registry's config may
            # not have run in this batch, so skip silently. An explicit request for it must
            # find the file.
            if filter_config == _CKPTBENCH_REGISTRY_CFG:
                raise FileNotFoundError(
                    f"no registry.bin at {src} — did the {_CKPTBENCH_REGISTRY_CFG} deterministic"
                    f" native run produce iter_{_CKPTBENCH_REGISTRY_ITER}?",
                )
            return
        CKPTBENCH_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, CKPTBENCH_REGISTRY)
        logger.info(f"[run_job] updated {CKPTBENCH_REGISTRY} from {src}")

    @staticmethod
    def _update_nvrx_baseline_files(filter_config: str | None) -> None:
        """Refresh the checked-in per-tensor md5 baseline (one file per iteration) from the
        latest run's output — the reference both providers compare against."""
        metadata_root = REPO_ROOT / "metadata"
        ci_runs_root = REPO_ROOT / "ci/ci-runs"
        iter_pattern = re.compile(r"^iter_(\d{7})$")
        # Matches the "{profile}_NxG" ci-runs dirs (e.g. pico_1x4, nano4_8x4).
        cfg_pattern = re.compile(r"^.+_\d+x\d+$")

        cfgs = (
            [filter_config]
            if filter_config is not None
            else sorted(d.name for d in ci_runs_root.iterdir() if d.is_dir() and cfg_pattern.match(d.name))
        )

        for cfg in cfgs:
            scale = re.match(r"^.+_(\d+)x\d+$", cfg)
            if scale is not None and int(scale.group(1)) > 2:  # noqa: PLR2004
                # No committed baseline exists at this scale: training there is
                # bit-reproducible only within one allocation, so the deterministic jobs
                # regenerate their reference in-job (the in-allocation reference check).
                logger.info(f"[run_job] skipping metadata baseline for {cfg}: gated by the in-allocation reference check")
                continue
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
                dst = dst_dir / "reference.metadata.info.xz"
                _write_compressed_baseline(src, dst)
                logger.info(f"[run_job] updated {dst} from {src}")


def _run_cfg(args: argparse.Namespace, nodes: int, gpus_per_node: int) -> str:
    """Run-namespace key for baselines, metadata, and the ci-runs dir: ``{config}_{scale}``, so
    two profiles pinned to the same node scale (e.g. nano3 and nano4 at 8x4) keep separate
    baseline/metadata trees."""
    return f"{args.config}_{nodes}x{gpus_per_node}"


def _log_determinism_versions() -> None:
    """Record, from inside the srun/container environment (rank 0 only), the versions of the
    packages whose kernels gate reproducibility, plus whether mamba's deterministic path is
    active. Deterministic mamba/conv1d backward needs mamba-ssm >= 2.3.0 and causal-conv1d
    >= 1.6.0; older bundled wheels make MAMBA_DETERMINISTIC=1 a silent no-op."""
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return
    from importlib.metadata import version as _pkg_version  # noqa: PLC0415

    parts: list[str] = []
    for dist in ("mamba-ssm", "causal-conv1d", "torch", "transformer-engine", "triton"):
        try:
            parts.append(f"{dist}={_pkg_version(dist)}")
        except Exception as exc:  # noqa: BLE001, PERF203
            parts.append(f"{dist}=<{type(exc).__name__}>")
    try:
        from mamba_ssm.utils.determinism import use_deterministic_mode  # pyright: ignore[reportMissingImports] # noqa: PLC0415

        parts.append(f"mamba_deterministic_active={use_deterministic_mode()}")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"mamba_deterministic_active=<{type(exc).__name__}>")
    parts.append(f"MAMBA_DETERMINISTIC={os.environ.get('MAMBA_DETERMINISTIC', '<unset>')}")
    logger.info("[run_job] determinism pkg versions: " + " ".join(parts))  # noqa: G003


class PretrainJob:
    @staticmethod
    def run_pretrain(args: argparse.Namespace) -> None:
        provider = _ckpt_provider(args.job)
        cfg = _run_cfg(args, args.nodes, args.gpus_per_node)
        extra_args = getattr(args, "extra_args", None)

        sys_path_insert(REPO_ROOT.parent)
        sys_path_insert(megatron_root())

        if in_srun():
            _log_determinism_versions()
            # GPUs-per-node arrives on argv (round-tripping it through the container env corrupted it).
            nodes = int(os.environ.get("SLURM_NNODES", "1"))
            cfg = _run_cfg(args, nodes, args.gpus_per_node)
            ctx = RunCtx(args, args.job, cfg, provider, platform_tag(), args.flavors)
            with _flavored_run(ctx):
                common = {"flavors": args.flavors, "extra_args": extra_args, "config_profile": args.config}
                # Valgrind keeps the plain routine at any scale: its gate is memcheck, and the
                # manifest equality it would re-run is covered by the plain deterministic job.
                if ctx.has("deterministic") and not ctx.has("valgrind") and nodes > 2:  # noqa: PLR2004
                    PretrainJob._run_in_allocation_reference_check(cfg, nodes, args.gpus_per_node, provider, common)
                else:
                    PretrainJob._run_two_phase(cfg, nodes, args.gpus_per_node, provider, common)
            return

        # nvrx deterministic via SLURM/multinode up to two nodes runs the two-run cross-run
        # determinism check; perf falls through to a single measured run (the check needs
        # deterministic references). Past two nodes training is bit-reproducible only within
        # one allocation, so two independent runs cannot be compared — the in-allocation
        # reference check (in-srun above) gates those instead.
        is_deterministic = any(f.name == "deterministic" for f in args.flavors)
        if provider == "nvrx" and is_deterministic and (args.nodes > 1 or args.slurm) and args.nodes <= 2:  # noqa: PLR2004
            PretrainJob._run_pretrain_determinism_check(args, cfg)
            return

        # Clear stale checkpoints once in the launcher so in-srun tasks don't race on rmtree.
        # Both phase dirs are cleared so phase 1 always starts from random init.
        for stale in (
            REPO_ROOT / f"ci/ci-runs/{cfg}/checkpoints",
            PretrainJob._phase1_ckpt_dir(cfg),
        ):
            if stale.exists():
                shutil.rmtree(stale)

        ctx = RunCtx(args, args.job, cfg, provider, platform_tag(), args.flavors)
        ctx.log_file = RESULTS_DIR / f"{ctx.stem}.log"
        # The <=2-node determinism check writes {stem}_run1/2.log, and the args capture keeps
        # the last block across those too — drop any stale pair a prior job on this runner left.
        for stale_log in RESULTS_DIR.glob(f"{ctx.stem}_run*.log"):
            stale_log.unlink()
        if args.nodes > 1 or args.slurm:
            # Launcher: training + worker epilogs run in the srun task; this launcher tees the log
            # and runs the perf epilog from it.
            cmd = PretrainJob._build_pretrain_srun_cmd(
                cfg,
                args.nodes,
                args.gpus_per_node,
                args.job,
                args.flavors,
                extra_args=extra_args,
                config_profile=args.config,
            )
            with _flavored_run(ctx):
                ctx.soft_fail = tee_run(cmd, ctx.log_file) == SOFT_FAIL_RC
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
                tee_run(cmd, ctx.log_file, env=env)
                PretrainJob._wait_for_phase1_checkpoint(cfg)
                cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, args.nodes, args.gpus_per_node, phase=2, **common)
                ctx.soft_fail = tee_run(cmd, ctx.log_file, env=env, append=True) == SOFT_FAIL_RC
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
        """Prep-then-run dispatch inside one allocation.

        The allocation-level srun (no container) lands a dispatcher on each
        node; task 0 runs two overlapping inner job steps. The prep step
        installs the native binaries into the shared checkout — the published
        wheel for perf, a source-built wheel for the other flavors — and the
        run step launches the workload. Every rank of the run step then finds
        installed binaries, so the run is identical across flavors and never
        compiles during the measured job.
        """
        image = image_path().as_posix()
        triton_dir = _triton_cache_dir()
        total_tasks = nodes * gpus_per_node

        recipe = Recipe(opts=[], env=Flavor.base_env(triton_dir))
        _apply_flavors(recipe, flavors)
        recipe.srun_time_min = PretrainJob._job_time_min(config_profile, flavors, "srun", recipe.srun_time_min)
        # The srun forwards only what the tasks cannot re-derive: a
        # runner-configured VCPKG_ROOT, plus — for the prep step's registry
        # fetch fallback — the CI coordinates. Everything else the in-srun
        # re-entry rebuilds from argv (the run env is injected into the phase
        # processes from the re-derived recipe), and the flavor build env
        # rides wheel_install's argv like the other per-run knobs.
        srun_common = {
            "time_minutes": recipe.srun_time_min,
            "job_name": f"{job}_{cfg}_{os.environ.get('CI_PIPELINE_ID', 'local')}",
            "image": image,
            "gpus_per_node": gpus_per_node,
            "extra_srun_args": recipe.srun_comments,
        }

        wheel_prep_mode = "--ensure-prebuilt" if [f.name for f in flavors] == ["perf"] else "--build-from-source"
        prep_container_env = "VCPKG_ROOT,CI_API_V4_URL,CI_PROJECT_ID,CI_JOB_TOKEN,CI_COMMIT_SHA,WHEEL_ARCH"
        prep_cmd = [
            *build_srun_cmd(nodes=1, ntasks=1, total_gpus=gpus_per_node, overlap=True, container_env=prep_container_env, **srun_common),
            "python3",
            "-u",
            str(Path(wheel_install.__file__).resolve()),
            wheel_prep_mode,
            *[f"--define={k}={v}" for k, v in recipe.build_env.items()],
        ]
        run_cmd = [
            *build_srun_cmd(
                nodes=nodes,
                ntasks=total_tasks,
                total_gpus=total_tasks,
                overlap=True,
                container_env="VCPKG_ROOT",
                **srun_common,
            ),
            # SLURM_PROCID will be set on each task, triggering in-srun mode.
            "python3",
            "-u",
            str(Path(__file__).resolve()),
            job,
            # Pass per-run knobs on argv so the task re-derives the same recipe from argv, not
            # the container env (a value collision there corrupted forwarded values).
            "--gpus-per-node",
            str(gpus_per_node),
            "--config",
            config_profile,
            *[tok for f in flavors for tok in ("--flavor", f.name)],
            *(extra_args or []),
        ]

        run_dir = REPO_ROOT / f"ci/ci-runs/{cfg}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dispatch = run_dir / "dispatch.sh"
        dispatch.write_text(
            "#!/bin/bash\n"
            "# Generated by run_job.py (_build_pretrain_srun_cmd): prep-then-run job steps\n"
            "# sharing one allocation. The allocation srun carries the run step's full\n"
            "# task footprint (a smaller request would cap the CPUs the inner steps may\n"
            "# use); task 0 drives both steps, every other task exits, and --overlap\n"
            "# lets the inner steps span the whole allocation.\n"
            "set -Eeuo pipefail\n"
            ': "${SLURM_JOB_ID:?dispatch must run inside the srun allocation}"\n'
            'if [ "${SLURM_PROCID:-0}" != "0" ]; then exit 0; fi\n'
            f"{shlex.join(prep_cmd)}\n"
            f"exec {shlex.join(run_cmd)}\n",
        )
        dispatch.chmod(0o755)

        return [
            *build_srun_cmd(
                nodes=nodes,
                ntasks=total_tasks,
                total_gpus=total_tasks,
                strip_container=True,
                label=False,
                container_env="",
                **srun_common,
            ),
            "bash",
            str(dispatch),
        ]

    @staticmethod
    def _phase1_ckpt_dir(cfg: str) -> Path:
        """Phase 1's own checkpoint dir, kept apart so the measured dir holds only later checkpoints."""
        return REPO_ROOT / f"ci/ci-runs/{cfg}/phase1-checkpoints"

    @staticmethod
    def _wait_for_phase1_checkpoint(cfg: str, timeout_seconds: int = 1800) -> None:
        """Block until phase 1's checkpoint is durable on disk, so the fresh next process can load it."""
        marker = PretrainJob._phase1_ckpt_dir(cfg) / f"iter_{_PHASE1_TRAIN_ITERS:07d}" / ".metadata"
        logger.info(f"[run_job] waiting for phase-1 checkpoint to flush: {marker}")
        wait_for_file(marker, timeout_seconds)

    @staticmethod
    def _run_two_phase(cfg: str, nodes: int, gpus_per_node: int, ckpt_provider: str, common: dict, *, routine: str = "") -> None:  # noqa: PLR0913
        """One full checkpoint routine: phase 1 trains from scratch and saves the handoff
        checkpoint, phase 2 loads it and saves the measured checkpoints. Two sequential
        processes in one task share one server lifecycle.

        With ``routine`` set (the reference check, which runs this 32-way inside one srun),
        each task's training output goes to its own file instead of the merged job log —
        nano4's per-expert save enumeration otherwise blows the CI log cap before the
        reference-check verdict prints. Rank 0's file is uploaded as an artifact; a failing
        phase surfaces its tail. Without ``routine`` (the plain path) output is inherited so
        the launcher's tee still feeds the perf parser."""
        redirect = bool(routine)
        log_file = None
        if redirect:
            procid = os.environ.get("SLURM_PROCID", "0")
            # Rank 0's routine log is an artifact (ci-results/*.log); worker ranks park theirs
            # out of the artifact glob to avoid 32-fold bloat, kept only for a failure tail.
            log_root = RESULTS_DIR if procid == "0" else (REPO_ROOT / f"ci/ci-runs/{cfg}/worker-logs")
            log_root.mkdir(parents=True, exist_ok=True)
            log_file = log_root / (f"{cfg}_{routine}-routine.log" if procid == "0" else f"{cfg}_{routine}-routine_rank{procid}.log")
            log_file.unlink(missing_ok=True)
        for phase in (1, 2):
            cmd, env = PretrainJob._pretrain_cmd_and_env(cfg, nodes, gpus_per_node, phase=phase, ckpt_provider=ckpt_provider, **common)
            if redirect:
                with log_file.open("ab") as fh:
                    rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
                if rc != 0:
                    tail = "\n".join(log_file.read_text(errors="replace").splitlines()[-40:])
                    logger.error(f"[run_job] {routine} routine phase {phase} failed (rc={rc}); tail of {log_file.name}:\n{tail}")
                    raise RuntimeError(f"{routine} routine phase {phase} failed with exit code {rc} — see {log_file}")
            else:
                subprocess.run(cmd, env=env, check=True)
            if phase == 1:
                budget = PretrainJob._job_time_min(common["config_profile"], common["flavors"], "phase1-flush", 30)
                PretrainJob._wait_for_phase1_checkpoint(cfg, budget * 60)
                # ValgrindFlavor explicitly owns one server around both phases and disables
                # client auto-launch. Keep that server alive so phase 2 can reuse it; the
                # flavor context stops it after phase 2 and collects the memcheck report.
                if not any(flavor.name == "valgrind" for flavor in common["flavors"]):
                    # Normal runs turn the daemon over at the phase boundary: phase 2 gets a
                    # fresh process whose session table and heap carry nothing from phase 1.
                    # Every local rank holds here until the node's daemon is gone — a sibling
                    # rank that entered phase 2 early would race the port handoff.
                    if os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")) == "0":
                        if not CheckpointServer._stop_checkpoint_server():
                            raise RuntimeError("checkpoint_server did not stop at the phase boundary")
                    else:
                        deadline = time.monotonic() + 90.0
                        while CheckpointServer._checkpoint_server_pids() and time.monotonic() < deadline:
                            time.sleep(0.5)
                        if CheckpointServer._checkpoint_server_pids():
                            raise RuntimeError("checkpoint_server still running 90s into the phase boundary")

    # The provider whose output is the equality reference the other provider is compared
    # against — committed metadata/ baselines at <=2 nodes, the in-allocation reference
    # routine past that.
    _REFERENCE_PROVIDER = "nvrx"
    # Manifest derivation streams every checkpoint byte for the md5s (minutes per iter dir at
    # 8x4), and the non-zero ranks sit at the barrier through four derivations plus the wipe
    # of the multi-TB reference checkpoints.
    _REFERENCE_SYNC_TIMEOUT_S = 7200

    @staticmethod
    def _checkpoint_iter_dirs(cfg: str) -> list[tuple[str, Path]]:
        """Every checkpoint one routine leaves on disk, as (group, iter_dir): the phase-1
        handoff save plus the measured phase-2 saves."""
        groups = (
            ("phase1", PretrainJob._phase1_ckpt_dir(cfg)),
            ("checkpoints", REPO_ROOT / f"ci/ci-runs/{cfg}/checkpoints"),
        )
        out: list[tuple[str, Path]] = []
        for group, root in groups:
            iter_dirs = sorted(d for d in root.glob("iter_*") if d.is_dir())
            if not iter_dirs:
                raise RuntimeError(f"no iter_* checkpoints under {root}")
            out.extend((group, d) for d in iter_dirs)
        return out

    @staticmethod
    def _common_state_lines(dir_path: Path) -> list[str]:
        """The training-STATE lines of common_pt_dump.info (iteration, FLOPs, LR schedule,
        rerun state...). The args.* lines record the invocation, and the two routines'
        invocations differ in provider selection by design (--ckpt-provider and the flags
        derived from it); argument drift has its own gate (_arguments_match_baseline)."""
        lines = (dir_path / "common_pt_dump.info").read_text().splitlines()
        return [ln for ln in lines if not ln.startswith("args.")]

    @staticmethod
    def _derive_manifest(iter_dir: Path, baseline_dir: Path | None = None) -> int:
        """Derive iter_dir's reference.metadata.info + common_pt_dump.info (distcp_trace),
        diffing the manifest against baseline_dir when given. Runs metadata_compare as a
        subprocess for env parity (see compare_iters_to_baseline). Returns its exit code:
        0 equal (or derive-only), 3 manifest mismatch, anything else structural failure."""
        env = os.environ.copy()
        env["TORCHINDUCTOR_WORKER_START"] = "fork"
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        # Derive-only: point at a dir that holds no baseline — the reference provider's lenient
        # bootstrap path then derives the manifest and skips the diff. With a baseline present
        # the provider only names the strict branch should the baseline turn out incomplete.
        provider = PretrainJob._REFERENCE_PROVIDER if baseline_dir is None else "native"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "nvrx_checkpointing.scripts.metadata_compare",
                "--checkpoint-dir",
                str(iter_dir),
                "--baseline-dir",
                str(baseline_dir if baseline_dir is not None else iter_dir / "unseeded"),
                "--provider",
                provider,
            ],
            env=env,
            check=False,
        ).returncode

    @staticmethod
    def _run_in_allocation_reference_check(cfg: str, nodes: int, gpus_per_node: int, provider: str, common: dict) -> None:  # noqa: C901, PLR0912
        """The >2-node deterministic gate: run the checkpoint routine twice inside this one
        allocation — first with the reference provider, then with the job's provider — and
        require bit-identical manifests for every checkpoint (the phase-1 handoff and each
        measured save).

        Training past two nodes is bit-reproducible only on a fixed node set (the reduction
        order of node-spanning collectives follows the allocation's topology), so the
        reference is regenerated here, where both routines share the allocation, instead of
        read from a committed baseline. An nvrx job runs two identical routines — a
        training+save determinism gate; a native job diffs its serialization of the same
        trajectory against the nvrx reference — a cross-provider equality gate."""
        run_dir = REPO_ROOT / f"ci/ci-runs/{cfg}"
        reference_root = run_dir / "reference-run"
        ready_marker = run_dir / "reference-run.ready"
        failed_marker = run_dir / "reference-run.failed"
        is_rank0 = os.environ.get("SLURM_PROCID", "0") == "0"

        # Stale state from a prior job on this runner is cleared before the reference routine
        # starts: every task reaches the marker wait only after that routine's collective
        # training, which cannot begin before this cleanup.
        if is_rank0:
            shutil.rmtree(reference_root, ignore_errors=True)
            ready_marker.unlink(missing_ok=True)
            failed_marker.unlink(missing_ok=True)

        PretrainJob._run_two_phase(cfg, nodes, gpus_per_node, PretrainJob._REFERENCE_PROVIDER, common, routine="reference")

        if is_rank0:
            # This task's phase-2 exit means rank 0's save finalization completed, and that
            # finalization gathers every rank's write results first — the checkpoints are whole.
            try:
                for group, iter_dir in PretrainJob._checkpoint_iter_dirs(cfg):
                    rc = PretrainJob._derive_manifest(iter_dir)
                    if rc != 0:
                        raise RuntimeError(f"reference manifest derivation failed for {group}/{iter_dir.name} (rc={rc})")  # noqa: TRY301
                    stash_dir = reference_root / group / iter_dir.name
                    stash_dir.mkdir(parents=True)
                    for name in ("reference.metadata.info", "common_pt_dump.info"):
                        shutil.copy2(iter_dir / name, stash_dir / name)
                for stale in (run_dir / "checkpoints", PretrainJob._phase1_ckpt_dir(cfg)):
                    shutil.rmtree(stale)
            except BaseException:
                failed_marker.touch()  # release the waiting tasks before propagating
                raise
            ready_marker.touch()
        else:
            wait_for_file(run_dir / "reference-run.*", PretrainJob._REFERENCE_SYNC_TIMEOUT_S, poll_s=15, log_every_s=120)
            if failed_marker.exists():
                raise RuntimeError("reference routine failed on the coordinator task — see its log above")

        PretrainJob._run_two_phase(cfg, nodes, gpus_per_node, provider, common, routine=f"measured-{provider}")

        if not is_rank0:
            return
        produced = PretrainJob._checkpoint_iter_dirs(cfg)
        stashed = {(g.name, d.name) for g in reference_root.iterdir() for d in g.iterdir()}
        if {(g, d.name) for g, d in produced} != stashed:
            raise RuntimeError(
                f"reference check: checkpoint sets differ — reference run saved {sorted(stashed)}, "
                f"{provider} run saved {sorted((g, d.name) for g, d in produced)}",
            )
        failures: list[str] = []
        for group, iter_dir in produced:
            stash_dir = reference_root / group / iter_dir.name
            rc = PretrainJob._derive_manifest(iter_dir, stash_dir)
            if rc != 0:
                failures.append(f"{group}/{iter_dir.name} (manifest rc={rc})")
                continue
            reference = PretrainJob._common_state_lines(stash_dir)
            current = PretrainJob._common_state_lines(iter_dir)
            if reference != current:
                diff = list(difflib.unified_diff(reference, current, fromfile="reference", tofile=provider, n=1))
                logger.info(f"[run_job] reference check: common.pt content differs for {group}/{iter_dir.name} ({len(diff)} diff lines)")
                for line in diff[:30]:
                    logger.info(line.rstrip("\n"))
                failures.append(f"{group}/{iter_dir.name} (common.pt)")
        if failures:
            raise RuntimeError(
                f"reference check FAILED on {len(failures)} checkpoint(s): {', '.join(failures)} — the {provider} "
                f"routine's checkpoints do not match the same-allocation {PretrainJob._REFERENCE_PROVIDER} reference",
            )
        logger.info(
            f"[run_job] reference check PASS: {len(produced)} checkpoint(s) bit-identical between the "
            f"{PretrainJob._REFERENCE_PROVIDER} reference and {provider} routines",
        )

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
        """Build the command and env to run the pretrain entry on a compute node. The first phase
        trains from scratch and saves; the second loads that and saves the measured checkpoints."""
        run_dir = REPO_ROOT / f"ci/ci-runs/{cfg}"
        ckpt_dir = run_dir / "checkpoints"
        phase1_dir = PretrainJob._phase1_ckpt_dir(cfg)
        tb_dir = run_dir / "tensorboard"
        datacache = run_dir / "data-cache"
        triton_dir = _triton_cache_dir()
        for d in [ckpt_dir, phase1_dir, tb_dir, datacache]:
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
            # Memcheck cannot host a second resident snapshot generation at 8-node scale
            # (multi-GB arena allocations stall the reader); save once over the full span.
            valgrind = any(f.name == "valgrind" for f in (flavors or []))
            span = PretrainRunConfig.save_interval * PretrainRunConfig.num_checkpoints
            config = PretrainRunConfig(
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                ckpt_dir=ckpt_dir,
                tb_dir=tb_dir,
                datacache_dir=datacache,
                ckpt_provider=ckpt_provider,
                load_dir=phase1_dir,
                config_profile=config_profile,
                **({"save_interval": span, "num_checkpoints": 1} if valgrind else {}),
            )
        opts = PretrainJob._pretrain_cli_args(config)
        recipe = Recipe(opts=opts, env=Flavor.base_env(triton_dir))
        _apply_flavors(recipe, flavors or [])
        opts = recipe.opts

        env = os.environ.copy()
        env.update(recipe.env)
        # The teardown diagnostics in this wrapper (dump_server_log,
        # _append_to_server_log) resolve the server-log template from their own
        # environment; use the value re-derived here — the container-forwarded
        # copy arrives corrupted through the launcher's --container-env value
        # forwarding on some clusters.
        os.environ["NATIVE_CKPT_SERVER_LOG"] = recipe.env["NATIVE_CKPT_SERVER_LOG"]

        if use_torchrun:
            launcher = ["torchrun", "--nproc_per_node", str(gpus_per_node)]
        else:
            logger.info(
                f"[run_job] in-srun mode: RANK={env['RANK']} LOCAL_RANK={env['LOCAL_RANK']} WORLD_SIZE={env['WORLD_SIZE']}"
                f"MASTER_ADDR={env['MASTER_ADDR']} MASTER_PORT={env['MASTER_PORT']}",
            )
            launcher = ["python3", "-u"]

        # Re-invoke this script in entry mode so the CI markers stay in this tree and the training
        # scripts stay unmodified; the phase follows the sentinel, the rest forwards to training.
        return [*launcher, str(SCRIPT_DIR / "run_job.py"), _PRETRAIN_ENTRY_CMD, str(phase), *opts, *(extra_args or [])], env

    @staticmethod
    def _phase1_config(**kwargs: object) -> PretrainRunConfig:
        """A first-phase config: train from scratch to the handoff iteration with one synchronous,
        durable save that the second phase loads."""
        # Must advertise the full run's iteration count, else the schedule and the weights the
        # second phase resumes from diverge.
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
        # The schedule spans the full run, so the first phase keeps that horizon even though it
        # exits early, matching a single-phase trajectory; later phases fall back to the exit interval.
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
            # nvrx async worker stages D2H copies through CPU shared memory instead
            # of pinned host buffers; pairs with dropping the pre-save GPU-headroom
            # release (empty_cache / free_overlap_buffers) in training.py.
            if "nvrx" in config.ckpt_provider:
                checkpoint_opts.append("--async-ckpt-use-cpu-shm")
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
        # batch to GPU count, nano/ultra3* pin them; eval-interval tracks the full horizon.
        dynamic = {
            "tensor-model-parallel-size": config.gpus_per_node,
            "expert-model-parallel-size": config.gpus_per_node,
            "global-batch-size": config.nodes * config.gpus_per_node,
            "seq-length": seq_len,
            "max-position-embeddings": seq_len,
            "data-cache-path": str(config.datacache_dir),
            "tensorboard-dir": str(config.tb_dir),
            "log-interval": save_interval,
            "eval-interval": train_iters * 2,
            "exit-interval": exit_interval,
            #            "train-iters": train_iters,
        }
        model = PretrainJob._load_model_config(config.config_profile)
        if config.synchronous_save:
            # Phase 1 must persist synchronously so its checkpoint is durable on exit; strip any
            # globally-enabled async-save flags or the wait for its commit marker would time out.
            for async_flag in ("async-save", "async-strategy", "use-persistent-ckpt-worker"):
                model.pop(async_flag, None)
        return _flatten_cli({**dynamic, **model}) + checkpoint_opts

    @staticmethod
    def _load_profile_config(profile: str) -> dict:
        """base.yaml deep-merged with the named profile override, including CI-scoped keys."""
        base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text()) or {}
        override = CONFIGS_DIR / f"{profile}.yaml"
        if not override.exists():
            raise FileNotFoundError(
                f"unknown config profile {profile!r}: {override} not found (available: {', '.join(_available_profiles())})",
            )
        return _deep_merge(base, yaml.safe_load(override.read_text()) or {})

    @staticmethod
    def _load_model_config(profile: str) -> dict:
        """Static model + training args: the profile config minus the CI-scoped keys."""
        return {k: v for k, v in PretrainJob._load_profile_config(profile).items() if k != "job_time_in_mins"}

    @staticmethod
    def _job_time_min(profile: str, flavors: list | None, use_case: str, default: int) -> int:
        """A flavor-keyed time budget from the profile config: job_time_in_mins[use_case]
        maps the sorted flavor-name chain (e.g. "deterministic-valgrind") to minutes,
        falling back to the use-case's "default" entry, then to ``default``."""
        table = (PretrainJob._load_profile_config(profile).get("job_time_in_mins") or {}).get(use_case) or {}
        chain = "-".join(sorted(f.name for f in (flavors or []))) or "default"
        return int(table.get(chain, table.get("default", default)))

    @staticmethod
    def _launch_pretrain_inline(ctx: RunCtx) -> None:
        """Run pretrain in-process for single-GPU runs, tee'ing output so the perf check can parse it."""
        args, cfg = ctx.args, ctx.cfg
        nodes, gpus_per_node = args.nodes, args.gpus_per_node
        run_dir = REPO_ROOT / f"ci/ci-runs/{cfg}"
        ckpt_dir = run_dir / "checkpoints"
        tb_dir = run_dir / "tensorboard"
        datacache = run_dir / "data-cache"
        triton_dir = _triton_cache_dir()
        for d in [ckpt_dir, tb_dir, datacache]:
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
        # Applied before the in-process import, since some env is read at import time.
        os.environ.update(recipe.env)
        # Single-GPU inline is a debugger smoke test: one short single-phase run, so the parser
        # scopes to the whole log. The two-phase resume is exercised by the other paths.

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
        """Call pretrain directly — in the calling process for one GPU, via spawned workers for more."""
        import time  # noqa: PLC0415

        sys_path_insert(REPO_ROOT.parent)
        sys_path_insert(megatron_root())

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
            PretrainJob.pretrain_main(program_start)
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
        PretrainJob.pretrain_main(program_start)

    @staticmethod
    def pretrain_main(program_start: float, ci_phase: int | None = None) -> None:
        """Call pretrain directly, bracketed by the CI-only stdout markers the perf parser reads
        (a phase banner up front and peak-memory on exit), emitted on rank 0 only."""
        import time  # noqa: PLC0415
        from functools import partial  # noqa: PLC0415

        sys_path_insert(REPO_ROOT.parent)
        sys_path_insert(megatron_root())

        from megatron.core.enums import ModelType  # pyright: ignore[reportMissingImports] # noqa: PLC0415
        from megatron.training import (  # pyright: ignore[reportMissingImports] # noqa: PLC0415
            inprocess_restart,
            pretrain,
            set_startup_timestamps,
        )
        from megatron.training.argument_utils import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
            hybrid_config_from_args,
            pretrain_cfg_container_from_args,
        )
        from megatron.training.arguments import parse_and_validate_args  # pyright: ignore[reportMissingImports] # noqa: PLC0415

        try:
            from megatron.post_training.arguments import add_modelopt_args  # pyright: ignore[reportMissingImports] # noqa: PLC0415

            has_modelopt = True
        except ImportError:
            has_modelopt = False

        import pretrain_hybrid as ph  # pyright: ignore[reportMissingImports] # noqa: PLC0415
        from hybrid_builders import hybrid_builder  # pyright: ignore[reportMissingImports] # noqa: PLC0415
        from model_provider import model_provider as _mp  # pyright: ignore[reportMissingImports] # noqa: PLC0415

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
        except Exception as exc:
            PretrainJob._diagnose_checkpoint_failure(exc)
            raise
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
    def _diagnose_checkpoint_failure(exc: BaseException) -> None:
        """Probe the per-node checkpoint server when a training rank fails. Pings its port and,
        only when the failure implicates the server, dumps its log (if dead) or its stack (if alive
        but stuck); stays quiet otherwise. One probe per node; never raises."""
        from nvrx_checkpointing.ckpt_diag import diagnose_checkpoint_server_failure  # pyright: ignore[reportMissingImports] # noqa: PLC0415

        diagnose_checkpoint_server_failure(exc, log_dir=RESULTS_DIR)

    @staticmethod
    def _run_pretrain_determinism_check(args: argparse.Namespace, cfg: str) -> None:  # noqa: C901, PLR0912, PLR0915 — sequential orchestration (run1 → save → wipe → run2 → diff)
        """Run pretrain twice and verify the two runs produce bit-identical checkpoints — the
        determinism gate. A soft-fail against the seeded baseline is tolerated; the cross-run diff
        is what gates."""
        extra_args = getattr(args, "extra_args", None)
        ckpt_dir = REPO_ROOT / f"ci/ci-runs/{cfg}/checkpoints"
        phase1_dir = PretrainJob._phase1_ckpt_dir(cfg)
        sidecar_dir = REPO_ROOT / f"ci/ci-runs/{cfg}/run1-references"
        machine = platform_tag()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = PretrainJob._build_pretrain_srun_cmd(
            cfg,
            args.nodes,
            args.gpus_per_node,
            args.job,
            args.flavors,
            extra_args=extra_args,
            config_profile=args.config,
        )
        stem = artifact_stem(machine, args.job, args.flavor_tag, cfg)

        # phase1-checkpoints is wiped too so each run's phase 1 starts from random init.
        for label, path in (("checkpoints", ckpt_dir), ("phase-1 checkpoints", phase1_dir), ("run-1 sidecar", sidecar_dir)):
            if path.exists():
                logger.info(f"[run_job] determinism: clearing prior {label} at {path}")
                shutil.rmtree(path)

        def _run(log_file: Path, banner: str) -> None:
            logger.info(f"[run_job] {banner}: {' '.join(shlex.quote(c) for c in cmd)}")
            shell_cmd = f"set -o pipefail; {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {shlex.quote(str(log_file))}"
            rc = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", check=False).returncode  # noqa: S602
            # Capture the resolved arguments before any failure check — model init (and the args
            # dump) precedes training, so even a timed-out or crashed run still yields the block.
            capture_arguments(stem)
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

        # Wipe so run 2 starts clean (otherwise it would resume from run 1's checkpoint and skip
        # training); the phase-1 dir is wiped too so its first phase retrains from scratch.
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
    """Config-profile names selectable on the command line (every profile but the shared base)."""
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml") if p.stem != "base")


def detect_local_gpu_count() -> int:
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
        "over configs/base.yaml. pico scales parallelism to the runner's GPUs; nano3/nano4 "
        "(8 nodes) and the ultra3 profiles (16/64 nodes) pin it.",
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
    ensure_vcpkg_root()  # before any `import nvrx_checkpointing` triggers the CMake build
    # Honor the runner's configured scheduler mode (SLURM_ACCOUNT set => SLURM),
    # so the CI YAML needn't pass --slurm in JOB_ARGS.
    if runner_uses_slurm():
        args.slurm = True
    if args.gpus_per_node is None:
        # Prefer the count configured at runner setup (GPUS_PER_NODE), so the CI
        # YAML stays cluster-agnostic; fall back to probing the local host.
        args.gpus_per_node = runner_gpus_per_node() or detect_local_gpu_count()

    # Flavors apply only to pretrain jobs; other jobs keep an empty tag (e.g.
    # ckptbench baselines stay {machine}_ckptbench_{cfg}).
    args.flavors = _resolve_flavors(args.flavor, args.nodes) if _is_pretrain_job(args.job) else []
    args.flavor_tag = "-".join(f.name for f in args.flavors)

    # Build env is read when the import triggers the build, so export it first; children inherit it.
    build_env = Recipe(opts=[], env={})
    _apply_flavors(build_env, args.flavors)
    os.environ.update(build_env.build_env)

    # Perf jobs can skip the source build when the pipeline's wheel job has
    # published a perf wheel for this commit + arch. Only the perf flavor
    # matches the wheel's build config (deterministic runs the plain build,
    # valgrind needs PHOENIX_BUILD_FOR_VALGRIND). The launcher installs once;
    # srun workers see the extracted files through the shared checkout, so
    # in-srun re-entries skip the lookup. The load probe only applies where
    # the binaries run on the runner host itself (non-slurm).
    if not in_srun() and [f.name for f in args.flavors] == ["perf"]:
        wheel_install.install_prebuilt_wheel(probe=not args.slurm)

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
        # Subprocess entry for the launched paths: drop the sentinel and phase, then run in-process.
        ci_phase = int(sys.argv[2])
        sys.argv = ["pretrain_hybrid.py", *sys.argv[3:]]
        PretrainJob.pretrain_main(time.time(), ci_phase=ci_phase)
    else:
        main()
