#!/usr/bin/env python3
"""CI job runner for megatron/native benchmarks and e2e tests.

Runs a job and automatically parses performance results afterwards.
By default only perf numbers are printed; use --skip-perf to see full output.

Jobs:
  ckptbench       Build and run the C++ checkpoint benchmark (1 node, 2 GPUs).
  pretrain-all    Run pretrain_mamba.py with both checkpoint providers (correctness).
  pretrain-native Run pretrain_mamba.py with native checkpoint provider (perf benchmark).
  pretrain-nvrx   Run pretrain_mamba.py with nvrx checkpoint provider (comparison baseline).
  valgrind        Same as pretrain-native, but pre-launches checkpoint_server under
                  valgrind (memcheck) so leaks/errors in the server are reported.
                  Local single-node only — SLURM/multi-node not supported.

Usage:
  python3 run_job.py pretrain-native                 # 1 node, run inline
  python3 run_job.py pretrain-native --nodes 16      # 16 nodes via SLURM
  python3 run_job.py pretrain-native --slurm         # 1 node via SLURM
  python3 run_job.py ckptbench                       # run inline (2 GPUs required locally)
  python3 run_job.py ckptbench --slurm               # run via SLURM
  python3 run_job.py pretrain-native --skip-perf     # tee full output, no perf parsing
  python3 run_job.py pretrain-native --update-baseline  # record current perf as new baseline

Cluster defaults (overridable via environment variables):
  SLURM_ACCOUNT           coreai_resiliency_osiris
  SLURM_PARTITION         batch
  ENROOT_IMAGE            ${HOME}/data/native.<hash>.sqsh
                            (hash = first 10 hex of sha256(megatron/native/scripts/prepare.sh);
                            changing prepare.sh causes a rebuild under a new filename)
  SLURM_GPU_FLAGS_ENABLED 1 (set to 0 on runners that don't support --gpus/--gpus-per-node;
                            the job will run with all GPUs allocated by the scheduler)

Outputs (auto-derived from job and config):
  Logs:      <megatron_root>/megatron/native/ci/ci-results/<machine>_<job>_<config>.log
  Reports:   <megatron_root>/megatron/native/ci/ci-results/<machine>_<job>_<config>_report.json
  Baselines: <script_dir>/../baselines/<machine>_<job>_<config>.yaml

Exit codes:
  0  Job succeeded and all perf metrics within tolerance (or --skip-perf)
  3  Job succeeded but one or more perf metrics regressed beyond tolerance
  1  Job or setup failed (raised as an exception)
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MEGATRON_ROOT = (SCRIPT_DIR / "../../../..").resolve()
BASELINES_DIR = (SCRIPT_DIR / "../baselines").resolve()

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
        sys.stderr.write(f"{prefix}Waiting for {path} to be available...\n")
        time.sleep(10)


def _prepare_sh_hash() -> str:
    """First 10 hex chars of sha256(prepare.sh) — namespaces the built sqsh image so any
    change to prepare.sh forces a rebuild rather than reusing a stale image. Must match
    the hash computed in scripts/enroot.sh."""
    return hashlib.sha256((ENROOT_SCRIPTS_DIR / "prepare.sh").read_bytes()).hexdigest()[:10]


def _default_enroot_image() -> str:
    return f"{Path.home()}/data/native.{_prepare_sh_hash()}.sqsh"


def _machine_tag() -> str:
    """Short machine/cluster identifier used to namespace baseline files.

    Override via MEGATRON_CI_MACHINE; falls back to the hostname stem.
    """
    tag = os.environ.get("MEGATRON_CI_MACHINE", "").strip()
    return tag or socket.gethostname().split(".")[0]


# Ensure the repo root is on sys.path so `import megatron` works regardless of
# how the script is invoked (e.g. from a GitLab runner without a proper install).
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))
RESULTS_DIR = MEGATRON_ROOT / "megatron" / "native" / "ci" / "ci-results"
GPUS_PER_NODE = 4
PERF_TOLERANCE = 0.05  # 5% tolerance — both regressions and improvements beyond this fail

# Stable vcpkg root on shared filesystem — persists installed packages across CI runs.
# Override via VCPKG_ROOT env var. Must be accessible inside the enroot container.
VCPKG_ROOT = os.environ.get("VCPKG_ROOT", str(Path.home() / "data/vcpkg"))
os.environ.setdefault("VCPKG_ROOT", VCPKG_ROOT)  # ensure subprocesses and os.environ.copy() inherit it
_BASELINE_CHANGE_THRESHOLD = 1e-6  # minimum relative change to count as a meaningful baseline update

# Maps each pretrain job name to the --ckpt-provider value it exercises.
_PRETRAIN_JOBS: dict[str, str] = {
    "pretrain-all": "nvrx,native",  # single-node correctness test (both providers)
    "pretrain-native": "native",  # single-node native-checkpoint perf benchmark
    "pretrain-nvrx": "nvrx",  # single-node nvrx baseline for comparison
    "valgrind": "native",  # like pretrain-native, but checkpoint_server runs under valgrind
}

# ---------------------------------------------------------------------------
# Enroot image build
# ---------------------------------------------------------------------------


def _ensure_enroot_image(image: Path) -> None:
    """Build the enroot .sqsh image if it doesn't exist by delegating to enroot.sh."""
    if image.exists():
        return
    base_image = image.parent / "pytorch2604.sqsh"
    sys.stderr.write(f"[run_job] enroot image not found: {image} — building now (this may take ~30 min)\n")

    enroot_sh = ENROOT_SCRIPTS_DIR / "enroot.sh"
    env = os.environ.copy()
    env["ENROOT_IMAGE_BASE"] = base_image.as_posix()
    env["ENROOT_IMAGE_OUT"] = image.as_posix()
    subprocess.run([shutil.which("bash"), enroot_sh.as_posix()], env=env, check=True)


# ---------------------------------------------------------------------------
# SLURM context detection
# ---------------------------------------------------------------------------


def in_srun() -> bool:
    return "SLURM_PROCID" in os.environ


def _cleanup_stale_checkpoint_server() -> None:
    """Kill stray checkpoint_server processes and clear named shm segments on this host.

    The daemon (``server_main.cpp``) calls ``setsid()`` and survives parent death, then
    pins ``messages.h:Message::Port`` (41000). Any prior CI job on this node that
    crashed before clean shutdown leaves the daemon listening on that port, so when
    new ``checkpoint_server`` forks try to bind they hit EADDRINUSE and exit with
    ``ErrorAlreadyRunning`` (``server.cpp:1427``). The clients then see "Failed to
    start or connect to checkpoint service" — observed on h200 nodes that hosted
    earlier failed pipelines on this branch.

    Kill any leftover daemon and remove the named shm queues at
    ``/dev/shm/checkpoint_request_*`` (named after
    ``Message::RequestChannelName`` in ``messages.h``). Use ``-f`` matching the
    full cmdline anchored on ``/megatron/native/build/checkpoint_server`` —
    ``-x`` matches the kernel ``comm`` field which is truncated to 15 chars and
    silently misses our 17-char binary name; the path anchor keeps us narrow
    enough to avoid hitting unrelated processes that mention the string.
    """
    subprocess.run(
        ["pkill", "-f", "/megatron/native/build/checkpoint_server"],
        check=False,
    )
    for shm in Path("/dev/shm").glob("checkpoint_request_*"):
        try:
            shm.unlink()
        except OSError as ex:
            sys.stderr.write(f"[run_job] could not remove stale shm {shm}: {ex}\n")


def _slurm_nodes_and_gpus() -> tuple[int, int]:
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    raw = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_NTASKS_PER_NODE") or os.environ.get("SLURM_TASKS_PER_NODE")
    if raw is None:
        slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
        raise RuntimeError(
            f"cannot infer gpus_per_node — SLURM_GPUS_PER_NODE and SLURM_TASKS_PER_NODE are both unset. SLURM environment: {slurm_vars}",
        )
    # SLURM_GPUS_PER_NODE may be "gpu:4" or plain "4"
    return nodes, int(str(raw).split(":")[-1])


def _slurm_gpu_flags_enabled() -> bool:
    """False on runners that don't support --gpus/--gpus-per-node (set SLURM_GPU_FLAGS_ENABLED=0)."""
    return os.environ.get("SLURM_GPU_FLAGS_ENABLED", "1").strip() not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Job: ckptbench
# ---------------------------------------------------------------------------


def _ckptbench_binary() -> Path:
    import megatron.native as _native  # noqa: F401, PLC0415 — triggers CMake build

    native_dir = MEGATRON_ROOT / "megatron/native"
    build_dir = (
        Path(os.environ["MEGATRON_NATIVE_BUILD_DIR"]).resolve() if os.environ.get("MEGATRON_NATIVE_BUILD_DIR") else native_dir / "build"
    )
    binary = build_dir / "ckptbench"
    if not binary.is_file():
        raise FileNotFoundError(f"ckptbench not found at {binary}")
    return binary


def run_ckptbench(args: argparse.Namespace) -> None:
    registry = MEGATRON_ROOT / "megatron/native/metadata/1x2_checkpoints_iter_0000250/registry.bin"
    if not registry.is_file():
        raise FileNotFoundError(f"stateregistry not found: {registry}")

    if in_srun():
        binary = _ckptbench_binary()
        bench_cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]
        subprocess.run(bench_cmd, check=True)
        return

    if args.slurm:
        account = os.environ.get("SLURM_ACCOUNT", "coreai_resiliency_osiris")
        partition = os.environ.get("SLURM_PARTITION")
        image = os.environ.get("ENROOT_IMAGE", _default_enroot_image())
        cmd = [
            "srun",
            f"--account={account}",
            *([f"--partition={partition}"] if partition else []),
            "--time=5",
            "--nodes=1",
            "--ntasks=1",
            *([f"--gpus-per-node={args.gpus_per_node}"] if _slurm_gpu_flags_enabled() else []),
            "--mem=0",
            f"--job-name=ckpt_benchmark_{os.environ.get('CI_PIPELINE_ID', 'local')}",
            f"--container-image={image}",
            f"--container-mounts={Path.home()}:{Path.home()},/lustre:/lustre",
            "--container-env=VCPKG_ROOT",  # forward by name — avoids KEY=VALUE ambiguity in pyxis/enroot
            "python3",
            "-u",
            str(Path(__file__).resolve()),
            "ckptbench",
        ]
    else:
        binary = _ckptbench_binary()
        cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]

    _run_and_report_perf(cmd, "ckptbench", f"1x{args.gpus_per_node}", args)


# ---------------------------------------------------------------------------
# Job: pretrain
# ---------------------------------------------------------------------------


def _checkpoint_server_binary() -> Path:
    """Path to the built checkpoint_server binary (built/copied by megatron.native import)."""
    import megatron.native as _native  # noqa: F401, PLC0415 — triggers CMake build

    binary = MEGATRON_ROOT / "megatron/native/checkpoint_server"
    if not binary.is_file():
        raise FileNotFoundError(f"checkpoint_server not found at {binary}")
    return binary


_VALGRIND_READY_MARKER = "Checkpoint server started"
# Server logs this (server.cpp ~L1088) when it begins orderly shutdown. We stop tee'ing
# new output to the console once we see it — the long auto-shutdown wait would otherwise
# repeat the same per-channel lines and valgrind's exit diagnostics — but the unified log
# file still captures everything for post-mortem.
_VALGRIND_STOP_MARKER = "Shutting down server thread"
_CHECKPOINT_SERVER_PORT = 41000


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _stop_stale_checkpoint_server(port: int = _CHECKPOINT_SERVER_PORT, timeout: float = 10.0) -> None:
    """Best-effort: ensure no checkpoint_server is bound to `port` before we launch our own.

    Tries the server's own HTTP /api/stop endpoint first (graceful, targets only the listener);
    if the port stays bound, falls back to `pkill -TERM`. The new server's signal handler logs
    a breadcrumb and exits, so SIGTERM is clean.
    """
    if not _is_port_in_use(port):
        return

    sys.stderr.write(f"[run_job] checkpoint_server port {port} is in use; attempting graceful stop\n")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stop", timeout=2.0):
            pass
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        sys.stderr.write(f"[run_job]   /api/stop failed ({exc}); falling back to pkill\n")
        subprocess.run(["pkill", "-TERM", "-f", "checkpoint_server"], check=False)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_in_use(port):
            sys.stderr.write("[run_job]   stale checkpoint_server gone, port is free\n")
            return
        time.sleep(0.25)

    # Last-resort SIGKILL on whatever is still hanging on to the port.
    sys.stderr.write("[run_job]   port still in use after graceful stop; sending SIGKILL\n")
    subprocess.run(["pkill", "-KILL", "-f", "checkpoint_server"], check=False)
    time.sleep(0.5)
    if _is_port_in_use(port):
        raise RuntimeError(
            f"checkpoint_server port {port} is still in use after stop+kill; cannot launch valgrind instance",
        )


def _watch_log_and_tee(
    server_log: Path,
    proc: subprocess.Popen,
    ready: threading.Event,
) -> None:
    """Tail the unified server log and tee new content to the parent's stderr so server-side
    log lines (written to the file via $NATIVE_CKPT_SERVER_LOG, never through the subprocess
    pipe) also surface in the CI console. Sets `ready` once the readiness marker appears.
    Stops tee'ing once the orderly-shutdown marker appears — the file still captures
    everything past that point, but the console doesn't need the auto-shutdown wait's
    repeated lines and valgrind's exit diagnostics. On process exit, does one final read
    so log lines written between the previous poll and exit aren't lost. Always sets
    `ready` so a stuck waiter doesn't deadlock; the caller disambiguates via proc.poll().
    """
    pos = 0
    ready_b = _VALGRIND_READY_MARKER.encode()
    stop_b = _VALGRIND_STOP_MARKER.encode()
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
            drain()  # Final read in case bytes landed between drain() and poll().
            ready.set()
            return
        time.sleep(0.25)


def _spawn_valgrind_server(
    cmd: list[str],
    server_log: Path,
    log_fp: IO[str],
) -> tuple[subprocess.Popen, threading.Event, threading.Thread]:
    """Spawn checkpoint_server under valgrind. The pump thread copies the subprocess pipe
    (valgrind's --log-fd=2 diagnostics plus the server's stdout "READY") into `log_fp`;
    the watcher thread tails the unified log and is the single source of truth for what
    reaches the parent's stderr (console). The server itself appends its stderr to the
    same file via $NATIVE_CKPT_SERVER_LOG, so the watcher surfaces both streams without
    duplication. The watcher is returned so the caller can join() it after proc.wait()
    — otherwise log lines written during the final ~0.25s of the server's life can be
    lost if main exits before the watcher's next poll.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=1,
        text=True,
    )
    ready = threading.Event()

    def _pump() -> None:
        assert proc.stdout is not None  # noqa: S101 — Popen with stdout=PIPE always has it
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
        # Pipe closed (process exited): signal so a stuck waiter doesn't deadlock.
        ready.set()

    threading.Thread(target=_pump, daemon=True).start()
    watcher = threading.Thread(
        target=_watch_log_and_tee,
        args=(server_log, proc, ready),
        daemon=True,
    )
    watcher.start()
    return proc, ready, watcher


@contextlib.contextmanager
def _valgrind_checkpoint_server(machine: str, cfg: str) -> Iterator[None]:
    """Pre-launch checkpoint_server under valgrind for the duration of the block.

    On enter: starts the server under memcheck, sets CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH=1
    so the in-process client (see checkpoint.cpp::SvcPing) skips its own server fork and
    connects to the pre-launched valgrind-wrapped instance, and blocks until the server
    prints its "Checkpoint server started" ready marker.

    On exit: waits for the server's ~150s no-activity auto-shutdown so valgrind can flush
    its leak report, then raises if valgrind detected errors.

    valgrind diagnostics use --log-fd=2 so they flow through the same pipe as the server's
    own stderr; the pump thread tees that combined stream to both the persisted log file
    and the parent's stderr (live console).
    """
    valgrind = shutil.which("valgrind")
    if valgrind is None:
        raise RuntimeError("valgrind not found in PATH — required for the 'valgrind' job")

    server = _checkpoint_server_binary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server_log = RESULTS_DIR / f"{machine}_valgrind_{cfg}_server.log"
    # Truncate any prior run's log before we (and the server) start appending to it.
    server_log.write_bytes(b"")

    cmd = [
        valgrind,
        "--tool=memcheck",
        "--leak-check=full",
        "--show-leak-kinds=all",
        # "--track-origins=yes",
        "--error-exitcode=42",
        "--log-fd=2",
        str(server),
    ]
    sys.stderr.write("[run_job] Launching checkpoint_server under valgrind\n")
    sys.stderr.write(f"[run_job]   server+valgrind log: {server_log}\n")

    _stop_stale_checkpoint_server()

    os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
    # Server (via RedirectStdio in server_main.cpp) appends its own stderr to this path.
    # Both Python (pump thread) and the server append; O_APPEND on the server side and
    # line-flushed writes here keep entries from interleaving mid-line.
    os.environ["NATIVE_CKPT_SERVER_LOG"] = str(server_log)
    log_fp = server_log.open("a")
    proc, ready, watcher = _spawn_valgrind_server(cmd, server_log, log_fp)

    try:
        if not ready.wait(timeout=600):
            raise RuntimeError(
                f"checkpoint_server did not print '{_VALGRIND_READY_MARKER}' within 600s under valgrind; see {server_log}",
            )
        if proc.poll() is not None:
            raise RuntimeError(
                f"checkpoint_server exited before becoming ready (rc={proc.returncode}); see {server_log}",
            )
        sys.stderr.write("[run_job] checkpoint_server is ready — proceeding with pretrain\n")
        yield
    finally:
        sys.stderr.write(
            "[run_job] Pretrain finished — waiting for checkpoint_server auto-shutdown (~150s) so valgrind can flush its report\n",
        )
        try:
            rc = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            sys.stderr.write("[run_job] checkpoint_server did not auto-shutdown; sending SIGTERM\n")
            proc.terminate()
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
        # Wait for the watcher to finish draining the log file before we close it,
        # otherwise the last ~0.25s of server output never reaches the console.
        watcher.join(timeout=5)
        log_fp.close()

        sys.stderr.write(f"[run_job] valgrind exited with code {rc}\n")
        sys.stderr.write(f"[run_job] server+valgrind log: {server_log}\n")
        if rc != 0:
            raise RuntimeError(f"valgrind reported errors in checkpoint_server (exit {rc}); see {server_log}")


def _maybe_valgrind_ctx(args: argparse.Namespace, cfg: str) -> contextlib.AbstractContextManager[None]:
    """Wrap the pretrain launch in a valgrind+checkpoint_server context when job=valgrind.

    In srun mode every rank sets CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH so its client skips
    the auto-fork; only global rank 0 hosts the valgrind-wrapped server.
    """
    if args.job != "valgrind":
        return contextlib.nullcontext()
    if os.environ.get("SLURM_PROCID", "0") != "0":
        os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
        return contextlib.nullcontext()
    return _valgrind_checkpoint_server(_machine_tag(), cfg)


def run_pretrain(args: argparse.Namespace) -> None:  # noqa: C901
    ckpt_provider = _PRETRAIN_JOBS[args.job]
    cfg = f"{args.nodes}x{args.gpus_per_node}"
    extra_args = getattr(args, "extra_args", None)
    if in_srun():
        nodes, gpus_per_node = _slurm_nodes_and_gpus()
        cfg = f"{nodes}x{gpus_per_node}"
        try:
            with _maybe_valgrind_ctx(args, cfg):
                cmd, env = _pretrain_cmd_and_env(cfg, nodes, gpus_per_node, ckpt_provider=ckpt_provider, extra_args=extra_args)
                subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError:
            _dump_nvrx_vs_native_diffs(cfg)
            raise
        # Keep the srun cgroup alive until the detached checkpoint_server has
        # flushed every iter's native artifacts. Returning here would let srun
        # tear down the cgroup and SIGKILL the server mid-write — most visible
        # at --exit-interval where there is no follow-up training to cover the
        # async drain.
        if args.job in _NATIVE_CKPT_JOBS:
            try:
                _wait_for_native_metadata_files(cfg, timeout_seconds=1800)
            except TimeoutError as ex:
                sys.stderr.write(f"[run_job] in-srun wait for native artifacts timed out: {ex}\n")
        # Run the cross-provider compare inside the srun task so it has access
        # to the container's torch/numpy/megatron.native installation. The
        # launcher process outside srun runs on bare-metal Python that lacks
        # these deps and would crash on `import megatron.native.debug`.
        if args.job == "pretrain-all":
            _run_equality_compare(cfg)
        return

    # Clear stale checkpoints once in the launcher so in-srun tasks don't race on rmtree.
    ckpt_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    # pretrain-all is the only dual-provider job; its correctness check must run even
    # when perf regresses, so capture the SystemExit(3) raised by _run_and_report_perf /
    # _launch_pretrain_inline, run the equality compare, then re-raise. Other SystemExit
    # codes (e.g. an unexpected internal exit) propagate immediately.
    perf_exit: SystemExit | None = None
    try:
        if args.nodes > 1 or args.slurm:
            # Launcher only — orchestration happens inside the srun task (in_srun branch above).
            cmd = _build_pretrain_srun_cmd(cfg, args.nodes, args.gpus_per_node, args.job, extra_args=extra_args)
            _run_and_report_perf(cmd, args.job, cfg, args)
        elif args.gpus_per_node > 1:
            # Multi-GPU devbox: use torchrun subprocess so all worker output is captured in the log.
            # mp.spawn workers don't inherit Python-level sys.stdout/stderr redirections.
            cmd, env = _pretrain_cmd_and_env(
                cfg,
                args.nodes,
                args.gpus_per_node,
                use_torchrun=True,
                ckpt_provider=ckpt_provider,
                extra_args=extra_args,
            )
            with _maybe_valgrind_ctx(args, cfg):
                _run_and_report_perf(cmd, args.job, cfg, args, env=env)
        else:
            # Single-GPU inline: call pretrain() directly in-process — no subprocess or torchrun.
            # This lets you attach a debugger to run_job.py and hit breakpoints in training code.
            with _maybe_valgrind_ctx(args, cfg):
                _launch_pretrain_inline(cfg, args)
    except SystemExit as exc:
        if args.job != "pretrain-all" or exc.code != 3:  # noqa: PLR2004 — 3 = perf regression (see _run_and_report_perf / _launch_pretrain_inline)
            raise
        perf_exit = exc

    # Reached only in the launcher. pretrain-all is the only dual-provider job
    # so the check is gated on that. For slurm runs the compare already ran in
    # the srun task (which has the container's deps); the launcher here is
    # bare-metal and lacks torch/numpy, so skip.
    if args.job == "pretrain-all" and not (args.nodes > 1 or args.slurm):
        _run_equality_compare(cfg)
    if perf_exit is not None:
        raise perf_exit


def _build_pretrain_srun_cmd(cfg: str, nodes: int, gpus_per_node: int, job: str, extra_args: list[str] | None = None) -> list[str]:
    account = os.environ.get("SLURM_ACCOUNT", "coreai_resiliency_osiris")
    partition = os.environ.get("SLURM_PARTITION")
    image = os.environ.get("ENROOT_IMAGE", _default_enroot_image())
    triton_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/triton-cache"
    total_tasks = nodes * gpus_per_node

    env_exports = (
        "NCCL_IB_TIMEOUT=24,UCX_MEM_MMAP_HOOK_MODE=none,UCX_MEM_CUDA_HOOK_MODE=none,"
        "UCX_MEM_MALLOC_HOOKS=n,UCX_ERROR_SIGNALS=,NVTE_FWD_LAYERNORM_SM_MARGIN=16,"
        "NVTE_BWD_LAYERNORM_SM_MARGIN=16,TORCHINDUCTOR_WORKER_START=fork,"
        f"QUANTIZATION_TYPE_DEBUG=1,CUDA_DEVICE_MAX_CONNECTIONS=1,TRITON_CACHE_DIR={triton_dir},"
        "VCPKG_ROOT"  # forward by name — avoids KEY=VALUE ambiguity in pyxis/enroot
    )
    if job == "valgrind":
        # Forward the valgrind build flag; CMakeLists.txt translates it into
        # CPPFORGE_DISABLE_MARCH_NATIVE plus a compile-time macro that shrinks
        # D2H buffer sizes so valgrind's shadow memory stays manageable.
        env_exports += ",PHOENIX_BUILD_FOR_VALGRIND"

    return [
        "srun",
        f"--account={account}",
        *([f"--partition={partition}"] if partition else []),
        f"--nodes={nodes}",
        f"--ntasks={total_tasks}",
        f"--time={90 if job == 'valgrind' else 90}",  # 30
        *([f"--gpus={total_tasks}", f"--gpus-per-node={gpus_per_node}"] if _slurm_gpu_flags_enabled() else []),
        "--mem=0",
        f"--job-name={job}_{cfg}_{os.environ.get('CI_PIPELINE_ID', 'local')}",
        f"--container-image={image}",
        f"--container-mounts={Path.home()}:{Path.home()},/lustre:/lustre",
        f"--container-env={env_exports}",
        # SLURM_PROCID will be set on each task, triggering in-srun mode.
        "python3",
        "-u",
        str(Path(__file__).resolve()),
        job,
        *(extra_args or []),
    ]


def _pretrain_cmd_and_env(  # noqa: PLR0913
    cfg: str,
    nodes: int,
    gpus_per_node: int,
    use_torchrun: bool = False,
    ckpt_provider: str = "nvrx,native",
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Return (cmd, env) to run pretrain_mamba.py on a compute node.

    use_torchrun=True  — devbox multi-GPU; torchrun spawns one process per GPU.
    use_torchrun=False — in-srun; srun already placed one process per GPU.
    """
    run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    datacache = run_dir / "data-cache"
    triton_dir = run_dir / "triton-cache"
    for d in [ckpt_dir, tb_dir, datacache, triton_dir]:
        d.mkdir(parents=True, exist_ok=True)

    opts = _pretrain_cli_args(
        PretrainRunConfig(
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            ckpt_dir=ckpt_dir,
            tb_dir=tb_dir,
            datacache_dir=datacache,
            ckpt_provider=ckpt_provider,
        ),
    )

    env = os.environ.copy()
    env.update(
        {
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
        },
    )

    if use_torchrun:
        launcher = ["torchrun", "--nproc_per_node", str(gpus_per_node)]
    else:
        # In-srun: set PyTorch distributed env vars from SLURM.
        # Port is derived from SLURM_JOB_ID so concurrent jobs don't collide.
        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        master_port = 20000 + (slurm_job_id % 40000)
        try:
            master_addr = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]], text=True).split()[0]  # noqa: S607
        except (subprocess.CalledProcessError, OSError, KeyError, IndexError):
            master_addr = "localhost"
        env.update(
            {
                "RANK": os.environ.get("SLURM_PROCID", "0"),
                "LOCAL_RANK": os.environ.get("SLURM_LOCALID", "0"),
                "WORLD_SIZE": os.environ.get("SLURM_NTASKS", "1"),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            },
        )
        launcher = ["python3", "-u"]

    return [*launcher, str(MEGATRON_ROOT / "pretrain_mamba.py"), *opts, *(extra_args or [])], env


@dataclass
class PretrainRunConfig:
    nodes: int
    gpus_per_node: int
    ckpt_dir: Path
    tb_dir: Path
    datacache_dir: Path
    save_interval: int = 250
    num_checkpoints: int = 3
    ckpt_provider: str = "nvrx,native"

    @property
    def scale(self) -> str:
        return f"{self.nodes}x{self.gpus_per_node}"


def _pretrain_cli_args(config: PretrainRunConfig) -> list[str]:
    ckpt_dir, tb_dir, datacache_dir = config.ckpt_dir, config.tb_dir, config.datacache_dir
    nodes, gpus_per_node = config.nodes, config.gpus_per_node
    save_interval, num_checkpoints = config.save_interval, config.num_checkpoints
    scale = config.scale
    seq_len = 1024

    exit_interval = save_interval * (num_checkpoints)

    if scale == "16node":
        gbs = 3072
        model = [
            "--num-layers",
            "44",
            "--hidden-size",
            "4096",
            "--num-attention-heads",
            "32",
            "--ffn-hidden-size",
            "2688",
            "--kv-channels",
            "128",
            "--mamba-num-heads",
            "128",
            "--hybrid-override-pattern",
            "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEME",
            "--num-experts",
            "512",
            "--moe-router-topk",
            "22",
            "--moe-shared-expert-intermediate-size",
            "5376",
            "--moe-latent-size",
            "1024",
            "--num-layers-at-start-in-bf16",
            "0",
            "--num-layers-at-end-in-bf16",
            "14",
        ]
        parallel = [
            "--tensor-model-parallel-size",
            "2",
            "--expert-model-parallel-size",
            "64",
            "--expert-tensor-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
        ]
    else:
        gbs = nodes * gpus_per_node
        tp = ep = gpus_per_node
        model = [
            "--num-layers",
            "3",
            "--hidden-size",
            "2048",
            "--num-attention-heads",
            "32",
            "--ffn-hidden-size",
            "1344",
            "--kv-channels",
            "128",
            "--mamba-num-heads",
            "64",
            "--hybrid-override-pattern",
            "M*E",
            "--num-experts",
            "32",
            "--moe-router-topk",
            "11",
            "--moe-shared-expert-intermediate-size",
            "2688",
            "--moe-latent-size",
            "1024",
            "--num-layers-at-start-in-bf16",
            "0",
            "--num-layers-at-end-in-bf16",
            "2",
        ]
        parallel = [
            "--tensor-model-parallel-size",
            str(tp),
            "--expert-model-parallel-size",
            str(ep),
            "--expert-tensor-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
        ]

    return [
        # MTP
        "--mtp-num-layers",
        "2",
        "--mtp-hybrid-override-pattern",
        "*E",
        "--calculate-per-token-loss",
        "--mtp-loss-scaling-factor",
        "0.3",
        # Architecture
        "--is-hybrid-model",
        "--first-last-layers-bf16",
        "--untie-embeddings-and-output-weights",
        "--init-method-std",
        "0.014",
        "--position-embedding-type",
        "none",
        "--bf16",
        "--use-mcore-models",
        "--spec",
        "megatron.core.models.mamba.mamba_layer_specs",
        "mamba_stack_spec",
        *model,
        # Parallelism
        *parallel,
        "--sequence-parallel",
        "--no-gradient-accumulation-fusion",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--override-opt_param-scheduler",
        # MoE
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-router-score-function",
        "sigmoid",
        "--moe-grouped-gemm",
        "--moe-aux-loss-coeff",
        "1e-4",
        "--moe-router-topk-scaling-factor",
        "5.0",
        "--moe-router-enable-expert-bias",
        "--moe-router-dtype",
        "fp32",
        "--moe-router-load-balancing-type",
        "seq_aux_loss",
        "--moe-permute-fusion",
        # Fused ops / attention
        "--attention-backend",
        "flash",
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl",
        "native",
        "--use-fused-weighted-squared-relu",
        "--squared-relu",
        "--te-rng-tracker",
        "--ddp-num-buckets",
        "10",
        # Checkpoint
        f"--ckpt-provider={config.ckpt_provider}",
        "--ckpt-format",
        "torch_dist",
        "--ckpt-assume-constant-structure",
        "--async-save",
        "--use-persistent-ckpt-worker",
        "--save",
        str(ckpt_dir),
        "--load",
        str(ckpt_dir),
        "--save-interval",
        str(save_interval),
        "--save-retain-interval",
        str(save_interval),
        # Data — NullTokenizer requires no vocab file; vocab_size matches the production tokenizer
        "--mock-data",
        "--data-cache-path",
        str(datacache_dir),
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "256000",
        # Schedule
        "--seq-length",
        str(seq_len),
        "--max-position-embeddings",
        str(seq_len),
        "--lr-wsd-decay-style",
        "minus_sqrt",
        # Optimizer
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(gbs),
        "--lr",
        "4.5e-4",
        "--min-lr",
        "4.5e-6",
        "--weight-decay",
        "0.1",
        "--clip-grad",
        "1.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        # Regularization
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        # Logging
        "--log-interval",
        str(save_interval),
        "--log-params-norm",
        "--log-num-zeros-in-grad",
        "--log-throughput",
        "--log-progress",
        "--logging-level",
        "20",
        "--disable-straggler-on-startup",
        "--straggler-minmax-count",
        "16",
        "--timing-log-option",
        "minmax",
        # Misc
        "--eval-interval",
        str(exit_interval),
        "--eval-iters",
        "14",
        "--no-mmap-bin-files",
        "--no-create-attention-mask-in-dataloader",
        "--distributed-timeout-minutes",
        "30",
        "--num-workers",
        "1",
        "--disable-gloo-process-groups",
        "--exit-interval",
        str(exit_interval),
        "--train-iters",
        str(exit_interval),
        "--tensorboard-dir",
        str(tb_dir),
    ]


# ---------------------------------------------------------------------------
# Inline pretrain (in-process, no torchrun subprocess)
# ---------------------------------------------------------------------------


def _launch_pretrain_inline(cfg: str, args: argparse.Namespace) -> None:  # noqa: C901
    """Run pretrain in-process (no subprocess/torchrun) for single-GPU devbox runs.

    Output is tee'd to the log file so perf metrics can be parsed afterwards.
    Multi-GPU devbox runs are routed through torchrun subprocess in run_pretrain().
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    machine = _machine_tag()
    log_file = RESULTS_DIR / f"{machine}_{args.job}_{cfg}.log"
    report_file = RESULTS_DIR / f"{machine}_{args.job}_{cfg}_report.json"
    baseline_file = BASELINES_DIR / f"{machine}_{args.job}_{cfg}.yaml"

    nodes, gpus_per_node = args.nodes, args.gpus_per_node
    run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    datacache = run_dir / "data-cache"
    triton_dir = run_dir / "triton-cache"
    for d in [ckpt_dir, tb_dir, datacache, triton_dir]:
        d.mkdir(parents=True, exist_ok=True)

    opts = _pretrain_cli_args(
        PretrainRunConfig(
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            ckpt_dir=ckpt_dir,
            tb_dir=tb_dir,
            datacache_dir=datacache,
            save_interval=1,
            num_checkpoints=2,
            ckpt_provider=_PRETRAIN_JOBS[args.job],
        ),
    )
    opts = [*opts, *(getattr(args, "extra_args", None) or [])]

    os.environ.update(
        {
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
        },
    )

    if args.skip_perf:
        _call_pretrain_inline(opts, nodes, gpus_per_node)
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

    with log_file.open("w") as log_fp:
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(saved_stdout, log_fp)
        sys.stderr = _Tee(saved_stderr, log_fp)
        try:
            _call_pretrain_inline(opts, nodes, gpus_per_node)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if rc != 0:
                raise RuntimeError(f"pretrain exited with code {rc} — full log: {log_file}") from None
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr

    if args.job in _NATIVE_CKPT_JOBS:
        _wait_for_native_metadata_files(cfg, timeout_seconds=1800)

    perf_passed = _parse_and_report(
        args.job,
        cfg,
        log_file,
        baseline_file,
        report_file,
        args.update_baseline,
        args.gpus_per_node,
    )
    if not perf_passed:
        sys.exit(3)


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
        _pretrain_main(program_start)
    else:
        import torch.multiprocessing as mp  # noqa: PLC0415

        mp.spawn(
            _pretrain_worker,
            args=(gpus_per_node, nodes, opts, program_start),
            nprocs=gpus_per_node,
            join=True,
        )


def _pretrain_worker(local_rank: int, gpus_per_node: int, nodes: int, opts: list[str], program_start: float) -> None:
    """Entry point for each torch.multiprocessing.spawn worker (multi-GPU inline)."""
    os.environ["RANK"] = str(local_rank)  # single-node: global rank == local rank
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(nodes * gpus_per_node)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    sys.argv = ["pretrain_mamba.py", *opts]
    _pretrain_main(program_start)


def _pretrain_main(program_start: float) -> None:
    """Mirror of pretrain_hybrid.py's __main__ block — calls pretrain() directly."""
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

    set_startup_timestamps(program_start=program_start, main_entry=time.time())

    ph.train_valid_test_datasets_provider.is_distributed = True

    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_modelopt else None,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    model_cfg = hybrid_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain_fn(
        full_config,
        ph.train_valid_test_datasets_provider,
        partial(_mp, hybrid_builder),
        ModelType.encoder_or_decoder,
        ph.forward_step,
        store=store,
    )


# ---------------------------------------------------------------------------
# Launch + perf post-processing
# ---------------------------------------------------------------------------


def _run_and_report_perf(cmd: list[str], job: str, config: str, args: argparse.Namespace, env: dict | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    machine = _machine_tag()
    log_file = RESULTS_DIR / f"{machine}_{job}_{config}.log"
    report_file = RESULTS_DIR / f"{machine}_{job}_{config}_report.json"
    baseline_file = BASELINES_DIR / f"{machine}_{job}_{config}.yaml"

    if args.skip_perf:
        rc = subprocess.run(cmd, env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Job failed with exit code {rc}")
        return

    sys.stderr.write(f"[run_job] Running command: {' '.join(shlex.quote(c) for c in cmd)}\n")
    # srun+enroot doesn't forward container stdout through a PIPE, so we tee via shell.
    shell_cmd = f"set -o pipefail; {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {shlex.quote(str(log_file))}"
    rc = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", env=env, check=False).returncode  # noqa: S602

    if rc != 0:
        raise RuntimeError(f"Job failed with exit code {rc} — full log: {log_file}")

    if job in _NATIVE_CKPT_JOBS:
        _wait_for_native_metadata_files(config, timeout_seconds=1800)

    sys.stderr.write(f"[run_job] Full log: {log_file}\n")
    perf_passed = _parse_and_report(job, config, log_file, baseline_file, report_file, args.update_baseline, args.gpus_per_node)
    if not perf_passed:
        sys.exit(3)


def _parse_and_report(  # noqa: PLR0913
    job: str,
    config: str,
    log_file: Path,
    baseline_file: Path,
    report_file: Path,
    update_baseline: bool,
    gpus_per_node: int,
) -> bool:
    """Parse perf metrics from log_file, compare to baseline, write report. Returns True if passed."""
    log_text = log_file.read_text(errors="replace")

    ckpt_lines = [ln for ln in log_text.splitlines() if "save-checkpoint" in ln]
    if ckpt_lines:
        sys.stderr.write("--- save-checkpoint times ---\n")
        for ln in ckpt_lines:
            sys.stderr.write(ln + "\n")
        sys.stderr.write("-----------------------------\n")

    metrics = _parse_metrics(log_text, job)

    if not metrics:
        if job == "pretrain-nvrx":
            # nvrx provider doesn't emit parseable metrics; times printed above.
            return True
        if in_srun():
            raise RuntimeError(f"no perf metrics found in {log_file}")

    if update_baseline:
        _update_baseline_file(baseline_file, metrics, job, config, gpus_per_node)
        return True

    baseline = {}
    if baseline_file.exists():
        with baseline_file.open() as f:
            baseline = yaml.safe_load(f)

    comparison, passed = _compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
    _print_perf_summary(comparison, passed)

    report = {
        "job": job,
        "config": config,
        "gpus_per_node": gpus_per_node,
        "log": str(log_file),
        "baseline": str(baseline_file),
        "passed": passed,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in metrics.items()},
        "comparison": comparison,
    }
    with report_file.open("w") as f:
        json.dump(report, f, indent=2)
    return passed


# ---------------------------------------------------------------------------
# Perf log parsers
# ---------------------------------------------------------------------------

_RE_NATIVE_BW = re.compile(
    r"\[native_ckpt:cli\]\[\d+\]\[INFO\]\[perf\] Finished "
    r"bw_mbps=([0-9.]+).*?wr_tnsr_us=([0-9]+).*?total_written_bytes=([0-9]+)",
    re.DOTALL,
)


def _make_metric(samples: list, unit: str, higher_is_better: bool | None) -> dict:
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "unit": unit,
        "higher_is_better": higher_is_better,
    }


def _parse_metrics(log_text: str, _job: str) -> dict:
    """Parse all perf metrics for the given job type from log_text."""
    result = {}

    bw_matches = list(_RE_NATIVE_BW.finditer(log_text))
    if bw_matches:
        result.update(
            {
                "bw_mbps": _make_metric([float(m.group(1)) for m in bw_matches], "MB/s", higher_is_better=True),
                "wr_tnsr_us": _make_metric([int(m.group(2)) for m in bw_matches], "us", higher_is_better=False),
                "total_written_bytes": _make_metric([int(m.group(3)) for m in bw_matches], "B", None),
            },
        )

    return result


# ---------------------------------------------------------------------------
# Baseline comparison + reporting
# ---------------------------------------------------------------------------


def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
    baseline_metrics = baseline.get("metrics", {})
    results = {}
    all_passed = True

    for metric_name, measured in metrics.items():
        higher_is_better = measured.get("higher_is_better")
        measured_val = measured.get("mean")
        if measured_val is None:
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


def _print_perf_summary(comparison: dict, passed: bool) -> None:
    sys.stderr.write("\n" + "=" * 70 + "\n")
    sys.stderr.write(" Performance Results\n")
    sys.stderr.write("=" * 70 + "\n")
    for name, data in comparison.items():
        status = data.get("status", "?")
        measured = data.get("measured")
        baseline = data.get("baseline")
        unit = data.get("unit", "")
        pct = data.get("pct_change")
        if status == "info":
            sys.stderr.write(f"  {name}: {measured:.4f} {unit}  [informational]\n")
        elif status == "no_baseline":
            sys.stderr.write(f"  {name}: {measured:.4f} {unit}  [no baseline]\n")
        elif status in ("PASS", "FAIL"):
            sign = "+" if pct and pct >= 0 else ""
            pct_str = f"{sign}{pct:.1f}%" if pct is not None else "N/A"
            tol = data.get("tolerance_pct", "?")
            samples = data.get("n_samples", "?")
            sys.stderr.write(
                f"  [{status}] {name}: {measured:.4f} {unit}  (baseline={baseline:.4f}, Δ={pct_str}, tol=±{tol}%, n={samples})\n",
            )
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f" Overall: {'PASS' if passed else 'FAIL'}\n")
    sys.stderr.write("=" * 70 + "\n\n")


def _baseline_run_cmds(job: str, config: str, gpus_per_node: int) -> tuple[str, str]:
    script = "python3 megatron/native/ci/scripts/run_job.py"
    gflag = f"--gpus-per-node {gpus_per_node}"
    if job == "ckptbench":
        flags = f"ckptbench --slurm {gflag}"
    elif job == "pretrain" and config.startswith("16"):
        flags = f"pretrain --nodes 16 {gflag}"
    else:
        flags = f"{job} --slurm {gflag}"
    return f"{script} {flags}", f"{script} {flags} --update-baseline"


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round so only the leftmost `sig_figs` digits are non-zero (e.g. 5742.485 -> 5700.0)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, sig_figs - 1 - magnitude)


def _round_for_baseline(mean: float, tolerance: float, higher_is_better: bool | None) -> float | int:
    """Round a measured mean to the precision implied by its tolerance.

    5% (the default for perf metrics) → 2 sig figs so values are easy to read at a glance.
    Exact match (tolerance==0, e.g. info metrics like total_written_bytes) → preserve as int.
    Custom tolerances → 6 decimals, since the user picked a non-default precision deliberately.
    """
    if tolerance == 0.0:
        return int(mean) if mean == int(mean) else round(mean, 6)
    if tolerance == PERF_TOLERANCE:
        rounded = _round_to_sig_figs(mean, 2)
        # Info-only metrics that landed in the perf-default branch (no old baseline) still
        # serialize cleanly as int when the rounded value is integral.
        if higher_is_better is None and rounded == int(rounded):
            return int(rounded)
        return rounded
    return round(mean, 6)


def _update_baseline_file(baseline_file: Path, metrics: dict, job: str, config: str, gpus_per_node: int) -> None:
    if baseline_file.exists():
        with baseline_file.open() as f:
            baseline = yaml.safe_load(f)
    else:
        run_cmd, update_cmd = _baseline_run_cmds(job, config, gpus_per_node)
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

        new_val = _round_for_baseline(mean, tol, higher_is_better)
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


# ---------------------------------------------------------------------------
# Baseline aggregation (update stage — no re-run)
# ---------------------------------------------------------------------------


def _parse_job_filter(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Return (filter_job, filter_config) from --filter or CI_JOB_NAME.

    CI_JOB_NAME format: <platform>:<job>[:<config>][:update_baselines]
      gb200:ckptbench:update_baselines        → (ckptbench, None)
      gb200:pretrain:1x4:update_baselines     → (pretrain, 1x4)
    """
    if args.filter:
        job_name, *rest = args.filter.split(":", 1)
        return job_name, rest[0] if rest else None

    ci_job_name = os.environ.get("CI_JOB_NAME", "")
    if ci_job_name:
        _, job_name, *rest = ci_job_name.removesuffix(":update_baselines").split(":")
        return job_name, rest[0] if rest else None

    return None, None


def run_update_baselines(args: argparse.Namespace) -> None:
    """Update baseline YAMLs from *_report.json files in RESULTS_DIR without re-running jobs.

    Job/config are auto-detected from CI_JOB_NAME; override with --filter.
    """
    machine = _machine_tag()
    filter_job, filter_config = _parse_job_filter(args)

    job_glob = filter_job or "*"
    reports = sorted(RESULTS_DIR.glob(f"{machine}_{job_glob}_*_report.json"))
    if not reports:
        filter_desc = f"{filter_job}:{filter_config}" if filter_config else (filter_job or "all jobs")
        raise RuntimeError(
            f"no report files found in {RESULTS_DIR} matching '{machine}_{job_glob}_*_report.json' "
            f"(filter: {filter_desc}) — did the test stage run first?",
        )

    for report_file in reports:
        with report_file.open() as f:
            report = json.load(f)

        job = report.get("job")
        config = report.get("config")
        gpus_per_node = report.get("gpus_per_node", args.gpus_per_node)
        metrics = report.get("metrics", {})

        if not job or not config or not metrics:
            continue

        if filter_config is not None and config != filter_config:
            continue

        baseline_file = BASELINES_DIR / f"{_machine_tag()}_{job}_{config}.yaml"
        _update_baseline_file(baseline_file, metrics, job, config, gpus_per_node)

    if filter_job == "pretrain-native":
        _update_native_metadata_files(filter_config)


# Jobs that exercise the native checkpoint provider — these spawn the async
# checkpoint_server, so we must wait for its background writes (registry.bin,
# {peerIndex}.metadata.info) before the test stage finishes.
_NATIVE_CKPT_JOBS = ("pretrain-native", "pretrain-all", "valgrind")


def _native_sidecar_dir(iter_dir: Path) -> Path:
    """Resolve where the native checkpoint_server's sidecars live for one iter_*.

    In dual-provider mode the native and nvrx writers each get their own
    sub-directory (iter_*/native, iter_*/nvrx); in single-provider mode the
    native writer drops its files directly into iter_*. Detect from disk so
    callers don't have to thread the config through.

    The fallback is intentionally silent: if iter_*/native is missing for any
    reason other than single-provider mode, callers will report "no files
    matching <pattern> in <iter_dir>" — pointing at the parent rather than
    the missing subdir. That trade is acceptable here because the only
    dual-provider job (pretrain-all) is a CI correctness test where missing
    files surface as a hard FileNotFoundError anyway."""
    native_subdir = iter_dir / "native"
    return native_subdir if native_subdir.is_dir() else iter_dir


def _run_equality_compare(cfg: str) -> None:
    """Compare every iter_*/native against iter_*/nvrx under the run's checkpoint dir.

    Used by the pretrain-all dual-provider correctness test. Runs two passes per iter:

      1. ``checkpoint_equality_compare.py`` — tensor + common.pt semantic equality
         via ``dcp.load`` and ``torch.load``. Asymmetric on common.pt missing keys
         (binding-gap policy) per the script's docstring.
      2. ``distcp_trace`` on each side, then ``compare_reference_metadata`` —
         byte-level metadata + per-chunk md5 diff. ``distcp_trace`` on the native
         side also verifies the server's own metadata claims against on-disk
         truth. The launcher only reaches here once the in-srun branch's
         ``_wait_for_native_metadata_files`` has drained the server's writes,
         so the files Pass 2 reads are already on disk.

    Any pass failing on any iter raises a RuntimeError — propagating as exit 1,
    not exit 3 (gitlab-ci-native.yml treats exit 3 as a perf regression).
    Both passes run as subprocesses so that ``megatron.native.debug``'s
    transitive import of ``megatron.core`` → ``torch._inductor`` reads the
    same ``TORCHINDUCTOR_WORKER_START`` overlay we hand the training
    subprocess; importing in-process from the in-srun parent (which doesn't
    have that env) trips an ``Invalid start method`` assertion in newer torch
    builds.
    """
    ckpt_root = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
    iter_dirs = sorted(ckpt_root.glob("iter_*"))
    if not iter_dirs:
        raise RuntimeError(f"no iter_* dirs in {ckpt_root}")
    # Thread MEGATRON_ROOT onto PYTHONPATH so `-m megatron.native.scripts.*` resolves
    # regardless of cwd (CI launches from $CI_PROJECT_DIR; manual runs don't).
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{MEGATRON_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(MEGATRON_ROOT)
    # Mirror _pretrain_cmd_and_env's torch overlay: the in-srun parent's env
    # came in via srun --container-env, which has parsed our KEY=VALUE,KEY=...
    # blob inconsistently in the past and produced a malformed
    # ``TORCHINDUCTOR_WORKER_START`` value that newer torch._inductor rejects
    # with "Invalid start method: fork=fork-". The training subprocess avoids
    # this because _pretrain_cmd_and_env writes the same overrides explicitly
    # — do the same here for the compare subprocesses.
    env["TORCHINDUCTOR_WORKER_START"] = "fork"
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    failures: list[str] = []
    for iter_dir in iter_dirs:
        native = iter_dir / "native"
        nvrx = iter_dir / "nvrx"
        missing = [name for name, d in (("native", native), ("nvrx", nvrx)) if not d.is_dir()]
        if missing:
            raise RuntimeError(
                f"equality-compare expected dual-provider layout under {iter_dir}, but {missing} subdir(s) missing",
            )
        sys.stderr.write(f"[run_job] equality-compare {iter_dir.name}\n")
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "megatron.native.scripts.checkpoint_equality_compare",
                "--native",
                str(native),
                "--nvrx",
                str(nvrx),
            ],
            env=env,
            check=False,
        ).returncode
        if rc != 0:
            failures.append(f"{iter_dir.name} (checkpoint_equality_compare rc={rc})")
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "megatron.native.scripts.metadata_compare",
                "--native",
                str(native),
                "--nvrx",
                str(nvrx),
            ],
            env=env,
            check=False,
        ).returncode
        if rc != 0:
            failures.append(f"{iter_dir.name} (metadata_compare rc={rc})")
    if failures:
        raise RuntimeError(
            f"compare failed for {len(failures)} iter(s): {'; '.join(failures)}; see stderr above for per-finding detail",
        )


def _dump_nvrx_vs_native_diffs(cfg: str, max_lines: int = 200) -> None:
    """Emit cross-provider diff between nvrx and native's independently-derived
    reference.metadata.info per iter dir, when both writers ran (dual-provider
    mode). Called from the in-srun pretrain failure path so the CI log captures
    what diverged — distcp_trace's own diff print is bounded by file size and
    the gb200 5313-diff case blows past it."""
    checkpoints_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
    if not checkpoints_dir.is_dir():
        return
    compared_iters = 0
    diverged_iters = 0
    for iter_dir in sorted(checkpoints_dir.glob("iter_*")):
        native_ref = iter_dir / "native" / "reference.metadata.info"
        nvrx_ref = iter_dir / "nvrx" / "reference.metadata.info"
        if not (native_ref.is_file() and nvrx_ref.is_file()):
            continue
        compared_iters += 1
        native_lines = native_ref.read_text().splitlines()
        nvrx_lines = nvrx_ref.read_text().splitlines()
        diff = list(difflib.unified_diff(nvrx_lines, native_lines, fromfile=str(nvrx_ref), tofile=str(native_ref), n=1))
        if not diff:
            continue
        diverged_iters += 1
        sys.stderr.write(f"[run_job] nvrx-vs-native metadata diff for {iter_dir.name} ({len(diff)} lines):\n")
        for line in diff[:max_lines]:
            sys.stderr.write(line + "\n")
        if len(diff) > max_lines:
            sys.stderr.write(f"[run_job] ... ({len(diff) - max_lines} more diff lines suppressed; full files in artifacts)\n")
        sys.stderr.flush()
    if compared_iters > 0 and diverged_iters == 0:
        # Signal that the diagnostic ran cleanly so log readers don't wonder
        # whether it was silently skipped.
        sys.stderr.write(f"[run_job] nvrx-vs-native: 0 cross-provider differences across {compared_iters} iter dirs\n")
        sys.stderr.flush()


def _foreach_native_metadata_files(
    filter_config: str | None,
    callback: Callable[[Path, Path], None],
    wait_seconds: int = 0,
) -> None:
    """Iterate over each (src, dst) native metadata file pair, invoking callback for each.

    If wait_seconds > 0, polls for each pattern's first match (per iter dir) before globbing
    — needed because checkpoint_server writes these files asynchronously in the background.
    `wait_only_patterns` are polled but not passed to callback — used to block on files the
    post-pretrain stage consumes downstream but that shouldn't enter the baseline snapshot.
    """
    metadata_root = MEGATRON_ROOT / "megatron/native/metadata"
    ci_runs_root = MEGATRON_ROOT / "megatron/native/ci/ci-runs"
    if not ci_runs_root.is_dir():
        raise FileNotFoundError(f"ci-runs root not found: {ci_runs_root}")

    # `[0-9]*.metadata.info` keeps the per-rank server dumps separate from
    # `reference.metadata.info` (the distcp_trace cross-check) so the latter
    # doesn't bleed into the baseline snapshot.
    files_to_copy = ("registry.bin", "[0-9]*.metadata.info", "*.txt")
    # `common.pt` is consumed by the post-pretrain equality-compare but isn't
    # part of the baseline snapshot — checkpoint_server writes it async, so the
    # post-pretrain wait must gate on it, but `_update_native_metadata_files`
    # must not copy multi-MB torch pickles into `megatron/native/metadata/`.
    wait_only_patterns = ("common.pt",)
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
            sidecar_dir = _native_sidecar_dir(src_iter_dir)
            for pattern in files_to_copy:
                if wait_seconds > 0:
                    wait_for_file(sidecar_dir / pattern, wait_seconds)
                matches = sorted(sidecar_dir.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"no files matching {pattern!r} in {sidecar_dir}")
                for src in matches:
                    dst = dst_dir / src.name
                    callback(src, dst)
            if wait_seconds > 0:
                for pattern in wait_only_patterns:
                    wait_for_file(sidecar_dir / pattern, wait_seconds)


def _wait_for_native_metadata_files(cfg: str, timeout_seconds: int) -> None:
    """Block until checkpoint_server has flushed registry.bin and {peerIndex}.metadata.info
    under ci-runs/<cfg>/checkpoints/iter_* (or iter_*/native in dual-provider mode). Call
    after pretrain returns but before the test stage finishes, so the downstream
    update_baselines stage finds the files."""

    def _log(src: Path, _dst: Path) -> None:
        sys.stderr.write(f"[run_job] Found native metadata file: {src}\n")

    _foreach_native_metadata_files(cfg, _log, wait_seconds=timeout_seconds)


def _update_native_metadata_files(filter_config: str | None) -> None:
    """Refresh registry.bin and {peerIndex}.metadata.info under megatron/native/metadata/.

    Driven by ci-runs/<cfg>/checkpoints/iter_NNNNNNN/ — every iter the just-completed
    pretrain-native run produced gets mirrored to metadata/<cfg>_checkpoints_iter_NNNNNNN/
    (creating the destination folder if needed). When filter_config is set, only that
    cfg is processed.
    """

    def log_and_copy(src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)
        sys.stderr.write(f"[run_job] updated {dst} from {src}\n")

    _foreach_native_metadata_files(filter_config, log_and_copy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    sys.stdout.write(shlex.join(sys.argv) + "\n")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "job",
        choices=["ckptbench", "pretrain", "pretrain-all", "pretrain-native", "pretrain-nvrx", "valgrind", "update_baselines"],
    )
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1 — runs inline; >1 uses SLURM automatically)")
    p.add_argument("--gpus-per-node", type=int, default=None, dest="gpus_per_node", help="GPUs per node (default: auto-detect)")
    p.add_argument("--slurm", action="store_true", help="Submit via SLURM even for single-node runs")
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
        help="update_baselines only: <job>[:<config>] to update, e.g. 'pretrain:1x4' or 'bench'. "
        "Auto-detected from CI_JOB_NAME when running in GitLab CI.",
    )
    args, extra_args = p.parse_known_args()
    args.extra_args = extra_args
    if args.gpus_per_node is None:
        args.gpus_per_node = _detect_local_gpu_count()

    if args.job == "valgrind":
        # Signal CMake that we're building for a valgrind run. CMakeLists.txt
        # turns this into CPPFORGE_DISABLE_MARCH_NATIVE (so valgrind doesn't trip
        # on host-specific instructions) and defines PHOENIX_BUILD_FOR_VALGRIND
        # which shrinks D2H ring-buffer sizes in messages.h. Must be set before
        # any `import megatron.native` triggers a CMake build; subprocess and
        # srun children inherit it via env=os.environ.copy() / container-env
        # forwarding.
        os.environ["PHOENIX_BUILD_FOR_VALGRIND"] = "1"

    # Clean stale checkpoint_server daemons + named shm before starting any
    # work, and again on exit, so the next pipeline lands on a node free of
    # this branch's pollution. Only meaningful on compute nodes: the slurm
    # launcher (args.slurm and not in_srun()) runs on the head node, where no
    # checkpoint_server ever runs, so we skip there.
    is_compute_node = in_srun() or not args.slurm
    if is_compute_node:
        _cleanup_stale_checkpoint_server()
    try:
        if args.job == "update_baselines":
            run_update_baselines(args)
            return

        image = Path(_default_enroot_image())
        if args.slurm:
            if args.job == "pretrain-all" and not in_srun():
                _ensure_enroot_image(image)
            wait_for_file(image, timeout_seconds=3600)
        if args.job == "ckptbench":
            run_ckptbench(args)
        else:
            run_pretrain(args)
    finally:
        if is_compute_node:
            _cleanup_stale_checkpoint_server()


if __name__ == "__main__":
    main()
