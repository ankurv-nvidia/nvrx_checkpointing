#!/usr/bin/env python3
"""CI runner for ckpt_stress_test (checkpoint web-service scaling).

Functional gate, no perf baseline. Two phases:

1. Daemon election: exactly one checkpoint_server instance may own the
   port. Start one daemon, then a second candidate; the second must exit
   promptly with "already running" while the first keeps serving. This
   catches any regression in the single-instance election that the
   per-rank launcher relies on (every rank spawns a candidate and the
   losers must yield to the winner).
2. Connect burst: replay a scaled-down synchronized connect storm against
   the in-process server (ckpt_stress_test --fail-on-errors) and fail on
   any refused/reset/timed-out/addr-unavailable connection.

Runs inline on the runner host; the workload is loopback-only and needs
no GPUs.
"""

from __future__ import annotations

import argparse
import contextlib
import http
import logging
import os
import shlex
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from daemon_tools import CHECKPOINT_SERVER
from slurm_job import (
    REPO_ROOT,
    RESULTS_DIR,
    ensure_vcpkg_root,
    platform_tag,
    runner_gpus_per_node,
    tee_run,
)

logger = logging.getLogger("native_ci")

READY_MARKER = "READY"
ELECTION_LOSER_TIMEOUT_S = 15
SERVER_READY_TIMEOUT_S = 60
SERVER_EXIT_TIMEOUT_S = 60
BASE_URL = CHECKPOINT_SERVER.base_url.removesuffix("/api")


def _artifact_stem(*parts: str) -> str:
    return "_".join(filter(None, parts))


def _build_binaries() -> tuple[Path, Path]:
    """Build (via the import-triggered CMake build) and return (stress binary, server binary)."""
    sys.path.insert(0, Path(__file__).parent.parent.parent.parent.as_posix())
    import nvrx_checkpointing as native  # noqa: PLC0415 — triggers CMake build

    build_dir = (
        Path(os.environ["MEGATRON_NATIVE_BUILD_DIR"]).resolve() if os.environ.get("MEGATRON_NATIVE_BUILD_DIR") else REPO_ROOT / "build"
    )
    stress = build_dir / "ckpt_stress_test"
    if not stress.is_file():
        raise FileNotFoundError(f"ckpt_stress_test not found at {stress}")
    server = native.SERVER_PATH
    if not server.is_file():
        raise FileNotFoundError(f"checkpoint_server not found at {server}")
    return stress, server


def _spawn_daemon(server: Path, log_file: Path) -> tuple[subprocess.Popen, threading.Event]:
    """Start a daemon candidate; the returned event fires when it prints READY."""
    env = dict(os.environ, NATIVE_CKPT_SERVER_LOG=str(log_file))
    proc = subprocess.Popen([str(server)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ready = threading.Event()

    def _watch() -> None:
        for line in proc.stdout or []:
            if READY_MARKER in line:
                ready.set()

    threading.Thread(target=_watch, daemon=True).start()
    return proc, ready


def _http(path: str, timeout: float = 5.0) -> int:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as resp:
        return resp.status


def check_daemon_election(server: Path) -> None:
    """Assert the second daemon candidate loses the election and the first keeps serving."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(platform_tag(), "ckpt-stress", "election")
    winner, ready = _spawn_daemon(server, RESULTS_DIR / f"{stem}_winner.%p.log")
    try:
        if not ready.wait(timeout=SERVER_READY_TIMEOUT_S):
            raise RuntimeError(f"checkpoint_server did not print '{READY_MARKER}' within {SERVER_READY_TIMEOUT_S}s")
        logger.info("[ckpt_stress] daemon up (pid %d); starting a second candidate", winner.pid)

        loser, _ = _spawn_daemon(server, RESULTS_DIR / f"{stem}_loser.%p.log")
        try:
            rc = loser.wait(timeout=ELECTION_LOSER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            loser.kill()
            raise RuntimeError(
                "single-instance election is broken: a second checkpoint_server candidate stayed up "
                f"alongside the first instead of exiting 'already running' within {ELECTION_LOSER_TIMEOUT_S}s",
            ) from None
        if rc != 0:
            raise RuntimeError(f"losing checkpoint_server candidate exited {rc}, expected a clean 0")

        status = _http("/api/ping")
        if status != http.HTTPStatus.OK:
            raise RuntimeError(f"daemon stopped serving after the election check: /api/ping -> {status}")
        logger.info("[ckpt_stress] election check passed: loser exited 0, winner still serving")
    finally:
        with contextlib.suppress(OSError):
            _http("/api/stop")
        try:
            winner_rc = winner.wait(timeout=SERVER_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            winner.kill()
            raise RuntimeError(f"checkpoint_server did not exit within {SERVER_EXIT_TIMEOUT_S}s of /api/stop") from None
        if winner_rc != 0:
            raise RuntimeError(f"checkpoint_server exited {winner_rc} after /api/stop, expected a clean 0")


def run_connect_burst(stress: Path, args: argparse.Namespace) -> None:
    """Scaled-down connect storm against the in-process server; hard-fails on any connection failure."""
    gpus = args.gpus_per_node or runner_gpus_per_node()
    if not gpus:
        raise RuntimeError("GPUS_PER_NODE or --gpus-per-node is required")
    stem = _artifact_stem(platform_tag(), "ckpt-stress", f"1x{gpus}")
    cmd = [
        str(stress),
        "--ramp",
        args.ramp,
        "--iters",
        str(args.iters),
        "--saves",
        str(args.saves),
        "--endpoint",
        "ping",
        "--fail-on-errors",
    ]
    # tee_run raises for exit codes other than 0/3; it returns 3 (the perf-drift code
    # other jobs tolerate), which this gate has no use for and treats as a failure.
    rc = tee_run(cmd, RESULTS_DIR / f"{stem}.log")
    if rc != 0:
        raise RuntimeError(f"ckpt_stress_test exited {rc}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(shlex.join(sys.argv))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ramp", default="64,256,1024", help="comma-separated concurrency levels for the burst phase")
    p.add_argument("--iters", type=int, default=3, help="messages each simulated node sends per save")
    p.add_argument("--saves", type=int, default=2, help="saves to replay the ramp over (>1 models steady state)")
    p.add_argument("--gpus-per-node", type=int, default=None, dest="gpus_per_node", help="artifact-name scale tag override")
    args = p.parse_args()

    ensure_vcpkg_root()  # before any `import nvrx_checkpointing` triggers the CMake build
    stress, server = _build_binaries()
    # Fail fast if a prior job left a checkpoint_server bound to the port; both
    # phases need exclusive ownership of it.
    CHECKPOINT_SERVER.require_stopped()
    check_daemon_election(server)
    run_connect_burst(stress, args)
    logger.info("[ckpt_stress] all gates passed")


if __name__ == "__main__":
    main()
