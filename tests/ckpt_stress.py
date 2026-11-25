#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Functional stress gate of the checkpoint web service, without a performance baseline.

1. Daemon election: one checkpoint server owns the port. A second candidate started beside a
   running daemon exits cleanly while the first keeps serving, which the per-rank launcher
   relies on when every rank starts a candidate.
2. Connect burst: a scaled-down synchronized connect storm against the in-process server sees
   no refused, reset, timed-out or address-unavailable connection.

The workload is loopback-only and needs no GPUs.

    ckpt_stress.py --stress STRESS_BINARY --server SERVER_BINARY --results-dir DIR
"""

from __future__ import annotations

import argparse
import contextlib
import http
import logging
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("ckpt_stress")

BASE_URL = "http://127.0.0.1:4100"
READY_MARKER = "READY"
ELECTION_LOSER_TIMEOUT_S = 15
SERVER_READY_TIMEOUT_S = 60
SERVER_EXIT_TIMEOUT_S = 60


def _http(path: str, timeout: float = 5.0) -> int:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
        return response.status


def _hold() -> str:
    """Hold a reference on the daemon until its returned token is released."""
    request = urllib.request.Request(f"{BASE_URL}/api/acquirelocalref/{os.getpid()}", method="POST")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        return response.read().decode().split()[0]


def _release(token: str) -> None:
    request = urllib.request.Request(f"{BASE_URL}/api/releaselocalref/{os.getpid()}/{token}", method="POST")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        response.read()


def _require_free_port() -> None:
    """Both phases need the port to themselves."""
    try:
        _http("/api/ping", timeout=1.0)
    except urllib.error.HTTPError:
        pass
    except OSError:
        return
    raise SystemExit(f"a checkpoint server already answers at {BASE_URL}")


def _spawn_daemon(server: Path, log_file: Path) -> tuple[subprocess.Popen[str], threading.Event]:
    """Start a daemon candidate; the returned event fires when it prints its ready marker."""
    env = {**os.environ, "NATIVE_CKPT_SERVER_LOG": str(log_file)}
    proc = subprocess.Popen([str(server)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ready = threading.Event()

    def watch() -> None:
        for line in proc.stdout or []:
            if READY_MARKER in line:
                ready.set()

    threading.Thread(target=watch, daemon=True).start()
    return proc, ready


def check_daemon_election(server: Path, results_dir: Path) -> None:
    """The second candidate loses the election and the first keeps serving."""
    winner, ready = _spawn_daemon(server, results_dir / "election_winner.%p.log")
    token = None
    try:
        if not ready.wait(timeout=SERVER_READY_TIMEOUT_S):
            raise SystemExit(f"the checkpoint server did not print '{READY_MARKER}' within {SERVER_READY_TIMEOUT_S}s")
        # Without a client reference the winner would wind down after its startup grace.
        token = _hold()
        logger.info("daemon up (pid %d); starting a second candidate", winner.pid)
        loser, _ = _spawn_daemon(server, results_dir / "election_loser.%p.log")
        try:
            rc = loser.wait(timeout=ELECTION_LOSER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            loser.kill()
            raise SystemExit(
                "single-instance election is broken: a second candidate stayed up beside the first "
                f"instead of exiting within {ELECTION_LOSER_TIMEOUT_S}s",
            ) from None
        if rc != 0:
            raise SystemExit(f"the losing candidate exited {rc}, expected a clean 0")
        status = _http("/api/ping")
        if status != http.HTTPStatus.OK:
            raise SystemExit(f"the daemon stopped serving after the election: /api/ping -> {status}")
        logger.info("election passed: the loser exited 0 and the winner still serves")
    finally:
        if token is not None:
            with contextlib.suppress(OSError):
                _release(token)
        try:
            winner_rc = winner.wait(timeout=SERVER_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            winner.kill()
            raise SystemExit(f"the checkpoint server did not wind down within {SERVER_EXIT_TIMEOUT_S}s of its last release") from None
        if winner_rc != 0:
            raise SystemExit(f"the checkpoint server exited {winner_rc} after winding down, expected a clean 0")


def run_connect_burst(stress: Path, args: argparse.Namespace) -> None:
    """A connect storm against the in-process server, failing on any connection failure."""
    cmd = [str(stress), "--ramp", args.ramp, "--iters", str(args.iters), "--saves", str(args.saves)]
    cmd += ["--endpoint", "ping", "--fail-on-errors"]
    with (args.results_dir / "connect_burst.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout or []:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"the stress binary exited {rc}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[ckpt_stress] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stress", type=Path, required=True, help="the connect-storm binary")
    parser.add_argument("--server", type=Path, required=True, help="the checkpoint server binary")
    parser.add_argument("--results-dir", type=Path, required=True, help="directory for the daemon and storm logs")
    parser.add_argument("--ramp", default="64,256,1024", help="comma-separated concurrency levels for the burst phase")
    parser.add_argument("--iters", type=int, default=3, help="messages each simulated node sends per save")
    parser.add_argument("--saves", type=int, default=2, help="saves to replay the ramp over")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    _require_free_port()
    check_daemon_election(args.server, args.results_dir)
    run_connect_burst(args.stress, args)
    logger.info("all gates passed")


if __name__ == "__main__":
    main()
