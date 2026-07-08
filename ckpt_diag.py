"""Diagnostic helpers for the per-node checkpoint server.

Pure-stdlib module — no megatron imports — so it can be imported both from
training code (tools.py) and from the CI runner
(ci/scripts/run_job.py) without pulling in heavy deps.
"""

import os
import shutil
import socket
import subprocess
import traceback
from pathlib import Path

# Substrings in an exception message that implicate the checkpoint server.
CKPT_SERVER_FAILURE_MARKERS = (
    "Failed to start or connect to checkpoint service",  # startup / connect
    "No heartbeat from reader",  # server reader stalled mid-save
)


def _print_log(log_dir: Path | str | None, rank: int) -> None:
    # hostname = socket.gethostname()
    logs: list[Path] = []
    if log_dir is None and "NATIVE_CKPT_SERVER_LOG" not in os.environ:
        raise ValueError("log_dir is None and NATIVE_CKPT_SERVER_LOG is not set; cannot find checkpoint server logs")
    log_dir: Path = Path(log_dir) if log_dir is not None else Path(os.environ["NATIVE_CKPT_SERVER_LOG"]).parent
    logs = sorted(p for p in log_dir.glob("checkpoint_server.*.log"))
    logs = list(dict.fromkeys(logs))  # dedupe, preserve order
    if not logs:
        return
    for log_path in logs[-2:]:
        print(f"[native_ckpt:diag][rank{rank}] ==== server log {log_path} (tail) ====", flush=True)  # noqa: T201
        try:
            with Path(log_path).open() as log_file:
                print("".join(log_file.readlines()[-300:]), flush=True)  # noqa: T201
        except OSError as read_err:
            print(f"[native_ckpt:diag][rank{rank}] could not read {log_path}: {read_err}", flush=True)  # noqa: T201


def _get_server_pid(pgrep: str) -> int | None:
    try:
        found = subprocess.run(
            [pgrep, "-f", "checkpoint_server"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        pids = [p for p in found.stdout.split() if p.isdigit()]
        return int(pids[0]) if pids else None
    except (OSError, subprocess.SubprocessError):
        return None


def _print_gdb_stack(gdb: str, pid: int, rank: int) -> None:
    try:
        bt = subprocess.run(
            [gdb, "-p", str(pid), "-batch", "-ex", "thread apply all bt"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        print(bt.stdout, flush=True)  # noqa: T201
        if bt.stderr.strip():
            print(f"[native_ckpt:diag][rank{rank}] gdb stderr: {bt.stderr}", flush=True)  # noqa: T201
    except (OSError, subprocess.SubprocessError) as gdb_err:
        print(f"[native_ckpt:diag][rank{rank}] gdb on server pid {pid} failed: {gdb_err}", flush=True)  # noqa: T201


def _print_stack(rank: int) -> None:
    pid = None
    pgrep = shutil.which("pgrep")
    gdb = shutil.which("gdb")
    if not pgrep or not gdb:
        print(f"[native_ckpt:diag][rank{rank}] cannot probe server stack: missing {'pgrep' if not pgrep else 'gdb'}", flush=True)  # noqa: T201
    pid = _get_server_pid(pgrep)
    if pid is None:
        return
    print(f"[native_ckpt:diag][rank{rank}] ==== server (pid {pid}) stack via gdb ====", flush=True)  # noqa: T201
    _print_gdb_stack(gdb, pid, rank)


def diagnose_checkpoint_server_failure(
    exc: BaseException,
    log_dir: Path | str | None = None,
) -> None:
    """Probe the per-node checkpoint server when a training rank fails.

    Pings its port and, only when the failure implicates the server, dumps its
    log (if dead) or its stack (if alive but stuck); stays quiet otherwise.
    One probe per node; never raises.

    Args:
        exc: The exception that triggered the diagnostic.
        log_dir: Directory where the server writes its log file
            (typically the CI results dir).  When None only /tmp logs are
            considered.
    """
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    host, port = "127.0.0.1", 41000  # checkpoint::ipc::Message::Port — per-node singleton
    marker_hit = any(marker in str(exc) for marker in CKPT_SERVER_FAILURE_MARKERS)

    alive = False
    try:
        with socket.create_connection((host, port), timeout=5):
            alive = True
    except OSError:
        alive = False

    # A healthy server on an unrelated training failure is not worth a diagnostic.
    if alive and not marker_hit:
        return

    print(f"[native_ckpt:diag][rank{rank}] client-side stack trace:", flush=True)  # noqa: T201
    traceback.print_exception(type(exc), exc, exc.__traceback__)

    # Only one rank per node probes that node's singleton server.
    if local_rank != "0":
        return
    if not alive:
        _print_log(log_dir, rank)
    else:
        _print_stack(rank)
