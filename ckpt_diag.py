# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Kernel core setup and post-failure diagnostics for the per-node checkpoint server."""

import atexit
import contextlib
import fcntl
import glob
import json
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_PROC_ROOT = Path("/proc")
_CORE_PATTERN_PATH = _PROC_ROOT / "sys/kernel/core_pattern"
_DEFAULT_SERVER_LOG = "/tmp/native_ckpt_server.%h_%p.log"  # noqa: S108 - daemon default
_CORE_TIMEOUT = 3600
_MAX_CORE_TIMEOUT = 2_147_483
_CORE_LOCK = threading.RLock()


@dataclass
class _CoreCapture:
    executable_fd: int
    started: float
    log_dirs: set[Path]
    core_pattern: str
    core_uses_pid: bool
    working_dir: Path
    executable_path: Path
    ambiguous: bool = False
    initial_cores: dict[Path, tuple[int, int, int, int]] = field(default_factory=dict)


_CORE_CAPTURE: _CoreCapture | None = None


def _core_warning(message: str) -> None:
    print(f"[native_ckpt:diag] core capture: {message}", file=sys.stderr, flush=True)  # noqa: T201


def _core_log_dirs(log_dir: Path | str | None = None) -> set[Path]:
    directories = {Path(os.environ.get("NATIVE_CKPT_SERVER_LOG", _DEFAULT_SERVER_LOG)).parent}
    if log_dir is not None:
        directories.add(Path(log_dir))
    return directories


def _core_pattern_tokens(pattern: str, uses_pid: bool = False) -> list[tuple[bool, str]]:
    tokens = [(part.startswith("%"), part.removeprefix("%")) for part in re.findall(r"[^%]+|%.|%$", pattern)]
    if any(expansion and not value for expansion, value in tokens):
        raise ValueError(f"unsupported incomplete kernel core pattern: {pattern!r}")
    if uses_pid and (True, "p") not in tokens:
        tokens.extend(((False, "."), (True, "p")))
    return tokens


def _core_pattern_info(capture: _CoreCapture) -> tuple[Path, str, re.Pattern[str]]:  # noqa: C901, PLR0912
    pattern = capture.core_pattern
    if not pattern or pattern.startswith(("|", "@")):
        raise ValueError(f"unsupported piped kernel core handler: {pattern!r}; file-core collection requires a file pattern")
    literals = {"%": "%", "h": socket.gethostname(), "u": str(os.getuid()), "g": str(os.getgid())}
    if "%E" in pattern:
        literals["E"] = str(capture.executable_path).replace("/", "!")

    parent, _separator, name = pattern.rpartition("/")
    directory_parts = []
    for expansion, value in _core_pattern_tokens(parent):
        if expansion and value not in literals:
            raise ValueError(f"unsupported variable directory in kernel core pattern: {pattern!r}")
        directory_parts.append(literals[value] if expansion else value)
    directory = Path("/" if pattern.startswith("/") and not parent else "".join(directory_parts))
    if not directory.is_absolute():
        directory = capture.working_dir / directory

    fields = {"p": "pid", "P": "host_pid", "t": "timestamp", "s": "signal"}
    expressions: list[str] = []
    wildcards: list[str] = []
    seen: set[str] = set()
    for expansion, value in _core_pattern_tokens(name, capture.core_uses_pid):
        if not expansion or value in literals:
            literal = literals[value] if expansion else value
            if expansion and value == "h":
                hosts = {literal, literal.split(".", 1)[0]}
                expressions.append("(?:" + "|".join(re.escape(host) for host in sorted(hosts)) + ")")
                wildcards.append("*")
            elif expansion and value == "E":
                expressions.append(re.escape(literal) + r"(?: \(deleted\))?")
                wildcards.append(glob.escape(literal) + "*")
            else:
                expressions.append(re.escape(literal))
                wildcards.append(glob.escape(literal))
        elif value in fields:
            field_name = fields[value]
            expressions.append(rf"(?P={field_name})" if field_name in seen else rf"(?P<{field_name}>\d+)")
            wildcards.append("*")
            seen.add(field_name)
        elif value == "e":
            expressions.append(r"[^/]+")
            wildcards.append("*")
        elif value in {"c", "d", "i", "I", "F"}:
            expressions.append(r"\d+")
            wildcards.append("*")
        else:
            raise ValueError(f"unsupported expansion %{value} in kernel core pattern: {name!r}")
    if "pid" not in seen:
        raise ValueError(f"unsupported kernel core pattern without a namespace process identifier: {name!r}")
    return directory.resolve(), "".join(wildcards), re.compile("".join(expressions))


def _core_paths(capture: _CoreCapture) -> list[Path]:
    directory, pattern, _matcher = _core_pattern_info(capture)
    return list(directory.glob(pattern))


def _core_fingerprint(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size


def enable_checkpoint_server_core_dumps(executable: Path | str, log_dir: Path | str | None = None) -> None:  # noqa: C901
    """Enable inherited kernel dumps and register collection at process exit."""
    global _CORE_CAPTURE  # noqa: PLW0603
    if sys.platform != "linux" or os.environ.get("NATIVE_CKPT_SERVER_CORE_DUMP", "1") == "0":
        return
    try:
        for kind, name, option in ((resource.RLIMIT_CORE, "core", "-c"), (resource.RLIMIT_FSIZE, "file size", "-f")):
            try:
                _soft, hard = resource.getrlimit(kind)
                try:
                    resource.setrlimit(kind, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                except (OSError, ValueError):
                    resource.setrlimit(kind, (hard, hard))
                    if hard != resource.RLIM_INFINITY:
                        _core_warning(
                            f"inherited hard {name} limit is {hard} bytes; full cores are not guaranteed. "
                            f"Set 'ulimit {option} unlimited' before launching the training job.",
                        )
            except (OSError, ValueError) as exc:
                _core_warning(f"cannot raise the inherited {name} limit: {exc}")
        try:
            (_PROC_ROOT / "self/coredump_filter").write_text("0x1ff")
        except OSError as exc:
            _core_warning(f"cannot include all CPU mapping types in inherited cores: {exc}")
        if os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")) != "0":
            return
        with _CORE_LOCK:
            if _CORE_CAPTURE is not None:
                pinned = os.fstat(_CORE_CAPTURE.executable_fd)
                requested = Path(executable).stat()
                if (pinned.st_dev, pinned.st_ino) != (requested.st_dev, requested.st_ino):
                    _CORE_CAPTURE.ambiguous = True
                    _core_warning(
                        "daemon executable changed during collection; "
                        "raw cores remain at the kernel destination to prevent incorrect pairing",
                    )
                    return
                _CORE_CAPTURE.log_dirs.update(_core_log_dirs(log_dir))
                return
            pattern = _CORE_PATTERN_PATH.read_text().strip()
            try:
                uses_pid = int((_PROC_ROOT / "sys/kernel/core_uses_pid").read_text().strip()) != 0
            except FileNotFoundError:
                uses_pid = False
            executable_fd = os.open(executable, os.O_RDONLY | os.O_CLOEXEC)
            try:
                capture = _CoreCapture(
                    executable_fd,
                    time.time(),
                    _core_log_dirs(log_dir),
                    core_pattern=pattern,
                    core_uses_pid=uses_pid,
                    working_dir=Path.cwd().resolve(),
                    executable_path=Path(f"/proc/self/fd/{executable_fd}").readlink(),
                )
                capture.initial_cores = {path: _core_fingerprint(path) for path in _core_paths(capture) if path.is_file()}
            except (OSError, ValueError, RuntimeError):
                os.close(executable_fd)
                raise
            _CORE_CAPTURE = capture
            atexit.register(_finish_core_collection)
    except (OSError, ValueError, RuntimeError) as exc:
        _core_warning(f"cannot initialize collection: {exc}")


def _core_pending(pid: int) -> bool:
    try:
        status = (_PROC_ROOT / str(pid) / "status").read_text()
    except FileNotFoundError:
        return False
    return bool(re.search(r"^CoreDumping:\s+1\b", status, re.MULTILINE)) or not re.search(r"^State:\s+Z\b", status, re.MULTILINE)


def _core_signal_from_log(log_path: Path) -> int:
    with log_path.open(errors="replace") as log:
        for line in log:
            if "[SIGNAL] received SIGQUIT (terminating)" in line:
                return int(signal.SIGQUIT)
            if "[CRASH]" in line and "crashed on" in line:
                for name in ("SIGABRT", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL"):
                    if name in line:
                        return int(getattr(signal, name))
    return 0


def collect_checkpoint_server_core_dumps(wait: bool = False) -> list[Path]:  # noqa: C901, PLR0912, PLR0915
    """Copy completed local daemon cores and their pinned executable beside their logs."""
    collected: list[Path] = []
    timeout = 0
    if wait:
        try:
            timeout = int(os.environ.get("NATIVE_CKPT_SERVER_CORE_DUMP_TIMEOUT_SECONDS", str(_CORE_TIMEOUT)))
        except ValueError:
            timeout = _CORE_TIMEOUT
        if not 1 <= timeout <= _MAX_CORE_TIMEOUT:
            timeout = _CORE_TIMEOUT
    deadline = time.monotonic() + timeout
    while True:
        pending = False
        with _CORE_LOCK:
            capture = _CORE_CAPTURE
            if capture is None or capture.ambiguous:
                return collected
            try:
                paths = _core_paths(capture)
            except (OSError, ValueError, RuntimeError) as exc:
                _core_warning(f"cannot discover daemon cores: {exc}")
                return collected
            for path in paths:
                try:
                    directory, _pattern, matcher = _core_pattern_info(capture)
                    match = matcher.fullmatch(path.name)
                    if path.parent != directory or match is None or path.is_symlink() or not path.is_file():
                        continue
                    metadata = path.stat()
                    if metadata.st_uid != os.getuid() or metadata.st_mtime < int(capture.started):
                        continue
                    if capture.initial_cores.get(path) == _core_fingerprint(path):
                        continue
                    fields = match.groupdict()
                    pid = int(fields["pid"])
                    timestamp = int(fields.get("timestamp") or metadata.st_mtime)
                    if timestamp < int(capture.started):
                        continue
                    host_pid = int(fields.get("host_pid") or pid)
                    fatal_signal = int(fields.get("signal") or 0)
                    hostname = socket.gethostname()
                    hosts = {hostname, hostname.split(".", 1)[0]}
                    directories = capture.log_dirs | _core_log_dirs()
                    pattern = os.environ.get("NATIVE_CKPT_SERVER_LOG", _DEFAULT_SERVER_LOG)
                    candidates = {Path(pattern.replace("%h", host).replace("%p", str(pid))) for host in hosts}
                    for directory in directories:
                        for host in hosts:
                            candidates.add(directory / f"checkpoint_server.{host}.{pid}.log")
                            candidates.update(directory.glob(f"checkpoint_server.*.{host}.{pid}.log"))
                            candidates.update(directory.glob(f"native_ckpt_server.{host}_{pid}.log"))
                    log_path = None
                    for candidate in sorted(candidates):
                        try:
                            metadata = candidate.stat()
                            if (
                                candidate.parent in directories
                                and metadata.st_uid == os.getuid()
                                and metadata.st_mtime >= int(capture.started)
                            ):
                                log_path = candidate
                                break
                        except OSError:
                            continue
                    if log_path is None or (
                        (True, "h") not in _core_pattern_tokens(capture.core_pattern) and not _core_signal_from_log(log_path)
                    ):
                        continue
                    if _core_pending(pid):
                        pending = True
                        continue
                    base = log_path
                    if any(Path(f"{base}{suffix}").exists() for suffix in (".core", ".exe", ".core.log")):
                        base = Path(f"{log_path}.{host_pid}.{timestamp}")
                    core, executable, metadata = (Path(f"{base}{suffix}") for suffix in (".core", ".exe", ".core.log"))
                    partials: list[Path] = []
                    published: list[Path] = []
                    try:
                        with path.open("rb") as source:
                            try:
                                fcntl.flock(source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            if not path.exists() or _core_pending(pid):
                                continue
                            details = {
                                "source": str(path),
                                "pid": pid,
                                "host_pid": host_pid,
                                "timestamp": timestamp,
                                "signal": fatal_signal or _core_signal_from_log(log_path),
                                "core": str(core),
                                "executable": str(executable),
                            }
                            for destination in (core, executable, metadata):
                                partial = Path(f"{destination}.partial")
                                with partial.open("xb") as output:
                                    partials.append(partial)
                                    os.fchmod(output.fileno(), 0o600)
                                    if destination.suffix == ".log":
                                        output.write((json.dumps(details, indent=2) + "\n").encode())
                                    else:
                                        descriptor = capture.executable_fd if destination.suffix == ".exe" else source.fileno()
                                        os.lseek(descriptor, 0, os.SEEK_SET)
                                        with os.fdopen(os.dup(descriptor), "rb") as input_file:
                                            shutil.copyfileobj(input_file, output)
                                    output.flush()
                                    os.fsync(output.fileno())
                            for destination in (executable, metadata, core):
                                os.link(Path(f"{destination}.partial"), destination)
                                published.append(destination)
                            published.clear()
                            path.unlink()
                    except OSError as exc:
                        _core_warning(f"cannot collect {path}: {exc}; the original core is retained")
                        for destination in published:
                            with contextlib.suppress(OSError):
                                destination.unlink()
                        continue
                    finally:
                        for partial in partials:
                            with contextlib.suppress(OSError):
                                partial.unlink()
                    print(f"[native_ckpt:diag] collected checkpoint_server core {core}", flush=True)  # noqa: T201
                    collected.append(core)
                except (OSError, ValueError, RuntimeError) as exc:
                    _core_warning(f"cannot inspect {path}: {exc}")
        if not pending or time.monotonic() >= deadline:
            return collected
        time.sleep(0.25)


def _finish_core_collection() -> None:
    global _CORE_CAPTURE  # noqa: PLW0603
    with _CORE_LOCK:
        capture = _CORE_CAPTURE
        if capture is not None:
            try:
                collect_checkpoint_server_core_dumps(wait=True)
            finally:
                _CORE_CAPTURE = None
                os.close(capture.executable_fd)


# Substrings in an exception message that implicate the checkpoint server.
CKPT_SERVER_FAILURE_MARKERS = (
    "Failed to start or connect to checkpoint service",  # startup / connect
    "No heartbeat from reader",  # server reader stalled mid-save
)


def diagnose_checkpoint_server_failure(  # noqa: C901, PLR0912, PLR0915
    exc: BaseException,
    log_dir: Path | str | None = None,
) -> None:
    """Probe the per-node checkpoint server when a training rank fails.

    Pings its port and, only when the failure implicates the server, dumps its
    local log and attempts to capture its stack; stays quiet otherwise.
    One probe per node; never raises.

    Args:
        exc: The exception that triggered the diagnostic.
        log_dir: Directory where the server writes its log file
            (typically the CI results dir).  When None only /tmp logs are
            considered.
    """
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    if local_rank == "0":
        collect_checkpoint_server_core_dumps(wait=True)
    host, port = "127.0.0.1", 4100  # checkpoint::ipc::Message::Port — per-node singleton
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
    if not alive or marker_hit:
        if log_dir is None and "NATIVE_CKPT_SERVER_LOG" not in os.environ:
            raise ValueError("log_dir is None and NATIVE_CKPT_SERVER_LOG is not set; cannot find checkpoint server logs")
        directory = Path(log_dir) if log_dir is not None else Path(os.environ["NATIVE_CKPT_SERVER_LOG"]).parent
        hostname = socket.gethostname().split(".", 1)[0]

        def _sort_key(path: Path) -> tuple[int, str]:
            try:
                return path.stat().st_mtime_ns, path.name
            except OSError:
                return 0, path.name

        logs = sorted(directory.glob(f"checkpoint_server.{hostname}.*.log"), key=_sort_key)
        if not logs:
            print(  # noqa: T201
                f"[native_ckpt:diag][rank{rank}] no checkpoint server logs found for host {hostname} in {directory}",
                flush=True,
            )
        # Multiple launch candidates can precede the node-local election.
        for log_path in logs[-4:]:
            print(f"[native_ckpt:diag][rank{rank}] ==== server log {log_path} (tail) ====", flush=True)  # noqa: T201
            try:
                with Path(log_path).open() as log_file:
                    print("".join(log_file.readlines()[-100:]), flush=True)  # noqa: T201
            except OSError as read_err:
                print(f"[native_ckpt:diag][rank{rank}] could not read {log_path}: {read_err}", flush=True)  # noqa: T201
    # A failed listener can leave a live process available for stack capture.
    if marker_hit:
        pgrep = shutil.which("pgrep")
        gdb = shutil.which("gdb")
        if not pgrep:
            print(f"[native_ckpt:diag][rank{rank}] cannot probe server stack: missing pgrep", flush=True)  # noqa: T201
            return
        try:
            found = subprocess.run(
                [pgrep, "-f", "[c]heckpoint_server"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            pids = [p for p in found.stdout.split() if p.isdigit()]
            pid = int(pids[0]) if pids else None
        except (OSError, subprocess.SubprocessError):
            pid = None
        if pid is None:
            print(f"[native_ckpt:diag][rank{rank}] no checkpoint_server process found for stack capture", flush=True)  # noqa: T201
            return
        if not gdb:
            print(f"[native_ckpt:diag][rank{rank}] cannot capture server pid {pid} stack: missing gdb", flush=True)  # noqa: T201
            return
        print(f"[native_ckpt:diag][rank{rank}] ==== server (pid {pid}) stack via gdb ====", flush=True)  # noqa: T201
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
