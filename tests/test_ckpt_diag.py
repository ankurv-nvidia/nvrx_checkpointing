# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

import fcntl
import json
import os
import resource
import signal
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import ckpt_diag

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

CoreEnvironment = tuple[ckpt_diag._CoreCapture, Path, Path]


@pytest.fixture
def core_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[CoreEnvironment]:
    proc = tmp_path / "proc"
    spool = tmp_path / "spool"
    logs = tmp_path / "logs"
    proc.mkdir()
    spool.mkdir()
    logs.mkdir()
    executable = tmp_path / "checkpoint_server"
    executable.write_bytes(b"original daemon binary")
    capture = ckpt_diag._CoreCapture(
        os.open(executable, os.O_RDONLY),
        1000.0,
        {logs},
        core_pattern=str(spool / "nvrx-core.%h.%p.%P.%t.%s"),
        core_uses_pid=False,
        working_dir=Path.cwd(),
        executable_path=executable.resolve(),
    )
    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", proc)
    monkeypatch.setattr(ckpt_diag, "_CORE_CAPTURE", capture)
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(logs / "checkpoint_server.%h.%p.log"))
    yield capture, executable, logs
    if ckpt_diag._CORE_CAPTURE is capture:
        os.close(capture.executable_fd)


def _write_core(  # noqa: PLR0913 - independent kernel filename fields
    capture: ckpt_diag._CoreCapture,
    logs: Path,
    *,
    host: str = "node-a.example.com",
    pid: int = 123,
    timestamp: int = 1001,
    comm: str = "",
    cycle: str = "",
) -> tuple[Path, Path]:
    source = Path(capture.core_pattern).parent / f"nvrx-core.{comm}{host}.{pid}.456.{timestamp}.11"
    source.write_bytes(b"kernel core payload")
    log = logs / f"checkpoint_server.{cycle}{host}.{pid}.log"
    log.write_text("daemon crashed\n")
    assert log.stat().st_mtime >= capture.started
    return source, log


@contextmanager
def _relative_core_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pattern: str = "core",
    uses_pid: bool = True,
    preexisting: str | None = None,
) -> Iterator[CoreEnvironment]:
    proc = tmp_path / "proc"
    (proc / "sys/kernel").mkdir(parents=True)
    (proc / "sys/kernel/core_pattern").write_text(pattern)
    (proc / "sys/kernel/core_uses_pid").write_text(str(int(uses_pid)))
    (proc / "self").mkdir()
    (proc / "self/coredump_filter").write_text("0x33")
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    executable = tmp_path / "checkpoint_server"
    executable.write_bytes(b"exact daemon executable")
    if preexisting is not None:
        old_core = working_dir / preexisting
        old_core.write_bytes(b"old core")
        future = time.time() + 3600
        os.utime(old_core, (future, future))
    monkeypatch.chdir(working_dir)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(logs / "checkpoint_server.%h.%p.log"))
    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", proc)
    monkeypatch.setattr(ckpt_diag, "_CORE_PATTERN_PATH", proc / "sys/kernel/core_pattern")
    monkeypatch.setattr(ckpt_diag, "_CORE_CAPTURE", None)
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", lambda _kind, _value: None)
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setattr(ckpt_diag.threading, "Thread", lambda **_kwargs: pytest.fail("setup starts no background collector"))
    monkeypatch.setattr(ckpt_diag.atexit, "register", lambda _callback: None)
    monkeypatch.setattr(ckpt_diag.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("collection needs no privileged command"))

    ckpt_diag.enable_checkpoint_server_core_dumps(executable, logs)
    capture = ckpt_diag._CORE_CAPTURE
    assert capture is not None
    try:
        yield capture, executable, logs
    finally:
        if ckpt_diag._CORE_CAPTURE is capture:
            os.close(capture.executable_fd)


def _write_relative_core(
    capture: ckpt_diag._CoreCapture,
    logs: Path,
    name: str = "core.123",
    *,
    pid: int = 123,
    log_name: str | None = None,
) -> tuple[Path, Path]:
    source = capture.working_dir / name
    source.write_bytes(b"new kernel core payload")
    log = logs / (log_name or f"checkpoint_server.node-a.example.com.{pid}.log")
    log.write_text("[native_ckpt:svr][CRASH] crashed on SIGSEGV (segfault)\n")
    for path in (source, log):
        os.utime(path, (capture.started + 1, capture.started + 1))
    return source, log


def test_relative_kernel_pattern_collects_from_daemon_working_directory_without_global_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        source, log = _write_relative_core(capture, logs)

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]

        assert Path(f"{log}.core").read_bytes() == b"new kernel core payload"
        assert Path(f"{log}.exe").read_bytes() == b"exact daemon executable"
        assert json.loads(Path(f"{log}.core.log").read_text())["signal"] == signal.SIGSEGV
        assert not source.exists()
        assert (ckpt_diag._PROC_ROOT / "sys/kernel/core_pattern").read_text() == "core"
        assert (ckpt_diag._PROC_ROOT / "sys/kernel/core_uses_pid").read_text() == "1"


def test_relative_kernel_collection_preserves_sigquit_dump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        source, log = _write_relative_core(capture, logs)
        log.write_text("[native_ckpt:svr][SIGNAL] received SIGQUIT (terminating)\n")

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]

        assert not source.exists()
        assert Path(f"{log}.core").read_bytes() == b"new kernel core payload"
        assert Path(f"{log}.exe").read_bytes() == b"exact daemon executable"
        assert json.loads(Path(f"{log}.core.log").read_text())["signal"] == signal.SIGQUIT


@pytest.mark.parametrize(
    ("pattern", "uses_pid", "filename"),
    [
        ("core.%p", True, "core.123"),
        ("core.%p", False, "core.123"),
        ("core.%P", True, "core.456.123"),
        ("core.%%p", True, "core.%p.123"),
    ],
)
def test_kernel_implicit_pid_suffix_obeys_percent_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pattern: str,
    uses_pid: bool,
    filename: str,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path, pattern=pattern, uses_pid=uses_pid) as (capture, _executable, logs):
        _source, log = _write_relative_core(capture, logs, filename)

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]


def test_relative_kernel_collection_is_scoped_to_this_hosts_daemon_pids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        _source, log = _write_relative_core(capture, logs)
        remote_core, _remote_log = _write_relative_core(
            capture,
            logs,
            "core.789",
            pid=789,
            log_name="checkpoint_server.node-b.example.com.789.log",
        )
        unrelated = capture.working_dir / "core.999"
        unrelated.write_bytes(b"another process")

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]

        assert remote_core.is_file()
        assert unrelated.is_file()


@pytest.mark.parametrize(
    ("pattern", "uses_pid", "filename"),
    [("core", True, "core.123"), ("core.%p.%s", False, "core.123.11"), ("core.%%h.%p", False, "core.%h.123")],
)
def test_hostname_free_core_does_not_belong_to_a_normally_exited_local_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pattern: str,
    uses_pid: bool,
    filename: str,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path, pattern=pattern, uses_pid=uses_pid) as (capture, _executable, logs):
        source, log = _write_relative_core(capture, logs, filename)
        log.write_text("[native_ckpt:svr][INFO] Daemon exiting normally\n")
        os.utime(log, (capture.started + 1, capture.started + 1))

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == []

        assert source.read_bytes() == b"new kernel core payload"
        assert not Path(f"{log}.core").exists()


@pytest.mark.parametrize("changed", [False, True])
def test_relative_kernel_collection_rejects_preexisting_core_after_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: bool,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path, preexisting="core.123") as (capture, _executable, logs):
        source = capture.working_dir / "core.123"
        log = logs / "checkpoint_server.node-a.example.com.123.log"
        log.write_text("[native_ckpt:svr][CRASH] crashed on SIGSEGV (segfault)\n")
        if changed:
            source, log = _write_relative_core(capture, logs)

        collected = ckpt_diag.collect_checkpoint_server_core_dumps()

        assert collected == ([Path(f"{log}.core")] if changed else [])
        if not changed:
            assert source.read_bytes() == b"old core"


def test_relative_kernel_collection_keeps_the_startup_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        _source, log = _write_relative_core(capture, logs)
        later = tmp_path / "later"
        later.mkdir()
        monkeypatch.chdir(later)

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]


def test_relative_kernel_collection_waits_for_the_kernel_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        source, log = _write_relative_core(capture, logs)
        process = ckpt_diag._PROC_ROOT / "123"
        process.mkdir()
        status = process / "status"
        status.write_text("State:\tZ (zombie)\nCoreDumping:\t1\n")

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
        assert source.exists()
        assert not Path(f"{log}.core").exists()

        status.unlink()

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]


@pytest.mark.parametrize("style", ["cycle", "custom", "default"])
def test_relative_core_collection_matches_daemon_log_patterns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    style: str,
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        patterns = {
            "cycle": "checkpoint_server.cycle002.%h.%p.log",
            "custom": "custom.%h.%p.log",
            "default": "native_ckpt_server.%h_%p.log",
        }
        pattern = patterns[style]
        monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(logs / pattern))
        log_name = pattern.replace("%h", "node-a.example.com").replace("%p", "123")
        _source, log = _write_relative_core(capture, logs, log_name=log_name)

        assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]


@pytest.mark.parametrize(
    ("pattern", "after_unlink"),
    [
        ("nvrx-core.%h.%p.%P.%t.%s", False),
        ("nvrx-core.%E.%h.%p.%P.%t.%s", False),
        ("nvrx-core.%E.%h.%p.%P.%t.%s", True),
    ],
)
def test_kernel_core_collection_preserves_exact_executable_after_replacement(
    core_environment: CoreEnvironment,
    pattern: str,
    after_unlink: bool,
) -> None:
    capture, executable, logs = core_environment
    capture.core_pattern = str(Path(capture.core_pattern).parent / pattern)
    if after_unlink:
        executable.unlink()
    encoded_executable = str(Path(f"/proc/self/fd/{capture.executable_fd}").readlink()).replace("/", "!")
    comm = f"{encoded_executable}." if "%E" in pattern else ""
    source, log = _write_core(capture, logs, comm=comm, cycle="cycle002.")
    if not after_unlink:
        executable.unlink()
    executable.write_bytes(b"replacement binary")

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]

    assert not source.exists()
    assert Path(f"{log}.core").read_bytes() == b"kernel core payload"
    assert Path(f"{log}.exe").read_bytes() == b"original daemon binary"
    metadata = json.loads(Path(f"{log}.core.log").read_text())
    assert {key: metadata[key] for key in ("signal", "pid", "host_pid")} == {"signal": 11, "pid": 123, "host_pid": 456}
    owner_read_write = 0o600
    assert Path(f"{log}.core").stat().st_mode & 0o777 == owner_read_write
    assert not list(logs.glob("*.partial"))


def test_core_discovery_error_preserves_original_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _relative_core_environment(monkeypatch, tmp_path) as (capture, _executable, logs):
        source, _log = _write_relative_core(capture, logs)
        moved = tmp_path / "moved"
        capture.working_dir.rename(moved)
        capture.working_dir.symlink_to(capture.working_dir, target_is_directory=True)
        monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))

        ckpt_diag.diagnose_checkpoint_server_failure(RuntimeError("original save failure"), logs)

        error = capsys.readouterr().err
        assert "cannot discover daemon cores" in error
        assert "original save failure" in error
        assert (moved / source.name).read_bytes() == b"new kernel core payload"


def test_reinitializing_with_another_executable_keeps_raw_cores(
    core_environment: CoreEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture, executable, logs = core_environment
    source, log = _write_core(capture, logs)
    executable.unlink()
    executable.write_bytes(b"different daemon executable")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", lambda _kind, _value: None)

    ckpt_diag.enable_checkpoint_server_core_dumps(executable, logs)

    assert capture.ambiguous
    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()
    assert not Path(f"{log}.core").exists()
    assert "executable changed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "identity",
    [{"host": "node-b.example.com"}, {"pid": 789}, {"timestamp": 999}, {"comm": "other_process."}],
)
def test_kernel_collection_does_not_take_unrelated_cores(core_environment: CoreEnvironment, identity: dict[str, str | int]) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs, **identity)
    if identity == {"pid": 789}:
        log.unlink()

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()


def test_kernel_collection_requires_owned_core(core_environment: CoreEnvironment, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, _executable, logs = core_environment
    source, _log = _write_core(capture, logs)
    uid = os.getuid()
    monkeypatch.setattr(ckpt_diag.os, "getuid", lambda: uid + 1)

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()


@pytest.mark.parametrize(
    "status",
    [
        "State:\tS (sleeping)\nCoreDumping:\t0\n",
        "State:\tD (disk sleep)\nCoreDumping:\t1\n",
        "State:\tZ (zombie)\nCoreDumping:\t1\n",
    ],
)
def test_kernel_collection_waits_for_process_to_finish(core_environment: CoreEnvironment, status: str) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)
    process = ckpt_diag._PROC_ROOT / "123"
    process.mkdir()
    status_path = process / "status"
    status_path.write_text(status)

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()
    assert not Path(f"{log}.core").exists()

    status_path.unlink()

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.core")]


def test_kernel_collection_copy_failure_preserves_original(
    core_environment: CoreEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)

    def fail_copy(*_args: object) -> None:
        raise OSError("filesystem full")

    monkeypatch.setattr(ckpt_diag.shutil, "copyfileobj", fail_copy)

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()
    assert not Path(f"{log}.core").exists()
    assert not list(logs.glob("*.partial"))


@pytest.mark.parametrize("failure", ["publish", "remove_source"])
def test_kernel_collection_retains_source_on_publication_or_removal_failure(
    core_environment: CoreEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)
    core = Path(f"{log}.core")
    if failure == "publish":
        link = os.link

        def fail_publish(partial: Path, destination: Path) -> None:
            if destination == core:
                raise OSError("cannot publish core")
            link(partial, destination)

        monkeypatch.setattr(ckpt_diag.os, "link", fail_publish)
    else:
        unlink = Path.unlink

        def fail_remove(path: Path, *, missing_ok: bool = False) -> None:
            if path == source:
                raise OSError("cannot remove source")
            unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_remove)

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.read_bytes() == b"kernel core payload"
    assert not list(logs.glob("*.partial"))
    for suffix in (".core", ".exe", ".core.log"):
        assert Path(f"{log}{suffix}").exists() == (failure == "remove_source")
    if failure == "remove_source":
        assert core.read_bytes() == source.read_bytes()
        assert Path(f"{log}.exe").read_bytes() == b"original daemon binary"


def test_kernel_collection_keeps_existing_artifacts(core_environment: CoreEnvironment) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)
    Path(f"{log}.core").write_bytes(b"existing core")
    Path(f"{log}.exe").write_bytes(b"existing executable")

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == [Path(f"{log}.456.1001.core")]

    assert not source.exists()
    assert Path(f"{log}.core").read_bytes() == b"existing core"
    assert Path(f"{log}.exe").read_bytes() == b"existing executable"


def test_kernel_collection_does_not_remove_another_partial(core_environment: CoreEnvironment) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)
    partial = Path(f"{log}.core.partial")
    partial.write_bytes(b"another collector")

    assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()
    assert partial.read_bytes() == b"another collector"


def test_kernel_collection_does_not_compete_with_another_collector(core_environment: CoreEnvironment) -> None:
    capture, _executable, logs = core_environment
    source, _log = _write_core(capture, logs)
    with source.open("rb") as other_collector:
        fcntl.flock(other_collector.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ckpt_diag.collect_checkpoint_server_core_dumps() == []
    assert source.exists()


def test_core_settings_preserve_hard_limit_and_warn_when_unlimited_is_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/coredump_filter").write_text("00000033")
    limits: list[tuple[int, int]] = []

    def set_limit(_kind: int, value: tuple[int, int]) -> None:
        limits.append(value)
        if value[1] == resource.RLIM_INFINITY:
            raise ValueError("not permitted")

    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", proc)
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, 1024))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", set_limit)
    monkeypatch.setenv("LOCAL_RANK", "1")

    ckpt_diag.enable_checkpoint_server_core_dumps("/does/not/exist")

    assert limits == [(resource.RLIM_INFINITY, resource.RLIM_INFINITY), (1024, 1024)] * 2
    assert (proc / "self/coredump_filter").read_text() == "0x1ff"
    assert "full cores are not guaranteed" in capsys.readouterr().err


@pytest.mark.parametrize("kernel_pattern", ["|/usr/lib/systemd/systemd-coredump %P %u %g %s %t", "core.%P"])
def test_unsupported_kernel_handler_does_not_start_a_file_collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kernel_pattern: str,
) -> None:
    pattern = tmp_path / "core_pattern"
    pattern.write_text(kernel_pattern)
    (tmp_path / "sys/kernel").mkdir(parents=True)
    (tmp_path / "sys/kernel/core_uses_pid").write_text("0")
    executable = tmp_path / "checkpoint_server"
    executable.write_bytes(b"daemon executable")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", tmp_path)
    monkeypatch.setattr(ckpt_diag, "_CORE_PATTERN_PATH", pattern)
    monkeypatch.setattr(ckpt_diag, "_CORE_CAPTURE", None)
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", lambda _kind, _value: None)
    monkeypatch.setattr(ckpt_diag.threading, "Thread", lambda **_kwargs: pytest.fail("unsupported routing needs no file collector"))
    monkeypatch.setattr(ckpt_diag.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("setup needs no privileged command"))

    ckpt_diag.enable_checkpoint_server_core_dumps(executable)

    assert ckpt_diag._CORE_CAPTURE is None
    assert pattern.read_text() == kernel_pattern
    assert capsys.readouterr().err


def test_nonzero_local_rank_sets_inherited_settings_without_global_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "self").mkdir()
    core_filter = tmp_path / "self/coredump_filter"
    core_filter.write_text("0x33")
    calls: list[int] = []
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("SLURM_LOCALID", "2")
    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", tmp_path)
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", lambda kind, _value: calls.append(kind))
    monkeypatch.setattr(ckpt_diag.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("setup needs no privileged command"))

    ckpt_diag.enable_checkpoint_server_core_dumps("/does/not/exist")

    assert calls == [resource.RLIMIT_CORE, resource.RLIMIT_FSIZE]
    assert core_filter.read_text() == "0x1ff"


def test_local_rank_zero_enables_kernel_dumps_without_background_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel").mkdir(parents=True)
    (proc / "self/coredump_filter").write_text("0x33")
    pattern = proc / "sys/kernel/core_pattern"
    pattern.write_text("core")
    (proc / "sys/kernel/core_uses_pid").write_text("1")
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "checkpoint_server"
    executable.write_bytes(b"exact daemon executable")
    logs = tmp_path / "logs"
    logs.mkdir()
    callbacks: list[Callable[[], None]] = []
    limits: list[tuple[int, int]] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ckpt_diag, "_CORE_CAPTURE", None)
    monkeypatch.setattr(ckpt_diag, "_PROC_ROOT", proc)
    monkeypatch.setattr(ckpt_diag, "_CORE_PATTERN_PATH", pattern)
    monkeypatch.setattr(ckpt_diag.resource, "getrlimit", lambda _kind: (0, resource.RLIM_INFINITY))
    monkeypatch.setattr(ckpt_diag.resource, "setrlimit", lambda _kind, value: limits.append(value))
    monkeypatch.setattr(ckpt_diag.threading, "Thread", lambda **_kwargs: pytest.fail("setup starts no background collector"))
    monkeypatch.setattr(ckpt_diag.atexit, "register", callbacks.append)
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(logs / "checkpoint_server.%h.%p.log"))

    ckpt_diag.enable_checkpoint_server_core_dumps(executable, logs)
    capture = ckpt_diag._CORE_CAPTURE
    assert capture is not None
    try:
        descriptor = capture.executable_fd
        source, log = _write_relative_core(capture, logs)
        ckpt_diag.enable_checkpoint_server_core_dumps(executable, tmp_path / "cycle002")

        assert pattern.read_text() == "core"
        assert (proc / "sys/kernel/core_uses_pid").read_text() == "1"
        assert (proc / "self/coredump_filter").read_text() == "0x1ff"
        assert callbacks == [ckpt_diag._finish_core_collection]
        assert capture.executable_fd == descriptor
        assert os.read(descriptor, 4096) == b"exact daemon executable"
        assert {tmp_path / "logs", tmp_path / "cycle002"} <= capture.log_dirs
        assert limits == [(resource.RLIM_INFINITY, resource.RLIM_INFINITY)] * 4
        assert source.exists()
        assert not Path(f"{log}.core").exists()
    finally:
        os.close(capture.executable_fd)


def test_exit_collection_copies_core_and_closes_pinned_executable(core_environment: CoreEnvironment) -> None:
    capture, _executable, logs = core_environment
    source, log = _write_core(capture, logs)
    descriptor = capture.executable_fd

    ckpt_diag._finish_core_collection()

    assert ckpt_diag._CORE_CAPTURE is None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptor)
    assert not source.exists()
    assert Path(f"{log}.core").read_bytes() == b"kernel core payload"
    assert Path(f"{log}.exe").read_bytes() == b"original daemon binary"
    assert json.loads(Path(f"{log}.core.log").read_text())["signal"] == signal.SIGSEGV
    ckpt_diag._finish_core_collection()
    assert ckpt_diag._CORE_CAPTURE is None


@pytest.mark.parametrize(("timeout", "expected"), [("60", 60), ("0", 3600), ("garbage", 3600), ("2147484", 3600)])
def test_core_timeout_is_bounded(
    core_environment: CoreEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    timeout: str,
    expected: int,
) -> None:
    capture, _executable, logs = core_environment
    _write_core(capture, logs)
    process = ckpt_diag._PROC_ROOT / "123"
    process.mkdir()
    (process / "status").write_text("CoreDumping:\t1\n")
    ticks = iter((0.0, float(expected - 1), float(expected)))
    sleeps: list[float] = []
    monkeypatch.setenv("NATIVE_CKPT_SERVER_CORE_DUMP_TIMEOUT_SECONDS", timeout)
    monkeypatch.setattr(ckpt_diag.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ckpt_diag.time, "sleep", sleeps.append)

    assert ckpt_diag.collect_checkpoint_server_core_dumps(wait=True) == []
    assert sleeps == [0.25]


def test_server_log_diagnostic_is_scoped_to_local_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "12")
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))
    (tmp_path / "checkpoint_server.node-a.100.log").write_text("local server failure\n")
    (tmp_path / "checkpoint_server.node-b.200.log").write_text("remote server noise\n")

    ckpt_diag.diagnose_checkpoint_server_failure(RuntimeError("training failed"), tmp_path)

    output = capsys.readouterr().out
    assert "checkpoint_server.node-a.100.log" in output
    assert "local server failure" in output
    assert "node-b" not in output
    assert "remote server noise" not in output


def test_connection_failure_probes_logs_and_process_even_when_port_is_dead(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []
    (tmp_path / "checkpoint_server.node-a.42.log").write_text("daemon crash log\n")
    monkeypatch.setenv("RANK", "112")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))
    monkeypatch.setattr(ckpt_diag.shutil, "which", lambda name: f"/usr/bin/{name}")

    def run(args: list[str], **_kwargs: object) -> ckpt_diag.subprocess.CompletedProcess[str]:
        calls.append(args)
        output = "42" if args[0] == "/usr/bin/pgrep" else "daemon thread backtrace"
        return ckpt_diag.subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(ckpt_diag.subprocess, "run", run)

    ckpt_diag.diagnose_checkpoint_server_failure(
        RuntimeError("Failed to start or connect to checkpoint service"),
        tmp_path,
    )

    assert calls == [
        ["/usr/bin/pgrep", "-f", "[c]heckpoint_server"],
        ["/usr/bin/gdb", "-p", "42", "-batch", "-ex", "thread apply all bt"],
    ]
    output = capsys.readouterr().out
    assert output.index("daemon crash log") < output.index("daemon thread backtrace")
    assert "[native_ckpt:diag][rank112]" in output


def test_stack_diagnostic_reports_when_no_server_process_remains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))
    monkeypatch.setattr(ckpt_diag.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ckpt_diag.subprocess, "run", lambda *_args, **_kwargs: ckpt_diag.subprocess.CompletedProcess([], 1, stdout=""))

    ckpt_diag.diagnose_checkpoint_server_failure(RuntimeError("Failed to start or connect to checkpoint service"), tmp_path)

    assert "no checkpoint_server process found for stack capture" in capsys.readouterr().out
