# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN001, ANN202, INP001, PLR2004, S101, SLF001
from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "ci" / "scripts"


def _load_daemon_tools():
    spec = importlib.util.spec_from_file_location("daemon_tools_test", SCRIPTS_DIR / "daemon_tools.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_server_uses_fixed_native_service_port() -> None:
    daemon_tools = _load_daemon_tools()

    assert daemon_tools.CHECKPOINT_SERVER.base_url == "http://127.0.0.1:4100/api"


def test_require_stopped_removes_configured_stale_files(monkeypatch, tmp_path: Path) -> None:
    daemon_tools = _load_daemon_tools()
    stale = tmp_path / "request.1"
    stale.touch()
    monkeypatch.setattr(
        daemon_tools.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: False)
    daemon = daemon_tools.LocalDaemon("worker", "[w]orker", "http://127.0.0.1:1", (str(tmp_path / "request.*"),))

    daemon.require_stopped()

    assert not stale.exists()


def test_stop_requests_shutdown_and_waits_for_exit(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.LocalDaemon("worker", "[w]orker", "http://127.0.0.1:1")
    states = iter((["42"], ["42"], [], []))
    requests = []
    cleaned = []
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: next(states))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda _self, endpoint, timeout: requests.append((endpoint, timeout)))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: cleaned.append(True))
    monkeypatch.setattr(daemon_tools.time, "sleep", lambda _seconds: None)

    daemon.stop(timeout=5)

    assert requests == [("spindown", 5)]
    assert cleaned == [True]


def test_stop_cleans_state_when_process_already_exited(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.LocalDaemon("worker", "[w]orker", "http://127.0.0.1:1")
    cleaned = []
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: [])
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: cleaned.append(True))

    daemon.stop()

    assert cleaned == [True]


def test_terminate_escalates_to_sigkill(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.LocalDaemon("worker", "[w]orker", "http://127.0.0.1:1")
    states = iter((["42"], ["42"], ["42"], ["42"], [], [], []))
    signals = []
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: next(states))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda *_args: None)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: False)
    monkeypatch.setattr(daemon_tools.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(daemon_tools.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(daemon_tools.time, "monotonic", lambda: 0)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)

    daemon.terminate(timeout=0)

    assert signals == [(42, signal.SIGTERM), (42, signal.SIGKILL)]


def test_terminate_stops_daemon_owned_by_another_user(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.LocalDaemon("worker", "[w]orker", "http://127.0.0.1:1")
    reachable = iter((True, False, False))
    requests = []
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: [])
    monkeypatch.setattr(
        daemon_tools.LocalDaemon,
        "request",
        lambda _self, endpoint, timeout: requests.append((endpoint, timeout)),
    )
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: next(reachable))
    monkeypatch.setattr(daemon_tools.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)

    daemon.terminate(timeout=5)

    assert requests == [("stop", 5)]


def test_terminate_waits_for_existing_kernel_core_before_request(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.CHECKPOINT_SERVER
    active = ["42"]
    events = []

    def wait(pid: int, timeout: float) -> bool:
        events.append(("core", pid, timeout))
        active.clear()
        return True

    monkeypatch.setenv("NATIVE_CKPT_SERVER_CORE_DUMP_TIMEOUT_SECONDS", "1800")
    monkeypatch.setattr(daemon_tools, "_wait_for_core_dump", wait)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: active.copy())
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda *_args: events.append(("request",)))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: False)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)
    monkeypatch.setattr(daemon_tools.os, "kill", lambda *_args: pytest.fail("completed core process receives no signal"))

    daemon.terminate(timeout=0)

    assert events == [("core", 42, 1800), ("request",)]


def test_terminate_waits_for_kernel_core_started_during_shutdown(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.CHECKPOINT_SERVER
    active = ["42"]
    signals = []
    core_started = False
    core_waited = False

    def kill(pid: int, sig: signal.Signals) -> None:
        nonlocal core_started
        signals.append((pid, sig))
        core_started = True

    def wait(_pid: int, _timeout: float) -> bool:
        nonlocal core_waited
        if not core_started:
            return False
        core_waited = True
        active.clear()
        return True

    monkeypatch.setattr(daemon_tools, "_wait_for_core_dump", wait)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: active.copy())
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda *_args: None)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: False)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)
    monkeypatch.setattr(daemon_tools.os, "kill", kill)

    daemon.terminate(timeout=0)

    assert core_waited
    assert signals == [(42, signal.SIGTERM)]


def test_kernel_core_timeout_prevents_shutdown_signals(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.CHECKPOINT_SERVER

    def wait(_pid: int, _timeout: float) -> bool:
        raise TimeoutError("kernel core is still being written")

    monkeypatch.setattr(daemon_tools, "_wait_for_core_dump", wait)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: ["42"])
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda *_args: pytest.fail("active core receives no RPC"))
    monkeypatch.setattr(daemon_tools.os, "kill", lambda *_args: pytest.fail("active core receives no signal"))

    with pytest.raises(TimeoutError, match="core is still being written"):
        daemon.terminate()


def test_stop_waits_for_kernel_core_before_shutdown_rpc(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.CHECKPOINT_SERVER
    active = ["42"]
    events = []

    def wait(_pid: int, _timeout: float) -> bool:
        events.append("core")
        active.clear()
        return True

    monkeypatch.setattr(daemon_tools, "_wait_for_core_dump", wait)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", lambda _self: active.copy())
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", lambda *_args: events.append("request"))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)

    daemon.stop()

    assert events == ["core", "request"]


def test_kernel_core_wait_preserves_normal_shutdown_budget(monkeypatch) -> None:
    daemon_tools = _load_daemon_tools()
    daemon = daemon_tools.CHECKPOINT_SERVER
    active = ["42", "43"]
    clock = 0
    requested = False
    shutdown_started = False
    signals = []
    reachable = iter((True, False, False))

    def pids(_self):
        if shutdown_started and clock >= 103:
            active.clear()
        return active.copy()

    def request(*_args: object) -> None:
        nonlocal requested
        requested = True

    def wait(pid: int, _timeout: float) -> bool:
        nonlocal clock
        if pid != 42 or not requested:
            return False
        active.remove("42")
        clock = 100
        return True

    def kill(pid: int, sig: signal.Signals) -> None:
        nonlocal shutdown_started
        signals.append((pid, sig))
        shutdown_started = True

    def sleep(_seconds: float) -> None:
        nonlocal clock
        clock += 1

    monkeypatch.setattr(daemon_tools, "_wait_for_core_dump", wait)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "pids", pids)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "request", request)
    monkeypatch.setattr(daemon_tools.LocalDaemon, "reachable", lambda _self: next(reachable))
    monkeypatch.setattr(daemon_tools.LocalDaemon, "require_stopped", lambda _self: None)
    monkeypatch.setattr(daemon_tools.os, "kill", kill)
    monkeypatch.setattr(daemon_tools.time, "monotonic", lambda: clock)
    monkeypatch.setattr(daemon_tools.time, "sleep", sleep)

    daemon.terminate(timeout=5)

    assert signals == [(43, signal.SIGTERM)]
    assert clock == 103


def test_kernel_dump_wait_returns_after_writer_finishes(monkeypatch, tmp_path: Path) -> None:
    daemon_tools = _load_daemon_tools()
    process = tmp_path / "123"
    process.mkdir()
    status = process / "status"
    status.write_text("State:\tD (disk sleep)\nCoreDumping:\t1\n")
    monkeypatch.setattr(daemon_tools, "_PROC_ROOT", tmp_path)
    monkeypatch.setattr(daemon_tools.time, "sleep", lambda _seconds: status.unlink())

    assert daemon_tools._wait_for_core_dump(123, timeout=10)
    assert not daemon_tools._wait_for_core_dump(123, timeout=10)


def test_kernel_dump_wait_reports_active_writer_timeout(monkeypatch, tmp_path: Path) -> None:
    daemon_tools = _load_daemon_tools()
    process = tmp_path / "123"
    process.mkdir()
    (process / "status").write_text("CoreDumping:\t1\n")
    monkeypatch.setattr(daemon_tools, "_PROC_ROOT", tmp_path)

    with pytest.raises(TimeoutError, match="remains active"):
        daemon_tools._wait_for_core_dump(123, timeout=0)


@pytest.mark.parametrize(("value", "expected"), [("60", 60), ("0", 3600), ("garbage", 3600), ("2147484", 3600)])
def test_kernel_dump_timeout_uses_validated_range(monkeypatch, value: str, expected: int) -> None:
    daemon_tools = _load_daemon_tools()
    monkeypatch.setenv("NATIVE_CKPT_SERVER_CORE_DUMP_TIMEOUT_SECONDS", value)

    assert daemon_tools._core_dump_timeout() == expected
