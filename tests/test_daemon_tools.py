# ruff: noqa: ANN001, ANN202, INP001, S101
from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "ci" / "scripts"


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

    assert daemon_tools.CHECKPOINT_SERVER.base_url == "http://127.0.0.1:41000/api"


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
