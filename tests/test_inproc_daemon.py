# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101
"""Independent training-process coverage for native daemon ownership."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_REPO = Path(__file__).resolve().parent.parent
_WORKER = Path(__file__).with_name("inproc_daemon_worker.py")
_BASE_URL = "http://127.0.0.1:4100/api"
_CLEAN_EXIT_LIMIT = 4.5
_STARTUP_GRACE_MIN = 4
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1" or sys.platform != "linux",
        reason="requires native checkpoint-server IPC and Linux process inspection",
    ),
]


def _ping() -> dict | None:
    try:
        with urllib.request.urlopen(f"{_BASE_URL}/ping", timeout=0.5) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _wait_for(predicate: Callable[[], bool], timeout: float = 4) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f"condition did not become true within {timeout} seconds")
        time.sleep(0.05)


def _build_dir() -> Path:
    configured = os.environ.get("NATIVE_CKPT_INPROC_TEST_BUILD_DIR") or os.environ.get("MEGATRON_NATIVE_BUILD_DIR")
    candidates = [Path(configured)] if configured else [_REPO, _REPO / "build", _REPO / "build-test", _REPO / "build-inproc"]
    for candidate in candidates:
        if any(candidate.glob("checkpoint*.so")) and (candidate / "checkpoint_server").is_file():
            return candidate.resolve()
    pytest.skip("native binaries are unavailable; set NATIVE_CKPT_INPROC_TEST_BUILD_DIR")


class _TrainingProcess:
    def __init__(  # noqa: PLR0913
        self,
        build: Path,
        directory: Path,
        rank: int,
        world: int,
        mode: str | None,
        *,
        ping_only: bool = False,
        telemetry: bool = False,
    ):
        self.log = directory / f"daemon.{rank}.log"
        self.telemetry_file = directory / "telemetry.jsonl"
        self.stderr_path = directory / f"training.{rank}.stderr"
        self.stderr_file = self.stderr_path.open("w")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("NATIVE_CKPT_")}
        environment.pop("CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH", None)
        environment.update(
            PYTHONPATH=str(build),
            LOCAL_RANK=str(rank),
            RANK=str(rank),
            NATIVE_CKPT_SERVER_LOG=str(self.log),
            NATIVE_CKPT_SERVER_HANDLER_THREADS="4",
            NATIVE_CKPT_SERVER_TCP_ACCEPTORS="1",
            NATIVE_CKPT_SHM_RING_BUFFER_SIZE="64KiB",
            NATIVE_CKPT_SHM_RING_BUFFER_SLOTS="4",
            NATIVE_CKPT_SHM_RING_BUFFER_NUM_QUEUES="1",
        )
        if mode is not None:
            environment["NATIVE_CKPT_SERVER_INPROC"] = mode
        if telemetry:
            environment.update(
                NATIVE_CKPT_OTEL_FILE=str(self.telemetry_file),
                NATIVE_CKPT_OTEL_EXPORT="daemon",
                NATIVE_CKPT_OTEL_LOGS="0",
            )
        arguments = [sys.executable, "-u", str(_WORKER), "--rank", str(rank), "--world", str(world)]
        if ping_only:
            arguments.append("--ping-only")
        self.process = subprocess.Popen(
            arguments,
            cwd=directory,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
        )
        self.events: queue.Queue[dict] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            self.events.put(event)

    def event(self, expected: str, timeout: float = 15) -> dict:
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"missing {expected} event; process={self.process.poll()}; stderr={self.stderr_path.read_text()}")
        assert event["event"] == expected, event
        return event

    def send(self, action: str, **fields: object) -> None:
        self.process.stdin.write(json.dumps({"action": action, **fields}) + "\n")
        self.process.stdin.flush()

    def request(self, action: str, **fields: object) -> dict:
        self.send(action, **fields)
        return self.event("exiting" if action in {"exit", "save_exit"} else action)

    def wait(self, timeout: float = 4) -> None:
        try:
            status = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"training process failed to exit; stderr={self.stderr_path.read_text()}")
        assert status == 0, self.stderr_path.read_text()

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process.stdin.close()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        self.stderr_file.close()


@pytest.fixture
def training(tmp_path: Path) -> Iterator[Callable[..., _TrainingProcess]]:
    build = _build_dir()
    if _ping() is not None:
        pytest.skip("port 4100 already has a checkpoint service")
    processes = []

    def start(
        rank: int = 0,
        world: int = 1,
        mode: str | None = "1",
        *,
        ping_only: bool = False,
        telemetry: bool = False,
    ) -> _TrainingProcess:
        process = _TrainingProcess(build, tmp_path, rank, world, mode, ping_only=ping_only, telemetry=telemetry)
        processes.append(process)
        return process

    try:
        yield start
    finally:
        if _ping() is not None:
            try:
                with urllib.request.urlopen(f"{_BASE_URL}/stop", timeout=1):
                    pass
            except OSError:
                pass
        for process in reversed(processes):
            process.cleanup()
        _wait_for(lambda: _ping() is None)


@pytest.mark.parametrize("mode", ["", "0"])
@pytest.mark.parametrize("release", ["close", "destroy"])
def test_inproc_presence_owns_daemon_and_releases_without_grace(training: Callable, mode: str, release: str) -> None:
    owner = training(mode=mode)
    ready = owner.event("ready")
    assert ready["children"] == []
    identity = _ping()
    assert identity["mode"] == "inproc"
    assert identity["pid"] == owner.process.pid
    owner.request("stderr")
    owner.request("signal")
    started = time.monotonic()
    owner.request(release)
    if release == "close":
        owner.request("close")
    owner.request("exit")
    owner.wait()
    assert time.monotonic() - started < _CLEAN_EXIT_LIMIT
    assert _ping() is None
    assert "TRAINING_STDERR_IS_USABLE" in owner.stderr_path.read_text()
    daemon_log = owner.log.read_text()
    assert "[native_ckpt:svr][INFO] In-process daemon started" in daemon_log
    assert "TRAINING_STDERR_IS_USABLE" not in daemon_log


def test_nonowner_waits_for_owner_and_keeps_it_alive(training: Callable) -> None:
    peer = training(rank=1, world=2)
    with pytest.raises(queue.Empty):
        peer.events.get(timeout=0.2)
    owner = training(world=2)
    assert owner.event("ready")["children"] == []
    assert peer.event("ready")["children"] == []
    identity = _ping()
    assert identity["mode"] == "inproc"
    assert identity["pid"] == owner.process.pid
    owner.request("close")
    owner.request("exit")
    time.sleep(0.2)
    assert owner.process.poll() is None
    assert _ping()["pid"] == owner.process.pid
    peer.request("close")
    peer.request("exit")
    peer.wait()
    owner.wait()


def test_dead_local_process_releases_its_reference(training: Callable) -> None:
    owner = training(world=2)
    owner.event("ready")
    peer = training(rank=1, world=2)
    peer.event("ready")
    owner.request("close")
    owner.request("exit")
    assert owner.process.poll() is None
    peer.process.kill()
    peer.process.wait(timeout=5)
    owner.wait()
    assert _ping() is None


def test_two_roots_in_one_process_keep_separate_references(training: Callable) -> None:
    owner = training()
    owner.event("ready")
    owner.request("new_root")
    owner.request("close_first")
    time.sleep(0.2)
    identity = _ping()
    assert identity["mode"] == "inproc"
    assert identity["pid"] == owner.process.pid
    owner.request("close")
    owner.request("exit")
    owner.wait()
    assert _ping() is None


def test_unused_inproc_daemon_has_startup_grace(training: Callable) -> None:
    owner = training(ping_only=True)
    owner.event("ready")
    started = time.monotonic()
    time.sleep(1)
    assert _ping()["pid"] == owner.process.pid
    _wait_for(lambda: _ping() is None, timeout=7)
    assert time.monotonic() - started >= _STARTUP_GRACE_MIN
    owner.request("exit")
    owner.wait()


def test_interpreter_exit_finishes_queued_checkpoint(training: Callable, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    owner = training()
    owner.event("ready")
    checkpoint_path = tmp_path / "checkpoint"
    owner.request("save_exit", path=str(checkpoint_path))
    owner.wait(timeout=30)
    assert (checkpoint_path / ".metadata").is_file()
    loaded = {"weight": torch.zeros(8, dtype=torch.float32)}
    dcp.load(loaded, checkpoint_id=checkpoint_path)
    torch.testing.assert_close(loaded["weight"], torch.arange(8, dtype=torch.float32))
    assert _ping() is None


def test_same_process_saves_after_daemon_restart(training: Callable, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    owner = training()
    owner.event("ready")
    first_identity = _ping()
    first_path = tmp_path / "first"
    owner.request("save", path=str(first_path))
    owner.request("close")
    _wait_for(lambda: _ping() is None)
    owner.request("new_root")
    second_identity = _ping()
    assert second_identity["mode"] == "inproc"
    assert second_identity["pid"] == first_identity["pid"] == owner.process.pid
    assert second_identity["generation"] != first_identity["generation"]
    second_path = tmp_path / "second"
    owner.request("save_exit", path=str(second_path))
    owner.wait(timeout=30)
    for path in (first_path, second_path):
        loaded = {"weight": torch.zeros(8, dtype=torch.float32)}
        dcp.load(loaded, checkpoint_id=path)
        torch.testing.assert_close(loaded["weight"], torch.arange(8, dtype=torch.float32))
    assert _ping() is None


@pytest.mark.parametrize("mode", ["1", None], ids=["inproc", "process"])
def test_root_destruction_after_explicit_spindown_can_restart(training: Callable, tmp_path: Path, mode: str | None) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    owner = training(mode=mode)
    owner.event("ready")
    first_path = tmp_path / "first"
    owner.request("save", path=str(first_path))
    owner.request("spindown_destroy")
    _wait_for(lambda: _ping() is None)
    owner.request("new_root")
    second_path = tmp_path / "second"
    owner.request("save_exit", path=str(second_path))
    owner.wait(timeout=30)
    _wait_for(lambda: _ping() is None)
    for path in (first_path, second_path):
        loaded = {"weight": torch.zeros(8, dtype=torch.float32)}
        dcp.load(loaded, checkpoint_id=path)
        torch.testing.assert_close(loaded["weight"], torch.arange(8, dtype=torch.float32))


@pytest.mark.parametrize("mode", ["1", None], ids=["inproc", "process"])
def test_interpreter_exit_flushes_checkpoint_telemetry(training: Callable, tmp_path: Path, mode: str | None) -> None:
    pytest.importorskip("torch")
    owner = training(mode=mode, telemetry=True)
    owner.event("ready")
    identity = _ping()
    owner.request("save_exit", path=str(tmp_path / "checkpoint"))
    owner.wait(timeout=30)
    _wait_for(lambda: _ping() is None)
    spans = []
    for line in owner.telemetry_file.read_text().splitlines():
        for resource in json.loads(line).get("resourceSpans", []):
            attributes = {attribute["key"]: attribute["value"] for attribute in resource["resource"]["attributes"]}
            pid = int(attributes["process.pid"]["intValue"])
            for scope in resource.get("scopeSpans", []):
                spans.extend((pid, span) for span in scope.get("spans", []))
    required = {"ckpt.save", "ckpt.svr.durable"}
    completed = {span["name"] for pid, span in spans if pid == identity["pid"] and int(span["endTimeUnixNano"]) > 0}
    assert required <= completed
    for pid, span in spans:
        if span["name"] in required:
            assert pid == identity["pid"]
            assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])


def test_inproc_rejects_existing_separate_daemon(training: Callable) -> None:
    separate = training(mode=None)
    separate.event("ready")
    identity = _ping()
    assert identity["mode"] == "process"
    assert identity["pid"] != separate.process.pid
    inproc = training()
    status = inproc.process.wait(timeout=10)
    assert status != 0
    message = inproc.stderr_path.read_text().lower()
    assert "in-process" in message or "inproc" in message
    assert _ping() == identity
    separate.request("close")
    separate.request("exit")
    separate.wait()
    _wait_for(lambda: _ping() is None)


def test_inproc_reports_log_open_failure(training: Callable, tmp_path: Path) -> None:
    (tmp_path / "daemon.0.log").mkdir()
    owner = training()
    assert owner.process.wait(timeout=10) != 0
    message = owner.stderr_path.read_text()
    assert "cannot open" in message
    assert str(owner.log) in message
    assert _ping() is None
