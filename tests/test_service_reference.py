# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101
"""Exercise daemon ownership through training-process lifetimes."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
        reason="requires checkpoint-server IPC support",
    ),
]

_COMMON = """
import gc
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from nvrx_checkpointing import checkpoint


def request(path, body=b''):
    with urlopen(Request('http://127.0.0.1:4100' + path, data=body), timeout=2) as response:
        return response.read()


def wait_stopped():
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            request('/api/reference/ping')
        except (URLError, TimeoutError, ConnectionError):
            return
        time.sleep(0.05)
    raise AssertionError('checkpoint service did not stop')


def create():
    return checkpoint.model_state.create_root(9100, 0, 1, ['localhost'], 1)

"""


def _request(path: str) -> bytes:
    with urlopen(Request("http://127.0.0.1:4100" + path), timeout=2) as response:
        return response.read()


def _wait_stopped() -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            _request("/api/reference/ping")
        except (URLError, TimeoutError, ConnectionError):
            return
        time.sleep(0.05)
    pytest.fail("checkpoint service did not stop")


@pytest.fixture(params=["inproc", "outproc"])
def daemon_env(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    for name in ("NATIVE_CKPT_SERVER_INPROC", "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH", "NATIVE_CKPT_OTEL_FILE"):
        env.pop(name, None)
    if request.param == "inproc":
        env["NATIVE_CKPT_SERVER_INPROC"] = ""
    env.update(
        NATIVE_CKPT_SERVER_LOG=str(tmp_path / "daemon.%p.log"),
        NATIVE_CKPT_SHM_RING_BUFFER_SIZE="8MiB",
        NATIVE_CKPT_SHM_RING_BUFFER_NUM_QUEUES="1",
        NATIVE_CKPT_SHM_RING_BUFFER_SLOTS="2",
        NATIVE_CKPT_SERVER_TCP_ACCEPTORS="1",
        NATIVE_CKPT_SERVER_HANDLER_THREADS="4",
    )
    try:
        yield env
    finally:
        with contextlib.suppress(URLError, TimeoutError, ConnectionError):
            _request("/api/stop")
        _wait_stopped()


def _run(env: dict[str, str], script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COMMON + textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("release", ["close", "destructor"])
def test_clean_release_has_no_shutdown_grace(daemon_env: dict[str, str], release: str) -> None:
    _run(
        daemon_env,
        f"""
        root = create()
        started = time.monotonic()
        if {release!r} == 'close':
            root.close()
            root.close()
            try:
                root.register_state_dict({{'value': 1}})
            except RuntimeError:
                pass
            else:
                raise AssertionError('closed client accepted work')
        else:
            del root
            gc.collect()
        checkpoint._shutdown()
        wait_stopped()
        assert time.monotonic() - started < 4, 'clean shutdown waited for startup grace'
        """,
    )


def test_global_root_is_closed_at_interpreter_exit(daemon_env: dict[str, str]) -> None:
    _run(daemon_env, "ROOT = create()")
    _wait_stopped()


def test_close_drains_async_cpu_save(daemon_env: dict[str, str], tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    _run(
        daemon_env,
        f"""
        import torch
        root = create()
        root.register_state_dict({{'weight': torch.arange(1_048_576, dtype=torch.float32)}})
        root.finalize_registry()
        root.save_async(1, {str(destination)!r})
        root.close()
        checkpoint._shutdown()
        wait_stopped()
        """,
    )
    import torch  # noqa: PLC0415
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    loaded = {"weight": torch.zeros(1_048_576)}
    dcp.load(loaded, checkpoint_id=str(destination))
    torch.testing.assert_close(loaded["weight"], torch.arange(1_048_576, dtype=torch.float32))


def test_unavailable_telemetry_collector_preserves_daemon(daemon_env: dict[str, str], tmp_path: Path) -> None:
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        daemon_env.update(
            NATIVE_CKPT_OTEL_FILE=str(tmp_path / "telemetry.jsonl"),
            NATIVE_CKPT_OTEL_COLLECTOR=f"127.0.0.1:{unavailable.getsockname()[1]}",
            NATIVE_CKPT_OTEL_EXPORT="daemon",
            NATIVE_CKPT_OTEL_LOGS="1",
        )
        _run(
            daemon_env,
            """
            root = create()
            try:
                request('/api/telemetry', b'{"resourceSpans":[]}')
            except (URLError, ConnectionError):
                pass
            time.sleep(1.2)
            root.close()
            checkpoint._shutdown()
            wait_stopped()
            """,
        )
    assert any("Telemetry export failed" in log.read_text() for log in tmp_path.glob("daemon.*.log"))


def test_immediate_reacquire_saves_after_service_restart(daemon_env: dict[str, str], tmp_path: Path) -> None:
    _run(
        daemon_env,
        f"""
        import torch
        from pathlib import Path
        for index in range(3):
            first = create()
            first.close()
            root = create()
            root.register_state_dict({{'weight': torch.arange(16, dtype=torch.float32)}})
            root.finalize_registry()
            destination = Path({str(tmp_path)!r}) / str(index)
            root.save_async(index, destination)
            root.close()
            checkpoint._shutdown()
            wait_stopped()
            assert (destination / '.metadata').is_file()
        """,
    )


def test_dead_training_process_releases_pid_reference(daemon_env: dict[str, str]) -> None:
    if "NATIVE_CKPT_SERVER_INPROC" in daemon_env:
        pytest.skip("the embedded daemon exits with its owner")
    _run(daemon_env, "root = create(); os._exit(0)")
    _wait_stopped()


def test_outproc_retries_election_while_old_service_closes(daemon_env: dict[str, str]) -> None:
    if "NATIVE_CKPT_SERVER_INPROC" in daemon_env:
        pytest.skip("standalone launchers use the election handshake")
    _run(
        daemon_env,
        """
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class ClosingService(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_error(503, 'Checkpoint service is shutting down')

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(('127.0.0.1', 4100), ClosingService) as old:
            worker = threading.Thread(target=old.serve_forever, daemon=True)
            worker.start()
            def finish():
                old.shutdown()
                old.server_close()
            timer = threading.Timer(1, finish)
            timer.start()
            try:
                root = create()
            finally:
                timer.join()
        root.close()
        checkpoint._shutdown()
        wait_stopped()
        """,
    )


@pytest.mark.parametrize("remote_exit", ["release", "dead"])
def test_coordinator_waits_for_remote_reference(daemon_env: dict[str, str], remote_exit: str) -> None:
    _run(
        daemon_env,
        f"""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Peer(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header('Content-Length', '3')
                self.end_headers()
                self.wfile.write(b'123')

            def log_message(self, *args):
                pass

        peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        thread = threading.Thread(target=peer.serve_forever, daemon=True)
        thread.start()
        try:
            root = create()
            request('/api/reference/remote/9100/123/remote-test',
                    f'127.0.0.1:{{peer.server_port}}'.encode())
            root.close()
            time.sleep(1.5)
            assert request('/api/reference/ping'), 'coordinator exited while remote was alive'
            if {remote_exit!r} == 'release':
                request('/api/reference/release/remote-test')
            else:
                peer.shutdown()
                peer.server_close()
            checkpoint._shutdown()
            wait_stopped()
        finally:
            peer.shutdown()
            peer.server_close()
            thread.join()
        """,
    )
