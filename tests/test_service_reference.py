# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101
"""Exercise daemon ownership through training-process lifetimes."""

from __future__ import annotations

import contextlib
import os
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
import sys
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
    root = checkpoint.model_state.create_root(9100, 0, 1, ['localhost'], 1)
    root.register_state_dict({'value': 1})
    root.finalize_registry()
    root.wait_for_async_completion()
    return root

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
    for name in (
        "NATIVE_CKPT_SERVER_DISABLE_INPROC",
        "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH",
        "NATIVE_CKPT_OTEL_FILE",
        "NATIVE_CKPT_OTEL_COLLECTOR",
    ):
        env.pop(name, None)
    if request.param == "outproc":
        env["NATIVE_CKPT_SERVER_DISABLE_INPROC"] = "1"
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


def test_clean_release_has_no_shutdown_grace(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        root = create()
        started = time.monotonic()
        del root
        gc.collect()
        wait_stopped()
        assert time.monotonic() - started < 4, 'clean shutdown waited for startup grace'
        """,
    )


def test_global_root_is_closed_at_interpreter_exit(daemon_env: dict[str, str]) -> None:
    _run(daemon_env, "ROOT = create()")
    _wait_stopped()


def test_destructor_drains_async_cpu_save(daemon_env: dict[str, str], tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    _run(
        daemon_env,
        f"""
        import torch
        root = checkpoint.model_state.create_root(9100, 0, 1, ['localhost'], 1)
        root.register_state_dict({{'weight': torch.arange(1_048_576, dtype=torch.float32)}})
        root.finalize_registry()
        root.save_async(1, {str(destination)!r})
        del root
        gc.collect()
        wait_stopped()
        """,
    )
    import torch  # noqa: PLC0415
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    loaded = {"weight": torch.zeros(1_048_576)}
    dcp.load(loaded, checkpoint_id=str(destination))
    torch.testing.assert_close(loaded["weight"], torch.arange(1_048_576, dtype=torch.float32))


def test_telemetry_is_flushed_before_shutdown(daemon_env: dict[str, str], tmp_path: Path) -> None:
    destination = tmp_path / "telemetry.jsonl"
    daemon_env.update(NATIVE_CKPT_OTEL_FILE=str(destination), NATIVE_CKPT_OTEL_EXPORT="daemon")
    _run(
        daemon_env,
        """
        root = create()
        request('/api/telemetry', b'{"resourceSpans":[],"test":"shutdown"}')
        del root
        gc.collect()
        wait_stopped()
        """,
    )
    assert '"test":"shutdown"' in destination.read_text()


def test_immediate_reacquire_saves_after_service_restart(daemon_env: dict[str, str], tmp_path: Path) -> None:
    _run(
        daemon_env,
        f"""
        import torch
        from pathlib import Path
        for index in range(3):
            first = create()
            del first
            gc.collect()
            root = checkpoint.model_state.create_root(9100, 0, 1, ['localhost'], 1)
            root.register_state_dict({{'weight': torch.arange(16, dtype=torch.float32)}})
            root.finalize_registry()
            destination = Path({str(tmp_path)!r}) / str(index)
            root.save_async(index, destination)
            del root
            gc.collect()
            wait_stopped()
            assert (destination / '.metadata').is_file()
        """,
    )


def test_dead_training_process_releases_pid_reference(daemon_env: dict[str, str]) -> None:
    if daemon_env.get("NATIVE_CKPT_SERVER_DISABLE_INPROC") != "1":
        pytest.skip("the embedded daemon exits with its owner")
    _run(daemon_env, "root = create(); os._exit(0)")
    _wait_stopped()


def test_outproc_retries_election_while_old_service_closes(daemon_env: dict[str, str]) -> None:
    if daemon_env.get("NATIVE_CKPT_SERVER_DISABLE_INPROC") != "1":
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
        del root
        gc.collect()
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
            del root
            gc.collect()
            time.sleep(1.5)
            assert request('/api/reference/ping'), 'coordinator exited while remote was alive'
            if {remote_exit!r} == 'release':
                request('/api/reference/release/remote-test')
            else:
                peer.shutdown()
                peer.server_close()
            wait_stopped()
        finally:
            peer.shutdown()
            peer.server_close()
            thread.join()
        """,
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_child_preserves_parent_reference(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        root = create()
        child = os.fork()
        if child == 0:
            del root
            gc.collect()
            sys.exit(0)
        _, status = os.waitpid(child, 0)
        assert status == 0, status
        assert request('/api/reference/ping'), 'child released the parent reference'
        del root
        gc.collect()
        wait_stopped()
        """,
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_child_releases_only_its_own_reference(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        root = create()
        ready_read, ready_write = os.pipe()
        exit_read, exit_write = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(ready_read)
            os.close(exit_write)
            child_root = create()
            del child_root, root
            gc.collect()
            os.write(ready_write, b'1')
            os.close(ready_write)
            os.read(exit_read, 1)
            os._exit(0)
        os.close(ready_write)
        os.close(exit_read)
        try:
            assert os.read(ready_read, 1) == b'1'
            assert request('/api/reference/ping'), 'child released the parent reference'
            del root
            gc.collect()
            wait_stopped()
        finally:
            os.write(exit_write, b'1')
            os.close(exit_write)
            _, status = os.waitpid(child, 0)
            assert status == 0, status
        """,
    )


def test_nonzero_local_rank_can_host(daemon_env: dict[str, str]) -> None:
    if daemon_env.get("NATIVE_CKPT_SERVER_DISABLE_INPROC") == "1":
        pytest.skip("checks embedded ownership")
    daemon_env["LOCAL_RANK"] = "1"
    _run(
        daemon_env,
        """
        from pathlib import Path
        root = checkpoint.model_state.create_root(9100, 1, 2, ['localhost'], 2)
        root.register_state_dict({'value': 1})
        root.finalize_registry()
        deadline = time.monotonic() + 10
        while True:
            try:
                request('/api/reference/ping')
                break
            except (URLError, ConnectionError):
                assert time.monotonic() < deadline, 'nonzero rank did not start a daemon'
                time.sleep(0.05)
        assert Path(os.environ['NATIVE_CKPT_SERVER_LOG'].replace('%p', str(os.getpid()))).is_file()
        leader = checkpoint.model_state.create_root(9100, 0, 2, ['localhost'], 2)
        leader.register_state_dict({'value': 1})
        leader.finalize_registry()
        root.wait_for_async_completion()
        leader.wait_for_async_completion()
        del root, leader
        gc.collect()
        wait_stopped()
        """,
    )


def test_another_process_can_own_the_replacement_daemon(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        import subprocess
        checkpoint.ping()
        checkpoint.svc_spindown()
        result = subprocess.run(
            [sys.executable, '-c', 'from nvrx_checkpointing import checkpoint; checkpoint.ping()'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        """,
    )
    _wait_stopped()


def test_client_exit_after_becoming_an_owner(daemon_env: dict[str, str]) -> None:
    if daemon_env.get("NATIVE_CKPT_SERVER_DISABLE_INPROC") == "1":
        pytest.skip("checks embedded ownership")
    _run(
        daemon_env,
        """
        os.environ['NATIVE_CKPT_SERVER_DISABLE_INPROC'] = '1'
        checkpoint.ping()
        checkpoint.svc_spindown()
        os.environ['NATIVE_CKPT_SERVER_DISABLE_INPROC'] = '0'
        checkpoint.ping()
        """,
    )
    _wait_stopped()


def test_ping_reacquires_after_forced_restart(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        checkpoint.ping()
        first = request('/api/reference/ping')
        checkpoint.svc_spindown()
        checkpoint.ping()
        assert request('/api/reference/ping') != first
        time.sleep(5.2)
        assert request('/api/reference/ping'), 'replacement service has no client reference'
        """,
    )
