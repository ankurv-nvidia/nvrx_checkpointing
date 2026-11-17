# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101
"""Exercise native daemon ownership through client process lifetimes."""

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
            request('/api/ping')
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
            with urlopen(Request("http://127.0.0.1:4100/api/stop"), timeout=2):
                pass


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
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with urlopen(Request("http://127.0.0.1:4100/api/ping"), timeout=2):
                pass
        except (URLError, TimeoutError, ConnectionError):
            return
        time.sleep(0.05)
    pytest.fail("checkpoint service outlived its client")


def test_clean_release_has_no_shutdown_grace(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        from pathlib import Path
        root = create()
        own_log = Path(os.environ['NATIVE_CKPT_SERVER_LOG'].replace('%p', str(os.getpid())))
        assert own_log.is_file() == (os.environ.get('NATIVE_CKPT_SERVER_DISABLE_INPROC') != '1')
        started = time.monotonic()
        del root
        gc.collect()
        wait_stopped()
        assert time.monotonic() - started < 4, 'clean shutdown waited for startup grace'
        """,
    )


def test_global_root_is_closed_at_interpreter_exit(daemon_env: dict[str, str]) -> None:
    _run(daemon_env, "ROOT = create()")


def test_destructor_finishes_async_cpu_save(daemon_env: dict[str, str], tmp_path: Path) -> None:
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


def test_ping_does_not_extend_startup_grace(daemon_env: dict[str, str]) -> None:
    _run(
        daemon_env,
        """
        started = time.monotonic()
        checkpoint.ping()
        time.sleep(3)
        checkpoint.ping()
        wait_stopped()
        assert time.monotonic() - started < 7.5, 'ping extended the startup grace'
        """,
    )


def test_reacquire_saves_after_daemon_restart(daemon_env: dict[str, str], tmp_path: Path) -> None:
    _run(
        daemon_env,
        f"""
        for index in range(2):
            root = create()
            destination = {str(tmp_path)!r} + '/' + str(index)
            root.save_async(index, destination)
            del root
            gc.collect()
            wait_stopped()
            assert os.path.isfile(destination + '/.metadata')
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
                request('/api/ping')
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
        assert request('/api/ping'), 'child released the parent reference'
        del root
        gc.collect()
        wait_stopped()
        """,
    )


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
