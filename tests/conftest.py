"""Test-suite-wide pytest hooks.

Each integration test starts once the checkpoint daemon of the test before it
has wound down, so the test registers with a daemon of its own.

One noisy torch logger is silenced because it otherwise breaks pytest-cmake's
test discovery on macOS.

`torch.distributed.elastic.multiprocessing.redirects` calls `get_libc()` at
module load time (see `libc = get_libc()` at the module body). On macOS that
goes down the `IS_MACOS` branch and logs:

    NOTE: Redirects are currently not supported in Windows or MacOs.

pytest-cmake's discoverer (`PytestAddTests.cmake`) captures both stdout and
stderr from `pytest --collect-only -q` into one buffer, then parses lines as
test IDs. The torch warning ends up smushed into a ctest test name and ctest
later fails with `file or directory not found: W0524 ... NOTE: Redirects ...`.

Pre-creating the logger here and pinning its level above WARNING makes the
subsequent `logger.warning(...)` inside `get_libc()` a no-op. Python's logger
registry is keyed by name, so by the time torch's module does
`logger = logging.getLogger(__name__)`, it gets the already-configured logger.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import urllib.request

import pytest

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

_WIND_DOWN_TIMEOUT_SECONDS = 30


def _checkpoint_daemon_answers() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:4100/api/ping", timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _earlier_checkpoint_daemon_wound_down(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("integration") is None:
        return
    # A failed test's frame stays referenced for post-mortem debugging and would keep its daemon references.
    for name in ("last_exc", "last_type", "last_value", "last_traceback"):
        if hasattr(sys, name):
            delattr(sys, name)
    gc.collect()
    deadline = time.monotonic() + _WIND_DOWN_TIMEOUT_SECONDS
    while _checkpoint_daemon_answers():
        if time.monotonic() >= deadline:
            raise TimeoutError("the checkpoint daemon of an earlier test did not wind down")
        time.sleep(0.05)
