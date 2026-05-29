"""Test-suite-wide pytest hooks.

The only thing here right now is silencing one noisy torch logger that
otherwise breaks pytest-cmake's test discovery on macOS.

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

import logging

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
