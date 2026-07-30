# ruff: noqa: INP001, S108
from __future__ import annotations

import contextlib
import http.client
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LocalDaemon:
    name: str
    process_pattern: str
    base_url: str
    cleanup_globs: tuple[str, ...] = ()

    def pids(self) -> list[str]:
        return subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", self.process_pattern],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()

    def require_stopped(self) -> None:
        if pids := self.pids():
            raise RuntimeError(f"{self.name} already running (pids {', '.join(pids)})")
        if self.reachable():
            raise RuntimeError(f"{self.name} is still reachable at {self.base_url}")
        for pattern in self.cleanup_globs:
            path = Path(pattern)
            for stale in path.parent.glob(path.name):
                stale.unlink(missing_ok=True)

    def request(self, endpoint: str, timeout: float) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/{endpoint}", timeout=timeout) as response:
                response.read()
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            pass

    def reachable(self, timeout: float = 0.25) -> bool:
        url = urlsplit(self.base_url)
        try:
            with socket.create_connection((url.hostname or "127.0.0.1", url.port or 80), timeout):
                return True
        except OSError:
            return False

    def stop(self, graceful: bool = True, timeout: int = 90) -> None:
        if not self.pids():
            self.require_stopped()
            return
        self.request("spindown" if graceful else "stop", timeout)
        deadline = time.monotonic() + timeout
        while self.pids() and time.monotonic() < deadline:
            time.sleep(0.25)
        if self.pids():
            raise RuntimeError(f"{self.name} did not exit")
        self.require_stopped()

    def terminate(self, timeout: int = 30) -> None:
        deadline = time.monotonic() + timeout
        self.request("stop", min(timeout, 5))
        while self.reachable() and time.monotonic() < deadline:
            time.sleep(0.25)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in self.pids():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pid), sig)
            while self.pids() and time.monotonic() < deadline:
                time.sleep(0.25)
            if not self.pids():
                break
        if self.pids() or self.reachable():
            raise RuntimeError(f"{self.name} did not exit")
        self.require_stopped()


_SERVICE_PORT = "41000"
CHECKPOINT_SERVER = LocalDaemon(
    name="checkpoint_server",
    process_pattern="[c]heckpoint_server",
    base_url=f"http://127.0.0.1:{_SERVICE_PORT}/api",
    cleanup_globs=(
        "/dev/shm/checkpoint_request_*",
        str(Path(tempfile.gettempdir()) / f"nvrx_ckpt_server.{_SERVICE_PORT}.lock"),
    ),
)


if __name__ == "__main__":
    CHECKPOINT_SERVER.terminate()
