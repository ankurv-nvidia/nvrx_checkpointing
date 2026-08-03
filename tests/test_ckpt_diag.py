# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import ckpt_diag

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_server_log_diagnostic_is_scoped_to_local_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ckpt_diag.socket, "gethostname", lambda: "node-a.example.com")
    (tmp_path / "checkpoint_server.node-a.100.log").write_text("local server failure\n")
    (tmp_path / "checkpoint_server.node-b.200.log").write_text("remote server noise\n")

    ckpt_diag._print_log(tmp_path, 12)

    output = capsys.readouterr().out
    assert "checkpoint_server.node-a.100.log" in output
    assert "local server failure" in output
    assert "node-b" not in output
    assert "remote server noise" not in output


def test_connection_failure_probes_logs_and_process_even_when_port_is_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setenv("RANK", "112")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))
    monkeypatch.setattr(ckpt_diag, "_print_log", lambda directory, rank: calls.append(("log", (directory, rank))))
    monkeypatch.setattr(ckpt_diag, "_print_stack", lambda rank: calls.append(("stack", rank)))

    ckpt_diag.diagnose_checkpoint_server_failure(
        RuntimeError("Failed to start or connect to checkpoint service"),
        "/logs",
    )

    assert calls == [("log", ("/logs", "112")), ("stack", "112")]


def test_stack_diagnostic_reports_when_no_server_process_remains(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ckpt_diag.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ckpt_diag, "_get_server_pid", lambda _pgrep: None)

    ckpt_diag._print_stack(20)

    assert "no checkpoint_server process found for stack capture" in capsys.readouterr().out
