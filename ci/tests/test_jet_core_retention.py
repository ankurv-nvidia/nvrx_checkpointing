# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

JET_DIR = Path(__file__).resolve().parents[2] / "ci" / "jet"


@pytest.fixture
def workload(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(JET_DIR))
    runtime = ModuleType("runtime")
    runtime.slurm_job = SimpleNamespace(_identity=lambda _args: ("ultra4_64x4", "native-perf", "ultra4"))
    runtime.parse_args = lambda _argv: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    spec = importlib.util.spec_from_file_location("jet_core_retention_test", JET_DIR / "run_workload.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cleanup_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    work = tmp_path / "1234"
    run = work / "ultra4"
    monkeypatch.setenv("JET_CLEAN_RUN_AFTER_FINALIZE", "1")
    monkeypatch.setenv("SLURMJOB_OPT_RUN_DIR", str(run))
    monkeypatch.setenv("SLURMJOB_OPT_RUN_NAME", run.name)
    monkeypatch.setenv("SLURMJOB_OPT_WORK_DIR", str(work))
    monkeypatch.setenv("UPSTREAM_PIPELINE_ID", work.name)
    return run


@pytest.mark.parametrize("suffix", [".core", ".core.partial", ".core.log", ".exe", ".exe.partial"])
@pytest.mark.parametrize("routine", ["", "native"])
def test_cleanup_preserves_crash_artifacts(
    workload: ModuleType,
    cleanup_environment: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
    routine: str,
) -> None:
    run = cleanup_environment
    logs = run / routine / "logs" / "native.ckpt"
    logs.mkdir(parents=True)
    core = logs / f"checkpoint_server.node.42.log{suffix}"
    core.write_bytes(b"crash state")

    workload.cleanup_finalized_run([], run.parent / "assets")

    assert core.read_bytes() == b"crash state"
    assert str(run) in capsys.readouterr().out


def test_cleanup_removes_archived_run_without_crash_artifacts(workload: ModuleType, cleanup_environment: Path) -> None:
    run = cleanup_environment
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "trainer.log").write_text("training finished\n")
    assets = run.parent / "assets"
    assets.mkdir()
    artifact_root = run.parent / "artifacts"
    workload._collect_run_artifacts(artifact_root, run)
    workload._archive(assets, artifact_root, "ultra4")

    workload.cleanup_finalized_run([], assets)

    assert not run.exists()
    assert workload._archive_path(assets, "ultra4").is_file()


def test_archive_contains_manifest_and_diagnostics_without_large_cores(
    workload: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "work" / "ultra4"
    logs = run / "native" / "logs" / "native.ckpt"
    logs.mkdir(parents=True)
    core = logs / "checkpoint_server.node.42.log.core"
    with core.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"core")
    executable = logs / "checkpoint_server.node.42.log.exe"
    executable.write_bytes(b"matching executable")
    partial = logs / "checkpoint_server.other.43.log.core.partial"
    partial.write_bytes(b"partial core")
    capture_log = logs / "checkpoint_server.node.42.log.core.log"
    capture_log.write_text("original_signal=6\nSaved corefile\n")
    artifact_root = tmp_path / "artifacts"

    workload._collect_run_artifacts(artifact_root, run)

    destination = artifact_root / "ci" / "ci-runs" / run.name
    manifest = json.loads((destination / "core_manifest.json").read_text())
    assert manifest["retained_run_root"] == str(run)
    files = {entry["path"]: entry["size_bytes"] for entry in manifest["files"]}
    assert files[str(core.relative_to(run))] == core.stat().st_size
    assert files[str(executable.relative_to(run))] == executable.stat().st_size
    assert files[str(partial.relative_to(run))] == partial.stat().st_size
    assert (destination / capture_log.relative_to(run)).read_text() == capture_log.read_text()
    assert not (destination / core.relative_to(run)).exists()
    assert not (destination / executable.relative_to(run)).exists()
    assert not (destination / partial.relative_to(run)).exists()

    monkeypatch.setenv("UPSTREAM_PIPELINE_ID", "1234")
    assets = tmp_path / "assets"
    assets.mkdir()
    workload._archive(assets, artifact_root, "ultra4")
    archive = workload._archive_path(assets, "ultra4")
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert any(name.endswith("core_manifest.json") for name in names)
    assert any(name.endswith(".core.log") for name in names)
    assert not any(name.endswith((".core", ".core.partial", ".exe", ".exe.partial")) for name in names)
    assert archive.stat().st_size < core.stat().st_size
    assert core.is_file()
