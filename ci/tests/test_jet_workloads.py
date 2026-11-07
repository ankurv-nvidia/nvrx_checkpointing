# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_SOURCE_KEYS = ("MEGATRON_REPO_URL", "MEGATRON_COMMIT", "MEGATRON_ROOT", "JET_PREPARED_MEGATRON_COMMIT", "JET_PREPARED_MEGATRON_ROOT")
_RUNTIME_KEYS = ("JET_RUNTIME_KEY", "JET_RUNTIME_ROOT", "JET_LOCAL_RUNTIME_ROOT", "JET_RUNTIME_ARCHIVE")


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("jet_workloads_test", _REPO / "ci" / "jet" / "generate_workloads.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("JET_CLUSTERS", json.dumps({"gb300": {"config": "team/test"}}))
    return module


def _exports(generator: ModuleType, config: str) -> dict[str, str]:
    workload = generator.Workload("gb300", "native", config, ("perf", "async", "native"), (), 64, 4)
    environment = generator._environment(generator.JetPlatform("gb300", "gb300", "aarch64", 4), workload, "/assets", "/shared path")
    script = environment.format(
        env=SimpleNamespace(UPSTREAM_COMMIT_SHA="source", UPSTREAM_PROJECT_ID="project", UPSTREAM_PIPELINE_ID="pipeline"),
    )
    keys = (*_SOURCE_KEYS, *_RUNTIME_KEYS)
    command = f"import os, json; print(json.dumps({{key: os.environ[key] for key in {keys!r} if key in os.environ}}))"
    script += f"\n{shlex.quote(sys.executable)} -c {shlex.quote(command)}"
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "1234"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("config", ["nano4", "ultra4"])
def test_profiles_use_the_prepared_image_checkout(generator: ModuleType, config: str) -> None:
    values = _exports(generator, config)

    assert values["MEGATRON_ROOT"] == values["JET_PREPARED_MEGATRON_ROOT"] == "/opt/megatron-lm"
    assert "MEGATRON_COMMIT" not in values
    assert "MEGATRON_REPO_URL" not in values
    assert "JET_PREPARED_MEGATRON_COMMIT" not in values
    assert values["JET_RUNTIME_KEY"] == generator.RUNTIME_KEY


def test_training_failure_allows_remaining_tasks_to_collect_cores(generator: ModuleType) -> None:
    platform = generator.JetPlatform("gb300", "gb300", "aarch64", 4)
    workload = generator.Workload("gb300", "native", "ultra4", ("perf", "native"), (), 64, 4)

    manifest = generator.recipe_manifest(platform, workload, "maintainer", "image")

    flags = manifest["launchers"]["type:slurm"]["srun_additional_flags"]
    assert flags["kill_on_bad_exit=0"] is True
    assert "kill_on_bad_exit=1" not in flags


@pytest.mark.parametrize("profile", ["h100", "b200"])
def test_small_native_profile_includes_inproc_checkpoint_coverage(generator: ModuleType, profile: str) -> None:
    platform = generator.JetPlatform(profile, "h100", "x86_64", 8)
    workloads = [workload for workload in generator.resolved_workloads(platform) if "inproc" in workload.flavors]

    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.config == "pico"
    assert workload.nodes == 1
    assert workload.provider == "native"
    assert workload.flavors == ("detr", "inproc")
    assert workload.runs == ("ref:nvrx", "cmp:native")
    assert "detr+inproc" in workload.job_args
