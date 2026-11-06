# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import test_slurm_job_manual as helpers

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

_COMMIT = helpers.DEFAULT_MEGATRON_COMMIT
_REPOSITORY = "https://github.com/NVIDIA/Megatron-LM.git"


@pytest.fixture
def slurm_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    module = helpers._load_slurm_job()
    helpers._clear_run_env(monkeypatch)
    for name in module._MODEL_ENV_NAMES | {"JET_PREPARED_MEGATRON_ROOT"}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return module


def test_ultra4_uses_the_shared_megatron_source(slurm_job: ModuleType) -> None:
    args, extra = slurm_job._parse_args(slurm_job._parser(), ["--config", "ultra4"])

    assert not extra
    assert args.megatron_commit == _COMMIT
    assert args.megatron_repo_url == _REPOSITORY


@pytest.mark.parametrize("override", ["cli", "environment"])
def test_explicit_source_overrides(
    slurm_job: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    commit = "b" * 40
    repository = "https://github.com/example/megatron.git"
    arguments = ["--config", "ultra4"]
    monkeypatch.setenv("MEGATRON_COMMIT", "a" * 40)
    monkeypatch.setenv("MEGATRON_REPO_URL", "https://github.com/example/environment.git")
    if override == "cli":
        arguments += ["--megatron-commit", commit, "--megatron-repo-url", repository]
    else:
        monkeypatch.setenv("MEGATRON_COMMIT", commit)
        monkeypatch.setenv("MEGATRON_REPO_URL", repository)

    args, _extra = slurm_job._parse_args(slurm_job._parser(), arguments)

    assert args.megatron_commit == commit
    assert args.megatron_repo_url == repository


def test_ultra4_uses_the_prepared_image_checkout(
    slurm_job: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared-source"
    monkeypatch.setenv("JET_PREPARED_MEGATRON_ROOT", str(prepared))
    args, _extra = slurm_job._parse_args(slurm_job._parser(), ["--config", "ultra4"])

    assert slurm_job._megatron_checkout(args) == prepared
    assert slurm_job._megatron_checkout(args, "b" * 40) != prepared


def _command(slurm_job: ModuleType, tmp_path: Path, mode: str, phase: int) -> tuple[list[str], dict[str, str]]:
    args = helpers._args(
        tmp_path,
        config="ultra4",
        megatron_commit=_COMMIT,
        megatron_repo_url=_REPOSITORY,
        flavors=[mode, "async", "native"],
    )
    slurm_job._configure_run(args)
    return slurm_job._phase_command("ultra4_64x4", args, "native", phase)


def _value(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


@pytest.mark.parametrize("phase", [1, 2])
def test_ultra4_phase_preserves_recipe_architecture_and_environment(slurm_job: ModuleType, tmp_path: Path, phase: int) -> None:
    command, environment = _command(slurm_job, tmp_path, "perf", phase)

    expected = {
        "--tensor-model-parallel-size": "2",
        "--expert-model-parallel-size": "64",
        "--optimizer": "muon",
        "--fp8-format": "e4m3",
        "--fp8-recipe": "mxfp8",
        "--step-batch-size-schedule": "0:4608,173342195712:9216,520026587136:13824",
    }
    assert {name: _value(command, name) for name in expected} == expected
    assert "--async-ckpt-use-cpu-shm" in command
    assert "--moe-shortcut-connection" in command
    assert "--moe-shortcut-parallel" in command
    assert "--mtp-hsm" in command
    for name in (
        "--wide-residual",
        "--global-batch-size",
        "--cuda-graph-impl",
        "--cuda-graph-modules",
        "--moe-router-enable-expert-bias",
        "--mtp-hybrid-override-pattern",
        "--mtp-num-layers",
        "--hybrid-override-pattern",
    ):
        assert name not in command
    assert environment["MEGATRON_COMMIT"] == _COMMIT
    assert environment["MEGATRON_REPO_URL"] == _REPOSITORY
    assert environment["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "40"
    assert environment["NVTE_BWD_LAYERNORM_SM_MARGIN"] == "40"
    assert environment["NVTE_INF_LAYERNORM_SM_MARGIN"] == "40"
    assert environment["NCCL_NVLS_ENABLE"] == "1"
    assert environment["NVTE_CPU_OFFLOAD_V1"] == "1"
    assert environment["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "64"
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in environment
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ


def test_explicit_environment_values_survive_profile_defaults(slurm_job: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "52")
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")

    slurm_job._configure_model_environment("ultra4")

    assert os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "52"
    assert os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] == "40"
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ


def test_determinism_overrides_the_recipes_training_fast_paths(slurm_job: ModuleType, tmp_path: Path) -> None:
    command, environment = _command(slurm_job, tmp_path, "detr", 2)

    assert _value(command, "--attention-backend") == "unfused"
    assert _value(command, "--moe-router-dtype") == "fp64"
    assert "--deterministic-mode" in command
    assert environment["NCCL_NVLS_ENABLE"] == "0"
    assert "--gtp-remat-nccl-ub" not in command
    assert "--gtp-expert-remat-nccl-ub" not in command
