# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, PLR0915, PLR2004, S101, S105, S607, SLF001
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_DIR = REPO_ROOT / "ci"
JET_DIR = CI_DIR / "jet"
SCRIPTS_DIR = CI_DIR / "scripts"
GITHUB_DIR = CI_DIR / "github"
for path in (str(JET_DIR), str(CI_DIR), str(SCRIPTS_DIR), str(GITHUB_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


generate_workloads = importlib.import_module("generate_workloads")
workload_matrix = importlib.import_module("workload_matrix")
wheel_install = importlib.import_module("wheel_install")
slurm_job = importlib.import_module("slurm_job")
run_workload = importlib.import_module("run_workload")
submit_workload = importlib.import_module("submit_workload")
generate_jet_workflows = importlib.import_module("generate_jet_workflows")
publish_wheels = importlib.import_module("publish_wheels")
# The wrapper owns the runner module under test.
slurm_job = run_workload.slurm_job


@pytest.fixture(autouse=True)
def _isolate_jet_parser_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", raising=False)


@pytest.fixture(autouse=True)
def _jet_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(workload_matrix.CLUSTERS_VARIABLE, (REPO_ROOT / workload_matrix.CLUSTERS_EXAMPLE).read_text())


def _jet_args(*args: str) -> object:
    inherited_run = os.environ.pop("SLURMJOB_OPT_RUN", None)
    try:
        return run_workload.runtime.parse_args(list(args))
    finally:
        if inherited_run is not None:
            os.environ["SLURMJOB_OPT_RUN"] = inherited_run


def test_blackwell_platform_topologies_and_wheel_paths() -> None:
    h100 = workload_matrix.jet_platform("h100")
    gb200 = workload_matrix.jet_platform("gb200")
    b200 = workload_matrix.jet_platform("b200")
    gb300 = workload_matrix.jet_platform("gb300")

    assert h100.max_timeout_seconds == 3 * 60 * 60
    assert gb200.max_timeout_seconds == 3 * 60 * 60
    assert (gb200.architecture, gb200.gpus_per_node, gb200.wheel_mode) == (
        "aarch64",
        4,
        "prebuilt",
    )
    assert (b200.architecture, b200.gpus_per_node, b200.wheel_mode) == ("x86_64", 8, "prebuilt")
    assert b200.max_timeout_seconds == 2 * 60 * 60
    assert (gb300.architecture, gb300.gpus_per_node, gb300.wheel_mode) == (
        "aarch64",
        4,
        "prebuilt",
    )
    assert gb300.max_nodes == 18
    assert gb300.max_timeout_seconds == 4 * 60 * 60
    assert not gb300.time_budget_overrides_seconds


def test_gb300_route_limits_runner_prepare_to_one_task() -> None:
    custom_config = generate_workloads.jet_custom_config(workload_matrix.jet_platform("gb300"))
    launcher = custom_config["launchers"]["gb300nvl_dlcluster"]
    environment = custom_config["executors"]["jet-ci"]["environments"]["gb300nvl_dlcluster"]
    before_script = environment["before_script"]

    assert launcher["partition"] == "gb300nvl72_ci"
    assert "sbatch_additional_flags" not in launcher
    assert environment["variables"] == {"GIT_STRATEGY": "none", "SLURM_NTASKS": "1"}
    assert len(before_script) == 1
    assert 'scontrol update "JobId=$(printenv SLURM_JOB_ID)" Requeue=0' in before_script[0]
    assert "${SLURM_JOB_ID" not in before_script[0]
    assert "${allocation_id" not in before_script[0]
    assert "Requeue=0" in before_script[0]
    assert before_script[0].endswith("unset SLURM_NTASKS\n")


def test_gb200_route_uses_oci_hsg() -> None:
    platform = workload_matrix.jet_platform("gb200")
    custom_config = generate_workloads.jet_custom_config(platform)

    assert workload_matrix.jet_cluster("gb200").config == "dlalgo-ci/oci-hsg"
    launcher = custom_config["launchers"]["oci-hsg"]
    assert "partition" not in launcher
    environment = custom_config["executors"]["jet-ci"]["environments"]["oci-hsg"]
    assert environment["variables"] == {"GIT_STRATEGY": "none"}
    assert launcher["sbatch_additional_flags"] == {
        "deadline": "now+3days",
        "gpus_per_node": 4,
    }


def test_jet_image_prepares_pinned_megatron_checkout() -> None:
    dockerfile = (JET_DIR / "image" / "Dockerfile").read_text()
    prepare_runtime = (JET_DIR / "image" / "prepare_runtime.sh").read_text()
    prepare_megatron = (JET_DIR / "image" / "prepare_megatron_runtime.sh").read_text()
    workload_generator = (JET_DIR / "generate_workloads.py").read_text()

    assert "ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3" in dockerfile
    assert "NVRX_RESILIENCY_EXT_VERSION=0.7.0" in dockerfile
    assert '"nvidia-resiliency-ext==${NVRX_RESILIENCY_EXT_VERSION}"' in prepare_runtime
    assert "--no-deps" in prepare_runtime
    assert "actual_torch_version" in prepare_runtime
    assert "base_numpy_version=$(python3 -c" in prepare_runtime
    assert '"numpy==${base_numpy_version}"' in prepare_runtime
    assert "retained base-image NumPy" in prepare_runtime
    assert 'm.version("nvidia-resiliency-ext")' in prepare_runtime
    assert 'local_prepared_marker="${local_runtime_root}/.prepared"' in prepare_runtime
    assert 'exec 8>"${local_runtime_root}.lock"' in prepare_runtime
    assert "reusing node-local native runtime" in prepare_runtime
    assert "archive_local_runtime" in prepare_runtime
    assert "archive_members+=(wheels)" not in prepare_runtime
    assert '"/tmp/nvrx-runtime/{env.UPSTREAM_PROJECT_ID}/"' in workload_generator
    toolchain = dockerfile.index("ci/jet/image/install_toolchain.sh")
    checkout = dockerfile.index('scripts/prepare.sh megatron_root "${MEGATRON_ROOT}"')
    megatron_source_deps = dockerfile.index("ci/jet/image/install_megatron_source_deps.sh")
    deepep = dockerfile.index("ci/jet/image/install_deepep_source.sh")
    source_revision_arg = dockerfile.index("ARG NVRX_SOURCE_REVISION")
    source_revision = dockerfile.index("ENV NVRX_SOURCE_REVISION=${NVRX_SOURCE_REVISION}")
    full_source = dockerfile.index("COPY . /workspace/nvrx_checkpointing")

    assert (
        toolchain
        < checkout
        < megatron_source_deps
        < deepep
        < source_revision_arg
        < source_revision
        < full_source
    )
    assert 'RUN bash scripts/prepare.sh megatron_root "${MEGATRON_ROOT}"' in dockerfile
    assert "scripts/prepare.sh ubuntu2404" not in dockerfile
    assert not re.search(r"scripts/prepare\.sh megatron(?!_root)", dockerfile)
    assert "ci/jet/image/build_inline_wheels.sh" not in dockerfile
    assert "NVRX_DEEPEP_ROOT=/opt/deepep-src" in dockerfile
    assert "NVRX_MEGATRON_EXTRAS=training,dev,otel,te,ssm" in dockerfile
    assert "NVRX_MEGATRON_CONSTRAINTS=" in dockerfile
    assert dockerfile.index("install_toolchain.sh") < dockerfile.index("NVRX_MEGATRON_EXTRAS=")
    assert "MAMBA_FORCE_BUILD=TRUE" in dockerfile
    assert "UV_UNMANAGED_INSTALL=/usr/local/bin" in dockerfile
    assert dockerfile.index("UV_UNMANAGED_INSTALL=/usr/local/bin") < checkout
    assert "pip uninstall --yes --break-system-packages nvidia-resiliency-ext" in dockerfile
    assert dockerfile.index("pip uninstall --yes --break-system-packages") < checkout
    megatron_source_deps_script = (
        JET_DIR / "image" / "install_megatron_source_deps.sh"
    ).read_text()
    assert "5e52befd5262c06289106338c308079d6adb391f" in megatron_source_deps_script
    assert "0048fbf2e7b2f214dcbe703ea3dec2b9647595e1" in megatron_source_deps_script
    assert "cd81f0413cad2fc1e6f17e785ac39f59aae690cd" in megatron_source_deps_script
    assert "f134af63deb2df17e1171a9ec1ea4a7d8604d5ca" in megatron_source_deps_script
    assert "b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16" in megatron_source_deps_script
    assert "sed -i 's/-std=c++17/-std=c++20/g'" in megatron_source_deps_script
    assert 'causal-conv1d = { path = \\"' in megatron_source_deps_script
    assert 'mamba-ssm = { path = \\"' in megatron_source_deps_script
    assert 'fast-hadamard-transform = { path = \\"' in megatron_source_deps_script
    assert 'emerging_optimizers = { path = \\"' in megatron_source_deps_script
    assert "ssm_cuda_arch.patch" not in dockerfile
    assert "ssm_cuda_arch.patch" not in megatron_source_deps_script
    assert 'git -C "${source_root}" status --short' in megatron_source_deps_script
    assert "submodule update --init --recursive --depth 1" in megatron_source_deps_script
    assert "--configure-only" in megatron_source_deps_script
    assert 'transformer-engine = { path = \\"${transformer_engine_root}\\" }' in (
        megatron_source_deps_script
    )
    deepep_source = (JET_DIR / "image" / "install_deepep_source.sh").read_text()
    assert "10d4dd7377d5bce900fbb4b80cce863764892b95" in deepep_source
    assert "1e0c869c39bb33f1034cb9920bd2a8a8406f04a3" in deepep_source
    assert "3994fd525c52c3d717bc36edd142796a1a0f12daa25e3e4a355f47b07359770e" in deepep_source
    assert 'git -C "${deepep_root}" fetch --depth 1' in deepep_source
    assert '"${deepep_root}/third-party/nccl/Makefile"' in deepep_source
    assert re.findall(r'scripts/prepare\.sh"? (\w+)', prepare_runtime) == ["megatron_root"]
    assert "prepare_megatron_runtime.sh" in prepare_runtime
    assert 'cp -a "${NVRX_DEEPEP_ROOT:?}/."' in prepare_megatron
    assert "extras=${NVRX_MEGATRON_EXTRAS:-training,dev,otel,te,ssm}" in prepare_megatron
    assert "NVRX_MEGATRON_CONSTRAINTS" in prepare_megatron
    assert "JET Megatron overlay matches the regular-CI package constraints" in prepare_megatron
    assert "import causal_conv1d, mamba_ssm" in prepare_megatron
    assert "export NVTE_FRAMEWORK=pytorch" in prepare_megatron
    assert 'm.version("nvidia-cutlass-dsl")' in prepare_megatron
    assert '"nvidia-cudnn-frontend==1.28.0"' in prepare_megatron
    assert "torch-c-dlpack-ext" not in prepare_megatron
    assert '--override "${overrides}"' in prepare_megatron
    assert "printf '%s\\n' \"quack-kernels; sys_platform == 'never'\" >> \"${overrides}\"" in prepare_megatron
    assert '"nvidia-cutlass-dsl[cu13]==${base_cutlass_dsl_version}"' in prepare_megatron
    assert '"${uv_args[@]}" --upgrade' in prepare_megatron
    assert '"${uv_args[@]}" --no-deps --reinstall' in prepare_megatron
    assert '"nvidia-cutlass-dsl-libs-cu13==${base_cutlass_dsl_version}"' in prepare_megatron
    assert 'hasattr(cutlass_dsl, "CuteExperimentalDSL")' in prepare_megatron
    assert "import flashinfer" in prepare_megatron
    assert "NVRX_MEGATRON_DEPS_ROOT}/TransformerEngine" not in prepare_megatron
    assert "command -v uv" in dockerfile
    constraints = (JET_DIR / "image" / "megatron-runtime-constraints.txt").read_text().splitlines()
    assert "quack-kernels==0.6.2" in constraints


def test_jet_image_tracks_the_regular_image_recipe() -> None:
    regular = (REPO_ROOT / "scripts" / "prepare.sh").read_text()
    dockerfile = (JET_DIR / "image" / "Dockerfile").read_text()
    prepare_megatron = (JET_DIR / "image" / "prepare_megatron_runtime.sh").read_text()
    deepep_source = (JET_DIR / "image" / "install_deepep_source.sh").read_text()
    megatron_source_deps = (JET_DIR / "image" / "install_megatron_source_deps.sh").read_text()
    workload_generator = (JET_DIR / "generate_workloads.py").read_text()

    base_image = re.search(r'DOCKER_IMAGES\["pytorch"\]="([^"]+)"', regular)[1]
    assert f"ARG FROM_IMAGE_NAME={base_image}" in dockerfile
    assert f'"FROM_IMAGE_NAME": "{base_image}"' in workload_generator
    deepep_commit = re.search(r"DEEPEP_COMMIT=([0-9a-f]{40})", regular)[1]
    assert f"deepep_commit={deepep_commit}" in deepep_source
    nvshmem = re.search(r"nvidia-nvshmem-cu13==([0-9.]+)", regular)[1]
    assert f"nvidia-nvshmem-cu13=={nvshmem}" in prepare_megatron
    cudnn_frontend = re.search(r"nvidia-cudnn-frontend\[cutedsl\]==([0-9.]+)", regular)[1]
    assert f'"nvidia-cudnn-frontend=={cudnn_frontend}"' in prepare_megatron
    extras = re.search(r'-e "\$\{MEGATRON_ROOT\}\[([a-z,]+)\]"', regular)[1]
    assert f"NVRX_MEGATRON_EXTRAS={extras}" in dockerfile
    assert f"extras=${{NVRX_MEGATRON_EXTRAS:-{extras}}}" in prepare_megatron
    for patch in ("scripts/deepep.patch", "scripts/deepep-nvshmem.patch"):
        assert Path(patch).name in regular
        assert f'"${{repo_root}}/{patch}"' in prepare_megatron
    assert 'ONLY_FUNCS="SendRecv"' in regular
    assert "export ONLY_FUNCS=SendRecv" in prepare_megatron
    assert "-std=c++20" in regular
    assert "-std=c++20" in megatron_source_deps
    assert "override-dependencies" in regular
    assert "override-dependencies" in prepare_megatron


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_jet_prebuilt_runtime_republishes_valid_node_local_cache(
    tmp_path: Path,
    architecture: str,
) -> None:
    runtime_root = tmp_path / "shared-runtime"
    local_runtime_root = tmp_path / "local-runtime"
    runtime_archive = runtime_root / "native-runtime.tar"
    python = local_runtime_root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    expected_state = f"key {architecture} 10.3 prebuilt local-bundle-v11"
    (local_runtime_root / ".prepared").write_text(f"{expected_state}\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(f"#!/bin/sh\necho {architecture}\n")
    fake_uname.chmod(0o755)

    result = subprocess.run(
        ["bash", str(JET_DIR / "image" / "prepare_runtime.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "JET_RUNTIME_ROOT": str(runtime_root),
            "JET_LOCAL_RUNTIME_ROOT": str(local_runtime_root),
            "JET_RUNTIME_ARCHIVE": str(runtime_archive),
            "JET_RUNTIME_KEY": "key",
            "NVRX_COMPUTE_CAPABILITY": "10.3",
            "JET_WHEEL_MODE": "prebuilt",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "reusing node-local native runtime" in result.stdout
    assert (runtime_root / ".prepared").read_text().strip() == expected_state
    with tarfile.open(runtime_archive) as archive:
        assert "venv/bin/python" in archive.getnames()


def test_slurm_time_uses_canonical_day_clock() -> None:
    workload = generate_workloads.Workload(
        "gb300",
        "native",
        "pico",
        ("perf", "native"),
        (),
        1,
        4,
        24 * 60 * 60,
    )

    assert workload.slurm_time == "1-00:00:00"


def test_native_setup_uses_minimal_shared_venv() -> None:
    prepare = (JET_DIR / "image" / "prepare_megatron_runtime.sh").read_text()
    dockerfile = (JET_DIR / "image" / "Dockerfile").read_text()
    toolchain = (JET_DIR / "image" / "install_toolchain.sh").read_text()

    assert 'uv venv --system-site-packages "${venv}"' in prepare
    assert 'python3 -m venv --system-site-packages "${venv}"' not in prepare
    assert "MAX_JOBS=${NVRX_MAX_JOBS:-4}" in prepare
    assert 'cd "${megatron_root}"' in prepare
    assert "Inherited PyTorch" in prepare
    assert 'uv_megatron pip install "${uv_args[@]}" --no-deps' in prepare
    assert '"nvidia-cutlass-dsl[cu13]==${base_cutlass_dsl_version}"' in prepare
    assert '"nvidia-cutlass-dsl-libs-cu13==${base_cutlass_dsl_version}"' in prepare
    assert "CUTLASS DSL changed from base-image" in prepare
    assert "CuteExperimentalDSL required by FlashInfer" in prepare
    assert "NVRX_RDMA_PREREQS_READY=1" in dockerfile
    assert "NVRX_USE_IMAGE_FLASH_ATTN=1" in dockerfile
    assert "Reusing image FlashAttention" in prepare
    assert "rdma-core" in toolchain
    assert "libibverbs-dev" in toolchain
    assert "libnvidia-ml-dev" in toolchain
    assert "slurm-client" not in toolchain


def test_gb300_matrix_enforces_node_and_proven_runtime_limits() -> None:
    supported, unsupported = generate_workloads.select_workloads(
        workload_matrix.jet_platform("gb300"),
        "full",
    )

    assert supported
    assert {workload.nodes for workload in unsupported} == {32}
    assert {workload.nodes for workload in supported} == {8}
    assert all(workload.timeout_seconds <= 4 * 60 * 60 for workload in supported)
    nano4_deterministic = [
        workload
        for workload in supported
        if workload.config == "nano4" and workload.flavors == ("detr",)
    ]
    assert {workload.provider for workload in nano4_deterministic} == {"native", "nvrx"}
    assert {workload.config for workload in [*supported, *unsupported]} == {"nano4"}


def test_generator_reports_unsupported_workloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "gb300.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_workloads.py",
            "--platform",
            "gb300",
            "--suite",
            "full",
            "--maintainer",
            "ankurv",
            "--output",
            str(output),
        ],
    )

    assert generate_workloads.main() == 0
    assert output.is_file()
    error = capsys.readouterr().err
    assert "skipping unsupported workload gb300-native-detr-nano4-32x4" in error
    assert "nodes=32 exceeds max_nodes=18" in error


def test_gb200_oci_runtime_cap_preserves_full_matrix() -> None:
    platform = workload_matrix.jet_platform("gb200")
    smoke, smoke_unsupported = generate_workloads.select_workloads(platform, "smoke")
    full, full_unsupported = generate_workloads.select_workloads(platform, "full")

    assert not smoke_unsupported
    assert not full_unsupported
    assert {workload.timeout_seconds for workload in smoke} == {3 * 60 * 60}
    assert {workload.timeout_seconds for workload in full} == {60 * 60, 2 * 60 * 60, 3 * 60 * 60}
    fp8_deterministic = next(
        workload
        for workload in full
        if workload.config == "ultra3fp8" and workload.flavors == ("detr",)
    )
    assert fp8_deterministic.timeout_seconds == 3 * 60 * 60


def test_distributed_timeout_is_jet_transport_metadata() -> None:
    gb200 = generate_workloads.resolved_workloads(workload_matrix.jet_platform("gb200"))
    gb300 = generate_workloads.resolved_workloads(workload_matrix.jet_platform("gb300"))

    for workload in [*gb200, *gb300]:
        if workload.nodes >= 16:
            assert workload.extra_args[-2:] == ("--distributed-timeout-minutes", "60")
        else:
            assert "--distributed-timeout-minutes" not in workload.extra_args

    deterministic = next(
        workload
        for workload in gb200
        if workload.nodes >= 16 and workload.runs
    )
    assert deterministic.job_args.index("--distributed-timeout-minutes") < deterministic.job_args.index("--run")


@pytest.mark.parametrize("platform_name", ["h100", "gb200", "b200", "gb300"])
def test_full_suite_collapses_regular_pipeline_partitions(platform_name: str) -> None:
    platform = workload_matrix.jet_platform(platform_name)
    resolved = generate_workloads.resolved_workloads(platform)
    supported, unsupported = generate_workloads.select_workloads(platform, "full")

    assert {workload.identity for workload in [*supported, *unsupported]} == {
        workload.identity for workload in resolved
    }
    assert {workload.pipeline_suite for workload in resolved} <= generate_workloads.POLICY_SUITES


@pytest.mark.parametrize("platform_name", ["h100", "gb200", "b200", "gb300"])
def test_jet_idle_gpu_exemption_matches_each_allocation(platform_name: str) -> None:
    platform = workload_matrix.jet_platform(platform_name)
    rendered, unsupported = generate_workloads.render(platform, "full", "ankurv")
    workloads, expected_unsupported = generate_workloads.select_workloads(platform, "full")
    documents = list(yaml.safe_load_all(rendered))

    assert unsupported == expected_unsupported
    assert len(documents) == len(workloads) + 1
    for workload, recipe in zip(workloads, documents[1:], strict=True):
        comment = recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]["comment"]
        reaper_json = comment.partition("\n")[0]
        reaper = json.loads(reaper_json.replace("{{", "{").replace("}}", "}").replace('\\"', '"'))[
            "OccupiedIdleGPUsJobReaper"
        ]
        assert reaper == {
            "exemptIdleTimeMins": str(workload.timeout_seconds // 60),
            "reason": "other",
            "description": "native-checkpoint-run",
        }


@pytest.mark.parametrize("platform_name", ["h100", "gb200", "b200"])
def test_perf_smoke_selects_one_platform_native_gate(platform_name: str) -> None:
    platform = workload_matrix.jet_platform(platform_name)

    selected, unsupported = generate_workloads.select_workloads(platform, "smoke-perf")

    assert not unsupported
    assert len(selected) == 1
    assert selected[0].identity == ("native", "pico", ("perf", "native"), (), 1)
    assert selected[0].gpus_per_node == platform.gpus_per_node
    assert selected[0].timeout_seconds == (
        3 * 60 * 60
        if platform_name in {"h100", "gb200"}
        else 2 * 60 * 60
        if platform_name == "b200"
        else 60 * 60
    )


@pytest.mark.parametrize(
    ("platform_name", "docker_platform", "wheel_mode", "tasks"),
    [
        ("h100", "linux/amd64", "prebuilt", 8),
        ("gb200", "linux/arm64", "prebuilt", 4),
        ("b200", "linux/amd64", "prebuilt", 8),
    ],
)
def test_smoke_manifests_encode_platform_contract(
    platform_name: str,
    docker_platform: str,
    wheel_mode: str,
    tasks: int,
) -> None:
    rendered, unsupported = generate_workloads.render(
        workload_matrix.jet_platform(platform_name),
        "smoke",
        "ankurv",
    )
    build, recipe = list(yaml.safe_load_all(rendered))

    assert not unsupported
    assert build["key_pattern"] == f"{platform_name}-ci-image"
    assert build["spec"]["platforms"] == [docker_platform]
    assert "parent" not in build["spec"]
    assert build["spec"]["source"]["dockerfile"] == "ci/jet/image/Dockerfile"
    assert build["spec"]["source"]["arguments"]["FROM_IMAGE_NAME"] == "nvcr.io/nvidia/pytorch:26.09-py3"
    assert "NVRX_WHEEL_MODE" not in build["spec"]["source"]["arguments"]
    assert recipe["key_pattern"] == f"{platform_name}-native-detr-pico-1x{tasks}"
    assert recipe["spec"]["image_source"] == {"build_key": f"build/{platform_name}-ci-image"}
    assert recipe["launchers"]["type:slurm"]["ntasks_per_node"] == tasks
    allocation_flags = recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]
    assert "deadline" not in allocation_flags
    assert allocation_flags["gpus_per_node"] == tasks
    assert "ntasks_per_node" not in allocation_flags
    assert "cpus_per_task" not in allocation_flags
    expected_time = (
        "03:00:00"
        if platform_name in {"h100", "gb200"}
        else "02:00:00"
        if platform_name == "b200"
        else "01:00:00"
    )
    assert allocation_flags["time"] == expected_time
    reaper = json.loads(
        allocation_flags["comment"].replace("{{", "{").replace("}}", "}").replace('\\"', '"'),
    )[
        "OccupiedIdleGPUsJobReaper"
    ]
    assert reaper == {
        "exemptIdleTimeMins": str(120 if platform_name == "b200" else 180),
        "reason": "other",
        "description": "native-checkpoint-run",
    }
    assert allocation_flags.get("gpus") is None
    assert allocation_flags.get("segment") is None
    assert recipe["launchers"]["type:slurm"]["srun_additional_flags"]["time"] == expected_time
    assert "gpu_bind" not in recipe["launchers"]["type:slurm"]["srun_additional_flags"]
    assert recipe["launchers"]["type:slurm"]["srun_additional_flags"]["kill_on_bad_exit=1"] is True
    assert recipe["spec"]["gpus"] == tasks
    assert recipe["spec"]["nodes"] == 1
    assert recipe["spec"]["time_limit"] is None
    assert "retry" not in rendered
    assert f"export PLATFORM={platform_name}" in recipe["spec"]["script"]
    assert f"export JET_WHEEL_MODE={wheel_mode}" in recipe["spec"]["batch_setup"]
    assert "ci/jet/image/prepare_runtime.sh" in recipe["spec"]["batch_setup"]
    assert "disable_slurm_requeue" not in recipe["spec"]["batch_setup"]
    assert "ci/jet/image/activate_runtime.sh" in recipe["spec"]["batch_setup"]
    assert "ci/jet/image/activate_runtime.sh" in recipe["spec"]["script"]
    assert "export JET_RUNTIME_ROOT=" in recipe["spec"]["batch_setup"]
    assert "export JET_RUNTIME_ARCHIVE=" in recipe["spec"]["batch_setup"]
    assert "export JET_LOCAL_RUNTIME_ROOT=" in recipe["spec"]["batch_setup"]
    assert "export NVRX_MAX_JOBS=16" in recipe["spec"]["batch_setup"]
    assert "export NCCL_IB_TIMEOUT=24" in recipe["spec"]["script"]
    assert "NVRX_SINGLE_GPU_PER_TASK" not in recipe["spec"]["script"]
    assert "NCCL_SHM_DISABLE" not in recipe["spec"]["script"]
    assert 'export TRITON_CACHE_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/triton-cache"' in recipe["spec"]["script"]
    assert 'export TRITON_CACHE_PATH="$TRITON_CACHE_DIR"' in recipe["spec"]["script"]
    assert 'export TORCHINDUCTOR_CACHE_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/torchinductor-cache"' in recipe["spec"]["script"]
    assert 'export TORCH_EXTENSIONS_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/torch-extensions"' in recipe["spec"]["script"]
    assert (
        'export SLURMJOB_OPT_TENSORBOARD_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/'
        'tensorboard/${{SLURMJOB_OPT_RUN_NAME:?}}"'
        in recipe["spec"]["script"]
    )
    assert "nvrx-ci-runs/{env.UPSTREAM_PIPELINE_ID}" in recipe["spec"]["batch_setup"]
    assert "nvrx-ci-artifacts/{env.UPSTREAM_PIPELINE_ID}" in recipe["spec"]["batch_setup"]
    assert "nvrx-ci-artifacts/{env.UPSTREAM_PIPELINE_ID}" in recipe["spec"]["script_teardown"]
    assert "export SLURMJOB_OPT_RUN_DIR=" in recipe["spec"]["batch_teardown"]
    assert "export SLURMJOB_OPT_CLEAN_REFERENCE_SERVER=1" in recipe["spec"]["script"]
    assert "export SLURMJOB_OPT_CLEAN_RUN=1" in recipe["spec"]["batch_setup"]
    assert "export JET_CLEAN_RUN_AFTER_FINALIZE=1" in recipe["spec"]["batch_teardown"]
    assert (
        f"export NVRX_DEEPEP_CUDA_ARCH_LIST={workload_matrix.jet_platform(platform_name).compute_capability}"
        in recipe["spec"]["batch_setup"]
    )
    assert "export MEGATRON_ROOT=/opt/megatron-lm" in recipe["spec"]["batch_setup"]
    assert "export JET_PREPARED_MEGATRON_ROOT=/opt/megatron-lm" in recipe["spec"]["batch_setup"]
    assert 'export PATH="$NVRX_VENV/bin:$PATH"' in recipe["spec"]["batch_setup"]
    assert "ci/jet/prepare_workload.py" in recipe["spec"]["batch_setup"]
    assert "ci/jet/run_workload.py" in recipe["spec"]["script"]
    assert "export SLURMJOB_OPT_ROUTINE_LABEL=jet" in recipe["spec"]["script"]
    assert "export SLURMJOB_OPT_ISOLATE_PHASE_SERVER_PORTS=1" in recipe["spec"]["script"]
    assert "ci/jet/finalize_workload.py" in recipe["spec"]["script_teardown"]
    assert "ci/jet/cleanup_workload.py" in recipe["spec"]["batch_teardown"]
    assert "SLURM_PROCID" not in recipe["spec"]["script_teardown"]
    assert "export JET_INLINE_WHEEL_DIR=" not in recipe["spec"]["batch_setup"]


def test_jet_uses_image_owned_default_megatron(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _jet_args("--config", "pico", "--flavor", "native", "--gpus-per-node", "8")
    prepared = tmp_path / "megatron-lm"
    monkeypatch.setenv("JET_PREPARED_MEGATRON_ROOT", str(prepared))
    monkeypatch.setattr(slurm_job, "WORK_ROOT", tmp_path / "work")

    assert slurm_job._megatron_checkout(args) == prepared.resolve()
    assert slurm_job._megatron_checkout(args, "1" * 40) != prepared.resolve()


def test_jet_restores_image_owned_megatron_checkout() -> None:
    dockerfile = (JET_DIR / "image" / "Dockerfile").read_text()
    prepare_runtime = (JET_DIR / "image" / "prepare_runtime.sh").read_text()

    restore = 'git -C "${MEGATRON_ROOT}" checkout --quiet -- pyproject.toml'
    assert restore in dockerfile
    assert restore in prepare_runtime


def test_large_gb200_recipes_share_one_storage_lane() -> None:
    rendered, unsupported = generate_workloads.render(
        workload_matrix.jet_platform("gb200"),
        "full",
        "ankurv",
    )
    assert not unsupported
    recipes = list(yaml.safe_load_all(rendered))[1:]
    large = [recipe for recipe in recipes if recipe["spec"]["nodes"] >= 16]
    small = [recipe for recipe in recipes if recipe["spec"]["nodes"] < 16]

    assert len(large) == 10
    assert {
        recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]["dependency"]
        for recipe in large
    } == {"singleton"}
    lane_names = {
        recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]["job_name"]
        for recipe in large
    }
    assert lane_names == {"nvrx-ci-gb200-large-{env.UPSTREAM_COMMIT_SHA}"}
    assert all(len(name.replace("{env.UPSTREAM_COMMIT_SHA}", "0" * 40)) <= 128 for name in lane_names)
    assert all(
        "dependency" not in recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]
        for recipe in small
    )


def test_github_ci_runs_the_default_jet_suite() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    jet = workflow["jobs"]["jet-gb200"]
    default, _unsupported = generate_workloads.select_workloads(workload_matrix.jet_platform("gb200"), "default")

    assert jet["uses"] == "./.github/workflows/jet.yml"
    assert jet["with"]["platform"] == "gb200"
    assert json.loads(jet["with"]["workloads"]) == [workload.display_name for workload in default]
    assert jet["with"]["wheel-aarch64-artifact-id"] == "${{ needs.wheel-aarch64.outputs.artifact-id }}"


def test_manual_jet_workflows_offer_a_checkbox_per_workload() -> None:
    platforms = generate_jet_workflows.platforms()

    assert {platform.name for platform in platforms} == {"gb200", "gb300", "h100", "b200"}
    for platform in platforms:
        path = generate_jet_workflows.workflow_path(platform)
        assert path.read_text() == generate_jet_workflows.render(platform), f"rerun ci/github/generate_jet_workflows.py for {path.name}"
        workflow = yaml.safe_load(path.read_text())
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        supported, _unsupported = generate_workloads.select_workloads(platform, "full")
        assert set(inputs) == {workload.display_name for workload in supported}
        assert len(inputs) <= generate_jet_workflows.MAX_INPUTS
        assert all(spec["type"] == "boolean" and spec["default"] is False for spec in inputs.values())
        jet = workflow["jobs"]["jet"]
        assert jet["uses"] == "./.github/workflows/jet.yml"
        assert jet["with"]["platform"] == platform.name
        assert jet["with"][f"wheel-{platform.architecture}-artifact-id"] == "${{ needs.wheel.outputs.artifact-id }}"


@pytest.mark.parametrize(
    ("ticked", "expected"),
    [
        ({"a-1x4": True, "b-8x4": False, "c-16x4": True}, ["a-1x4", "c-16x4"]),
        ({"a-1x4": False}, None),
    ],
)
def test_jet_selection_lists_the_ticked_workloads(tmp_path: Path, ticked: dict, expected: list | None) -> None:
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    result = subprocess.run(
        ["bash", str(GITHUB_DIR / "jet.sh"), "select"],
        env={**os.environ, "JET_INPUTS": json.dumps(ticked), "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        check=False,
    )

    if expected is None:
        assert result.returncode != 0
        assert "Tick at least one workload." in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f"workloads={json.dumps(expected)}\n"
        assert all(f"`{name}`" in summary.read_text() for name in expected)


def test_perf_soft_failure_names_each_missed_metric(tmp_path: Path) -> None:
    results = tmp_path / "jet-results"
    (results / "ci" / "ci-results").mkdir(parents=True)
    fields = ("status", "measured", "baseline", "tolerance_pct", "unit", "statistic")
    comparison = {
        "bw_mbps": dict(zip(fields, ("FAIL", 137803.5, 148600.0, 5.0, "MB/s", "median"), strict=True)),
        "wr_tnsr_us": dict(zip(fields, ("FAIL", 177560.5, 165000.0, 5.0, "us", "median"), strict=True)),
        "ckpt_durable_ms": dict(zip(fields, ("PASS", 317.2, 1871.0, 500.0, "ms", "mean"), strict=True)),
    }
    (results / "ci" / "ci-results" / "gb200_perf_report.json").write_text(json.dumps({"comparison": comparison}))
    summary = tmp_path / "summary.md"

    result = subprocess.run(
        ["python3", str(GITHUB_DIR / "perf_warning.py"), "gb200-native-perf-super3fp4-16x4", str(results)],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == (
        "::warning title=gb200-native-perf-super3fp4-16x4%3A 2 metrics missed the baseline::"
        "bw_mbps median 137,803.5 MB/s vs baseline 148,600.0 MB/s (-7.3%25, tolerance 5.0%25)%0A"
        "wr_tnsr_us median 177,560.5 us vs baseline 165,000.0 us (+7.6%25, tolerance 5.0%25)\n"
    )
    table = summary.read_text()
    assert "| `bw_mbps` | median | 137,803.5 MB/s | 148,600.0 MB/s | -7.3% | 5.0% |" in table
    assert "| `wr_tnsr_us` | median | 177,560.5 us | 165,000.0 us | +7.6% | 5.0% |" in table
    assert "ckpt_durable_ms" not in table


def test_perf_soft_failure_without_a_report_still_warns(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python3", str(GITHUB_DIR / "perf_warning.py"), "gb200-x", str(tmp_path / "missing")],
        env={key: value for key, value in os.environ.items() if key != "GITHUB_STEP_SUMMARY"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "::warning title=gb200-x%3A performance missed the baseline::See the jet-results artifact.\n"


def _changed_options(results: Path, run: str, changes: list[dict]) -> None:
    record = results / "ci" / "ci-runs" / run / "logs" / "baselines" / "gb200_perf_arguments_differences.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(changes))


def test_perf_soft_failure_names_each_changed_option(tmp_path: Path) -> None:
    results = tmp_path / "jet-results"
    change = {"option": "recompute_granularity", "recorded": "None", "run": "selective"}
    # Both compared runs of a workload record the same change once each.
    _changed_options(results, "ref", [change])
    _changed_options(results, "cmp", [change])
    summary = tmp_path / "summary.md"

    result = subprocess.run(
        ["python3", str(GITHUB_DIR / "perf_warning.py"), "gb200-x", str(results)],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == (
        "::warning title=gb200-x%3A 1 option differs from the recorded run::recompute_granularity: None -> selective\n"
    )
    assert "| `recompute_granularity` | None | selective |" in summary.read_text()


def test_perf_soft_failure_names_missed_metrics_and_changed_options(tmp_path: Path) -> None:
    results = tmp_path / "jet-results"
    (results / "ci" / "ci-results").mkdir(parents=True)
    fields = ("status", "measured", "baseline", "tolerance_pct", "unit", "statistic")
    comparison = {"bw_mbps": dict(zip(fields, ("FAIL", 137803.5, 148600.0, 5.0, "MB/s", "median"), strict=True))}
    (results / "ci" / "ci-results" / "gb200_perf_report.json").write_text(json.dumps({"comparison": comparison}))
    _changed_options(results, "run", [{"option": "micro_batch_size", "recorded": "1", "run": "2"}])

    result = subprocess.run(
        ["python3", str(GITHUB_DIR / "perf_warning.py"), "gb200-x", str(results)],
        env={key: value for key, value in os.environ.items() if key != "GITHUB_STEP_SUMMARY"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == (
        "::warning title=gb200-x%3A 1 metric missed the baseline; 1 option differs from the recorded run::"
        "bw_mbps median 137,803.5 MB/s vs baseline 148,600.0 MB/s (-7.3%25, tolerance 5.0%25)%0A"
        "micro_batch_size: 1 -> 2\n"
    )


def _wheel_name(build: int, arch: str) -> str:
    return f"nvrx_checkpointing-0.0.1.dev{build}+g{build:08x}-cp312-abi3-linux_{arch}.whl"


def _release_asset(name: str, asset_id: int, digest: str = "") -> dict:
    return {
        "id": asset_id,
        "name": name,
        "browser_download_url": f"https://github.com/o/r/releases/download/wheels/{name}",
        **({"digest": f"sha256:{digest}"} if digest else {}),
    }


class _FakeReleases:
    """Answers the GitHub release calls the publisher makes and records each one."""

    def __init__(self, existing: list[dict] | None, *, rename: bool = False) -> None:
        self.release = None if existing is None else {"id": 7}
        self.assets = list(existing or [])
        self.rename = rename
        self.calls: list[tuple[str, str, bytes | None]] = []

    def __call__(self, method: str, url: str, body: bytes | None = None, _content_type: str = "") -> object:
        self.calls.append((method, url, body))
        if url.endswith("/releases/tags/wheels"):
            if self.release is None:
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            return self.release
        if method == "POST" and url.endswith("/releases"):
            self.release = {"id": 7}
            return self.release
        if "/releases/7/assets?per_page=100&page=" in url:
            return list(self.assets) if url.endswith("page=1") else []
        if method == "POST" and url.startswith(publish_wheels.UPLOADS):
            name = urllib.parse.unquote(url.rsplit("name=", 1)[1])
            asset = _release_asset(name.replace("+", ".") if self.rename else name, 100 + len(self.assets))
            self.assets.append(asset)
            return asset
        if method == "DELETE":
            return None
        raise AssertionError(f"unexpected {method} {url}")


def _built_wheels(root: Path, build: int) -> Path:
    for arch in ("x86_64", "aarch64"):
        wheel = root / f"wheel-{arch}" / "dist" / _wheel_name(build, arch)
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(arch.encode())
    return root


def test_publishing_adds_the_build_and_keeps_the_newest_builds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(publish_wheels, "KEEP_BUILDS", 2)
    existing = [
        _release_asset(_wheel_name(build, arch), build * 10 + offset, digest=f"{build:064x}")
        for build in (3, 9)
        for offset, arch in enumerate(("x86_64", "aarch64"))
    ]
    api = _FakeReleases(existing)
    site = tmp_path / "site"

    published = publish_wheels.publish(api, "o/r", "abc", _built_wheels(tmp_path / "wheels", 12), site)

    expected = sorted(_wheel_name(build, arch) for build in (9, 12) for arch in ("x86_64", "aarch64"))
    assert sorted(asset["name"] for asset in published) == expected
    deleted = [url.rsplit("/", 1)[1] for method, url, _body in api.calls if method == "DELETE"]
    assert sorted(deleted) == ["30", "31"]
    page = (site / "simple" / "nvrx-checkpointing" / "index.html").read_text()
    assert f'#sha256={hashlib.sha256(b"x86_64").hexdigest()}">{_wheel_name(12, "x86_64")}</a>' in page
    assert f"#sha256={9:064x}" in page
    assert _wheel_name(3, "x86_64") not in page
    assert page.index(_wheel_name(12, "aarch64")) < page.index(_wheel_name(9, "aarch64"))
    assert 'href="nvrx-checkpointing/"' in (site / "simple" / "index.html").read_text()


def test_publishing_creates_the_rolling_release_once(tmp_path: Path) -> None:
    api = _FakeReleases(None)

    publish_wheels.publish(api, "o/r", "abc", _built_wheels(tmp_path / "wheels", 1), tmp_path / "site")
    publish_wheels.publish(api, "o/r", "abc", _built_wheels(tmp_path / "again", 1), tmp_path / "site")

    created = [json.loads(body) for method, url, body in api.calls if method == "POST" and url.endswith("/releases")]
    assert len(created) == 1
    assert created[0]["tag_name"] == "wheels"
    assert created[0]["prerelease"] is True
    assert "https://o.github.io/r/simple/" in created[0]["body"]
    assert len([call for call in api.calls if call[1].startswith(publish_wheels.UPLOADS)]) == 2


def test_publishing_refuses_a_wheel_github_renamed(tmp_path: Path) -> None:
    api = _FakeReleases([], rename=True)

    with pytest.raises(RuntimeError, match="pip cannot match"):
        publish_wheels.publish(api, "o/r", "abc", _built_wheels(tmp_path / "wheels", 5), tmp_path / "site")


def test_wheels_publish_only_from_main_after_every_gate() -> None:
    jobs = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    publish = jobs["publish-wheels"]

    assert publish["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert set(publish["needs"]) == set(jobs) - {"publish-wheels"}
    assert publish["permissions"] == {"contents": "write", "pages": "write", "id-token": "write"}
    assert publish["environment"]["name"] == "github-pages"
    assert publish_wheels.index_url("ankurv-nvidia/nvrx_checkpointing") in (REPO_ROOT / "README.md").read_text()


def test_workflows_use_node24_actions() -> None:
    uses = re.findall(r"uses: (actions/[\w-]+@v\d+)", "".join(p.read_text() for p in (REPO_ROOT / ".github" / "workflows").glob("*.yml")))

    assert set(uses) == {
        "actions/checkout@v7",
        "actions/upload-artifact@v7",
        "actions/download-artifact@v8",
        "actions/upload-pages-artifact@v5",
        "actions/deploy-pages@v5",
    }


def test_native_runtime_follows_prepare_sh_and_wheels_follow_the_run() -> None:
    rendered, _unsupported = generate_workloads.render(workload_matrix.jet_platform("gb200"), "default", "ankurv")
    _build, recipe, *_rest = yaml.safe_load_all(rendered)
    setup = recipe["spec"]["batch_setup"]
    key = hashlib.sha256((REPO_ROOT / "scripts" / "prepare.sh").read_bytes()).hexdigest()[:10]

    assert f"export JET_RUNTIME_KEY={key}\n" in setup
    assert "export JET_RUNTIME_CACHE_DIR=/nvrx-ci-cache\n" in setup
    assert f"/nvrx-ci-cache/nvrx-runtime/gb200/{key}/native-runtime.tar" in setup
    assert f"/tmp/nvrx-runtime/{{env.UPSTREAM_PROJECT_ID}}/gb200/{key}" in setup  # noqa: S108
    assert "nvrx-runtime/gb200/{env.UPSTREAM_COMMIT_SHA}" not in setup
    assert "{shared_dir}/nvrx-ci-runs/{env.UPSTREAM_PIPELINE_ID}/wheels" in setup


def test_runtime_cache_directory_is_a_mount_the_cluster_creates() -> None:
    clusters = {platform.name: workload_matrix.jet_cluster(platform.name) for platform in workload_matrix.JET_PLATFORMS}
    cached = {name for name, cluster in clusters.items() if cluster.cache}

    assert cached == {"gb200", "h100", "b200"}
    for name in cached:
        cluster = clusters[name]
        config = generate_workloads.jet_custom_config(workload_matrix.jet_platform(name))
        environment = config["executors"]["jet-ci"]["environments"][cluster.launcher]
        assert cluster.runtime_cache_dir == "/nvrx-ci-cache"
        assert config["launchers"][cluster.launcher]["mounts"] == {"/nvrx-ci-cache": cluster.cache}
        assert environment["before_script"][0] == f"mkdir -p {cluster.cache}"


def test_cluster_settings_come_from_the_repository_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    clusters = {
        "h100": {
            "config": "dlalgo-ci/elsewhere",
            "launcher": "elsewhere",
            "cache": "/scratch/ci-cache",
            "custom_config": {"executors": {"jet-ci": {"environments": {"elsewhere": {"before_script": ["date"]}}}}},
        },
    }
    monkeypatch.setenv("JET_CLUSTERS", json.dumps(clusters))
    config = generate_workloads.jet_custom_config(workload_matrix.jet_platform("h100"))

    assert workload_matrix.jet_cluster("h100").config == "dlalgo-ci/elsewhere"
    # The cluster creates the cache directory before the platform's own setup runs.
    assert config["executors"]["jet-ci"]["environments"]["elsewhere"]["before_script"] == ["mkdir -p /scratch/ci-cache", "date"]
    assert config["launchers"]["elsewhere"]["mounts"] == {"/nvrx-ci-cache": "/scratch/ci-cache"}
    assert config["retrier"] == generate_workloads.RETRIER_CONFIG["retrier"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", r"JET_CLUSTERS is unset; .*ci/jet/clusters\.example\.json"),
        ("{", "JET_CLUSTERS is not valid JSON"),
        ('{"gb200": {"config": "dlalgo-ci/oci-hsg"}}', r"no entry for platform 'h100'; it lists gb200"),
        ('{"h100": {"config": "c", "partition": "p"}}', "unknown fields: partition"),
        ('{"h100": {"launcher": "l"}}', "names no JET config"),
        ('{"h100": {"config": "c", "cache": "/scratch"}}', "sets a cache without the launcher"),
    ],
)
def test_cluster_settings_name_what_is_missing(monkeypatch: pytest.MonkeyPatch, value: str, message: str) -> None:
    monkeypatch.setenv("JET_CLUSTERS", value)

    with pytest.raises(RuntimeError, match=message):
        workload_matrix.jet_cluster("h100")


@pytest.mark.parametrize("value", ['["h100"]', '{"h100": "dlalgo-ci/coreweave-dfw"}'])
def test_cluster_settings_must_be_objects(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("JET_CLUSTERS", value)

    with pytest.raises(TypeError, match="must be a JSON object"):
        workload_matrix.jet_cluster("h100")


def test_cluster_example_names_every_jet_platform() -> None:
    example = json.loads((REPO_ROOT / workload_matrix.CLUSTERS_EXAMPLE).read_text())

    assert set(example) == {platform.name for platform in workload_matrix.JET_PLATFORMS}


def test_jet_workflow_reads_the_cluster_variable() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "jet.yml").read_text())

    assert workflow["env"]["JET_CLUSTERS"] == "${{ vars.JET_CLUSTERS }}"


def test_platforms_without_a_cache_directory_keep_the_runtime_for_one_job() -> None:
    platform = workload_matrix.jet_platform("gb300")
    assert workload_matrix.jet_cluster("gb300").runtime_cache_dir is None
    rendered, _unsupported = generate_workloads.render(platform, "full", "ankurv")
    _build, recipe, *_rest = yaml.safe_load_all(rendered)
    setup = recipe["spec"]["batch_setup"]
    key = hashlib.sha256((REPO_ROOT / "scripts" / "prepare.sh").read_bytes()).hexdigest()[:10]

    assert f"{{shared_dir}}/nvrx-runtime/gb300/{key}/native-runtime.tar" in setup
    assert "JET_RUNTIME_CACHE_DIR" not in setup


def test_jet_runtime_setup_refuses_an_unmounted_cache(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(JET_DIR / "image" / "prepare_runtime.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "JET_RUNTIME_CACHE_DIR": str(tmp_path / "cache"),
            "JET_RUNTIME_ROOT": str(tmp_path / "cache" / "nvrx-runtime" / "gb200" / "key"),
            "JET_LOCAL_RUNTIME_ROOT": str(tmp_path / "local-runtime"),
            "JET_RUNTIME_ARCHIVE": str(tmp_path / "cache" / "nvrx-runtime" / "gb200" / "key" / "native-runtime.tar"),
            "JET_RUNTIME_KEY": "key",
            "NVRX_COMPUTE_CAPABILITY": "10.0",
            "JET_WHEEL_MODE": "prebuilt",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"JET runtime cache {tmp_path / 'cache'} is not mounted" in result.stderr
    assert not (tmp_path / "cache").exists()


def test_recipes_receive_the_wheel_token_and_image_builds_do_not() -> None:
    rendered, _unsupported = generate_workloads.render(workload_matrix.jet_platform("gb200"), "default", "ankurv")
    build, *recipes = yaml.safe_load_all(rendered)

    assert build["type"] == "build"
    assert "secrets" not in build
    assert [recipe["key_pattern"] for recipe in recipes] == ["gb200-native-detr-pico-1x4", "gb200-native-perf-pico-1x4"]
    assert all(recipe["secrets"] == {"NVRX_GITHUB_TOKEN": "user/<current>/github/nvrx-ci-artifacts"} for recipe in recipes)


def test_gb200_recipes_keep_multinode_workloads_in_nvlink_segments() -> None:
    rendered, unsupported = generate_workloads.render(
        workload_matrix.jet_platform("gb200"),
        "full",
        "ankurv",
    )
    assert not unsupported
    recipes = list(yaml.safe_load_all(rendered))[1:]

    for recipe in recipes:
        nodes = recipe["spec"]["nodes"]
        flags = recipe["launchers"]["type:slurm"]["sbatch_additional_flags"]
        if nodes == 1:
            assert "segment" not in flags
        else:
            assert flags["segment"] == min(nodes, 16)


def _gb300_segments() -> dict[str, int | None]:
    rendered, _unsupported = generate_workloads.render(workload_matrix.jet_platform("gb300"), "full", "ankurv")
    return {
        recipe["key_pattern"]: recipe["launchers"]["type:slurm"]["sbatch_additional_flags"].get("segment")
        for recipe in list(yaml.safe_load_all(rendered))[1:]
    }


def test_the_cluster_segment_keeps_multinode_workloads_in_one_nvlink_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cluster whose racks are separate NVLink domains places each allocation within one.
    assert set(_gb300_segments().values()) == {None}
    clusters = {"gb300": {"config": "dlalgo-ci/nsc-lhr-slurm-1", "launcher": "b300_nsc_lhr_slurm_1", "segment": 16}}
    monkeypatch.setenv("JET_CLUSTERS", json.dumps(clusters))

    segments = _gb300_segments()

    assert segments
    assert all(key.endswith("-8x4") and segment == 8 for key, segment in segments.items()), segments


@pytest.mark.parametrize("segment", [0, "8", True])
def test_the_cluster_segment_must_be_a_positive_node_count(monkeypatch: pytest.MonkeyPatch, segment: object) -> None:
    monkeypatch.setenv("JET_CLUSTERS", json.dumps({"gb300": {"config": "c", "segment": segment}}))

    with pytest.raises(TypeError, match="segment must be a positive node count"):
        workload_matrix.jet_cluster("gb300")


def test_jet_cleanup_removes_only_exact_archived_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "nvrx-ci-runs" / "1234"
    run_dir = work_dir / "h100-native-perf-pico-1x8"
    sibling = work_dir / "keep-me"
    run_dir.mkdir(parents=True)
    sibling.mkdir()
    (run_dir / "checkpoint.bin").write_text("large")
    assets = tmp_path / "assets"
    artifact_ci = tmp_path / "artifact" / "ci"
    artifact_ci.mkdir(parents=True)
    (artifact_ci / "result.txt").write_text("preserved")
    args = ["pretrain-native", "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf"]
    parsed = run_workload.runtime.parse_args(args)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setenv("UPSTREAM_PIPELINE_ID", work_dir.name)
    _cfg, _flavor, stem = slurm_job._identity(parsed)
    assets.mkdir()
    with tarfile.open(run_workload._archive_path(assets, stem), "w:gz") as archive:
        archive.add(artifact_ci, arcname="ci")
    monkeypatch.setenv("JET_CLEAN_RUN_AFTER_FINALIZE", "1")
    monkeypatch.setenv("SLURMJOB_OPT_RUN_DIR", str(run_dir))
    monkeypatch.setenv("SLURMJOB_OPT_RUN_NAME", run_dir.name)
    monkeypatch.setenv("SLURMJOB_OPT_WORK_DIR", str(work_dir))

    run_workload.cleanup_finalized_run(args, assets)

    assert not run_dir.exists()
    assert sibling.is_dir()


def test_jet_cleanup_refuses_unscoped_or_unarchived_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "1234"
    run_dir = work_dir / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("JET_CLEAN_RUN_AFTER_FINALIZE", "1")
    monkeypatch.setenv("SLURMJOB_OPT_RUN_DIR", str(run_dir))
    monkeypatch.setenv("SLURMJOB_OPT_RUN_NAME", "different-run")
    monkeypatch.setenv("SLURMJOB_OPT_WORK_DIR", str(work_dir))
    monkeypatch.setenv("UPSTREAM_PIPELINE_ID", "1234")

    with pytest.raises(RuntimeError, match="unsafe JET run cleanup"):
        run_workload.cleanup_finalized_run(
            ["pretrain-native", "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf"],
            tmp_path / "assets",
        )

    assert run_dir.is_dir()


def test_jet_prepare_cleans_only_pipeline_scoped_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline_id = "1234"
    work_dir = tmp_path / "nvrx-ci-runs" / pipeline_id
    run_dir = work_dir / "h100-native-perf-pico-1x8"
    results_dir = run_dir / "logs"
    sibling_run = work_dir / "keep-me"
    artifact_root = tmp_path / "assets" / "nvrx-ci-artifacts" / pipeline_id
    sibling_artifact = artifact_root.parent / "9999"
    results_dir.mkdir(parents=True)
    sibling_run.mkdir()
    artifact_root.mkdir(parents=True)
    sibling_artifact.mkdir()
    (results_dir / "stale.log").write_text("old run")
    (artifact_root / "stale.txt").write_text("old artifact")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_dir)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("UPSTREAM_PIPELINE_ID", pipeline_id)
    monkeypatch.setenv("SLURMJOB_OPT_RUN_NAME", run_dir.name)
    monkeypatch.setenv("SLURMJOB_OPT_WORK_DIR", str(work_dir))
    monkeypatch.setenv("CI_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("CI_BASELINE_CANDIDATE_DIR", str(artifact_root / "ci" / "baselines"))

    run_workload.runtime._reset_stale_outputs(SimpleNamespace(clean_run=True))

    assert results_dir.is_dir()
    assert not (results_dir / "stale.log").exists()
    assert not artifact_root.exists()
    assert sibling_run.is_dir()
    assert sibling_artifact.is_dir()


def test_upstream_wheel_artifact_tracks_architecture_and_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_PROJECT_PATH", "owner/repo")
    monkeypatch.setenv("NVRX_GITHUB_TOKEN", "token")
    monkeypatch.setenv("UPSTREAM_WHEEL_X86_64_ARTIFACT_ID", "11")
    monkeypatch.setenv("WHEEL_ARCH", "x86_64")
    monkeypatch.setenv("WHEEL_VARIANT", "valgrind")
    monkeypatch.delenv("UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID", raising=False)

    with pytest.raises(RuntimeError, match="UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID"):
        wheel_install.fetch_upstream_wheel()


def test_required_wheel_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wheel_install, "install_prebuilt_wheel", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(wheel_install, "source_revision", lambda: "a" * 40)
    monkeypatch.setenv("WHEEL_ARCH", "x86_64")
    monkeypatch.setenv("WHEEL_VARIANT", "normal")

    with pytest.raises(RuntimeError, match="required normal x86_64 wheel"):
        wheel_install.require_prebuilt_wheel()


def test_bootstrap_report_writes_exact_baseline_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "ci" / "baselines"
    log = tmp_path / "run.log"
    log.write_text(
        "[native_ckpt:phase] phase=2\n"
        "[native_ckpt:mem] gpu_peak_bytes=1024 cpu_rss_bytes=2048\n",
    )
    (tmp_path / "checkpoint_server.cycle1.host.1.log").write_text(
        "[native_ckpt:svr][75][PERF] Rank=0 Begin span=ckpt.save "
        "started_at_ns=123 snapshot=1\n"
        "[native_ckpt:svr][75][PERF] Rank=0 Finished bw_mbps=12345.0 | Time(us):  "
        "wr_tnsr_us=456us snapshot_us=1us total_us=457us | "
        "Count: num_tensors=1 total_d2h=1 | Bytes: total_written_bytes=789 "
        "avg_d2h_rq=789 avg_tnsr=789 snapshot=1\n",
    )
    # A platform with no recorded baselines bootstraps them from its first run.
    monkeypatch.setenv("PLATFORM", "newgpu")
    monkeypatch.setenv("JET_BASELINE_BOOTSTRAP", "1")
    monkeypatch.setenv("CI_BASELINE_CANDIDATE_DIR", str(candidate_dir))
    stem = "newgpu_native-perf_pico_1x8"

    baseline = slurm_job.PerfReport.baseline(stem)
    passed = slurm_job.PerfReport.report(
        "native-perf",
        "pico_1x8",
        "native-perf",
        log,
        baseline,
        tmp_path / f"{stem}_report.json",
        8,
        otel_reporting=False,
    )
    args = _jet_args(
        "--nodes",
        "1",
        "--gpus-per-node",
        "8",
        "--config",
        "pico",
        "--flavor",
        "perf+native",
    )
    run_workload._write_baseline_candidate(args, log, stem)

    assert passed
    candidate = yaml.safe_load((candidate_dir / f"{stem}.yaml").read_text())
    assert candidate["metrics"] == {
        "wr_tnsr_us": 456,
        "total_written_bytes": 789,
        "gpu_peak_bytes": 1024,
        "training_process_peak_rss_bytes": 2048,
    }
    bandwidths = yaml.safe_load((candidate_dir / "platform_bw_mbps.yaml").read_text())
    assert bandwidths["newgpu"] == 12345


def test_jet_artifact_collection_excludes_checkpoint_payloads(tmp_path: Path) -> None:
    run_root = tmp_path / "shared" / "test-run"
    iteration = run_root / "checkpoints" / "iter_0000001"
    logs = run_root / "logs"
    iteration.mkdir(parents=True)
    logs.mkdir()
    (iteration / "registry.bin").write_bytes(b"registry")
    (iteration / ".metadata").write_bytes(b"metadata")
    (iteration / "payload.distcp").write_bytes(b"large payload")
    (logs / "worker.log").write_text("worker output\n")
    (logs / "run_checkpoint_digest.xz").write_bytes(b"digest")
    (logs / "run_checkpoint_summary.log").write_text("summary\n")
    artifact_root = tmp_path / "assets" / "nvrx-ci-artifacts"

    run_workload._collect_run_artifacts(artifact_root, run_root)

    collected = artifact_root / "ci" / "ci-runs" / run_root.name
    assert (collected / "checkpoints/iter_0000001/registry.bin").is_file()
    assert (collected / "checkpoints/iter_0000001/.metadata").is_file()
    assert (collected / "logs/worker.log").is_file()
    assert (collected / "logs/run_checkpoint_digest.xz").is_file()
    assert (collected / "logs/run_checkpoint_summary.log").is_file()
    assert not (collected / "checkpoints/iter_0000001/payload.distcp").exists()


def test_jet_combined_log_prefers_authoritative_workload_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "run" / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", results_dir.parent)
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf+native",
    )
    stem = slurm_job._identity(args)[2]
    (results_dir / f"{stem}.log").write_text("authoritative training and checkpoint output\n")
    daemon_log = results_dir / "checkpoint_server.cycle1.host.123.log"
    daemon_log.write_text("measured native bandwidth\n")
    otel_file = results_dir / "otel" / "cycle1" / "native_ckpt_otel.jsonl"
    otel_file.parent.mkdir(parents=True)
    otel_file.write_text("collected checkpoint spans\n")
    rank_log = results_dir / "jet_worker_rank0.log"
    rank_log.write_text("thin JET wrapper output\n")

    log, actual_stem = run_workload._combined_log(args, tmp_path / "artifacts", [rank_log])

    assert actual_stem == stem
    assert log.read_text() == "authoritative training and checkpoint output\n"
    assert (log.parent / daemon_log.name).read_text() == "measured native bandwidth\n"
    assert (log.parent / "otel" / "cycle1" / otel_file.name).read_text() == "collected checkpoint spans\n"


def test_jet_combined_log_prefers_rank_zero_routine_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "run" / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", results_dir.parent)
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf+native",
    )
    stem = slurm_job._identity(args)[2]
    (results_dir / f"{stem}.log").write_text("launcher preamble only\n")
    (results_dir / "pico_1x8_run-routine_rank0.log").write_text("successfully saved checkpoint\n")
    log, actual_stem = run_workload._combined_log(args, tmp_path / "artifacts", [])

    assert actual_stem == stem
    assert log.read_text() == "successfully saved checkpoint\n"


def test_jet_combined_log_concatenates_deterministic_routines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "run" / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", results_dir.parent)
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "detr",
        "--run", "ref:nvrx", "--run", "cmp:native",
    )
    stem = slurm_job._identity(args)[2]
    reference = results_dir.parent / "nvrx.ref" / "logs"
    measured = results_dir.parent / "native.cmp" / "logs"
    reference.mkdir(parents=True)
    measured.mkdir(parents=True)
    (reference / "pico_1x8_ref-routine_rank0.log").write_text(
        "reference checkpoint output\n",
    )
    (measured / "pico_1x8_cmp-routine_rank0.log").write_text(
        "measured checkpoint output\n",
    )
    rank_log = results_dir / "jet_worker_rank0.log"
    rank_log.write_text("thin JET wrapper output\n")

    log, actual_stem = run_workload._combined_log(args, tmp_path / "artifacts", [rank_log])

    assert actual_stem == stem
    assert log.read_text() == "reference checkpoint output\nmeasured checkpoint output\n"


def test_jet_combined_log_prefers_complete_jet_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "run" / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", results_dir.parent)
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "detr",
        "--run", "ref:nvrx", "--run", "cmp:native",
    )
    assets_dir = tmp_path / "assets"
    output_dir = assets_dir / "jet_assets" / "output_logs"
    output_dir.mkdir(parents=True)
    (output_dir / "output_script-0.log").write_text(
        "reference save\nmeasured save\nPASS: 0 differences vs baseline\n",
    )
    (results_dir / "pico_1x8_ref-routine_rank0.log").write_text(
        "incomplete routine-only output\n",
    )

    log, stem = run_workload._combined_log(
        args,
        tmp_path / "artifact-root",
        [],
        assets_dir,
    )

    assert stem == slurm_job._identity(args)[2]
    assert log.read_text() == "reference save\nmeasured save\nPASS: 0 differences vs baseline\n"


def test_jet_combined_log_falls_back_to_rank_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "run" / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", results_dir.parent)
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf+native",
    )
    rank_logs = [results_dir / "jet_worker_rank0.log", results_dir / "jet_worker_rank1.log"]
    rank_logs[0].write_text("rank zero\n")
    rank_logs[1].write_text("rank one\n")

    log, _stem = run_workload._combined_log(args, tmp_path / "artifacts", rank_logs)

    assert "===== JET rank 0: jet_worker_rank0.log =====\nrank zero\n" in log.read_text()
    assert "===== JET rank 1: jet_worker_rank1.log =====\nrank one\n" in log.read_text()


def test_jet_coordinates_server_preflight_before_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    checks: list[str] = []
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(terminate=lambda: checks.append("cleaned")),
    )
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURMD_NODENAME", "node-0")

    run_workload._coordinate_server_preflight(SimpleNamespace(nodes=1))

    assert checks == ["cleaned"]
    assert os.environ["SLURMJOB_OPT_SERVER_PREFLIGHT_DONE"] == "1"
    marker = run_root / ".jet-coordination/1234/server-preflight-node-0.json"
    assert json.loads(marker.read_text())["exit_code"] == 0


def test_jet_perf_soft_failure_is_nonblocking_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PLATFORM", "h100")
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf+native",
    )
    stem = slurm_job._identity(args)[2]
    log = tmp_path / f"{stem}.log"
    log.write_text("performance output\n")
    run_root = tmp_path / "live-run"
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    epilog_args = []

    def soft_failure(args: list[str]) -> NoReturn:
        epilog_args.extend(args)
        raise SystemExit(slurm_job.SOFT_FAIL_RC)

    monkeypatch.setattr(
        slurm_job,
        "_epilog",
        soft_failure,
    )

    assert run_workload._run_epilog(args, log) == 0
    marker = json.loads((tmp_path / f"{stem}_soft_failure.json").read_text())
    assert marker == {
        "exit_code": slurm_job.SOFT_FAIL_RC,
        "reason": "performance baseline comparison failed",
        "report": f"{stem}_report.json",
    }
    assert epilog_args[epilog_args.index("--run-root") + 1] == str(run_root)
    assert "JET PERFORMANCE BASELINE FAILED" in capsys.readouterr().err


def test_jet_epilog_keeps_options_outside_run_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PLATFORM", "h100")
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "detr",
        "--run", "ref:nvrx", "--run", "cmp:native",
    )
    log = tmp_path / f"{slurm_job._identity(args)[2]}.log"
    log.write_text("deterministic output\n")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", tmp_path / "run")
    captured: list[str] = []
    monkeypatch.setattr(slurm_job, "_epilog", captured.extend)

    assert run_workload._run_epilog(args, log) == 0
    first_run = captured.index("--run")
    assert captured.index("--log-file") < first_run
    assert captured.index("--run-root") < first_run
    assert captured.index("--baseline-json") < first_run


def _nvrx_detr_args() -> object:
    return _jet_args(
        "--nodes", "8", "--gpus-per-node", "4", "--config", "nano3", "--flavor", "detr",
        "--run", "ref:nvrx", "--run", "cmp:nvrx",
    )


def test_jet_epilog_grades_nvrx_detr_against_its_own_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _nvrx_detr_args()
    log = tmp_path / f"{slurm_job._identity(args)[2]}.log"
    log.write_text("deterministic output\n")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", tmp_path / "run")
    captured: list[str] = []
    monkeypatch.setattr(slurm_job, "_epilog", captured.extend)

    assert run_workload._run_epilog(args, log) == 0
    baseline = json.loads(captured[captured.index("--baseline-json") + 1])

    # Without a native run the job has no daemon metrics to grade.
    assert baseline == slurm_job.PerfReport.baseline("gb200_detr-nvrx_nano3_8x4")
    assert "bw_mbps" not in baseline["metrics"]


def test_jet_nvrx_detr_candidate_keeps_its_own_baseline_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PLATFORM", "gb200")
    candidate_dir = tmp_path / "ci" / "baselines"
    monkeypatch.setenv("CI_BASELINE_CANDIDATE_DIR", str(candidate_dir))
    args = _nvrx_detr_args()
    stem = slurm_job._identity(args)[2]
    log = tmp_path / f"{stem}.log"
    metric = {"min": 1, "median": 2, "mean": 2, "max": 3}
    (tmp_path / f"{stem}_report.json").write_text(json.dumps({"metrics": {"gpu_peak_bytes": metric}}))

    run_workload._write_baseline_candidate(args, log, stem)

    assert [path.name for path in candidate_dir.iterdir()] == ["gb200_detr-nvrx_nano3_8x4.yaml"]


def test_jet_arguments_candidate_removes_allocation_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "------------------------ arguments ------------------------\n"
        "  data_cache_path ................................. /jet/shared/cache\n"
        "  load ............................................ /jet/shared/load\n"
        "  micro_batch_size ................................ 1\n"
        "  save ............................................ /jet/shared/save\n"
        "  tensorboard_dir ................................. /jet/shared/tensorboard\n"
        "-------------------- end of arguments ---------------------\n",
    )
    candidate_dir = tmp_path / "baselines"
    monkeypatch.setenv("CI_BASELINE_CANDIDATE_DIR", str(candidate_dir))

    run_workload._write_arguments_candidate(log, "gb200_pretrain-native_perf_nano3_8x4")

    candidate = (candidate_dir / "gb200_pretrain-native_perf_nano3_8x4_arguments.txt").read_text()
    assert "micro_batch_size" in candidate
    assert "/jet/shared" not in candidate
    assert "data_cache_path" not in candidate
    assert "tensorboard_dir" not in candidate


def test_jet_non_perf_epilog_failure_remains_hard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _jet_args(
        "--nodes", "1", "--gpus-per-node", "8", "--config", "pico", "--flavor", "perf+native",
    )
    log = tmp_path / f"{slurm_job._identity(args)[2]}.log"
    log.write_text("incomplete output\n")
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setattr(
        slurm_job,
        "_epilog",
        lambda _args: (_ for _ in ()).throw(RuntimeError("missing checkpoints")),
    )

    assert run_workload._run_epilog(args, log) == 1
    assert not list(tmp_path.glob("*_soft_failure.json"))


def test_jet_first_surviving_rank_finalizes_and_publishes_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(run_workload.runtime, "parse_args", lambda _args: object())
    monkeypatch.setattr(run_workload.runtime, "configure", lambda _args: None)
    monkeypatch.setattr(run_workload.runtime, "require_prepared", lambda _args: {})
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(run_workload, "finalize", lambda _args, _assets: 7)
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    result = run_workload.coordinated_finalize(["workload"], tmp_path / "assets")

    assert result == 7
    assert (run_root / ".jet-coordination/1234/finalizer.leader").read_text() == "3\n"
    assert (run_root / ".jet-coordination/1234/finalizer.rc").read_text() == "7\n"


def test_jet_nonzero_rank_returns_published_finalizer_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    coordination = run_root / ".jet-coordination/5678"
    coordination.mkdir(parents=True)
    (coordination / "finalizer.rc").write_text("9\n")
    monkeypatch.setattr(run_workload.runtime, "parse_args", lambda _args: object())
    monkeypatch.setattr(run_workload.runtime, "configure", lambda _args: None)
    monkeypatch.setattr(run_workload.runtime, "require_prepared", lambda _args: {})
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(
        run_workload,
        "finalize",
        lambda _args, _assets: pytest.fail("nonzero rank must not finalize"),
    )
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_JOB_ID", "5678")

    assert run_workload.coordinated_finalize(["workload"], tmp_path / "assets") == 9


def test_jet_worker_synchronizes_inventory_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    results_dir = run_root / "logs"
    results_dir.mkdir(parents=True)
    args = object()
    monkeypatch.setattr(run_workload.runtime, "parse_args", lambda _args: args)
    monkeypatch.setattr(run_workload.runtime, "configure", lambda _args: None)
    monkeypatch.setattr(
        run_workload.runtime,
        "require_prepared",
        lambda _args: (_ for _ in ()).throw(RuntimeError("bad preparation")),
    )
    monkeypatch.setattr(
        slurm_job,
        "_worker",
        lambda _args: pytest.fail("worker must not run after inventory setup fails"),
    )
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "9012")
    monkeypatch.setattr(sys, "argv", ["run_workload.py", "--assets-dir", str(tmp_path), "--", "workload"])

    assert run_workload.main() == 1
    inventory = yaml.safe_load((results_dir / "jet_inventory_rank0.json").read_text())
    assert inventory == {"rank": 0, "error": "RuntimeError: bad preparation"}


def test_jet_failed_worker_does_not_wait_for_peer_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    results_dir = run_root / "logs"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "32")
    monkeypatch.setenv("SLURM_JOB_ID", "2468")

    assert run_workload._synchronize(17, {"rank": 5}) == 17
    assert (run_root / ".jet-coordination/2468/worker-5.rc").read_text() == "17\n"


def test_jet_successful_worker_propagates_existing_peer_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    results_dir = run_root / "logs"
    coordination = run_root / ".jet-coordination/8642"
    results_dir.mkdir(parents=True)
    coordination.mkdir(parents=True)
    (coordination / "worker-5.rc").write_text("23\n")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run_root)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results_dir)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "32")
    monkeypatch.setenv("SLURM_JOB_ID", "8642")

    assert run_workload._synchronize(0, {"rank": 0}) == 23


def test_trace_cursor_prints_complete_lines_until_the_job_finishes() -> None:
    cursor = submit_workload.TraceCursor(job_id=1)

    assert cursor.advance("first\nsec", final=False) == ["first"]
    assert cursor.advance("first\nsecond\nthi", final=False) == ["second"]
    assert cursor.advance("first\nsecond\nthi", final=False) == []
    assert cursor.advance("first\nsecond\nthird", final=True) == ["third"]
    assert cursor.done
    assert cursor.advance("first\nsecond\nthird", final=True) == []
    assert submit_workload.TraceCursor(job_id=2).advance("section_start:1:x\r\x1b[0Kbody\n", final=False) == [
        "section_start:1:x\r\x1b[0Kbody",
    ]


def test_jet_log_filter_mirrors_rank_zero_and_keeps_diagnostics() -> None:
    assert submit_workload.visible("0: iteration 10")
    assert submit_workload.visible("[jet] restoring native runtime")
    assert submit_workload.visible("section_start:1:step_script")
    assert not submit_workload.visible("7: iteration 10")
    assert not submit_workload.visible("\x1b[0K12: iteration 10")
    assert submit_workload.visible("12: Traceback (most recent call last):")
    assert submit_workload.visible("5: JET PERFORMANCE BASELINE FAILED (nonblocking exit code 3)")


class _FakeJetGitLab:
    """Replay successive GitLab views of one submitted JET pipeline."""

    def __init__(self, views: list[dict]) -> None:
        self.views = views
        self.index = 0
        self.fetches: list[int] = []
        self.sleeps: list[float] = []
        self.canceled: list[int] = []

    @property
    def view(self) -> dict:
        return self.views[self.index]

    def advance(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.index += 1

    def get_pipeline(self, _pipeline_id: int) -> dict:
        if self.view.get("error"):
            raise ConnectionError(self.view["error"])
        return {"status": self.view["pipeline"]}

    def get_pipeline_bridges(self, _pipeline_id: int) -> list[dict]:
        return [{"downstream_pipeline": {"id": 77}}] if self.view.get("jobs") is not None else [{}]

    def get_all_pipeline_jobs(self, child_id: int) -> list[dict]:
        assert child_id == 77
        return self.view["jobs"]

    def get_job_trace(self, job_id: int) -> str:
        self.fetches.append(job_id)
        return self.view["traces"][job_id]

    def cancel_pipeline(self, pipeline_id: int) -> None:
        self.canceled.append(pipeline_id)


def _jet_job(job_id: int, phase: int, label: str, status: str, key: str = "recipe/b200-native-perf-pico-1x8") -> dict:
    return {
        "id": job_id,
        "name": f"{key} b200_nsc_svg_slurm_1 00 [{phase} {label}]",
        "status": status,
        "web_url": f"https://gitlab.example/jobs/{job_id}",
    }


def test_follower_streams_each_attempt_completely_until_the_pipeline_finishes() -> None:
    other = _jet_job(9, 2, "b200_nsc_svg_slurm_1", "running", key="recipe/b200-native-detr-pico-1x8")
    banner = "1: JET PERFORMANCE BASELINE FAILED (nonblocking exit code 3)"
    first_attempt = f"0: setup\n3: noise\n{banner}\n0: train\n0: crashed"
    views = [
        {"pipeline": "running", "jobs": None},
        {
            "pipeline": "running",
            "jobs": [_jet_job(10, 1, "logs_before", "success"), _jet_job(11, 2, "b200_nsc_svg_slurm_1", "running"), other],
            "traces": {11: "0: setup\n3: noise\n"},
        },
        {
            "pipeline": "running",
            "jobs": [
                _jet_job(10, 1, "logs_before", "success"),
                _jet_job(11, 2, "b200_nsc_svg_slurm_1", "failed"),
                _jet_job(21, 2, "b200_nsc_svg_slurm_1", "running"),
                other,
            ],
            "traces": {11: first_attempt, 21: "0: retried setup\n"},
        },
        {
            "pipeline": "running",
            "jobs": [
                _jet_job(10, 1, "logs_before", "success"),
                _jet_job(21, 2, "b200_nsc_svg_slurm_1", "success"),
                other,
            ],
            "traces": {21: "0: retried setup\n0: done"},
        },
        {"error": "connection reset"},
        {
            "pipeline": "success",
            "jobs": [
                _jet_job(10, 1, "logs_before", "success"),
                _jet_job(21, 2, "b200_nsc_svg_slurm_1", "success"),
                _jet_job(22, 3, "logs_after", "success"),
                other,
            ],
            "traces": {},
        },
    ]
    gitlab = _FakeJetGitLab(views)
    out = io.StringIO()

    result = submit_workload.Follower(
        gitlab,
        5,
        "recipe/b200-native-perf-pico-1x8",
        out=out,
        sleep=gitlab.advance,
    ).run()

    text = out.getvalue()
    before_retry, after_retry = text.split("JET retried the workload; following https://gitlab.example/jobs/21")
    assert result.pipeline_status == "success"
    assert result.succeeded
    assert not result.soft_failure_banner
    assert [result.jobs[phase]["id"] for phase in sorted(result.jobs)] == [10, 21, 22]
    assert "0: setup\n" in before_retry
    assert "3: noise" not in text
    assert banner in before_retry
    assert "0: train\n0: crashed\n" in before_retry
    assert after_retry.index("0: retried setup\n") < after_retry.index("0: done\n")
    assert "polling JET failed (1/20); retrying: connection reset" in after_retry
    assert "recipe/b200-native-detr-pico-1x8" not in text
    assert gitlab.fetches == [11, 11, 21, 21]


def test_follower_reports_the_soft_failure_banner_of_the_final_attempt() -> None:
    gitlab = _FakeJetGitLab(
        [
            {
                "pipeline": "success",
                "jobs": [_jet_job(10, 1, "logs_before", "success"), _jet_job(11, 2, "b200_nsc_svg_slurm_1", "success")],
                "traces": {11: "4: JET PERFORMANCE BASELINE FAILED (nonblocking exit code 3)\n"},
            },
        ],
    )

    result = submit_workload.Follower(gitlab, 5, "recipe/b200-native-perf-pico-1x8", out=io.StringIO(), sleep=gitlab.advance).run()

    assert result.soft_failure_banner
    assert submit_workload.verdict(result, soft_failure=False, log_code="0") == 3


def test_follower_gives_up_after_the_poll_failure_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit_workload, "MAX_POLL_FAILURES", 2)
    gitlab = _FakeJetGitLab([{"error": "down"}] * 3)

    with pytest.raises(ConnectionError, match="down"):
        submit_workload.Follower(gitlab, 5, "build/b200-ci-image", out=io.StringIO(), sleep=gitlab.advance).run()


def test_follower_polls_large_traces_less_often() -> None:
    trace = "0: x\n" * 400_000
    gitlab = _FakeJetGitLab(
        [
            {"pipeline": "running", "jobs": [_jet_job(11, 2, "b200_nsc_svg_slurm_1", "running")], "traces": {11: trace}},
            {"pipeline": "success", "jobs": [_jet_job(11, 2, "b200_nsc_svg_slurm_1", "success")], "traces": {11: trace}},
        ],
    )

    submit_workload.Follower(gitlab, 5, "recipe/b200-native-perf-pico-1x8", out=io.StringIO(), sleep=gitlab.advance).run()

    assert gitlab.sleeps == [len(trace) * 60 / submit_workload.TRACE_BYTES_PER_MINUTE]
    assert gitlab.sleeps[0] > submit_workload.POLL_SECONDS


def test_follower_mirrors_only_diagnostics_past_the_log_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit_workload, "MIRROR_BUDGET_BYTES", 10)
    gitlab = _FakeJetGitLab(
        [
            {
                "pipeline": "success",
                "jobs": [_jet_job(11, 2, "b200_nsc_svg_slurm_1", "failed")],
                "traces": {11: "0: first line\n0: second line\n0: Traceback (most recent call last):\n0: last"},
            },
        ],
    )
    out = io.StringIO()

    submit_workload.Follower(gitlab, 5, "recipe/b200-native-perf-pico-1x8", out=out, sleep=gitlab.advance).run()

    text = out.getvalue()
    assert "0: first line\n" in text
    assert "0: second line" not in text
    assert "log budget reached; mirroring diagnostics only; full log: https://gitlab.example/jobs/11" in text
    assert "0: Traceback (most recent call last):\n" in text
    assert "0: last" not in text


def test_follower_reports_a_failed_submission_without_workload_jobs() -> None:
    gitlab = _FakeJetGitLab([{"pipeline": "failed", "jobs": None}])

    result = submit_workload.Follower(gitlab, 5, "build/b200-ci-image", out=io.StringIO(), sleep=gitlab.advance).run()

    assert result.pipeline_status == "failed"
    assert not result.jobs
    assert not result.succeeded


def test_submission_manifests_run_one_workload_on_the_built_image() -> None:
    platform = workload_matrix.jet_platform("b200")
    image = "gitlab-master.nvidia.com:5005/dl/jet/ci/nvrx-ci-b200:main_1"

    key, documents = submit_workload.manifests(platform, maintainer="ankurv", workload_key=None, image_tag=None)
    assert key == "build/b200-ci-image"
    assert [document["type"] for document in documents] == ["build"]

    key, documents = submit_workload.manifests(
        platform,
        maintainer="ankurv",
        workload_key="b200-native-perf-pico-1x8",
        image_tag=image,
    )
    assert key == "recipe/b200-native-perf-pico-1x8"
    assert len(documents) == 1
    assert documents[0]["key_pattern"] == "b200-native-perf-pico-1x8"
    assert documents[0]["spec"]["image_source"] == {"image_tag": image}
    with pytest.raises(ValueError, match="no built image"):
        submit_workload.manifests(platform, maintainer="ankurv", workload_key="b200-native-perf-pico-1x8", image_tag=None)
    with pytest.raises(ValueError, match="has no JET workload"):
        submit_workload.manifests(platform, maintainer="ankurv", workload_key="b200-unknown", image_tag=image)
    assert submit_workload.image_variable(platform) == "JET_IMAGE_TAG_B200"


def test_submission_variables_forward_the_exact_wheels() -> None:
    variables = submit_workload.submission_variables(
        {
            "UPSTREAM_WHEEL_X86_64_ARTIFACT_ID": "11",
            "UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID": "12",
            "UPSTREAM_WHEEL_AARCH64_ARTIFACT_ID": "",
            "UNRELATED": "x",
        },
    )

    assert variables == {
        "UPSTREAM_WHEEL_X86_64_ARTIFACT_ID": "11",
        "UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID": "12",
        "JET_API_TAG": "v3",
        "JET_SCRIPT_GENERATION_VALIDATION": "error",
    }


def test_submission_variables_describe_the_github_run() -> None:
    variables = submit_workload.submission_variables(
        {
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_REPOSITORY_ID": "777",
            "GITHUB_RUN_ID": "40586524912",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": "c" * 40,
            "GITHUB_HEAD_REF": "feature",
            "GITHUB_REF_NAME": "12/merge",
            "UPSTREAM_WHEEL_AARCH64_ARTIFACT_ID": "13",
        },
    )

    assert variables == {
        "UPSTREAM_PROJECT_URL": "https://github.com/owner/repo",
        "UPSTREAM_PROJECT_ID": "777",
        "UPSTREAM_PROJECT_PATH": "owner/repo",
        "UPSTREAM_PIPELINE_ID": "40586524912",
        "UPSTREAM_PIPELINE_SOURCE": "pull_request",
        "UPSTREAM_COMMIT_SHA": "c" * 40,
        "UPSTREAM_COMMIT_REF_NAME": "feature",
        "UPSTREAM_WHEEL_AARCH64_ARTIFACT_ID": "13",
        "JET_API_TAG": "v3",
        "JET_SCRIPT_GENERATION_VALIDATION": "error",
    }


def test_verdict_mirrors_the_regular_job_exit_codes() -> None:
    ok = {"status": "success"}
    passed = submit_workload.FollowResult("success", {1: ok, 2: ok, 3: ok})
    failed = submit_workload.FollowResult("failed", {1: ok, 2: {"status": "failed"}})

    assert submit_workload.verdict(passed, soft_failure=False, log_code="0") == 0
    assert submit_workload.verdict(passed, soft_failure=False, log_code=None) == 0
    assert submit_workload.verdict(passed, soft_failure=True, log_code="0") == 3
    assert submit_workload.verdict(passed, soft_failure=False, log_code="1.1.1") == 1
    assert submit_workload.verdict(failed, soft_failure=True, log_code="0") == 1
    assert submit_workload.verdict(submit_workload.FollowResult("failed"), soft_failure=False, log_code=None) == 1
    assert submit_workload.verdict(submit_workload.FollowResult("canceled", {1: ok, 3: ok}), soft_failure=False, log_code=None) == 1
    assert submit_workload.verdict(submit_workload.FollowResult("failed", {1: ok, 2: ok}), soft_failure=False, log_code=None) == 1
    assert submit_workload.verdict(submit_workload.FollowResult("success", {1: ok, 3: ok}), soft_failure=False, log_code=None) == 1


class _FakeAsset:
    def __init__(self, name: str, source: Path) -> None:
        self.name = name
        self.source = source

    def download(self, path: Path) -> None:
        path.write_bytes(self.source.read_bytes())


class _FakeJetLog:
    def __init__(self, log: dict, assets: list | None = None) -> None:
        self.log = log
        self.assets = assets or []

    def get_assets(self) -> list:
        return self.assets


def test_publish_results_extracts_the_workload_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive-root"
    (root / "ci" / "ci-results").mkdir(parents=True)
    (root / "ci" / "ci-results" / "b200_report.json").write_text("{}")
    (root / "ci" / "ci-results" / "b200_soft_failure.json").write_text("{}")
    archive = tmp_path / "b200_pipeline-5_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "ci", arcname="ci")
    log = _FakeJetLog({}, [_FakeAsset("gitlab_out.log", archive), _FakeAsset(archive.name, archive)])
    destination = tmp_path / "jet-results"

    assert submit_workload.publish_results(log, destination)
    assert (destination / "ci" / "ci-results" / "b200_report.json").is_file()
    assert not (destination / archive.name).exists()
    assert not submit_workload.publish_results(_FakeJetLog({}), tmp_path / "empty")

    (root / "ci" / "ci-results" / "b200_soft_failure.json").unlink()
    clean = tmp_path / "clean_pipeline-6_artifacts.tar.gz"
    with tarfile.open(clean, "w:gz") as output:
        output.add(root / "ci", arcname="ci")
    assert not submit_workload.publish_results(_FakeJetLog({}, [_FakeAsset(clean.name, clean)]), tmp_path / "clean")
    assert (tmp_path / "clean" / "ci" / "ci-results" / "b200_report.json").is_file()


def test_publish_results_refuses_archive_members_outside_the_results(tmp_path: Path) -> None:
    (tmp_path / "escape.txt").write_text("outside")
    archive = tmp_path / "evil_pipeline-7_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(tmp_path / "escape.txt", arcname="../escaped.txt")
    destination = tmp_path / "results"

    assert not submit_workload.publish_results(_FakeJetLog({}, [_FakeAsset("../../" + archive.name, archive)]), destination)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path.parent / archive.name).exists()
    assert list(destination.iterdir()) == []


def test_workload_log_matches_the_followed_attempt_and_is_described() -> None:
    early = _FakeJetLog({"workload": {"key": "recipe/a"}, "created": submit_workload.EARLIEST, "id": "old", "ci": {"job_id": 11}})
    late = _FakeJetLog(
        {
            "workload": {"key": "recipe/a"},
            "created": submit_workload.EARLIEST.replace(year=2026),
            "id": "new",
            "ci": {"job_id": 21},
            "status": {"code": "1.1.1", "name": "error > workload > python", "message": "training failed"},
            "slurm": {"job": 42, "queue_time": 3725.0, "duration": 61.0, "state": "FAILED"},
        },
    )
    other = _FakeJetLog({"workload": {"key": "recipe/b"}, "id": "other", "ci": {"job_id": 21}})

    assert submit_workload.workload_log([early, other, late], "recipe/a", 11) is early
    assert submit_workload.workload_log([early, other], "recipe/a", 21) is None
    log = submit_workload.workload_log([early, other, late], "recipe/a", None)

    assert log is late
    assert submit_workload.workload_log([other], "recipe/a", None) is None
    lines = submit_workload.describe(log)
    assert lines[0] == "[jet] JET status: error > workload > python (1.1.1) training failed"
    assert lines[1] == "[jet] JET log: https://jet.nvidia.com/ui/logs/new"
    assert lines[2] == "[jet] Slurm job 42: queued 1h02m05s, ran 0h01m01s, state FAILED"


class _FakeJetPipeline:
    def __init__(self, jet_id: int) -> None:
        self.jet_id = jet_id
        self.generation_details = SimpleNamespace(
            job_url="https://gitlab.example/generate",
            job_trace="generate: invalid manifest",
        )


class _FakeJetClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[list[dict], dict]] = []
        self.pipeline = _FakeJetPipeline(555)
        self.workloads = self

    def submit(self, documents: list[dict], **kwargs: object) -> _FakeJetPipeline:
        self.submissions.append((documents, kwargs))
        return self.pipeline

    def get_pipeline(self, pipeline_id: int) -> _FakeJetPipeline:
        assert pipeline_id == self.pipeline.jet_id
        return self.pipeline


def _submit_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: object,
    log: object | None,
    on_run: object = None,
) -> tuple[_FakeJetClient, _FakeJetGitLab, dict]:
    monkeypatch.chdir(tmp_path)
    client = _FakeJetClient()
    gitlab = _FakeJetGitLab([])
    handlers: dict = {}
    monkeypatch.setattr(submit_workload, "_client", lambda: client)
    monkeypatch.setattr(submit_workload, "_gitlab", lambda _client: gitlab)
    monkeypatch.setattr(submit_workload.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(submit_workload, "wait_for_log", lambda _pipeline, _key, _job_id: log)

    class FakeFollower:
        def __init__(self, follower_gitlab: object, pipeline_id: int, key: str) -> None:
            assert follower_gitlab is gitlab
            assert pipeline_id == client.pipeline.jet_id
            self.key = key

        def run(self) -> object:
            assert (tmp_path / ".jet-pipeline").read_text() == "555\n"
            if on_run is not None:
                on_run(handlers)
            return result

    monkeypatch.setattr(submit_workload, "Follower", FakeFollower)
    for name in ("JET_CI_MAINTAINER", "JET_CI_DRY_RUN", "JET_IMAGE_TAG_B200"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITLAB_USER_LOGIN", "ankurv")
    monkeypatch.setenv("UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID", "4242")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    return client, gitlab, handlers


_ALL_PASSED = {1: {"status": "success", "id": 10}, 2: {"status": "success", "id": 11}, 3: {"status": "success", "id": 12}}


def test_image_job_exports_the_built_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = "gitlab-master.nvidia.com:5005/dl/jet/ci/nvrx-ci-b200:main_555"
    log = _FakeJetLog({"id": "x", "status": {"code": "0", "name": "success"}, "output_image": {"name": image}})
    client, _gitlab, handlers = _submit_harness(
        monkeypatch,
        tmp_path,
        submit_workload.FollowResult("success", _ALL_PASSED),
        log,
    )

    assert submit_workload.main(["--platform", "b200", "--image"]) == 0

    assert (tmp_path / "jet-image.env").read_text() == f"JET_IMAGE_TAG_B200={image}\n"
    assert not (tmp_path / ".jet-pipeline").exists()
    documents, kwargs = client.submissions[0]
    assert [document["type"] for document in documents] == ["build"]
    assert documents[0]["maintainers"] == ["ankurv"]
    assert kwargs["config_id"] == "dlalgo-ci/nsc-svg-slurm-1"
    assert (tmp_path / "summary.md").read_text() == "`build/b200-ci-image` was submitted to JET config `dlalgo-ci/nsc-svg-slurm-1`.\n"
    assert kwargs["dry_run"] is False
    assert kwargs["wait_for_validation"] is False
    assert kwargs["custom_config"]["retrier"]["default_retry_on"] == ["1.2", "1.2.*"]
    assert kwargs["additional_variables"]["UPSTREAM_WHEEL_VALGRIND_X86_64_ARTIFACT_ID"] == "4242"
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}


@pytest.mark.parametrize(
    ("log", "result"),
    [
        (None, submit_workload.FollowResult("success", _ALL_PASSED)),
        (_FakeJetLog({"id": "x", "status": {"code": "0"}, "output_image": None}), submit_workload.FollowResult("success", _ALL_PASSED)),
        (
            _FakeJetLog({"id": "x", "status": {"code": "1.1.3"}}),
            submit_workload.FollowResult("failed", {2: {"status": "failed", "id": 11}}),
        ),
    ],
)
def test_image_job_fails_without_a_built_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log: object | None,
    result: object,
) -> None:
    _submit_harness(monkeypatch, tmp_path, result, log)

    assert submit_workload.main(["--platform", "b200", "--image"]) == 1
    assert not (tmp_path / "jet-image.env").exists()


def _results_log(tmp_path: Path, *, soft_failure: bool, code: str = "0") -> _FakeJetLog:
    root = tmp_path / "workload-root"
    (root / "ci" / "ci-results").mkdir(parents=True)
    (root / "ci" / "ci-results" / "b200_report.json").write_text("{}")
    if soft_failure:
        (root / "ci" / "ci-results" / "b200_soft_failure.json").write_text("{}")
    archive = tmp_path / "b200_pipeline-555_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "ci", arcname="ci")
    return _FakeJetLog({"id": "w", "status": {"code": code}}, [_FakeAsset(archive.name, archive)])


@pytest.mark.parametrize(
    ("pipeline_status", "soft_failure", "code", "expected"),
    [("success", False, "0", 0), ("success", True, "0", 3), ("success", False, "1.1.1", 1), ("failed", False, "1.1.1", 1)],
)
def test_workload_job_exits_like_the_regular_job(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline_status: str,
    soft_failure: bool,
    code: str,
    expected: int,
) -> None:
    image = "gitlab-master.nvidia.com:5005/dl/jet/ci/nvrx-ci-b200:main_554"
    jobs = dict(_ALL_PASSED) if pipeline_status == "success" else {2: {"status": "failed", "id": 11}}
    client, _gitlab, _handlers = _submit_harness(
        monkeypatch,
        tmp_path,
        submit_workload.FollowResult(pipeline_status, jobs),
        _results_log(tmp_path, soft_failure=soft_failure, code=code),
    )
    monkeypatch.setenv("JET_IMAGE_TAG_B200", image)
    monkeypatch.setenv("JET_CI_MAINTAINER", "jdoe")

    assert submit_workload.main(["--platform", "b200", "--workload", "b200-native-perf-pico-1x8"]) == expected

    documents, _kwargs = client.submissions[0]
    assert documents[0]["maintainers"] == ["jdoe"]
    assert documents[0]["spec"]["image_source"] == {"image_tag": image}
    assert (tmp_path / "jet-results" / "ci" / "ci-results" / "b200_report.json").is_file()
    assert not (tmp_path / "jet-image.env").exists()


def test_dry_run_hands_a_placeholder_image_to_the_workloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _gitlab, _handlers = _submit_harness(monkeypatch, tmp_path, submit_workload.FollowResult("success"), None)
    monkeypatch.setenv("JET_CI_DRY_RUN", "true")

    assert submit_workload.main(["--platform", "b200", "--image"]) == 0
    placeholder = submit_workload.dry_run_image(workload_matrix.jet_platform("b200"))
    assert f"JET_IMAGE_TAG_B200={placeholder}\n" in (tmp_path / "jet-image.env").read_text()
    monkeypatch.setenv("JET_IMAGE_TAG_B200", placeholder)
    assert submit_workload.main(["--platform", "b200", "--workload", "b200-native-perf-pico-1x8"]) == 0
    assert [kwargs["dry_run"] for _documents, kwargs in client.submissions] == [True, True]


def test_failed_submission_prints_the_generate_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _submit_harness(monkeypatch, tmp_path, submit_workload.FollowResult("failed"), None)

    assert submit_workload.main(["--platform", "b200", "--image"]) == 1
    output = capsys.readouterr().out
    assert "[jet] JET generate job: https://gitlab.example/generate" in output
    assert "generate: invalid manifest" in output


def test_termination_cancels_the_submitted_jet_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def terminate(handlers: dict) -> None:
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    _client, gitlab, _handlers = _submit_harness(
        monkeypatch,
        tmp_path,
        submit_workload.FollowResult("success", _ALL_PASSED),
        None,
        on_run=terminate,
    )

    with pytest.raises(SystemExit) as exit_info:
        submit_workload.main(["--platform", "b200", "--image"])
    assert exit_info.value.code == 1
    assert gitlab.canceled == [555]


def test_cancel_command_cancels_one_jet_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    gitlab = _FakeJetGitLab([])
    monkeypatch.setattr(submit_workload, "_client", lambda: object())
    monkeypatch.setattr(submit_workload, "_gitlab", lambda _client: gitlab)

    assert submit_workload.main(["--cancel", "7"]) == 0
    assert gitlab.canceled == [7]
    with pytest.raises(SystemExit) as exit_info:
        submit_workload.main(["--image"])
    assert exit_info.value.code == 2
