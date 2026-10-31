# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: E402, INP001, SLF001
"""Shared setup for repository workers inside a JET Slurm allocation."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import install_run_dependency
import slurm_job
import wheel_install

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger("native_ci")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse a normal slurm_job invocation without selecting submission mode."""
    parser = slurm_job._parser()
    args, args.extra_args = slurm_job._parse_args(parser, argv)
    try:
        args.flavors = slurm_job.FlavorSelection(args.flavor)
    except ValueError as error:
        parser.error(str(error))
    slurm_job._validate_args(parser, args)
    return args


def configure(args: argparse.Namespace) -> None:
    """Resolve the shared run paths and Python target used by every rank."""
    slurm_job._configure_run(args)
    slurm_job.ensure_vcpkg_root()


def _reset_stale_outputs(args: argparse.Namespace) -> None:
    """Clean only this pipeline/workload before JET starts its GPU step."""
    if not args.clean_run:
        return
    pipeline_id = os.environ.get("UPSTREAM_PIPELINE_ID", "").strip()
    run_name = os.environ.get("SLURMJOB_OPT_RUN_NAME", "").strip()
    work_value = os.environ.get("SLURMJOB_OPT_WORK_DIR", "").strip()
    artifact_value = os.environ.get("CI_ARTIFACT_ROOT", "").strip()
    candidate_value = os.environ.get("CI_BASELINE_CANDIDATE_DIR", "").strip()
    if not all((pipeline_id, run_name, work_value, artifact_value, candidate_value)):
        raise RuntimeError("JET preparation cleanup requires exact run and artifact identifiers")
    if slurm_job.RUN_ROOT is None or slurm_job.RESULTS_DIR is None:
        raise RuntimeError("JET run directory is not configured")

    run = slurm_job.RUN_ROOT
    work = Path(work_value)
    artifact = Path(artifact_value)
    candidate = Path(candidate_value)
    resolved_run = run.resolve()
    resolved_work = work.resolve()
    resolved_artifact = artifact.resolve()
    expected_candidate = artifact / "ci" / "baselines"
    if (
        not pipeline_id.isdecimal()
        or not run.is_absolute()
        or not work.is_absolute()
        or not artifact.is_absolute()
        or resolved_run == Path("/")
        or resolved_work == Path("/")
        or resolved_artifact == Path("/")
        or run != resolved_run
        or work != resolved_work
        or artifact != resolved_artifact
        or run.is_symlink()
        or work.is_symlink()
        or artifact.is_symlink()
        or run.name != run_name
        or run.parent != work
        or work.name != pipeline_id
        or artifact.name != pipeline_id
        or artifact.parent.name != "nvrx-ci-artifacts"
        or candidate != expected_candidate
    ):
        raise RuntimeError(
            f"refusing unsafe JET preparation cleanup: run={run}, work={work}, artifact={artifact}",
        )

    for path in (run, artifact):
        if path.exists():
            shutil.rmtree(path)
            logger.info("[jet] removed stale pipeline-scoped output: %s", path)
    run.mkdir(parents=True)
    slurm_job.RESULTS_DIR.mkdir(parents=True)


def wheel_variant(args: argparse.Namespace) -> str:
    """Select the native binary variant required by a workload."""
    return "valgrind" if "vg" in args.flavors.names else "normal"


def _verify_architecture() -> None:
    expected = os.environ.get("WHEEL_ARCH", "").strip()
    actual = platform.machine()
    if expected and actual != expected:
        raise RuntimeError(f"JET image architecture mismatch: expected {expected}, found {actual}")


def _install_wheel(args: argparse.Namespace) -> None:
    variant = wheel_variant(args)
    os.environ["WHEEL_VARIANT"] = variant
    root = Path(os.environ.get("JET_PREBUILT_WHEEL_DIR", "/opt/nvrx-wheels"))
    variant_root = root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    if not any(variant_root.glob("*.whl")):
        wheel_install.fetch_upstream_wheel(variant_root)
    wheel_install.require_prebuilt_wheel(variant_root, probe=False)
    wheel_install.report_install()


def _install_nvrx(args: argparse.Namespace) -> None:
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    target = slurm_job._nvrx_target(cfg, args.nvrx_commit)
    if target is None:
        return
    install_run_dependency.install_nvrx(
        args.nvrx_repo_url,
        args.nvrx_commit,
        target,
        slurm_job.nvrx_repo_root(),
    )


def prepare(args: argparse.Namespace) -> Path:
    """Install exact inputs once into the allocation's shared run directory."""
    configure(args)
    # Single-task setup resets the exact workload paths before GPU ranks start.
    _reset_stale_outputs(args)
    _verify_architecture()
    _install_wheel(args)
    _install_nvrx(args)
    if slurm_job.RUN_ROOT is None:
        raise RuntimeError("run directory is not configured")
    for marker in slurm_job.RUN_ROOT.glob(".jet-worker-*.rc"):
        marker.unlink()
    for name in (".jet-final.rc", ".jet-finalizer.rc", ".jet-workers.rc"):
        (slurm_job.RUN_ROOT / name).unlink(missing_ok=True)
    state = {
        "platform": slurm_job.platform_tag(),
        "architecture": platform.machine(),
        "source_revision": wheel_install.source_revision(),
        "wheel_variant": wheel_variant(args),
    }
    prepared = slurm_job.RUN_ROOT / ".jet-prepared.json"
    prepared.write_text(json.dumps(state, indent=2))
    logger.info("[jet] prepared %s", prepared)
    subprocess.run([sys.executable, "-m", "pip", "list"], check=True)
    return prepared


def require_prepared(args: argparse.Namespace) -> dict:
    """Validate the preparation marker before a GPU rank starts."""
    if slurm_job.RUN_ROOT is None:
        raise RuntimeError("run directory is not configured")
    prepared = slurm_job.RUN_ROOT / ".jet-prepared.json"
    if not prepared.is_file():
        raise RuntimeError(f"JET batch setup did not produce {prepared}")
    state = json.loads(prepared.read_text())
    expected = {
        "platform": slurm_job.platform_tag(),
        "architecture": platform.machine(),
        "source_revision": wheel_install.source_revision(),
        "wheel_variant": wheel_variant(args),
    }
    if state != expected:
        raise RuntimeError(f"JET preparation state mismatch: expected {expected}, found {state}")
    return state


def inventory(args: argparse.Namespace) -> dict:
    """Record the rank-visible GPU and Slurm topology."""
    expected_world = args.nodes * args.gpus_per_node
    world = int(os.environ.get("SLURM_NTASKS", "0"))
    if world != expected_world:
        raise RuntimeError(f"JET task topology mismatch: expected {expected_world} tasks, found {world}")
    allocated_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", str(args.nodes)))
    if allocated_nodes != args.nodes:
        raise RuntimeError(f"JET node topology mismatch: expected {args.nodes} nodes, found {allocated_nodes}")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is not installed or not found in PATH")
    gpu_query = subprocess.run(
        [nvidia_smi, "--query-gpu=name,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_query.returncode:
        raise RuntimeError(f"nvidia-smi failed: {gpu_query.stderr.strip()}")
    return {
        "rank": int(os.environ["SLURM_PROCID"]),
        "local_rank": int(os.environ["SLURM_LOCALID"]),
        "world_size": world,
        "nodes": allocated_nodes,
        "platform": slurm_job.platform_tag(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "single_gpu_per_task": os.environ.get("NVRX_SINGLE_GPU_PER_TASK") == "1",
        "visible_gpus": gpu_query.stdout.splitlines(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }
