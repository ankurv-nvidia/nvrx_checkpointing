#!/usr/bin/env python3
"""CI runner for the C++ checkpoint benchmark (ckptbench).

Build and run the native checkpoint benchmark (1 node). Inline by default;
``--slurm`` launches it through srun. Parses perf metrics from the run log and
gates on the per-platform baseline.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from run_job import (  # companion script, shares run_job's build helpers
    BASELINES_DIR,
    CKPTBENCH_REGISTRY,
    MEGATRON_ROOT,
    RESULTS_DIR,
    SOFT_FAIL_RC,
    CheckpointServer,
    PerfReport,
    artifact_stem,
    build_srun_cmd,
    detect_local_gpu_count,
    ensure_vcpkg_root,
    image_path,
    in_srun,
    platform_tag,
    runner_gpus_per_node,
    runner_uses_slurm,
    tee_run,
)

logger = logging.getLogger("native_ci")


def _ckptbench_binary() -> Path:
    import megatron.native as _native  # noqa: F401, PLC0415 — triggers CMake build

    native_dir = MEGATRON_ROOT / "megatron/native"
    build_dir = (
        Path(os.environ["MEGATRON_NATIVE_BUILD_DIR"]).resolve() if os.environ.get("MEGATRON_NATIVE_BUILD_DIR") else native_dir / "build"
    )
    binary = build_dir / "ckptbench"
    if not binary.is_file():
        raise FileNotFoundError(f"ckptbench not found at {binary}")
    return binary


def run_ckptbench(args: argparse.Namespace) -> None:
    registry = CKPTBENCH_REGISTRY
    if not registry.is_file():
        raise FileNotFoundError(f"stateregistry not found: {registry}")

    if in_srun():
        binary = _ckptbench_binary()
        bench_cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]
        subprocess.run(bench_cmd, check=True)
        return

    if args.slurm:
        image = image_path().as_posix()
        cmd = [
            *build_srun_cmd(
                nodes=1,
                ntasks=1,
                time_minutes=10,
                job_name=f"ckpt_benchmark_{os.environ.get('CI_PIPELINE_ID', 'local')}",
                image=image,
                # container-env forwarded by name — avoids KEY=VALUE ambiguity in pyxis/enroot
                container_env="VCPKG_ROOT",
                gpus_per_node=args.gpus_per_node,
                total_gpus=args.gpus_per_node,  # single node
            ),
            "python3",
            "-u",
            str(Path(__file__).resolve()),
        ]
    else:
        binary = _ckptbench_binary()
        cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]

    _run_and_report_perf(cmd, "ckptbench", f"1x{args.gpus_per_node}", args)


def _run_and_report_perf(cmd: list[str], job: str, config: str, args: argparse.Namespace, env: dict | None = None) -> None:
    """Tee a non-pretrain job (ckptbench), parse its perf metrics, gate on the baseline."""
    stem = artifact_stem(platform_tag(), job, args.flavor_tag, config)
    log_file = RESULTS_DIR / f"{stem}.log"
    if args.skip_perf:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(cmd, env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Job failed with exit code {rc}")
        return
    rc = tee_run(cmd, log_file, env)
    passed = PerfReport.parse_and_report(
        job,
        config,
        args.flavor_tag,
        log_file,
        BASELINES_DIR / f"{stem}.yaml",
        RESULTS_DIR / f"{stem}_report.json",
        args.update_baseline,
        args.gpus_per_node,
    )
    if not passed or rc == SOFT_FAIL_RC:
        sys.exit(SOFT_FAIL_RC)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(shlex.join(sys.argv))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        dest="gpus_per_node",
        help="GPUs per node (default: GPUS_PER_NODE env set at runner setup, else auto-detect)",
    )
    p.add_argument("--slurm", action="store_true", help="Submit via SLURM even for single-node runs")
    p.add_argument("--skip-perf", action="store_true", dest="skip_perf", help="Skip perf parsing; tee full output to console instead")
    p.add_argument(
        "--update-baseline",
        action="store_true",
        dest="update_baseline",
        help="Record current measurements as the new baseline and exit",
    )
    args, extra_args = p.parse_known_args()
    args.extra_args = extra_args
    args.flavor_tag = ""
    ensure_vcpkg_root()  # before any `import megatron.native` triggers the CMake build
    # Honor the runner's configured scheduler mode (SLURM_PARTITION set => SLURM).
    if runner_uses_slurm():
        args.slurm = True
    if args.gpus_per_node is None:
        # Prefer the count configured at runner setup (GPUS_PER_NODE), so the CI
        # YAML stays cluster-agnostic; fall back to probing the local host.
        args.gpus_per_node = runner_gpus_per_node() or detect_local_gpu_count()

    # Fail fast if a prior job left a checkpoint_server bound to the port. Only
    # meaningful on compute nodes — the slurm launcher runs on the head node,
    # where no server runs. The benchmark's own server exits with the process
    # (--inproc).
    if in_srun() or not args.slurm:
        CheckpointServer.require_no_checkpoint_server()
    run_ckptbench(args)


if __name__ == "__main__":
    main()
