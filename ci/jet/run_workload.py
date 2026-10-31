#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: BLE001, SLF001
"""Run and finalize one repository workload inside a JET Slurm step."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import TextIO

import runtime
import yaml

slurm_job = runtime.slurm_job

_ARGUMENTS_RE = re.compile(
    r"-+ arguments -+\n.*?-+ end of arguments -+",
    re.DOTALL,
)
_VOLATILE_ARGUMENT_RE = re.compile(r"^  (?:data_cache_path|load|save|tensorboard_dir) ")


class Tee(io.TextIOBase):
    """Write rank output to both JET's stream and a repository artifact."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def _exit_code(error: BaseException) -> int:
    if isinstance(error, SystemExit) and isinstance(error.code, int):
        return error.code
    return 1


def _wait_for_count(directory: Path, pattern: str, count: int, timeout_seconds: int = 3600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while len(list(directory.glob(pattern))) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {count} files matching {directory / pattern}")
        time.sleep(0.25)


def _coordination_root() -> Path:
    run_root = slurm_job.RUN_ROOT
    if run_root is None:
        raise RuntimeError("run directory is not configured")
    allocation_id = os.environ.get("SLURM_JOB_ID")
    if not allocation_id:
        raise RuntimeError("SLURM_JOB_ID is required for JET rank coordination")
    root = run_root / ".jet-coordination" / allocation_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coordinate_server_preflight(args: argparse.Namespace) -> None:
    """Prove each node is clean before any rank is allowed to start a server."""
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    coordination = _coordination_root()
    node = os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME", f"rank-{rank}"))
    safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", node)
    marker = coordination / f"server-preflight-{safe_node}.json"
    if local_rank == 0:
        result: dict[str, object] = {"node": node, "rank": rank, "exit_code": 0}
        try:
            slurm_job.CHECKPOINT_SERVER.terminate()
        except BaseException as error:
            result.update(
                {
                    "exit_code": _exit_code(error),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(marker)
    _wait_for_count(coordination, "server-preflight-*.json", args.nodes)
    results = [json.loads(path.read_text()) for path in coordination.glob("server-preflight-*.json")]
    if failures := [result for result in results if result["exit_code"]]:
        raise RuntimeError(f"JET checkpoint-server preflight failed: {failures}")
    # Worker ranks consume this completed node preflight.
    os.environ["SLURMJOB_OPT_SERVER_PREFLIGHT_DONE"] = "1"


def _combined_log(
    args: argparse.Namespace,
    artifact_root: Path,
    rank_logs: list[Path],
    assets_dir: Path | None = None,
) -> tuple[Path, str]:
    cfg, flavor, stem = slurm_job._identity(args)
    del flavor
    result_dir = artifact_root / "ci" / "ci-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    # Performance reports consume daemon logs and collected OTLP data beside the combined trainer log.
    for daemon_log in slurm_job.RESULTS_DIR.glob("checkpoint_server.*.log"):
        shutil.copy2(daemon_log, result_dir / daemon_log.name)
    if (otel_dir := slurm_job.RESULTS_DIR / "otel").is_dir():
        shutil.copytree(otel_dir, result_dir / "otel", dirs_exist_ok=True)
    log = result_dir / f"{stem}.log"
    jet_output = assets_dir / "jet_assets" / "output_logs" / "output_script-0.log" if assets_dir else None
    if jet_output and jet_output.is_file():
        shutil.copy2(jet_output, log)
        return log, stem
    routine_logs = []
    run_root = slurm_job.RUN_ROOT or slurm_job.RESULTS_DIR.parent
    for spec in slurm_job._run_specs(args):
        pattern = f"**/{cfg}_{spec.name or 'run'}-routine_rank0.log"
        routine_logs.extend(sorted(run_root.glob(pattern)))
    if routine_logs:
        with log.open("w") as output:
            for routine_log in routine_logs:
                text = routine_log.read_text(errors="replace")
                output.write(text)
                if text and not text.endswith("\n"):
                    output.write("\n")
        return log, stem
    workload_log = slurm_job.RESULTS_DIR / f"{stem}.log"
    if workload_log.is_file():
        shutil.copy2(workload_log, log)
        return log, stem
    with log.open("w") as output:
        for rank, rank_log in enumerate(rank_logs):
            output.write(f"===== JET rank {rank}: {rank_log.name} =====\n")
            output.write(rank_log.read_text(errors="replace"))
            output.write("\n")
    return log, stem


def _write_arguments_candidate(log: Path, stem: str) -> None:
    candidate_dir_value = os.environ.get("CI_BASELINE_CANDIDATE_DIR")
    if not candidate_dir_value:
        return
    matches = _ARGUMENTS_RE.findall(log.read_text(errors="replace"))
    if not matches:
        return
    candidate_dir = Path(candidate_dir_value)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    normalized = "\n".join(
        line for line in matches[-1].rstrip().splitlines() if not _VOLATILE_ARGUMENT_RE.match(line)
    )
    (candidate_dir / f"{stem}_arguments.txt").write_text(f"{normalized}\n")


def _write_baseline_candidate(args: argparse.Namespace, log: Path, stem: str) -> None:
    candidate_dir_value = os.environ.get("CI_BASELINE_CANDIDATE_DIR")
    report_path = log.with_name(f"{stem}_report.json")
    if not candidate_dir_value or not report_path.is_file():
        return
    metrics = json.loads(report_path.read_text()).get("metrics", {})
    candidate_metrics = {}
    for name, metric in metrics.items():
        if name in {"bw_mbps", "bw_socket_imbalance"}:
            continue
        statistic = slurm_job.PerfReport._statistic(name, metric)
        value = metric[statistic]
        candidate_metrics[name] = round(value, 2) if name.endswith("_pct") else round(value)
    if not candidate_metrics:
        return
    candidate_dir = Path(candidate_dir_value)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline_stem = slurm_job._performance_baseline_stem(args)
    baseline = slurm_job.PerfReport.baseline(baseline_stem)
    candidate = {
        "description": "auto-generated by CI",
        "metrics": candidate_metrics,
        **({"tolerances": baseline["tolerances"]} if baseline.get("tolerances") else {}),
    }
    (candidate_dir / f"{baseline_stem}.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    flavors = slurm_job._selected_flavors(args)
    bandwidth = metrics.get("bw_mbps", {}).get("median")
    if (
        os.environ.get("JET_BASELINE_BOOTSTRAP") == "1"
        and "perf" in flavors.names
        and slurm_job._provider(flavors) == "native"
        and args.nodes == 1
        and bandwidth is not None
    ):
        bandwidths = slurm_job._load_yaml(slurm_job.BANDWIDTH_BASELINE_FILE)
        bandwidths[slurm_job.platform_tag()] = round(bandwidth)
        (candidate_dir / slurm_job.BANDWIDTH_BASELINE_FILE.name).write_text(
            yaml.safe_dump(bandwidths, sort_keys=False),
        )


def _run_epilog(args: argparse.Namespace, log: Path) -> int:
    common_args = [
        "--nodes",
        str(args.nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--config",
        args.config,
        "--flavor",
        "+".join(args.flavors.names),
    ]
    if getattr(args, "checkpoint_interval", None) is not None:
        common_args.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
    if getattr(args, "checkpoints_per_run", None) is not None:
        common_args.extend(["--checkpoints-per-run", str(args.checkpoints_per_run)])
    if getattr(args, "repeat", None) is not None:
        common_args.extend(["--repeat", str(args.repeat)])
    _cfg, _flavor, stem = slurm_job._identity(args)
    epilog_args = [
        *common_args,
        "--log-file",
        str(log),
        "--run-root",
        str(slurm_job.RUN_ROOT),
        "--baseline-json",
        json.dumps(slurm_job.PerfReport.baseline(slurm_job._performance_baseline_stem(args)), separators=(",", ":")),
        *(token for run in (getattr(args, "run", None) or ()) for token in ("--run", run)),
    ]
    try:
        slurm_job._epilog(epilog_args)
    except BaseException as error:
        exit_code = _exit_code(error)
        if exit_code == slurm_job.SOFT_FAIL_RC:
            marker = log.with_name(f"{stem}_soft_failure.json")
            failure = {
                "exit_code": exit_code,
                "reason": "performance baseline comparison failed",
                "report": f"{stem}_report.json",
            }
            marker.write_text(json.dumps(failure, indent=2))
            report = log.with_name(f"{stem}_report.json")
            details = report.read_text(errors="replace") if report.is_file() else "report not found"
            print(  # noqa: T201 - prominent, nonblocking JET job warning
                "\n"
                "=" * 80
                + "\nJET PERFORMANCE BASELINE FAILED (nonblocking exit code 3)\n"
                + f"workload: {stem}\nreport: {report}\n{details}\n"
                + "=" * 80,
                file=sys.stderr,
            )
            return 0
        traceback.print_exc()
        return exit_code
    return 0


def _write_provenance(artifact_root: Path, stem: str, inventories: list[dict], state: dict) -> None:
    result_dir = artifact_root / "ci" / "ci-results"
    result = {
        "upstream": {
            name: os.environ.get(name)
            for name in (
                "UPSTREAM_PROJECT_ID",
                "UPSTREAM_PROJECT_URL",
                "UPSTREAM_PIPELINE_ID",
                "UPSTREAM_COMMIT_SHA",
                "UPSTREAM_COMMIT_REF_NAME",
            )
        },
        "jet": {
            "pipeline_id": os.environ.get("JET_PIPELINE_ID"),
            "launcher_name": os.environ.get("JET_LAUNCHER_NAME"),
            "launcher_type": os.environ.get("JET_LAUNCHER_TYPE"),
        },
        "prepared": state,
        "ranks": inventories,
    }
    (result_dir / f"{stem}_jet_provenance.json").write_text(json.dumps(result, indent=2))


def _collect_run_artifacts(artifact_root: Path, run_root: Path) -> None:
    destination = artifact_root / "ci" / "ci-runs" / run_root.name
    patterns = (
        "logs/*.log",
        "logs/*_report.json",
        "logs/*_checkpoint_digest.xz",
        "logs/*_checkpoint_summary.log",
        "logs/baselines/*",
        "checkpoints/iter_*/registry.bin",
        "checkpoints/iter_*/reference.metadata.info",
        "checkpoints/iter_*/*.txt",
        "checkpoints/iter_*/.metadata",
        "checkpoints/iter_*/common_pt_dump.info",
        "phase1-checkpoints/iter_*/reference.metadata.info",
        "phase1-checkpoints/iter_*/common_pt_dump.info",
        "reference-run/*/iter_*/reference.metadata.info",
        "reference-run/*/iter_*/common_pt_dump.info",
        "run1-references/iter_*/reference.metadata.info",
        "*/logs/*.log",
        "*/logs/*_report.json",
        "*/logs/*_checkpoint_digest.xz",
        "*/logs/*_checkpoint_summary.log",
        "*/logs/baselines/*",
        "*/checkpoints/iter_*/registry.bin",
        "*/checkpoints/iter_*/reference.metadata.info",
        "*/checkpoints/iter_*/*.txt",
        "*/checkpoints/iter_*/.metadata",
        "*/checkpoints/iter_*/common_pt_dump.info",
        "*/phase1-checkpoints/iter_*/reference.metadata.info",
        "*/phase1-checkpoints/iter_*/common_pt_dump.info",
        "*/reference-run/*/iter_*/reference.metadata.info",
        "*/reference-run/*/iter_*/common_pt_dump.info",
        "*/run1-references/iter_*/reference.metadata.info",
    )
    for pattern in patterns:
        for source in run_root.glob(pattern):
            if not source.is_file():
                continue
            target = destination / source.relative_to(run_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _archive_path(assets_dir: Path, stem: str) -> Path:
    pipeline_id = os.environ.get("UPSTREAM_PIPELINE_ID", "").strip()
    if not re.fullmatch(r"[0-9]+", pipeline_id):
        raise RuntimeError(f"invalid upstream pipeline ID for JET artifacts: {pipeline_id!r}")
    return assets_dir / f"{stem}_pipeline-{pipeline_id}_artifacts.tar.gz"


def _archive(assets_dir: Path, artifact_root: Path, stem: str) -> None:
    archive = _archive_path(assets_dir, stem)
    with tarfile.open(archive, "w:gz") as output:
        output.add(artifact_root / "ci", arcname="ci")


def cleanup_finalized_run(workload_argv: list[str], assets_dir: Path) -> None:
    """Safely remove one exact run only after its artifact archive exists."""
    if os.environ.get("JET_CLEAN_RUN_AFTER_FINALIZE") != "1":
        return
    args = runtime.parse_args(workload_argv)
    run_dir_value = os.environ.get("SLURMJOB_OPT_RUN_DIR")
    run_name = os.environ.get("SLURMJOB_OPT_RUN_NAME")
    work_dir_value = os.environ.get("SLURMJOB_OPT_WORK_DIR")
    pipeline_id = os.environ.get("UPSTREAM_PIPELINE_ID")
    if not all((run_dir_value, run_name, work_dir_value, pipeline_id)):
        raise RuntimeError("JET run cleanup requires the exact run, work, and upstream pipeline identifiers")
    run_dir = Path(run_dir_value)
    work_dir = Path(work_dir_value)
    resolved_run = run_dir.resolve()
    resolved_work = work_dir.resolve()
    if (
        not run_dir.is_absolute()
        or not work_dir.is_absolute()
        or resolved_run == Path("/")
        or resolved_work == Path("/")
        or resolved_run.name != run_name
        or resolved_run.parent != resolved_work
        or resolved_work.name != pipeline_id
    ):
        raise RuntimeError(f"refusing unsafe JET run cleanup: run={run_dir}, work={work_dir}")
    if run_dir.is_symlink() or run_dir != resolved_run or work_dir != resolved_work:
        raise RuntimeError(f"refusing to clean symlinked JET run: {run_dir}")
    _cfg, _flavor, stem = slurm_job._identity(args)
    archive = _archive_path(assets_dir, stem)
    if not archive.is_file():
        raise RuntimeError(f"refusing to clean JET run without artifact archive: {archive}")
    with tarfile.open(archive, "r:gz") as contents:
        if not any(member.name == "ci" or member.name.startswith("ci/") for member in contents):
            raise RuntimeError(f"refusing to clean JET run with invalid artifact archive: {archive}")
    if resolved_run.exists():
        shutil.rmtree(resolved_run)
        print(f"[jet] removed finalized run directory: {resolved_run}")  # noqa: T201 - CI progress


def _synchronize(worker_rc: int, inventory: dict) -> int:
    run_root = slurm_job.RUN_ROOT
    if run_root is None:
        raise RuntimeError("run directory is not configured")
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    results_dir = slurm_job.RESULTS_DIR
    (results_dir / f"jet_inventory_rank{rank}.json").write_text(json.dumps(inventory, indent=2))
    coordination = _coordination_root()
    marker = coordination / f"worker-{rank}.rc"
    temporary = marker.with_suffix(".rc.tmp")
    temporary.write_text(f"{worker_rc}\n")
    temporary.replace(marker)
    # Any rank failure releases the synchronization barrier.
    deadline = time.monotonic() + 3600
    while True:
        markers = list(coordination.glob("worker-*.rc"))
        marker_rcs = [int(path.read_text()) for path in markers]
        if failure := next((code for code in marker_rcs if code), 0):
            return failure
        if len(markers) >= world:
            if rank == 0:
                result = coordination / "workers.rc"
                result.write_text("0\n")
            else:
                _wait_for_count(coordination, "workers.rc", 1)
            return 0
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {world} JET worker results")
        time.sleep(0.25)


def finalize(workload_argv: list[str], assets_dir: Path) -> int:
    """Finalize synchronized rank outputs from JET's per-rank teardown tasks."""
    args = runtime.parse_args(workload_argv)
    runtime.configure(args)
    state = runtime.require_prepared()
    run_root = slurm_job.RUN_ROOT
    if run_root is None:
        raise RuntimeError("run directory is not configured")
    world = args.nodes * args.gpus_per_node
    results_dir = slurm_job.RESULTS_DIR
    coordination = _coordination_root()
    deadline = time.monotonic() + 3600
    while True:
        markers = list(coordination.glob("worker-*.rc"))
        marker_rcs = [int(path.read_text()) for path in markers]
        if worker_rc := next((code for code in marker_rcs if code), 0):
            break
        workers_result = coordination / "workers.rc"
        if len(markers) >= world and workers_result.is_file():
            worker_rc = int(workers_result.read_text())
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {world} JET worker results")
        time.sleep(0.25)
    final_rc = worker_rc
    try:
        rank_logs = [
            path
            for index in range(world)
            if (path := results_dir / f"jet_worker_rank{index}.log").is_file()
        ]
        artifact_root = Path(os.environ.get("CI_ARTIFACT_ROOT", assets_dir / "nvrx-ci-artifacts"))
        log, stem = _combined_log(args, artifact_root, rank_logs, assets_dir)
        if final_rc == 0:
            final_rc = _run_epilog(args, log)
        _write_arguments_candidate(log, stem)
        _write_baseline_candidate(args, log, stem)
        inventories = [
            json.loads(path.read_text())
            for index in range(world)
            if (path := results_dir / f"jet_inventory_rank{index}.json").is_file()
        ]
        _write_provenance(artifact_root, stem, inventories, state)
        _collect_run_artifacts(artifact_root, run_root)
        _archive(assets_dir, artifact_root, stem)
    except BaseException:
        traceback.print_exc()
        final_rc = worker_rc or 1
    return final_rc


def coordinated_finalize(workload_argv: list[str], assets_dir: Path) -> int:
    """Elect one surviving rank to finalize and publish its result to every rank."""
    args = runtime.parse_args(workload_argv)
    runtime.configure(args)
    runtime.require_prepared()
    run_root = slurm_job.RUN_ROOT
    if run_root is None:
        raise RuntimeError("run directory is not configured")
    rank = int(os.environ["SLURM_PROCID"])
    coordination = _coordination_root()
    result = coordination / "finalizer.rc"
    leader = coordination / "finalizer.leader"
    if result.is_file():
        return int(result.read_text())
    try:
        descriptor = os.open(leader, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        descriptor = None
    if descriptor is not None:
        with os.fdopen(descriptor, "w") as output:
            output.write(f"{rank}\n")
        try:
            final_rc = finalize(workload_argv, assets_dir)
        except BaseException:
            traceback.print_exc()
            final_rc = 1
        temporary = result.with_suffix(".rc.tmp")
        temporary.write_text(f"{final_rc}\n")
        temporary.replace(result)
        return final_rc
    _wait_for_count(result.parent, result.name, 1)
    return int(result.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("workload_args", nargs=argparse.REMAINDER)
    wrapper_args = parser.parse_args()
    workload_argv = wrapper_args.workload_args
    if workload_argv[:1] == ["--"]:
        workload_argv = workload_argv[1:]
    args = runtime.parse_args(workload_argv)
    runtime.configure(args)
    rank = int(os.environ["SLURM_PROCID"])
    rank_log = slurm_job.RESULTS_DIR / f"jet_worker_rank{rank}.log"
    worker_rc = 0
    inventory: dict = {"rank": rank}
    with rank_log.open("w") as output:
        tee = Tee(sys.stdout, output)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                runtime.require_prepared()
                inventory = runtime.inventory(args)
                _coordinate_server_preflight(args)
                slurm_job._worker(args)
            except BaseException as error:
                traceback.print_exc()
                worker_rc = _exit_code(error)
                inventory["error"] = f"{type(error).__name__}: {error}"
    return _synchronize(worker_rc, inventory)


if __name__ == "__main__":
    sys.exit(main())
