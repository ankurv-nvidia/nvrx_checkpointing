#!/usr/bin/env python3
"""CI job runner for megatron/native benchmarks and e2e tests.

Runs a job and automatically parses performance results afterwards.
By default only perf numbers are printed; use --skip-perf to see full output.

Jobs:
  ckptbench       Build and run the C++ checkpoint benchmark (1 node, 2 GPUs).
  pretrain-all    Run pretrain_mamba.py with both checkpoint providers (correctness).
  pretrain-native Run pretrain_mamba.py with native checkpoint provider (perf benchmark).
  pretrain-nvrx   Run pretrain_mamba.py with nvrx checkpoint provider (comparison baseline).
  valgrind        Same as pretrain-native, but pre-launches checkpoint_server under
                  valgrind (memcheck) so leaks/errors in the server are reported.
                  Local single-node only — SLURM/multi-node not supported.

Usage:
  python3 run_job.py pretrain-native                 # 1 node, run inline
  python3 run_job.py pretrain-native --nodes 16      # 16 nodes via SLURM
  python3 run_job.py pretrain-native --slurm         # 1 node via SLURM
  python3 run_job.py ckptbench                       # run inline (2 GPUs required locally)
  python3 run_job.py ckptbench --slurm               # run via SLURM
  python3 run_job.py pretrain-native --skip-perf     # tee full output, no perf parsing
  python3 run_job.py pretrain-native --update-baseline  # record current perf as new baseline

Cluster defaults (overridable via environment variables):
  SLURM_ACCOUNT           coreai_resiliency_osiris
  SLURM_PARTITION         batch
  ENROOT_IMAGE            ${HOME}/data/native.<hash>.sqsh
                            (hash = first 10 hex of sha256(megatron/native/scripts/prepare.sh);
                            changing prepare.sh causes a rebuild under a new filename)
  SLURM_GPU_FLAGS_ENABLED 1 (set to 0 on runners that don't support --gpus/--gpus-per-node;
                            the job will run with all GPUs allocated by the scheduler)

Outputs (auto-derived from job and config):
  Logs:      <megatron_root>/megatron/native/ci/ci-results/<machine>_<job>_<config>.log
  Reports:   <megatron_root>/megatron/native/ci/ci-results/<machine>_<job>_<config>_report.json
  Baselines: <script_dir>/../baselines/<machine>_<job>_<config>.yaml

Exit codes:
  0  Job succeeded and all perf metrics within tolerance (or --skip-perf)
  3  Job succeeded but one or more perf metrics regressed beyond tolerance
  1  Job or setup failed (raised as an exception)
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MEGATRON_ROOT = (SCRIPT_DIR / "../../../..").resolve()
BASELINES_DIR = (SCRIPT_DIR / "../baselines").resolve()

sys.path.insert(0, str(MEGATRON_ROOT))

ENROOT_SCRIPTS_DIR = (MEGATRON_ROOT / "megatron/native/scripts").resolve()
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.04-py3")


def wait_for_file(path: Path, timeout_seconds: int) -> None:
    """Wait for a file to exist, polling every 10 seconds, up to timeout_seconds.

    If path's basename contains a glob wildcard (*?[), waits for any match under path.parent
    instead — useful for files whose exact name isn't known ahead of time (e.g. async-written
    checkpoint metadata).
    """
    has_wildcard = any(c in path.name for c in "*?[")
    start_time = time.time()
    while True:
        if has_wildcard:
            if any(path.parent.glob(path.name)):
                return
        elif path.is_file():
            return
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {path} after {timeout_seconds} seconds")
        prefix = "[in-srun] " if "SLURM_PROCID" in os.environ else ""
        sys.stderr.write(f"{prefix}Waiting for {path} to be available...\n")
        time.sleep(10)


def _prepare_sh_hash() -> str:
    """First 10 hex chars of sha256(prepare.sh) — namespaces the built sqsh image so any
    change to prepare.sh forces a rebuild rather than reusing a stale image. Must match
    the hash computed in scripts/enroot.sh."""
    return hashlib.sha256((ENROOT_SCRIPTS_DIR / "prepare.sh").read_bytes()).hexdigest()[:10]


def _default_enroot_image() -> str:
    return f"{Path.home()}/data/native.{_prepare_sh_hash()}.sqsh"


def _machine_tag() -> str:
    """Short machine/cluster identifier used to namespace baseline files.

    Override via MEGATRON_CI_MACHINE; falls back to the hostname stem.
    """
    tag = os.environ.get("MEGATRON_CI_MACHINE", "").strip()
    return tag or socket.gethostname().split(".")[0]


# Ensure the repo root is on sys.path so `import megatron` works regardless of
# how the script is invoked (e.g. from a GitLab runner without a proper install).
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))
RESULTS_DIR = MEGATRON_ROOT / "megatron" / "native" / "ci" / "ci-results"
GPUS_PER_NODE = 4
PERF_TOLERANCE = 0.05  # 5% tolerance — both regressions and improvements beyond this fail

# Stable vcpkg root on shared filesystem — persists installed packages across CI runs.
# Override via VCPKG_ROOT env var. Must be accessible inside the enroot container.
VCPKG_ROOT = os.environ.get("VCPKG_ROOT", str(Path.home() / "data/vcpkg"))
os.environ.setdefault("VCPKG_ROOT", VCPKG_ROOT)  # ensure subprocesses and os.environ.copy() inherit it
_BASELINE_CHANGE_THRESHOLD = 1e-6  # minimum relative change to count as a meaningful baseline update

# Maps each pretrain job name to the --ckpt-provider value it exercises.
_PRETRAIN_JOBS: dict[str, str] = {
    "pretrain-all": "nvrx,native",  # single-node correctness test (both providers)
    "pretrain-native": "native",  # single-node native-checkpoint perf benchmark
    "pretrain-nvrx": "nvrx",  # single-node nvrx baseline for comparison
    "valgrind": "native",  # like pretrain-native, but checkpoint_server runs under valgrind
}

# ---------------------------------------------------------------------------
# Enroot image build
# ---------------------------------------------------------------------------


def _ensure_enroot_image(image: Path) -> None:
    """Build the enroot .sqsh image if it doesn't exist by delegating to enroot.sh."""
    if image.exists():
        return
    base_image = image.parent / "pytorch2604.sqsh"
    sys.stderr.write(f"[run_job] enroot image not found: {image} — building now (this may take ~30 min)\n")

    enroot_sh = ENROOT_SCRIPTS_DIR / "enroot.sh"
    env = os.environ.copy()
    env["ENROOT_IMAGE_BASE"] = base_image.as_posix()
    env["ENROOT_IMAGE_OUT"] = image.as_posix()
    subprocess.run([shutil.which("bash"), enroot_sh.as_posix()], env=env, check=True)


# ---------------------------------------------------------------------------
# SLURM context detection
# ---------------------------------------------------------------------------


def in_srun() -> bool:
    return "SLURM_PROCID" in os.environ


def _slurm_nodes_and_gpus() -> tuple[int, int]:
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    raw = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_NTASKS_PER_NODE") or os.environ.get("SLURM_TASKS_PER_NODE")
    if raw is None:
        slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
        raise RuntimeError(
            f"cannot infer gpus_per_node — SLURM_GPUS_PER_NODE and SLURM_TASKS_PER_NODE are both unset. SLURM environment: {slurm_vars}",
        )
    # SLURM_GPUS_PER_NODE may be "gpu:4" or plain "4"
    return nodes, int(str(raw).split(":")[-1])


def _slurm_gpu_flags_enabled() -> bool:
    """False on runners that don't support --gpus/--gpus-per-node (set SLURM_GPU_FLAGS_ENABLED=0)."""
    return os.environ.get("SLURM_GPU_FLAGS_ENABLED", "1").strip() not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Job: ckptbench
# ---------------------------------------------------------------------------


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
    registry = MEGATRON_ROOT / "megatron/native/metadata/1x2_checkpoints_iter_0000250/registry.bin"
    if not registry.is_file():
        raise FileNotFoundError(f"stateregistry not found: {registry}")

    if in_srun():
        binary = _ckptbench_binary()
        bench_cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]
        subprocess.run(bench_cmd, check=True)
        return

    if args.slurm:
        account = os.environ.get("SLURM_ACCOUNT", "coreai_resiliency_osiris")
        partition = os.environ.get("SLURM_PARTITION")
        image = os.environ.get("ENROOT_IMAGE", _default_enroot_image())
        cmd = [
            "srun",
            f"--account={account}",
            *([f"--partition={partition}"] if partition else []),
            "--time=5",
            "--nodes=1",
            "--ntasks=1",
            *([f"--gpus-per-node={args.gpus_per_node}"] if _slurm_gpu_flags_enabled() else []),
            "--mem=0",
            f"--job-name=ckpt_benchmark_{os.environ.get('CI_PIPELINE_ID', 'local')}",
            f"--container-image={image}",
            f"--container-mounts={Path.home()}:{Path.home()},/lustre:/lustre",
            "--container-env=VCPKG_ROOT",  # forward by name — avoids KEY=VALUE ambiguity in pyxis/enroot
            "python3",
            "-u",
            str(Path(__file__).resolve()),
            "ckptbench",
        ]
    else:
        binary = _ckptbench_binary()
        cmd = [str(binary), "--stateregistry", str(registry), "--benchmark_min_iters", "5", "--benchmark_max_iters", "5", "--inproc"]

    _run_and_report_perf(cmd, "ckptbench", f"1x{args.gpus_per_node}", args)


# ---------------------------------------------------------------------------
# Job: pretrain
# ---------------------------------------------------------------------------


def _checkpoint_server_binary() -> Path:
    """Path to the built checkpoint_server binary (built/copied by megatron.native import)."""
    import megatron.native as _native  # noqa: F401, PLC0415 — triggers CMake build

    binary = MEGATRON_ROOT / "megatron/native/checkpoint_server"
    if not binary.is_file():
        raise FileNotFoundError(f"checkpoint_server not found at {binary}")
    return binary


_VALGRIND_READY_MARKER = "Checkpoint server started"


def _spawn_valgrind_server(cmd: list[str], log_fp: IO[str]) -> tuple[subprocess.Popen, threading.Event]:
    """Spawn checkpoint_server under valgrind with a pump thread that tees output to
    both `log_fp` and the parent's stderr (live console), and signals the returned Event
    once the ready marker appears (or the process exits, so a stuck waiter doesn't deadlock).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=1,
        text=True,
    )
    ready = threading.Event()

    def _pump() -> None:
        assert proc.stdout is not None  # noqa: S101 — Popen with stdout=PIPE always has it
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
            if not ready.is_set() and _VALGRIND_READY_MARKER in line:
                ready.set()
        ready.set()

    threading.Thread(target=_pump, daemon=True).start()
    return proc, ready


@contextlib.contextmanager
def _valgrind_checkpoint_server(machine: str, cfg: str) -> Iterator[None]:
    """Pre-launch checkpoint_server under valgrind for the duration of the block.

    On enter: starts the server under memcheck, sets CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH=1
    so the in-process client (see checkpoint.cpp::SvcPing) skips its own server fork and
    connects to the pre-launched valgrind-wrapped instance, and blocks until the server
    prints its "Checkpoint server started" ready marker.

    On exit: waits for the server's ~150s no-activity auto-shutdown so valgrind can flush
    its leak report, then raises if valgrind detected errors.

    valgrind diagnostics use --log-fd=2 so they flow through the same pipe as the server's
    own stderr; the pump thread tees that combined stream to both the persisted log file
    and the parent's stderr (live console).
    """
    valgrind = shutil.which("valgrind")
    if valgrind is None:
        raise RuntimeError("valgrind not found in PATH — required for the 'valgrind' job")

    server = _checkpoint_server_binary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server_log = RESULTS_DIR / f"{machine}_valgrind_{cfg}_server.log"

    cmd = [
        valgrind,
        "--tool=memcheck",
        "--leak-check=full",
        "--show-leak-kinds=all",
        # "--track-origins=yes",
        "--error-exitcode=42",
        "--log-fd=2",
        str(server),
    ]
    sys.stderr.write("[run_job] Launching checkpoint_server under valgrind\n")
    sys.stderr.write(f"[run_job]   server+valgrind log: {server_log}\n")

    os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
    log_fp = server_log.open("w")
    proc, ready = _spawn_valgrind_server(cmd, log_fp)

    try:
        if not ready.wait(timeout=600):
            raise RuntimeError(
                f"checkpoint_server did not print '{_VALGRIND_READY_MARKER}' within 600s under valgrind; see {server_log}",
            )
        if proc.poll() is not None:
            raise RuntimeError(
                f"checkpoint_server exited before becoming ready (rc={proc.returncode}); see {server_log}",
            )
        sys.stderr.write("[run_job] checkpoint_server is ready — proceeding with pretrain\n")
        yield
    finally:
        sys.stderr.write(
            "[run_job] Pretrain finished — waiting for checkpoint_server auto-shutdown (~150s) so valgrind can flush its report\n",
        )
        try:
            rc = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            sys.stderr.write("[run_job] checkpoint_server did not auto-shutdown; sending SIGTERM\n")
            proc.terminate()
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
        log_fp.close()

        sys.stderr.write(f"[run_job] valgrind exited with code {rc}\n")
        sys.stderr.write(f"[run_job] server+valgrind log: {server_log}\n")
        if rc != 0:
            raise RuntimeError(f"valgrind reported errors in checkpoint_server (exit {rc}); see {server_log}")


def _maybe_valgrind_ctx(args: argparse.Namespace, cfg: str) -> contextlib.AbstractContextManager[None]:
    """Wrap the pretrain launch in a valgrind+checkpoint_server context when job=valgrind.

    In srun mode every rank sets CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH so its client skips
    the auto-fork; only global rank 0 hosts the valgrind-wrapped server.
    """
    if args.job != "valgrind":
        return contextlib.nullcontext()
    if os.environ.get("SLURM_PROCID", "0") != "0":
        os.environ["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"
        return contextlib.nullcontext()
    return _valgrind_checkpoint_server(_machine_tag(), cfg)


def run_pretrain(args: argparse.Namespace) -> None:
    ckpt_provider = _PRETRAIN_JOBS[args.job]
    cfg = f"{args.nodes}x{args.gpus_per_node}"
    extra_args = getattr(args, "extra_args", None)
    if in_srun():
        nodes, gpus_per_node = _slurm_nodes_and_gpus()
        cfg = f"{nodes}x{gpus_per_node}"
        try:
            with _maybe_valgrind_ctx(args, cfg):
                cmd, env = _pretrain_cmd_and_env(cfg, nodes, gpus_per_node, ckpt_provider=ckpt_provider, extra_args=extra_args)
                subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError:
            _dump_nvrx_vs_native_diffs(cfg)
            raise
        # Keep the srun cgroup alive until the detached checkpoint_server has
        # flushed every iter's native artifacts. Returning here would let srun
        # tear down the cgroup and SIGKILL the server mid-write — most visible
        # at --exit-interval where there is no follow-up training to cover the
        # async drain.
        if args.job in _NATIVE_CKPT_JOBS:
            try:
                _wait_for_native_metadata_files(cfg, timeout_seconds=1800)
            except TimeoutError as ex:
                sys.stderr.write(f"[run_job] in-srun wait for native artifacts timed out: {ex}\n")
        return

    # Clear stale checkpoints once in the launcher so in-srun tasks don't race on rmtree.
    ckpt_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    if args.nodes > 1 or args.slurm:
        # Launcher only — orchestration happens inside the srun task (in_srun branch above).
        cmd = _build_pretrain_srun_cmd(cfg, args.nodes, args.gpus_per_node, args.job, extra_args=extra_args)
        _run_and_report_perf(cmd, args.job, cfg, args)
    elif args.gpus_per_node > 1:
        # Multi-GPU devbox: use torchrun subprocess so all worker output is captured in the log.
        # mp.spawn workers don't inherit Python-level sys.stdout/stderr redirections.
        cmd, env = _pretrain_cmd_and_env(
            cfg,
            args.nodes,
            args.gpus_per_node,
            use_torchrun=True,
            ckpt_provider=ckpt_provider,
            extra_args=extra_args,
        )
        with _maybe_valgrind_ctx(args, cfg):
            _run_and_report_perf(cmd, args.job, cfg, args, env=env)
    else:
        # Single-GPU inline: call pretrain() directly in-process — no subprocess or torchrun.
        # This lets you attach a debugger to run_job.py and hit breakpoints in training code.
        with _maybe_valgrind_ctx(args, cfg):
            _launch_pretrain_inline(cfg, args)


def _build_pretrain_srun_cmd(cfg: str, nodes: int, gpus_per_node: int, job: str, extra_args: list[str] | None = None) -> list[str]:
    account = os.environ.get("SLURM_ACCOUNT", "coreai_resiliency_osiris")
    partition = os.environ.get("SLURM_PARTITION")
    image = os.environ.get("ENROOT_IMAGE", _default_enroot_image())
    triton_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/triton-cache"
    total_tasks = nodes * gpus_per_node

    env_exports = (
        "NCCL_IB_TIMEOUT=24,UCX_MEM_MMAP_HOOK_MODE=none,UCX_MEM_CUDA_HOOK_MODE=none,"
        "UCX_MEM_MALLOC_HOOKS=n,UCX_ERROR_SIGNALS=,NVTE_FWD_LAYERNORM_SM_MARGIN=16,"
        "NVTE_BWD_LAYERNORM_SM_MARGIN=16,TORCHINDUCTOR_WORKER_START=fork,"
        f"QUANTIZATION_TYPE_DEBUG=1,CUDA_DEVICE_MAX_CONNECTIONS=1,TRITON_CACHE_DIR={triton_dir},"
        "VCPKG_ROOT"  # forward by name — avoids KEY=VALUE ambiguity in pyxis/enroot
    )
    if job == "valgrind":
        # Forward the valgrind build flag; CMakeLists.txt translates it into
        # CPPFORGE_DISABLE_MARCH_NATIVE plus a compile-time macro that shrinks
        # D2H buffer sizes so valgrind's shadow memory stays manageable.
        env_exports += ",PHOENIX_BUILD_FOR_VALGRIND"

    return [
        "srun",
        f"--account={account}",
        *([f"--partition={partition}"] if partition else []),
        f"--nodes={nodes}",
        f"--ntasks={total_tasks}",
        f"--time={90 if job == 'valgrind' else 90}",  # 30
        *([f"--gpus={total_tasks}", f"--gpus-per-node={gpus_per_node}"] if _slurm_gpu_flags_enabled() else []),
        "--mem=0",
        f"--job-name={job}_{cfg}_{os.environ.get('CI_PIPELINE_ID', 'local')}",
        f"--container-image={image}",
        f"--container-mounts={Path.home()}:{Path.home()},/lustre:/lustre",
        f"--container-env={env_exports}",
        # SLURM_PROCID will be set on each task, triggering in-srun mode.
        "python3",
        "-u",
        str(Path(__file__).resolve()),
        job,
        *(extra_args or []),
    ]


def _pretrain_cmd_and_env(  # noqa: PLR0913
    cfg: str,
    nodes: int,
    gpus_per_node: int,
    use_torchrun: bool = False,
    ckpt_provider: str = "nvrx,native",
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Return (cmd, env) to run pretrain_mamba.py on a compute node.

    use_torchrun=True  — devbox multi-GPU; torchrun spawns one process per GPU.
    use_torchrun=False — in-srun; srun already placed one process per GPU.
    """
    run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    datacache = run_dir / "data-cache"
    triton_dir = run_dir / "triton-cache"
    for d in [ckpt_dir, tb_dir, datacache, triton_dir]:
        d.mkdir(parents=True, exist_ok=True)

    opts = _pretrain_cli_args(
        PretrainRunConfig(
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            ckpt_dir=ckpt_dir,
            tb_dir=tb_dir,
            datacache_dir=datacache,
            ckpt_provider=ckpt_provider,
        ),
    )

    env = os.environ.copy()
    env.update(
        {
            "NCCL_IB_TIMEOUT": "24",
            "UCX_MEM_MMAP_HOOK_MODE": "none",
            "UCX_MEM_CUDA_HOOK_MODE": "none",
            "UCX_MEM_MALLOC_HOOKS": "n",
            "UCX_ERROR_SIGNALS": "",
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
            "TORCHINDUCTOR_WORKER_START": "fork",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "QUANTIZATION_TYPE_DEBUG": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TRITON_CACHE_DIR": str(triton_dir),
        },
    )

    if use_torchrun:
        launcher = ["torchrun", "--nproc_per_node", str(gpus_per_node)]
    else:
        # In-srun: set PyTorch distributed env vars from SLURM.
        # Port is derived from SLURM_JOB_ID so concurrent jobs don't collide.
        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        master_port = 20000 + (slurm_job_id % 40000)
        try:
            master_addr = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]], text=True).split()[0]  # noqa: S607
        except (subprocess.CalledProcessError, OSError, KeyError, IndexError):
            master_addr = "localhost"
        env.update(
            {
                "RANK": os.environ.get("SLURM_PROCID", "0"),
                "LOCAL_RANK": os.environ.get("SLURM_LOCALID", "0"),
                "WORLD_SIZE": os.environ.get("SLURM_NTASKS", "1"),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            },
        )
        launcher = ["python3", "-u"]

    return [*launcher, str(MEGATRON_ROOT / "pretrain_mamba.py"), *opts, *(extra_args or [])], env


@dataclass
class PretrainRunConfig:
    nodes: int
    gpus_per_node: int
    ckpt_dir: Path
    tb_dir: Path
    datacache_dir: Path
    save_interval: int = 250
    num_checkpoints: int = 3
    ckpt_provider: str = "nvrx,native"

    @property
    def scale(self) -> str:
        return f"{self.nodes}x{self.gpus_per_node}"


def _pretrain_cli_args(config: PretrainRunConfig) -> list[str]:
    ckpt_dir, tb_dir, datacache_dir = config.ckpt_dir, config.tb_dir, config.datacache_dir
    nodes, gpus_per_node = config.nodes, config.gpus_per_node
    save_interval, num_checkpoints = config.save_interval, config.num_checkpoints
    scale = config.scale
    seq_len = 1024

    exit_interval = save_interval * (num_checkpoints)

    if scale == "16node":
        gbs = 3072
        model = [
            "--num-layers",
            "44",
            "--hidden-size",
            "4096",
            "--num-attention-heads",
            "32",
            "--ffn-hidden-size",
            "2688",
            "--kv-channels",
            "128",
            "--mamba-num-heads",
            "128",
            "--hybrid-override-pattern",
            "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEME",
            "--num-experts",
            "512",
            "--moe-router-topk",
            "22",
            "--moe-shared-expert-intermediate-size",
            "5376",
            "--moe-latent-size",
            "1024",
            "--num-layers-at-start-in-bf16",
            "0",
            "--num-layers-at-end-in-bf16",
            "14",
        ]
        parallel = [
            "--tensor-model-parallel-size",
            "2",
            "--expert-model-parallel-size",
            "64",
            "--expert-tensor-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
        ]
    else:
        gbs = nodes * gpus_per_node
        tp = ep = gpus_per_node
        model = [
            "--num-layers",
            "3",
            "--hidden-size",
            "2048",
            "--num-attention-heads",
            "32",
            "--ffn-hidden-size",
            "1344",
            "--kv-channels",
            "128",
            "--mamba-num-heads",
            "64",
            "--hybrid-override-pattern",
            "M*E",
            "--num-experts",
            "32",
            "--moe-router-topk",
            "11",
            "--moe-shared-expert-intermediate-size",
            "2688",
            "--moe-latent-size",
            "1024",
            "--num-layers-at-start-in-bf16",
            "0",
            "--num-layers-at-end-in-bf16",
            "2",
        ]
        parallel = [
            "--tensor-model-parallel-size",
            str(tp),
            "--expert-model-parallel-size",
            str(ep),
            "--expert-tensor-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
        ]

    return [
        # MTP
        "--mtp-num-layers",
        "2",
        "--mtp-hybrid-override-pattern",
        "*E",
        "--calculate-per-token-loss",
        "--mtp-loss-scaling-factor",
        "0.3",
        # Architecture
        "--is-hybrid-model",
        "--first-last-layers-bf16",
        "--untie-embeddings-and-output-weights",
        "--init-method-std",
        "0.014",
        "--position-embedding-type",
        "none",
        "--bf16",
        "--use-mcore-models",
        "--spec",
        "megatron.core.models.mamba.mamba_layer_specs",
        "mamba_stack_spec",
        *model,
        # Parallelism
        *parallel,
        "--sequence-parallel",
        "--no-gradient-accumulation-fusion",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--override-opt_param-scheduler",
        # MoE
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-router-score-function",
        "sigmoid",
        "--moe-grouped-gemm",
        "--moe-aux-loss-coeff",
        "1e-4",
        "--moe-router-topk-scaling-factor",
        "5.0",
        "--moe-router-enable-expert-bias",
        "--moe-router-dtype",
        "fp32",
        "--moe-router-load-balancing-type",
        "seq_aux_loss",
        "--moe-permute-fusion",
        # Fused ops / attention
        "--attention-backend",
        "flash",
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl",
        "native",
        "--use-fused-weighted-squared-relu",
        "--squared-relu",
        "--te-rng-tracker",
        "--ddp-num-buckets",
        "10",
        # Checkpoint
        f"--ckpt-provider={config.ckpt_provider}",
        "--ckpt-format",
        "torch_dist",
        "--ckpt-assume-constant-structure",
        "--async-save",
        "--use-persistent-ckpt-worker",
        "--save",
        str(ckpt_dir),
        "--load",
        str(ckpt_dir),
        "--save-interval",
        str(save_interval),
        "--save-retain-interval",
        str(save_interval),
        # Data — NullTokenizer requires no vocab file; vocab_size matches the production tokenizer
        "--mock-data",
        "--data-cache-path",
        str(datacache_dir),
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "256000",
        # Schedule
        "--seq-length",
        str(seq_len),
        "--max-position-embeddings",
        str(seq_len),
        "--lr-wsd-decay-style",
        "minus_sqrt",
        # Optimizer
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(gbs),
        "--lr",
        "4.5e-4",
        "--min-lr",
        "4.5e-6",
        "--weight-decay",
        "0.1",
        "--clip-grad",
        "1.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        # Regularization
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        # Logging
        "--log-interval",
        str(save_interval),
        "--log-params-norm",
        "--log-num-zeros-in-grad",
        "--log-throughput",
        "--log-progress",
        "--logging-level",
        "20",
        "--disable-straggler-on-startup",
        "--straggler-minmax-count",
        "16",
        "--timing-log-option",
        "minmax",
        # Misc
        "--eval-interval",
        str(exit_interval),
        "--eval-iters",
        "14",
        "--no-mmap-bin-files",
        "--no-create-attention-mask-in-dataloader",
        "--distributed-timeout-minutes",
        "30",
        "--num-workers",
        "1",
        "--disable-gloo-process-groups",
        "--exit-interval",
        str(exit_interval),
        "--train-iters",
        str(exit_interval),
        "--tensorboard-dir",
        str(tb_dir),
    ]


# ---------------------------------------------------------------------------
# Inline pretrain (in-process, no torchrun subprocess)
# ---------------------------------------------------------------------------


def _launch_pretrain_inline(cfg: str, args: argparse.Namespace) -> None:  # noqa: C901
    """Run pretrain in-process (no subprocess/torchrun) for single-GPU devbox runs.

    Output is tee'd to the log file so perf metrics can be parsed afterwards.
    Multi-GPU devbox runs are routed through torchrun subprocess in run_pretrain().
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    machine = _machine_tag()
    log_file = RESULTS_DIR / f"{machine}_{args.job}_{cfg}.log"
    report_file = RESULTS_DIR / f"{machine}_{args.job}_{cfg}_report.json"
    baseline_file = BASELINES_DIR / f"{machine}_{args.job}_{cfg}.yaml"

    nodes, gpus_per_node = args.nodes, args.gpus_per_node
    run_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}"
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    datacache = run_dir / "data-cache"
    triton_dir = run_dir / "triton-cache"
    for d in [ckpt_dir, tb_dir, datacache, triton_dir]:
        d.mkdir(parents=True, exist_ok=True)

    opts = _pretrain_cli_args(
        PretrainRunConfig(
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            ckpt_dir=ckpt_dir,
            tb_dir=tb_dir,
            datacache_dir=datacache,
            save_interval=1,
            num_checkpoints=2,
            ckpt_provider=_PRETRAIN_JOBS[args.job],
        ),
    )
    opts = [*opts, *(getattr(args, "extra_args", None) or [])]

    os.environ.update(
        {
            "NCCL_IB_TIMEOUT": "24",
            "UCX_MEM_MMAP_HOOK_MODE": "none",
            "UCX_MEM_CUDA_HOOK_MODE": "none",
            "UCX_MEM_MALLOC_HOOKS": "n",
            "UCX_ERROR_SIGNALS": "",
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
            "TORCHINDUCTOR_WORKER_START": "fork",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "QUANTIZATION_TYPE_DEBUG": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TRITON_CACHE_DIR": str(triton_dir),
        },
    )

    if args.skip_perf:
        _call_pretrain_inline(opts, nodes, gpus_per_node)
        return

    class _Tee:
        def __init__(self, stream: IO[str], logfp: IO[str]) -> None:
            self._stream, self._logfp = stream, logfp

        def write(self, data: str) -> int:
            self._stream.write(data)
            self._logfp.write(data)
            return len(data)

        def flush(self) -> None:
            self._stream.flush()
            self._logfp.flush()

        def fileno(self) -> int:
            return self._stream.fileno()

        def __getattr__(self, name: str) -> object:
            return getattr(self._stream, name)

    with log_file.open("w") as log_fp:
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(saved_stdout, log_fp)
        sys.stderr = _Tee(saved_stderr, log_fp)
        try:
            _call_pretrain_inline(opts, nodes, gpus_per_node)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if rc != 0:
                raise RuntimeError(f"pretrain exited with code {rc} — full log: {log_file}") from None
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr

    if args.job in _NATIVE_CKPT_JOBS:
        _wait_for_native_metadata_files(cfg, timeout_seconds=1800)

    perf_passed = _parse_and_report(
        args.job,
        cfg,
        log_file,
        baseline_file,
        report_file,
        args.update_baseline,
        args.gpus_per_node,
    )
    if not perf_passed:
        sys.exit(3)


def _call_pretrain_inline(opts: list[str], nodes: int, gpus_per_node: int) -> None:
    """Set sys.argv and call pretrain() directly.

    gpus_per_node==1: runs in the calling process (debugger-friendly).
    gpus_per_node>1:  uses torch.multiprocessing.spawn.
    """
    import time  # noqa: PLC0415

    if str(MEGATRON_ROOT) not in sys.path:
        sys.path.insert(0, str(MEGATRON_ROOT))

    program_start = time.time()

    if "MASTER_PORT" not in os.environ:
        import socket as _socket  # noqa: PLC0415

        with _socket.socket() as _s:
            _s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(_s.getsockname()[1])

    if gpus_per_node == 1:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        # Force num-workers=0 so DataLoader runs in-process with no subprocess spawning.
        clean = []
        skip = False
        for tok in opts:
            if skip:
                skip = False
                continue
            if tok == "--num-workers":
                skip = True
                continue
            clean.append(tok)
        opts = [*clean, "--num-workers", "0"]
        sys.argv = ["pretrain_mamba.py", *opts]
        _pretrain_main(program_start)
    else:
        import torch.multiprocessing as mp  # noqa: PLC0415

        mp.spawn(
            _pretrain_worker,
            args=(gpus_per_node, nodes, opts, program_start),
            nprocs=gpus_per_node,
            join=True,
        )


def _pretrain_worker(local_rank: int, gpus_per_node: int, nodes: int, opts: list[str], program_start: float) -> None:
    """Entry point for each torch.multiprocessing.spawn worker (multi-GPU inline)."""
    os.environ["RANK"] = str(local_rank)  # single-node: global rank == local rank
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(nodes * gpus_per_node)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    sys.argv = ["pretrain_mamba.py", *opts]
    _pretrain_main(program_start)


def _pretrain_main(program_start: float) -> None:
    """Mirror of pretrain_mamba.py's __main__ block — calls pretrain() directly."""
    import time  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415

    from megatron.core.enums import ModelType  # noqa: PLC0415
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps  # noqa: PLC0415

    try:
        from megatron.post_training.arguments import add_modelopt_args  # noqa: PLC0415

        has_modelopt = True
    except ImportError:
        has_modelopt = False

    import pretrain_mamba as pm  # noqa: PLC0415
    from mamba_builders import mamba_builder  # noqa: PLC0415
    from model_provider import model_provider as _mp  # noqa: PLC0415

    set_startup_timestamps(program_start=program_start, main_entry=time.time())

    pm.train_valid_test_datasets_provider.is_distributed = True

    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    pretrain_fn(
        pm.train_valid_test_datasets_provider,
        partial(_mp, mamba_builder),
        ModelType.encoder_or_decoder,
        pm.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        store=store,
        extra_args_provider=add_modelopt_args if has_modelopt else None,
    )


# ---------------------------------------------------------------------------
# Launch + perf post-processing
# ---------------------------------------------------------------------------


def _run_and_report_perf(cmd: list[str], job: str, config: str, args: argparse.Namespace, env: dict | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    machine = _machine_tag()
    log_file = RESULTS_DIR / f"{machine}_{job}_{config}.log"
    report_file = RESULTS_DIR / f"{machine}_{job}_{config}_report.json"
    baseline_file = BASELINES_DIR / f"{machine}_{job}_{config}.yaml"

    if args.skip_perf:
        rc = subprocess.run(cmd, env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Job failed with exit code {rc}")
        return

    sys.stderr.write(f"[run_job] Running command: {' '.join(shlex.quote(c) for c in cmd)}\n")
    # srun+enroot doesn't forward container stdout through a PIPE, so we tee via shell.
    shell_cmd = f"set -o pipefail; {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {shlex.quote(str(log_file))}"
    rc = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", env=env, check=False).returncode  # noqa: S602

    if rc != 0:
        raise RuntimeError(f"Job failed with exit code {rc} — full log: {log_file}")

    if job in _NATIVE_CKPT_JOBS:
        _wait_for_native_metadata_files(config, timeout_seconds=1800)

    sys.stderr.write(f"[run_job] Full log: {log_file}\n")
    perf_passed = _parse_and_report(job, config, log_file, baseline_file, report_file, args.update_baseline, args.gpus_per_node)
    if not perf_passed:
        sys.exit(3)


def _parse_and_report(  # noqa: PLR0913
    job: str,
    config: str,
    log_file: Path,
    baseline_file: Path,
    report_file: Path,
    update_baseline: bool,
    gpus_per_node: int,
) -> bool:
    """Parse perf metrics from log_file, compare to baseline, write report. Returns True if passed."""
    log_text = log_file.read_text(errors="replace")

    ckpt_lines = [ln for ln in log_text.splitlines() if "save-checkpoint" in ln]
    if ckpt_lines:
        sys.stderr.write("--- save-checkpoint times ---\n")
        for ln in ckpt_lines:
            sys.stderr.write(ln + "\n")
        sys.stderr.write("-----------------------------\n")

    metrics = _parse_metrics(log_text, job)

    if not metrics:
        if job == "pretrain-nvrx":
            # nvrx provider doesn't emit parseable metrics; times printed above.
            return True
        raise RuntimeError(f"no perf metrics found in {log_file}")

    if update_baseline:
        _update_baseline_file(baseline_file, metrics, job, config, gpus_per_node)
        return True

    baseline = {}
    if baseline_file.exists():
        with baseline_file.open() as f:
            baseline = yaml.safe_load(f)

    comparison, passed = _compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
    _print_perf_summary(comparison, passed)

    report = {
        "job": job,
        "config": config,
        "gpus_per_node": gpus_per_node,
        "log": str(log_file),
        "baseline": str(baseline_file),
        "passed": passed,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in metrics.items()},
        "comparison": comparison,
    }
    with report_file.open("w") as f:
        json.dump(report, f, indent=2)
    return passed


# ---------------------------------------------------------------------------
# Perf log parsers
# ---------------------------------------------------------------------------

_RE_NATIVE_BW = re.compile(
    r"\[native_ckpt:cli\]\[\d+\]\[INFO\]\[perf\] Finished "
    r"bw_mbps=([0-9.]+).*?wr_tnsr_us=([0-9]+).*?total_written_bytes=([0-9]+)",
    re.DOTALL,
)


def _make_metric(samples: list, unit: str, higher_is_better: bool | None) -> dict:
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "unit": unit,
        "higher_is_better": higher_is_better,
    }


def _parse_metrics(log_text: str, _job: str) -> dict:
    """Parse all perf metrics for the given job type from log_text."""
    result = {}

    bw_matches = list(_RE_NATIVE_BW.finditer(log_text))
    if bw_matches:
        result.update(
            {
                "bw_mbps": _make_metric([float(m.group(1)) for m in bw_matches], "MB/s", higher_is_better=True),
                "wr_tnsr_us": _make_metric([int(m.group(2)) for m in bw_matches], "us", higher_is_better=False),
                "total_written_bytes": _make_metric([int(m.group(3)) for m in bw_matches], "B", None),
            },
        )

    return result


# ---------------------------------------------------------------------------
# Baseline comparison + reporting
# ---------------------------------------------------------------------------


def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
    baseline_metrics = baseline.get("metrics", {})
    results = {}
    all_passed = True

    for metric_name, measured in metrics.items():
        higher_is_better = measured.get("higher_is_better")
        measured_val = measured.get("mean")
        if measured_val is None:
            continue
        if metric_name in baseline_metrics:
            baseline_val = baseline_metrics[metric_name]
            tolerances = baseline.get("tolerances") or {}
            # Info-only metrics (no direction) must match exactly; directional perf metrics use default_tolerance.
            tol = tolerances.get(metric_name, 0.0 if higher_is_better is None else default_tolerance)
            lo = baseline_val * (1.0 - tol)
            hi = baseline_val * (1.0 + tol)
            passed = lo <= measured_val <= hi
            pct_change = (measured_val - baseline_val) / baseline_val * 100 if baseline_val else 0.0
            if not passed:
                all_passed = False
            results[metric_name] = {
                "status": "PASS" if passed else "FAIL",
                "measured": measured_val,
                "baseline": baseline_val,
                "lo": lo,
                "hi": hi,
                "pct_change": round(pct_change, 2),
                "tolerance_pct": round(tol * 100, 1),
                "unit": measured.get("unit", ""),
                "higher_is_better": higher_is_better,
                "min": measured.get("min"),
                "max": measured.get("max"),
                "median": measured.get("median"),
                "n_samples": len(measured.get("samples", [])),
            }
        elif higher_is_better is None:
            results[metric_name] = {"status": "info", "measured": measured_val}
        else:
            results[metric_name] = {
                "status": "no_baseline",
                "measured": measured_val,
                "unit": measured.get("unit", ""),
                "higher_is_better": higher_is_better,
                "min": measured.get("min"),
                "max": measured.get("max"),
                "median": measured.get("median"),
                "n_samples": len(measured.get("samples", [])),
            }
    return results, all_passed


def _print_perf_summary(comparison: dict, passed: bool) -> None:
    sys.stderr.write("\n" + "=" * 70 + "\n")
    sys.stderr.write(" Performance Results\n")
    sys.stderr.write("=" * 70 + "\n")
    for name, data in comparison.items():
        status = data.get("status", "?")
        measured = data.get("measured")
        baseline = data.get("baseline")
        unit = data.get("unit", "")
        pct = data.get("pct_change")
        if status == "info":
            sys.stderr.write(f"  {name}: {measured:.4f} {unit}  [informational]\n")
        elif status == "no_baseline":
            sys.stderr.write(f"  {name}: {measured:.4f} {unit}  [no baseline]\n")
        elif status in ("PASS", "FAIL"):
            sign = "+" if pct and pct >= 0 else ""
            pct_str = f"{sign}{pct:.1f}%" if pct is not None else "N/A"
            tol = data.get("tolerance_pct", "?")
            samples = data.get("n_samples", "?")
            sys.stderr.write(
                f"  [{status}] {name}: {measured:.4f} {unit}  (baseline={baseline:.4f}, Δ={pct_str}, tol=±{tol}%, n={samples})\n",
            )
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f" Overall: {'PASS' if passed else 'FAIL'}\n")
    sys.stderr.write("=" * 70 + "\n\n")


def _baseline_run_cmds(job: str, config: str, gpus_per_node: int) -> tuple[str, str]:
    script = "python3 megatron/native/ci/scripts/run_job.py"
    gflag = f"--gpus-per-node {gpus_per_node}"
    if job == "ckptbench":
        flags = f"ckptbench --slurm {gflag}"
    elif job == "pretrain" and config.startswith("16"):
        flags = f"pretrain --nodes 16 {gflag}"
    else:
        flags = f"{job} --slurm {gflag}"
    return f"{script} {flags}", f"{script} {flags} --update-baseline"


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round so only the leftmost `sig_figs` digits are non-zero (e.g. 5742.485 -> 5700.0)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, sig_figs - 1 - magnitude)


def _round_for_baseline(mean: float, tolerance: float, higher_is_better: bool | None) -> float | int:
    """Round a measured mean to the precision implied by its tolerance.

    5% (the default for perf metrics) → 2 sig figs so values are easy to read at a glance.
    Exact match (tolerance==0, e.g. info metrics like total_written_bytes) → preserve as int.
    Custom tolerances → 6 decimals, since the user picked a non-default precision deliberately.
    """
    if tolerance == 0.0:
        return int(mean) if mean == int(mean) else round(mean, 6)
    if tolerance == PERF_TOLERANCE:
        rounded = _round_to_sig_figs(mean, 2)
        # Info-only metrics that landed in the perf-default branch (no old baseline) still
        # serialize cleanly as int when the rounded value is integral.
        if higher_is_better is None and rounded == int(rounded):
            return int(rounded)
        return rounded
    return round(mean, 6)


def _update_baseline_file(baseline_file: Path, metrics: dict, job: str, config: str, gpus_per_node: int) -> None:
    if baseline_file.exists():
        with baseline_file.open() as f:
            baseline = yaml.safe_load(f)
    else:
        run_cmd, update_cmd = _baseline_run_cmds(job, config, gpus_per_node)
        baseline = {
            "description": "auto-generated by CI",
            "run_cmd": run_cmd,
            "update_cmd": update_cmd,
            "metrics": {},
        }

    existing = baseline.setdefault("metrics", {})
    tolerances = baseline.get("tolerances") or {}
    changed: list[str] = []
    for name, data in metrics.items():
        mean = data.get("mean")
        if mean is None:
            continue
        old_val = existing.get(name)
        higher_is_better = data.get("higher_is_better")
        # Effective tolerance mirrors _compare_to_baseline; for new metrics (no old_val) we
        # fall back to higher_is_better since type-of-baseline isn't available yet.
        if name in tolerances:
            tol = tolerances[name]
        elif old_val is not None:
            tol = 0.0 if isinstance(old_val, int) else PERF_TOLERANCE
        else:
            tol = 0.0 if higher_is_better is None else PERF_TOLERANCE

        new_val = _round_for_baseline(mean, tol, higher_is_better)
        if old_val is None or abs(new_val - old_val) / max(abs(old_val), 1e-9) > _BASELINE_CHANGE_THRESHOLD:
            existing[name] = new_val
            changed.append(f"  {name}: {old_val} -> {new_val}")

    if not changed:
        return

    for _line in changed:
        pass
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    with baseline_file.open("w") as f:
        yaml.dump(baseline, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Baseline aggregation (update stage — no re-run)
# ---------------------------------------------------------------------------


def _parse_job_filter(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Return (filter_job, filter_config) from --filter or CI_JOB_NAME.

    CI_JOB_NAME format: <platform>:<job>[:<config>][:update_baselines]
      gb200:ckptbench:update_baselines        → (ckptbench, None)
      gb200:pretrain:1x4:update_baselines     → (pretrain, 1x4)
    """
    if args.filter:
        job_name, *rest = args.filter.split(":", 1)
        return job_name, rest[0] if rest else None

    ci_job_name = os.environ.get("CI_JOB_NAME", "")
    if ci_job_name:
        _, job_name, *rest = ci_job_name.removesuffix(":update_baselines").split(":")
        return job_name, rest[0] if rest else None

    return None, None


def run_update_baselines(args: argparse.Namespace) -> None:
    """Update baseline YAMLs from *_report.json files in RESULTS_DIR without re-running jobs.

    Job/config are auto-detected from CI_JOB_NAME; override with --filter.
    """
    machine = _machine_tag()
    filter_job, filter_config = _parse_job_filter(args)

    job_glob = filter_job or "*"
    reports = sorted(RESULTS_DIR.glob(f"{machine}_{job_glob}_*_report.json"))
    if not reports:
        filter_desc = f"{filter_job}:{filter_config}" if filter_config else (filter_job or "all jobs")
        raise RuntimeError(
            f"no report files found in {RESULTS_DIR} matching '{machine}_{job_glob}_*_report.json' "
            f"(filter: {filter_desc}) — did the test stage run first?",
        )

    for report_file in reports:
        with report_file.open() as f:
            report = json.load(f)

        job = report.get("job")
        config = report.get("config")
        gpus_per_node = report.get("gpus_per_node", args.gpus_per_node)
        metrics = report.get("metrics", {})

        if not job or not config or not metrics:
            continue

        if filter_config is not None and config != filter_config:
            continue

        baseline_file = BASELINES_DIR / f"{_machine_tag()}_{job}_{config}.yaml"
        _update_baseline_file(baseline_file, metrics, job, config, gpus_per_node)

    if filter_job == "pretrain-native":
        _update_native_metadata_files(filter_config)


# Jobs that exercise the native checkpoint provider — these spawn the async
# checkpoint_server, so we must wait for its background writes (registry.bin,
# *.native.metadata.info) before the test stage finishes.
_NATIVE_CKPT_JOBS = ("pretrain-native", "pretrain-all", "valgrind")


def _dump_nvrx_vs_native_diffs(cfg: str, max_lines: int = 200) -> None:
    """Emit unified diff between reference.metadata.info (nvrx) and
    0.native.metadata.info (native server) for every iter dir that has both
    files on disk. Called from the in-srun pretrain failure path so the CI
    log captures what diverged — distcp_trace itself only prints the diff
    body when it's under half the file size, which the gb200 5313-diff case
    blows past."""
    checkpoints_dir = MEGATRON_ROOT / f"megatron/native/ci/ci-runs/{cfg}/checkpoints"
    if not checkpoints_dir.is_dir():
        return
    for iter_dir in sorted(checkpoints_dir.glob("iter_*")):
        ref = iter_dir / "reference.metadata.info"
        nat = iter_dir / "0.native.metadata.info"
        if not (ref.is_file() and nat.is_file()):
            continue
        ref_lines = ref.read_text().splitlines()
        nat_lines = nat.read_text().splitlines()
        diff = list(difflib.unified_diff(ref_lines, nat_lines, fromfile=str(ref), tofile=str(nat), n=1))
        if not diff:
            continue
        sys.stderr.write(f"[run_job] nvrx-vs-native metadata diff for {iter_dir.name} ({len(diff)} lines):\n")
        for line in diff[:max_lines]:
            sys.stderr.write(line + "\n")
        if len(diff) > max_lines:
            sys.stderr.write(f"[run_job] ... ({len(diff) - max_lines} more diff lines suppressed; full files in artifacts)\n")
        sys.stderr.flush()


def _foreach_native_metadata_files(
    filter_config: str | None,
    callback: Callable[[Path, Path], None],
    wait_seconds: int = 0,
) -> None:
    """Iterate over each (src, dst) native metadata file pair, invoking callback for each.

    If wait_seconds > 0, polls for each pattern's first match (per iter dir) before globbing
    — needed because checkpoint_server writes these files asynchronously in the background.
    """
    metadata_root = MEGATRON_ROOT / "megatron/native/metadata"
    ci_runs_root = MEGATRON_ROOT / "megatron/native/ci/ci-runs"
    if not ci_runs_root.is_dir():
        raise FileNotFoundError(f"ci-runs root not found: {ci_runs_root}")

    files_to_copy = ("registry.bin", "*.native.metadata.info", "*.txt")
    iter_pattern = re.compile(r"^iter_(\d{7})$")
    cfg_pattern = re.compile(r"^\d+x\d+$")

    cfgs = (
        [filter_config]
        if filter_config is not None
        else sorted(d.name for d in ci_runs_root.iterdir() if d.is_dir() and cfg_pattern.match(d.name))
    )

    for cfg in cfgs:
        ckpt_root = ci_runs_root / cfg / "checkpoints"
        if not ckpt_root.is_dir():
            raise FileNotFoundError(
                f"checkpoint dir not found for {cfg}: {ckpt_root} — did the pretrain-native test stage run on this runner?",
            )

        iter_dirs = [d for d in sorted(ckpt_root.iterdir()) if d.is_dir() and iter_pattern.match(d.name)]
        if not iter_dirs:
            raise FileNotFoundError(f"no iter_* checkpoints found under {ckpt_root}")

        for src_iter_dir in iter_dirs:
            iter_str = iter_pattern.match(src_iter_dir.name).group(1)
            dst_dir = metadata_root / f"{cfg}_checkpoints_iter_{iter_str}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for pattern in files_to_copy:
                if wait_seconds > 0:
                    wait_for_file(src_iter_dir / pattern, wait_seconds)
                matches = sorted(src_iter_dir.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"no files matching {pattern!r} in {src_iter_dir}")
                for src in matches:
                    dst = dst_dir / src.name
                    callback(src, dst)


def _wait_for_native_metadata_files(cfg: str, timeout_seconds: int) -> None:
    """Block until checkpoint_server has flushed registry.bin and *.native.metadata.info
    under ci-runs/<cfg>/checkpoints/iter_*. Call after pretrain returns but before the
    test stage finishes, so the downstream update_baselines stage finds the files."""

    def _log(src: Path, _dst: Path) -> None:
        sys.stderr.write(f"[run_job] Found native metadata file: {src}\n")

    _foreach_native_metadata_files(cfg, _log, wait_seconds=timeout_seconds)


def _update_native_metadata_files(filter_config: str | None) -> None:
    """Refresh registry.bin and native.metadata.info under megatron/native/metadata/.

    Driven by ci-runs/<cfg>/checkpoints/iter_NNNNNNN/ — every iter the just-completed
    pretrain-native run produced gets mirrored to metadata/<cfg>_checkpoints_iter_NNNNNNN/
    (creating the destination folder if needed). When filter_config is set, only that
    cfg is processed.
    """

    def log_and_copy(src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)
        sys.stderr.write(f"[run_job] updated {dst} from {src}\n")

    _foreach_native_metadata_files(filter_config, log_and_copy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _detect_local_gpu_count() -> int:
    """Return the number of visible GPUs on this host, falling back to 4."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and cuda_visible not in ("NoDevFiles", "none", ""):
        return len(cuda_visible.split(","))
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output([nvidia_smi, "--list-gpus"], text=True, stderr=subprocess.DEVNULL)
            n = sum(1 for ln in out.splitlines() if ln.strip())
            return max(n, 1)
        except (subprocess.CalledProcessError, OSError):
            pass
    return 4


def main() -> None:
    sys.stdout.write(shlex.join(sys.argv) + "\n")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "job",
        choices=["ckptbench", "pretrain", "pretrain-all", "pretrain-native", "pretrain-nvrx", "valgrind", "update_baselines"],
    )
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1 — runs inline; >1 uses SLURM automatically)")
    p.add_argument("--gpus-per-node", type=int, default=None, dest="gpus_per_node", help="GPUs per node (default: auto-detect)")
    p.add_argument("--slurm", action="store_true", help="Submit via SLURM even for single-node runs")
    p.add_argument("--skip-perf", action="store_true", dest="skip_perf", help="Skip perf parsing; tee full output to console instead")
    p.add_argument(
        "--update-baseline",
        action="store_true",
        dest="update_baseline",
        help="Record current measurements as the new baseline and exit",
    )
    p.add_argument(
        "--filter",
        default=None,
        dest="filter",
        help="update_baselines only: <job>[:<config>] to update, e.g. 'pretrain:1x4' or 'bench'. "
        "Auto-detected from CI_JOB_NAME when running in GitLab CI.",
    )
    args, extra_args = p.parse_known_args()
    args.extra_args = extra_args
    if args.gpus_per_node is None:
        args.gpus_per_node = _detect_local_gpu_count()

    if args.job == "valgrind":
        # Signal CMake that we're building for a valgrind run. CMakeLists.txt
        # turns this into CPPFORGE_DISABLE_MARCH_NATIVE (so valgrind doesn't trip
        # on host-specific instructions) and defines PHOENIX_BUILD_FOR_VALGRIND
        # which shrinks D2H ring-buffer sizes in messages.h. Must be set before
        # any `import megatron.native` triggers a CMake build; subprocess and
        # srun children inherit it via env=os.environ.copy() / container-env
        # forwarding.
        os.environ["PHOENIX_BUILD_FOR_VALGRIND"] = "1"

    if args.job == "update_baselines":
        run_update_baselines(args)
        return

    image = Path(_default_enroot_image())
    if args.slurm:
        if args.job == "pretrain-all" and not in_srun():
            _ensure_enroot_image(image)
        wait_for_file(image, timeout_seconds=3600)
    if args.job == "ckptbench":
        run_ckptbench(args)
    else:
        run_pretrain(args)


if __name__ == "__main__":
    main()
