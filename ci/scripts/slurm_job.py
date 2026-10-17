#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Slurm CI launcher.

LLM/agent maintenance constraints:

1. Training environment variables must not depend on model configuration or scale. Set
   them globally for every run, or let a flavor add/remove them uniformly.
2. Flavors must not inspect model configuration. Do not branch training policy on node
   count, GPU count, or model name without explicit user approval.
3. Model YAML is flavor-agnostic. Keep it focused on architecture and the full-performance
   recipe, including performance features such as CUDA graphs; do not put environment or
   flavor policy in model configs.
4. Cluster and scale may select unavoidable scheduler mechanics (allocation topology,
   partition, timeout, and paths), but must not silently change training semantics.
5. Do not inject model options into GitLab ``JOB_ARGS``. Put architecture/precision options
   in model YAML and uniform execution-policy transforms in the owning flavor.
6. Keep generated GitLab CI files synchronized with their template/matrix, and update
   argument baselines whenever an intentional option change affects them.
7. Keep experiment-only runner tags and pipeline scoping on ``expr/*`` branches.
8. Explain and obtain approval before changing Python or C++ checkpointing implementation.
9. Existing scale/config policy exceptions are legacy behavior. Do not copy or extend them
   without explicit approval; migrate them separately when their ownership is clarified.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import lzma
import mmap
import os
import re
import shlex
import shutil
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.request
from collections import namedtuple
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import log_report
import otel_report
import otel_to_perfetto
import wheel_install
import yaml


def _configure_worker_tmpdir() -> Path | None:
    """Use the container-local short path required by AF_UNIX consumers."""
    if "SLURM_PROCID" not in os.environ:
        return None
    tmpdir = Path("/tmp")  # noqa: S108 - writable container-local tmpfs
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    return tmpdir


_configure_worker_tmpdir()

from daemon_tools import CHECKPOINT_SERVER  # noqa: E402

# ruff: noqa: FBT003, PLR0913, PLR0917, PLR2004, TC003
# fmt: off
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR / "../..").resolve()
BASELINES_DIR = (SCRIPT_DIR / "../baselines").resolve()
CONFIGS_DIR = (SCRIPT_DIR / "../configs").resolve()
# One entry per hardware platform, shared with the pipeline generator.
JOBS_FILE = (SCRIPT_DIR / "../jobs.json").resolve()
FLAVORS_FILE = SCRIPT_DIR / "flavors.yaml"
TEMPLATES_DIR = (SCRIPT_DIR / "../templates").resolve()
ENROOT_SCRIPTS_DIR = REPO_ROOT / "scripts"
BANDWIDTH_BASELINE_FILE = BASELINES_DIR / "platform_bw_mbps.yaml"
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.09-py3")
DEFAULT_MEGATRON_REPO_URL = "https://github.com/NVIDIA/Megatron-LM.git"
DEFAULT_MEGATRON_COMMIT = "16251ac1221f4814cd39350def05cefa3c36734a"
DEFAULT_NVRX_REPO_URL = "https://github.com/NVIDIA/nvidia-resiliency-ext.git"
DEFAULT_NVRX_COMMIT = "73b17d9bcff543a0a5f64e95d64f26899d730653"
_NVRX_DISTRIBUTION = "nvidia-resiliency-ext"
_NVRX_MODULE = "nvidia_resiliency_ext"
SOFT_FAIL_RC = 3
PERF_TOLERANCE = 0.05
HOST_MEMORY_TOLERANCE = 0.10
MAX_BW_SOCKET_IMBALANCE = 1.30
# Durability spans the whole distributed write, so it carries the shared filesystem's
# variance rather than the checkpointer's: on a busy Lustre the same job can differ
# several-fold. Track the number, but only fail on a change too large for that to
# explain. A baseline's own `tolerances` entry still wins over this.
WIDE_TOLERANCE_METRICS = {
    "ckpt_durable_ms": 5.0,
    "ckpt_daemon_durable_ms": 2.0,
}
HOST_MEMORY_METRICS = frozenset({
    "checkpoint_daemon_peak_rss_bytes",
    "training_process_peak_rss_bytes",
})
GPU_MEMORY_METRICS = frozenset({"gpu_peak_bytes", "gpu_hbm_peak_bytes"})
MEMORY_REGRESSION_METRICS = HOST_MEMORY_METRICS | GPU_MEMORY_METRICS
RESULTS_DIR = REPO_ROOT / "ci/ci-results"
RUN_ROOT: Path | None = None
# The allocation directory owns every named run.
JOB_ROOT: Path | None = None
# Parent of every run directory. Holds what runs share: the dependency
# checkouts, the native build and the compiler caches, none of which depend on
# what a run is testing.
WORK_ROOT: Path | None = None
RUN_NAME: str | None = None
logger = logging.getLogger("native_ci")
_PRETRAIN_ENTRY_CMD = "__pretrain_entry__"
_EPILOG_ENTRY_CMD = "__epilog__"
_RANK_LOGS_ENTRY_CMD = "__rank_logs__"
_FAILURE_SUMMARY_ENTRY_CMD = "__failure_summary__"
_RECOVER_OTEL_ENTRY_CMD = "__recover_otel__"
# GitLab traces are size limited. Rank 0 is already mirrored in full while the
# run is active, so retain the failure and traceback tail for the ranks that
# Slurm reports exiting.
_RANK_LOG_TAIL_LINES = 120
_RANK_LOG_LIMIT = 4
_DIGEST_DIFF_DETAIL_LIMIT = 12
_FAILURE_SUMMARY_LIMIT = 12
_FAILURE_SUMMARY_TAIL_BYTES = 512 * 1024
_SLURM_STATUS_PERIOD_SECONDS = 5 * 60
_PHASE1_TRAIN_ITERS = 100
_LONG_RUN_TRAIN_SAMPLES = 1_000_000_000
_CLEANUP_GRACE_MINUTES = 5
_PRETRAIN_ENTRYPOINTS = frozenset({"gpt", "hybrid"})

# One array task owns one NVLink domain: TP/EP collectives stay on NVLink and
# --ft-segment can evict and replace a whole domain at once. Any job wider than
# one domain launches as a shared-rendezvous job array of domain-sized tasks.
_JOB_ARRAY_SEGMENT_DEFAULT = 16
# Ranks coordinate through files under this directory of a run, never beside
# its logs.
SYNC_DIR_NAME = ".sync"
_TORCH_STORAGE_ALIGNMENT = 64
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_PROC_STATUS_RSS_RE = re.compile(r"^VmRSS:\s+(\d+)\s+kB$", re.MULTILINE)
_PROC_ROOT = Path("/proc")
_WORKTREE_REVISION = "HEAD"
# A run directory is named for what it holds: the model it trained, the sources
# it trained on, and the variants it carried. Two runs differing in any of those
# keep their own checkpoints and logs; two runs alike in all of them share a
# directory and the later one supersedes the earlier.
_DEFAULT_NAME = "{run}{config}_{nodes}n-{flavors}{sources}"
_TRAIN_ENV_DEFAULTS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    "MASTER_PORT": "29500",
    "NATIVE_CKPT_OTEL_EXPORT": "daemon",
}
_MEGATRON_OTEL_ENV_DEFAULTS = {
    "MEGATRON_OTEL_ENABLED": "1",
    "MEGATRON_OTEL_METRICS_ENABLED": "1",
    "MEGATRON_OTEL_EXPORT_STRATEGY": "all_ranks",
    "MEGATRON_OTEL_SPAN_GROUPS": "default,load_checkpoint",
    "MEGATRON_OTEL_TRACES_ENABLED": "1",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4100",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
}
_PRETRAIN_ENV_NAMES = {
    "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH",
    "CUBLAS_WORKSPACE_CONFIG",
    "ENROOT_IMAGE",
    "GPUS_PER_NODE",
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MAMBA_DETERMINISTIC",
    "MASTER_ADDR",
    "NATIVE_CKPT_OTEL_COLLECTOR",
    "NATIVE_CKPT_OTEL_FILE",
    "NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS",
    "NATIVE_CKPT_SERVER_LOG",
    "NCCL_ALGO",
    "NCCL_NET",
    "NCCL_NVLS_ENABLE",
    "NCCL_PROTO",
    "NCCL_TUNER_PLUGIN",
    "NVRX_OTEL_USE_SYSTEM_RANDOM",
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NVRX_CURRENT_CYCLE_INFO",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "NVTE_CPU_OFFLOAD_V1",
    "NVTE_FLASH_ATTN_V4",
    "RANK",
    "PYTHONDONTWRITEBYTECODE",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_PROCID",
    "SLURM_STEP_NODELIST",
    "SLURM_TASKS_PER_NODE",
    "TORCHELASTIC_RESTART_COUNT",
    "WORLD_SIZE",
    *_MEGATRON_OTEL_ENV_DEFAULTS,
    "MEGATRON_OTEL_RUN_ID",
}
_ARGUMENTS_RE = re.compile(
    r"-{24} arguments -{24}.*?-{20} end of arguments -{21}",
    re.DOTALL,
)

def sys_path_insert(path: Path) -> None:
    value = path.as_posix()
    if value not in sys.path:
        sys.path.insert(0, value)
    entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(":") if entry]
    if value not in entries:
        os.environ["PYTHONPATH"] = ":".join([value, *entries])

def _otel_reporting_enabled(flavors: FlavorSelection | None = None) -> bool:
    return (flavors is None or flavors.policy.otel_reporting) and (
        bool(os.environ.get("NATIVE_CKPT_OTEL_FILE"))
        or os.environ.get("MEGATRON_OTEL_ENABLED", _MEGATRON_OTEL_ENV_DEFAULTS["MEGATRON_OTEL_ENABLED"]) == "1"
    )


def _job_array_segment(args: argparse.Namespace) -> int:
    """Nodes per job-array task; 0 keeps the classic single-allocation launch.

    Unset (no --job-array-segment and no SLURMJOB_OPT_JOB_ARRAY_SEGMENT) launches
    any job wider than one 16-node NVLink domain as a shared-rendezvous array of
    16-node tasks.
    """
    value = getattr(args, "job_array_segment", None)
    if value is not None:
        return value
    return _JOB_ARRAY_SEGMENT_DEFAULT if args.nodes > _JOB_ARRAY_SEGMENT_DEFAULT else 0


def _in_job_array() -> bool:
    """Whether the run trains under ft_launcher across a Slurm job-array shared rendezvous."""
    return int(os.environ.get("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "") or 0) > 0


def _rank_env(primary: str, fallback: str, default: str) -> str:
    """Read a distributed-rank variable, preferring the launcher's names in job-array mode.

    The ft_launcher srun step runs one Slurm task per node, so the SLURM_* rank
    variables describe launcher agents — they repeat on every node and count nodes,
    not ranks. The launcher sets RANK, LOCAL_RANK, WORLD_SIZE and LOCAL_WORLD_SIZE
    for the worker processes it spawns, so those win when it drives the launch.
    """
    if _in_job_array():
        primary, fallback = fallback, primary
    return os.environ.get(primary, os.environ.get(fallback, default))


def _sync_job_id() -> str:
    """The job id every rank of this run shares, for file-based synchronization.

    Array tasks carry distinct SLURM_JOB_IDs, so the shared-rendezvous world keys
    its markers by the array id instead.
    """
    if not _in_job_array():
        return os.environ.get("SLURM_JOB_ID", "local")
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    # ft_launcher re-executes the worker entrypoint on every elastic cycle.
    # Markers from a previous cycle must not satisfy the new cycle's barriers
    # (and a stale .failed marker must not poison it permanently).
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    return f"{job_id}-restart{restart}"


def _install_torch_serialization_compat() -> bool:
    """Supply config omitted by the PyTorch wheel in the 26.07 image.

    PyTorch's own serialization.py imports torch.utils.serialization.config for
    torch.save/load settings, but the nv26.7 wheel does not contain that package.
    Preserve the upstream defaults and install the shim only for that exact
    missing-module failure. Returns whether a shim was installed.
    """
    try:
        importlib.import_module("torch.utils.serialization")
    except ModuleNotFoundError as error:
        if error.name != "torch.utils.serialization":
            raise
    else:
        return False

    torch_utils = importlib.import_module("torch.utils")
    config = ModuleType("torch.utils.serialization.config")
    config.load = SimpleNamespace(  # type: ignore[attr-defined]
        mmap=False,
        endianness=None,
        mmap_flags=None if sys.platform == "win32" else mmap.MAP_PRIVATE,
        calculate_storage_offsets=False,
    )
    config.save = SimpleNamespace(  # type: ignore[attr-defined]
        compute_crc32=True,
        use_pinned_memory_for_d2h=False,
        storage_alignment=_TORCH_STORAGE_ALIGNMENT,
    )
    package = ModuleType("torch.utils.serialization")
    package.__path__ = []  # type: ignore[attr-defined]
    package.config = config  # type: ignore[attr-defined]
    sys.modules[package.__name__] = package
    sys.modules[config.__name__] = config
    torch_utils.serialization = package
    return True


def shell_join(command: list[str]) -> str:
    refs = sorted(((v, n) for n in _SHELL_ENV_REFS if (v := os.environ.get(n))), key=lambda x: len(x[0]), reverse=True)
    words = []
    for token in command:
        rendered, placeholders = token, {}
        for index, (value, name) in enumerate(refs):
            if value not in rendered:
                continue
            placeholder = f"\0{index}\0"
            if name == "HOME" and rendered.startswith("--container-mounts="):
                prefix, mounts = rendered.split("=", 1)
                pairs = [mount.partition(":") for mount in mounts.split(",")]
                updated = prefix + "=" + ",".join(host.replace(value, placeholder) + sep + target for host, sep, target in pairs)
                if updated == rendered:
                    continue
                rendered = updated
            elif "/" not in value and len(value) < 20 and rendered != value:
                if not rendered.endswith(f"={value}"):
                    continue
                rendered = rendered[: -len(value)] + placeholder
            else:
                rendered = rendered.replace(value, placeholder)
            placeholders[placeholder] = name
        if not placeholders:
            words.append(shlex.quote(token))
            continue
        rendered = rendered.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        for placeholder, name in placeholders.items():
            rendered = rendered.replace(placeholder, f"${{{name}}}")
        words.append(f'"{rendered}"')
    return " ".join(words)

def wait_for_file(path: Path, timeout_seconds: int, *, poll_s: int = 5) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if any(path.parent.glob(path.name)) if any(c in path.name for c in "*?[") else path.exists():
            return
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for {path}")

def image_path() -> Path:
    digest = hashlib.sha256((ENROOT_SCRIPTS_DIR / "prepare.sh").read_bytes()).hexdigest()[:10]
    return Path(os.environ.get("ENROOT_IMAGE", f"{Path.home()}/data/native.{digest}.sqsh")).expanduser().resolve()
def megatron_root() -> Path:
    if value := os.environ.get("MEGATRON_ROOT"):
        root = Path(value).expanduser().resolve()
        if (root / ".git").exists():
            return root
    out = subprocess.check_output(["bash", str(ENROOT_SCRIPTS_DIR / "megatron_root.sh")], text=True)  # noqa: S607
    return Path(out.strip()).resolve()
def platform_tag() -> str:
    return os.environ.get("PLATFORM", "").strip() or socket.gethostname().split(".")[0]
def platform_gpus_per_node() -> int | None:
    """How many GPUs a node of this platform has, from the platform matrix.

    The count is a property of the hardware, so naming the platform is enough
    and a run only states it to deviate.
    """
    if not (platform := platform_tag()):
        return None
    for entry in json.loads(JOBS_FILE.read_text())["platforms"]:
        if entry["name"] == platform:
            return int(str(entry["gpus"]).rpartition("x")[2])
    return None

def _gpu_compute_capability() -> str:
    """Compute capability of this node's first GPU, read from the driver without initializing CUDA."""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        if _rank_env("SLURM_PROCID", "RANK", "0") == "0":
            logger.warning(f"[slurm_job:env] GPU compute capability unavailable ({error}); attention backend defaults unchanged")
        return ""

def cluster_tag() -> str:
    return os.environ.get("CLUSTER", "").strip()
def _run_dir(cfg: str) -> Path:
    return RUN_ROOT if RUN_ROOT is not None else REPO_ROOT / "ci/ci-runs" / cfg
def nvrx_repo_root() -> Path | None:
    value = os.environ.get("NVRX_REPO_ROOT")
    return Path(value).expanduser().resolve() if value else None

def _normalize_nvrx_repo_root() -> None:
    if (resolved := nvrx_repo_root()) is None:
        return
    if any(c in resolved.as_posix() for c in ",:"):
        raise SystemExit("NVRX_REPO_ROOT may not contain ',' or ':'")
    os.environ["NVRX_REPO_ROOT"] = resolved.as_posix()

def _nvrx_target_root(cfg: str) -> Path:
    return (JOB_ROOT or _run_dir(cfg)) / "deps"

def _nvrx_target(cfg: str, commit: str) -> Path:
    return _nvrx_target_root(cfg) / f"nvidia-resiliency-ext-{commit}"

@dataclass(frozen=True)
class Toolchain:
    """Megatron-LM and NVIDIA Resiliency Extension coordinates a run trains on.

    The checkout is keyed by commit, so runs of different sources keep their own
    directories and a training subprocess is pointed at one through its
    environment.
    """

    megatron_url: str
    megatron_commit: str | None
    megatron_root: Path
    nvrx_url: str
    nvrx_commit: str | None
    nvrx_target: Path | None
    nvrx_target_root: Path

    def describe(self, label: str) -> str:
        nvrx = f"{nvrx_repo_root()}@HEAD" if self.nvrx_commit == _WORKTREE_REVISION else f"{self.nvrx_url}@{self.nvrx_commit}"
        return f"{label}: megatron={self.megatron_url}@{self.megatron_commit or 'HEAD'} nvrx={nvrx}"

    def apply(self, env: dict[str, str], *, include_nvrx: bool = True) -> None:
        """Point one training subprocess at this toolchain."""
        env["MEGATRON_ROOT"] = self.megatron_root.as_posix()
        env["MEGATRON_REPO_URL"] = self.megatron_url
        if self.megatron_commit:
            env["MEGATRON_COMMIT"] = self.megatron_commit
        else:
            env.pop("MEGATRON_COMMIT", None)
        if self.megatron_commit == _WORKTREE_REVISION:
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["NVRX_REPO_URL"] = self.nvrx_url
        if include_nvrx and self.nvrx_commit:
            env["NVRX_COMMIT"] = self.nvrx_commit
        else:
            env.pop("NVRX_COMMIT", None)
        entries = [entry for entry in env.get("PYTHONPATH", "").split(":") if entry]
        entries = [
            entry
            for entry in entries
            if not (
                Path(entry).parent == self.nvrx_target_root
                and Path(entry).name.startswith("nvidia-resiliency-ext")
            )
        ]
        if include_nvrx and self.nvrx_target is not None:
            head = self.nvrx_target.as_posix()
            entries = [head, *(entry for entry in entries if entry != head)]
        if entries:
            env["PYTHONPATH"] = ":".join(entries)
        else:
            env.pop("PYTHONPATH", None)

def _toolchain(cfg: str, args: argparse.Namespace) -> Toolchain:
    """Resolve the Megatron-LM and extension sources this run trains on.

    A run trains on one toolchain. Comparing two of them is comparing the
    digests two runs wrote, which needs neither to know about the other.
    """
    megatron_commit = args.megatron_commit or DEFAULT_MEGATRON_COMMIT
    nvrx_commit = args.nvrx_commit or DEFAULT_NVRX_COMMIT
    megatron_checkout = (
        Path(os.environ["MEGATRON_ROOT"]).expanduser().resolve()
        if megatron_commit == _WORKTREE_REVISION
        else _megatron_checkout(args, megatron_commit)
    )
    return Toolchain(
        args.megatron_repo_url or DEFAULT_MEGATRON_REPO_URL,
        megatron_commit,
        megatron_checkout,
        args.nvrx_repo_url or DEFAULT_NVRX_REPO_URL,
        nvrx_commit,
        _nvrx_target(cfg, nvrx_commit),
        _nvrx_target_root(cfg),
    )
def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}
def _repo_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip()[:8]

def _megatron_checkout(args: argparse.Namespace, commit: str | None = None) -> Path:
    """Where this run's Megatron-LM lives, keyed so runs of one commit share it."""
    root = WORK_ROOT if WORK_ROOT is not None else REPO_ROOT / "workdir"
    revision = commit or args.megatron_commit or DEFAULT_MEGATRON_COMMIT
    return root / f"deps/megatron-lm-{revision[:8]}"

def _source_suffix(args: argparse.Namespace) -> str:
    """Names only source deviations common to every run in the job.

    A job on the pinned sources says nothing. Per-run deviations belong to the
    child directory for that run rather than making the allocation directory
    read as though it represents only one side of the comparison.
    """
    parts = []
    if args.megatron_commit and args.megatron_commit != DEFAULT_MEGATRON_COMMIT:
        parts.append(f"-mlm-{args.megatron_commit[:8]}")
    if args.nvrx_commit and args.nvrx_commit != DEFAULT_NVRX_COMMIT:
        parts.append(f"-nvrx-{args.nvrx_commit[:8]}")
    return "".join(parts)

def _run_name(template: str | None, args: argparse.Namespace, run: str = "") -> str:
    job_id = os.environ.get("CI_JOB_ID", "").strip()
    default = _DEFAULT_NAME
    template = (template or ("{gitlab_pipeline_jobid}_" if job_id else "") + default).strip()
    if not job_id:
        template = template.replace("{gitlab_pipeline_jobid}_", "")
    values = {
        "gitlab_pipeline_jobid": job_id,
        "flavors": _selected_flavors(args).tag,
        "config": args.config,
        "nodes": args.nodes,
        "mlm_abbreviated_sha_commit": (args.megatron_commit or "image")[:8],
        "nvrx_abbreviated_sha_commit": (args.nvrx_commit or "image")[:8],
        "sources": _source_suffix(args),
        "run": f"{run}." if run else "",
    }
    try:
        name = template.format_map(values)
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid run-name template: {error}") from error
    if not _NAME_RE.fullmatch(name):
        raise SystemExit(
            "--name must expand to only letters, digits, '.', '_' and '-' and may not start with punctuation",
        )
    return name
def _configure_run(args: argparse.Namespace) -> None:
    global JOB_ROOT, RESULTS_DIR, RUN_NAME, RUN_ROOT, WORK_ROOT  # noqa: PLW0603
    _normalize_nvrx_repo_root()
    os.environ["SLURMJOB_OPT_LAUNCHER_ROOT"] = REPO_ROOT.as_posix()
    name = _run_name(args.name, args)
    root = (args.work_dir or REPO_ROOT / "workdir").expanduser().resolve()
    WORK_ROOT = root
    RUN_NAME, JOB_ROOT = name, root / name
    RUN_ROOT, RESULTS_DIR = JOB_ROOT, JOB_ROOT / "logs"
    # Every run writes inside this directory, so removing it removes the whole
    # job. Cleaning happens on the launcher because the runs themselves are
    # performed by every rank of an allocation at once.
    if args.clean_run and not in_srun() and RUN_ROOT.exists():
        if RUN_ROOT.is_symlink():
            raise SystemExit(f"--clean-run refuses to remove a symlink: {RUN_ROOT}")
        logger.info(f"[slurm_job] cleaning run directory: {RUN_ROOT}")
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not in_srun():
        logger.info(f"[slurm_job] run directory: {RUN_ROOT}")
    args.name, args.work_dir = name, root
    args.image = args.image.expanduser().resolve() if args.image else image_path()
    for dest, option in _RUN_OPTIONS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if option.action in ("append", "extend"):
            os.environ[option.env] = ",".join(value)
        else:
            os.environ[option.env] = str(int(value) if isinstance(value, bool) else value)
    os.environ.update(
        SLURMJOB_OPT_RUN_DIR=str(RUN_ROOT),
        SLURMJOB_OPT_WORK_ROOT=str(WORK_ROOT),
        SLURMJOB_OPT_PYTHON_INSTALL_DIR=str(WORK_ROOT / "deps/python"),
        MEGATRON_ROOT=os.environ.get("MEGATRON_ROOT", str(_megatron_checkout(args))),
        MEGATRON_NATIVE_BUILD_DIR=str(WORK_ROOT / f"build/nvrx_checkpointing-{_repo_sha()}"),
        ENROOT_IMAGE_DIR=str(args.image.parent),
        # Publish the resolved array segment so workers and the materialized
        # script see the value the launcher was generated with, not the unset
        # (auto) form.
        SLURMJOB_OPT_JOB_ARRAY_SEGMENT=str(_job_array_segment(args)),
    )
    for name, default in _TRAIN_ENV_DEFAULTS.items():
        os.environ.setdefault(name, default)
    # Compiler and dataset caches are keyed by their own contents, so every run
    # reads the one another run filled instead of paying the first compile again.
    for name, directory in (
        ("TRITON_CACHE_DIR", "caches/triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "caches/torchinductor"),
        ("TORCH_EXTENSIONS_DIR", "caches/torch-extensions"),
    ):
        os.environ.setdefault(name, str(WORK_ROOT / directory))
    os.environ.setdefault("TRITON_CACHE_PATH", os.environ["TRITON_CACHE_DIR"])
    os.environ.setdefault("NATIVE_CKPT_SERVER_LOG", str(RESULTS_DIR / "checkpoint_server.%h.%p.log"))

def ensure_vcpkg_root() -> None:
    os.environ.setdefault("VCPKG_ROOT", str(WORK_ROOT / "deps/vcpkg" if WORK_ROOT else REPO_ROOT.parent / "vcpkg"))
    os.environ.setdefault(
        "SLURMJOB_OPT_PYTHON_INSTALL_DIR",
        str(WORK_ROOT / "deps/python" if WORK_ROOT else REPO_ROOT / "build/wheel-install"),
    )
    sys_path_insert(Path(os.environ["SLURMJOB_OPT_PYTHON_INSTALL_DIR"]))

def uninstall_nvrx_for_native() -> None:  # noqa: C901
    """Hide NVRx inside a marked native worker container and prove it is absent.

    ``pip uninstall`` can block indefinitely when several allocated nodes
    mutate their writable container overlays together. Renaming the package
    and its metadata on the same filesystem is atomic, bounded, and reversible
    until the ephemeral container exits.
    """
    if RUN_ROOT is None:
        raise RuntimeError("run directory is not configured")
    host = socket.gethostname().split(".", 1)[0]
    lock_path = RUN_ROOT / f".nvrx-uninstall-{host}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        # Ranks on a node may share the writable container filesystem. Serialize their
        # package inspection/removal while retaining one lock per node at large scale.
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            distribution = importlib.metadata.distribution(_NVRX_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            logger.info(f"[slurm_job] {_NVRX_DISTRIBUTION} is already uninstalled")
        else:
            spec = importlib.util.find_spec(_NVRX_MODULE)
            locations = list(spec.submodule_search_locations or ()) if spec is not None else []
            module_path = Path(locations[0] if locations else spec.origin or "") if spec is not None else None
            metadata_path = Path(getattr(distribution, "_path", ""))
            if module_path is None or module_path.name != _NVRX_MODULE:
                raise RuntimeError(f"cannot safely locate {_NVRX_MODULE} for native perf verification")
            if not metadata_path.name.endswith((".dist-info", ".egg-info")):
                raise RuntimeError(f"cannot safely locate {_NVRX_DISTRIBUTION} metadata: {metadata_path}")
            quarantine = f"ci-disabled-{os.environ.get('SLURM_JOB_ID', 'allocation')}-{host}"
            logger.info(f"[slurm_job] quarantining {_NVRX_DISTRIBUTION} for native perf verification")
            for source in (module_path, metadata_path):
                target = source.with_name(f".{source.name}.{quarantine}")
                if target.exists():
                    raise RuntimeError(f"native perf quarantine already exists: {target}")
                source.rename(target)
                logger.info(f"[slurm_job] quarantined {source} as {target}")
        importlib.invalidate_caches()
        try:
            importlib.metadata.distribution(_NVRX_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"{_NVRX_DISTRIBUTION} remains installed after uninstall")
        if importlib.util.find_spec(_NVRX_MODULE) is not None:
            raise RuntimeError(f"{_NVRX_MODULE} remains importable after uninstall")
        logger.info(f"[slurm_job] verified {_NVRX_DISTRIBUTION} is not installed or importable")

def in_srun() -> bool:
    return "SLURM_PROCID" in os.environ
def runner_gpus_per_node() -> int | None:
    return int(os.environ["GPUS_PER_NODE"]) if os.environ.get("GPUS_PER_NODE", "").strip() else None

def _load_template(name: str) -> str:
    cluster = cluster_tag()
    if not cluster:
        raise FileNotFoundError(f"CLUSTER is required to select Slurm template {name!r}")
    cluster_name = f"{cluster}.{name}"
    paths = [
        Path(os.environ.get("CI_DIR", str(Path.home() / "data/ci"))) / cluster_name,
        TEMPLATES_DIR / cluster_name,
    ]
    for path in paths:
        if path.is_file():
            return path.read_text()
    raise FileNotFoundError(f"Slurm template {name!r} not found for cluster {cluster!r}: {paths}")

def build_cmd_from_template(name: str, values: dict[str, object], payload: list[str]) -> list[str]:
    text = "\n".join(
        line for line in _load_template(name).splitlines() if not line.lstrip().startswith("#")
    )
    return [*shlex.split(string.Template(text).safe_substitute(values)), *payload]

def _srun_tokens(values: dict[str, object]) -> list[str]:
    tokens = build_cmd_from_template("srun.tmpl", values, [])
    if not tokens or tokens[0] != "srun":
        raise RuntimeError("srun.tmpl must render to an srun command")
    required = [RUN_ROOT.parent.resolve()] if RUN_ROOT is not None else []
    required += [root.parent for root in (REPO_ROOT, nvrx_repo_root()) if root is not None]
    if required:
        for index, token in enumerate(tokens):
            if token.startswith("--container-mounts="):
                hosts = [Path(m.split(":", 1)[0]).expanduser().resolve() for m in token.split("=", 1)[1].split(",")]
                for parent in dict.fromkeys(required):
                    if not any(parent.is_relative_to(host) for host in hosts):
                        tokens[index] += f",{parent}:{parent}"
                        hosts.append(parent)
                break
    return tokens

def _override_command_option(tokens: list[str], option: str, value: str | None) -> list[str]:
    if not value:
        return tokens
    prefix = f"{option}="
    found = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith(prefix):
            tokens[index] = prefix + value
            found = True
        elif token == option:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"{option} in srun.tmpl is missing its value")
            tokens[index + 1] = value
            found = True
            index += 1
        index += 1
    if not found:
        tokens.insert(1, prefix + value)
    return tokens

def _slurm_values(
    nodes: int, ntasks: int, time_minutes: int, job_name: str, image: str,
    container_env: str, gpus_per_node: int, total_gpus: int,
) -> dict[str, object]:
    account = os.environ.get("SLURM_ACCOUNT", "").strip()
    if not account:
        raise RuntimeError("SLURM_ACCOUNT is not set")
    return {
        "account": account,
        "nodes": nodes,
        "ntasks": ntasks,
        "time_minutes": time_minutes,
        "slurm_time": f"{time_minutes // 60:02d}:{time_minutes % 60:02d}:00",
        "job_name": job_name,
        "image": image,
        "container_env": container_env,
        # Lyris names its hardware partitions after the platform. Other
        # cluster templates do not consume this value.
        "partition": platform_tag(),
        "gpus_per_node": gpus_per_node,
        "total_gpus": total_gpus,
        "home": Path.home(),
        "root": Path(image).parent,
    }

def _reaper(time_minutes: int) -> str:
    value = {
        "OccupiedIdleGPUsJobReaper": {"exemptIdleTimeMins": str(time_minutes), "reason": "other", "description": "native-checkpoint-run"},
    }
    return "--comment=" + json.dumps(value, separators=(",", ":"))
def build_srun_cmd(
    *, nodes: int, ntasks: int, time_minutes: int, job_name: str, image: str,
    container_env: str, gpus_per_node: int, total_gpus: int, extra_srun_args: list[str] | None = None,
    output: Path | None = None, append_output: bool = False, container: bool = True,
    partition: str | None = None, kill_on_bad_exit: bool = True,
) -> list[str]:
    values = _slurm_values(nodes, ntasks, time_minutes, job_name, image, container_env, gpus_per_node, total_gpus)
    output_args = [f"--output={output}", f"--open-mode={'append' if append_output else 'truncate'}"] if output else []
    tokens = _override_command_option(_srun_tokens(values), "--partition", partition)
    if not container:
        tokens = [token for token in tokens if not token.startswith("--container-")]
    return [
        *tokens,
        f"--kill-on-bad-exit={int(kill_on_bad_exit)}",
        _reaper(time_minutes),
        "--label",
        *output_args,
        *(extra_srun_args or []),
    ]

def build_sbatch_script(
    *, nodes: int, ntasks: int, time_minutes: int, job_name: str, image: str,
    container_env: str, gpus_per_node: int, total_gpus: int, body: str,
    partition: str | None = None, nodes_per_task: int = 0,
) -> tuple[str, str]:
    active_tasks = 0
    if nodes_per_task:
        # One NVLink domain per task; the extra active task is the hot
        # spare and the rest of the array backfills consumed tasks.
        active_tasks = nodes // nodes_per_task + 1
        nodes, ntasks = nodes_per_task, nodes_per_task * gpus_per_node
    if ntasks != nodes * gpus_per_node:
        raise ValueError("exclusive allocations require one task per GPU")
    values = _slurm_values(
        nodes, ntasks, time_minutes,
        job_name, image, container_env, gpus_per_node, total_gpus,
    )
    step_only = ("--container-", "--gpus=")
    tokens = _override_command_option(_srun_tokens(values), "--partition", partition)
    flags = [f for f in tokens[1:] if f not in ("-l", "--label") and not f.startswith(step_only)]
    flags += [
        f"--ntasks-per-node={gpus_per_node}",
        "--dependency=singleton",
        f"--signal=TERM@{_CLEANUP_GRACE_MINUTES * 60}",
    ]
    if nodes_per_task:
        flags += [f"--array=0-{2 * active_tasks - 1}%{active_tasks}", "--no-requeue"]
    segment = os.environ.get("SLURM_SEGMENT", "").strip() or (
        str(nodes_per_task) if nodes_per_task else ("16" if nodes >= 16 else "")
    )
    if segment:
        flags.append(f"--segment={segment}")
    flags.append("--exclusive")
    reaper_flag, reaper_json = _reaper(time_minutes).split("=", 1)
    reaper = f"{reaper_flag}={shlex.quote(reaper_json)}"
    return "#!/bin/bash\n" + "\n".join(f"#SBATCH {flag}" for flag in flags) + f"\n\n{body.rstrip()}\n", reaper

def _materialize(script: str, path: Path, args: list[str], env: dict[str, str] | None, batch_output: Path | None = None) -> None:
    head, body = script.split("\n\n", 1)
    source = env or os.environ
    names = _RUN_ENV_NAMES | {key for key in source if key.startswith(("CUDA_", "NCCL_", "NVTE_", "TORCHINDUCTOR_", "UCX_"))}
    derived = {}
    image = source.get("ENROOT_IMAGE")
    if image and source.get("ENROOT_IMAGE_DIR") == str(Path(image).parent):
        derived["ENROOT_IMAGE_DIR"] = '"${ENROOT_IMAGE%/*}"'
    run_dir = source.get("SLURMJOB_OPT_RUN_DIR")
    if run_dir and source.get("SLURMJOB_OPT_WORK_DIR") and source.get("SLURMJOB_OPT_RUN_NAME"):
        derived["SLURMJOB_OPT_RUN_DIR"] = '"${SLURMJOB_OPT_WORK_DIR}/${SLURMJOB_OPT_RUN_NAME}"'
        if source.get("SLURMJOB_OPT_WORK_ROOT") == source["SLURMJOB_OPT_WORK_DIR"]:
            derived["SLURMJOB_OPT_WORK_ROOT"] = '"${SLURMJOB_OPT_WORK_DIR}"'
    for parent, keys in (
        ("SLURMJOB_OPT_RUN_DIR", (("NATIVE_CKPT_SERVER_LOG", "/logs/checkpoint_server.%h.%p.log"),)),
        ("SLURMJOB_OPT_WORK_ROOT", (
            ("SLURMJOB_OPT_PYTHON_INSTALL_DIR", "/deps/python"),
            ("TRITON_CACHE_DIR", "/caches/triton"),
            ("TRITON_CACHE_PATH", "/caches/triton"),
            ("TORCHINDUCTOR_CACHE_DIR", "/caches/torchinductor"),
            ("TORCH_EXTENSIONS_DIR", "/caches/torch-extensions"),
            ("VCPKG_ROOT", "/deps/vcpkg"),
        )),
    ):
        if not (base := source.get(parent)):
            continue
        for key, suffix in keys:
            if source.get(key) == base + suffix:
                derived[key] = f'"${{{parent}}}{suffix}"'
    exports = [
        *(f"export {key}={shlex.quote(source[key])}" for key in sorted(names - derived.keys()) if source.get(key) is not None),
        *(f"export {key}={value}" for key, value in derived.items()),
    ]
    export_text = "\n".join(exports)
    directives = "\n".join([*(f"#SBATCH {arg}" for arg in args), f"#SBATCH --output={batch_output or os.devnull}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{head}\n# Generated by ci/scripts/slurm_job.py; submit with: sbatch {shlex.quote(path.name)}\n"
        f"{directives}\n\n{export_text}\n\n{body}",
    )
    path.chmod(0o755)

def _record_submission(output: str, script_path: Path) -> str:
    job_id = output.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch did not return a numeric job ID: {output.strip()!r}")
    root = RUN_ROOT or script_path.parent
    (root / "job_id").write_text(f"{job_id}\n")
    if gitlab_job_id := os.environ.get("CI_JOB_ID", "").strip():
        (root / "gitlab_job_id").write_text(f"{gitlab_job_id}\n")
    logger.info(f"[slurm_job] submitted job {job_id}")
    return job_id

def _cancel_slurm_job(job_id: str) -> None:
    logger.warning(f"[slurm_job] canceling Slurm job {job_id}")
    proc = subprocess.run(["scancel", "--", job_id], check=False)  # noqa: S607
    if proc.returncode:
        logger.warning(f"[slurm_job] scancel exited with code {proc.returncode} for job {job_id}")

def _report_queue_status(job_id: str) -> bool:
    """Print pending scheduler state; return whether queue polling should continue."""
    command = [
        "squeue", "--noheader", "--jobs", job_id,
        "--format=%i|%T|%M|%l|%S|%R",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S607
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if proc.returncode:
        logger.warning(
            f"[slurm_job:status] {timestamp} squeue failed for job {job_id}: "
            f"{proc.stderr.strip() or f'exit {proc.returncode}'}",
        )
        return True
    rows = proc.stdout.strip().splitlines()
    if not rows:
        logger.info(f"[slurm_job:status] {timestamp} job {job_id} is no longer listed by squeue")
        return False
    logger.info("[slurm_job:status] JobID|State|Elapsed|TimeLimit|StartTime|Nodes/Reason")
    for row in rows:
        logger.info(f"[slurm_job:status] {timestamp}|{row}")
    return all(len(parts := row.split("|", 2)) > 1 and parts[1] == "PENDING" for row in rows)

def _wait_for_submission(
    submit: subprocess.Popen,
    job_id: str,
    period_seconds: int = _SLURM_STATUS_PERIOD_SECONDS,
) -> int:
    """Wait for sbatch, reporting scheduler state only while the job is pending."""
    while True:
        try:
            return submit.wait(timeout=period_seconds)
        except subprocess.TimeoutExpired:
            if not _report_queue_status(job_id):
                return submit.wait()

def _report_allocation_accounting(job_id: str) -> None:
    """Print allocation and step accounting without changing job outcome."""
    command = [
        "sacct", "-n", "-P", "--local", "-j", job_id,
        "-o", "JobIDRaw,JobName%30,State,Elapsed,Timelimit,ExitCode,DerivedExitCode,NodeList%40,MaxRSS",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        logger.warning(
            f"[slurm_job:sacct] accounting unavailable for job {job_id}: "
            f"{proc.stderr.strip() or f'exit {proc.returncode}'}",
        )
        return
    logger.info(
        "[slurm_job:sacct] JobID|JobName|State|Elapsed|TimeLimit|ExitCode|"
        "DerivedExitCode|NodeList|MaxRSS",
    )
    for row in proc.stdout.strip().splitlines():
        logger.info(f"[slurm_job:sacct] {row}")


_SACCT_ACTIVE_STATES = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}


def _terminal_state(row: list[str]) -> bool:
    return bool(row[1].strip()) and row[1].split()[0].rstrip("+") not in _SACCT_ACTIVE_STATES


def _clean(row: list[str]) -> bool:
    return row[1].split()[0].rstrip("+") == "COMPLETED" and all(code == "0:0" for code in row[2:])


def _soft_failed(row: list[str]) -> bool:
    return (
        row[1].split()[0].rstrip("+") == "FAILED"
        and all(code.endswith(":0") for code in row[2:])
        and any(code.startswith(f"{SOFT_FAIL_RC}:") for code in row[2:])
    )


def _drained(row: list[str]) -> bool:
    """A replaced/drained peer or a cancelled cold spare.

    Slurm owns the state when infrastructure removes a task, so a launcher
    successfully replaced by a hot spare can finish as NODE_FAIL or PREEMPTED
    instead of reaching the wrapper's ordinary exit-1 path.
    """
    state = row[1].split()[0].rstrip("+")
    return state in {"CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REVOKED"} or (
        state == "FAILED"
        and all(code.endswith(":0") for code in row[2:])
        and any(code.startswith("1:") for code in row[2:])
    )


def _verify_job_array(job_id: str, rows: list[list[str]]) -> int:
    """Verdict for a finished job array: task 0 carries it; peers only drain."""
    lead = next((r for r in rows if r[0] == f"{job_id}_0"), None)
    if lead is None:
        # No indexed task rows at all: every row belongs to this array but none
        # names task 0, and a task-0 failure reads exactly like a drained peer,
        # so the verdict passes only when every task is clean or cancelled.
        if all(_clean(r) or r[1].split()[0].rstrip("+") == "CANCELLED" for r in rows):
            return 0
        raise RuntimeError(
            f"Slurm job array {job_id} has no identifiable task 0 and a failed task: "
            + ", ".join(f"{r[0]} state={r[1]} exit={r[2]}" for r in rows),
        )
    if not (_clean(lead) or _soft_failed(lead)):
        raise RuntimeError(
            f"Slurm job array {job_id} task 0 was not successful: state={lead[1]} exit={lead[2]} derived_exit={lead[3]}",
        )
    if bad := [r for r in rows if r is not lead and not _clean(r) and not _drained(r)]:
        raise RuntimeError(
            f"Slurm job array {job_id} peer tasks failed: "
            + ", ".join(f"{r[0]} state={r[1]} exit={r[2]}" for r in bad),
        )
    return SOFT_FAIL_RC if _soft_failed(lead) else 0


def _sacct_rows(job_id: str, job_array: bool) -> list[list[str]]:
    """Poll sacct until the allocation — every task, for an array — reports terminal.

    Queries the display JobID column rather than JobIDRaw: array tasks always show
    the indexed 123_0 form there, while JobIDRaw is a bare per-task numeric on some
    Slurm versions, which cannot name task 0.
    """
    command = ["sacct", "-X", "-n", "-P", "--local", "-j", job_id, "-o", "JobID,State,ExitCode,DerivedExitCode"]

    rows: list[list[str]] = []
    for _attempt in range(30):
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"sacct failed for Slurm job {job_id}: {proc.stderr.strip()}")
        rows = [r for r in (line.split("|") for line in proc.stdout.splitlines() if line.strip()) if len(r) == 4]
        if job_array:
            key = next((r for r in rows if r[0] == f"{job_id}_0"), None)
            if any(r[0].startswith(f"{job_id}_") for r in rows):
                rows = [r for r in rows if r[0].startswith(f"{job_id}_")]
                if key is not None and all(_terminal_state(r) for r in rows):
                    break
            elif rows and all(_terminal_state(r) for r in rows):
                # No indexed rows at all: every row returned by -j is this array's.
                break
        elif (row := next((r for r in rows if r[0] == job_id), None)) is not None and _terminal_state(row):
            break
        time.sleep(1)
    return rows


def _verify_allocation(job_id: str, rc: int, job_array: bool = False) -> int | None:
    """Check the finished Slurm job; returns the verdict rc in job-array mode.

    A job array's verdict lives on task 0: peers exit 1 when task 0's RDZV_CLOSED
    drains the generation and cold spares are cancelled, so their rows carry no
    signal. sbatch --wait does not propagate a per-task exit code for arrays, so
    the sacct verdict replaces rc. The display JobID column carries the indexed
    123_0 task form on every Slurm version; a version whose JobIDRaw is a bare
    per-task numeric cannot name task 0, so an unindexed row set only passes when
    every task is clean or cancelled — a drained peer reads exactly like a failed
    task 0 there.
    """
    rows = _sacct_rows(job_id, job_array)
    if job_array:
        if not rows or not all(_terminal_state(r) for r in rows):
            raise RuntimeError(f"sacct did not report a finished Slurm job array {job_id}")
        return _verify_job_array(job_id, rows)

    row = next((r for r in rows if r[0] == job_id), None)
    if row is None or not _terminal_state(row):
        raise RuntimeError(f"sacct did not report a finished Slurm job {job_id}")
    _report_allocation_accounting(job_id)
    soft_rc = rc == SOFT_FAIL_RC and _soft_failed(row)
    if not ((rc == 0 and _clean(row)) or soft_rc):
        raise RuntimeError(
            f"Slurm job {job_id} finished unsuccessfully: state={row[1]} "
            f"exit={row[2]} derived_exit={row[3]} sbatch_rc={rc}",
        )
    return None

@contextlib.contextmanager
def _cancel_on_signal(job_id: str) -> Generator[None, None, None]:
    previous = {}

    def cancel(signum: int, _frame: object) -> None:
        _cancel_slurm_job(job_id)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, cancel)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

def sbatch_run(
    script: str, stream_log: Path, *, extra_sbatch_args: list[str] | None = None,
    env: dict[str, str] | None = None, background: bool = False, dry_run: bool = False,
    job_array: bool = False,
) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = RUN_ROOT / "run.sbatch" if RUN_ROOT else stream_log.with_suffix(".sbatch")
    stream_log.parent.mkdir(parents=True, exist_ok=True)
    stream_log.write_text("")
    if job_array:
        # A named run can be submitted again without --clean-run. Its primary
        # stream is truncated above; do the same for the peer streams so the
        # new epilog cannot count checkpoints or metrics from an older array.
        for stale in stream_log.parent.glob(f"{stream_log.stem}.task*.log"):
            stale.unlink()
        stream_log.with_suffix(".combined.log").unlink(missing_ok=True)
    # Array tasks need a real batch output on the run directory: anything a task
    # writes before its exec redirect — including a failing redirect — otherwise
    # lands in /dev/null.
    batch_output = RESULTS_DIR / "slurm_batch.%A_%a.log" if job_array else None
    _materialize(script, script_path, list(extra_sbatch_args or []), env, batch_output=batch_output)
    if dry_run:
        logger.info(f"[slurm_job] wrote sbatch reproducer: {script_path}")
        return 0
    cmd = ["sbatch", "--parsable", *(["--wait"] if not background else []), "--export=ALL", str(script_path)]
    logger.info(shlex.join(cmd))
    if background:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"sbatch submission failed (rc={proc.returncode}): {proc.stderr.strip()}")
        _record_submission(proc.stdout, script_path)
        return 0
    tail = subprocess.Popen(["tail", "-n", "+1", "-F", str(stream_log)])  # noqa: S607
    submit = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    try:
        if submit.stdout is None:
            raise RuntimeError("failed to capture sbatch submission output")
        job_id = _record_submission(submit.stdout.readline(), script_path)
        with _cancel_on_signal(job_id):
            rc = _wait_for_submission(submit, job_id)
    finally:
        if submit.poll() is None:
            submit.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                submit.wait(timeout=5)
        tail.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            tail.wait(timeout=5)
    verdict = _verify_allocation(job_id, rc, job_array=job_array)
    return rc if verdict is None else verdict

def _transient_nccl_proxy_failure(log: Path) -> bool:
    """Recognize the recurrent NCCL InfiniBand proxy crash, and only that crash.

    Reads the task logs as well when they exist: under a job array the failing
    rank's output lands in the log of the task that hosted it, not the one the
    launcher tails.
    """
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    for task_log in sorted(log.parent.glob(f"{log.stem}.task*.log")):
        text += task_log.read_text(errors="replace")
    return all(marker in text for marker in ("Segfault encountered", "ncclIbMultiSend", "sendProxyProgress"))

def _clean_run_for_retry() -> None:
    """Remove one failed job's outputs while retaining its shared build caches."""
    if RUN_ROOT is None or RESULTS_DIR is None or WORK_ROOT is None or RUN_NAME is None:
        raise RuntimeError("run directory is not configured")
    if RUN_ROOT.is_symlink() or RUN_ROOT.parent != WORK_ROOT or RUN_ROOT.name != RUN_NAME:
        raise RuntimeError(f"refusing to clean unsafe retry directory: {RUN_ROOT}")
    logger.warning(f"[slurm_job:retry] cleaning failed run before retry: {RUN_ROOT}")
    shutil.rmtree(RUN_ROOT)
    RESULTS_DIR.mkdir(parents=True)

def _sbatch_run_with_retry(
    script: str,
    log: Path,
    *,
    extra_sbatch_args: list[str],
    background: bool,
    dry_run: bool,
    job_array: bool = False,
) -> int:
    """Retry one clean allocation only for the known transient NCCL proxy crash."""
    try:
        return sbatch_run(
            script, log,
            extra_sbatch_args=extra_sbatch_args,
            background=background,
            dry_run=dry_run,
            job_array=job_array,
        )
    except RuntimeError:
        if background or dry_run or not _transient_nccl_proxy_failure(log):
            raise
        logger.warning(
            "[slurm_job:retry] NCCL InfiniBand proxy segfault detected; "
            "retrying once on a clean allocation",
        )
        _clean_run_for_retry()
        return sbatch_run(script, log, extra_sbatch_args=extra_sbatch_args, job_array=job_array)

def _tee_rc(
    cmd: list[str],
    log_file: Path,
    env: dict | None = None,
    *,
    append: bool = False,
    mirror_stdout: bool = True,
) -> int:
    if not mirror_stdout:
        with log_file.open("a" if append else "w") as output:
            return subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, env=env, check=False).returncode
    flag = "-a " if append else ""
    shell = f"set -o pipefail; {shlex.join(cmd)} 2>&1 | tee {flag}{shlex.quote(str(log_file))}"
    return subprocess.run(shell, shell=True, executable="/bin/bash", env=env, check=False).returncode  # noqa: S602
def tee_run(cmd: list[str], log_file: Path, env: dict | None = None, *, append: bool = False) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rc = _tee_rc(cmd, log_file, env, append=append)
    if rc not in (0, SOFT_FAIL_RC):
        raise RuntimeError(f"job failed with exit code {rc}; log: {log_file}")
    return rc

def _process_rss_bytes(pid: str) -> int | None:
    """Read the current RSS for one live Linux process."""
    try:
        status = (_PROC_ROOT / pid / "status").read_text()
    except (OSError, UnicodeError):
        return None
    match = _PROC_STATUS_RSS_RE.search(status)
    return int(match[1]) * 1024 if match else None


def _with_host_memory_metrics(metrics: dict, results_dir: Path, *, native: bool) -> dict:
    """Add complete-run peak RSS for the training ranks and daemon nodes."""
    result = dict(metrics)
    if training := result.pop("cpu_rss_bytes", None):
        result.setdefault("training_process_peak_rss_bytes", training)
    if native and "checkpoint_daemon_peak_rss_bytes" not in result:
        result.update(otel_report.complete_run_process_memory_peaks(results_dir))
    return result

class PerfReport:
    @staticmethod
    def _platform_bandwidth_baseline(platform: str) -> int:
        values = _load_yaml(BANDWIDTH_BASELINE_FILE)
        if platform not in values:
            known = ", ".join(sorted(values))
            hint = (
                " PLATFORM is unset, so this is the hostname; set it to the hardware "
                f"the run is on ({known})."
                if not os.environ.get("PLATFORM", "").strip()
                else f" Known platforms: {known}."
            )
            raise RuntimeError(f"no bandwidth baseline for platform {platform!r}.{hint}")
        return round(float(values[platform]))

    @staticmethod
    def baseline(stem: str) -> dict:
        path = BASELINES_DIR / f"{stem}.yaml"
        baseline = dict(_load_yaml(path) if path.exists() else {})
        metrics = dict(baseline.get("metrics", {}))
        if "nvrx" not in stem:
            metrics.setdefault("bw_mbps", PerfReport._platform_bandwidth_baseline(platform_tag()))
        baseline["metrics"] = metrics
        return baseline

    @staticmethod
    def _log_daemon_durability(metrics: dict) -> None:
        """Report the coordinator's own view of durability.

        Reported apart from the per-routine spans above because the daemon measures the
        job's writes rather than one routine's, and only the native provider runs one.
        """
        metric = metrics.get("ckpt_daemon_durable_ms")
        if not metric:
            return
        samples = ", ".join(f"{value:.2f}" for value in metric["samples"])
        logger.info("checkpoint durability, coordinator daemon (min, max ms across saves)")
        logger.info(
            "    daemon activate -> durable ".ljust(52, ".")
            + f": ({metric['min']:.2f}, {metric['max']:.2f})  mean {metric['mean']:.2f}"
            + f"  over {len(metric['samples'])} save(s) [{samples}]",
        )

    @staticmethod
    def _statistic(name: str, measured: dict) -> str:
        if name in {
            *HOST_MEMORY_METRICS,
            "gpu_hbm_peak_bytes",
            "megatron.training.memory_allocated_gb",
        } and "max" in measured:
            return "max"
        if name in {
            "bw_mbps",
            "wr_tnsr_us",
        } and "median" in measured:
            return "median"
        return "mean"

    @staticmethod
    def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
        expected, tolerances = baseline.get("metrics", {}), baseline.get("tolerances", {})
        result, passed = {}, True
        for name in sorted(expected.keys() - metrics.keys()):
            logger.warning(f"[slurm_job:perf] WARNING required baseline metric {name!r} was not measured")
            result[name] = {
                "status": "FAIL",
                "measured": None,
                "baseline": expected[name],
                "reason": "missing_metric",
            }
            passed = False
        for name, measured in metrics.items():
            statistic = PerfReport._statistic(name, measured)
            value, higher = measured[statistic], measured["higher_is_better"]
            improved_beyond_tolerance = False
            if name == "bw_socket_imbalance":
                ok, target, tolerance = value <= MAX_BW_SOCKET_IMBALANCE, MAX_BW_SOCKET_IMBALANCE, 0
            elif name not in expected:
                result[name] = {"status": "info" if higher is None else "no_baseline", "measured": value}
                continue
            else:
                target = float(expected[name])
                fallback = WIDE_TOLERANCE_METRICS.get(name, 0 if higher is None else default_tolerance)
                tolerance = tolerances.get(name, fallback)
                low, high = target * (1 - tolerance), target * (1 + tolerance)
                if higher is True:
                    ok = value >= low
                    improved_beyond_tolerance = value > high
                elif higher is False:
                    ok = value <= high
                    improved_beyond_tolerance = value < low
                else:
                    ok = low <= value <= high
            passed &= ok
            result[name] = {
                "status": "WARN" if improved_beyond_tolerance else ("PASS" if ok else "FAIL"),
                "measured": value,
                "baseline": target,
                "tolerance_pct": tolerance * 100,
                "unit": measured["unit"],
                "statistic": statistic,
            }
        return result, passed

    @staticmethod
    def report(
        job: str,
        config: str,
        flavor: str,
        log: Path,
        baseline: dict,
        report_path: Path,
        gpus: int,
        *,
        results_dir: Path | None = None,
        otel_reporting: bool | None = None,
        enforced_metrics: frozenset[str] | None = None,
    ) -> bool:
        flavor_names = flavor.split("-")
        results_dir = results_dir or log.parent
        native = "nvrx" not in flavor_names and (
            "native" in flavor_names or any(results_dir.glob("checkpoint_server.cycle*.log"))
        )
        if otel_reporting is None:
            otel_reporting = _otel_reporting_enabled()
        processed = (
            otel_report.post_process(results_dir, gpus, native)
            if otel_reporting
            else log_report.post_process(log, results_dir, gpus, native)
        )
        metrics = _with_host_memory_metrics(processed["metrics"], results_dir, native=native)
        PerfReport._log_daemon_durability(metrics)
        if not metrics:
            if processed["source"] == "logs" and not native:
                return True
            raise RuntimeError(f"no perf metrics found in {processed['source']} report data for {results_dir}")
        comparison, passed = PerfReport._compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
        if enforced_metrics is not None:
            passed = all(comparison.get(name, {}).get("status") != "FAIL" for name in enforced_metrics)
        validation = processed["validation"]
        if validation:
            comparison[validation["name"]] = {
                "status": "PASS" if validation["valid"] else "FAIL",
                "measured": validation["measured"],
                "baseline": validation["baseline"],
            }
            if enforced_metrics is None:
                passed &= validation["valid"]
        for name, value in comparison.items():
            metric = metrics.get(name, {})
            unit = f" {metric['unit']}" if metric.get("unit") else ""
            if all(statistic in metric for statistic in ("min", "median", "max")):
                statistic = value.get("statistic") or PerfReport._statistic(name, metric)
                measured = (
                    f"{statistic}={value['measured']}{unit} "
                    f"(min={metric['min']}{unit}, median={metric['median']}{unit}, max={metric['max']}{unit})"
                )
            else:
                measured = f"measured={value['measured']}{unit}"
            details = [measured]
            if "baseline" in value:
                details.append(f"baseline={value['baseline']}{unit}")
            if "tolerance_pct" in value:
                details.append(f"tolerance={value['tolerance_pct']}%")
            report_metric = logger.warning if value["status"] == "WARN" else logger.info
            report_metric(f"[{value['status']}] {name}: {'; '.join(details)}")
        report = {
            "job": job,
            "config": config,
            "flavor_tag": flavor,
            "cluster": cluster_tag(),
            "telemetry_source": processed["source"],
            "passed": passed,
            "metrics": {name: {k: v for k, v in metric.items() if k not in ("samples", "ranks")} for name, metric in metrics.items()},
            "comparison": comparison,
        }
        if validation:
            report[validation["report_field"]] = validation["valid"]
        report_path.write_text(json.dumps(report, indent=2))
        return passed

def _argument_values(text: str) -> dict[str, str]:
    """The options an arguments file records, by name.

    The block a run prints is delimited by rules of dashes, which carry no
    option, so a comparison reads the named settings and not the decoration.
    """
    values = {}
    for line in text.splitlines():
        name, _, value = line.strip().partition(" ")
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            values[name] = value.strip().strip(".").strip()
    return values

def _arguments_stem(args: argparse.Namespace, variants: tuple[str, ...] | None = None) -> str:
    """The baseline name for one run's resolved Megatron options."""
    stem = _identity(args)[2]
    if variants is None:
        variants = next((spec.variants for spec in reversed(_run_specs(args)) if spec.variants), ())
    return f"{stem}-{'-'.join(variants)}" if variants else stem

def _compare_arguments(args: argparse.Namespace, log: Path, variants: tuple[str, ...] = ()) -> bool:
    """Report how the options this run trained on differ from the recorded ones.

    An option that changes changes what the run's checkpoints mean, so a run
    that no longer trains on what was recorded reports every difference and
    marks the job. The candidate beside this run's logs is what a reviewer
    adopts to accept the change.
    """
    stem = _arguments_stem(args, variants)
    candidate = log.parent / "baselines" / f"{stem}_arguments.txt"
    recorded = BASELINES_DIR / f"{stem}_arguments.txt"
    if not candidate.exists():
        logger.warning(f"[slurm_job:args] WARNING {stem}: this run recorded no arguments")
        return False
    if not recorded.exists():
        logger.warning(
            f"[slurm_job:args] WARNING {stem}: no recorded arguments to compare against; "
            f"adopt {candidate.name} from this job's artifacts into {BASELINES_DIR.name}/",
        )
        return False
    produced, expected = (_argument_values(path.read_text(errors="replace")) for path in (candidate, recorded))
    # Two files that yield no options agree on nothing, not on everything. A
    # run whose arguments could not be read has compared nothing and says so.
    for name, values, path in (("recorded", expected, recorded), ("produced", produced, candidate)):
        if not values:
            logger.warning(f"[slurm_job:args] WARNING {stem}: no options read from the {name} arguments in {path}")
            return False
    differences = [
        f"{name}: {expected.get(name, '<unset>')} -> {produced.get(name, '<unset>')}"
        for name in sorted(expected.keys() | produced.keys())
        if expected.get(name) != produced.get(name)
    ]
    for difference in differences:
        logger.warning(f"[slurm_job:args] WARNING {stem} {difference}")
    report = logger.warning if differences else logger.info
    report(f"[slurm_job:args] {stem}: {len(differences)} of {len(expected)} options differ from the recorded run")
    return not differences

# The line may carry a timestamp or stream prefix from the log reader, so
# match the srun diagnostic anywhere within it. Ranks reported as Terminated
# are consequences of another task failing and deliberately do not match.
_SRUN_EXITED_RE = re.compile(r"srun: error: \S+: tasks? ([\d,-]+): Exited with exit code \d+")

def _failing_ranks(text: str) -> list[int]:
    """Return the ranks that srun reports exiting before it terminates the step."""
    ranks: list[int] = []
    for spec in _SRUN_EXITED_RE.findall(text):
        for part in spec.split(","):
            first, separator, last = part.partition("-")
            ranks.extend(range(int(first), int(last if separator else first) + 1))
    return sorted(dict.fromkeys(ranks))

def _rank_logs(run_log: Path, rank: int) -> list[Path]:
    """Find a rank's logs in both single-run and nested multi-run layouts."""
    run_root = run_log.parent.parent
    return sorted(path for path in run_root.glob(f"**/logs/*_rank{rank}.log") if path.is_file())

def _report_failing_rank_logs(
    run_log: Path,
    *,
    rank_limit: int = _RANK_LOG_LIMIT,
    tail_lines: int = _RANK_LOG_TAIL_LINES,
) -> None:
    """Mirror failing ranks' log tails into the Slurm and GitLab job trace."""
    ranks = _failing_ranks(run_log.read_text(errors="replace")) if run_log.exists() else []
    if not ranks:
        return
    logger.info(
        f"[slurm_job:rank] srun reports rank(s) {', '.join(map(str, ranks))} exited; "
        "the remaining ranks were terminated",
    )
    for rank in ranks[:rank_limit]:
        paths = _rank_logs(run_log, rank)
        if not paths:
            logger.warning(f"[slurm_job:rank] no rank log found for exited rank {rank}")
            continue
        for path in paths:
            tail = path.read_text(errors="replace").splitlines()[-tail_lines:]
            display = path.relative_to(run_log.parent.parent)
            logger.info(f"[slurm_job:rank] ---- last {len(tail)} lines of {display} ----")
            for line in tail:
                logger.info(f"[slurm_job:rank{rank}] {line}")
    if len(ranks) > rank_limit:
        logger.info(
            f"[slurm_job:rank] {len(ranks) - rank_limit} additional exited rank(s) "
            "remain available in the job artifacts",
        )


_PRIMARY_FAILURE_RE = re.compile(
    r"(?:RuntimeError: (?:checkpoints differ|reference check failed|manifest derivation failed)|"
    r"Unexpected result nan|found NaN|CUDA out of memory|OutOfMemoryError|"
    r"Fatal Python error|Segmentation fault|AssertionError:|ValueError:)",
    re.IGNORECASE,
)
_FAILURE_DETAIL_RE = re.compile(
    r"(?:\[slurm_job:diff\].*(?:only in|missing=[1-9]\d*|extra=[1-9]\d*|changed=[1-9]\d*|"
    r":\s+(?:changed|missing|extra)\s+\S)|ncclIbMultiSend|sendProxyProgress)",
    re.IGNORECASE,
)
_FALLBACK_FAILURE_RE = re.compile(r"(?:RuntimeError:|srun: error:.*Exited with exit code|FAILED.*exit)", re.IGNORECASE)


def _log_tail(path: Path, limit: int = _FAILURE_SUMMARY_TAIL_BYTES) -> str:
    """Read a bounded log tail without loading a multi-rank training log in full."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def _failure_summary_lines(run_log: Path) -> list[str]:
    """Select causal errors and their useful details from an allocation's log tail."""
    paths = [run_log, *sorted(run_log.parent.glob(f"{run_log.stem}.task*.log"))]
    categories: tuple[list[str], list[str], list[str]] = ([], [], [])
    seen: set[str] = set()
    for path in paths:
        for raw in _log_tail(path).splitlines():
            line = re.sub(r"^\s*\d+:\s*", "", raw).strip()
            if not line or line in seen:
                continue
            for regex, bucket in zip(
                (_PRIMARY_FAILURE_RE, _FAILURE_DETAIL_RE, _FALLBACK_FAILURE_RE),
                categories,
            ):
                if regex.search(line):
                    seen.add(line)
                    bucket.append(line)
                    break
    selected = [*categories[0], *categories[1]] or categories[2]
    return selected[:_FAILURE_SUMMARY_LIMIT]


def _report_failure_summary(run_log: Path) -> None:
    """Print the underlying allocation failure after verbose rank and cleanup output."""
    lines = _failure_summary_lines(run_log)
    if not lines:
        return
    logger.error("[slurm_job:failure] ===== underlying failure summary =====")
    for line in lines:
        logger.error(f"[slurm_job:failure] {line}")
    logger.error(f"[slurm_job:failure] full logs: {run_log.parent}")

def _write_arguments_artifact(args: argparse.Namespace, log: Path, variants: tuple[str, ...] = ()) -> Path:
    """Write the stable argument block from one run beside that run's logs."""
    output = log.parent / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    text = re.sub(r"(?m)^\s*\d+:\s?", "", log.read_text(errors="replace"))
    blocks = _ARGUMENTS_RE.findall(text)
    if not blocks:
        raise RuntimeError(f"no Megatron arguments block found in {log}")
    lines = [
        line for line in blocks[-1].splitlines()
        if line.startswith("-") or re.match(r"^  [A-Za-z_]\w*(?: \.+ | {2})", line)
    ]
    volatile = re.compile(r"^  (?:ckpt_pg_tensors_cache_path|data_cache_path|load|save|tensorboard_dir) ")
    arguments = output / f"{_arguments_stem(args, variants)}_arguments.txt"
    arguments.write_text("\n".join(line.rstrip() for line in lines if not volatile.match(line)) + "\n")
    return arguments

def _argument_log(_args: argparse.Namespace, batch_log: Path, spec: RunSpec) -> Path:
    """Return the rank-0 log containing a particular run's argument blocks."""
    cfg = f"{spec.args.config}_{spec.args.nodes}x{spec.args.gpus_per_node}"
    run_root = batch_log.parent.parent
    run = run_root / _run_dir_name(spec) if spec.name else run_root
    return run / "logs" / f"{cfg}_{spec.name or 'run'}-routine_rank0.log"

def _write_and_compare_arguments(args: argparse.Namespace, batch_log: Path, *, strict: bool = True) -> bool:
    """Generate, compare and retain arguments for the runs under test."""
    results = []
    all_specs = _run_specs(args)
    native_specs = tuple(spec for spec in all_specs if _provider(spec.flavors) == "native")
    specs = native_specs if args.flavors.policy.compares_routines and native_specs else all_specs
    for spec in specs:
        log = _argument_log(args, batch_log, spec)
        # A unit/local epilog may only have the combined log. Real multi-run
        # jobs always have one rank-0 log per run and must not fall back.
        if not log.exists() and len(all_specs) == 1:
            log = batch_log
        if not log.exists():
            message = f"argument log for run {spec.name!r} does not exist: {log}"
            if strict:
                raise RuntimeError(message)
            logger.warning(f"[slurm_job:args] WARNING {message}")
            continue
        try:
            # Run-scoped variants already participate in the artifact identity.
            _write_arguments_artifact(spec.args, log, ())
        except RuntimeError:
            if strict:
                raise
            logger.warning(f"[slurm_job:args] WARNING no complete arguments in {log}")
            continue
        results.append(_compare_arguments(spec.args, log, ()))
    return len(results) == len(specs) and all(results)

def _log_otel_checkpoint_timings(args: argparse.Namespace, batch_log: Path) -> None:
    specs = _run_specs(args)
    for spec in specs:
        results_dir = _argument_log(args, batch_log, spec).parent
        otel_files = sorted(
            results_dir.glob("otel/*/native_ckpt_otel.jsonl"),
            key=lambda path: (
                path.parent.name != "phase1",
                int(path.parent.name.removeprefix("cycle"))
                if path.parent.name.startswith("cycle") and path.parent.name.removeprefix("cycle").isdigit()
                else 0,
            ),
        )
        if not otel_files:
            raise RuntimeError(f"no checkpoint OTLP files found in {results_dir / 'otel'}")
        for path in otel_files:
            label = f"{spec.name} {path.parent.name}" if len(specs) > 1 else path.parent.name
            otel_report.log_checkpoint_timings(path, None, label)

def _write_performance_baseline(args: argparse.Namespace, results_dir: Path) -> None:
    """Write measured metrics from the job-level report as a candidate baseline."""
    flavors = _selected_flavors(args)
    if not ({"perf", "detr"} & set(flavors.names)):
        return
    _cfg, _flavor, report_stem = _identity(args)
    report_spec = next(
        (spec for spec in reversed(_run_specs(args)) if _provider(spec.flavors) == "native"),
        _run_specs(args)[-1],
    )
    nvrx = "nvrx" in report_spec.flavors.names
    stem = _performance_baseline_stem(args)
    output = results_dir / "baselines"
    output.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / f"{report_stem}_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    measured = report.get("metrics", {})
    source = report.get("telemetry_source")
    if source not in {"logs", "otlp"}:
        logger.warning(f"[slurm_job:perf] WARNING refusing baseline candidate with unknown telemetry source: {source!r}")
        return
    required = (
        GPU_MEMORY_METRICS | {"training_process_peak_rss_bytes"}
        if nvrx
        else otel_report.REQUIRED_PERF_METRICS if source == "otlp" else log_report.REQUIRED_PERF_METRICS
    )
    validation_field = "ckpt_save_spans_valid" if source == "otlp" else "ckpt_save_markers_valid"
    missing = required - measured.keys()
    invalid_save_records = not nvrx and report.get(validation_field) is not True
    if invalid_save_records or missing:
        details = []
        if invalid_save_records:
            details.append("ckpt.save records were missing or invalid")
        if missing:
            details.append(f"required metrics were absent: {', '.join(sorted(missing))}")
        logger.warning(f"[slurm_job:perf] WARNING refusing to write an incomplete baseline candidate: {'; '.join(details)}")
        return
    metrics = {}
    rank_statistics = {}
    for name, value in measured.items():
        # Socket imbalance is an in-job topology diagnostic rather than a
        # workload baseline. The platform bandwidth value is only a fallback
        # when a workload has no recorded measurement.
        if name == "bw_socket_imbalance":
            continue
        statistic = PerfReport._statistic(name, value)
        if statistic in value:
            measured_value = value[statistic]
            preserve_decimals = name.endswith("_pct") or name.startswith("megatron.training.")
            metrics[name] = round(measured_value, 2) if preserve_decimals else round(measured_value)
        if {"rank_min", "rank_median", "rank_max"} <= value.keys():
            rank_statistics[name] = {
                "min": round(value["rank_min"], 2),
                "median": round(value["rank_median"], 2),
                "max": round(value["rank_max"], 2),
            }
    if metrics:
        baseline = {"description": "auto-generated by CI", "source": source, "metrics": metrics}
        memory_tolerances = {name: HOST_MEMORY_TOLERANCE for name in HOST_MEMORY_METRICS if name in metrics}
        if memory_tolerances:
            baseline["tolerances"] = memory_tolerances
        if rank_statistics:
            baseline["rank_statistics"] = rank_statistics
        (output / f"{stem}.yaml").write_text(yaml.safe_dump(baseline, sort_keys=False))

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result

def _flatten(config: dict) -> list[str]:
    output: list[str] = []
    for key, value in config.items():
        if value is True:
            output.append(f"--{key}")
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            output.extend([f"--{key}", *(str(v) for v in value)])
        else:
            output.extend([f"--{key}", str(value)])
    return output

def _drop(opts: list[str], *flags: str) -> None:
    """Remove each flag and the value it carries, however that value is spelled.

    An option states its value either as the token after the flag or joined to
    it by '=', so a flag is matched both ways.
    """
    for flag in flags:
        start = next((i for i, opt in enumerate(opts) if opt == flag or opt.startswith(f"{flag}=")), None)
        if start is None:
            continue
        end = start + 1
        if "=" not in opts[start]:
            while end < len(opts) and not opts[end].startswith("--"):
                end += 1
        del opts[start:end]

def _set(opts: list[str], flag: str, value: str | None = None) -> None:
    _drop(opts, flag)
    opts.extend([flag, value] if value is not None else [flag])

def _base_env() -> dict[str, str]:
    rank = _rank_env("SLURM_PROCID", "RANK", "0")
    caches = {
        name: str(Path(os.environ.get(name, default)) / f"rank-{rank}")
        for name, default in (
            ("TRITON_CACHE_DIR", "/tmp/triton-cache"),  # noqa: S108
            ("TRITON_CACHE_PATH", "/tmp/triton-cache"),  # noqa: S108
            ("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor-cache"),  # noqa: S108
            ("TORCH_EXTENSIONS_DIR", "/tmp/torch-extensions"),  # noqa: S108
        )
    }
    for cache in caches.values():
        Path(cache).mkdir(parents=True, exist_ok=True)
    local_rank = _rank_env("SLURM_LOCALID", "LOCAL_RANK", "0")
    world = _rank_env("SLURM_NTASKS", "WORLD_SIZE", "1")
    nodes = os.environ.get("SLURM_STEP_NODELIST") or os.environ["SLURM_JOB_NODELIST"]
    if "MASTER_ADDR" not in os.environ:
        hosts = subprocess.check_output(["scontrol", "show", "hostnames", nodes], text=True)  # noqa: S607
        os.environ["MASTER_ADDR"] = hosts.splitlines()[0]
    return {
        "RANK": rank,
        "LOCAL_RANK": local_rank,
        "WORLD_SIZE": world,
        "SLURM_JOB_NODELIST": nodes,
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
        **{name: os.environ.get(name, default) for name, default in _TRAIN_ENV_DEFAULTS.items()},
        **caches,
        "NATIVE_CKPT_SERVER_LOG": os.environ.get(
            "NATIVE_CKPT_SERVER_LOG",
            str(RESULTS_DIR / "checkpoint_server.%h.%p.log"),
        ),
    }

# A run trains phase 1 from scratch and phase 2 resuming from it. A variant
# applies to both unless it names the phase it belongs to.
_SCOPES = ("all", "phase1", "phase2")
OptionEdit = namedtuple("OptionEdit", "scope add drop")  # noqa: PYI024

def _scope_matches(scope: str, phase: int) -> bool:
    return scope in ("all", f"phase{phase}")

def _scopes_overlap(left: str, right: str) -> bool:
    return "all" in (left, right) or left == right

def _flag(option: str) -> str:
    """The option a token names, whether or not it carries a joined value."""
    return option.partition("=")[0]

@dataclass
class FlavorPolicy:
    phase1_train_iters: int = _PHASE1_TRAIN_ITERS
    phase1_save_interval: int | None = None
    phase1_checkpoints: int = 1
    phase2_save_interval: int = 250
    phase2_checkpoints: int = 3
    checkpoint_routines: int = 1
    manages_native_server: bool = True
    otel_reporting: bool = True
    requires_nvrx_absent: bool = False
    compares_routines: bool = False
    # The variant a single-routine run trains as, so its options are the
    # ones a comparison would have applied to the routine under test.
    routine_variant: str = "reference"
    # Contributed by the mixin flavors; an explicit run option still wins.
    option_edits: tuple[OptionEdit, ...] = ()
    wheel_prep_mode: str = "--ensure-prebuilt"
    wheel_defines: tuple[str, ...] = ()
    install_prebuilt: bool = True
    requires_finite_run: bool = False
    worker: Callable[[str, argparse.Namespace, str], None] | None = None
    server: Callable[[str, int], contextlib.AbstractContextManager] | None = None
    reporter: Callable[[argparse.Namespace, Path], bool] | None = None

    def train_config(
        self,
        cfg: str,
        args: argparse.Namespace,
        provider: str,
        phase: int,
        cycle: int = 0,
        *,
        resume_iteration: int | None = None,
        checkpoints_override: int | None = None,
    ) -> TrainConfig:
        run, phase1 = _run_dir(cfg), _run_dir(cfg) / "phase1-checkpoints"
        extend_training = args.checkpoints_per_run is not None or args.repeat != 1
        if phase == 1:
            save_interval = self.phase1_save_interval or self.phase1_train_iters
            return TrainConfig(
                args.nodes, args.gpus_per_node, phase1, None, args.config, provider,
                1, save_interval, self.phase1_checkpoints, extend_training,
            )
        checkpoints = self.phase2_checkpoints if args.checkpoints_per_run is None else args.checkpoints_per_run
        if checkpoints_override is not None:
            checkpoints = checkpoints_override
        interval = self.phase2_save_interval if args.checkpoint_interval is None else args.checkpoint_interval
        start_iteration = self.phase1_train_iters
        if cycle:
            start_iteration = (self.phase1_train_iters // interval + cycle * checkpoints) * interval
        load = phase1 if cycle == 0 else run / "checkpoints"
        if resume_iteration is not None:
            start_iteration = resume_iteration
            load = run / "checkpoints"
        return TrainConfig(
            args.nodes,
            args.gpus_per_node,
            run / "checkpoints",
            load,
            args.config,
            provider,
            2,
            interval,
            checkpoints,
            extend_training,
            start_iteration,
        )

    def server_context(self, stem: str, phase: int) -> contextlib.AbstractContextManager:
        return self.server(stem, phase) if self.server else contextlib.nullcontext()

    def run_worker(self, cfg: str, args: argparse.Namespace, provider: str, label: str = "") -> None:
        (self.worker or _run_routine)(cfg, args, provider, label)

    def stop_server(self, _provider: str, tag: str, graceful: bool = True) -> None:
        if self.manages_native_server:
            def stop() -> None:
                pids = CHECKPOINT_SERVER.pids()
                CHECKPOINT_SERVER.stop(graceful)
                if graceful:
                    pattern = os.environ["NATIVE_CKPT_SERVER_LOG"]
                    _reject_server_errors(
                        Path(pattern.replace("%h", socket.gethostname()).replace("%p", pid)) for pid in pids
                    )

            _node_once(tag, stop)

    def apply_options(self, opts: list[str], cfg: str, phase: int) -> list[str]:
        """Add and drop the options the selected variants ask of this phase.

        Returns each change as ``add <option>`` or ``drop <option>`` so a run
        evidences the variants it carried.
        """
        applied = []
        for edit in self.option_edits:
            if not _scope_matches(edit.scope, phase):
                continue
            _drop(opts, *edit.drop)
            added = _extra_options(cfg, [shlex.join(edit.add)]) if edit.add else []
            opts += added
            applied += [f"drop {flag}" for flag in edit.drop] + [f"add {token}" for token in added]
        return applied

    def prep_args(self) -> list[str]:
        return [self.wheel_prep_mode, *(arg for define in self.wheel_defines for arg in ("--define", define))]

    def epilog(self, args: argparse.Namespace, log: Path) -> bool:
        return self.reporter(args, log) if self.reporter else True


class Flavor:
    name = ""
    summary = ""
    priority = 0
    # An execution mode owns the routine a run performs and what it reports.
    # A variant only edits the options that routine trains with.
    execution_mode = False
    edits: tuple[OptionEdit, ...] = ()

    def configure(self, _policy: FlavorPolicy) -> None:
        pass

    def transform(self, _opts: list[str], _env: dict[str, str]) -> None:
        pass


class PerfFlavor(Flavor):
    execution_mode = True
    name = "perf"
    summary = "Measure one routine against a recorded throughput baseline."
    edits = (
        OptionEdit(
            "all",
            (
                "--ckpt-fully-parallel-save-process-group=ep_dp",
                "--ckpt-fully-parallel-load-process-group=ep_dp",
                "--ckpt-pg-tensors-cache-path=@PG_CACHE_DIR@",
                "--ckpt-fully-parallel-load-per-rank-objects",
                "--ckpt-drop-redundant-extra-state",
                "--no-ckpt-load-validate-sharding-integrity",
            ),
            (
                "--ckpt-fully-parallel-save-process-group",
                "--ckpt-fully-parallel-load-process-group",
            ),
        ),
        OptionEdit("phase1", ("--ckpt-pg-tensors-cache-create",), ()),
    )

    def configure(self, policy: FlavorPolicy) -> None:
        policy.requires_nvrx_absent = True
        policy.reporter = self.report
        policy.option_edits += self.edits

    def report(self, args: argparse.Namespace, log: Path) -> bool:
        cfg, tag, stem = _identity(args)
        baseline = args.baseline if hasattr(args, "baseline") else PerfReport.baseline(stem)
        return PerfReport.report(
            _selected_flavors(args).tag,
            cfg,
            tag,
            log,
            baseline,
            log.with_name(f"{stem}_report.json"),
            args.gpus_per_node,
        )

    def transform(self, opts: list[str], _env: dict[str, str]) -> None:
        # A measured run reports the steady-state save path, which resolves the
        # checkpoint structure on the first save and reuses it after that.
        _set(opts, "--ckpt-assume-constant-structure")
        if "--cuda-graph-impl" in opts and "--recompute-granularity" in opts:
            # Attention-only graph capture and selective MoE recomputation have
            # completed the 16-node Super3 BF16 workload together. Retain the
            # profile's memory policy: dropping it leaves less free HBM than
            # fused cross entropy's single 3.91 GiB allocation requires.
            if "--async-save" in opts:
                _set(opts, "--async-ckpt-use-cpu-shm")


class DeterministicFlavor(Flavor):
    execution_mode = True
    name = "detr"
    summary = "Compare a reference routine's checkpoints against a second routine's, byte for byte."

    def configure(self, policy: FlavorPolicy) -> None:
        policy.reporter = self.report
        policy.requires_finite_run = True
        policy.phase1_train_iters = 50
        policy.phase1_save_interval = 25
        policy.phase1_checkpoints = 2
        # Determinism compares checkpoint contents, so retain three phase-2
        # checkpoints while shortening the training between them. A job that
        # names two runs trains both, and the former 250-iteration cadence
        # exceeded Ultra3's six-hour allocation.
        policy.phase2_save_interval = 25
        policy.compares_routines = True

    def report(self, args: argparse.Namespace, log: Path) -> bool:
        cfg, tag, stem = _identity(args)
        specs = _run_specs(args)
        report_spec = next(
            (spec for spec in reversed(specs) if _provider(spec.flavors) == "native"),
            specs[-1],
        )
        # Multi-run determinism keeps each routine's daemon logs beneath that
        # routine, while the batch log and report live at the job root. Report
        # the native routine when one exists; otherwise report the final run.
        baseline = args.baseline if hasattr(args, "baseline") else PerfReport.baseline(_performance_baseline_stem(args))
        source_log = _argument_log(args, log, report_spec)
        report_tag = "-".join(dict.fromkeys((*_selected_flavors(args).names, *report_spec.flavors.names)))
        return PerfReport.report(
            tag,
            cfg,
            report_tag,
            source_log,
            baseline,
            log.with_name(f"{stem}_report.json"),
            args.gpus_per_node,
            results_dir=source_log.parent,
            enforced_metrics=MEMORY_REGRESSION_METRICS,
        )

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        # Deterministic backward graph construction fails with the pinned Megatron and Transformer Engine versions.
        _set(opts, "--attention-backend", "unfused")
        _drop(
            opts,
            "--cross-entropy-loss-fusion",
            "--cross-entropy-fusion-impl",
            # The fused weighted-squared-ReLU backward is run-varying on GB200.
            "--use-fused-weighted-squared-relu",
            "--cuda-graph-impl",
            "--cuda-graph-modules",
            "--moe-router-fusion",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        )
        # Recomputation reproduces the forward it replaces rather than altering
        # it, so a deterministic run keeps whatever its configuration asks for
        # and spends the memory on unfused attention and an FP64 router.
        _set(opts, "--moe-router-dtype", "fp64")
        _set(opts, "--deterministic-mode")
        _drop(opts, "--async-save", "--use-persistent-ckpt-worker", "--async-ckpt-use-cpu-shm")
        env.update(
            PYTHONHASHSEED="0",
            # The OTel SDK uses Python's process-global random module for IDs,
            # exemplar sampling, and HTTP retry jitter. Deterministic training
            # saves that state, so isolate OTel's draws.
            NVRX_OTEL_USE_SYSTEM_RANDOM="1",
            NCCL_ALGO="Ring",
            NCCL_PROTO="Simple",
            NCCL_NVLS_ENABLE="0",
            NCCL_TUNER_PLUGIN="none",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
            MAMBA_DETERMINISTIC="1",
            FLA_TILELANG="0",
        )


_VALGRIND_NATIVE_ENV = {
    "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH": "1",
    "NATIVE_CKPT_SERVER_AUTO_SHUTDOWN_TIMEOUT_SECONDS": "0",
    "NATIVE_CKPT_SERVER_HANDLER_THREADS": "4",
    "NATIVE_CKPT_SERVER_MAX_ACTIVE_SNAPSHOTS": "0",
    "NATIVE_CKPT_SERVER_TCP_ACCEPTORS": "1",
    "NATIVE_CKPT_SERVER_TCP_REUSE_PORT": "0",
    "NATIVE_CKPT_SHM_RING_BUFFER_NUM_QUEUES": "2",
    "NATIVE_CKPT_SHM_RING_BUFFER_SIZE": "128MiB",
    "NATIVE_CKPT_SHM_RING_BUFFER_SLOTS": "4",
    "NATIVE_CKPT_SHM_RING_BUFFER_WRITE_TIMEOUT_SECONDS": "300",
}


class ValgrindFlavor(Flavor):
    execution_mode = True
    name = "vg"
    summary = "Run the checkpoint server under memcheck."
    priority = 100

    def configure(self, policy: FlavorPolicy) -> None:
        policy.checkpoint_routines = 1
        policy.phase2_save_interval = 1
        policy.manages_native_server = False
        policy.otel_reporting = False
        policy.server = self.server
        policy.reporter = None

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        _drop(
            opts,
            "--cuda-graph-impl",
            "--cuda-graph-modules",
            "--recompute-granularity",
            "--recompute-modules",
        )
        # Memcheck can perturb model numerics enough to request Megatron's
        # cross-GPU rerun (exit 16), which is not useful for this gate. Keep the
        # independent training NaN abort enabled: corruption may surface there
        # even when memcheck does not diagnose it.
        _set(opts, "--rerun-mode", "disabled")
        # Keep third-party OTel/protobuf internals outside the daemon's Memcheck gate.
        env.update(_VALGRIND_NATIVE_ENV)
        env["MEGATRON_OTEL_ENABLED"] = "0"
        env.pop("NATIVE_CKPT_OTEL_FILE", None)

    @contextlib.contextmanager
    def server(self, stem: str, phase: int) -> Generator[None, None, None]:
        host = socket.gethostname().split(".")[0]
        marker = _sync_root() / f"{_sync_job_id()}-{stem}-{host}"
        if _rank_env("SLURM_LOCALID", "LOCAL_RANK", "0") != "0":
            wait_for_file(marker.with_suffix(".ready"), 600)
            yield
            wait_for_file(marker.with_suffix(".done"), 3600)
            return
        import nvrx_checkpointing  # noqa: PLC0415

        valgrind = shutil.which("valgrind")
        if not valgrind:
            raise RuntimeError("valgrind not found")
        CHECKPOINT_SERVER.require_stopped()
        report = RESULTS_DIR / f"{stem}_valgrind_phase{phase}.{host}.log"
        server_log = RESULTS_DIR / f"{stem}_server_phase{phase}.{host}.log"
        os.environ.update(CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH="1", NATIVE_CKPT_SERVER_LOG=str(server_log))
        cmd = [
            valgrind,
            "--tool=memcheck",
            "--leak-check=full",
            "--show-leak-kinds=all",
            "--errors-for-leak-kinds=definite,indirect",
            "--error-exitcode=42",
            f"--log-file={report}",
            str(nvrx_checkpointing.SERVER_PATH),
        ]
        server_env = {**os.environ, **_VALGRIND_NATIVE_ENV}
        server_env.pop("NATIVE_CKPT_OTEL_FILE", None)
        with server_log.open("w") as output:
            # Listener readiness precedes lazy queue creation, and stdout must not share stderr's log offset.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=output,
                start_new_session=True,
                env=server_env,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline and proc.poll() is None:
                if CHECKPOINT_SERVER.reachable():
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"checkpoint_server failed to start under valgrind; see {server_log}")
            marker.with_suffix(".ready").touch()
            try:
                yield
                CHECKPOINT_SERVER.request("spindown", 3600)
                rc = proc.wait(timeout=3600)
                if rc:
                    raise RuntimeError(f"valgrind exited {rc}; see {report}")
                _reject_server_errors([server_log])
            finally:
                if proc.poll() is None:
                    proc.kill()
                marker.with_suffix(".done").touch()


_FLAVOR_KEYS = ("summary", "priority", *_SCOPES)


class DeclaredFlavor(Flavor):
    """A checkpoint variant declared in ``ci/flavors.yaml``.

    A variant that only adds or drops options, or settles a run option, needs no
    code: it names the axis it puts under test and the options that express it.
    """

    def __init__(self, name: str, spec: dict):
        if unknown := sorted(set(spec) - set(_FLAVOR_KEYS)):
            raise ValueError(f"flavor {name} in {FLAVORS_FILE.name} has unknown keys: {', '.join(unknown)}")
        self.name = name
        self.summary = str(spec.get("summary", ""))
        self.priority = int(spec.get("priority", 0))
        self.edits = tuple(
            OptionEdit(scope, tuple(spec[scope].get("add") or ()), tuple(spec[scope].get("drop") or ()))
            for scope in _SCOPES
            if scope in spec
        )

    def configure(self, policy: FlavorPolicy) -> None:
        policy.option_edits += self.edits


def _flavor_index(flavors: Iterable[Flavor]) -> dict[str, Flavor]:
    """Index every flavor by the one name it answers to."""
    index: dict[str, Flavor] = {}
    for flavor in flavors:
        if index.setdefault(flavor.name, flavor) is not flavor:
            raise ValueError(f"{flavor.name} is declared twice")
    return index


_EXECUTION_MODES = (PerfFlavor(), DeterministicFlavor(), ValgrindFlavor())
_FLAVOR_LIST = (
    *_EXECUTION_MODES,
    *(DeclaredFlavor(name, spec or {}) for name, spec in _load_yaml(FLAVORS_FILE).items()),
)
_FLAVORS = _flavor_index(_FLAVOR_LIST)
_FLAVOR_NAMES = tuple(flavor.name for flavor in _FLAVOR_LIST)
_DEFAULT_FLAVOR = _FLAVORS[PerfFlavor.name]
# The checkpoint server exists only for the native provider, so the run loop
# and the daemon lifecycle ask which one a run named. Everything else about a
# provider is the `--ckpt-provider` option its flavor carries.
_DEFAULT_PROVIDER = "nvrx"

def _provider(flavors: FlavorSelection) -> str:
    return next((name for name in flavors.names if name in ("native", "nvrx")), _DEFAULT_PROVIDER)


_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

def _run_entry(value: str) -> list[str]:
    """Validate one --run argument: a name, optionally with '+'-joined variants."""
    entries = []
    for item in value.split(","):
        name, _, variants = item.strip().partition(":")
        if not _RUN_NAME_RE.fullmatch(name):
            raise argparse.ArgumentTypeError(f"run name {name!r} must be letters, digits, '_' or '-'")
        _flavor_names(variants)
        entries.append(item.strip())
    return entries

def _flavor_names(value: str) -> list[str]:
    """Resolve one ``--flavor`` argument, which may name several joined by '+'."""
    names = [name.strip() for name in re.split(r"[+,]", value) if name.strip()]
    if unknown := [name for name in names if name not in _FLAVORS]:
        raise argparse.ArgumentTypeError(
            f"unknown flavor: {', '.join(unknown)}; choose from {', '.join(_FLAVOR_NAMES)}",
        )
    return [_FLAVORS[name].name for name in names]


class FlavorSelection:
    def __init__(self, names: list[str] | tuple[str, ...] | None = None):
        selected = [_FLAVORS[name] for name in names] if names else [_DEFAULT_FLAVOR]
        # Named order is kept, so a run directory reads the way the run was
        # asked for. What a combination does is settled in priority order, so
        # it never depends on how it was typed.
        self.items = tuple({flavor.name: flavor for flavor in selected}.values())
        self.settled = tuple(sorted(self.items, key=lambda flavor: (flavor.priority, flavor.name)))
        self._reject_contradictions()
        self.policy = FlavorPolicy()
        for flavor in self.settled:
            flavor.configure(self.policy)

    def _reject_contradictions(self) -> None:
        """Refuse a combination where one variant undoes another.

        Two variants contradict when one adds an option the other drops in a
        scope they share, which is what a run cannot carry both of. A variant
        replacing its own option is not a contradiction.
        """
        for adder in self.items:
            for dropper in self.items:
                if adder is dropper:
                    continue
                clash = {
                    flag
                    for added in adder.edits
                    for dropped in dropper.edits
                    if _scopes_overlap(added.scope, dropped.scope)
                    for flag in {_flag(option) for option in added.add} & set(dropped.drop)
                }
                if clash:
                    raise ValueError(
                        f"{adder.name} adds {', '.join(sorted(clash))}, which {dropper.name} drops",
                    )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(flavor.name for flavor in self.items)

    @property
    def tag(self) -> str:
        """How the run named this combination, in the order it named it."""
        return "-".join(self.names)

    @property
    def settled_tag(self) -> str:
        """One spelling per combination, whatever order it was named in.

        A baseline, a report and a digest identify a configuration, so they
        must not move when the same combination is typed another way.
        """
        return "-".join(flavor.name for flavor in self.settled)

    @property
    def table_keys(self) -> tuple[str, ...]:
        """Names a per-flavor table is looked up by, most specific first.

        A checkpoint variant mixed into a run changes the tag but not what the
        run costs or where it belongs, so the execution mode answers for the
        combinations no table names.
        """
        mode = next(
            (flavor.name for flavor in self.settled if flavor.execution_mode),
            self.items[0].name,
        )
        # A table names a combination, not the order it was typed in, so the
        # settled spelling is looked up alongside the one the run named.
        return tuple(dict.fromkeys((self.tag, self.settled_tag, mode)))

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        for flavor in self.settled:
            flavor.transform(opts, env)


@dataclass(frozen=True)
class RunBlock:
    """Represents one ``--run`` header and the arguments scoped beneath it."""

    entry: str
    argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSpec:
    name: str
    variants: tuple[str, ...]
    flavors: FlavorSelection
    args: argparse.Namespace
    directory_tags: tuple[str, ...]


_RUN_SCOPED_OPTIONS = frozenset(
    {
        "flavor",
        "megatron_commit",
        "megatron_repo_url",
        "nvrx_commit",
        "nvrx_repo_url",
        "checkpoints_per_run",
        "checkpoint_interval",
        "repeat",
    },
)

def _run_dir_name(spec: RunSpec) -> str:
    """The directory one run writes to, inside the job's own directory.

    Run-specific flavors and overrides lead the name, while common options stay
    in the parent job directory. This keeps the parent compact without allowing
    two materially different children to share a path.
    """
    return f"{'-'.join(spec.directory_tags)}.{spec.name}" if spec.directory_tags else spec.name

def _run_directory_tags(common: argparse.Namespace, resolved: argparse.Namespace) -> tuple[str, ...]:
    """Identifies only what one run adds to the common job options."""
    common_flavors = set(_selected_flavors(common).names)
    tags = [name for name in resolved.flavors.names if name not in common_flavors]
    for field, label in (("megatron_commit", "mlm"), ("nvrx_commit", "nvrx")):
        value = getattr(resolved, field)
        if value != getattr(common, field):
            tags.append(f"{label}-{value[:8]}")
    for field, label in (
        ("checkpoints_per_run", "ckpts"),
        ("checkpoint_interval", "interval"),
        ("repeat", "repeat"),
    ):
        value = getattr(resolved, field)
        if value != getattr(common, field):
            tags.append(f"{label}-{value if value is not None else 'default'}")
    scoped_extra = resolved.extra_args[len(common.extra_args):]
    if scoped_extra:
        digest = hashlib.sha256("\0".join(scoped_extra).encode()).hexdigest()[:8]
        tags.append(f"opts-{digest}")
    return tuple(tags)

def _run_arguments(args: argparse.Namespace, variants: tuple[str, ...], argv: tuple[str, ...]) -> argparse.Namespace:
    """Overlays one run block on the fully resolved common arguments."""
    overrides, extra = _parser().parse_known_args(list(argv))
    if disallowed := sorted(set(vars(overrides)) - _RUN_SCOPED_OPTIONS):
        rendered = ", ".join("--" + name.replace("_", "-") for name in disallowed)
        raise ValueError(f"allocation option(s) must precede the first --run: {rendered}")
    resolved = argparse.Namespace(**vars(args))
    common_flavors = list(args.flavor or [])
    run_flavors = list(getattr(overrides, "flavor", None) or [])
    resolved.flavor = [*common_flavors, *variants, *run_flavors]
    for name, value in vars(overrides).items():
        if name != "flavor":
            setattr(resolved, name, value)
    resolved.flavors = FlavorSelection(resolved.flavor)
    resolved.extra_args = [*args.extra_args, *extra]
    return resolved

def _run_specs(args: argparse.Namespace) -> tuple[RunSpec, ...]:
    """The runs this job performs, each with the variants it adds to the common set.

    A run is a full two-phase training that records what its checkpoints
    contain. Two runs differ by the variants they name, and the diff between
    their digests is the comparison.
    """
    blocks = getattr(args, "run_blocks", None)
    if not blocks:
        blocks = tuple(RunBlock(entry) for entry in (args.run or [""]))
    specs = tuple(
        RunSpec(name, tuple(variants), run_args.flavors, run_args, _run_directory_tags(args, run_args))
        for block in blocks
        for name, variants in (
            (block.entry.partition(":")[0], _flavor_names(block.entry.partition(":")[2])),
        )
        for run_args in (_run_arguments(args, tuple(variants), block.argv),)
    )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"--run names must be unique: {', '.join(names)}")
    return specs

def _selected_flavors(args: argparse.Namespace) -> FlavorSelection:
    if not isinstance(args.flavors, FlavorSelection):
        args.flavors = FlavorSelection(args.flavors)
    return args.flavors


def _profile_config(name: str, parents: tuple[str, ...] = ()) -> dict:
    if name in parents:
        chain = " -> ".join((*parents, name))
        raise ValueError(f"cyclic model config inheritance: {chain}")
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown model config {name!r}")
    profile = _load_yaml(path)
    parent = profile.pop("inherits", None)
    if parent is None:
        return profile
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"model config {name!r} has invalid inherits value {parent!r}")
    return _deep_merge(_profile_config(parent, (*parents, name)), profile)


def _model_config(name: str) -> dict:
    base = _load_yaml(CONFIGS_DIR / "base.yaml")
    return _deep_merge(base, _profile_config(name))

def _pretrain_entrypoint(config_name: str) -> str:
    metadata = _model_config(config_name).get("ci") or {}
    entrypoint = metadata.get("pretrain-entry", "hybrid")
    if entrypoint not in _PRETRAIN_ENTRYPOINTS:
        choices = ", ".join(sorted(_PRETRAIN_ENTRYPOINTS))
        raise ValueError(f"model config {config_name!r} has invalid CI pretrain entrypoint {entrypoint!r}; choose {choices}")
    return entrypoint


def _time_budget(config_name: str, flavors: FlavorSelection, use: str, default: int) -> int:
    table = (_model_config(config_name).get("job_time_in_mins") or {}).get(use) or {}
    selected = next((table[key] for key in flavors.table_keys if key in table), table.get("default", default))
    return int(selected)

def _job_partition(config_name: str, flavors: FlavorSelection) -> str | None:
    table = (_model_config(config_name).get("job_partition") or {}).get(cluster_tag()) or {}
    selected = next((table[key] for key in flavors.table_keys if key in table), table.get("default", ""))
    return str(selected).strip() or None

@dataclass
class TrainConfig:
    nodes: int
    gpus: int
    save: Path
    load: Path | None
    model_config: str
    provider: str
    phase: int
    save_interval: int = 250
    checkpoints: int = 3
    extend_training: bool = False
    start_iteration: int = 0


@dataclass(frozen=True)
class RoutineStart:
    phase: int = 1
    cycle: int = 0
    resume_iteration: int | None = None
    checkpoints: int | None = None
    complete: bool = False


def _complete_checkpoint_iterations(root: Path) -> list[int]:
    """Iterations whose distributed checkpoint metadata was committed."""
    complete = []
    for path in root.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", path.name)
        if match and (path / ".metadata").is_file():
            complete.append(int(match[1]))
    return sorted(complete)


def _routine_start(cfg: str, args: argparse.Namespace, policy: FlavorPolicy) -> RoutineStart:
    """Resume the current routine after an ft_launcher worker restart.

    A fresh Slurm submission intentionally starts a fresh validation even when
    its named run directory already exists. Only a launcher-owned restart may
    consume checkpoints written by the preceding elastic cycle.
    """
    if not _in_job_array() or int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")) == 0:
        return RoutineStart()

    run = _run_dir(cfg)
    phase1 = run / f"phase1-checkpoints/iter_{policy.phase1_train_iters:07d}/.metadata"
    if not phase1.is_file():
        return RoutineStart()

    iterations = _complete_checkpoint_iterations(run / "checkpoints")
    checkpoints = policy.phase2_checkpoints if args.checkpoints_per_run is None else args.checkpoints_per_run
    if checkpoints == 0:
        return RoutineStart(phase=2, resume_iteration=iterations[-1] if iterations else None, checkpoints=0)

    completed = len(iterations)
    cycle, within_cycle = divmod(completed, checkpoints)
    if args.repeat and cycle >= args.repeat:
        return RoutineStart(phase=2, cycle=cycle, complete=True)
    remaining = checkpoints - within_cycle if within_cycle else None
    return RoutineStart(
        phase=2,
        cycle=cycle,
        resume_iteration=iterations[-1] if iterations else None,
        checkpoints=remaining,
    )

def _train_options(cfg: str, config: TrainConfig) -> list[str]:
    run = _run_dir(cfg)
    phase1 = run / "phase1-checkpoints"
    total_iters = (
        (config.start_iteration // config.save_interval + config.checkpoints) * config.save_interval
        if config.checkpoints
        else None
    )
    dynamic = {
        "tensor-model-parallel-size": config.gpus,
        "expert-model-parallel-size": config.gpus,
        "global-batch-size": config.nodes * config.gpus,
        "seq-length": 1024,
        "max-position-embeddings": 1024,
        "data-cache-path": str(run / "data-cache"),
        "tensorboard-dir": str(run / "tensorboard"),
        "log-interval": config.save_interval,
        "eval-interval": total_iters * 2 if total_iters else _LONG_RUN_TRAIN_SAMPLES,
        "exit-interval": total_iters,
    }
    model = {
        k: v
        for k, v in _model_config(config.model_config).items()
        if k not in {"ci", "job_partition", "job_time_in_mins"}
    }
    if config.extend_training:
        model["train-samples"] = _LONG_RUN_TRAIN_SAMPLES
    elif total_iters:
        model["train-samples"] = max(int(model["train-samples"]), total_iters * config.nodes * config.gpus)
    synchronous = config.phase == 1
    if synchronous:
        for key in ("async-save", "async-strategy", "use-persistent-ckpt-worker"):
            model.pop(key, None)
    checkpoint = {
        "ckpt-format": "torch_dist",
        "ckpt-assume-constant-structure": True,
        "save": str(config.save),
        "load": str(config.load) if config.load else None,
        "save-interval": config.save_interval,
        "save-retain-interval": config.save_interval,
        "async-save": not synchronous,
        "use-persistent-ckpt-worker": not synchronous,
    }
    for path in (config.save, phase1, run / "data-cache", run / "tensorboard"):
        path.mkdir(parents=True, exist_ok=True)
    return _flatten({**dynamic, **model, **checkpoint})

def _sync_root() -> Path:
    """Where ranks leave the files they hand off to each other through.

    A rank per checkpoint per barrier is thousands of empty files, and they
    are handshakes rather than results, so they stay out of the directory a
    run's logs are read from. The launcher removes this directory once the
    allocation it coordinated has exited.

    A run directory holds them once one is configured. Before that they belong
    under the results directory, which is a place the repository ignores.
    """
    root = (RUN_ROOT or RESULTS_DIR) / SYNC_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root

def _node_once(tag: str, action: Callable[[], None]) -> None:
    host = socket.gethostname().split(".")[0]
    marker = _sync_root() / f"{_sync_job_id()}-{tag}-{host}"
    local_rank = int(_rank_env("SLURM_LOCALID", "LOCAL_RANK", "0"))
    tasks_spec = (
        (os.environ.get("LOCAL_WORLD_SIZE") if _in_job_array() else None)
        or os.environ.get("SLURM_NTASKS_PER_NODE")
        or os.environ.get("SLURM_TASKS_PER_NODE")
        or os.environ.get("LOCAL_WORLD_SIZE", "1")
    )
    local_tasks = int(tasks_spec.partition("(")[0])
    marker.with_name(f"{marker.name}.ready-{local_rank}").touch()
    deadline = time.monotonic() + 3600
    while len(list(marker.parent.glob(f"{marker.name}.ready-*"))) < local_tasks:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for node-local ranks before {tag}")
        time.sleep(0.25)
    failed = marker.with_suffix(".failed")
    if local_rank == 0:
        try:
            action()
        except BaseException:
            failed.touch()
            raise
        finally:
            marker.touch()
    else:
        wait_for_file(marker, 3600, poll_s=1)
        if failed.exists():
            raise RuntimeError(f"node-local action {tag} failed")

def _sync_ranks(tag: str) -> None:
    marker = _sync_root() / f"{_sync_job_id()}-{tag}"
    rank = int(_rank_env("SLURM_PROCID", "RANK", "0"))
    world = int(_rank_env("SLURM_NTASKS", "WORLD_SIZE", "1"))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.with_name(f"{marker.name}.ready-{rank}").touch()
    deadline = time.monotonic() + 3600
    while len(list(marker.parent.glob(f"{marker.name}.ready-*"))) < world:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ranks after {tag}")
        time.sleep(0.25)

def _reject_server_errors(logs: Iterable[Path]) -> None:
    contents = {log: log.read_text(errors="replace") for log in logs if log.exists()}
    hard_stops = [log for log, text in contents.items() if "spindown timed out before idle" in text]
    if hard_stops:
        raise RuntimeError(f"checkpoint server aborted in-flight work; see {', '.join(map(str, hard_stops))}")
    failed = [log for log, text in contents.items() if re.search(r"(?m)^(?:.*\[ERROR\]|Error:)", text)]
    if failed:
        raise RuntimeError(f"checkpoint server reported errors; see {', '.join(map(str, failed))}")

def _heartbeat_server(stopped: threading.Event) -> None:
    while not stopped.wait(60):
        CHECKPOINT_SERVER.request("ping", 1)


@contextlib.contextmanager
def _server_heartbeat(enabled: bool) -> Generator[None, None, None]:
    if not enabled or _rank_env("SLURM_LOCALID", "LOCAL_RANK", "0") != "0":
        yield
        return
    stopped = threading.Event()
    thread = threading.Thread(target=_heartbeat_server, args=(stopped,), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()

def _extra_options(cfg: str, selected: list[str]) -> list[str]:
    """Split option text, resolving the run-scoped cache directory in it."""
    if not any(selected):
        return []
    cache = _run_dir(cfg) / "pg-tensors-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return [
        token
        for text in selected
        for token in shlex.split(
            string.Template(text.replace("@PG_CACHE_DIR@", "$PG_CACHE_DIR")).safe_substitute(
                PG_CACHE_DIR=cache.as_posix(),
            ),
        )
    ]

def _phase_command(
    cfg: str,
    args: argparse.Namespace,
    provider: str,
    phase: int,
    cycle: int = 0,
    *,
    resume_iteration: int | None = None,
    checkpoints_override: int | None = None,
) -> tuple[list[str], dict]:
    flavors = _selected_flavors(args)
    config = flavors.policy.train_config(
        cfg,
        args,
        provider,
        phase,
        cycle,
        resume_iteration=resume_iteration,
        checkpoints_override=checkpoints_override,
    )
    opts, env = _train_options(cfg, config), {**os.environ, **_base_env()}
    for name, value in _MEGATRON_OTEL_ENV_DEFAULTS.items():
        env.setdefault(name, value)
    if "NVTE_FLASH_ATTN_V4" not in env and _gpu_compute_capability() == "9.0":
        # FlashAttention 4's Hopper backward pass stalls, so Hopper trains with FlashAttention 2.
        env["NVTE_FLASH_ATTN_V4"] = "0"
    include_nvrx = not (provider == "native" and flavors.policy.requires_nvrx_absent)
    _toolchain(cfg, args).apply(env, include_nvrx=include_nvrx)
    flavors.transform(opts, env)
    applied = flavors.policy.apply_options(opts, cfg, phase)
    if provider == "native":
        # Native comparisons omit MCore's unused process-group cache options.
        for flag in ("--ckpt-pg-tensors-cache-path", "--ckpt-pg-tensors-cache-create"):
            if any(_flag(option) == flag for option in opts):
                _drop(opts, flag)
                applied.append(f"drop {flag}")
    _print_flavor_options(flavors, phase, applied)
    expert = opts[opts.index("--expert-model-parallel-size") + 1]
    env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = expert
    if "--fine-grained-activation-offloading" in opts:
        env["NVTE_CPU_OFFLOAD_V1"] = "1"
    os.environ["NATIVE_CKPT_SERVER_LOG"] = env["NATIVE_CKPT_SERVER_LOG"]
    entrypoint = _pretrain_entrypoint(config.model_config)
    return [
        "python3", "-u", str(Path(__file__).resolve()), _PRETRAIN_ENTRY_CMD,
        str(phase), entrypoint, *opts, *args.extra_args,
    ], env
def _print_flavor_options(flavors: FlavorSelection, phase: int, applied: list[str]) -> None:
    """Record what the selected variants changed about this phase's command."""
    if _rank_env("SLURM_PROCID", "RANK", "0") != "0" or not applied:
        return
    print(f"[slurm_job:flavor] {flavors.tag} phase{phase} {' '.join(applied)}", flush=True)  # noqa: T201

def _print_pretrain_env(env: dict[str, str], cycle_tag: str) -> None:
    if _rank_env("SLURM_PROCID", "RANK", "0") != "0":
        return
    print(f"[slurm_job:env] {cycle_tag} rank0 environment", flush=True)  # noqa: T201
    for name in sorted(_RUN_ENV_NAMES | _PRETRAIN_ENV_NAMES | _TRAIN_ENV_DEFAULTS.keys()):
        if name in env:
            print(f"[slurm_job:env] {name}={shlex.quote(env[name])}", flush=True)  # noqa: T201

def _run_routine(cfg: str, args: argparse.Namespace, provider: str, label: str = "") -> None:
    flavors = _selected_flavors(args)
    stem = f"{platform_tag()}_{flavors.settled_tag}_{cfg}"
    start = _routine_start(cfg, args, flavors.policy)
    phase, cycle = start.phase, start.cycle
    if start.complete:
        if _rank_env("SLURM_PROCID", "RANK", "0") == "0":
            logger.info(f"[slurm_job:restart] {label or 'run'} already completed; reusing durable checkpoints")
        return
    resume_iteration, checkpoints_override = start.resume_iteration, start.checkpoints
    if resume_iteration is not None and _rank_env("SLURM_PROCID", "RANK", "0") == "0":
        logger.info(
            f"[slurm_job:restart] {label or 'run'} resuming phase 2 from iteration {resume_iteration}",
        )
    while phase == 1 or args.repeat == 0 or cycle < args.repeat:
        expected_checkpoints = flavors.policy.phase1_checkpoints if phase == 1 else flavors.policy.phase2_checkpoints
        if phase == 2 and args.checkpoints_per_run is not None:
            expected_checkpoints = args.checkpoints_per_run
        if checkpoints_override is not None:
            expected_checkpoints = checkpoints_override
        cmd, env = _phase_command(
            cfg,
            args,
            provider,
            phase,
            cycle,
            resume_iteration=resume_iteration,
            checkpoints_override=checkpoints_override,
        )
        resume_iteration = checkpoints_override = None
        rank = _rank_env("SLURM_PROCID", "RANK", "0")
        log = RESULTS_DIR / f"{cfg}_{label}-routine_rank{rank}.log" if label else None
        cycle_tag = "phase1" if phase == 1 else f"cycle{cycle + 1}"
        server_log = str(RESULTS_DIR / f"checkpoint_server.{cycle_tag}.%h.%p.log")
        os.environ["NATIVE_CKPT_SERVER_LOG"] = server_log
        env["NATIVE_CKPT_SERVER_LOG"] = server_log
        otel_cycle_dir = RESULTS_DIR / "otel" / cycle_tag
        otel_cycle_dir.mkdir(parents=True, exist_ok=True)
        otel_file = otel_cycle_dir / "native_ckpt_otel.jsonl"
        megatron_otel_enabled = env.get("MEGATRON_OTEL_ENABLED") == "1"
        collect_otel = flavors.policy.manages_native_server and (
            provider == "native" or megatron_otel_enabled or "detr" in flavors.names
        )
        if collect_otel:
            os.environ["NATIVE_CKPT_OTEL_FILE"] = str(otel_file)
            env["NATIVE_CKPT_OTEL_FILE"] = str(otel_file)
        else:
            os.environ.pop("NATIVE_CKPT_OTEL_FILE", None)
            env.pop("NATIVE_CKPT_OTEL_FILE", None)
        otel_reporting = flavors.policy.otel_reporting and (
            bool(env.get("NATIVE_CKPT_OTEL_FILE")) or megatron_otel_enabled
        )
        perfetto_file = otel_cycle_dir / "native_ckpt_perfetto.json"
        if collect_otel and rank == "0":
            perfetto_file.unlink(missing_ok=True)
            otel_report.analysis_path(otel_file).unlink(missing_ok=True)
        relay_otel = provider != "native" and collect_otel
        collector_node = (
            socket.gethostname().split(".", 1)[0] == env["MASTER_ADDR"].split(".", 1)[0]
            if relay_otel
            else int(rank) < args.gpus_per_node
        )
        collector = "" if not relay_otel else ("file" if collector_node else f"{env['MASTER_ADDR']}:4100")
        if collector:
            os.environ["NATIVE_CKPT_OTEL_COLLECTOR"] = collector
            env["NATIVE_CKPT_OTEL_COLLECTOR"] = collector
        else:
            os.environ.pop("NATIVE_CKPT_OTEL_COLLECTOR", None)
            env.pop("NATIVE_CKPT_OTEL_COLLECTOR", None)
        job_id = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID", "local")
        slurm_restart = "0" if _in_job_array() else env.get("SLURM_RESTART_COUNT", "0")
        elastic_restart = env.get("TORCHELASTIC_RESTART_COUNT", "0")
        run_id = f"{job_id}-{RUN_NAME}-{cycle_tag}-sr{slurm_restart}-te{elastic_restart}"
        env["MEGATRON_OTEL_RUN_ID"] = run_id
        os.environ["MEGATRON_OTEL_RUN_ID"] = run_id
        server_enabled = collect_otel
        if server_enabled:
            from nvrx_checkpointing import checkpoint  # noqa: PLC0415

            if relay_otel:
                if collector_node:
                    _node_once(f"{label}-{cycle_tag}-otel-collector".strip("-"), checkpoint.ping)
                _sync_ranks(f"{label}-{cycle_tag}-otel-collector-ready".strip("-"))
                if not collector_node:
                    _node_once(f"{label}-{cycle_tag}-otel-relay".strip("-"), checkpoint.ping)
            else:
                _node_once(f"{label}-{cycle_tag}-otel-listener".strip("-"), checkpoint.ping)
            _sync_ranks(f"{label}-{cycle_tag}-otel-listeners-ready".strip("-"))
        if rank == "0":
            print(f"[native_ckpt:run] {cycle_tag}", flush=True)  # noqa: T201
        with (
            flavors.policy.server_context(f"{stem}-{cycle_tag}", phase),
            _server_heartbeat(server_enabled),
        ):
            _print_pretrain_env(env, cycle_tag)
            rc = (
                _tee_rc(
                    cmd,
                    log,
                    env,
                    append=True,
                    # Retain complete per-rank logs as artifacts, but mirror only
                    # rank 0 into the size-limited GitLab job trace.
                    mirror_stdout=rank == "0",
                )
                if log
                else subprocess.run(cmd, env=env, check=False).returncode
            )
            if rc:
                raise RuntimeError(f"training phase {phase} failed (rc={rc})")
            _sync_ranks(f"{label}-{cycle_tag}".strip("-"))
        if phase == 1:
            wait_for_file(
                _run_dir(cfg) / f"phase1-checkpoints/iter_{flavors.policy.phase1_train_iters:07d}/.metadata",
                _time_budget(args.config, flavors, "phase1-flush", 30) * 60,
            )
        if server_enabled and not collector_node:
            flavors.policy.stop_server(provider, f"{label}-{cycle_tag}-relay-stop")
        _sync_ranks(f"{label}-{cycle_tag}-relays-stopped".strip("-"))
        if server_enabled and collector_node:
            flavors.policy.stop_server(provider, f"{label}-{cycle_tag}-collector-stop")
        _sync_ranks(f"{label}-{cycle_tag}-stopped".strip("-"))
        if rank == "0":
            otel_data = otel_report.OtelData([])
            if collect_otel:
                if not otel_file.is_file():
                    raise RuntimeError(f"no collected OTLP file: {otel_file}")
                otel_data = otel_report.OtelData([otel_file])
                if otel_reporting:
                    otel_to_perfetto.convert_records(otel_data.records, perfetto_file)
                    analysis = otel_report.validate_collected(
                        otel_data,
                        run_id,
                        args.nodes,
                        args.gpus_per_node,
                        provider,
                        expected_checkpoints,
                    )
                    otel_report.write_analysis(otel_file, analysis)
            otel_report.write_process_memory_charts(otel_data, otel_cycle_dir)
        if phase == 1:
            phase = 2
        else:
            cycle += 1
def _manifest(iter_dir: Path, baseline: Path | None = None) -> int:
    """Record what one checkpoint holds, and diff it against a baseline if given.

    Reading a checkpoint means importing what wrote it, so the tool is given
    the Megatron-LM checkout rather than relying on it being importable
    already. This package comes from the installed wheel: putting the checkout
    on the path ahead of it makes the import build from source instead.
    """
    entries = [megatron_root().as_posix(), *os.environ.get("PYTHONPATH", "").split(":")]
    env = {
        **os.environ,
        "PYTHONPATH": ":".join(dict.fromkeys(entry for entry in entries if entry)),
        "TORCHINDUCTOR_WORKER_START": "fork",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "nvrx_checkpointing.metadata_compare",
            "--checkpoint-dir",
            str(iter_dir),
            "--baseline-dir",
            str(baseline or iter_dir / "unseeded"),
            "--provider",
            "native" if baseline else "nvrx",
        ],
        env=env,
        check=False,
    ).returncode

def _manifest_batch(  # noqa: C901
    tag: str,
    iterations: list[tuple[str, Path]],
    baseline_root: Path | None,
    stash_root: Path | None,
    args: argparse.Namespace,
) -> list[str]:
    """Distribute manifest work across node leaders and synchronize all ranks.

    Each iteration at index i is assigned to node leader i % nodes. Node
    leaders run manifests concurrently; all ranks synchronize after via
    _sync_ranks. Returns a sorted list of iteration paths that failed.

    When baseline_root is set, each manifest is compared against the
    baseline for that iteration. When stash_root is set, the produced
    .info files are copied there (used during reference-manifest creation).
    """
    rank = int(_rank_env("SLURM_PROCID", "RANK", "0"))
    local_rank = int(_rank_env("SLURM_LOCALID", "LOCAL_RANK", "0"))
    node_index = rank // args.gpus_per_node
    is_node_leader = local_rank == 0
    job_id = _sync_job_id()

    if is_node_leader and iterations:
        mine = [c for i, c in enumerate(iterations) if i % args.nodes == node_index]
        if mine:
            done_f = _sync_root() / f"{job_id}-{tag}-node{node_index}.done"
            fail_f = _sync_root() / f"{job_id}-{tag}-node{node_index}.failed"
            try:
                for group, iter_dir in mine:
                    baseline = (baseline_root / group / iter_dir.name) if baseline_root else None
                    if _manifest(iter_dir, baseline):
                        raise RuntimeError(f"manifest failed for {iter_dir}")  # noqa: TRY301
                    if stash_root is not None:
                        target = stash_root / group / iter_dir.name
                        target.mkdir(parents=True, exist_ok=True)
                        for name in ("reference.metadata.info", "common_pt_dump.info"):
                            shutil.copy2(iter_dir / name, target / name)
                done_f.touch()
            except Exception:  # noqa: BLE001
                fail_f.touch()

    _sync_ranks(tag)

    failures: list[str] = []
    for node_idx in range(min(args.nodes, len(iterations))):
        assigned = [c for i, c in enumerate(iterations) if i % args.nodes == node_idx]
        if not assigned:
            continue
        done_f = _sync_root() / f"{job_id}-{tag}-node{node_idx}.done"
        fail_f = _sync_root() / f"{job_id}-{tag}-node{node_idx}.failed"
        if fail_f.exists() or not done_f.exists():
            failures.extend(str(c[1]) for c in assigned)
    return sorted(failures)

# Everything a checkpoint records about itself: the per-chunk tensor manifest
# and the dump of the common state beside it, both `key=value` text.
_MANIFESTS = (("tensor", "reference.metadata.info"), ("common", "common_pt_dump.info"))

def _write_checkpoint_digest(
    run: Path,
    stem: str,
    baseline_for: Callable[[str, str], Path] | None = None,
) -> Path:
    """Write everything this run's checkpoints recorded, and how it compares.

    The digest lists one ``<group>/<iter> <source>/<key>=<value>`` line per
    entry, sorted, so the digests of two runs diff directly and a run needs no
    second routine beside it to be comparable. It is compressed because a
    manifest of every tensor runs to tens of megabytes. The summary beside it
    stays plain text: it counts what each checkpoint holds, which is what a
    reader needs before deciding whether to open the digest at all.
    """
    lines, summary = [], []
    for group in ("phase1-checkpoints", "checkpoints"):
        for iteration in sorted((run / group).glob("iter_*")):
            produced = _checkpoint_entries(iteration, run)
            if produced is None:
                continue
            label = f"{group}/{iteration.name}"
            lines += [f"{label} {key}={value}\n" for key, value in sorted(produced.items())]
            recorded = _checkpoint_entries(baseline_for(group, iteration.name)) if baseline_for else None
            if recorded is None:
                summary.append(f"{label}: entries={len(produced)}")
                continue
            summary.append(f"{label}: entries={len(produced)} {_compare_entries(produced, recorded)}")
    if not lines:
        raise RuntimeError(f"no checkpoint metadata found under {run}; digest was not written")
    (RESULTS_DIR / f"{stem}_checkpoint_summary.log").write_text("\n".join(summary) + "\n")
    output = RESULTS_DIR / f"{stem}_checkpoint_digest.xz"
    with lzma.open(output, "wt") as digest:
        digest.writelines(lines)
    for line in summary:
        logger.info(f"[slurm_job:digest] {stem} {line}")
    logger.info(f"[slurm_job:digest] wrote {output}")
    return output

def _compare_entries(produced: dict[str, str], recorded: dict[str, str]) -> str:
    """Count how one checkpoint's entries differ from what it is compared against.

    A recorded baseline holds the tensor manifest alone, so only the sources
    both sides recorded are counted; a source one side never wrote is reported
    as unrecorded rather than as every one of its entries going missing.
    """
    sources = {key.partition("/")[0] for key in recorded}
    scoped = {key: value for key, value in produced.items() if key.partition("/")[0] in sources}
    missing = sum(key not in scoped for key in recorded)
    extra = sum(key not in recorded for key in scoped)
    changed = sum(scoped[key] != recorded[key] for key in scoped.keys() & recorded.keys())
    unrecorded = sorted({key.partition("/")[0] for key in produced} - sources)
    counted = f"baseline={len(recorded)} missing={missing} extra={extra} changed={changed}"
    return f"{counted} unrecorded={','.join(unrecorded)}" if unrecorded else counted

# Recorded state that names where or how a run wrote its checkpoint rather than
# what it computed. DCP may also select a different rank's copy of replicated
# common state. Comparing these fields would report equivalent checkpoints as
# different while model and training arguments remain fully checked.
_RUN_SCOPED_KEYS = frozenset(
    {
        "common/args.async_ckpt_use_cpu_shm",
        "common/args.async_save",
        "common/args.async_strategy",
        "common/args.ckpt_fully_parallel_save",
        "common/args.ckpt_fully_parallel_save_deprecated",
        "common/args.ckpt_load_validate_sharding_integrity",
        "common/args.ckpt_provider",
        "common/args.local_rank",
        "common/args.rank",
        "common/args.use_persistent_ckpt_worker",
    },
)

_CHUNK_RE = re.compile(r"^(?P<fqn>.+)\.chunks\.(?P<index>\d+)\.(?P<field>[a-z0-9_]+)$")

def _by_chunk_identity(entries: dict[str, str]) -> dict[str, str]:
    """Key each chunk by where it sits in the tensor, not by where it was listed.

    Two writers may enumerate or replicate the same chunks in their own orders,
    which is not a difference in what they wrote. Keying by the offsets a chunk
    covers compares like with like; a chunk one side never wrote is still
    absent from the other.
    """
    offsets = {
        (m["fqn"], m["index"]): value
        for key, value in entries.items()
        if (m := _CHUNK_RE.match(key)) and m["field"] == "offsets"
    }
    rekeyed = {}
    for key, value in entries.items():
        match = _CHUNK_RE.match(key)
        where = offsets.get((match["fqn"], match["index"])) if match else None
        rekeyed[f"{match['fqn']}.chunk{where}.{match['field']}" if where else key] = value
    return rekeyed

def _checkpoint_entries(path: Path, run: Path | None = None) -> dict[str, str] | None:
    """Read everything one checkpoint recorded, keyed by the source that holds it."""
    entries: dict[str, str] = {}
    for source, name in _MANIFESTS:
        text = _manifest_text(path / name)
        if text is None:
            continue
        entries.update(
            (f"{source}/{key}", value)
            for key, value in _by_chunk_identity(
                {
                    key: value
                    for key, _, value in (line.partition("=") for line in text.splitlines() if "=" in line)
                },
            ).items()
        )
    located = run.as_posix() if run else None
    return {
        key: value
        for key, value in entries.items()
        if key not in _RUN_SCOPED_KEYS and not (located and located in value)
    } or None

def _manifest_text(path: Path) -> str | None:
    """Read one manifest, or the compressed form a recorded baseline keeps."""
    compressed = path.with_suffix(f"{path.suffix}.xz")
    if path.is_file():
        return path.read_text(errors="replace")
    if compressed.is_file():
        with lzma.open(compressed, "rt", errors="replace") as handle:
            return handle.read()
    return None

def _report_comparison(run: Path, baseline_for: Callable[[str, str], Path], stem: str = "run") -> None:
    """Diff every checkpoint the measured routine wrote against its baseline."""
    _write_checkpoint_digest(run, f"{stem}_compare", baseline_for)
    failures = [
        str(iteration)
        for group in ("phase1-checkpoints", "checkpoints")
        for iteration in sorted((run / group).glob("iter_*"))
        if _manifest(iteration, baseline_for(group, iteration.name))
    ]
    if failures:
        raise RuntimeError(f"reference check failed: {', '.join(failures)}")

@contextlib.contextmanager
def _in_run(spec: RunSpec, _args: argparse.Namespace) -> Generator[None, None, None]:
    """Point the run-scoped paths and variants at one run for its duration."""
    global RESULTS_DIR, RUN_ROOT  # noqa: PLW0603
    previous = (RUN_ROOT, RESULTS_DIR)
    scoped_env = (
        "MEGATRON_ROOT",
        "MEGATRON_REPO_URL",
        "MEGATRON_COMMIT",
        "NVRX_REPO_URL",
        "NVRX_COMMIT",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
    )
    previous_env = {name: os.environ.get(name) for name in scoped_env}
    # A run writes inside the job that owns it, in a directory named for the
    # variants it adds, so the job directory holds the whole comparison and
    # one path says which side of it this is.
    if spec.name:
        RUN_ROOT = previous[0] / _run_dir_name(spec)
        RESULTS_DIR = RUN_ROOT / "logs"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SLURMJOB_OPT_RUN_DIR"] = str(RUN_ROOT)
    os.environ["NATIVE_CKPT_SERVER_LOG"] = str(RESULTS_DIR / "checkpoint_server.%h.%p.log")
    cfg = f"{spec.args.config}_{spec.args.nodes}x{spec.args.gpus_per_node}"
    provider = _provider(spec.flavors)
    include_nvrx = not (provider == "native" and spec.flavors.policy.requires_nvrx_absent)
    _toolchain(cfg, spec.args).apply(os.environ, include_nvrx=include_nvrx)
    try:
        yield
    finally:
        RUN_ROOT, RESULTS_DIR = previous
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        os.environ["SLURMJOB_OPT_RUN_DIR"] = str(RUN_ROOT)
        os.environ["NATIVE_CKPT_SERVER_LOG"] = str(RESULTS_DIR / "checkpoint_server.%h.%p.log")


def _perform_runs(cfg: str, args: argparse.Namespace) -> None:
    """Train every run this job names, then compare what they wrote.

    Each run is independent: its own variants, its own directory, its own
    digest. Nothing one run writes is read by another, so the comparison is a
    diff of digests rather than a second routine checking the first.
    """
    rank0 = _rank_env("SLURM_PROCID", "RANK", "0") == "0"
    specs, digests = _run_specs(args), []
    comparable = len(specs) > 1 or args.baseline is not None or any(
        spec.flavors.policy.compares_routines for spec in specs
    )
    for spec in specs:
        with _in_run(spec, args):
            run_args, flavors = spec.args, spec.flavors
            if rank0:
                logger.info(f"[slurm_job:run] {spec.name or 'run'} {flavors.tag} "
                            f"{_toolchain(cfg, run_args).describe('sources')}")
            # Naming the routine keeps every rank's output in a per-rank file
            # and mirrors only rank 0 into the job trace. Without it 256 ranks
            # write straight to the trace and bury the run in its own logs.
            flavors.policy.run_worker(cfg, run_args, _provider(flavors), spec.name or "run")
            run = _run_dir(cfg)
            # Recording what a checkpoint holds costs a pass over every tensor,
            # which only earns its place when something will read it: another
            # run to diff against, a baseline to diff against, or a flavor whose
            # whole purpose is the comparison.
            iterations = [
                (group, iter_dir)
                for group in ("phase1-checkpoints", "checkpoints")
                for iter_dir in sorted((run / group).glob("iter_*"))
            ] if comparable else []
            # Node leaders share the manifest work, so a run's digest costs the
            # same whether it wrote four checkpoints or sixty-four.
            if failures := _manifest_batch(f"{spec.name or 'run'}-manifest", iterations, None, None, run_args):
                raise RuntimeError(f"manifest derivation failed: {', '.join(failures)}")
            if not rank0 or not comparable:
                continue
            stem = _identity(run_args)[2]
            digests.append(_write_checkpoint_digest(run, stem))
    if rank0:
        _compare_runs(args, specs, digests)


def _compare_runs(args: argparse.Namespace, specs: tuple[RunSpec, ...], digests: list[Path]) -> None:
    """Diff every run against the baseline, which is the first run by default.

    A single run with nothing named to compare against has nothing to diff, so
    it records its digest and stops.
    """
    baseline = _baseline_digest(args.baseline) if args.baseline else None
    if baseline is None and len(digests) < 2:
        return
    against = baseline or digests[0]
    failures = []
    for spec, digest in zip(specs, digests, strict=True):
        if digest == against:
            continue
        differences = _digest_differences(against, digest)
        if differences:
            failures.append(f"{spec.name or 'run'} ({differences} differences)")
    if failures:
        raise RuntimeError(f"checkpoints differ from {against.name}: {', '.join(failures)}")


def _baseline_digest(path: Path) -> Path:
    """Resolve a baseline to the digest it holds, given either the file or its run."""
    if path.is_file():
        return path
    found = sorted(path.glob("**/*_checkpoint_digest.xz"))
    if not found:
        raise SystemExit(f"no checkpoint digest under {path}")
    return found[0]


def _checkout_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"{revision or 'unknown'}{'+dirty' if dirty else ''}"

def _print_source_provenance() -> None:
    """Record the Megatron-LM and NVIDIA Resiliency Extension this phase loaded.

    Each routine of a cross-source comparison selects its own checkouts through
    ``MEGATRON_ROOT`` and ``PYTHONPATH``. Resolving them from inside the
    training process, after every import has bound, is what makes a comparison
    self-evidencing: the log names the revisions that ran rather than the ones
    that were requested.
    """
    root = megatron_root()
    print(f"[native_ckpt:sources] megatron={root}@{_checkout_revision(root)}", flush=True)  # noqa: T201
    try:
        import nvidia_resiliency_ext  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    except ImportError:
        # pretrain-native's perf flavor verifies the provider works with the
        # extension uninstalled, so its absence is a result, not a failure.
        print("[native_ckpt:sources] nvidia_resiliency_ext=absent", flush=True)  # noqa: T201
        return
    try:
        version = importlib.metadata.version(_NVRX_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    source = ""
    if os.environ.get("NVRX_COMMIT") == _WORKTREE_REVISION and (nvrx_root := nvrx_repo_root()) is not None:
        source = f" source={nvrx_root}@{_checkout_revision(nvrx_root)}"
    elif requested := os.environ.get("NVRX_COMMIT"):
        source = f" requested={requested}"
    print(  # noqa: T201
        f"[native_ckpt:sources] nvidia_resiliency_ext={nvidia_resiliency_ext.__file__}@{version}{source}",
        flush=True,
    )

class _PeakHbmSampler:
    """Sample whole-device HBM use and training-process RSS."""

    def __init__(self, torch_module: ModuleType, interval_seconds: float = 1.0):
        self.torch = torch_module
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.total_bytes = 0
        self.device: int | None = None
        self.rss_samples: list[tuple[int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="peak-hbm-sampler", daemon=True)

    def _sample_once(self) -> None:
        if (rss_bytes := _process_rss_bytes(str(os.getpid()))) is not None:
            self.rss_samples.append((time.time_ns(), rss_bytes))
        if not self.torch.cuda.is_initialized():
            return
        if self.device is None:
            requested = int(os.environ.get("LOCAL_RANK", "0"))
            count = self.torch.cuda.device_count()
            self.device = requested if requested < count else self.torch.cuda.current_device()
        free, total = self.torch.cuda.mem_get_info(self.device)
        self.peak_bytes = max(self.peak_bytes, total - free)
        self.total_bytes = total

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with contextlib.suppress(Exception):
                self._sample_once()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        with contextlib.suppress(Exception):
            self._sample_once()


def _emit_training_process_memory(samples: list[tuple[int, int]]) -> None:
    """Export one training rank's RSS samples through its node-local OTLP daemon."""
    if not samples:
        return

    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (  # noqa: PLC0415
        ExportLogsServiceRequest,
    )

    rank = int(os.environ.get("RANK", "0"))
    request = ExportLogsServiceRequest()
    resource_logs = request.resource_logs.add()
    for key, value in {
        "service.name": otel_report.TRAINING_SERVICE,
        "host.name": socket.gethostname().split(".", 1)[0],
        "process.pid": os.getpid(),
        "nemo.run.id": os.environ["MEGATRON_OTEL_RUN_ID"],
        "dl.rank": rank,
    }.items():
        attribute = resource_logs.resource.attributes.add()
        attribute.key = key
        if isinstance(value, int):
            attribute.value.int_value = value
        else:
            attribute.value.string_value = value

    scope_logs = resource_logs.scope_logs.add()
    scope_logs.scope.name = "nvrx.checkpoint.ci"
    peak_rss_bytes = 0
    for timestamp, rss_bytes in samples:
        peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
        record = scope_logs.log_records.add()
        record.time_unix_nano = timestamp
        record.observed_time_unix_nano = timestamp
        record.event_name = "process.memory"
        record.body.string_value = "process.memory"
        for key, value in {
            "nvrx.rank": rank,
            "nvrx.memory.rss_bytes": rss_bytes,
            "nvrx.memory.peak_rss_bytes": peak_rss_bytes,
            "nvrx.memory.trigger": "sampler",
        }.items():
            attribute = record.attributes.add()
            attribute.key = key
            if isinstance(value, int):
                attribute.value.int_value = value
            else:
                attribute.value.string_value = value

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if not endpoint:
        endpoint = f"{os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://127.0.0.1:4318').rstrip('/')}/v1/logs"
    http_request = urllib.request.Request(
        endpoint,
        data=request.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=30) as response:
        response.read()


def _emit_checkpoint_resource_metrics(
    provider: str,
    *,
    gpu_peak_bytes: int,
    gpu_hbm_peak_bytes: int,
    gpu_total_bytes: int,
    cpu_rss_bytes: int,
) -> None:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter  # noqa: PLC0415
    from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # noqa: PLC0415
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415

    resource = Resource.create(
        {
            "service.name": "nvrx-checkpoint-ci",
            "nemo.run.id": os.environ["MEGATRON_OTEL_RUN_ID"],
            "dl.rank": int(os.environ.get("RANK", "0")),
            "checkpoint.provider": provider,
        },
    )
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=60_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = meter_provider.get_meter("nvrx.checkpoint.ci")
    values = {
        "nvrx.checkpoint.run.gpu_peak_bytes": gpu_peak_bytes,
        "nvrx.checkpoint.run.gpu_hbm_peak_bytes": gpu_hbm_peak_bytes,
        "nvrx.checkpoint.run.gpu_total_bytes": gpu_total_bytes,
        "nvrx.checkpoint.run.cpu_rss_bytes": cpu_rss_bytes,
    }
    instruments = [meter.create_gauge(name, unit="By") for name in values]
    for instrument, value in zip(instruments, values.values()):
        instrument.set(value)
    meter_provider.shutdown()

def _install_selected_checkpoint_provider() -> ModuleType:
    """Install native hooks only for a native comparison run."""
    from nvrx_checkpointing.megatron import provider as megatron_provider  # noqa: PLC0415

    if megatron_provider.selected_provider() == megatron_provider.NATIVE_PROVIDER:
        from nvrx_checkpointing.megatron import integration as megatron_integration  # noqa: PLC0415

        megatron_integration.install()
    return megatron_provider


def _prepare_otel_randomness() -> None:
    if os.environ.get("NVRX_OTEL_USE_SYSTEM_RANDOM") != "1":
        return

    from nvrx_checkpointing.telemetry import isolate_opentelemetry_randomness  # noqa: PLC0415

    isolated = isolate_opentelemetry_randomness()
    if isolated and _rank_env("SLURM_PROCID", "RANK", "0") == "0":
        print(f"[slurm_job:otel] isolated_rng={','.join(isolated)}", flush=True)  # noqa: T201


def _fill_uninitialized_memory_if_deterministic(args: argparse.Namespace, torch_module: ModuleType) -> None:
    """Fill uninitialized allocations in deterministic runs.

    Checkpoints carry optimizer padding that is allocated but never written, so
    only a filled allocation keeps a deterministic run's checkpoint bytes
    reproducible. Other runs keep the setting Megatron chose.
    """
    if args.deterministic_mode:
        torch_module.utils.deterministic.fill_uninitialized_memory = True


def _pretrain_entry(phase: int, entrypoint: str) -> None:
    sys_path_insert(megatron_root())
    _prepare_otel_randomness()
    if _install_torch_serialization_compat() and _rank_env("SLURM_PROCID", "RANK", "0") == "0":
        print("[slurm_job] installed missing torch.utils.serialization config compatibility", file=sys.stderr, flush=True)  # noqa: T201

    # Reference runs execute without native integration patches.
    megatron_provider = _install_selected_checkpoint_provider()
    from megatron.core.enums import ModelType  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    from megatron.training import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        inprocess_restart,
        pretrain,
        set_startup_timestamps,
    )
    from megatron.training.argument_utils import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        pretrain_cfg_container_from_args,
        resolve_tokenizer_vocab_size,
    )
    from megatron.training.arguments import parse_and_validate_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    from megatron.training.global_vars import initialize_runtime_services  # noqa: PLC0415 # pyright: ignore[reportMissingImports]

    try:
        from megatron.post_training.arguments import add_modelopt_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    except ImportError:
        add_modelopt_args = None
    if entrypoint == "gpt":
        import pretrain_gpt as pretrain_module  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        from megatron.training.argument_utils import gpt_config_from_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    elif entrypoint == "hybrid":
        import pretrain_hybrid as pretrain_module  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        from hybrid_builders import hybrid_builder as model_builder  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        from megatron.training.argument_utils import hybrid_config_from_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        from model_provider import model_provider  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    else:
        choices = ", ".join(sorted(_PRETRAIN_ENTRYPOINTS))
        raise ValueError(f"unknown pretrain entrypoint {entrypoint!r}; choose {choices}")

    rank0 = os.environ.get("RANK", "0") == "0"
    if rank0:
        print(f"[native_ckpt:phase] phase={phase}", flush=True)  # noqa: T201
        _print_source_provenance()
    start = time.time()
    set_startup_timestamps(program_start=start, main_entry=start)
    pretrain_module.train_valid_test_datasets_provider.is_distributed = True
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(
        extra_args_provider=megatron_provider.compose_extra_args_provider(add_modelopt_args),
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    import torch  # noqa: PLC0415

    _fill_uninitialized_memory_if_deterministic(args, torch)
    model_cfg = hybrid_config_from_args(args) if entrypoint == "hybrid" else gpt_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(full_config, args.padded_vocab_size)

    hbm = _PeakHbmSampler(torch)
    hbm.start()
    try:
        pretrain_kwargs = {
            "train_valid_test_dataset_provider": pretrain_module.train_valid_test_datasets_provider,
            "model_type": ModelType.encoder_or_decoder,
            "forward_step_func": pretrain_module.forward_step,
            "store": store,
        }
        if entrypoint == "gpt":
            pretrain_kwargs["get_embedding_ranks"] = pretrain_module.get_embedding_ranks
        else:
            pretrain_kwargs["model_provider"] = partial(model_provider, model_builder)
        pretrain_fn(full_config, **pretrain_kwargs)
    finally:
        import resource  # noqa: PLC0415

        hbm.close()
        if os.environ.get("NATIVE_CKPT_OTEL_FILE"):
            _emit_training_process_memory(hbm.rss_samples)
        gpu_peak = torch.cuda.max_memory_allocated(hbm.device) if torch.cuda.is_initialized() else 0
        cpu_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if os.environ.get("MEGATRON_OTEL_ENABLED") == "1":
            _emit_checkpoint_resource_metrics(
                args.ckpt_provider,
                gpu_peak_bytes=gpu_peak,
                gpu_hbm_peak_bytes=hbm.peak_bytes,
                gpu_total_bytes=hbm.total_bytes,
                cpu_rss_bytes=cpu_rss,
            )
        else:
            print(  # noqa: T201
                f"[native_ckpt:mem] rank={os.environ.get('RANK', '0')} gpu_peak_bytes={gpu_peak} "
                f"gpu_hbm_peak_bytes={hbm.peak_bytes} gpu_total_bytes={hbm.total_bytes} "
                f"cpu_rss_bytes={cpu_rss}",
                flush=True,
            )

def _worker(args: argparse.Namespace) -> None:
    flavors = _selected_flavors(args)
    if any(
        _provider(spec.flavors) == "native" and spec.flavors.policy.requires_nvrx_absent
        for spec in _run_specs(args)
    ) and not _in_job_array():
        # The array's ft_launcher is itself an NVRx entry point; a mid-run
        # uninstall would pull its modules from under the launcher agents.
        uninstall_nvrx_for_native()
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    if _in_job_array():
        # A launcher restart re-runs this worker on nodes whose previous attempt
        # auto-launched the checkpoint daemon; the leftover belongs to this run,
        # so terminate it instead of refusing to start.
        CHECKPOINT_SERVER.terminate()
    else:
        CHECKPOINT_SERVER.require_stopped()
    try:
        _perform_runs(cfg, args)
    except BaseException:
        local_rank = int(_rank_env("SLURM_LOCALID", "LOCAL_RANK", "0"))
        if flavors.policy.manages_native_server and local_rank == 0:
            with contextlib.suppress(Exception):
                CHECKPOINT_SERVER.request("stop", 5)
        raise

def _identity(args: argparse.Namespace) -> tuple[str, str, str]:
    """The profile, the combination and the stem a run reports under."""
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    flavors = _selected_flavors(args)
    return cfg, flavors.tag, f"{platform_tag()}_{flavors.settled_tag}_{cfg}"


def _provider_baseline_stem(args: argparse.Namespace, report_spec: RunSpec) -> str:
    """Name a provider-specific baseline for a multi-routine report."""
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    names = dict.fromkeys((*_selected_flavors(args).names, *report_spec.flavors.names))
    return f"{platform_tag()}_{FlavorSelection(tuple(names)).settled_tag}_{cfg}"


def _performance_baseline_stem(args: argparse.Namespace) -> str:
    """Select the baseline for the provider represented by the job report."""
    stem = _identity(args)[2]
    flavors = _selected_flavors(args)
    if "detr" not in flavors.names:
        return stem
    specs = _run_specs(args)
    report_spec = next(
        (spec for spec in reversed(specs) if _provider(spec.flavors) == "native"),
        specs[-1],
    )
    return stem if _provider(report_spec.flavors) == "native" else _provider_baseline_stem(args, report_spec)


def _clean_run_lines(*, lead_only: bool) -> list[str]:
    """Shell lines that safely clear one run before any shared writes begin."""
    condition = '"${SLURMJOB_OPT_CLEAN_RUN:-0}" == 1'
    if lead_only:
        condition += ' && "${SLURM_ARRAY_TASK_ID}" == "0"'
    return [
        f"if [[ {condition} ]]; then",
        '  [[ "${SLURMJOB_OPT_RUN_DIR:?}" != / && "${SLURMJOB_OPT_RUN_DIR##*/}" == '
        '"${SLURMJOB_OPT_RUN_NAME:?}" ]] || exit 2',
        ('  find "${SLURMJOB_OPT_RUN_DIR}" -mindepth 1 -maxdepth 1 '
         '! -path "${SLURMJOB_OPT_RUN_DIR}/run.sbatch" '
         '! -path "${SLURMJOB_OPT_RUN_DIR}/job_id" '
         '! -path "${SLURMJOB_OPT_RUN_DIR}/gitlab_job_id" '
         '! -path "${SLURMJOB_OPT_RUN_DIR}/logs" -exec rm -rf -- {} +'),
        '  mkdir -p "${SLURMJOB_OPT_RUN_DIR}/logs"',
        '  find "${SLURMJOB_OPT_RUN_DIR}/logs" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +',
        "fi",
    ]


def _classic_job_body(
    *, prep: list[str], run: list[str], epilog: list[str], rank_logs: list[str],
    host_cleanup: list[str], log: Path,
) -> list[str]:
    """Build the ordinary single-allocation batch lifecycle."""
    return [
        "set -Eeuo pipefail",
        "allocation_epilog() {",
        "  local rc=$?",
        "  trap - EXIT",
        "  set +e",
        f"  {shell_join(host_cleanup)}",
        "  local cleanup_rc=$?",
        '  (( rc != 0 )) && exit "${rc}"',
        '  exit "${cleanup_rc}"',
        "}",
        "trap allocation_epilog EXIT",
        *_clean_run_lines(lead_only=False),
        f"exec > >({shell_join(['tee', str(log)])}) 2>&1",
        'printf "%s\\n" "${SLURM_JOB_ID:?}" > "${SLURMJOB_OPT_RUN_DIR:?}/job_id"',
        shell_join(prep),
        # Only rank 0 is streamed during a run. If srun fails, show the logs of
        # the ranks it says exited before propagating the original return code.
        f"{shell_join(run)} && run_rc=0 || run_rc=$?",
        "if (( run_rc != 0 )); then",
        f"  {shell_join(rank_logs)} || true",
        '  exit "${run_rc}"',
        "fi",
        shell_join(epilog),
    ]


def _job_array_body(
    args: argparse.Namespace,
    *,
    segment: int,
    work_minutes: int,
    common: dict[str, object],
    container_env: str,
    writable: bool,
    prep: list[str],
    worker_args: list[str],
    epilog: list[str],
    rank_logs: list[str],
    host_cleanup: list[str],
    log: Path,
) -> list[str]:
    """Build one task's shared-rendezvous array lifecycle.

    Task 0 owns preparation, the rendezvous store, final verification and the
    generation drain. Every task owns one NVLink domain; additional queued
    tasks provide replacement domains to ``ft_launcher``.
    """

    def launcher_step(*, lead: bool) -> list[str]:
        return build_srun_cmd(
            nodes=segment,
            ntasks=segment,
            time_minutes=work_minutes,
            total_gpus=segment * args.gpus_per_node,
            container_env=f"{container_env},RDZV_HOST",
            extra_srun_args=["--ntasks-per-node=1", *(["--container-writable"] if writable else [])],
            kill_on_bad_exit=not lead,
            **common,
        )

    launcher_options = " ".join([
        f"--nnodes {args.nodes}:{args.nodes + segment}",
        f"--nproc-per-node {args.gpus_per_node}",
        '--rdzv-id "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"',
        "--rdzv-backend c10d",
        '--rdzv-endpoint "${RDZV_HOST}:29400"',
        "--rdzv-conf join_timeout=12000,store_connect_wait_seconds=600",
        "--max-restarts 8",
        "--ft-restart-policy any-failed",
        f"--ft-segment {segment}",
        "--ft-rank-section-timeouts setup:1800,step:600,checkpointing:420",
        "--ft-rank-out-of-section-timeout 3600",
        '--ft-cycle-info-dir "${SLURMJOB_OPT_RUN_DIR}/nvrx/'
        '${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/cycle_infos"',
    ])
    # ft_launcher invokes the first positional argument with Python. Passing
    # ``python3`` would instead make it look for /workspace/python3.
    worker_command = shlex.join([str(Path(__file__).resolve()), *worker_args])
    launch_command = f"ft_launcher {launcher_options} {worker_command}"
    lead_launch = f"{shell_join(launcher_step(lead=True))} sh -c {shlex.quote(launch_command)}"
    peer_launch = f"{shell_join(launcher_step(lead=False))} sh -c {shlex.quote(launch_command)}"

    return [
        "set -Eeuo pipefail",
        'SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"',
        'NVRX_ARRAY_DIR="${SLURMJOB_OPT_RUN_DIR:?}/nvrx/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"',
        'RDZV_FILE="${NVRX_ARRAY_DIR}/control/rdzv_host"',
        'RDZV_CLOSED="${NVRX_ARRAY_DIR}/control/rdzv_closed"',
        'CLEAN_DONE="${NVRX_ARRAY_DIR}/control/clean_done"',
        'PREP_DONE="${NVRX_ARRAY_DIR}/control/prep_done"',
        "cancel_generation() {",
        '  echo "Cancelling peer tasks of array ${SLURM_ARRAY_JOB_ID}."',
        '  scancel "${SLURM_ARRAY_JOB_ID}_[1-${SLURM_ARRAY_TASK_MAX}]" 2>/dev/null || true',
        "}",
        "job_array_exit() {",
        "  local rc=$?",
        "  trap - EXIT",
        "  set +e",
        f"  {shell_join(host_cleanup)}",
        "  local cleanup_rc=$?",
        '  if (( cleanup_rc != 0 )); then echo "WARNING: host cleanup exited '
        '${cleanup_rc}; continuing array drain"; fi',
        '  if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        '    mkdir -p "${NVRX_ARRAY_DIR}/control"',
        '    echo "Task 0 exiting (rc=${rc}), creating RDZV_CLOSED file"',
        '    touch "${RDZV_CLOSED}"',
        "    if (( rc != 0 )); then",
        "      cancel_generation",
        "    else",
        '      scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true',
        "    fi",
        "  fi",
        '  exit "${rc}"',
        "}",
        "trap job_array_exit EXIT",
        *_clean_run_lines(lead_only=True),
        # Clean-run removal includes shared state and logs. Hold peer tasks here
        # until task 0 has finished, then let every task create its output.
        'if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        '  mkdir -p "${NVRX_ARRAY_DIR}/control"',
        '  touch "${CLEAN_DONE}"',
        "else",
        "  CLEAN_WAIT=0",
        '  while [[ ! -f "${CLEAN_DONE}" ]]; do',
        '    if [[ -f "${RDZV_CLOSED}" ]]; then',
        '      echo "RDZV_CLOSED while waiting for cleanup. Exiting task ${SLURM_ARRAY_TASK_ID}."',
        "      exit 1",
        "    fi",
        "    if (( CLEAN_WAIT >= 600 )); then",
        '      echo "cleanup not done after ${CLEAN_WAIT}s. Exiting task ${SLURM_ARRAY_TASK_ID}."',
        "      exit 1",
        "    fi",
        "    sleep 1",
        "    CLEAN_WAIT=$(( CLEAN_WAIT + 1 ))",
        "  done",
        "fi",
        # Task 0 is the streamed control log; peer tasks retain separate logs.
        'if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        f"  exec > >({shell_join(['tee', str(log)])}) 2>&1",
        "else",
        f"  exec >>{shell_join([str(log.with_suffix(''))])}.task${{SLURM_ARRAY_TASK_ID}}.log 2>&1",
        "fi",
        'printf "%s\\n" "${SLURM_ARRAY_JOB_ID}" > "${SLURMJOB_OPT_RUN_DIR:?}/job_id"',
        # Publish atomically so peers see either no endpoint or a complete one.
        'if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        '  RDZV_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | head -n1 || true)"',
        '  if [[ -z "${RDZV_HOST}" ]]; then RDZV_HOST="$(hostname)"; fi',
        '  if [[ -z "${RDZV_HOST}" ]]; then echo "FATAL: cannot determine rendezvous host" >&2; exit 1; fi',
        '  printf \'%s\\n\' "${RDZV_HOST}" > "${RDZV_FILE}.tmp" && mv -f "${RDZV_FILE}.tmp" "${RDZV_FILE}"',
        "fi",
        "RH_WAIT=0",
        "while true; do",
        '  if [[ -f "${RDZV_CLOSED}" ]]; then',
        '    echo "RDZV_CLOSED present on task ${SLURM_ARRAY_TASK_ID}: task 0 exited. Exiting."',
        "    exit 1",
        "  fi",
        '  if [[ -s "${RDZV_FILE}" ]]; then break; fi',
        "  if (( RH_WAIT >= 600 )); then",
        '    echo "rdzv_host not published after ${RH_WAIT}s. Exiting task ${SLURM_ARRAY_TASK_ID}."',
        "    exit 1",
        "  fi",
        "  sleep 1",
        "  RH_WAIT=$(( RH_WAIT + 1 ))",
        "done",
        'export RDZV_HOST="$(cat "${RDZV_FILE}")"',
        # Preparation writes shared dependencies, so peers wait for task 0.
        'if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        f"  {shell_join(prep)}",
        '  touch "${PREP_DONE}"',
        "else",
        "  PREP_WAIT=0",
        '  while [[ ! -f "${PREP_DONE}" ]]; do',
        '    if [[ -f "${RDZV_CLOSED}" ]]; then',
        '      echo "RDZV_CLOSED while waiting for prep. Exiting task ${SLURM_ARRAY_TASK_ID}."',
        "      exit 1",
        "    fi",
        "    if (( PREP_WAIT >= 3600 )); then",
        '      echo "prep not done after ${PREP_WAIT}s. Exiting task ${SLURM_ARRAY_TASK_ID}."',
        "      exit 1",
        "    fi",
        "    sleep 5",
        "    PREP_WAIT=$(( PREP_WAIT + 5 ))",
        "  done",
        "fi",
        'if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then',
        f"  {lead_launch} && run_rc=0 || run_rc=$?",
        "  if (( run_rc != 0 )); then",
        f"    {shell_join(rank_logs)} || true",
        "  fi",
        # Task 0 owns the result. Its launcher may lose a local agent while a
        # spare replaces that domain, so the checkpoint epilog is authoritative.
        f"  {shell_join(epilog)} && epilog_rc=0 || epilog_rc=$?",
        '  if (( epilog_rc != 0 )); then exit "${epilog_rc}"; fi',
        "else",
        f"  {peer_launch} && run_rc=0 || run_rc=$?",
        '  if (( run_rc != 0 )); then exit "${run_rc}"; fi',
        "fi",
    ]


def _build_job(args: argparse.Namespace) -> tuple[str, str, Path]:
    flavors = _selected_flavors(args)
    cfg, flavor, stem = _identity(args)
    log = RESULTS_DIR / f"{stem}.log"
    minutes = args.time_minutes or _time_budget(args.config, flavors, "srun", 30)
    partition = _job_partition(args.config, flavors)
    work_minutes = minutes - _CLEANUP_GRACE_MINUTES
    job_name = RUN_NAME or f"{flavor}_{cfg}"
    image, total_tasks = str(image_path()), args.nodes * args.gpus_per_node
    common = {
        "job_name": job_name,
        "image": image,
        "gpus_per_node": args.gpus_per_node,
        "partition": partition,
    }
    def step(nodes: int, tasks: int, env: str, output: Path | None = None, writable: bool = False) -> list[str]:
        return build_srun_cmd(
            nodes=nodes, ntasks=tasks, time_minutes=work_minutes,
            total_gpus=nodes * args.gpus_per_node, container_env=env, output=output,
            extra_srun_args=["--container-writable"] if writable else None, **common,
        )
    prep_log = RESULTS_DIR / f"{stem}_prep.log"
    spec_toolchains = [(spec, _toolchain(cfg, spec.args)) for spec in _run_specs(args)]
    toolchains = list(dict.fromkeys(toolchain for _spec, toolchain in spec_toolchains))
    megatron_sources = {
        (toolchain.megatron_root, toolchain.megatron_commit, toolchain.megatron_url): toolchain
        for toolchain in toolchains
    }
    prep_commands = [
        [
            "env",
            f"MEGATRON_COMMIT={toolchain.megatron_commit}",
            f"MEGATRON_REPO_URL={toolchain.megatron_url}",
            "bash",
            str(ENROOT_SCRIPTS_DIR / "megatron_root.sh"),
            str(toolchain.megatron_root),
        ]
        for toolchain in megatron_sources.values()
        if toolchain.megatron_commit != _WORKTREE_REVISION
    ]
    prep_commands.append([
        "python3", "-u", str(Path(wheel_install.__file__).resolve()),
        *flavors.policy.prep_args(),
    ])
    nvrx_sources = {
        toolchain.nvrx_target: toolchain
        for spec, toolchain in spec_toolchains
        if not (_provider(spec.flavors) == "native" and spec.flavors.policy.requires_nvrx_absent)
    }
    prep_commands.extend(
        [
            "python3", "-u", str(SCRIPT_DIR / "install_run_dependency.py"),
            "--repo-url", selected.nvrx_url,
            "--commit", selected.nvrx_commit,
            "--target", str(selected.nvrx_target),
            *(("--repo-root", str(nvrx_repo_root())) if nvrx_repo_root() else ()),
        ]
        for selected in nvrx_sources.values()
    )
    prep_commands.append(["python3", "-m", "pip", "list"])
    prep_commands.append(["python3", "-u", str(Path(wheel_install.__file__).resolve()), "--report-install"])
    prep = [*step(1, 1, ",".join(_env_names("p")), prep_log, writable=True),
            "/bin/bash", "-c", " && ".join(shell_join(command) for command in prep_commands)]
    common_worker_args = [
        "--nodes", str(args.nodes), "--gpus-per-node", str(args.gpus_per_node), "--config", args.config,
        "--flavor", "+".join(flavors.names),
        *args.extra_args,
    ]
    run_blocks = getattr(args, "run_blocks", None)
    if run_blocks is None:
        run_blocks = tuple(RunBlock(entry) for entry in (args.run or []))
    run_args = [
        token
        for block in run_blocks
        for token in ("--run", block.entry, *block.argv)
    ]
    worker_args = [*common_worker_args, *run_args]
    segment = _job_array_segment(args)
    job_array = segment > 0
    container_env = ",".join(_env_names("w"))
    native_perf_verification = any(
        _provider(spec.flavors) == "native" and spec.flavors.policy.requires_nvrx_absent
        for spec, _toolchain_ in spec_toolchains
    )
    run = [*step(args.nodes, total_tasks, container_env, writable=native_perf_verification),
           "python3", "-u", str(Path(__file__).resolve()), *worker_args]
    epilog = [
        "python3", "-u", str(Path(__file__).resolve()), _EPILOG_ENTRY_CMD, *common_worker_args,
        "--log-file", str(log),
        "--baseline-json", json.dumps(PerfReport.baseline(_performance_baseline_stem(args)), separators=(",", ":")),
        *run_args,
    ]
    rank_logs = [
        "python3", "-u", str(Path(__file__).resolve()), _RANK_LOGS_ENTRY_CMD, *common_worker_args,
        "--log-file", str(log),
        *run_args,
    ]
    cleanup_nodes = segment if job_array else args.nodes

    host_cleanup = [
        *build_srun_cmd(
            nodes=cleanup_nodes, ntasks=cleanup_nodes,
            total_gpus=cleanup_nodes * args.gpus_per_node, container_env="",
            time_minutes=_CLEANUP_GRACE_MINUTES, container=False,
            extra_srun_args=["--ntasks-per-node=1"], **common,
        ),
        "python3", "-u", str(SCRIPT_DIR / "daemon_tools.py"),
    ]
    if job_array:
        body = _job_array_body(
            args,
            segment=segment,
            work_minutes=work_minutes,
            common=common,
            container_env=container_env,
            writable=native_perf_verification,
            prep=prep,
            worker_args=worker_args,
            epilog=epilog,
            rank_logs=rank_logs,
            host_cleanup=host_cleanup,
            log=log,
        )
    else:
        body = _classic_job_body(
            prep=prep,
            run=run,
            epilog=epilog,
            rank_logs=rank_logs,
            host_cleanup=host_cleanup,
            log=log,
        )
    script, reaper = build_sbatch_script(
        nodes=args.nodes, ntasks=total_tasks, time_minutes=minutes, job_name=job_name, image=image,
        container_env="", gpus_per_node=args.gpus_per_node, total_gpus=total_tasks, body="\n".join(body),
        partition=partition, nodes_per_task=segment,
    )
    return script, reaper, log

def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True
    raise argparse.ArgumentTypeError("must be a boolean")
def _validated(value: str, convert: Callable, valid: Callable, error: str) -> object:
    converted = convert(value)
    if not valid(converted):
        raise argparse.ArgumentTypeError(error)
    return converted
def _path(value: str) -> Path:
    path = Path(value)
    if any(c in path.expanduser().resolve().as_posix() for c in ",:"):
        raise argparse.ArgumentTypeError("may not contain ',' or ':'")
    return path
_positive_int = partial(_validated, convert=int, valid=lambda value: value >= 1, error="must be at least 1")
_nonnegative_int = partial(_validated, convert=int, valid=lambda value: value >= 0, error="must be at least 0")
def _commit(value: str) -> str:
    if value == _WORKTREE_REVISION or _COMMIT_RE.fullmatch(value):
        return value
    raise argparse.ArgumentTypeError("must be HEAD or a 7-40 character hexadecimal commit")

_url = partial(_validated, convert=str, valid=lambda value: not re.match(r"https?://[^/@]+@", value),
               error="may not contain credentials because it is written to the reproducer")

RunOption = namedtuple("RunOption", "env default convert action choices scopes", defaults=(None, str, None, (), "r"))  # noqa: PYI024
_CONFIGS = tuple(sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml") if path.stem != "base"))
_RUN_OPTIONS = {
    # The hardware, the cluster and the account to charge. Each keeps its
    # environment name, so a shell that already exports it needs no argument.
    "platform": RunOption("PLATFORM", scopes="rspw"),
    "cluster": RunOption("CLUSTER", scopes="rspw"),
    "slurm_account": RunOption("SLURM_ACCOUNT", scopes="rsp"),
    "nodes": RunOption("SLURMJOB_OPT_NODES", 1, _positive_int),
    "config": RunOption("SLURMJOB_OPT_CONFIG", "pico", choices=_CONFIGS),
    "gpus_per_node": RunOption("GPUS_PER_NODE", None, _positive_int),
    # Variants every run carries, joined by '+' or given one option at a time.
    "flavor": RunOption("SLURMJOB_OPT_FLAVOR", convert=_flavor_names, action="extend"),
    # A run this job performs, as <name> or <name>:<variants>. Repeat for more
    # than one; without any, the job performs a single unnamed run.
    "run": RunOption("SLURMJOB_OPT_RUN", convert=_run_entry, action="extend", scopes="rw"),
    # Digest, or run directory holding one, every run is compared against. The
    # first run is the baseline when this names nothing.
    "baseline": RunOption("SLURMJOB_OPT_BASELINE", convert=_path, scopes="rw"),
    "name": RunOption("SLURMJOB_OPT_RUN_NAME", scopes="rspw"),
    "work_dir": RunOption("SLURMJOB_OPT_WORK_DIR", convert=_path, scopes="rspw"),
    "image": RunOption("ENROOT_IMAGE", convert=Path, scopes="rsw"),
    "megatron_commit": RunOption("MEGATRON_COMMIT", DEFAULT_MEGATRON_COMMIT, _commit, scopes="rspw"),
    "megatron_repo_url": RunOption("MEGATRON_REPO_URL", DEFAULT_MEGATRON_REPO_URL, _url, scopes="rspw"),
    "nvrx_commit": RunOption("NVRX_COMMIT", DEFAULT_NVRX_COMMIT, _commit, scopes="rsw"),
    "nvrx_repo_url": RunOption("NVRX_REPO_URL", DEFAULT_NVRX_REPO_URL, _url, scopes="rsw"),
    "checkpoints_per_run": RunOption("SLURMJOB_OPT_CHECKPOINTS_PER_RUN", convert=_nonnegative_int, scopes="rw"),
    "checkpoint_interval": RunOption("SLURMJOB_OPT_CHECKPOINT_INTERVAL", convert=_positive_int, scopes="rw"),
    "repeat": RunOption("SLURMJOB_OPT_REPEAT", 1, _nonnegative_int, scopes="rw"),
    "time_minutes": RunOption("SLURMJOB_OPT_TIME_MINUTES", convert=_positive_int, scopes="rw"),
    "clean_run": RunOption("SLURMJOB_OPT_CLEAN_RUN", False, _parse_bool, argparse.BooleanOptionalAction),
    "background": RunOption("SLURMJOB_OPT_BACKGROUND", False, _parse_bool, argparse.BooleanOptionalAction),
    # Shared-rendezvous job array segment size in nodes; 0 keeps the classic
    # single-allocation launch. Unset: jobs wider than one 16-node NVLink
    # domain run as arrays of 16-node tasks.
    "job_array_segment": RunOption("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", None, _nonnegative_int, None, (), "rw"),
    "dry_run": RunOption("SLURMJOB_OPT_DRY_RUN", False, _parse_bool, argparse.BooleanOptionalAction),
}
_ENV_SCOPES = {
    **dict.fromkeys(_TRAIN_ENV_DEFAULTS, "rw"),
    "NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS": "rw",
    "VCPKG_ROOT": "rspw", "MEGATRON_NATIVE_BUILD_DIR": "rspw", "NVRX_REPO_ROOT": "rspw",
    "SLURMJOB_OPT_PYTHON_INSTALL_DIR": "rspw", "ENROOT_IMAGE_DIR": "rs", "HOME": "rs", "MEGATRON_ROOT": "rspw",
    "SLURMJOB_OPT_LAUNCHER_ROOT": "rs", "SLURMJOB_OPT_RUN_DIR": "rsw", "SLURMJOB_OPT_WORK_ROOT": "rspw",
    "PYTHONPATH": "rpw",
    # Set by the GB300 runner for the Nano4 production recipe. Keep it scoped
    # to the training worker instead of changing every profile's defaults.
    "NVTE_FUSED_ATTN": "rw",
    "CI_API_V4_URL": "rp", "CI_COMMIT_SHA": "rp", "CI_PROJECT_ID": "rp", "WHEEL_ARCH": "rp", "CI_JOB_TOKEN": "p",
    "CI_DIR": "r", "DOCKER_IMAGE_BASE": "r", "LD_LIBRARY_PATH": "r", "PATH": "r",
    "SLURM_SEGMENT": "r", "TRITON_CACHE_DIR": "rw", "TRITON_CACHE_PATH": "rw",
    "TORCHINDUCTOR_CACHE_DIR": "rw", "TORCH_EXTENSIONS_DIR": "rw", "NATIVE_CKPT_SERVER_LOG": "rw",
    "NATIVE_CKPT_OTEL_COLLECTOR": "rw", "NATIVE_CKPT_OTEL_FILE": "rw",
    "MEGATRON_OTEL_RUN_ID": "rw",
    **dict.fromkeys(_MEGATRON_OTEL_ENV_DEFAULTS, "rw"),
}
def _env_names(scope: str) -> list[str]:
    return [option.env for option in _RUN_OPTIONS.values() if scope in option.scopes] + [
        name for name, scopes in _ENV_SCOPES.items() if scope in scopes
    ]
_SHELL_ENV_REFS = tuple(_env_names("s"))
_RUN_ENV_NAMES = set(_env_names("r"))

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # One training process checkpoints indefinitely within a four-hour allocation.
  slurm_job.py pretrain-native --config nano3 --nodes 4 --work-dir /lustre/runs \\
    --clean-run --checkpoints-per-run 0 --time-minutes 240 --background

  # Restart after each default three-checkpoint cycle on the same allocation.
  slurm_job.py pretrain-native --config nano3 --nodes 4 --work-dir /lustre/runs \\
    --clean-run --repeat 0 --time-minutes 240 --background
""",
    )
    for dest, option in _RUN_OPTIONS.items():
        kwargs: dict[str, object] = {"dest": dest, "default": argparse.SUPPRESS}
        kwargs["action" if option.action else "type"] = option.action or option.convert
        if option.action == "extend":
            kwargs["type"] = option.convert
        if option.choices:
            kwargs["choices"] = option.choices
        parser.add_argument("--" + dest.replace("_", "-"), **kwargs)
    return parser

def _split_run_blocks(argv: list[str]) -> tuple[list[str], tuple[RunBlock, ...]]:
    """Splits arguments into common options and ordered run blocks."""
    common: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    active: list[str] | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--run" or argument.startswith("--run="):
            if argument == "--run":
                index += 1
                if index == len(argv):
                    raise argparse.ArgumentTypeError("--run requires a name")
                entry = argv[index]
            else:
                entry = argument.partition("=")[2]
            parsed = _run_entry(entry)
            if len(parsed) != 1:
                raise argparse.ArgumentTypeError("each scoped --run must name exactly one run")
            active = []
            blocks.append((parsed[0], active))
        else:
            (active if active is not None else common).append(argument)
        index += 1
    return common, tuple(RunBlock(entry, tuple(tokens)) for entry, tokens in blocks)

def _parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        common, cli_blocks = _split_run_blocks(raw)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    args, extra = parser.parse_known_args(common)
    for dest, option in _RUN_OPTIONS.items():
        if hasattr(args, dest):
            continue
        raw = os.environ.get(option.env)
        multi = option.action in ("append", "extend")
        values = raw.split(",") if raw and option.action == "append" else [raw]
        try:
            converted = [option.convert(value.strip()) for value in values if value and value.strip()]
        except (argparse.ArgumentTypeError, TypeError, ValueError) as error:
            parser.error(f"{option.env} has invalid value: {error}")
        if option.action == "extend":
            converted = [item for group in converted for item in group]
        value = converted if multi and converted else converted[0] if converted else option.default
        candidates = (value or []) if multi else ([] if value is None else [value])
        if option.choices and any(item not in option.choices for item in candidates):
            parser.error(f"{option.env} must be one of: {', '.join(option.choices)}")
        setattr(args, dest, value)
    if cli_blocks:
        args.run = [block.entry for block in cli_blocks]
        args.run_blocks = cli_blocks
    else:
        args.run_blocks = tuple(RunBlock(entry) for entry in (args.run or []))
    return args, extra

def _rank_logs_entry(argv: list[str]) -> None:
    """Preserve available arguments and show logs of ranks named by srun."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    parser.add_argument("--log-file", type=Path, required=True)
    args, args.extra_args = _parse_args(parser, argv)
    args.flavors = FlavorSelection(args.flavor)
    _selected_flavors(args)
    _report_failing_rank_logs(args.log_file)
    # The regular epilog is not reached after srun fails. Retain and compare
    # every argument block that was completed before the failure so failed jobs
    # provide the same diagnostic artifacts as successful jobs.
    _write_and_compare_arguments(args, args.log_file, strict=False)


def _failure_summary_entry(argv: list[str]) -> None:
    """Summarize logs owned by the current GitLab job after its script fails."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 1:
        raise SystemExit(f"usage: {_FAILURE_SUMMARY_ENTRY_CMD} RUNS_ROOT")
    root = Path(argv[0])
    if not root.is_dir():
        return
    job_id = os.environ.get("CI_JOB_ID", "").strip()
    runs = sorted(root.glob(f"{job_id}_*")) if job_id else sorted(root.iterdir())
    for run in runs:
        for log in sorted((run / "logs").glob("*.log")):
            _report_failure_summary(log)


def _recover_otel_entry(argv: list[str]) -> None:
    """Create missing memory graphs from telemetry left by an interrupted run."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 1:
        raise SystemExit(f"usage: {_RECOVER_OTEL_ENTRY_CMD} RUNS_ROOT")
    root = Path(argv[0])
    if not root.is_dir():
        return
    job_id = os.environ.get("CI_JOB_ID", "").strip()
    runs = sorted(root.glob(f"{job_id}_*")) if job_id else sorted(root.iterdir())
    for run in runs:
        for source in sorted(run.glob("**/logs/otel/*/native_ckpt_otel.jsonl")):
            output = source.parent / "process_memory_all.svg"
            if output.is_file() or not source.stat().st_size:
                continue
            try:
                data = otel_report.OtelData([source], tolerate_truncated_tail=True)
                outputs = otel_report.write_process_memory_charts(data, source.parent)
            except (OSError, ValueError, KeyError, TypeError) as error:
                logger.warning(f"[slurm_job:otel] WARNING could not recover {source}: {error}")
                continue
            if outputs:
                logger.info(f"[slurm_job:otel] recovered partial memory graph: {outputs[0]}")


def _epilog(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--baseline-json", required=True)
    args, args.extra_args = _parse_args(parser, argv)
    args.flavors = FlavorSelection(args.flavor)
    _selected_flavors(args)
    args.baseline = json.loads(args.baseline_json)
    otel_reporting = _otel_reporting_enabled(args.flavors)
    log_file = args.log_file
    if _in_job_array():
        # The launcher's rank 0 lands on whichever array task joins the rendezvous
        # first, so the training stream the report reads lives in that task's log,
        # not necessarily in the one task 0 tees. The combined view is the job's log.
        task_logs = sorted(log_file.parent.glob(f"{log_file.stem}.task*.log"))
        combined = log_file.with_suffix(".combined.log")
        combined.write_text(log_file.read_text(errors="replace") + "".join(p.read_text(errors="replace") for p in task_logs))
        log_file = combined
    if not otel_reporting:
        specs = _run_specs(args)
        expected_phase2 = sum(
            (spec.flavors.policy.phase2_checkpoints if spec.args.checkpoints_per_run is None else spec.args.checkpoints_per_run)
            * (spec.args.repeat or 1)
            for spec in specs
        )
        log_report.validate_checkpoint_saves(log_file, len(specs), expected_phase2)
    passed = args.flavors.policy.epilog(args, log_file)
    _write_performance_baseline(args, log_file.parent)
    # The arguments a run trained on are part of what its result means, so they
    # are compared with the same standing as the numbers it produced.
    passed = _write_and_compare_arguments(args, log_file) and passed
    if otel_reporting:
        _log_otel_checkpoint_timings(args, log_file)
    if not passed:
        raise SystemExit(SOFT_FAIL_RC)

# Options that named a second toolchain for one run to hold. Unrecognized
# options reach Megatron, so naming one here fails on the launcher rather than
# in phase 1 with an allocation already held.
_RETIRED_OPTIONS = {
    f"--{dependency}-{field}-compare" for dependency in ("megatron", "nvrx") for field in ("commit", "repo-url")
}

def _validate_comparison_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject comparison settings that cannot hold, before the allocation."""
    if retired := sorted(_RETIRED_OPTIONS.intersection(args.extra_args)):
        parser.error(
            f"{', '.join(retired)}: a run trains on one toolchain. Run it once per toolchain and compare the "
            "digests with --diff-digests.",
        )
    if args.baseline and not args.baseline.exists():
        parser.error(f"--baseline {args.baseline} does not exist")

def _checkout_for_head(parser: argparse.ArgumentParser, option: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if not value:
        parser.error(f"{option}=HEAD requires {env_name}")
    root = Path(value).expanduser().resolve()
    if not (root / ".git").exists():
        parser.error(f"{option}=HEAD requires {env_name} to be a Git checkout: {root}")

def _export_settings(args: argparse.Namespace) -> None:
    """Publish the settings the rest of the run reads from the environment.

    The hardware, the cluster and the account are read wherever they are
    needed rather than threaded through, so an argument for one has to reach
    the environment before anything asks.
    """
    for dest in ("platform", "cluster", "slurm_account"):
        if value := getattr(args, dest, None):
            os.environ[_RUN_OPTIONS[dest].env] = str(value)

def _validate_run_spec(
    parser: argparse.ArgumentParser,
    spec: RunSpec,
    common_modes: set[str],
) -> None:
    run_args = spec.args
    run_modes = {flavor.name for flavor in spec.flavors.items if flavor.execution_mode}
    if run_modes != common_modes:
        parser.error(f"run {spec.name or 'run'}: execution flavors must be common options before the first --run")
    if run_args.checkpoints_per_run == 0 and run_args.repeat != 1:
        parser.error(f"run {spec.name or 'run'}: an unlimited checkpoint run cannot be repeated")
    if spec.flavors.policy.requires_finite_run and (run_args.checkpoints_per_run == 0 or run_args.repeat == 0):
        parser.error(f"run {spec.name or 'run'}: the selected flavor requires finite checkpoint and repeat counts")
    if run_args.nvrx_commit == _WORKTREE_REVISION:
        _checkout_for_head(parser, "--nvrx-commit", "NVRX_REPO_ROOT")
    if run_args.megatron_commit == _WORKTREE_REVISION:
        _checkout_for_head(parser, "--megatron-commit", "MEGATRON_ROOT")
    _validate_comparison_options(parser, run_args)

def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.gpus_per_node is None:
        args.gpus_per_node = platform_gpus_per_node()
    if args.gpus_per_node is None:
        parser.error(
            "--gpus-per-node or GPUS_PER_NODE is required; PLATFORM names it for "
            f"{', '.join(sorted(e['name'] for e in json.loads(JOBS_FILE.read_text())['platforms']))}",
        )
    if args.time_minutes is not None and args.time_minutes <= _CLEANUP_GRACE_MINUTES:
        parser.error(f"--time-minutes must exceed the {_CLEANUP_GRACE_MINUTES}-minute cleanup reserve")
    segment = _job_array_segment(args)
    if segment and (segment > args.nodes or args.nodes % segment):
        parser.error(
            f"--job-array allocates {segment}-node NVLink domains per array task; "
            "--nodes must be a positive multiple of --job-array-segment",
        )
    _validate_comparison_options(parser, args)
    try:
        specs = _run_specs(args)
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))
    common_modes = {flavor.name for flavor in args.flavors.items if flavor.execution_mode}
    for spec in specs:
        _validate_run_spec(parser, spec, common_modes)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    args, args.extra_args = _parse_args(parser)
    try:
        args.flavors = FlavorSelection(args.flavor)
    except ValueError as error:
        parser.error(str(error))
    _selected_flavors(args)
    _export_settings(args)
    _validate_args(parser, args)
    _configure_run(args)
    ensure_vcpkg_root()
    if in_srun():
        _worker(args)
        return
    if not args.dry_run and args.flavors.policy.install_prebuilt:
        wheel_install.install_prebuilt_wheel(probe=False)
    script, reaper, log = _build_job(args)
    rc = _sbatch_run_with_retry(
        script, log,
        extra_sbatch_args=[reaper],
        background=args.background,
        dry_run=args.dry_run,
        job_array=_job_array_segment(args) > 0,
    )
    if not args.background and not args.dry_run:
        _drop_sync_files()
    if rc == SOFT_FAIL_RC:
        raise SystemExit(rc)
def _drop_sync_files() -> None:
    """Remove what the ranks coordinated through, now that they have exited.

    A backgrounded job is still running when the launcher returns, so its
    handshakes are left for the next run of the same job to clear.
    """
    for path in sorted(RUN_ROOT.glob(f"**/{SYNC_DIR_NAME}")):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)


def _digest_value(value: str, limit: int = 160) -> str:
    """Keep one digest value useful without letting metadata blobs flood the trace."""
    shortened = value if len(value) <= limit else f"{value[:limit - 3]}..."
    return repr(shortened)


def _entry_difference_details(
    produced: dict[str, str],
    recorded: dict[str, str],
    limit: int = _DIGEST_DIFF_DETAIL_LIMIT,
) -> list[str]:
    """Describe missing, extra and changed entries with the values that explain them."""
    details = []
    for key in sorted(recorded.keys() - produced.keys()):
        details.append(f"missing {key}: baseline={_digest_value(recorded[key])}")
        if len(details) == limit:
            return details
    for key in sorted(produced.keys() - recorded.keys()):
        details.append(f"extra {key}: candidate={_digest_value(produced[key])}")
        if len(details) == limit:
            return details
    for key in sorted(produced.keys() & recorded.keys()):
        if produced[key] != recorded[key]:
            details.append(
                f"changed {key}: baseline={_digest_value(recorded[key])} "
                f"candidate={_digest_value(produced[key])}",
            )
            if len(details) == limit:
                return details
    return details


def _digest_differences(left: Path, right: Path) -> int:
    """Count how the checkpoints two digests describe differ, reporting each."""
    sides = [_digest_entries(path) for path in (left, right)]
    differences = 0
    details_shown = 0
    for label in sorted(sides[0].keys() | sides[1].keys()):
        produced, recorded = sides[1].get(label), sides[0].get(label)
        if produced is None or recorded is None:
            logger.info(f"[slurm_job:diff] {label}: only in {(left if recorded else right).name}")
            differences += 1
            continue
        counts = _compare_entries(produced, recorded)
        logger.info(f"[slurm_job:diff] {label}: entries={len(produced)} {counts}")
        checkpoint_differences = sum(
            int(field.split("=")[1])
            for field in counts.split()
            if field.startswith(("missing", "extra", "changed"))
        )
        differences += checkpoint_differences
        if checkpoint_differences and details_shown < _DIGEST_DIFF_DETAIL_LIMIT:
            remaining = _DIGEST_DIFF_DETAIL_LIMIT - details_shown
            for detail in _entry_difference_details(produced, recorded, remaining):
                logger.info(f"[slurm_job:diff] {label}: {detail}")
                details_shown += 1
    if differences > details_shown:
        logger.info(
            f"[slurm_job:diff] {differences - details_shown} additional entry difference(s) omitted; "
            f"inspect {left.name} and {right.name} in the job artifacts",
        )
    return differences

def _diff_digests(left: Path, right: Path) -> None:
    """Report how two runs' checkpoints differ, from their digests alone.

    Each digest names the run that wrote it, so a comparison needs neither run
    to still exist and neither to have been configured against the other.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    differences = _digest_differences(left, right)
    logger.info(f"[slurm_job:diff] {'PASS: 0 differences' if not differences else f'FAIL: {differences} differences'}")
    raise SystemExit(1 if differences else 0)

def _digest_entries(path: Path) -> dict[str, dict[str, str]]:
    """Read one digest back into per-checkpoint entries."""
    opener = lzma.open if path.suffix == ".xz" else open
    entries: dict[str, dict[str, str]] = {}
    with opener(path, "rt", errors="replace") as handle:  # type: ignore[operator]
        for line in handle:
            label, _, entry = line.rstrip("\n").partition(" ")
            key, _, value = entry.partition("=")
            # Filter here as well as while generating a digest so comparisons
            # against already-recorded runs get the same normalization.
            if key and key not in _RUN_SCOPED_KEYS:
                entries.setdefault(label, {})[key] = value
    return entries

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--diff-digests":
        _diff_digests(Path(sys.argv[2]), Path(sys.argv[3]))
    if len(sys.argv) > 1 and sys.argv[1] == _PRETRAIN_ENTRY_CMD:
        phase = int(sys.argv[2])
        entrypoint = sys.argv[3]
        sys.argv = [f"pretrain_{entrypoint}.py", *sys.argv[4:]]
        _pretrain_entry(phase, entrypoint)
    elif len(sys.argv) > 1 and sys.argv[1] == _EPILOG_ENTRY_CMD:
        _epilog(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == _RANK_LOGS_ENTRY_CMD:
        _rank_logs_entry(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == _FAILURE_SUMMARY_ENTRY_CMD:
        _failure_summary_entry(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == _RECOVER_OTEL_ENTRY_CMD:
        _recover_otel_entry(sys.argv[2:])
    else:
        main()
