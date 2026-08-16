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
import statistics
import string
import subprocess
import sys
import threading
import time
from collections import namedtuple
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.07-py3")
DEFAULT_MEGATRON_REPO_URL = "https://github.com/NVIDIA/Megatron-LM.git"
DEFAULT_MEGATRON_COMMIT = "77c87725e71c9f5022ac289071b16941876ca858"
DEFAULT_NVRX_REPO_URL = "https://github.com/NVIDIA/nvidia-resiliency-ext.git"
_NVRX_DISTRIBUTION = "nvidia-resiliency-ext"
_NVRX_MODULE = "nvidia_resiliency_ext"
SOFT_FAIL_RC = 3
PERF_TOLERANCE = 0.05
MAX_BW_SOCKET_IMBALANCE = 1.30
# Durability spans the whole distributed write, so it carries the shared filesystem's
# variance rather than the checkpointer's: on a busy Lustre the same job can differ
# several-fold. Track the number, but only fail on a change too large for that to
# explain. A baseline's own `tolerances` entry still wins over this.
WIDE_TOLERANCE_METRICS = {
    "ckpt_durable_ms": 2.0,
    "ckpt_daemon_durable_ms": 2.0,
}
RESULTS_DIR = REPO_ROOT / "ci/ci-results"
RUN_ROOT: Path | None = None
# Parent of every run directory. Holds what runs share: the dependency
# checkouts, the native build and the compiler caches, none of which depend on
# what a run is testing.
WORK_ROOT: Path | None = None
RUN_NAME: str | None = None
logger = logging.getLogger("native_ci")
_PRETRAIN_ENTRY_CMD = "__pretrain_entry__"
_EPILOG_ENTRY_CMD = "__epilog__"
_RANK_LOGS_ENTRY_CMD = "__rank_logs__"
# GitLab traces are size limited. Rank 0 is already mirrored in full while the
# run is active, so retain the failure and traceback tail for the ranks that
# Slurm reports exiting.
_RANK_LOG_TAIL_LINES = 120
_RANK_LOG_LIMIT = 4
_SLURM_STATUS_PERIOD_SECONDS = 5 * 60
_PHASE1_TRAIN_ITERS = 100
_LONG_RUN_TRAIN_SAMPLES = 1_000_000_000
_CLEANUP_GRACE_MINUTES = 5
# Ranks coordinate through files under this directory of a run, never beside
# its logs.
SYNC_DIR_NAME = ".sync"
_TORCH_STORAGE_ALIGNMENT = 64
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
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
}
_PRETRAIN_ENV_NAMES = {
    "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH",
    "CUBLAS_WORKSPACE_CONFIG",
    "ENROOT_IMAGE",
    "GPUS_PER_NODE",
    "LOCAL_RANK",
    "MAMBA_DETERMINISTIC",
    "MASTER_ADDR",
    "NATIVE_CKPT_SERVER_LOG",
    "NCCL_ALGO",
    "NCCL_NVLS_ENABLE",
    "NCCL_PROTO",
    "NCCL_TUNER_PLUGIN",
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "NVTE_CPU_OFFLOAD_V1",
    "RANK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_PROCID",
    "SLURM_STEP_NODELIST",
    "SLURM_TASKS_PER_NODE",
    "WORLD_SIZE",
}
# The daemon emits the perf counters, one line per rank per save, into its own
# per-host log rather than the trainer's output.
_BW_RE = re.compile(
    r"\[native_ckpt:svr\]\[[^\]]*\]\[PERF\] Rank=(\d+) Finished "
    r"bw_mbps=([0-9.]+)[^\n]*?wr_tnsr_us=([0-9]+)[^\n]*?total_written_bytes=([0-9]+)",
)
_MEM_RE = re.compile(
    r"\[native_ckpt:mem\](?: rank=(\d+))? gpu_peak_bytes=(\d+)"
    r"(?: gpu_hbm_peak_bytes=(\d+) gpu_total_bytes=(\d+))? cpu_rss_bytes=(\d+)",
)
_PHASE2_MARKER = "[native_ckpt:phase] phase=2"
_SAVE_RE = re.compile(r"successfully saved (?:async )?checkpoint from iteration\s+\d+\s+to\s+(\S+)")
# The coordinator daemon reports a checkpoint durable once every shard and the metadata
# are on disk. Native-only: the nvrx provider never starts this daemon.
_DAEMON_DURABLE_RE = re.compile(
    r"\[native_ckpt:svr\]\[[^\]]*\]\[PERF\] Snapshot=(\d+) Durable[^\n]*?activate_to_durable_us=(\d+)",
)
# Megatron prints this pair around every save, and the native writer mirrors it
# byte-for-byte, so the gap measures durability the same way whichever provider ran.
# Matched as one alternation so the two events stay in the order the run emitted them.
_CKPT_EVENT_RE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] "
    r"(saving checkpoint at iteration|successfully saved (?:\S+ )?checkpoint from iteration)"
    r"\s+(\d+)",
)
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

def cluster_tag() -> str:
    return os.environ.get("CLUSTER", "").strip()
def _run_dir(cfg: str) -> Path:
    return RUN_ROOT if RUN_ROOT is not None else REPO_ROOT / "ci/ci-runs" / cfg
def nvrx_repo_root() -> Path | None:
    value = os.environ.get("NVRX_REPO_ROOT")
    return Path(value).expanduser().resolve() if value else None
def _nvrx_target(cfg: str, commit: str | None) -> Path | None:
    if not commit and nvrx_repo_root() is None:
        return None
    suffix = f"-{commit}" if commit else ""
    return _run_dir(cfg) / f"deps/nvidia-resiliency-ext{suffix}"

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

    def describe(self, label: str) -> str:
        nvrx = f"{self.nvrx_url}@{self.nvrx_commit}" if self.nvrx_commit else "image"
        return f"{label}: megatron={self.megatron_url}@{self.megatron_commit or 'HEAD'} nvrx={nvrx}"

    def apply(self, env: dict[str, str]) -> None:
        """Point one training subprocess at this toolchain."""
        env["MEGATRON_ROOT"] = self.megatron_root.as_posix()
        env["MEGATRON_REPO_URL"] = self.megatron_url
        if self.megatron_commit:
            env["MEGATRON_COMMIT"] = self.megatron_commit
        env["NVRX_REPO_URL"] = self.nvrx_url
        if self.nvrx_commit:
            env["NVRX_COMMIT"] = self.nvrx_commit
        entries = [entry for entry in env.get("PYTHONPATH", "").split(":") if entry]
        if self.nvrx_target is not None:
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
    return Toolchain(
        args.megatron_repo_url or DEFAULT_MEGATRON_REPO_URL,
        args.megatron_commit,
        Path(os.environ.get("MEGATRON_ROOT") or _megatron_checkout(args)).expanduser(),
        args.nvrx_repo_url or DEFAULT_NVRX_REPO_URL,
        args.nvrx_commit,
        _nvrx_target(cfg, args.nvrx_commit),
    )
def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}
def _repo_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip()[:8]

def _megatron_checkout(args: argparse.Namespace) -> Path:
    """Where this run's Megatron-LM lives, keyed so runs of one commit share it."""
    root = WORK_ROOT if WORK_ROOT is not None else REPO_ROOT / "workdir"
    return root / f"deps/megatron-lm-{(args.megatron_commit or 'head')[:8]}"

def _source_suffix(args: argparse.Namespace) -> str:
    """Name only the sources this run was told to deviate on.

    A run on the pinned Megatron-LM and the extension in the image is the
    common case and says nothing; a run given either explicitly carries it, so
    two runs of different sources never share a directory.
    """
    parts = []
    if args.megatron_commit and args.megatron_commit != DEFAULT_MEGATRON_COMMIT:
        parts.append(f"-mlm-{args.megatron_commit[:8]}")
    if args.nvrx_commit:
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
    global RESULTS_DIR, RUN_NAME, RUN_ROOT, WORK_ROOT  # noqa: PLW0603
    if value := os.environ.get("NVRX_REPO_ROOT"):
        resolved = Path(value).expanduser().resolve()
        if any(c in resolved.as_posix() for c in ",:"):
            raise SystemExit("NVRX_REPO_ROOT may not contain ',' or ':'")
        os.environ["NVRX_REPO_ROOT"] = resolved.as_posix()
    os.environ["SLURMJOB_OPT_LAUNCHER_ROOT"] = REPO_ROOT.as_posix()
    name = _run_name(args.name, args)
    root = (args.work_dir or REPO_ROOT / "workdir").expanduser().resolve()
    WORK_ROOT = root
    RUN_NAME, RUN_ROOT, RESULTS_DIR = name, root / name, root / name / "logs"
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

def uninstall_nvrx_for_native() -> None:
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
    partition: str | None = None,
) -> list[str]:
    values = _slurm_values(nodes, ntasks, time_minutes, job_name, image, container_env, gpus_per_node, total_gpus)
    output_args = [f"--output={output}", f"--open-mode={'append' if append_output else 'truncate'}"] if output else []
    tokens = _override_command_option(_srun_tokens(values), "--partition", partition)
    if not container:
        tokens = [token for token in tokens if not token.startswith("--container-")]
    return [*tokens, "--kill-on-bad-exit=1", _reaper(time_minutes), "--label", *output_args, *(extra_srun_args or [])]

def build_sbatch_script(
    *, nodes: int, ntasks: int, time_minutes: int, job_name: str, image: str,
    container_env: str, gpus_per_node: int, total_gpus: int, body: str,
    partition: str | None = None,
) -> tuple[str, str]:
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
    if segment := os.environ.get("SLURM_SEGMENT", "16" if nodes >= 16 else "").strip():
        flags.append(f"--segment={segment}")
    flags.append("--exclusive")
    reaper_flag, reaper_json = _reaper(time_minutes).split("=", 1)
    reaper = f"{reaper_flag}={shlex.quote(reaper_json)}"
    return "#!/bin/bash\n" + "\n".join(f"#SBATCH {flag}" for flag in flags) + f"\n\n{body.rstrip()}\n", reaper

def _materialize(script: str, path: Path, args: list[str], env: dict[str, str] | None) -> None:
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
    directives = "\n".join([*(f"#SBATCH {arg}" for arg in args), f"#SBATCH --output={os.devnull}"])
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

def _report_queue_status(job_id: str) -> None:
    """Print a non-fatal scheduler heartbeat for a submitted allocation."""
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
        return
    rows = proc.stdout.strip().splitlines()
    if not rows:
        logger.info(f"[slurm_job:status] {timestamp} job {job_id} is no longer listed by squeue")
        return
    logger.info("[slurm_job:status] JobID|State|Elapsed|TimeLimit|StartTime|Nodes/Reason")
    for row in rows:
        logger.info(f"[slurm_job:status] {timestamp}|{row}")

def _wait_for_submission(
    submit: subprocess.Popen,
    job_id: str,
    period_seconds: int = _SLURM_STATUS_PERIOD_SECONDS,
) -> int:
    """Wait for sbatch while reporting scheduler state at a bounded cadence."""
    while True:
        try:
            return submit.wait(timeout=period_seconds)
        except subprocess.TimeoutExpired:
            _report_queue_status(job_id)

def _report_allocation_accounting(job_id: str) -> None:
    """Print allocation and step accounting without changing job outcome."""
    command = [
        "sacct", "-n", "-P", "--local", "-j", job_id,
        "-o", "JobIDRaw,JobName%30,State,Elapsed,Timelimit,ExitCode,DerivedExitCode,NodeList%40,MaxRSS",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S607
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

def _verify_allocation(job_id: str, rc: int) -> None:
    command = ["sacct", "-X", "-n", "-P", "--local", "-j", job_id, "-o", "JobIDRaw,State,ExitCode,DerivedExitCode"]
    for _attempt in range(30):
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"sacct failed for Slurm job {job_id}: {proc.stderr.strip()}")
        row = next((line.split("|") for line in proc.stdout.splitlines() if line.split("|", 1)[0] == job_id), None)
        state = row[1].split()[0].rstrip("+") if row and len(row) == 4 else ""
        if state and state not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"sacct did not report Slurm allocation {job_id}")
    _report_allocation_accounting(job_id)
    codes = row[2:]
    clean = all(code == "0:0" for code in codes)
    soft = rc == SOFT_FAIL_RC and state == "FAILED" and all(code.endswith(":0") for code in codes) and any(
        code.startswith(f"{SOFT_FAIL_RC}:") for code in codes
    )
    if not ((rc == 0 and state == "COMPLETED" and clean) or soft):
        raise RuntimeError(
            f"Slurm allocation {job_id} was not successful: state={row[1]} "
            f"exit={row[2]} derived_exit={row[3]} sbatch_rc={rc}",
        )

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
) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = RUN_ROOT / "run.sbatch" if RUN_ROOT else stream_log.with_suffix(".sbatch")
    stream_log.parent.mkdir(parents=True, exist_ok=True)
    stream_log.write_text("")
    _materialize(script, script_path, list(extra_sbatch_args or []), env)
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
    _verify_allocation(job_id, rc)
    return rc

def _transient_nccl_proxy_failure(log: Path) -> bool:
    """Recognize the recurrent NCCL InfiniBand proxy crash, and only that crash."""
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
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
) -> int:
    """Retry one clean allocation only for the known transient NCCL proxy crash."""
    try:
        return sbatch_run(
            script, log,
            extra_sbatch_args=extra_sbatch_args,
            background=background,
            dry_run=dry_run,
        )
    except RuntimeError:
        if background or dry_run or not _transient_nccl_proxy_failure(log):
            raise
        logger.warning(
            "[slurm_job:retry] NCCL InfiniBand proxy segfault detected; "
            "retrying once on a clean allocation",
        )
        _clean_run_for_retry()
        return sbatch_run(script, log, extra_sbatch_args=extra_sbatch_args)

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

def _rank_memory_measurements(job_log: Path) -> str:
    """Collect final memory records from every rank log belonging to a job."""
    if job_log.parent.name != "logs":
        return ""
    run_root = job_log.parent.parent
    records = []
    for path in sorted(run_root.glob("**/logs/*-routine_rank*.log")):
        records.extend(match.group(0) for match in _MEM_RE.finditer(path.read_text(errors="replace")))
    return "\n".join(records)

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
        baseline["metrics"] = {**baseline.get("metrics", {}), "bw_mbps": PerfReport._platform_bandwidth_baseline(platform_tag())}
        return baseline

    @staticmethod
    def _metric(samples: list[float | int], unit: str, higher: bool | None, ranks: list[int] | None = None) -> dict:
        return {
            "samples": samples,
            "ranks": ranks,
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "unit": unit,
            "higher_is_better": higher,
        }

    @staticmethod
    def _routine_rank_logs(job_log: Path) -> dict[str, dict[int, Path]]:
        """The per-rank routine logs a job wrote, grouped by the routine that wrote them.

        A determinism job runs two routines, `--run=ref:... --run=cmp:...`, each into its
        own directory. They are what the job exists to compare, so both are reported. The
        report is handed the job log, so the rank logs are found beside it the way the
        per-rank memory records are.
        """
        if job_log.parent.name != "logs":
            return {}
        grouped: dict[str, dict[int, Path]] = {}
        for path in job_log.parent.parent.glob("**/logs/*-routine_rank*.log"):
            if match := re.fullmatch(r"(?P<label>.+)-routine_rank(?P<rank>\d+)", path.stem):
                grouped.setdefault(match["label"], {})[int(match["rank"])] = path
        return dict(sorted(grouped.items()))

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _log_checkpoint_timings(job_log: Path) -> None:
        """Report what each routine a job ran cost to checkpoint.

        Megatron's Timers.log writes to the last rank, while the save/saved pair a
        durability span is measured from is printed by rank 0, so each routine is read
        from both ends of its own rank set.
        """
        for label, ranks in PerfReport._routine_rank_logs(job_log).items():
            last = ranks[max(ranks)]
            timers = [
                line.rstrip()
                for line in PerfReport._read(last).splitlines()
                if "save-checkpoint" in line or "load-checkpoint" in line
            ]
            if timers:
                logger.info(f"{label}: checkpoint timers from {last.name} (min, max ms across ranks)")
                for line in timers:
                    logger.info(line)
            spans = PerfReport._durability_ms(PerfReport._read(ranks[min(ranks)]))
            if spans:
                samples = ", ".join(f"{value:.2f}" for value in spans)
                logger.info(f"{label}: checkpoint durability (min, max ms across saves)")
                logger.info(
                    "    save -> durable ".ljust(52, ".")
                    + f": ({min(spans):.2f}, {max(spans):.2f})  mean {statistics.mean(spans):.2f}"
                    + f"  over {len(spans)} save(s) [{samples}]",
                )

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
    def _durability_ms(text: str) -> list[float]:
        """Milliseconds from each save starting to its checkpoint being durable.

        Reported in the same unit as Megatron's checkpoint timers so a job log can be read
        without converting between the two.

        Measured in the trainer rather than the daemon, because that is the only side both
        providers share: the nvrx provider never starts the checkpoint daemon. Covers the
        whole save, staging included, so it is the number a training run feels.

        Read in document order and consumed on use, because a determinism run saves the same
        iteration once per cycle: keyed only by iteration, an early save would pair with a
        later cycle's start and report a negative span.
        """
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        started: dict[int, str] = {}
        elapsed = []
        for match in _CKPT_EVENT_RE.finditer(text):
            timestamp, event, iteration = match[1], match[2], int(match[3])
            if event.startswith("saving"):
                started[iteration] = timestamp
                continue
            start = started.pop(iteration, None)
            if start is None:
                continue  # save began before the measured phase; its start is not in `text`
            span = datetime.strptime(timestamp, fmt) - datetime.strptime(start, fmt)
            elapsed.append(round(span.total_seconds() * 1000, 2))
        return elapsed

    @staticmethod
    def _daemon_text(results_dir: Path) -> str:
        """Concatenate the daemon logs of the newest measured cycle.

        Every host writes its own log and the daemon restarts each cycle, so the newest
        cycle is the routine a baseline is made of — the same routine the trainer log's
        last phase-2 marker selects.
        """
        by_cycle: dict[int, list[Path]] = {}
        for path in results_dir.glob("checkpoint_server.cycle*.log"):
            tag = path.name.split(".")[1]
            if tag.startswith("cycle") and tag[5:].isdigit():
                by_cycle.setdefault(int(tag[5:]), []).append(path)
        if not by_cycle:
            return ""
        return "\n".join(p.read_text(errors="replace") for p in sorted(by_cycle[max(by_cycle)]))

    @staticmethod
    def _parse_metrics(text: str, gpus_per_node: int | None = None, daemon_text: str | None = None) -> dict:
        text = text[text.rfind(_PHASE2_MARKER) :] if _PHASE2_MARKER in text else text
        # The memory records come from the ranks; the bandwidth counters from the daemon.
        matches = list(_BW_RE.finditer(text if daemon_text is None else daemon_text))
        metrics: dict = {}
        if matches:
            ranks = [int(m[1]) for m in matches]
            bw = [round(float(m[2])) for m in matches]
            write = [int(m[3]) for m in matches]
            metrics = {
                "bw_mbps": PerfReport._metric(bw, "MB/s", True, ranks),
                "wr_tnsr_us": PerfReport._metric(write, "us", False, ranks),
                "total_written_bytes": PerfReport._metric([int(m[4]) for m in matches], "B", None, ranks),
            }
            unique = sorted(set(ranks))
            per_rank = [statistics.mean(v for r, v in zip(ranks, bw) if r == rank) for rank in unique]
            if gpus_per_node and len(per_rank) == gpus_per_node and len(per_rank) >= 4 and len(per_rank) % 2 == 0:
                half = len(per_rank) // 2
                means = statistics.mean(per_rank[:half]), statistics.mean(per_rank[half:])
                metrics["bw_socket_imbalance"] = PerfReport._metric([max(means) / min(means)], "ratio", None)
        durable = PerfReport._durability_ms(text)
        if durable:
            metrics["ckpt_durable_ms"] = PerfReport._metric(durable, "ms", False)
        if daemon_text:
            daemon_durable = [int(m[2]) / 1e3 for m in _DAEMON_DURABLE_RE.finditer(daemon_text)]
            if daemon_durable:
                metrics["ckpt_daemon_durable_ms"] = PerfReport._metric(daemon_durable, "ms", False)
        memory = list(_MEM_RE.finditer(text))
        if memory:
            ranks = [int(m[1]) for m in memory if m[1] is not None]
            memory_ranks = ranks if len(ranks) == len(memory) else None
            metrics["gpu_peak_bytes"] = PerfReport._metric(
                [int(m[2]) for m in memory], "B", False, memory_ranks,
            )
            hbm = [(int(m[3]), int(m[4])) for m in memory if m[3] is not None and m[4] is not None]
            if hbm:
                hbm_ranks = ranks if len(ranks) == len(hbm) else None
                metrics["gpu_hbm_peak_bytes"] = PerfReport._metric(
                    [used for used, _total in hbm], "B", False, hbm_ranks,
                )
                metrics["gpu_hbm_peak_pct"] = PerfReport._metric(
                    [used * 100 / total for used, total in hbm], "%", False, hbm_ranks,
                )
        return metrics

    @staticmethod
    def _statistic(name: str, measured: dict) -> str:
        if name in {"gpu_hbm_peak_bytes", "gpu_hbm_peak_pct"} and "max" in measured:
            return "max"
        if name in {"bw_mbps", "wr_tnsr_us"} and "median" in measured:
            return "median"
        return "mean"

    @staticmethod
    def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
        expected, tolerances = baseline.get("metrics", {}), baseline.get("tolerances", {})
        result, passed = {}, True
        for name, measured in metrics.items():
            statistic = PerfReport._statistic(name, measured)
            value, higher = measured[statistic], measured["higher_is_better"]
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
                elif higher is False:
                    ok = value <= high
                else:
                    ok = low <= value <= high
            passed &= ok
            result[name] = {
                "status": "PASS" if ok else "FAIL",
                "measured": value,
                "baseline": target,
                "tolerance_pct": tolerance * 100,
                "unit": measured["unit"],
                "statistic": statistic,
            }
        return result, passed

    @staticmethod
    def parse_and_report(job: str, config: str, flavor: str, log: Path, baseline: dict, report_path: Path, gpus: int) -> bool:
        text = log.read_text(errors="replace")
        # Rank 0 is the only routine log mirrored into the combined job log.
        # Read the short per-rank memory records so the HBM peak describes the
        # most heavily loaded GPU rather than whichever GPU owns global rank 0.
        memory = _rank_memory_measurements(log)
        metric_text = f"{_MEM_RE.sub('', text)}\n{memory}" if memory else text
        metrics = PerfReport._parse_metrics(metric_text, gpus, PerfReport._daemon_text(log.parent))
        # What the run cost to checkpoint, gathered in one place once it is over: the
        # per-rank timers Megatron kept, then how long each save took to become durable.
        PerfReport._log_checkpoint_timings(log)
        PerfReport._log_daemon_durability(metrics)
        # The extension's own writer emits checkpoint times rather than the
        # bandwidth counters a baseline is made of, so it has nothing to compare.
        if not metrics and "native" not in flavor.split("-"):
            return True
        if not metrics:
            raise RuntimeError(f"no perf metrics found in {log} or the daemon logs beside it")
        comparison, passed = PerfReport._compare_to_baseline(metrics, baseline, PERF_TOLERANCE)
        for name, value in comparison.items():
            logger.info(f"[{value['status']}] {name}: {value['measured']}")
        report = {
            "job": job,
            "config": config,
            "flavor_tag": flavor,
            "cluster": cluster_tag(),
            "passed": passed,
            "metrics": {name: {k: v for k, v in metric.items() if k not in ("samples", "ranks")} for name, metric in metrics.items()},
            "comparison": comparison,
        }
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
    volatile = re.compile(r"^  (?:data_cache_path|load|save|tensorboard_dir) ")
    arguments = output / f"{_arguments_stem(args, variants)}_arguments.txt"
    arguments.write_text("\n".join(line for line in lines if not volatile.match(line)) + "\n")
    return arguments

def _argument_log(args: argparse.Namespace, batch_log: Path, spec: RunSpec) -> Path:
    """Return the rank-0 log containing a particular run's argument blocks."""
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    run_root = batch_log.parent.parent
    run = run_root / _run_dir_name(spec) if spec.name else run_root
    return run / "logs" / f"{cfg}_{spec.name or 'run'}-routine_rank0.log"

def _write_and_compare_arguments(args: argparse.Namespace, batch_log: Path, *, strict: bool = True) -> bool:
    """Generate, compare and retain arguments for every run the job executed."""
    results = []
    specs = _run_specs(args)
    for spec in specs:
        log = _argument_log(args, batch_log, spec)
        # A unit/local epilog may only have the combined log. Real multi-run
        # jobs always have one rank-0 log per run and must not fall back.
        if not log.exists() and len(specs) == 1:
            log = batch_log
        if not log.exists():
            message = f"argument log for run {spec.name!r} does not exist: {log}"
            if strict:
                raise RuntimeError(message)
            logger.warning(f"[slurm_job:args] WARNING {message}")
            continue
        try:
            _write_arguments_artifact(args, log, spec.variants)
        except RuntimeError:
            if strict:
                raise
            logger.warning(f"[slurm_job:args] WARNING no complete arguments in {log}")
            continue
        results.append(_compare_arguments(args, log, spec.variants))
    return len(results) == len(specs) and all(results)

def _write_performance_baseline(args: argparse.Namespace, log: Path) -> None:
    """Write measured metrics from the job-level report as a candidate baseline."""
    flavors = _selected_flavors(args)
    # NVRX does not expose the native writer's bandwidth counters. Keep its
    # report as raw diagnostic output without pretending the memory-only data
    # is a comparable checkpoint-performance baseline.
    if "perf" not in flavors.names or "nvrx" in flavors.names:
        return
    _cfg, _flavor, stem = _identity(args)
    output = log.parent / "baselines"
    output.mkdir(parents=True, exist_ok=True)

    report_path = log.with_name(f"{stem}_report.json")
    if not report_path.exists():
        return
    measured = json.loads(report_path.read_text()).get("metrics", {})
    metrics = {}
    for name, value in measured.items():
        # Socket imbalance is an in-job topology diagnostic rather than a
        # workload baseline. The measured bandwidth belongs in the candidate
        # even though the shared platform floor remains the enforced gate.
        if name == "bw_socket_imbalance":
            continue
        statistic = PerfReport._statistic(name, value)
        if statistic in value:
            measured_value = value[statistic]
            metrics[name] = round(measured_value, 2) if name.endswith("_pct") else round(measured_value)
    if metrics:
        baseline = {"description": "auto-generated by CI", "metrics": metrics}
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
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
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
    local_rank = os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0"))
    world = os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1"))
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
    phase2_save_interval: int = 250
    phase2_checkpoints: int = 3
    checkpoint_routines: int = 1
    manages_native_server: bool = True
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

    def train_config(self, cfg: str, args: argparse.Namespace, provider: str, phase: int, cycle: int = 0) -> TrainConfig:
        run, phase1 = _run_dir(cfg), _run_dir(cfg) / "phase1-checkpoints"
        extend_training = args.checkpoints_per_run is not None or args.repeat != 1
        if phase == 1:
            return TrainConfig(
                args.nodes, args.gpus_per_node, phase1, None, args.config, provider,
                1, self.phase1_train_iters, 1, extend_training,
            )
        checkpoints = self.phase2_checkpoints if args.checkpoints_per_run is None else args.checkpoints_per_run
        interval = self.phase2_save_interval if args.checkpoint_interval is None else args.checkpoint_interval
        start_iteration = self.phase1_train_iters
        if cycle:
            start_iteration = (self.phase1_train_iters // interval + cycle * checkpoints) * interval
        return TrainConfig(
            args.nodes,
            args.gpus_per_node,
            run / "checkpoints",
            phase1 if cycle == 0 else run / "checkpoints",
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

    def stop_server(self, provider: str, tag: str, graceful: bool = True) -> None:
        if self.manages_native_server and provider == "native":
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

    def configure(self, policy: FlavorPolicy) -> None:
        policy.requires_nvrx_absent = True
        policy.reporter = self.report

    def report(self, args: argparse.Namespace, log: Path) -> bool:
        cfg, tag, stem = _identity(args)
        baseline = args.baseline if hasattr(args, "baseline") else PerfReport.baseline(stem)
        if "nvrx" in _selected_flavors(args).names:
            baseline = {}
        return PerfReport.parse_and_report(
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
        policy.phase1_train_iters = 25
        # Determinism compares checkpoint contents, so retain three phase-2
        # checkpoints while shortening the training between them. A job that
        # names two runs trains both, and the former 250-iteration cadence
        # exceeded Ultra3's six-hour allocation.
        policy.phase2_save_interval = 25
        policy.compares_routines = True

    def report(self, args: argparse.Namespace, log: Path) -> bool:
        if not PerfFlavor().report(args, log):
            logger.warning("performance baseline failed; determinism comparisons passed")
        return True

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        # Megatron 77c87725 / TE in the 26.07 image fails while constructing
        # deterministic backward graphs for Pico, Nano, and Ultra on GB200
        # (`Could not infer dtype of _WeakRefTensor`). Perf retains full capture.
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
            NCCL_ALGO="Ring",
            NCCL_PROTO="Simple",
            NCCL_NVLS_ENABLE="0",
            NCCL_TUNER_PLUGIN="none",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
            MAMBA_DETERMINISTIC="1",
        )


class ValgrindFlavor(Flavor):
    execution_mode = True
    name = "vg"
    summary = "Run the checkpoint server under memcheck."
    priority = 100

    def configure(self, policy: FlavorPolicy) -> None:
        policy.checkpoint_routines = 1
        policy.manages_native_server = False
        policy.wheel_prep_mode = "--build-from-source"
        policy.wheel_defines = ("PHOENIX_BUILD_FOR_VALGRIND=1",)
        policy.install_prebuilt = False
        policy.worker = self.run
        policy.server = self.server
        policy.reporter = None

    def run(self, cfg: str, args: argparse.Namespace, provider: str, label: str = "") -> None:
        if provider == "native":
            checkpoints = (
                _selected_flavors(args).policy.phase2_checkpoints
                if args.checkpoints_per_run is None
                else args.checkpoints_per_run
            )
            if checkpoints:
                args = argparse.Namespace(**vars(args))
                args.checkpoints_per_run, args.repeat = 1, checkpoints * args.repeat
        _run_routine(cfg, args, provider, label)

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
        env["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"

    @contextlib.contextmanager
    def server(self, stem: str, phase: int) -> Generator[None, None, None]:
        host = socket.gethostname().split(".")[0]
        marker = _sync_root() / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{stem}-{host}"
        if os.environ.get("SLURM_LOCALID", "0") != "0":
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
            "--error-exitcode=42",
            f"--log-file={report}",
            str(nvrx_checkpointing.SERVER_PATH),
        ]
        with server_log.open("w") as output:
            proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline and proc.poll() is None:
                if "Checkpoint server started" in server_log.read_text(errors="replace"):
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


RunSpec = namedtuple("RunSpec", "name variants flavors")  # noqa: PYI024

def _run_dir_name(spec: RunSpec) -> str:
    """The directory one run writes to, inside the job's own directory.

    The variants a run adds lead the name and the run's own name follows, so
    two runs of one job read as what distinguishes them.
    """
    return f"{'-'.join(spec.variants)}.{spec.name}" if spec.variants else spec.name

def _run_specs(args: argparse.Namespace) -> tuple[RunSpec, ...]:
    """The runs this job performs, each with the variants it adds to the common set.

    A run is a full two-phase training that records what its checkpoints
    contain. Two runs differ by the variants they name, and the diff between
    their digests is the comparison.
    """
    common = list(args.flavor or [])
    specs = tuple(
        RunSpec(name, tuple(variants), FlavorSelection(list(variants) + common))
        for name, variants in (
            (entry.partition(":")[0], _flavor_names(entry.partition(":")[2]))
            for entry in (args.run or [""])
        )
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
        if k not in {"job_partition", "job_time_in_mins"}
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
    marker = _sync_root() / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{tag}-{host}"
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    tasks_spec = (
        os.environ.get("SLURM_NTASKS_PER_NODE")
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
    marker = _sync_root() / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{tag}"
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
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
    if not enabled or os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")) != "0":
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
) -> tuple[list[str], dict]:
    flavors = _selected_flavors(args)
    config = flavors.policy.train_config(cfg, args, provider, phase, cycle)
    opts, env = _train_options(cfg, config), {**os.environ, **_base_env()}
    _toolchain(cfg, args).apply(env)
    flavors.transform(opts, env)
    applied = flavors.policy.apply_options(opts, cfg, phase)
    _print_flavor_options(flavors, phase, applied)
    expert = opts[opts.index("--expert-model-parallel-size") + 1]
    env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = expert
    if "--fine-grained-activation-offloading" in opts:
        env["NVTE_CPU_OFFLOAD_V1"] = "1"
    os.environ["NATIVE_CKPT_SERVER_LOG"] = env["NATIVE_CKPT_SERVER_LOG"]
    return ["python3", "-u", str(Path(__file__).resolve()), _PRETRAIN_ENTRY_CMD, str(phase), *opts, *args.extra_args], env
def _print_flavor_options(flavors: FlavorSelection, phase: int, applied: list[str]) -> None:
    """Record what the selected variants changed about this phase's command."""
    if os.environ.get("SLURM_PROCID", "0") != "0" or not applied:
        return
    print(f"[slurm_job:flavor] {flavors.tag} phase{phase} {' '.join(applied)}", flush=True)  # noqa: T201

def _print_pretrain_env(env: dict[str, str], cycle_tag: str) -> None:
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return
    print(f"[slurm_job:env] {cycle_tag} rank0 environment", flush=True)  # noqa: T201
    for name in sorted(_RUN_ENV_NAMES | _PRETRAIN_ENV_NAMES | _TRAIN_ENV_DEFAULTS.keys()):
        if name in env:
            print(f"[slurm_job:env] {name}={shlex.quote(env[name])}", flush=True)  # noqa: T201
def _run_routine(cfg: str, args: argparse.Namespace, provider: str, label: str = "") -> None:
    flavors = _selected_flavors(args)
    stem = f"{platform_tag()}_{flavors.settled_tag}_{cfg}"
    phase, cycle = 1, 0
    while phase == 1 or args.repeat == 0 or cycle < args.repeat:
        cmd, env = _phase_command(cfg, args, provider, phase, cycle)
        log = RESULTS_DIR / f"{cfg}_{label}-routine_rank{os.environ.get('SLURM_PROCID', '0')}.log" if label else None
        cycle_tag = "phase1" if phase == 1 else f"cycle{cycle + 1}"
        # The daemon carries the perf counters now, and it is restarted per cycle. Name its
        # log for the cycle so the report can tell the measured routine from the reference.
        os.environ["NATIVE_CKPT_SERVER_LOG"] = str(RESULTS_DIR / f"checkpoint_server.{cycle_tag}.%h.%p.log")
        if os.environ.get("SLURM_PROCID", "0") == "0":
            print(f"[native_ckpt:run] {cycle_tag}", flush=True)  # noqa: T201
        with (
            flavors.policy.server_context(f"{stem}-{cycle_tag}", phase),
            _server_heartbeat(flavors.policy.manages_native_server and provider == "native"),
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
                    mirror_stdout=os.environ.get("SLURM_PROCID", "0") == "0",
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
            phase = 2
        else:
            cycle += 1
        flavors.policy.stop_server(provider, f"{label}-{cycle_tag}-stop")
        _sync_ranks(f"{label}-{cycle_tag}-stopped".strip("-"))
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
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    node_index = rank // args.gpus_per_node
    is_node_leader = local_rank == 0
    job_id = os.environ.get("SLURM_JOB_ID", "local")

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

def _write_checkpoint_digest(run: Path, stem: str, baseline_for: Callable[[str, str], Path] | None = None) -> None:
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
        return
    (RESULTS_DIR / f"{stem}_checkpoint_summary.log").write_text("\n".join(summary) + "\n")
    with lzma.open(RESULTS_DIR / f"{stem}_checkpoint_digest.xz", "wt") as digest:
        digest.writelines(lines)
    for line in summary:
        logger.info(f"[slurm_job:digest] {stem} {line}")

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
    }
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
def _in_run(spec: RunSpec, args: argparse.Namespace) -> Generator[None, None, None]:
    """Point the run-scoped paths and variants at one run for its duration."""
    global RESULTS_DIR, RUN_ROOT  # noqa: PLW0603
    previous = (RUN_ROOT, RESULTS_DIR, args.flavors)
    args.flavors = spec.flavors
    # A run writes inside the job that owns it, in a directory named for the
    # variants it adds, so the job directory holds the whole comparison and
    # one path says which side of it this is.
    if spec.name:
        RUN_ROOT = previous[0] / _run_dir_name(spec)
        RESULTS_DIR = RUN_ROOT / "logs"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SLURMJOB_OPT_RUN_DIR"] = str(RUN_ROOT)
    os.environ["NATIVE_CKPT_SERVER_LOG"] = str(RESULTS_DIR / "checkpoint_server.%h.%p.log")
    try:
        yield
    finally:
        RUN_ROOT, RESULTS_DIR, args.flavors = previous


def _perform_runs(cfg: str, args: argparse.Namespace) -> None:
    """Train every run this job names, then compare what they wrote.

    Each run is independent: its own variants, its own directory, its own
    digest. Nothing one run writes is read by another, so the comparison is a
    diff of digests rather than a second routine checking the first.
    """
    rank0 = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")) == "0"
    specs, digests = _run_specs(args), []
    comparable = len(specs) > 1 or args.baseline is not None or any(
        spec.flavors.policy.compares_routines for spec in specs
    )
    for spec in specs:
        with _in_run(spec, args):
            flavors = spec.flavors
            if rank0:
                logger.info(f"[slurm_job:run] {spec.name or 'run'} {flavors.tag} "
                            f"{_toolchain(cfg, args).describe('sources')}")
            # Naming the routine keeps every rank's output in a per-rank file
            # and mirrors only rank 0 into the job trace. Without it 256 ranks
            # write straight to the trace and bury the run in its own logs.
            flavors.policy.run_worker(cfg, args, _provider(flavors), spec.name or "run")
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
            if failures := _manifest_batch(f"{spec.name or 'run'}-manifest", iterations, None, None, args):
                raise RuntimeError(f"manifest derivation failed: {', '.join(failures)}")
            if not rank0 or not comparable:
                continue
            stem = _identity(args)[2]
            _write_checkpoint_digest(run, stem)
            digests.append(RESULTS_DIR / f"{stem}_checkpoint_digest.xz")
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
    failures = [
        spec.name or "run"
        for spec, digest in zip(specs, digests, strict=True)
        if digest != against and _digest_differences(against, digest)
    ]
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


def _print_source_provenance() -> None:
    """Record the Megatron-LM and NVIDIA Resiliency Extension this phase loaded.

    Each routine of a cross-source comparison selects its own checkouts through
    ``MEGATRON_ROOT`` and ``PYTHONPATH``. Resolving them from inside the
    training process, after every import has bound, is what makes a comparison
    self-evidencing: the log names the revisions that ran rather than the ones
    that were requested.
    """
    root = megatron_root()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"[native_ckpt:sources] megatron={root}@{revision or 'unknown'}", flush=True)  # noqa: T201
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
    print(f"[native_ckpt:sources] nvidia_resiliency_ext={nvidia_resiliency_ext.__file__}@{version}", flush=True)  # noqa: T201

class _PeakHbmSampler:
    """Sample whole-device HBM use after training initializes its CUDA device."""

    def __init__(self, torch_module: ModuleType, interval_seconds: float = 1.0):
        self.torch = torch_module
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.total_bytes = 0
        self.device: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="peak-hbm-sampler", daemon=True)

    def _sample_once(self) -> None:
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

def _pretrain_entry(phase: int) -> None:
    sys_path_insert(megatron_root())
    if _install_torch_serialization_compat() and os.environ.get("SLURM_PROCID", "0") == "0":
        print("[slurm_job] installed missing torch.utils.serialization config compatibility", file=sys.stderr, flush=True)  # noqa: T201

    from nvrx_checkpointing import megatron_integration  # noqa: PLC0415

    megatron_integration.install()
    from megatron.core.enums import ModelType  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    from megatron.training import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        inprocess_restart,
        pretrain,
        set_startup_timestamps,
    )
    from megatron.training.argument_utils import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
        hybrid_config_from_args,
        pretrain_cfg_container_from_args,
    )
    from megatron.training.arguments import parse_and_validate_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]

    try:
        from megatron.post_training.arguments import add_modelopt_args  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    except ImportError:
        add_modelopt_args = None
    import pretrain_hybrid as ph  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    from hybrid_builders import hybrid_builder  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
    from model_provider import model_provider  # noqa: PLC0415 # pyright: ignore[reportMissingImports]

    rank0 = os.environ.get("RANK", "0") == "0"
    if rank0:
        print(f"[native_ckpt:phase] phase={phase}", flush=True)  # noqa: T201
        _print_source_provenance()
    start = time.time()
    set_startup_timestamps(program_start=start, main_entry=start)
    ph.train_valid_test_datasets_provider.is_distributed = True
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(
        extra_args_provider=megatron_integration.compose_extra_args_provider(add_modelopt_args),
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    model_cfg = hybrid_config_from_args(args)
    import torch  # noqa: PLC0415

    hbm = _PeakHbmSampler(torch)
    hbm.start()
    try:
        pretrain_fn(
            pretrain_cfg_container_from_args(args, model_cfg),
            train_valid_test_dataset_provider=ph.train_valid_test_datasets_provider,
            model_provider=partial(model_provider, hybrid_builder),
            model_type=ModelType.encoder_or_decoder,
            forward_step_func=ph.forward_step,
            store=store,
        )
    finally:
        import resource  # noqa: I001, PLC0415

        hbm.close()
        gpu_peak = torch.cuda.max_memory_allocated(hbm.device) if torch.cuda.is_initialized() else 0
        print(  # noqa: T201
            f"[native_ckpt:mem] rank={os.environ.get('RANK', '0')} gpu_peak_bytes={gpu_peak} "
            f"gpu_hbm_peak_bytes={hbm.peak_bytes} gpu_total_bytes={hbm.total_bytes} "
            f"cpu_rss_bytes={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}",
            flush=True,
        )

def _worker(args: argparse.Namespace) -> None:
    flavors = _selected_flavors(args)
    provider = _provider(flavors)
    if provider == "native" and flavors.policy.requires_nvrx_absent:
        uninstall_nvrx_for_native()
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    if target := _nvrx_target(cfg, args.nvrx_commit):
        sys_path_insert(target)
    CHECKPOINT_SERVER.require_stopped()
    try:
        _perform_runs(cfg, args)
    except BaseException:
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
        if flavors.policy.manages_native_server and provider == "native" and local_rank == 0:
            with contextlib.suppress(Exception):
                CHECKPOINT_SERVER.request("stop", 5)
        raise

def _identity(args: argparse.Namespace) -> tuple[str, str, str]:
    """The profile, the combination and the stem a run reports under."""
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    flavors = _selected_flavors(args)
    return cfg, flavors.tag, f"{platform_tag()}_{flavors.settled_tag}_{cfg}"

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
    toolchain = _toolchain(cfg, args)
    prep_commands = [["bash", str(ENROOT_SCRIPTS_DIR / "megatron_root.sh")]]
    prep_commands.append([
        "python3", "-u", str(Path(wheel_install.__file__).resolve()),
        *flavors.policy.prep_args(),
    ])
    if toolchain.nvrx_target:
        prep_commands.append([
            "python3", "-u", str(SCRIPT_DIR / "install_run_dependency.py"),
            "--repo-url", args.nvrx_repo_url,
            *(("--commit", args.nvrx_commit) if args.nvrx_commit else ()),
            "--target", str(toolchain.nvrx_target),
            *(("--repo-root", str(nvrx_repo_root())) if nvrx_repo_root() else ()),
        ])
    prep_commands.append(["python3", "-m", "pip", "list"])
    prep_commands.append(["python3", "-u", str(Path(wheel_install.__file__).resolve()), "--report-install"])
    prep = [*step(1, 1, ",".join(_env_names("p")), prep_log, writable=True),
            "/bin/bash", "-c", " && ".join(shell_join(command) for command in prep_commands)]
    worker_args = [
        "--nodes", str(args.nodes), "--gpus-per-node", str(args.gpus_per_node), "--config", args.config,
        "--flavor", "+".join(flavors.names),
        *(token for entry in (args.run or []) for token in ("--run", entry)),
    ]
    epilog_args = [*worker_args]
    worker_args.extend(args.extra_args)
    container_env = ",".join(_env_names("w"))
    native_perf_verification = _provider(flavors) == "native" and flavors.policy.requires_nvrx_absent
    run = [*step(args.nodes, total_tasks, container_env, writable=native_perf_verification),
           "python3", "-u", str(Path(__file__).resolve()), *worker_args]
    epilog = [
        "python3", "-u", str(Path(__file__).resolve()), _EPILOG_ENTRY_CMD, *epilog_args,
        "--log-file", str(log), "--baseline-json", json.dumps(PerfReport.baseline(stem), separators=(",", ":")),
    ]
    rank_logs = [
        "python3", "-u", str(Path(__file__).resolve()), _RANK_LOGS_ENTRY_CMD, *epilog_args,
        "--log-file", str(log),
    ]
    host_cleanup = [
        *build_srun_cmd(
            nodes=args.nodes, ntasks=args.nodes, total_gpus=total_tasks, container_env="",
            time_minutes=_CLEANUP_GRACE_MINUTES, container=False,
            extra_srun_args=["--ntasks-per-node=1"], **common,
        ),
        "python3", "-u", str(SCRIPT_DIR / "daemon_tools.py"),
    ]
    body = [
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
        'if [[ "${SLURMJOB_OPT_CLEAN_RUN:-0}" == 1 ]]; then',
        '  [[ "${SLURMJOB_OPT_RUN_DIR:?}" != / && "${SLURMJOB_OPT_RUN_DIR##*/}" == "${SLURMJOB_OPT_RUN_NAME:?}" ]] || exit 2',
        ('  find "${SLURMJOB_OPT_RUN_DIR}" -mindepth 1 -maxdepth 1 ! -path "${SLURMJOB_OPT_RUN_DIR}/run.sbatch" '
        '! -path "${SLURMJOB_OPT_RUN_DIR}/job_id" ! -path "${SLURMJOB_OPT_RUN_DIR}/gitlab_job_id" '
        '! -path "${SLURMJOB_OPT_RUN_DIR}/logs" -exec rm -rf -- {} +'),
        '  mkdir -p "${SLURMJOB_OPT_RUN_DIR}/logs"',
        '  find "${SLURMJOB_OPT_RUN_DIR}/logs" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +',
        "fi",
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
    script, reaper = build_sbatch_script(
        nodes=args.nodes, ntasks=total_tasks, time_minutes=minutes, job_name=job_name, image=image,
        container_env="", gpus_per_node=args.gpus_per_node, total_gpus=total_tasks, body="\n".join(body),
        partition=partition,
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
_commit = partial(_validated, convert=str, valid=_COMMIT_RE.fullmatch, error="must be a 7-40 character hexadecimal commit")
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
    "nvrx_commit": RunOption("NVRX_COMMIT", convert=_commit, scopes="rsw"),
    "nvrx_repo_url": RunOption("NVRX_REPO_URL", DEFAULT_NVRX_REPO_URL, _url, scopes="rsw"),
    "checkpoints_per_run": RunOption("SLURMJOB_OPT_CHECKPOINTS_PER_RUN", convert=_nonnegative_int, scopes="rw"),
    "checkpoint_interval": RunOption("SLURMJOB_OPT_CHECKPOINT_INTERVAL", convert=_positive_int, scopes="rw"),
    "repeat": RunOption("SLURMJOB_OPT_REPEAT", 1, _nonnegative_int, scopes="rw"),
    "time_minutes": RunOption("SLURMJOB_OPT_TIME_MINUTES", convert=_positive_int, scopes="rw"),
    "clean_run": RunOption("SLURMJOB_OPT_CLEAN_RUN", False, _parse_bool, argparse.BooleanOptionalAction),
    "background": RunOption("SLURMJOB_OPT_BACKGROUND", False, _parse_bool, argparse.BooleanOptionalAction),
    "dry_run": RunOption("SLURMJOB_OPT_DRY_RUN", False, _parse_bool, argparse.BooleanOptionalAction),
}
_ENV_SCOPES = {
    **dict.fromkeys(_TRAIN_ENV_DEFAULTS, "rw"),
    "VCPKG_ROOT": "rspw", "MEGATRON_NATIVE_BUILD_DIR": "rspw", "NVRX_REPO_ROOT": "rspw",
    "SLURMJOB_OPT_PYTHON_INSTALL_DIR": "rspw", "ENROOT_IMAGE_DIR": "rs", "HOME": "rs", "MEGATRON_ROOT": "rspw",
    "SLURMJOB_OPT_LAUNCHER_ROOT": "rs", "SLURMJOB_OPT_RUN_DIR": "rsw", "SLURMJOB_OPT_WORK_ROOT": "rspw",
    "PYTHONPATH": "rpw",
    "CI_API_V4_URL": "rp", "CI_COMMIT_SHA": "rp", "CI_PROJECT_ID": "rp", "WHEEL_ARCH": "rp", "CI_JOB_TOKEN": "p",
    "CI_DIR": "r", "DOCKER_IMAGE_BASE": "r", "LD_LIBRARY_PATH": "r", "PATH": "r",
    "SLURM_SEGMENT": "r", "TRITON_CACHE_DIR": "rw", "TRITON_CACHE_PATH": "rw",
    "TORCHINDUCTOR_CACHE_DIR": "rw", "TORCH_EXTENSIONS_DIR": "rw", "NATIVE_CKPT_SERVER_LOG": "rw",
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

def _parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    args, extra = parser.parse_known_args(argv)
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
    return args, extra

def _rank_logs_entry(argv: list[str]) -> None:
    """Preserve available arguments and show logs of ranks named by srun."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    parser.add_argument("--log-file", type=Path, required=True)
    args, _extra = _parse_args(parser, argv)
    args.flavors = FlavorSelection(args.flavor)
    _selected_flavors(args)
    _report_failing_rank_logs(args.log_file)
    # The regular epilog is not reached after srun fails. Retain and compare
    # every argument block that was completed before the failure so failed jobs
    # provide the same diagnostic artifacts as successful jobs.
    _write_and_compare_arguments(args, args.log_file, strict=False)

def _epilog(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--baseline-json", required=True)
    args, _extra = _parse_args(parser, argv)
    args.flavors = FlavorSelection(args.flavor)
    _selected_flavors(args)
    args.baseline = json.loads(args.baseline_json)
    text = args.log_file.read_text(errors="replace")
    saves = _SAVE_RE.findall(text)
    phase1 = sum("/phase1-checkpoints" in path for path in saves)
    phase2 = sum("/checkpoints" in path and "/phase1-checkpoints" not in path for path in saves)
    # A recorded baseline stands in for the reference routine, so only the
    # compare routine writes checkpoints.
    routines = len(args.run or [""])
    expected_phase1 = routines
    expected_phase2 = args.flavors.policy.phase2_checkpoints if args.checkpoints_per_run is None else args.checkpoints_per_run
    expected_phase2 *= (args.repeat or 1) * routines
    logger.info(
        f"[slurm_job:epilog] checkpoint saves: phase1={phase1}, phase2={phase2}, "
        f"expected_phase1={expected_phase1}, expected_phase2={expected_phase2}",
    )
    if phase1 < expected_phase1 or (expected_phase2 and phase2 < expected_phase2):
        raise RuntimeError(
            f"incomplete checkpoint cycle in {args.log_file}: "
            f"found {phase1} phase-1 and {phase2} phase-2 saves; "
            f"expected at least {expected_phase1} and {expected_phase2}",
        )
    passed = args.flavors.policy.epilog(args, args.log_file)
    _write_performance_baseline(args, args.log_file)
    # The arguments a run trained on are part of what its result means, so they
    # are compared with the same standing as the numbers it produced.
    passed = _write_and_compare_arguments(args, args.log_file) and passed
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

def _export_settings(args: argparse.Namespace) -> None:
    """Publish the settings the rest of the run reads from the environment.

    The hardware, the cluster and the account are read wherever they are
    needed rather than threaded through, so an argument for one has to reach
    the environment before anything asks.
    """
    for dest in ("platform", "cluster", "slurm_account"):
        if value := getattr(args, dest, None):
            os.environ[_RUN_OPTIONS[dest].env] = str(value)

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
    if args.checkpoints_per_run == 0 and args.repeat != 1:
        parser.error("an unlimited checkpoint run cannot be repeated because it does not return")
    if args.flavors.policy.requires_finite_run and (args.checkpoints_per_run == 0 or args.repeat == 0):
        parser.error("the selected flavor requires finite checkpoint and repeat counts")
    _validate_comparison_options(parser, args)

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
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    if target := _nvrx_target(cfg, args.nvrx_commit):
        sys_path_insert(target)
    if not args.dry_run and args.flavors.policy.install_prebuilt:
        wheel_install.install_prebuilt_wheel(probe=False)
    script, reaper, log = _build_job(args)
    rc = _sbatch_run_with_retry(
        script, log,
        extra_sbatch_args=[reaper],
        background=args.background,
        dry_run=args.dry_run,
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
def _digest_differences(left: Path, right: Path) -> int:
    """Count how the checkpoints two digests describe differ, reporting each."""
    sides = [_digest_entries(path) for path in (left, right)]
    differences = 0
    for label in sorted(sides[0].keys() | sides[1].keys()):
        produced, recorded = sides[1].get(label), sides[0].get(label)
        if produced is None or recorded is None:
            logger.info(f"[slurm_job:diff] {label}: only in {(left if recorded else right).name}")
            differences += 1
            continue
        counts = _compare_entries(produced, recorded)
        logger.info(f"[slurm_job:diff] {label}: entries={len(produced)} {counts}")
        differences += sum(
            int(field.split("=")[1])
            for field in counts.split()
            if field.startswith(("missing", "extra", "changed"))
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
        phase = int(sys.argv.pop(2))
        sys.argv = ["pretrain_hybrid.py", *sys.argv[2:]]
        _pretrain_entry(phase)
    elif len(sys.argv) > 1 and sys.argv[1] == _EPILOG_ENTRY_CMD:
        _epilog(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == _RANK_LOGS_ENTRY_CMD:
        _rank_logs_entry(sys.argv[2:])
    else:
        main()
