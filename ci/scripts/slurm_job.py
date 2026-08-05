#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
TEMPLATES_DIR = (SCRIPT_DIR / "../templates").resolve()
ENROOT_SCRIPTS_DIR = REPO_ROOT / "scripts"
BANDWIDTH_BASELINE_FILE = BASELINES_DIR / "platform_bw_mbps.yaml"
DOCKER_IMAGE_BASE = os.environ.get("DOCKER_IMAGE_BASE", "nvcr.io/nvidia/pytorch:26.07-py3")
DEFAULT_NVRX_REPO_URL = "https://github.com/NVIDIA/nvidia-resiliency-ext.git"
_NVRX_DISTRIBUTION = "nvidia-resiliency-ext"
_NVRX_MODULE = "nvidia_resiliency_ext"
SOFT_FAIL_RC = 3
PERF_TOLERANCE = 0.05
MAX_BW_SOCKET_IMBALANCE = 1.30
RESULTS_DIR = REPO_ROOT / "ci/ci-results"
RUN_ROOT: Path | None = None
RUN_NAME: str | None = None
logger = logging.getLogger("native_ci")
_PRETRAIN_ENTRY_CMD = "__pretrain_entry__"
_EPILOG_ENTRY_CMD = "__epilog__"
_PHASE1_TRAIN_ITERS = 100
_LONG_RUN_TRAIN_SAMPLES = 1_000_000_000
_CLEANUP_GRACE_MINUTES = 5
_TORCH_STORAGE_ALIGNMENT = 64
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_DEFAULT_NAME = (
    "{job_name}_{config}_{nodes}n_mlm.{mlm_abbreviated_sha_commit}_"
    "nvrx.{nvrx_abbreviated_sha_commit}_nckpt.{native_checkpointing_abbreviated_sha_commit}"
)
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
_BW_RE = re.compile(
    r"\[native_ckpt:cli\]\[(\d+)\]\[INFO\]\[perf\] Finished "
    r"bw_mbps=([0-9.]+).*?wr_tnsr_us=([0-9]+).*?total_written_bytes=([0-9]+)",
    re.DOTALL,
)
_MEM_RE = re.compile(r"\[native_ckpt:mem\] gpu_peak_bytes=(\d+) cpu_rss_bytes=(\d+)")
_PHASE2_MARKER = "[native_ckpt:phase] phase=2"
_SAVE_RE = re.compile(r"successfully saved (?:async )?checkpoint from iteration\s+\d+\s+to\s+(\S+)")
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
def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}
def _run_name(template: str | None, args: argparse.Namespace) -> str:
    job_id = os.environ.get("CI_JOB_ID", "").strip()
    template = (template or ("{gitlab_pipeline_jobid}_" if job_id else "") + _DEFAULT_NAME).strip()
    if not job_id:
        template = template.replace("{gitlab_pipeline_jobid}_", "")
    values = {
        "gitlab_pipeline_jobid": job_id,
        "job_name": f"{args.job}_{_selected_flavors(args).tag}",
        "config": args.config,
        "nodes": args.nodes,
        "mlm_abbreviated_sha_commit": (args.megatron_commit or "image")[:8],
        "nvrx_abbreviated_sha_commit": (args.nvrx_commit or "image")[:8],
        "native_checkpointing_abbreviated_sha_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
            text=True,
        ).strip()[:8],
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
    global RESULTS_DIR, RUN_NAME, RUN_ROOT
    if value := os.environ.get("NVRX_REPO_ROOT"):
        resolved = Path(value).expanduser().resolve()
        if any(c in resolved.as_posix() for c in ",:"):
            raise SystemExit("NVRX_REPO_ROOT may not contain ',' or ':'")
        os.environ["NVRX_REPO_ROOT"] = resolved.as_posix()
    os.environ["SLURMJOB_OPT_LAUNCHER_ROOT"] = REPO_ROOT.as_posix()
    name = _run_name(args.name, args)
    root = (args.work_dir or REPO_ROOT / "workdir").expanduser().resolve()
    RUN_NAME, RUN_ROOT, RESULTS_DIR = name, root / name, root / name / "logs"
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
        if value is not None:
            os.environ[option.env] = ",".join(value) if option.action == "append" else str(int(value) if isinstance(value, bool) else value)
    os.environ.update(
        SLURMJOB_OPT_RUN_DIR=str(RUN_ROOT),
        SLURMJOB_OPT_PYTHON_INSTALL_DIR=str(RUN_ROOT / "deps/python"),
        MEGATRON_ROOT=os.environ.get("MEGATRON_ROOT", str(RUN_ROOT / "deps/megatron-lm")),
        MEGATRON_NATIVE_BUILD_DIR=str(RUN_ROOT / "build/nvrx_checkpointing"),
        ENROOT_IMAGE_DIR=str(args.image.parent),
    )
    for name, default in _TRAIN_ENV_DEFAULTS.items():
        os.environ.setdefault(name, default)
    os.environ.setdefault("TRITON_CACHE_DIR", str(RUN_ROOT / "triton-cache"))
    os.environ.setdefault("TRITON_CACHE_PATH", os.environ["TRITON_CACHE_DIR"])
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(RUN_ROOT / "torchinductor-cache"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(RUN_ROOT / "torch-extensions"))
    os.environ.setdefault("NATIVE_CKPT_SERVER_LOG", str(RESULTS_DIR / "checkpoint_server.%h.%p.log"))

def ensure_vcpkg_root() -> None:
    os.environ.setdefault("VCPKG_ROOT", str(RUN_ROOT / "deps/vcpkg" if RUN_ROOT else REPO_ROOT.parent / "vcpkg"))
    os.environ.setdefault(
        "SLURMJOB_OPT_PYTHON_INSTALL_DIR",
        str(RUN_ROOT / "deps/python" if RUN_ROOT else REPO_ROOT / "build/wheel-install"),
    )
    sys_path_insert(Path(os.environ["SLURMJOB_OPT_PYTHON_INSTALL_DIR"]))

def uninstall_nvrx_for_native() -> None:
    """Remove NVRx inside a marked native worker container and prove it is absent."""
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
            importlib.metadata.distribution(_NVRX_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            logger.info(f"[slurm_job] {_NVRX_DISTRIBUTION} is already uninstalled")
        else:
            logger.info(f"[slurm_job] uninstalling {_NVRX_DISTRIBUTION} for native perf verification")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "uninstall",
                    "--yes",
                    "--break-system-packages",
                    _NVRX_DISTRIBUTION,
                ],
                check=True,
            )
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
        for key, suffix in (
            ("MEGATRON_NATIVE_BUILD_DIR", "/build/nvrx_checkpointing"),
            ("MEGATRON_ROOT", "/deps/megatron-lm"),
            ("SLURMJOB_OPT_PYTHON_INSTALL_DIR", "/deps/python"),
            ("TRITON_CACHE_DIR", "/triton-cache"),
            ("TRITON_CACHE_PATH", "/triton-cache"),
            ("TORCHINDUCTOR_CACHE_DIR", "/torchinductor-cache"),
            ("TORCH_EXTENSIONS_DIR", "/torch-extensions"),
            ("VCPKG_ROOT", "/deps/vcpkg"),
            ("NATIVE_CKPT_SERVER_LOG", "/logs/checkpoint_server.%h.%p.log"),
        ):
            if source.get(key) == run_dir + suffix:
                derived[key] = f'"${{SLURMJOB_OPT_RUN_DIR}}{suffix}"'
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
            rc = submit.wait()
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

class PerfReport:
    @staticmethod
    def _platform_bandwidth_baseline(platform: str) -> int:
        values = _load_yaml(BANDWIDTH_BASELINE_FILE)
        if platform not in values:
            raise RuntimeError(f"missing bw_mbps baseline for {platform!r}")
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
    def _parse_metrics(text: str, gpus_per_node: int | None = None) -> dict:
        text = text[text.rfind(_PHASE2_MARKER) :] if _PHASE2_MARKER in text else text
        matches = list(_BW_RE.finditer(text))
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
        memory = list(_MEM_RE.finditer(text))
        if memory:
            metrics["gpu_peak_bytes"] = PerfReport._metric([int(m[1]) for m in memory], "B", False)
        return metrics

    @staticmethod
    def _compare_to_baseline(metrics: dict, baseline: dict, default_tolerance: float) -> tuple[dict, bool]:
        expected, tolerances = baseline.get("metrics", {}), baseline.get("tolerances", {})
        result, passed = {}, True
        for name, measured in metrics.items():
            statistic = "median" if name in {"bw_mbps", "wr_tnsr_us"} and "median" in measured else "mean"
            value, higher = measured[statistic], measured["higher_is_better"]
            if name == "bw_socket_imbalance":
                ok, target, tolerance = value <= MAX_BW_SOCKET_IMBALANCE, MAX_BW_SOCKET_IMBALANCE, 0
            elif name not in expected:
                result[name] = {"status": "info" if higher is None else "no_baseline", "measured": value}
                continue
            else:
                target = float(expected[name])
                tolerance = tolerances.get(name, 0 if higher is None else default_tolerance)
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
        for line in text.splitlines():
            if "save-checkpoint" in line or "load-checkpoint" in line:
                logger.info(line)
        metrics = PerfReport._parse_metrics(text, gpus)
        if not metrics and job == "pretrain-nvrx":
            return True
        if not metrics:
            raise RuntimeError(f"no perf metrics found in {log}")
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

def _write_baseline_artifacts(args: argparse.Namespace, log: Path) -> None:
    _cfg, _flavor, stem = _identity(args)
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
    (output / f"{stem}_arguments.txt").write_text("\n".join(line for line in lines if not volatile.match(line)) + "\n")

    report_path = log.with_name(f"{stem}_report.json")
    if not report_path.exists():
        return
    measured = json.loads(report_path.read_text()).get("metrics", {})
    metrics = {}
    for name, value in measured.items():
        if name in {"bw_mbps", "bw_socket_imbalance"}:
            continue
        statistic = "median" if name == "wr_tnsr_us" else "mean"
        if statistic in value:
            metrics[name] = round(value[statistic])
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
    for flag in flags:
        if flag not in opts:
            continue
        start, end = opts.index(flag), opts.index(flag) + 1
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

@dataclass
class FlavorPolicy:
    phase2_save_interval: int = 250
    phase2_checkpoints: int = 3
    checkpoint_routines: int = 1
    manages_native_server: bool = True
    requires_nvrx_absent: bool = False
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
                1, _PHASE1_TRAIN_ITERS, 1, extend_training,
            )
        checkpoints = self.phase2_checkpoints if args.checkpoints_per_run is None else args.checkpoints_per_run
        interval = self.phase2_save_interval if args.checkpoint_interval is None else args.checkpoint_interval
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
        )

    def server_context(self, stem: str, phase: int) -> contextlib.AbstractContextManager:
        return self.server(stem, phase) if self.server else contextlib.nullcontext()

    def run_worker(self, cfg: str, args: argparse.Namespace, provider: str) -> None:
        (self.worker or _run_routine)(cfg, args, provider)

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

    def prep_args(self) -> list[str]:
        return [self.wheel_prep_mode, *(arg for define in self.wheel_defines for arg in ("--define", define))]

    def epilog(self, args: argparse.Namespace, log: Path) -> bool:
        return self.reporter(args, log) if self.reporter else True


class Flavor:
    name = ""
    priority = 0
    exclusive_groups: tuple[str, ...] = ()

    def configure(self, _policy: FlavorPolicy) -> None:
        pass

    def transform(self, _opts: list[str], _env: dict[str, str]) -> None:
        pass


class PerfFlavor(Flavor):
    name = "perf"
    exclusive_groups = ("execution-mode",)

    def configure(self, policy: FlavorPolicy) -> None:
        policy.worker = self.run
        policy.requires_nvrx_absent = True
        policy.reporter = self.report

    def run(self, cfg: str, args: argparse.Namespace, provider: str) -> None:
        _run_routine(cfg, args, provider, async_checkpoint=args.nodes >= 16)

    def report(self, args: argparse.Namespace, log: Path) -> bool:
        cfg, tag, stem = _identity(args)
        return PerfReport.parse_and_report(
            args.job,
            cfg,
            tag,
            log,
            args.baseline if hasattr(args, "baseline") else PerfReport.baseline(stem),
            log.with_name(f"{stem}_report.json"),
            args.gpus_per_node,
        )

    def transform(self, opts: list[str], _env: dict[str, str]) -> None:
        if "--cuda-graph-impl" in opts and "--recompute-granularity" in opts:
            _drop(opts, "--recompute-granularity", "--recompute-modules")
            if "--async-save" in opts:
                _set(opts, "--async-ckpt-use-cpu-shm")


class DeterministicFlavor(Flavor):
    name = "deterministic"
    exclusive_groups = ("execution-mode",)

    def configure(self, policy: FlavorPolicy) -> None:
        policy.worker = self.run
        policy.reporter = self.report
        policy.requires_finite_run = True
        policy.checkpoint_routines = 2

    def run(self, cfg: str, args: argparse.Namespace, provider: str) -> None:
        _reference_check(cfg, args, provider, async_reference=provider == "native" and args.nodes >= 16)

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
            "--recompute-granularity",
            "--recompute-modules",
        )
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
    name = "valgrind"
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

    def run(self, cfg: str, args: argparse.Namespace, provider: str) -> None:
        if provider == "native":
            checkpoints = (
                _selected_flavors(args).policy.phase2_checkpoints
                if args.checkpoints_per_run is None
                else args.checkpoints_per_run
            )
            if checkpoints:
                args = argparse.Namespace(**vars(args))
                args.checkpoints_per_run, args.repeat = 1, checkpoints * args.repeat
        _run_routine(cfg, args, provider)

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        _drop(
            opts,
            "--cuda-graph-impl",
            "--cuda-graph-modules",
            "--recompute-granularity",
            "--recompute-modules",
        )
        # Memcheck can perturb model numerics enough to trip Megatron's result
        # validation and request a cross-GPU rerun (exit 16). This gate validates
        # the checkpoint server's memory safety, not training numerics.
        _set(opts, "--rerun-mode", "disabled")
        env["CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH"] = "1"

    @contextlib.contextmanager
    def server(self, stem: str, phase: int) -> Generator[None, None, None]:
        host = socket.gethostname().split(".")[0]
        marker = RESULTS_DIR / f".{os.environ.get('SLURM_JOB_ID', 'local')}-{stem}-{host}"
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


_FLAVORS = {flavor.name: flavor for flavor in (PerfFlavor(), DeterministicFlavor(), ValgrindFlavor())}
_DEFAULT_FLAVOR = _FLAVORS[PerfFlavor.name]


class FlavorSelection:
    def __init__(self, names: list[str] | tuple[str, ...] | None = None):
        selected = [_FLAVORS[name] for name in names] if names else [_DEFAULT_FLAVOR]
        self.items = tuple(sorted({flavor.name: flavor for flavor in selected}.values(), key=lambda flavor: (flavor.priority, flavor.name)))
        owners: dict[str, Flavor] = {}
        for flavor in self.items:
            for group in flavor.exclusive_groups:
                if previous := owners.get(group):
                    raise ValueError(f"{previous.name} and {flavor.name} are mutually exclusive")
                owners[group] = flavor
        self.policy = FlavorPolicy()
        for flavor in self.items:
            flavor.configure(self.policy)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(flavor.name for flavor in self.items)

    @property
    def tag(self) -> str:
        return "-".join(self.names)

    def transform(self, opts: list[str], env: dict[str, str]) -> None:
        for flavor in self.items:
            flavor.transform(opts, env)


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
    return int(table.get(flavors.tag, table.get("default", default)))

def _job_partition(config_name: str, flavors: FlavorSelection) -> str | None:
    table = (_model_config(config_name).get("job_partition") or {}).get(cluster_tag()) or {}
    value = str(table.get(flavors.tag, table.get("default", ""))).strip()
    return value or None

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

def _train_options(cfg: str, config: TrainConfig) -> list[str]:
    run = _run_dir(cfg)
    phase1 = run / "phase1-checkpoints"
    total_iters = config.save_interval * config.checkpoints if config.checkpoints else None
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
        "ckpt-provider": config.provider,
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

def _node_once(tag: str, action: Callable[[], None]) -> None:
    host = socket.gethostname().split(".")[0]
    marker = RESULTS_DIR / f".{os.environ.get('SLURM_JOB_ID', 'local')}-{tag}-{host}"
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
    marker = RESULTS_DIR / f".{os.environ.get('SLURM_JOB_ID', 'local')}-{tag}"
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

def _phase_command(
    cfg: str,
    args: argparse.Namespace,
    provider: str,
    phase: int,
    cycle: int = 0,
    *,
    async_checkpoint: bool = False,
) -> tuple[list[str], dict]:
    flavors = _selected_flavors(args)
    config = flavors.policy.train_config(cfg, args, provider, phase, cycle)
    opts, env = _train_options(cfg, config), {**os.environ, **_base_env()}
    # Deterministic transforms disable async options for normal routines. Preserve
    # a model's CPU-shm opt-in when the caller explicitly restores async checkpointing.
    async_ckpt_use_cpu_shm = "--async-ckpt-use-cpu-shm" in opts
    flavors.transform(opts, env)
    if async_checkpoint:
        _set(opts, "--async-save")
        _set(opts, "--async-strategy", "nvrx")
        _set(opts, "--use-persistent-ckpt-worker")
        if async_ckpt_use_cpu_shm:
            _set(opts, "--async-ckpt-use-cpu-shm")
    expert = opts[opts.index("--expert-model-parallel-size") + 1]
    env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = expert
    if "--fine-grained-activation-offloading" in opts:
        env["NVTE_CPU_OFFLOAD_V1"] = "1"
    os.environ["NATIVE_CKPT_SERVER_LOG"] = env["NATIVE_CKPT_SERVER_LOG"]
    return ["python3", "-u", str(Path(__file__).resolve()), _PRETRAIN_ENTRY_CMD, str(phase), *opts, *args.extra_args], env
def _print_pretrain_env(env: dict[str, str], cycle_tag: str) -> None:
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return
    print(f"[slurm_job:env] {cycle_tag} rank0 environment", flush=True)  # noqa: T201
    for name in sorted(_RUN_ENV_NAMES | _PRETRAIN_ENV_NAMES | _TRAIN_ENV_DEFAULTS.keys()):
        if name in env:
            print(f"[slurm_job:env] {name}={shlex.quote(env[name])}", flush=True)  # noqa: T201
def _run_routine(
    cfg: str,
    args: argparse.Namespace,
    provider: str,
    label: str = "",
    async_checkpoint: bool = False,
) -> None:
    flavors = _selected_flavors(args)
    stem = f"{platform_tag()}_{args.job}_{flavors.tag}_{cfg}"
    phase, cycle = 1, 0
    while phase == 1 or args.repeat == 0 or cycle < args.repeat:
        cmd, env = _phase_command(cfg, args, provider, phase, cycle, async_checkpoint=async_checkpoint)
        log = RESULTS_DIR / f"{cfg}_{label}-routine_rank{os.environ.get('SLURM_PROCID', '0')}.log" if label else None
        cycle_tag = "phase1" if phase == 1 else f"cycle{cycle + 1}"
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
                _run_dir(cfg) / f"phase1-checkpoints/iter_{_PHASE1_TRAIN_ITERS:07d}/.metadata",
                _time_budget(args.config, flavors, "phase1-flush", 30) * 60,
            )
            phase = 2
        else:
            cycle += 1
        flavors.policy.stop_server(provider, f"{label}-{cycle_tag}-stop")
        _sync_ranks(f"{label}-{cycle_tag}-stopped".strip("-"))
def _manifest(iter_dir: Path, baseline: Path | None = None) -> int:
    env = {**os.environ, "TORCHINDUCTOR_WORKER_START": "fork", "TORCHINDUCTOR_COMPILE_THREADS": "1"}
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
            done_f = RESULTS_DIR / f".{job_id}-{tag}-node{node_index}.done"
            fail_f = RESULTS_DIR / f".{job_id}-{tag}-node{node_index}.failed"
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
        done_f = RESULTS_DIR / f".{job_id}-{tag}-node{node_idx}.done"
        fail_f = RESULTS_DIR / f".{job_id}-{tag}-node{node_idx}.failed"
        if fail_f.exists() or not done_f.exists():
            failures.extend(str(c[1]) for c in assigned)
    return sorted(failures)

def _reference_check(  # noqa: C901
    cfg: str, args: argparse.Namespace, provider: str, *, async_reference: bool = False,
) -> None:
    run, reference = _run_dir(cfg), _run_dir(cfg) / "reference-run"
    ready, failed = run / "reference.ready", run / "reference.failed"
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    rank0 = rank == 0

    if rank0:
        shutil.rmtree(reference, ignore_errors=True)
        ready.unlink(missing_ok=True)
        failed.unlink(missing_ok=True)

    _run_routine(cfg, args, "nvrx", "reference", async_checkpoint=async_reference)

    ref_iters = [
        (group, iter_dir)
        for group in ("phase1-checkpoints", "checkpoints")
        for iter_dir in sorted((run / group).glob("iter_*"))
    ]

    ref_failures = _manifest_batch("ref-manifest", ref_iters, None, reference, args)

    if ref_failures:
        if rank0:
            failed.touch()
        raise RuntimeError(f"manifest derivation failed: {', '.join(ref_failures)}")

    if rank0:
        try:
            for group, iter_dir in ref_iters:
                target = reference / group / iter_dir.name
                for name in ("reference.metadata.info", "common_pt_dump.info"):
                    if not (target / name).exists():
                        raise RuntimeError(f"missing reference file: {target / name}")  # noqa: TRY301
            for group in ("phase1-checkpoints", "checkpoints"):
                shutil.rmtree(run / group)
            ready.touch()
        except BaseException:
            failed.touch()
            raise
    else:
        wait_for_file(run / "reference.*", 7200)
        if failed.exists():
            raise RuntimeError("reference routine failed")

    _run_routine(cfg, args, provider, f"measured-{provider}")

    meas_iters = [
        (group, iter_dir)
        for group in ("phase1-checkpoints", "checkpoints")
        for iter_dir in sorted((run / group).glob("iter_*"))
    ]

    compare_failures = _manifest_batch("meas-manifest", meas_iters, reference, None, args)
    if compare_failures:
        raise RuntimeError(f"reference check failed: {', '.join(compare_failures)}")

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
    start = time.time()
    set_startup_timestamps(program_start=start, main_entry=start)
    ph.train_valid_test_datasets_provider.is_distributed = True
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(
        extra_args_provider=megatron_integration.compose_extra_args_provider(add_modelopt_args),
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    model_cfg = hybrid_config_from_args(args)
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
        if rank0:
            import resource  # noqa: I001, PLC0415
            import torch  # noqa: PLC0415

            print(  # noqa: T201
                f"[native_ckpt:mem] gpu_peak_bytes={torch.cuda.max_memory_allocated()} "
                f"cpu_rss_bytes={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}",
                flush=True,
            )

def _worker(args: argparse.Namespace) -> None:
    provider = args.job.split("-", 1)[1]
    flavors = _selected_flavors(args)
    if provider == "native" and flavors.policy.requires_nvrx_absent:
        uninstall_nvrx_for_native()
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    if target := _nvrx_target(cfg, args.nvrx_commit):
        sys_path_insert(target)
    CHECKPOINT_SERVER.require_stopped()
    try:
        flavors.policy.run_worker(cfg, args, provider)
    except BaseException:
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
        if flavors.policy.manages_native_server and provider == "native" and local_rank == 0:
            with contextlib.suppress(Exception):
                CHECKPOINT_SERVER.request("stop", 5)
        raise

def _identity(args: argparse.Namespace) -> tuple[str, str, str]:
    cfg = f"{args.config}_{args.nodes}x{args.gpus_per_node}"
    flavor = _selected_flavors(args).tag
    return cfg, flavor, f"{platform_tag()}_{args.job}_{flavor}_{cfg}"

def _build_job(args: argparse.Namespace) -> tuple[str, str, Path]:
    flavors = _selected_flavors(args)
    cfg, flavor, stem = _identity(args)
    log = RESULTS_DIR / f"{stem}.log"
    minutes = args.time_minutes or _time_budget(args.config, flavors, "srun", 30)
    partition = _job_partition(args.config, flavors)
    work_minutes = minutes - _CLEANUP_GRACE_MINUTES
    job_name = RUN_NAME or f"{args.job}_{flavor}_{cfg}"
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
    prep_commands = [
        ["bash", str(ENROOT_SCRIPTS_DIR / "megatron_root.sh")],
        [
            "python3", "-u", str(Path(wheel_install.__file__).resolve()),
            *flavors.policy.prep_args(),
        ],
    ]
    cfg_target = _nvrx_target(cfg, args.nvrx_commit)
    if cfg_target:
        prep_commands.append([
            "python3", "-u", str(SCRIPT_DIR / "install_run_dependency.py"),
            "--repo-url", args.nvrx_repo_url,
            *(("--commit", args.nvrx_commit) if args.nvrx_commit else ()),
            "--target", str(cfg_target),
            *(("--repo-root", str(nvrx_repo_root())) if nvrx_repo_root() else ()),
        ])
    prep_commands.append(["python3", "-m", "pip", "list"])
    prep_commands.append(["python3", "-u", str(Path(wheel_install.__file__).resolve()), "--report-install"])
    prep = [*step(1, 1, ",".join(_env_names("p")), prep_log, writable=True),
            "/bin/bash", "-c", " && ".join(shell_join(command) for command in prep_commands)]
    worker_args = [
        args.job, "--nodes", str(args.nodes), "--gpus-per-node", str(args.gpus_per_node), "--config", args.config,
        *(token for name in flavors.names for token in ("--flavor", name)),
    ]
    epilog_args = [*worker_args]
    worker_args.extend(args.extra_args)
    container_env = ",".join(_env_names("w"))
    native_perf_verification = args.job == "pretrain-native" and flavors.policy.requires_nvrx_absent
    run = [*step(args.nodes, total_tasks, container_env, writable=native_perf_verification),
           "python3", "-u", str(Path(__file__).resolve()), *worker_args]
    epilog = [
        "python3", "-u", str(Path(__file__).resolve()), _EPILOG_ENTRY_CMD, *epilog_args,
        "--log-file", str(log), "--baseline-json", json.dumps(PerfReport.baseline(stem), separators=(",", ":")),
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
        shell_join(run),
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
    "nodes": RunOption("SLURMJOB_OPT_NODES", 1, _positive_int),
    "config": RunOption("SLURMJOB_OPT_CONFIG", "pico", choices=_CONFIGS),
    "gpus_per_node": RunOption("GPUS_PER_NODE", None, _positive_int),
    "flavor": RunOption("SLURMJOB_OPT_FLAVOR", choices=tuple(sorted(_FLAVORS)), action="append"),
    "name": RunOption("SLURMJOB_OPT_RUN_NAME", scopes="rspw"),
    "work_dir": RunOption("SLURMJOB_OPT_WORK_DIR", convert=_path, scopes="rspw"),
    "image": RunOption("ENROOT_IMAGE", convert=Path, scopes="rsw"),
    "megatron_commit": RunOption("MEGATRON_COMMIT", "77c87725e71c9f5022ac289071b16941876ca858", _commit, scopes="rspw"),
    "megatron_repo_url": RunOption("MEGATRON_REPO_URL", "https://github.com/NVIDIA/Megatron-LM.git", _url, scopes="rspw"),
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
    "SLURMJOB_OPT_LAUNCHER_ROOT": "rs", "SLURMJOB_OPT_RUN_DIR": "rsw", "SLURM_ACCOUNT": "rs", "PYTHONPATH": "rpw",
    "CI_API_V4_URL": "rp", "CI_COMMIT_SHA": "rp", "CI_PROJECT_ID": "rp", "WHEEL_ARCH": "rp", "CI_JOB_TOKEN": "p",
    "CI_DIR": "r", "CLUSTER": "r", "DOCKER_IMAGE_BASE": "r", "LD_LIBRARY_PATH": "r",
    "PATH": "r", "PLATFORM": "r",
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
    parser.add_argument("job", choices=["pretrain-native", "pretrain-nvrx"])
    for dest, option in _RUN_OPTIONS.items():
        kwargs: dict[str, object] = {"dest": dest, "default": argparse.SUPPRESS}
        kwargs["action" if option.action else "type"] = option.action or option.convert
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
        values = raw.split(",") if raw and option.action == "append" else [raw]
        try:
            converted = [option.convert(value.strip()) for value in values if value and value.strip()]
        except (argparse.ArgumentTypeError, TypeError, ValueError) as error:
            parser.error(f"{option.env} has invalid value: {error}")
        value = converted if option.action == "append" and converted else converted[0] if converted else option.default
        if option.choices and any(item not in option.choices for item in ((value or []) if option.action == "append" else [value])):
            parser.error(f"{option.env} must be one of: {', '.join(option.choices)}")
        setattr(args, dest, value)
    return args, extra

def _epilog(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--baseline-json", required=True)
    args, _extra = _parse_args(parser, argv)
    args.flavors = FlavorSelection(args.flavor)
    args.baseline = json.loads(args.baseline_json)
    text = args.log_file.read_text(errors="replace")
    saves = _SAVE_RE.findall(text)
    phase1 = sum("/phase1-checkpoints" in path for path in saves)
    phase2 = sum("/checkpoints" in path and "/phase1-checkpoints" not in path for path in saves)
    routines = args.flavors.policy.checkpoint_routines
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
    _write_baseline_artifacts(args, args.log_file)
    if not passed:
        raise SystemExit(SOFT_FAIL_RC)

def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.gpus_per_node is None:
        parser.error("--gpus-per-node or GPUS_PER_NODE is required")
    if args.time_minutes is not None and args.time_minutes <= _CLEANUP_GRACE_MINUTES:
        parser.error(f"--time-minutes must exceed the {_CLEANUP_GRACE_MINUTES}-minute cleanup reserve")
    if args.checkpoints_per_run == 0 and args.repeat != 1:
        parser.error("an unlimited checkpoint run cannot be repeated because it does not return")
    if args.flavors.policy.requires_finite_run and (args.checkpoints_per_run == 0 or args.repeat == 0):
        parser.error("the selected flavor requires finite checkpoint and repeat counts")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser()
    args, args.extra_args = _parse_args(parser)
    try:
        args.flavors = FlavorSelection(args.flavor)
    except ValueError as error:
        parser.error(str(error))
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
    rc = sbatch_run(
        script, log,
        extra_sbatch_args=[reaper],
        background=args.background,
        dry_run=args.dry_run,
    )
    if rc == SOFT_FAIL_RC:
        raise SystemExit(rc)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _PRETRAIN_ENTRY_CMD:
        phase = int(sys.argv.pop(2))
        sys.argv = ["pretrain_hybrid.py", *sys.argv[2:]]
        _pretrain_entry(phase)
    elif len(sys.argv) > 1 and sys.argv[1] == _EPILOG_ENTRY_CMD:
        _epilog(sys.argv[2:])
    else:
        main()
