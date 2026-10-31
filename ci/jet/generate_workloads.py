# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001
"""Build the JET image and recipe manifests for the workloads in ci/jobs.json."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

CI_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = CI_DIR / "configs"
sys.path.insert(0, str(CI_DIR))

from workload_matrix import (  # noqa: E402
    RUNTIME_CACHE_MOUNT,
    JetPlatform,
    jet_cluster,
    platform_profile,
)

DEFAULT_TIMEOUT_SECONDS = 8 * 60 * 60
LARGE_CHECKPOINT_MIN_NODES = 16
# The submitting user's read-only GitHub token downloads the upstream wheel artifacts.
WHEEL_TOKEN_SECRETS = {"NVRX_GITHUB_TOKEN": "user/<current>/github/nvrx-ci-artifacts"}
# The native runtime cache follows prepare.sh, so editing that file forces a rebuild.
RUNTIME_KEY = hashlib.sha256((CI_DIR.parent / "scripts" / "prepare.sh").read_bytes()).hexdigest()[:10]
# JET retries an attempt only for platform-classified infrastructure failures.
RETRIER_CONFIG = {
    "retrier": {
        "enabled": True,
        "environment": "jet-auto-retrier",
        "default_retry_on": ["1.2", "1.2.*"],
        "default_max_retries": 1,
        "waiting_time": 60,
    },
    "executors": {
        "jet-ci": {
            "environments": {
                "jet-auto-retrier": {
                    "tags": [
                        "purpose/auto-retrier",
                        "arch/amd64",
                        "env/prod",
                        "owner/jet-core",
                        "origin/jet-fleet",
                    ],
                    "image": "gitlab-master.nvidia.com/dl/jet/api:v3",
                    "before_script": [
                        'jet secrets jwt-login jwt/nvidia/gitlab-master jet-ci-ci "$(printenv VAULT_JWT_TOKEN)"',
                    ],
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Workload:
    """One Slurm runner invocation that JET runs on a platform."""

    platform: str
    provider: str
    config: str
    flavors: tuple[str, ...]
    runs: tuple[str, ...]
    nodes: int
    gpus_per_node: int
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    pipeline_suite: str = "optional"
    checkpoint_interval: int | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def flavor_tag(self) -> str:
        return "-".join(self.flavors)

    @property
    def scale(self) -> str:
        return f"{self.nodes}x{self.gpus_per_node}"

    @property
    def job_args(self) -> list[str]:
        args = [
            "--nodes",
            str(self.nodes),
            "--gpus-per-node",
            str(self.gpus_per_node),
            "--config",
            self.config,
            "--flavor",
            "+".join(self.flavors),
            *self.extra_args,
            *(token for run in self.runs for token in ("--run", run)),
        ]
        if "perf" in self.flavors and self.checkpoint_interval is not None:
            args.extend(["--checkpoint-interval", str(self.checkpoint_interval)])
        return args

    @property
    def safe_name(self) -> str:
        value = f"{self.platform}-{self.provider}-{self.flavor_tag}-{self.config}-{self.scale}"
        return re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower()

    @property
    def display_name(self) -> str:
        """Short, scan-friendly JET key without changing runtime identities."""
        mode = "-".join(
            flavor for flavor in self.flavors if flavor not in {"native", "nvrx", "async", "shm"}
        )
        value = f"{self.platform}-{self.provider}-{mode}-{self.config}-{self.scale}"
        return re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower()

    @property
    def slurm_time(self) -> str:
        days, remainder = divmod(self.timeout_seconds, 24 * 60 * 60)
        hours, remainder = divmod(remainder, 60 * 60)
        minutes, seconds = divmod(remainder, 60)
        clock = f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{days}-{clock}" if days else clock


def _timeout_seconds(value: str | None) -> int:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    match = re.fullmatch(r"(\d+)([dhm])", value)
    if match is None:
        raise ValueError(f"unsupported timeout {value!r}")
    factors = {"d": 24 * 60 * 60, "h": 60 * 60, "m": 60}
    return int(match[1]) * factors[match[2]]


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            _deep_merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def _config_time_budget_seconds(config: str, flavors: tuple[str, ...]) -> int:
    """Return the srun budget that the model config sets for these flavors."""
    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text())
    override = yaml.safe_load((CONFIGS_DIR / f"{config}.yaml").read_text())
    table = (_deep_merge(base, override).get("job_time_in_mins") or {}).get("srun") or {}
    flavor_tag = "-".join(flavors)
    execution_mode = next(
        (flavor for flavor in flavors if flavor in {"perf", "detr", "vg"}),
        None,
    )
    return int(table.get(flavor_tag, table.get(execution_mode, table.get("default", 30)))) * 60


def _apply_transport_args(platform: JetPlatform, workload: Workload) -> Workload:
    """Add the platform's collective timeout to workloads with enough nodes."""
    if platform.distributed_timeout_minutes is None:
        return workload
    if workload.nodes < (platform.distributed_timeout_min_nodes or 1):
        return workload
    return replace(
        workload,
        extra_args=(
            *workload.extra_args,
            "--distributed-timeout-minutes",
            str(platform.distributed_timeout_minutes),
        ),
    )


def resolved_workloads(platform: JetPlatform) -> list[Workload]:
    """Resolve every workload of a platform's profile in ci/jobs.json."""
    profile = platform_profile(platform.profile)
    workloads = []
    for provider, flavors in profile["jobs"]:
        selected = (flavors,) if isinstance(flavors, str) else tuple(flavors)
        runs = (
            ("ref:nvrx", f"cmp:{provider}")
            if selected[0] == "detr" and "vg" not in selected
            else ()
        )
        common = selected if runs else (*selected, provider)
        workloads.append(
            Workload(
                platform.name,
                provider,
                "pico",
                common,
                runs,
                1,
                platform.gpus_per_node,
                pipeline_suite=(
                    "main-branch-runs" if "vg" in selected else "nvrx" if provider == "nvrx" else "default"
                ),
            ),
        )
    for entry in profile["multinode"]:
        for provider, flavors in entry["jobs"]:
            selected = (flavors,) if isinstance(flavors, str) else tuple(flavors)
            common = selected
            if entry["nodes"] >= LARGE_CHECKPOINT_MIN_NODES and (
                staging := entry.get("staging", {}).get(selected[0], "async")
            ):
                common = (*common, staging)
            runs = (
                ("ref:nvrx", f"cmp:{provider}")
                if selected[0] == "detr" and "vg" not in selected
                else ()
            )
            if not runs:
                common = (*common, provider)
            workloads.append(
                Workload(
                    platform.name,
                    provider,
                    entry["config"],
                    common,
                    runs,
                    entry["nodes"],
                    platform.gpus_per_node,
                    timeout_seconds=_timeout_seconds(entry.get("timeout")),
                    pipeline_suite=(
                        ("main-branch-runs" if "vg" in selected else None)
                        or ("nvrx" if provider == "nvrx" else None)
                        or entry.get("pipeline_suite")
                        or "optional"
                    ),
                    checkpoint_interval=entry.get("checkpoint_interval") if "perf" in selected else None,
                ),
            )
    if platform.max_timeout_seconds is not None:
        workloads = [
            replace(workload, timeout_seconds=platform.max_timeout_seconds)
            if workload.nodes == 1 and workload.timeout_seconds > platform.max_timeout_seconds
            else workload
            for workload in workloads
        ]
    workloads = [_apply_transport_args(platform, workload) for workload in workloads]
    if platform.use_regular_time_budget:
        budgeted_workloads = []
        for workload in workloads:
            timeout_seconds = _config_time_budget_seconds(workload.config, workload.flavors)
            if platform.max_timeout_seconds is not None:
                timeout_seconds = min(timeout_seconds, platform.max_timeout_seconds)
            budgeted_workloads.append(replace(workload, timeout_seconds=timeout_seconds))
        workloads = budgeted_workloads
    return workloads


def select_workloads(platform: JetPlatform) -> tuple[list[Workload], list[Workload]]:
    """Split a platform's workloads into those its limits allow and those they exclude."""
    supported, unsupported = [], []
    for workload in resolved_workloads(platform):
        exceeds_nodes = platform.max_nodes is not None and workload.nodes > platform.max_nodes
        exceeds_time = (
            platform.max_timeout_seconds is not None
            and workload.timeout_seconds > platform.max_timeout_seconds
        )
        target = unsupported if exceeds_nodes or exceeds_time else supported
        target.append(workload)
    return supported, unsupported


def find_workload(platform: JetPlatform, key: str) -> Workload:
    """Return the supported workload with the given JET key."""
    workloads = {workload.display_name: workload for workload in select_workloads(platform)[0]}
    if key not in workloads:
        raise ValueError(f"{platform.name} has no JET workload {key!r}; known: {', '.join(sorted(workloads))}")
    return workloads[key]


def jet_custom_config(platform: JetPlatform) -> dict:
    """Return the JET generator config overrides for the platform's cluster."""
    cluster = jet_cluster(platform.name)
    config = _deep_merge(RETRIER_CONFIG, cluster.custom_config)
    if not cluster.cache:
        return config
    environment = config.get("executors", {}).get("jet-ci", {}).get("environments", {}).get(cluster.launcher, {})
    cache = {
        "executors": {
            "jet-ci": {
                "environments": {
                    cluster.launcher: {"before_script": [f"mkdir -p {cluster.cache}", *environment.get("before_script", [])]},
                },
            },
        },
        "launchers": {cluster.launcher: {"mounts": {RUNTIME_CACHE_MOUNT: cluster.cache}}},
    }
    return _deep_merge(config, cache)


def _environment(platform: JetPlatform, workload: Workload, assets_dir: str, shared_dir: str) -> str:
    values = {
        "NVRX_SOURCE_REVISION": "{env.UPSTREAM_COMMIT_SHA}",
        "PLATFORM": platform.name,
        "WHEEL_ARCH": platform.architecture,
        "MEGATRON_ROOT": "/opt/megatron-lm",
        "JET_PREPARED_MEGATRON_ROOT": "/opt/megatron-lm",
        "NVRX_MAX_JOBS": "16",
        "NVRX_DEEPEP_CUDA_ARCH_LIST": platform.compute_capability,
        "SLURMJOB_OPT_WORK_DIR": f"{shared_dir}/nvrx-ci-runs/{{env.UPSTREAM_PIPELINE_ID}}",
        "SLURMJOB_OPT_RUN_NAME": workload.safe_name,
        "SLURMJOB_OPT_RUN_DIR": (
            f"{shared_dir}/nvrx-ci-runs/{{env.UPSTREAM_PIPELINE_ID}}/{workload.safe_name}"
        ),
        # Single-task setup removes only this exact workload path.
        "SLURMJOB_OPT_CLEAN_RUN": "1",
        "SLURMJOB_OPT_CLEAN_REFERENCE_SERVER": "1",
        "JET_CLEAN_RUN_AFTER_FINALIZE": "1",
        # Pipeline scoping prevents stale assets from entering the archive.
        "CI_BASELINE_CANDIDATE_DIR": (
            f"{assets_dir}/nvrx-ci-artifacts/{{env.UPSTREAM_PIPELINE_ID}}/ci/baselines"
        ),
        "CI_ARTIFACT_ROOT": f"{assets_dir}/nvrx-ci-artifacts/{{env.UPSTREAM_PIPELINE_ID}}",
    }
    cache_dir = jet_cluster(platform.name).runtime_cache_dir
    if cache_dir:
        values["JET_RUNTIME_CACHE_DIR"] = cache_dir
    # A mounted cache directory persists between jobs; the shared directory lasts one job.
    runtime_root = f"{cache_dir or shared_dir}/nvrx-runtime/{platform.name}/{RUNTIME_KEY}"
    local_runtime_root = (
        "/tmp/nvrx-runtime/{env.UPSTREAM_PROJECT_ID}/"  # noqa: S108 - allocation-local cache
        f"{platform.name}/{RUNTIME_KEY}"
    )
    values.update(
        {
            "JET_RUNTIME_KEY": RUNTIME_KEY,
            "JET_RUNTIME_ROOT": runtime_root,
            "JET_RUNTIME_ARCHIVE": f"{runtime_root}/native-runtime.tar",
            "JET_LOCAL_RUNTIME_ROOT": local_runtime_root,
            "NVRX_VENV": f"{runtime_root}/venv",
            "VIRTUAL_ENV": f"{runtime_root}/venv",
        },
    )
    values["JET_PREBUILT_WHEEL_DIR"] = f"{shared_dir}/nvrx-ci-runs/{{env.UPSTREAM_PIPELINE_ID}}/wheels"
    exports = [
        *(f"export {name}={shlex.quote(value)}" for name, value in values.items()),
        "export NCCL_IB_TIMEOUT=24",
        # Compiler caches stay on node-local storage.
        'export TRITON_CACHE_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/triton-cache"',
        'export TRITON_CACHE_PATH="$TRITON_CACHE_DIR"',
        'export TORCHINDUCTOR_CACHE_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/torchinductor-cache"',
        'export TORCH_EXTENSIONS_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/torch-extensions"',
        # TensorBoard output stays on node-local storage.
        (
            'export SLURMJOB_OPT_TENSORBOARD_DIR="/tmp/nvrx-${{SLURM_JOB_ID:?}}/'
            'tensorboard/${{SLURMJOB_OPT_RUN_NAME:?}}"'
        ),
        'export PATH="$NVRX_VENV/bin:$PATH"',
    ]
    return "\n".join(exports)


def build_manifest(platform: JetPlatform, maintainer: str) -> dict:
    return {
        "type": "build",
        "format_version": 1,
        # Stable build identity keeps the pipeline label concise.
        "key_pattern": f"{platform.name}-ci-image",
        "maintainers": [maintainer],
        "labels": {
            "origin": "nvrx-checkpointing",
            "platform": platform.name,
        },
        "spec": {
            "name": f"nvrx-ci-{platform.name}",
            "platforms": [f"linux/{'amd64' if platform.architecture == 'x86_64' else 'arm64'}"],
            "source": {
                "repo": "{env.UPSTREAM_PROJECT_URL}.git",
                "ref": "{env.UPSTREAM_COMMIT_SHA}",
                "path": ".",
                "dockerfile": "ci/jet/image/Dockerfile",
                "arguments": {
                    "FROM_IMAGE_NAME": "nvcr.io/nvidia/pytorch:26.09-py3",
                    "NVRX_COMPUTE_CAPABILITY": platform.compute_capability,
                    "NVRX_SOURCE_REVISION": "{env.UPSTREAM_COMMIT_SHA}",
                },
                "submodules": {"update": None},
            },
        },
    }


def recipe_manifest(platform: JetPlatform, workload: Workload, maintainer: str, image_tag: str) -> dict:
    assets_dir = "{assets_dir}"
    environment = _environment(platform, workload, assets_dir, "{shared_dir}")
    args = shlex.join(workload.job_args)
    activate_runtime = "source ci/jet/image/activate_runtime.sh\n"
    native_setup = f"bash ci/jet/image/prepare_runtime.sh\n{activate_runtime}"
    # The allocation comment exempts exact-SHA wheel setup from the idle-GPU reaper.
    allocation_comment = (
        json.dumps(
            {
                "OccupiedIdleGPUsJobReaper": {
                    "exemptIdleTimeMins": str(workload.timeout_seconds // 60),
                    "reason": "other",
                    "description": "native-checkpoint-run",
                },
            },
            separators=(",", ":"),
        )
        .replace('"', '\\"')
        .replace("{", "{{")
        .replace("}", "}}")
    )
    allocation_flags = {
        "gpus_per_node": platform.gpus_per_node,
        "time": workload.slurm_time,
        "comment": allocation_comment,
    }
    segment = jet_cluster(platform.name).segment
    if segment and workload.nodes > 1:
        # Segments keep a multi-node allocation within the cluster's NVLink domains.
        allocation_flags["segment"] = min(workload.nodes, segment)
    if platform.name == "gb200" and workload.nodes >= LARGE_CHECKPOINT_MIN_NODES:
        # Large GB200 checkpoints serialize against the shared Lustre quota.
        allocation_flags.update(
            {
                # Revision scoping isolates singleton lanes from obsolete pipelines.
                "job_name": "nvrx-ci-gb200-large-{env.UPSTREAM_COMMIT_SHA}",
                "dependency": "singleton",
            },
        )
    return {
        "type": "recipe",
        "format_version": 1,
        "key_pattern": workload.display_name,
        "maintainers": [maintainer],
        "secrets": dict(WHEEL_TOKEN_SECRETS),
        "labels": {
            "origin": "nvrx-checkpointing",
            "platform": platform.name,
            "provider": workload.provider,
            "flavor": workload.flavor_tag,
        },
        "launchers": {
            "type:slurm": {
                "ntasks_per_node": platform.gpus_per_node,
                # Canonical Slurm clocks avoid launcher-specific duration rendering.
                "sbatch_additional_flags": allocation_flags,
                "srun_additional_flags": {
                    "time": workload.slurm_time,
                    # The keyed value renders as one Slurm argument.
                    "kill_on_bad_exit=1": True,
                },
            },
        },
        "spec": {
            "model": "nvrx-checkpointing",
            "variant": workload.safe_name,
            "scope": f"{workload.provider}-{workload.flavor_tag}",
            "image_source": {"image_tag": image_tag},
            "precision": workload.config,
            "platforms": [],
            "nodes": workload.nodes,
            "gpus": platform.gpus_per_node,
            "batch_size": 1,
            "workspace": "/workspace/nvrx_checkpointing",
            "time_limit": None,
            "batch_setup": (
                f"set -Eeuo pipefail\n{environment}\n{native_setup}python3 -u ci/jet/prepare_workload.py {args}\n"
            ),
            "script": (
                f"set -Eeuo pipefail\n{environment}\n"
                "export SLURMJOB_OPT_ISOLATE_PHASE_SERVER_PORTS=1\n"
                f"export SLURMJOB_OPT_ROUTINE_LABEL=jet\n{activate_runtime}"
                f"python3 -u ci/jet/run_workload.py --assets-dir {shlex.quote(assets_dir)} -- {args}\n"
            ),
            "script_teardown": (
                f"set +e\n{environment}\n"
                f"python3 -u ci/jet/finalize_workload.py --assets-dir {shlex.quote(assets_dir)} -- {args}\n"
                "finalize_rc=$?\n"
                "python3 -u ci/scripts/daemon_tools.py\n"
                "exit $finalize_rc\n"
            ),
            "batch_teardown": (
                f"set +e\n{environment}\n"
                "python3 -u ci/scripts/daemon_tools.py\n"
                "daemon_rc=$?\n"
                f"python3 -u ci/jet/cleanup_workload.py --assets-dir {shlex.quote(assets_dir)} -- {args}\n"
                "cleanup_rc=$?\n"
                "(( daemon_rc != 0 )) && exit $daemon_rc\n"
                "exit $cleanup_rc\n"
            ),
        },
    }
