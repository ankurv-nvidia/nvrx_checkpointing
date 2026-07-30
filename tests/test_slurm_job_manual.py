# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: ANN202, INP001, PLR0915, S101, S607, SLF001
from __future__ import annotations

import importlib.util
import io
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "ci" / "scripts"
_MUTATED_ENV = (
    "CI_JOB_ID",
    "ENROOT_IMAGE",
    "ENROOT_IMAGE_DIR",
    "MEGATRON_COMMIT",
    "MEGATRON_NATIVE_BUILD_DIR",
    "MEGATRON_REPO_URL",
    "MEGATRON_ROOT",
    "MASTER_PORT",
    "NATIVE_CKPT_SERVER_LOG",
    "NCCL_IB_TIMEOUT",
    "NVTE_BWD_LAYERNORM_SM_MARGIN",
    "NVTE_FWD_LAYERNORM_SM_MARGIN",
    "PYTORCH_CUDA_ALLOC_CONF",
    "QUANTIZATION_TYPE_DEBUG",
    "SLURMJOB_OPT_BACKGROUND",
    "SLURMJOB_OPT_CHECKPOINTS_PER_RUN",
    "SLURMJOB_OPT_CLEAN_RUN",
    "SLURMJOB_OPT_CONFIG",
    "SLURMJOB_OPT_DRY_RUN",
    "SLURMJOB_OPT_FLAVOR",
    "SLURMJOB_OPT_NODES",
    "NVRX_COMMIT",
    "SLURMJOB_OPT_LAUNCHER_ROOT",
    "SLURMJOB_OPT_PYTHON_INSTALL_DIR",
    "NVRX_REPO_ROOT",
    "NVRX_REPO_URL",
    "SLURMJOB_OPT_REPEAT",
    "SLURMJOB_OPT_RUN_DIR",
    "SLURMJOB_OPT_RUN_NAME",
    "SLURMJOB_OPT_TIME_MINUTES",
    "SLURMJOB_OPT_WORK_DIR",
    "GPUS_PER_NODE",
    "SLURM_SEGMENT",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_WORKER_START",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_CACHE_PATH",
    "UCX_ERROR_SIGNALS",
    "UCX_MEM_CUDA_HOOK_MODE",
    "UCX_MEM_MALLOC_HOOKS",
    "UCX_MEM_MMAP_HOOK_MODE",
    "VCPKG_ROOT",
    "CUDA_DEVICE_MAX_CONNECTIONS",
)


yaml.SafeLoader.add_constructor("!reference", lambda loader, node: loader.construct_sequence(node))


def _load_slurm_job():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("slurm_job_manual_test", SCRIPTS_DIR / "slurm_job.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prepare_image():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("prepare_image_manual_test", SCRIPTS_DIR / "prepare_image.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_run_dependency():
    spec = importlib.util.spec_from_file_location("install_run_dependency_test", SCRIPTS_DIR / "install_run_dependency.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_once_waits_for_local_peers_before_action(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results)
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node.example")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "2(x4)")
    marker = results / ".123-phase1-stop-node"
    marker.with_name(f"{marker.name}.ready-1").touch()
    calls = []

    slurm_job._node_once("phase1-stop", lambda: calls.append("stop"))

    assert calls == ["stop"]
    assert marker.exists()


def test_sync_ranks_waits_for_global_peers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    marker = results / ".123-phase1"
    marker.with_name(f"{marker.name}.ready-1").touch()

    slurm_job._sync_ranks("phase1")

    assert marker.with_name(f"{marker.name}.ready-0").exists()


def test_server_hard_stop_is_a_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "checkpoint_server.node.42.log"
    log.write_text("[ERROR] spindown timed out before idle; hard-stopping (aborting in-flight work)\n")
    monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(tmp_path / "checkpoint_server.%h.%p.log"))
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node")
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(pids=lambda: ["42"], stop=lambda _graceful: None),
    )
    monkeypatch.setattr(slurm_job, "_node_once", lambda _tag, action: action())

    with pytest.raises(RuntimeError, match="aborted in-flight work"):
        slurm_job.FlavorPolicy().stop_server("native", "phase2")


def test_valgrind_server_uses_phase_in_log_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(slurm_job.shutil, "which", lambda _name: "/usr/bin/valgrind")
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", SimpleNamespace(require_stopped=lambda: None))
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing", SimpleNamespace(SERVER_PATH=tmp_path / "server"))
    monkeypatch.setattr(slurm_job.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("popen reached")))

    with pytest.raises(RuntimeError, match="popen reached"), slurm_job.ValgrindFlavor().server("test", 2):
        pass


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "name": "ultra3-64n",
        "work_dir": tmp_path,
        "image": None,
        "megatron_commit": None,
        "megatron_repo_url": None,
        "nvrx_commit": None,
        "nvrx_repo_url": "https://github.com/NVIDIA/nvidia-resiliency-ext.git",
        "checkpoints_per_run": None,
        "repeat": 1,
        "time_minutes": None,
        "clean_run": False,
        "background": False,
        "dry_run": False,
        "job": "pretrain-native",
        "nodes": 64,
        "gpus_per_node": 4,
        "config": "ultra3fp8",
        "flavors": ["perf"],
        "extra_args": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _clear_run_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MUTATED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_run_options_apply_default_env_cli_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    defaults, _ = slurm_job._parse_args(slurm_job._parser(), ["pretrain-native"])
    assert (defaults.nodes, defaults.config, defaults.repeat, defaults.background) == (1, "pico", 1, False)
    assert defaults.flavor is None

    monkeypatch.setenv("SLURMJOB_OPT_NODES", "8")
    monkeypatch.setenv("SLURMJOB_OPT_CONFIG", "nano3")
    monkeypatch.setenv("SLURMJOB_OPT_FLAVOR", "perf,deterministic")
    monkeypatch.setenv("SLURMJOB_OPT_BACKGROUND", "1")
    from_env, _ = slurm_job._parse_args(slurm_job._parser(), ["pretrain-native"])
    assert (from_env.nodes, from_env.config, from_env.flavor, from_env.background) == (
        8, "nano3", ["perf", "deterministic"], True,
    )

    cli, _ = slurm_job._parse_args(
        slurm_job._parser(),
        ["pretrain-native", "--nodes", "16", "--config", "nano4", "--flavor", "valgrind", "--no-background"],
    )
    assert (cli.nodes, cli.config, cli.flavor, cli.background) == (16, "nano4", ["valgrind"], False)


def test_run_option_environment_values_use_the_cli_validator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURMJOB_OPT_NODES", "0")

    with pytest.raises(SystemExit):
        slurm_job._parse_args(slurm_job._parser(), ["pretrain-native"])
    assert "SLURMJOB_OPT_NODES has invalid value" in capsys.readouterr().err


def test_build_environment_exports_run_wheel_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    install_dir = tmp_path / "run/deps/python"
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(install_dir)])
    monkeypatch.setattr(slurm_job, "RUN_ROOT", tmp_path / "run")
    monkeypatch.setenv("PYTHONPATH", "/existing")

    slurm_job.ensure_vcpkg_root()

    assert sys.path[0] == str(install_dir)
    assert slurm_job.os.environ["PYTHONPATH"] == f"{install_dir}:/existing"
    assert slurm_job.os.environ["SLURMJOB_OPT_PYTHON_INSTALL_DIR"] == str(install_dir)


def test_shell_join_keeps_container_root_literal_when_home_is_root(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("ENROOT_IMAGE_DIR", "/root/data")

    command = slurm_job.shell_join(["srun", "--container-mounts=/root:/root,/root/data:/root"])

    assert '"--container-mounts=${HOME}:/root,${ENROOT_IMAGE_DIR}:/root"' in command


def test_named_run_resolves_all_durable_paths_under_workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    slurm_job._configure_run(_args(tmp_path))
    slurm_job.ensure_vcpkg_root()

    expected = tmp_path / "ultra3-64n"
    assert expected == slurm_job.RUN_ROOT
    assert expected / "logs" == slurm_job.RESULTS_DIR
    assert slurm_job._run_dir("ultra3fp8_64x4") == expected
    assert Path(slurm_job.os.environ["SLURMJOB_OPT_RUN_DIR"]) == expected
    assert Path(slurm_job.os.environ["MEGATRON_ROOT"]) == expected / "deps" / "megatron-lm"
    assert Path(slurm_job.os.environ["MEGATRON_NATIVE_BUILD_DIR"]) == expected / "build" / "nvrx_checkpointing"
    assert Path(slurm_job.os.environ["SLURMJOB_OPT_PYTHON_INSTALL_DIR"]) == expected / "deps" / "python"
    assert Path(slurm_job.os.environ["VCPKG_ROOT"]) == expected / "deps" / "vcpkg"


def test_unnamed_runs_generate_commit_keyed_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    commit = "1" * 40
    monkeypatch.setenv("CI_JOB_ID", "98765")
    native = subprocess.check_output(
        ["git", "-C", str(slurm_job.REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()[:8]

    slurm_job._configure_run(_args(tmp_path, name=None, work_dir=None, megatron_commit=commit))

    name = f"98765_pretrain-native_perf_ultra3fp8_64n_mlm.{commit[:8]}_nvrx.image_nckpt.{native}"
    assert name == slurm_job.RUN_NAME
    assert slurm_job.REPO_ROOT / "workdir" / name == slurm_job.RUN_ROOT
    assert Path(slurm_job.os.environ["MEGATRON_ROOT"]) == slurm_job.RUN_ROOT / "deps/megatron-lm"


def test_run_name_template_expands_before_paths_are_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(
        tmp_path,
        name="{job_name}_{config}_{nodes}n_mlm.{mlm_abbreviated_sha_commit}_nvrx.{nvrx_abbreviated_sha_commit}",
        megatron_commit="1" * 40,
        nvrx_commit="a" * 40,
    )

    slurm_job._configure_run(args)

    expected = "pretrain-native_perf_ultra3fp8_64n_mlm.11111111_nvrx.aaaaaaaa"
    assert expected == slurm_job.RUN_NAME
    assert tmp_path / expected == slurm_job.RUN_ROOT
    assert slurm_job.os.environ["SLURMJOB_OPT_RUN_NAME"] == expected


def test_gitlab_job_id_template_prefix_is_optional(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(tmp_path, name="{gitlab_pipeline_jobid}_{config}")

    slurm_job._configure_run(args)

    assert slurm_job.RUN_NAME == "ultra3fp8"


def test_nvrx_repo_root_selects_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    nvrx = tmp_path / "sources" / "nvidia-resiliency-ext"
    monkeypatch.setenv("NVRX_REPO_ROOT", str(nvrx))
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path)

    slurm_job._configure_run(args)
    script, _reaper, _log = slurm_job._build_job(args)
    slurm_job.sbatch_run(script, slurm_job.RESULTS_DIR / "scheduler.log", dry_run=True)
    reproducer = (tmp_path / "ultra3-64n" / "run.sbatch").read_text()

    assert slurm_job.nvrx_repo_root() == nvrx
    assert slurm_job.wheel_install.REPO_ROOT == slurm_job.REPO_ROOT
    assert "SLURMJOB_OPT_NATIVE_CKPT_REPO_ROOT" not in reproducer
    assert "export NVRX_REPO_ROOT=" + str(nvrx) in reproducer
    assert '--repo-root "${NVRX_REPO_ROOT}"' in reproducer
    assert '--target "${SLURMJOB_OPT_RUN_DIR}/deps/nvidia-resiliency-ext"' in reproducer

def test_nvrx_repo_root_reuses_existing_checkout_without_switching_it(tmp_path: Path) -> None:
    installer = _load_run_dependency()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    tracked = source / "tracked"
    tracked.write_text("one")
    subprocess.run(["git", "add", "tracked"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=source, check=True)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    tracked.write_text("two")
    subprocess.run(["git", "commit", "-qam", "two"], cwd=source, check=True)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    checkout = tmp_path / "nvidia-resiliency-ext"

    requirement = installer._source_requirement(str(source), first, checkout)
    assert requirement == f"git+{checkout.as_uri()}@{first}"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == first

    installer._source_requirement(str(source), second, checkout)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == first


def test_named_run_rejects_path_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    with pytest.raises(SystemExit, match="--name must expand"):
        slurm_job._configure_run(_args(tmp_path, name="../escape"))


def test_long_run_checkpoint_controls_resume_latest_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(tmp_path, checkpoints_per_run=4, repeat=2)
    slurm_job._configure_run(args)
    policy = slurm_job.FlavorSelection(["perf"]).policy
    cfg = "nano3_4x4"

    first = policy.train_config(cfg, args, "native", 2, 0)
    second = policy.train_config(cfg, args, "native", 2, 1)
    bootstrap = policy.train_config(cfg, args, "native", 1)
    opts = slurm_job._train_options(cfg, first)
    bootstrap_opts = slurm_job._train_options(cfg, bootstrap)

    assert slurm_job._LONG_RUN_TRAIN_SAMPLES == 1_000_000_000  # noqa: PLR2004
    assert first.checkpoints == 4  # noqa: PLR2004
    assert first.load == tmp_path / "ultra3-64n" / "phase1-checkpoints"
    assert second.load == tmp_path / "ultra3-64n" / "checkpoints"
    assert bootstrap_opts[bootstrap_opts.index("--train-samples") + 1] == str(slurm_job._LONG_RUN_TRAIN_SAMPLES)
    assert opts[opts.index("--exit-interval") + 1] == "1000"
    assert opts[opts.index("--train-samples") + 1] == str(slurm_job._LONG_RUN_TRAIN_SAMPLES)


def test_unlimited_checkpoint_run_omits_iteration_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(tmp_path, checkpoints_per_run=0)
    slurm_job._configure_run(args)
    config = slurm_job.FlavorSelection(["perf"]).policy.train_config("nano3_4x4", args, "native", 2)

    opts = slurm_job._train_options("nano3_4x4", config)

    assert "--exit-interval" not in opts
    assert opts[opts.index("--save-interval") + 1] == "250"
    assert opts[opts.index("--train-samples") + 1] == str(slurm_job._LONG_RUN_TRAIN_SAMPLES)


def test_repeat_runs_phase1_once_then_cycles_phase2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, repeat=3)
    args.flavors = slurm_job.FlavorSelection(["perf"])
    calls = []
    monkeypatch.setattr(
        slurm_job,
        "_phase_command",
        lambda _cfg, _args, _provider, phase, cycle=0: calls.append((phase, cycle)) or (["true"], {}),
    )
    monkeypatch.setattr(slurm_job, "wait_for_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(slurm_job.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    slurm_job._run_routine("nano3_4x4", args, "nvrx")

    assert calls == [(1, 0), (2, 0), (2, 1), (2, 2)]


@pytest.mark.parametrize(
    "case",
    [
        ("pretrain-native", None, 2, (1, 6)),
        ("pretrain-native", 0, 1, (0, 1)),
        ("pretrain-nvrx", None, 2, (None, 2)),
    ],
)
def test_native_restarts_the_daemon_for_each_finite_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[str, int | None, int, tuple[int | None, int]],
) -> None:
    slurm_job = _load_slurm_job()
    job, checkpoints, repeat, expected = case
    args = _args(tmp_path, checkpoints_per_run=checkpoints, repeat=repeat)
    args.job = job
    args.flavors = slurm_job.FlavorSelection(["perf"])
    calls = []
    args.flavors.policy.worker = lambda _cfg, run_args, _provider: calls.append(
        (run_args.checkpoints_per_run, run_args.repeat),
    )
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", SimpleNamespace(require_stopped=lambda: None))

    slurm_job._worker(args)

    assert calls == [expected]
    assert args.checkpoints_per_run == checkpoints
    assert args.repeat == repeat


def test_long_run_options_are_exported_and_override_allocation_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, checkpoints_per_run=6, repeat=2, time_minutes=240)
    slurm_job._configure_run(args)

    script, reaper, _log = slurm_job._build_job(args)
    slurm_job.sbatch_run(
        script, slurm_job.RESULTS_DIR / "scheduler.log", extra_sbatch_args=[reaper], dry_run=True,
    )
    reproducer = (tmp_path / "ultra3-64n" / "run.sbatch").read_text()

    assert "#SBATCH --time=04:05:00" in reproducer
    assert "--time=04:00:00" in reproducer
    assert '#SBATCH --comment=\'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240"' in reproducer
    assert "export SLURMJOB_OPT_CHECKPOINTS_PER_RUN=6" in reproducer
    assert "export SLURMJOB_OPT_REPEAT=2" in reproducer
    assert "export SLURMJOB_OPT_TIME_MINUTES=240" in reproducer
    assert "--checkpoints-per-run" not in reproducer.split("set -Eeuo pipefail", 1)[1]
    assert reproducer.count("--kill-on-bad-exit=1") == len(("allocation-epilog", "prepare", "train"))


def test_clean_run_removes_named_run_once_before_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    run = tmp_path / "ultra3-64n"
    (run / "logs").mkdir(parents=True)
    stale = run / "logs" / "old.log"
    stale.write_text("old")
    args = _args(tmp_path, clean_run=True)

    slurm_job._configure_run(args)

    assert not stale.exists()
    assert (run / "logs").is_dir()
    assert slurm_job.os.environ["SLURMJOB_OPT_CLEAN_RUN"] == "1"
    worker_file = run / "worker-must-not-clean"
    worker_file.write_text("keep")
    monkeypatch.setenv("SLURM_PROCID", "0")
    slurm_job._configure_run(args)
    assert worker_file.read_text() == "keep"


def test_clean_run_intent_is_reproducible_from_sbatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, clean_run=True)
    slurm_job._configure_run(args)
    script, _reaper, run_log = slurm_job._build_job(args)
    slurm_job.sbatch_run(script, run_log, dry_run=True)
    run = tmp_path / "ultra3-64n"
    reproducer = (run / "run.sbatch").read_text()
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "stale").write_text("old")
    (run / "logs" / "old.log").write_text("old")
    (run / "job_id").write_text("123\n")
    (run / "gitlab_job_id").write_text("456\n")
    start = reproducer.index('if [[ "${SLURMJOB_OPT_CLEAN_RUN:-0}"')
    end = reproducer.index("\nfi", start) + len("\nfi")
    env = {**slurm_job.os.environ, "SLURMJOB_OPT_CLEAN_RUN": "1"}

    subprocess.run(["bash", "-c", "set -Eeuo pipefail\n" + reproducer[start:end]], env=env, check=True)

    assert (run / "run.sbatch").is_file()
    assert not run_log.exists()
    assert not (run / "checkpoints").exists()
    assert not (run / "logs" / "old.log").exists()
    assert (run / "job_id").read_text() == "123\n"
    assert (run / "gitlab_job_id").read_text() == "456\n"
    assert "SLURMJOB_OPT_SBATCH_LOG" not in reproducer
    assert 'printf "%s\\n" "${SLURM_JOB_ID:?}" > "${SLURMJOB_OPT_RUN_DIR:?}/job_id"' in reproducer


def test_dry_run_writes_self_contained_exclusive_reproducer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    monkeypatch.setenv("PLATFORM", "gb200")
    monkeypatch.setenv("NVRX_COMMIT", "a" * 40)
    monkeypatch.setenv("CI_PIPELINE_ID", "98765")
    monkeypatch.setenv("CI_JOB_TOKEN", "must-not-be-written")
    args = _args(tmp_path, nvrx_commit="a" * 40)
    slurm_job._configure_run(args)
    script, reaper, run_log = slurm_job._build_job(args)

    assert slurm_job.sbatch_run(script, run_log, extra_sbatch_args=[reaper], dry_run=True) == 0

    reproducer_path = tmp_path / "ultra3-64n" / "run.sbatch"
    reproducer = reproducer_path.read_text()
    subprocess.run(["bash", "-n", str(reproducer_path)], check=True)
    assert "#SBATCH --exclusive" in reproducer
    assert "#SBATCH --nodes=64" in reproducer
    assert "#SBATCH --ntasks=256" in reproducer
    assert "#SBATCH --ntasks-per-node=4" in reproducer
    assert "#SBATCH --gpus=256" not in reproducer
    assert "#SBATCH --dependency=singleton" in reproducer
    assert "#SBATCH --signal=TERM@120" in reproducer
    assert "#SBATCH --job-name=ultra3-64n" in reproducer
    assert "98765" not in reproducer
    assert "#SBATCH --segment=16" in reproducer
    assert not any("${" in line for line in reproducer.splitlines() if line.startswith("#SBATCH"))
    assert "#SBATCH --output=/dev/null" in reproducer
    assert "_sbatch.log" not in reproducer
    assert "export ENROOT_IMAGE=" in reproducer
    assert 'export ENROOT_IMAGE_DIR="${ENROOT_IMAGE%/*}"' in reproducer
    assert "export NVRX_COMMIT=" + "a" * 40 in reproducer
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in reproducer
    assert 'export TRITON_CACHE_DIR="${SLURMJOB_OPT_RUN_DIR}/triton-cache"' in reproducer
    assert 'export TRITON_CACHE_PATH="${SLURMJOB_OPT_RUN_DIR}/triton-cache"' in reproducer
    assert 'export TORCHINDUCTOR_CACHE_DIR="${SLURMJOB_OPT_RUN_DIR}/torchinductor-cache"' in reproducer
    assert 'export TORCH_EXTENSIONS_DIR="${SLURMJOB_OPT_RUN_DIR}/torch-extensions"' in reproducer
    assert 'export NATIVE_CKPT_SERVER_LOG="${SLURMJOB_OPT_RUN_DIR}/logs/checkpoint_server.%h.%p.log"' in reproducer
    assert 'export SLURMJOB_OPT_RUN_DIR="${SLURMJOB_OPT_WORK_DIR}/${SLURMJOB_OPT_RUN_NAME}"' in reproducer
    assert 'export MEGATRON_NATIVE_BUILD_DIR="${SLURMJOB_OPT_RUN_DIR}/build/nvrx_checkpointing"' in reproducer
    assert 'export SLURMJOB_OPT_PYTHON_INSTALL_DIR="${SLURMJOB_OPT_RUN_DIR}/deps/python"' in reproducer
    assert "must-not-be-written" not in reproducer
    assert reproducer.index("export NVRX_COMMIT=") < reproducer.index("srun ")
    assert "${SLURMJOB_OPT_WORK_DIR}:${SLURMJOB_OPT_WORK_DIR}" in reproducer
    assert "VCPKG_ROOT,MEGATRON_NATIVE_BUILD_DIR,NVRX_REPO_ROOT" in reproducer
    worker_step = next(line for line in reproducer.splitlines() if " --ntasks=256 " in line)
    for name in slurm_job._TRAIN_ENV_DEFAULTS:
        assert name in worker_step
    prep_step = next(line for line in reproducer.splitlines() if "wheel_install.py" in line)
    body = reproducer.split("set -Eeuo pipefail", 1)[1]
    assert '"--container-image=${ENROOT_IMAGE}"' in body
    assert "${ENROOT_IMAGE_DIR}:/root" in body
    assert '"--account=${SLURM_ACCOUNT}"' in body
    assert "${SLURMJOB_OPT_LAUNCHER_ROOT}/ci/scripts/slurm_job.py" in body
    assert str(slurm_job.image_path()) not in body
    assert str(tmp_path / "ultra3-64n") not in body
    assert "--container-writable" in prep_step
    assert "install_run_dependency.py" in prep_step
    assert "python3 -m pip list" in prep_step
    assert "--report-install" in prep_step
    assert not any(line != prep_step and "install_run_dependency.py" in line for line in reproducer.splitlines())
    for name in ("SLURMJOB_OPT_REPEAT", "SLURMJOB_OPT_TIME_MINUTES"):
        assert name in reproducer
    assert "NATIVE_CKPT_SERVICE_PORT" not in reproducer
    assert "NATIVE_CKPT_SLURM_JOB_ID" not in reproducer
    host_cleanup = next(line for line in reproducer.splitlines() if "daemon_tools.py" in line)
    assert "--ntasks-per-node=1" in host_cleanup
    assert "--container-image=" not in host_cleanup
    assert reproducer.index("allocation_epilog() {") < reproducer.index(host_cleanup)
    assert reproducer.index(host_cleanup) < reproducer.index("\n}")
    assert reproducer.index("trap allocation_epilog EXIT") < reproducer.index(prep_step)
    assert '(( rc != 0 )) && exit "${rc}"' in reproducer
    assert "__epilog__" in reproducer


def test_manual_training_environment_override_is_reproduced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("PLATFORM", "gb200")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    args = _args(tmp_path, nodes=8, config="nano3", job="pretrain-nvrx", flavors=[])
    slurm_job._configure_run(args)
    script, _reaper, log = slurm_job._build_job(args)

    slurm_job.sbatch_run(script, log.with_name("scheduler.log"), dry_run=True)

    reproducer = (tmp_path / "ultra3-64n" / "run.sbatch").read_text()
    assert "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" in reproducer
    worker_step = next(line for line in reproducer.splitlines() if " --ntasks=32 " in line)
    assert "PYTORCH_CUDA_ALLOC_CONF" in worker_step


def test_ci_default_job_name_is_keyed_by_job_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("PLATFORM", "gb200")
    monkeypatch.setenv("CI_PIPELINE_ID", "98765")
    monkeypatch.setenv("CI_JOB_ID", "12345")
    args = _args(tmp_path, name=None, work_dir=None, nodes=2)
    slurm_job._configure_run(args)

    script, _reaper, _log = slurm_job._build_job(args)

    assert f"#SBATCH --job-name={slurm_job.RUN_NAME}" in script
    assert slurm_job.RUN_NAME.startswith("12345_")
    assert "98765" not in script


def test_srun_output_is_redirected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    log = tmp_path / "run.log"

    cmd = slurm_job.build_srun_cmd(
        nodes=2,
        ntasks=8,
        time_minutes=30,
        job_name="manual",
        image="/images/custom.sqsh",
        container_env="PYTHONPATH",
        gpus_per_node=4,
        total_gpus=8,
        output=log,
    )

    assert "--kill-on-bad-exit=1" in cmd
    assert f"--output={log}" in cmd
    assert "--open-mode=truncate" in cmd


@pytest.mark.parametrize(
    "case",
    [
        ("", 1, None, None),
        ("lyris", 2, None, None),
        ("ocihsg", 8, None, None),
        ("ocihsg", 16, None, "16"),
        ("", 64, None, "16"),
        ("lyris", 2, "8", "8"),
    ],
)
def test_sbatch_segment_matches_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[str, int, str | None, str | None],
) -> None:
    cluster, nodes, override, expected = case
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", cluster)
    if override is None:
        monkeypatch.delenv("SLURM_SEGMENT", raising=False)
    else:
        monkeypatch.setenv("SLURM_SEGMENT", override)

    script, _reaper = slurm_job.build_sbatch_script(
        nodes=nodes,
        ntasks=nodes * 4,
        time_minutes=30,
        job_name="segment-policy",
        image=str(tmp_path / "image.sqsh"),
        container_env="",
        gpus_per_node=4,
        total_gpus=nodes * 4,
        body="true",
    )

    directives = script.splitlines()
    expected_directive = [] if expected is None else [f"#SBATCH --segment={expected}"]
    assert [line for line in directives if line.startswith("#SBATCH --segment=")] == expected_directive
    assert "#SBATCH --exclusive" in directives


def test_prepare_image_template_api_is_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    template = tmp_path / "image_prep.tmpl"
    template.write_text("# ignored\nsrun --nodes=$nodes\n")
    monkeypatch.setattr(slurm_job, "_load_template", lambda _name: template.read_text())

    cmd = slurm_job.build_cmd_from_template("image_prep.tmpl", {"nodes": 2}, ["echo", "ready"])

    assert cmd == ["srun", "--nodes=2", "echo", "ready"]


def test_enroot_import_requests_runner_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("CLUSTER", "")

    cmd = slurm_job.build_cmd_from_template(
        "enroot_import.tmpl",
        {"account": "account", "time_minutes": 20, "gpus_per_node": 4},
        ["enroot", "import"],
    )

    assert "--gpus-per-node=4" in cmd


def test_image_build_selects_account_compatible_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_image = _load_prepare_image()

    def run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        if cmd[0] == "sinfo":
            return SimpleNamespace(returncode=0, stdout="batch*\ngb200\n", stderr="")
        return SimpleNamespace(returncode=0 if "--partition=gb200" in cmd else 1, stdout="", stderr="invalid")

    monkeypatch.setattr(prepare_image.subprocess, "run", run)
    cmd = ["srun", "--account=account", "--partition=batch", "--gpus-per-node=4", "enroot", "import"]

    partition = prepare_image._valid_partition(cmd)

    assert partition == "gb200"
    assert prepare_image._set_partition(cmd, partition)[1] == "--partition=gb200"


def test_slurm_job_consumers_import() -> None:
    subprocess.run(
        [sys.executable, "-c", "import ckpt_stress, prepare_image"],
        cwd=SCRIPTS_DIR,
        check=True,
    )


def test_auto_workloads_only_allow_perf_gate_warnings() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())

    assert config[".slurm_job"]["allow_failure"] == {"exit_codes": [3]}
    manual_optional = config[".manual_optional"]
    assert "allow_failure" not in manual_optional
    assert manual_optional["rules"][0] == {
        "if": '$PIPELINE_SUITE == "nvrx" || $PIPELINE_SUITE == "large-scale"',
        "when": "never",
    }
    assert manual_optional["rules"][1] == {
        "if": '$RUN_PIPELINE == "true" || $CI_PIPELINE_SOURCE == "merge_request_event"',
        "when": "on_success",
    }
    assert manual_optional["rules"][2]["allow_failure"] is True
    for suite in ("nvrx", "large_scale"):
        policy = config[f".{suite}_pipeline"]
        assert policy["allow_failure"] is False
        assert policy["rules"][-1] == {"when": "never"}
    assert ".manual_nvrx" not in config


def test_generated_pipeline_suites_and_node_tags() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    workloads = {
        name: job for name, job in config.items()
        if isinstance(job, dict) and isinstance(job.get("variables"), dict) and "JOB_ARGS" in job["variables"]
    }

    for name, job in workloads.items():
        args = shlex.split(job["variables"]["JOB_ARGS"])
        nodes = int(args[args.index("--nodes") + 1]) if "--nodes" in args else 1
        platform = next(tag for tag in ("devbox", "gb200", "h100") if f":{tag}:" in f":{name}:")
        assert job["tags"] == [platform, f"{nodes}n"]

    nvrx = {name: job for name, job in workloads.items() if ":nvrx:" in name}
    assert any(":ultra3bf16_16x4" in name for name in nvrx)
    assert any(":ultra3fp8_64x4" in name for name in nvrx)
    assert all(job["extends"][-1] == ".nvrx_pipeline" for job in nvrx.values())

    large = {
        name: job for name, job in workloads.items()
        if ":native:ultra3bf16_16x4" in name or ":native:ultra3fp8_64x4" in name
    }
    assert set(large) == {
        "[DETR]:gb200:native:ultra3bf16_16x4",
        "[PERF]:gb200:native:ultra3bf16_16x4",
        "[DETR]:gb200:native:ultra3fp8_64x4",
        "[PERF]:gb200:native:ultra3fp8_64x4",
    }
    assert all(job["extends"][-1] == ".large_scale_pipeline" for job in large.values())


def test_canceled_ci_job_cancels_only_its_recorded_slurm_job(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    cancel_script = config[".slurm_job"]["after_script"][0]
    for name, gitlab_id, slurm_id in (("target", "456", "123"), ("other", "999", "789")):
        run = tmp_path / "ci" / "ci-runs" / name
        run.mkdir(parents=True)
        (run / "gitlab_job_id").write_text(f"{gitlab_id}\n")
        (run / "job_id").write_text(f"{slurm_id}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scancel = bin_dir / "scancel"
    scancel.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "${SCANCEL_LOG}"\n')
    scancel.chmod(0o755)
    cancel_log = tmp_path / "scancel.log"

    subprocess.run(
        ["bash", "-c", cancel_script],
        cwd=tmp_path,
        env={
            "CI_JOB_ID": "456",
            "CI_JOB_STATUS": "canceled",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "SCANCEL_LOG": str(cancel_log),
        },
        check=True,
    )

    assert cancel_log.read_text() == "-- 123\n"


def test_foreground_submission_records_job_id_and_uses_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setenv("CI_JOB_ID", "456")
    commands: list[list[str]] = []

    class Process:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = io.StringIO(stdout)
            self.returncode: int | None = None

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

    def popen(command: list[str], **_kwargs: object) -> Process:
        commands.append(command)
        return Process("123;cluster\n" if command[0] == "sbatch" else "")

    monkeypatch.setattr(slurm_job.subprocess, "Popen", popen)
    verified = []
    monkeypatch.setattr(slurm_job, "_verify_allocation", lambda job_id, rc: verified.append((job_id, rc)))

    assert slurm_job.sbatch_run("#!/bin/bash\n\necho test\n", run / "scheduler.log") == 0
    assert ["sbatch", "--parsable", "--wait", "--export=ALL", str(run / "run.sbatch")] in commands
    assert verified == [("123", 0)]
    assert (run / "job_id").read_text() == "123\n"
    assert (run / "gitlab_job_id").read_text() == "456\n"


def test_background_submission_uses_only_the_streamed_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    log = logs / "train.log"
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", logs)
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(returncode=0, stdout="123\n", stderr="")
        if command[0] == "sbatch" else SimpleNamespace(returncode=0),
    )

    assert slurm_job.sbatch_run("#!/bin/bash\n\necho test\n", log, background=True) == 0

    assert list(logs.glob("*.log")) == [log]
    reproducer = (run / "run.sbatch").read_text()
    assert "#SBATCH --output=/dev/null" in reproducer
    assert "_sbatch.log" not in reproducer


def test_signal_cancellation_calls_scancel(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    commands: list[list[str]] = []
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )

    slurm_job._cancel_slurm_job("123")

    assert commands == [["scancel", "--", "123"]]


def test_sacct_rejects_canceled_allocation_with_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="123|CANCELLED by 456|0:0|0:15\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=r"state=CANCELLED by 456 exit=0:0 derived_exit=0:15"):
        slurm_job._verify_allocation("123", 0)


def test_sacct_accepts_only_the_perf_soft_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="123|FAILED|3:0|0:0\n", stderr=""),
    )

    slurm_job._verify_allocation("123", slurm_job.SOFT_FAIL_RC)


def test_workflow_runs_mr_updates_but_not_ordinary_pushes() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    rules = config["workflow"]["rules"]

    assert config["variables"]["PIPELINE_SUITE"]["options"] == ["default", "nvrx", "large-scale"]
    assert {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'} in rules
    assert {"if": '$CI_PIPELINE_SOURCE == "push" && $RUN_PIPELINE == "true"'} in rules
    assert any("pipeline" in rule.get("if", "") and rule.get("when") != "never" for rule in rules)
    assert any(rule.get("when") == "never" and "PIPELINE_SUITE !=" in rule.get("if", "") for rule in rules)
    assert rules[-1] == {"when": "never"}


def test_default_pipeline_has_manual_suite_launchers() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    launcher = config[".pipeline_launcher"]

    assert launcher["stage"] == "launch"
    assert launcher["rules"][0] == {
        "if": '$CI_PIPELINE_SOURCE == "pipeline" || $CI_PIPELINE_SOURCE == "parent_pipeline"',
        "when": "never",
    }
    assert launcher["rules"][-1] == {"when": "manual", "allow_failure": True}
    assert launcher["image"] == "curlimages/curl:8.15.0"
    assert launcher["tags"] == ["perflab", "type/docker", "os/linux"]
    assert launcher["variables"]["GIT_STRATEGY"] == "none"
    assert "trigger" not in launcher
    script = launcher["script"][0]
    assert "CI_JOB_TOKEN" in script
    assert "variables[PIPELINE_SUITE]" in script
    assert "variables[UPSTREAM_PIPELINE_SHA]" in script
    assert "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME" in script
    assert config["launch:nvrx-pipeline"]["variables"]["PIPELINE_SUITE"] == "nvrx"
    assert config["launch:large-scale-pipeline"]["variables"]["PIPELINE_SUITE"] == "large-scale"


def test_training_is_teed_without_a_postprocess_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "")
    monkeypatch.setenv("PLATFORM", "gb200")
    slurm_job._configure_run(_args(tmp_path))

    script, _reaper, log = slurm_job._build_job(_args(tmp_path))

    assert "#SBATCH --exclusive" in script
    assert f'exec > >(tee "${{SLURMJOB_OPT_RUN_DIR}}/logs/{log.name}") 2>&1' in script
    assert "__postprocess__" not in script
    assert "__epilog__" in script
    assert script.count("tee ") == 1
    assert script.count("--kill-on-bad-exit=1") == len(("allocation-epilog", "prepare", "train"))


def test_deterministic_flavor_keeps_perf_epilog_unless_valgrind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    calls = []
    monkeypatch.setattr(slurm_job.PerfReport, "parse_and_report", lambda *args: calls.append(args) or True)
    deterministic_args = _args(tmp_path, flavors=["deterministic"], baseline={})
    deterministic_args.flavors = slurm_job.FlavorSelection(deterministic_args.flavors)

    assert deterministic_args.flavors.policy.epilog(deterministic_args, tmp_path / "run.log")
    assert len(calls) == 1
    valgrind_args = _args(tmp_path, flavors=["valgrind", "deterministic"], baseline={})
    valgrind_args.flavors = slurm_job.FlavorSelection(valgrind_args.flavors)

    assert valgrind_args.flavors.policy.epilog(valgrind_args, tmp_path / "run.log")
    assert len(calls) == 1


def test_flavors_compose_execution_policy_independent_of_cli_order() -> None:
    slurm_job = _load_slurm_job()

    flavors = slurm_job.FlavorSelection(["valgrind", "deterministic"])

    assert flavors.names == ("deterministic", "valgrind")
    assert flavors.tag == "deterministic-valgrind"
    assert flavors.policy.phase2_save_interval == slurm_job.FlavorPolicy.phase2_save_interval
    assert flavors.policy.phase2_checkpoints == slurm_job.FlavorPolicy.phase2_checkpoints
    assert not flavors.policy.manages_native_server
    assert not flavors.policy.install_prebuilt
    assert flavors.policy.prep_args() == ["--build-from-source", "--define", "PHOENIX_BUILD_FOR_VALGRIND=1"]
    assert flavors.policy.worker.__self__.__class__ is slurm_job.ValgrindFlavor
    assert flavors.policy.server.__self__.__class__ is slurm_job.ValgrindFlavor
    assert flavors.policy.reporter is None


def test_execution_mode_conflicts_are_declared_by_flavors() -> None:
    slurm_job = _load_slurm_job()

    with pytest.raises(ValueError, match="mutually exclusive"):
        slurm_job.FlavorSelection(["perf", "deterministic"])


def test_batch_epilog_reports_from_embedded_baseline_without_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "_load_yaml", lambda *_: pytest.fail("batch epilog must not load YAML"))
    log = tmp_path / "run.log"
    log.write_text(
        "successfully saved checkpoint from iteration 100 to /run/phase1-checkpoints\n"
        "[native_ckpt:phase] phase=2\n"
        "successfully saved checkpoint from iteration 250 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 500 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 750 to /run/checkpoints\n"
        "[native_ckpt:cli][0][INFO][perf] Finished bw_mbps=150000 wr_tnsr_us=10 total_written_bytes=100",
    )
    baseline = '{"metrics":{"bw_mbps":140000,"wr_tnsr_us":10,"total_written_bytes":100}}'

    slurm_job._epilog(
        [
            "pretrain-native",
            "--nodes",
            "1",
            "--gpus-per-node",
            "1",
            "--config",
            "pico",
            "--flavor",
            "perf",
            "--checkpoints-per-run",
            "3",
            "--log-file",
            str(log),
            "--baseline-json",
            baseline,
        ],
    )

    assert list(tmp_path.glob("*_report.json"))


def test_batch_epilog_rejects_incomplete_checkpoint_cycle(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "run.log"
    log.write_text(
        "successfully saved checkpoint from iteration 100 to /run/phase1-checkpoints\n"
        "successfully saved checkpoint from iteration 250 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 500 to /run/checkpoints",
    )

    with pytest.raises(RuntimeError, match=r"found 1 phase-1 and 2 phase-2 saves; expected at least 1 and 3"):
        slurm_job._epilog(
            [
                "pretrain-native",
                "--nodes",
                "1",
                "--gpus-per-node",
                "1",
                "--config",
                "pico",
                "--flavor",
                "perf",
                "--checkpoints-per-run",
                "3",
                "--log-file",
                str(log),
                "--baseline-json",
                '{"metrics":{"bw_mbps":140000}}',
            ],
        )


def test_perf_parser_supports_python_without_strict_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    builtin_zip = zip
    monkeypatch.setattr(slurm_job, "zip", builtin_zip, raising=False)
    text = "\n".join(
        f"[native_ckpt:cli][{rank}][INFO][perf] Finished bw_mbps={bandwidth} wr_tnsr_us=10 total_written_bytes=100"
        for rank, bandwidth in enumerate((100, 100, 80, 80))
    )

    metrics = slurm_job.PerfReport._parse_metrics(text, 4)

    assert metrics["bw_mbps"]["median"] == 90  # noqa: PLR2004
    assert metrics["bw_socket_imbalance"]["mean"] == 1.25  # noqa: PLR2004


def test_source_wheel_build_skips_vcpkg_bootstrap_dependency_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    commands = []
    wheel = tmp_path / "nvrx_checkpointing-linux_x86_64.whl"
    build_dir = tmp_path / "run" / "build" / "nvrx_checkpointing"
    monkeypatch.setenv("MEGATRON_NATIVE_BUILD_DIR", str(build_dir))
    monkeypatch.setenv("VCPKG_ROOT", str(tmp_path / "vcpkg"))
    tools = {
        "cmake": "/usr/bin/cmake",
        "clang-22": "/usr/bin/clang-22",
        "clang++-22": "/usr/bin/clang++-22",
    }
    monkeypatch.setattr(wheel_install.shutil, "which", tools.get)
    monkeypatch.setattr(
        wheel_install.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(wheel_install, "_find_local_wheel", lambda _directory: wheel)
    monkeypatch.setattr(wheel_install, "_pip_install", lambda _wheel: None)

    wheel_install.build_and_install_from_source()

    assert "-DVCPKG_BOOTSTRAP_OPTIONS=-skipDependencyChecks" in commands[0]
    assert f"-DVCPKG_ROOT={tmp_path / 'vcpkg'}" in commands[0]
    assert "-DCMAKE_C_COMPILER=/usr/bin/clang-22" in commands[0]
    assert "-DCMAKE_CXX_COMPILER=/usr/bin/clang++-22" in commands[0]
    assert commands[0][commands[0].index("-B") + 1] == str(build_dir)
    assert commands[1][2] == str(build_dir)


def test_wheel_install_uses_pip_target_instead_of_extracting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    wheel = tmp_path / "nvrx_checkpointing-0.0.1-cp312-abi3-linux_aarch64.whl"
    wheel.write_text("wheel")
    target = tmp_path / "run/deps/python"
    commands: list[list[str]] = []
    monkeypatch.setenv("SLURMJOB_OPT_PYTHON_INSTALL_DIR", str(target))
    monkeypatch.setenv("WHEEL_ARCH", "aarch64")
    monkeypatch.setattr(wheel_install, "_source_revision", lambda: "a" * 40)

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        staging = Path(command[command.index("--target") + 1])
        package = staging / "nvrx_checkpointing"
        package.mkdir(parents=True)
        for name in wheel_install._WHEEL_BINARIES:
            (package / name).write_text(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wheel_install.subprocess, "run", run)

    wheel_install._pip_install(wheel)

    assert commands[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert commands[0][commands[0].index("--implementation") + 1] == "cp"
    assert commands[0][commands[0].index("--python-version") + 1] == "3.12"
    assert commands[0][commands[0].index("--abi") + 1] == "abi3"
    assert commands[0][commands[0].index("--platform") + 1] == "linux_aarch64"
    assert commands[0][commands[0].index("--target") + 1] != str(target)
    assert (target / ".nvrx-checkpointing-commit").read_text() == "a" * 40 + "\n"
    assert all((target / "nvrx_checkpointing" / name).is_file() for name in wheel_install._WHEEL_BINARIES)
    source = (REPO_ROOT / "ci/scripts/wheel_install.py").read_text()
    assert "_extract_binaries" not in source
    assert "zipfile" not in source


def test_registry_wheel_downloads_canonical_filename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel_install = _load_slurm_job().wheel_install
    filename = "nvrx_checkpointing-0.0.1-cp312-abi3-linux_aarch64.whl"
    monkeypatch.setenv("CI_JOB_TOKEN", "test-token")
    monkeypatch.setenv("WHEEL_ARCH", "aarch64")
    monkeypatch.setattr(
        wheel_install,
        "registry_wheel_url",
        lambda name=None: f"https://example.invalid/{name or 'nvrx_checkpointing-linux_aarch64.txt'}",
    )

    def response(request: object, **_kwargs: object) -> io.BytesIO:
        if str(request.full_url).endswith(".txt"):
            return io.BytesIO(f"{filename}\n".encode())
        return io.BytesIO(b"wheel contents")

    monkeypatch.setattr(
        wheel_install.urllib.request,
        "urlopen",
        response,
    )

    downloaded = wheel_install._fetch_registry_wheel(tmp_path)

    assert downloaded == tmp_path / filename
    assert downloaded.read_bytes() == b"wheel contents"
    template = (REPO_ROOT / "ci/gitlab-ci-native.template.yml").read_text()
    assert '--upload-file "${WHEEL}" "${WHEEL_URL}"' in template
    assert "printf '%s\\n' \"${WHEEL_NAME}\"" in template


def test_wheel_install_does_not_trust_an_untracked_site_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    wheel = tmp_path / "nvrx_checkpointing-0.0.1-cp312-abi3-linux_x86_64.whl"
    wheel.write_text("wheel")
    installed: list[Path] = []
    monkeypatch.delenv("SLURMJOB_OPT_PYTHON_INSTALL_DIR", raising=False)
    monkeypatch.setattr(wheel_install, "_binaries_installed", lambda: True)
    monkeypatch.setattr(wheel_install, "_find_local_wheel", lambda _directory: wheel)
    monkeypatch.setattr(wheel_install, "_pip_install", installed.append)

    assert wheel_install.install_prebuilt_wheel(tmp_path, fetch_remote=False, probe=False)
    assert installed == [wheel]


def test_wheel_packages_only_required_metadata_compare() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    template = (REPO_ROOT / "ci/gitlab-ci-native.template.yml").read_text()

    assert "metadata_compare.py" in cmake
    assert "scripts@scripts" not in cmake
    assert "python -m nvrx_checkpointing.metadata_compare --help" in template


def test_source_wheel_build_prepares_ubuntu_when_clang_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    prepared = False

    def which(name: str) -> str | None:
        if name in ("clang-22", "clang++-22") and prepared:
            return f"/usr/bin/{name}"
        return None

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal prepared
        prepared = True
        assert command == ["/bin/bash", str(REPO_ROOT / "scripts" / "prepare.sh"), "ubuntu2404"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wheel_install.shutil, "which", which)
    monkeypatch.setattr(wheel_install.subprocess, "run", run)

    wheel_install._ensure_source_toolchain()

    assert prepared


def test_vcpkg_bootstrap_invalidates_cached_executable() -> None:
    cmake = (REPO_ROOT / "cmake" / "vcpkg.cmake").read_text()
    bootstrap = cmake.index('COMMAND "${Z_VCPKG_BOOTSTRAP_SCRIPT}"')
    cache_reset = cmake.index("unset(VCPKG_EXE CACHE)", bootstrap)
    required_lookup = cmake.index("find_program(VCPKG_EXE vcpkg REQUIRED", bootstrap)

    assert bootstrap < cache_reset < required_lookup


def test_native_auto_build_passes_vcpkg_root_explicitly() -> None:
    package = (REPO_ROOT / "__init__.py").read_text()

    assert "*get_vcpkg_cmake_args()," in package
    assert 'f"-DVCPKG_ROOT={Path(root).expanduser().resolve()}"' in package


def test_worker_uses_step_node_order_for_checkpoint_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "batch-[7,3]")
    monkeypatch.setenv("SLURM_STEP_NODELIST", "step-[3,7]")
    calls = []
    monkeypatch.setattr(slurm_job.subprocess, "check_output", lambda command, **_: calls.append(command) or "step-3\nstep-7\n")

    env = slurm_job._base_env()

    assert env["SLURM_JOB_NODELIST"] == "step-[3,7]"
    assert env["MASTER_ADDR"] == "step-3"
    assert env["TRITON_CACHE_DIR"].endswith("/rank-0")
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("/rank-0")
    assert env["TORCH_EXTENSIONS_DIR"].endswith("/rank-0")
    assert calls == [["scontrol", "show", "hostnames", "step-[3,7]"]]


def test_pretrain_environment_snapshot_is_rank0_and_excludes_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURM_PROCID", "0")
    slurm_job._print_pretrain_env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "NCCL_IB_TIMEOUT": "24",
            "RANK": "0",
            "CI_JOB_TOKEN": "must-not-be-printed",
        },
        "phase1",
    )

    output = capsys.readouterr().out
    assert "[slurm_job:env] phase1 rank0 environment" in output
    assert "[slurm_job:env] PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" in output
    assert "[slurm_job:env] NCCL_IB_TIMEOUT=24" in output
    assert "[slurm_job:env] RANK=0" in output
    assert "CI_JOB_TOKEN" not in output
    assert "must-not-be-printed" not in output

    monkeypatch.setenv("SLURM_PROCID", "1")
    slurm_job._print_pretrain_env({"PYTORCH_CUDA_ALLOC_CONF": "value"}, "phase1")
    assert capsys.readouterr().out == ""


def test_failed_nvrx_phase_does_not_retry_one_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    args = SimpleNamespace(
        job="pretrain-nvrx",
        flavors=[],
        nodes=8,
        gpus_per_node=4,
        config="nano4",
        checkpoints_per_run=None,
        repeat=1,
        extra_args=[],
    )
    calls = []
    monkeypatch.setattr(slurm_job, "_phase_command", lambda *_: (["train"], {}))
    monkeypatch.setattr(slurm_job, "wait_for_file", lambda *_: None)
    results = iter((0, 1))
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(returncode=next(results)),
    )

    with pytest.raises(RuntimeError, match=r"training phase 2 failed \(rc=1\)"):
        slurm_job._run_routine("nano4_8x4", args, "nvrx")

    assert len(calls) == 2  # noqa: PLR2004


def test_flavor_transforms_are_profile_independent() -> None:
    slurm_job = _load_slurm_job()
    perf_opts = [
        "--cuda-graph-impl",
        "local",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "moe",
        "--async-save",
    ]
    perf_env = {}
    slurm_job.FlavorSelection(["perf"]).transform(perf_opts, perf_env)

    assert perf_opts == ["--cuda-graph-impl", "local", "--async-save", "--async-ckpt-use-cpu-shm"]
    assert not perf_env
    deterministic_opts = ["--cuda-graph-impl", "local", "--cuda-graph-modules", "mamba", "attn", "moe_router"]
    deterministic_env = {}
    slurm_job.FlavorSelection(["deterministic"]).transform(deterministic_opts, deterministic_env)
    assert deterministic_opts == [
        "--attention-backend",
        "unfused",
        "--deterministic-mode",
    ]
    source = (REPO_ROOT / "ci/scripts/slurm_job.py").read_text()
    assert "ProfileOptionPolicy" not in source
    assert "profile_policies" not in source
