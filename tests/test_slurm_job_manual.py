# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: ANN202, INP001, PLR0915, S101, S108, S607, SLF001
from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import lzma
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "ci" / "scripts"
DEFAULT_MEGATRON_COMMIT = "16251ac1221f4814cd39350def05cefa3c36734a"
DEFAULT_NVRX_COMMIT = "73b17d9bcff543a0a5f64e95d64f26899d730653"
# A cross-source determinism run materializes one checkout of each dependency
# per routine.
_MUTATED_ENV = (
    "TMPDIR",
    "CI_JOB_ID",
    "CI_JOB_TOKEN",
    "CI_SERVER_HOST",
    "UPSTREAM_PIPELINE_ID",
    "UPSTREAM_WHEEL_AARCH64_JOB_ID",
    "UPSTREAM_WHEEL_X86_64_JOB_ID",
    "ENROOT_IMAGE",
    "ENROOT_IMAGE_DIR",
    "MEGATRON_COMMIT",
    "MEGATRON_NATIVE_BUILD_DIR",
    "MEGATRON_OTEL_ENABLED",
    "MEGATRON_OTEL_METRICS_ENABLED",
    "MEGATRON_OTEL_EXPORT_STRATEGY",
    "MEGATRON_OTEL_RUN_ID",
    "MEGATRON_OTEL_TRACES_ENABLED",
    "MEGATRON_REPO_URL",
    "MEGATRON_ROOT",
    "MASTER_PORT",
    "NATIVE_CKPT_SERVER_LOG",
    "NATIVE_CKPT_OTEL_EXPORT",
    "NATIVE_CKPT_OTEL_FILE",
    "NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS",
    "NVRX_OTEL_USE_SYSTEM_RANDOM",
    "NCCL_IB_TIMEOUT",
    "NCCL_NET",
    "NVTE_BWD_LAYERNORM_SM_MARGIN",
    "NVTE_FWD_LAYERNORM_SM_MARGIN",
    "NVTE_FUSED_ATTN",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "PYTORCH_CUDA_ALLOC_CONF",
    "QUANTIZATION_TYPE_DEBUG",
    "SLURMJOB_OPT_BACKGROUND",
    "SLURMJOB_OPT_JOB_ARRAY_SEGMENT",
    "SLURMJOB_OPT_CHECKPOINTS_PER_RUN",
    "SLURMJOB_OPT_CHECKPOINT_INTERVAL",
    "SLURMJOB_OPT_CLEAN_RUN",
    "SLURMJOB_OPT_CONFIG",
    "SLURMJOB_OPT_DRY_RUN",
    "SLURMJOB_OPT_FLAVOR",
    "SLURMJOB_OPT_RUN",
    "SLURMJOB_OPT_BASELINE",
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

# Results a test produces belong outside the checkout it is testing.
_SCRATCH_RESULTS = Path(tempfile.mkdtemp(prefix="slurm-job-tests-"))


def _perf_finished(
    rank: int = 0,
    bandwidth: float = 150000,
    write_us: int = 10,
    written_bytes: int = 100,
    snapshot: int = 1,
) -> str:
    return (
        f"[native_ckpt:svr][0][PERF] Rank={rank} Finished bw_mbps={bandwidth:.2f} "
        f"| Time(us):  wr_tnsr_us={write_us}us snapshot_us=11us total_us=12us "
        f"| Count: num_tensors=2 total_d2h=2 | Bytes: total_written_bytes={written_bytes} "
        f"avg_d2h_rq=50 avg_tnsr=50 snapshot={snapshot}"
    )


def _perf_span(
    rank: int = 0,
    bandwidth: float = 150000,
    write_us: int = 10,
    written_bytes: int = 100,
    snapshot: int = 1,
) -> str:
    return (
        f"[native_ckpt:svr][0][PERF] Rank={rank} "
        f"Begin span=ckpt.save started_at_ns=123456 snapshot={snapshot}\n"
        f"{_perf_finished(rank, bandwidth, write_us, written_bytes, snapshot)}\n"
    )


def _native_otel_sample(
    rank: int = 0,
    bandwidth: float = 150000,
    write_us: int = 10,
    written_bytes: int = 100,
    snapshot: int = 1,
) -> dict:
    return {
        "session_id": 0,
        "rank": rank,
        "snapshot_id": snapshot,
        "bw_mbps": bandwidth,
        "wr_tnsr_us": write_us,
        "snapshot_us": 11,
        "total_us": 12,
        "num_tensors": 2,
        "total_d2h": 2,
        "total_tensor_bytes": written_bytes,
        "total_written_bytes": written_bytes,
        "avg_d2h_rq": written_bytes // 2,
        "avg_tnsr": written_bytes // 2,
        "start_time_unix_nano": 1,
        "end_time_unix_nano": 2,
    }


def _otel_data(
    *samples: dict,
    megatron_spans: list[dict] | None = None,
    megatron_metrics: list[dict] | None = None,
    resource_metrics: list[dict] | None = None,
) -> dict:
    return {
        "native_saves": list(samples),
        "daemon_durable_spans": [],
        "megatron_spans": megatron_spans or [],
        "megatron_metrics": megatron_metrics or [],
        "resource_metrics": resource_metrics or [],
    }


def _checkpoint_span(
    name: str,
    start_ms: float,
    end_ms: float,
    *,
    call_idx: int | list[int] | None = None,
    rank: int = 0,
) -> dict:
    attributes = {} if call_idx is None else {"nvrx.call_idx": call_idx}
    return {
        "name": name,
        "resource": {"dl.rank": rank},
        "attributes": attributes,
        "start_time_unix_nano": round(start_ms * 1e6),
        "end_time_unix_nano": round(end_ms * 1e6),
    }


def _megatron_perf_metric_samples(*ranks: int) -> list[dict]:
    values = {
        "megatron.training.memory_allocated_gb": 60.0,
    }
    return [
        {
            "name": name,
            "resource": {"dl.rank": rank},
            "value": value + rank,
            "time_unix_nano": 10,
        }
        for rank in ranks
        for name, value in values.items()
    ]


def _load_slurm_job(*, gpu_probe: bool = False):
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("slurm_job_manual_test", SCRIPTS_DIR / "slurm_job.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # A job points this at its work directory before anything runs. Until it
    # does the module names a path inside the checkout, so a test exercising a
    # routine directly would write results into the source tree.
    module.RESULTS_DIR = _SCRATCH_RESULTS
    if not gpu_probe:
        # Commands build the same environment on any host unless a test asks for the driver's answer.
        module._gpu_compute_capability = lambda: ""
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


_CONSTANT_STRUCTURE = "--ckpt-assume-constant-structure"


@pytest.mark.parametrize(("selected", "expected_installs"), [("native", 1), ("nvrx", 0)])
def test_comparison_launcher_installs_only_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
    expected_installs: int,
) -> None:
    slurm_job = _load_slurm_job()
    installs: list[None] = []
    package = ModuleType("nvrx_checkpointing")
    megatron_package = ModuleType("nvrx_checkpointing.megatron")
    megatron_package.provider = SimpleNamespace(
        NATIVE_PROVIDER="native",
        selected_provider=lambda: selected,
    )
    megatron_package.integration = SimpleNamespace(install=lambda: installs.append(None))
    package.megatron = megatron_package
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing", package)
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing.megatron", megatron_package)
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing.megatron.provider", megatron_package.provider)
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing.megatron.integration", megatron_package.integration)

    assert slurm_job._install_selected_checkpoint_provider() is megatron_package.provider
    assert len(installs) == expected_installs


def test_node_once_waits_for_local_peers_before_action(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", results)
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node.example")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "2(x4)")
    sync = results / slurm_job.SYNC_DIR_NAME
    sync.mkdir(exist_ok=True)
    marker = sync / "123-phase1-stop-node"
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
    sync = results / slurm_job.SYNC_DIR_NAME
    sync.mkdir(exist_ok=True)
    marker = sync / "123-phase1"
    marker.with_name(f"{marker.name}.ready-1").touch()

    slurm_job._sync_ranks("phase1")

    assert marker.with_name(f"{marker.name}.ready-0").exists()


def test_array_sync_markers_are_scoped_to_launcher_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "16")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")

    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    assert slurm_job._sync_job_id() == "123-restart0"
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "2")
    assert slurm_job._sync_job_id() == "123-restart2"


def test_a_run_writes_its_handshakes_beside_its_own_logs(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    slurm_job.RUN_ROOT = tmp_path / "nano3_8n-detr" / "nvrx.ref"
    slurm_job.RESULTS_DIR = slurm_job.RUN_ROOT / "logs"

    root = slurm_job._sync_root()

    # Beside the logs of the run that made them, never among them.
    assert root == slurm_job.RUN_ROOT / slurm_job.SYNC_DIR_NAME
    assert root.parent == slurm_job.RESULTS_DIR.parent
    assert slurm_job.SYNC_DIR_NAME not in {path.name for path in slurm_job.RESULTS_DIR.glob("*")}


def test_the_suite_writes_nothing_into_the_checkout() -> None:
    slurm_job = _load_slurm_job()

    # Nothing has configured a run, so the module still names its own default.
    # A test that exercises a routine must not have that land in the sources.
    assert slurm_job.REPO_ROOT not in slurm_job.RESULTS_DIR.parents


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


@pytest.mark.parametrize(
    "message",
    [
        "[native_ckpt:svr][ERROR] peer finalize request failed\n",
        "Error: Received finalize notification for non-existent snapshot. Ignoring.\n",
    ],
)
def test_server_log_error_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "checkpoint_server.node.42.log"
    log.write_text(message)
    monkeypatch.setenv("NATIVE_CKPT_SERVER_LOG", str(tmp_path / "checkpoint_server.%h.%p.log"))
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node")
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(pids=lambda: ["42"], stop=lambda _graceful: None),
    )
    monkeypatch.setattr(slurm_job, "_node_once", lambda _tag, action: action())

    with pytest.raises(RuntimeError, match="reported errors"):
        slurm_job.FlavorPolicy().stop_server("native", "phase2")


def test_host_memory_metrics_distinguish_training_process_and_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    daemon = slurm_job.otel_report.metric([123456], "B", higher_is_better=False)
    daemon["per_node"] = {"node-a": 123456}
    monkeypatch.setattr(
        slurm_job.otel_report,
        "complete_run_process_memory_peaks",
        lambda _results_dir: {"checkpoint_daemon_peak_rss_bytes": daemon},
    )
    metrics = slurm_job._with_host_memory_metrics(
        {
            "cpu_rss_bytes": slurm_job.otel_report.metric(
                [100, 104],
                "B",
                higher_is_better=False,
                ranks=[0, 1],
            ),
        },
        tmp_path,
        native=True,
    )

    assert "cpu_rss_bytes" not in metrics
    assert metrics["training_process_peak_rss_bytes"]["max"] == 104  # noqa: PLR2004
    assert metrics["checkpoint_daemon_peak_rss_bytes"]["max"] == 123456
    assert metrics["checkpoint_daemon_peak_rss_bytes"]["per_node"] == {"node-a": 123456}
    assert slurm_job.PerfReport._statistic(
        "training_process_peak_rss_bytes",
        metrics["training_process_peak_rss_bytes"],
    ) == "max"


def test_log_report_uses_each_ranks_peak_rss_across_all_phases() -> None:
    slurm_job = _load_slurm_job()
    text = "\n".join(
        f"[native_ckpt:mem] rank={rank} gpu_peak_bytes=1 cpu_rss_bytes={rss}"
        for rank, rss in ((0, 100), (1, 120), (0, 130), (1, 110))
    )

    measured = slurm_job.log_report._metrics(text, "", 4)["cpu_rss_bytes"]

    assert measured["samples"] == [130, 120]
    assert measured["per_rank_median"] == {0: 130, 1: 120}


def test_stopping_server_does_not_write_custom_memory_files(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    recorded = []
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(pids=lambda: ["42"], stop=lambda graceful: recorded.append(("stop", graceful))),
    )
    monkeypatch.setattr(slurm_job, "_node_once", lambda _tag, action: action())

    slurm_job.FlavorPolicy().stop_server("nvrx", "reference", graceful=False)
    slurm_job.FlavorPolicy().stop_server("native", "measured", graceful=False)

    assert recorded == [("stop", False), ("stop", False)]


def test_server_heartbeat_pings_only_from_node_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    requests = []

    class Stop:
        waits = 0

        def wait(self, timeout: int) -> bool:
            assert timeout == 60  # noqa: PLR2004
            self.waits += 1
            return self.waits == 2  # noqa: PLR2004

        def set(self) -> None:
            pass

    class Thread:
        def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
            assert daemon
            self.target, self.args = target, args

        def start(self) -> None:
            self.target(*self.args)

        def join(self) -> None:
            pass

    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr(slurm_job.threading, "Event", Stop)
    monkeypatch.setattr(slurm_job.threading, "Thread", Thread)
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", SimpleNamespace(request=lambda *args: requests.append(args)))

    with slurm_job._server_heartbeat(enabled=True):
        pass
    monkeypatch.setenv("SLURM_LOCALID", "1")
    with slurm_job._server_heartbeat(enabled=True):
        pass

    assert requests == [("ping", 1)]


def test_valgrind_server_uses_phase_in_log_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(slurm_job.shutil, "which", lambda _name: "/usr/bin/valgrind")
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", SimpleNamespace(require_stopped=lambda: None))
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing", SimpleNamespace(SERVER_PATH=tmp_path / "server"))
    def popen(*_args: object, **kwargs: object) -> None:
        assert "--errors-for-leak-kinds=definite,indirect" in _args[0]
        assert {
            name: kwargs["env"][name]
            for name in slurm_job._VALGRIND_NATIVE_ENV
        } == slurm_job._VALGRIND_NATIVE_ENV
        assert "NATIVE_CKPT_OTEL_FILE" not in kwargs["env"]
        raise RuntimeError("popen reached")

    monkeypatch.setattr(slurm_job.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="popen reached"), slurm_job.ValgrindFlavor().server("test", 2):
        pass


def test_valgrind_server_waits_for_listener_before_releasing_peers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(slurm_job.shutil, "which", lambda _name: "/usr/bin/valgrind")
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing", SimpleNamespace(SERVER_PATH=tmp_path / "server"))
    calls: list[object] = []

    class Process:
        finished = False

        def poll(self):
            return 0 if self.finished else None

        def wait(self, *, timeout: int):
            calls.append(("wait", timeout))
            self.finished = True
            return 0

        def kill(self):
            calls.append("kill")

    def popen(*_args: object, **kwargs: object) -> Process:
        stderr = kwargs["stderr"]
        assert isinstance(stderr, io.TextIOBase)
        calls.append(("stdio", kwargs["stdout"], stderr.name))
        return Process()

    reachable = iter((False, True))
    checkpoint_server = SimpleNamespace(
        require_stopped=lambda: calls.append("stopped"),
        reachable=lambda: calls.append("reachable") or next(reachable),
        request=lambda *args: calls.append(("request", *args)),
    )
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", checkpoint_server)
    monkeypatch.setattr(slurm_job.subprocess, "Popen", popen)
    monkeypatch.setattr(slurm_job.time, "sleep", lambda _seconds: None)

    with slurm_job.ValgrindFlavor().server("test", 2):
        assert (tmp_path / slurm_job.SYNC_DIR_NAME / "123-test-node.ready").exists()

    assert calls == [
        "stopped",
        ("stdio", subprocess.DEVNULL, str(tmp_path / "test_server_phase2.node.log")),
        "reachable",
        "reachable",
        ("request", "spindown", 3600),
        ("wait", 3600),
    ]


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "name": "ultra3-64n",
        "work_dir": tmp_path,
        "image": None,
        "megatron_commit": DEFAULT_MEGATRON_COMMIT,
        "megatron_repo_url": None,
        "nvrx_commit": DEFAULT_NVRX_COMMIT,
        "nvrx_repo_url": "https://github.com/NVIDIA/nvidia-resiliency-ext.git",
        "async_compare": None,
        "recorded_baseline": False,
        "async_strategy_compare": None,
        "compare_extra_args": None,
        "compare_extra_args_phase1": None,
        "reference_extra_args": None,
        "baseline_name": None,
        "parallel_save_compare": None,
        "fresh_structure_reference": None,
        "flavor": None,
        "run": None,
        "baseline": None,
        "checkpoints_per_run": None,
        "checkpoint_interval": None,
        "repeat": 1,
        "time_minutes": None,
        "clean_run": False,
        "background": False,
        "dry_run": False,
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


def test_worker_tmpdir_uses_short_container_path(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURM_PROCID", "7")

    configured = slurm_job._configure_worker_tmpdir()

    assert configured == Path("/tmp")
    assert configured.is_dir()
    assert slurm_job.os.environ["TMPDIR"] == str(configured)


def test_torch_serialization_compat_supplies_upstream_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    torch = ModuleType("torch")
    torch.__path__ = []  # type: ignore[attr-defined]
    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []  # type: ignore[attr-defined]
    torch.utils = torch_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.delitem(sys.modules, "torch.utils.serialization", raising=False)
    monkeypatch.delitem(sys.modules, "torch.utils.serialization.config", raising=False)

    assert slurm_job._install_torch_serialization_compat()
    package = importlib.import_module("torch.utils.serialization")
    config = importlib.import_module("torch.utils.serialization.config")
    assert package.config is config
    assert config.save.compute_crc32 is True
    assert config.save.storage_alignment == slurm_job._TORCH_STORAGE_ALIGNMENT
    assert config.save.use_pinned_memory_for_d2h is False
    assert config.load.mmap is False
    assert config.load.calculate_storage_offsets is False
    assert not slurm_job._install_torch_serialization_compat()


def test_run_options_apply_default_env_cli_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    defaults, _ = slurm_job._parse_args(slurm_job._parser(), ["pretrain-native"])
    assert (defaults.nodes, defaults.config, defaults.repeat, defaults.background) == (1, "pico", 1, False)
    assert defaults.flavor is None
    assert defaults.nvrx_commit == DEFAULT_NVRX_COMMIT

    monkeypatch.setenv("SLURMJOB_OPT_NODES", "8")
    monkeypatch.setenv("SLURMJOB_OPT_CONFIG", "nano3")
    monkeypatch.setenv("SLURMJOB_OPT_FLAVOR", "perf,detr")
    monkeypatch.setenv("SLURMJOB_OPT_BACKGROUND", "1")
    from_env, _ = slurm_job._parse_args(slurm_job._parser(), ["pretrain-native"])
    assert (from_env.nodes, from_env.config, from_env.flavor, from_env.background) == (
        8, "nano3", ["perf", "detr"], True,
    )

    cli, _ = slurm_job._parse_args(
        slurm_job._parser(),
        ["--nodes", "16", "--config", "nano4", "--flavor", "vg", "--no-background"],
    )
    assert (cli.nodes, cli.config, cli.flavor, cli.background) == (16, "nano4", ["vg"], False)


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
    install_dir = tmp_path / "deps/python"
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(install_dir)])
    monkeypatch.setattr(slurm_job, "WORK_ROOT", tmp_path)
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

    # What a run produces is its own; what it merely consumes is shared, so a
    # second run reads the checkout, build and compiled kernels the first left.
    expected = tmp_path / "ultra3-64n"
    assert expected == slurm_job.RUN_ROOT
    assert expected / "logs" == slurm_job.RESULTS_DIR
    assert slurm_job._run_dir("ultra3fp8_64x4") == expected
    assert Path(slurm_job.os.environ["SLURMJOB_OPT_RUN_DIR"]) == expected

    assert tmp_path == slurm_job.WORK_ROOT
    assert Path(slurm_job.os.environ["SLURMJOB_OPT_WORK_ROOT"]) == tmp_path
    for name, relative in (
        ("MEGATRON_ROOT", f"deps/megatron-lm-{DEFAULT_MEGATRON_COMMIT[:8]}"),
        ("SLURMJOB_OPT_PYTHON_INSTALL_DIR", "deps/python"),
        ("VCPKG_ROOT", "deps/vcpkg"),
        ("TRITON_CACHE_DIR", "caches/triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "caches/torchinductor"),
        ("TORCH_EXTENSIONS_DIR", "caches/torch-extensions"),
    ):
        assert Path(slurm_job.os.environ[name]) == tmp_path / relative, name
    # The build is shared too, but only among runs of the same revision.
    build = Path(slurm_job.os.environ["MEGATRON_NATIVE_BUILD_DIR"])
    assert build.parent == tmp_path / "build"
    assert build.name.startswith("nvrx_checkpointing-")


def test_unnamed_runs_generate_commit_keyed_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    commit = "1" * 40
    monkeypatch.setenv("CI_JOB_ID", "98765")

    slurm_job._configure_run(_args(tmp_path, name=None, work_dir=None, megatron_commit=commit))

    # A run names the sources it was told to deviate on, and only those.
    name = f"98765_ultra3fp8_64n-perf-mlm-{commit[:8]}"
    assert name == slurm_job.RUN_NAME
    assert slurm_job.REPO_ROOT / "workdir" / name == slurm_job.RUN_ROOT
    assert Path(slurm_job.os.environ["MEGATRON_ROOT"]) == slurm_job.WORK_ROOT / f"deps/megatron-lm-{commit[:8]}"


def test_reading_a_checkpoint_is_given_what_wrote_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    megatron = tmp_path / "megatron-lm"
    (megatron / ".git").mkdir(parents=True)
    monkeypatch.setenv("MEGATRON_ROOT", str(megatron))
    monkeypatch.setenv("PYTHONPATH", "/already/here")
    captured: dict = {}
    monkeypatch.setattr(
        slurm_job.subprocess, "run",
        lambda *_a, **kwargs: captured.update(kwargs) or SimpleNamespace(returncode=0),
    )

    slurm_job._manifest(tmp_path / "iter_0000100")

    # The tool imports what wrote the checkpoint, so it is handed the checkout
    # rather than left to find one; what was already there is kept.
    path = captured["env"]["PYTHONPATH"].split(":")
    assert path[0] == str(megatron)
    assert "/already/here" in path
    # This package comes from the installed wheel. The checkout ahead of it
    # would import from source, which builds the extension rather than loading
    # it, and the build fails on a compute node with no network.
    assert str(slurm_job.REPO_ROOT.parent) not in path


def test_only_a_run_that_will_be_read_records_its_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    written: list[str] = []
    monkeypatch.setattr(slurm_job, "_manifest_batch", lambda *_a, **_k: [])
    monkeypatch.setattr(slurm_job, "_run_routine", lambda *_a, **_k: None)
    monkeypatch.setattr(slurm_job, "_compare_runs", lambda *_a, **_k: None)
    monkeypatch.setattr(slurm_job, "_write_checkpoint_digest", lambda _run, stem, *_a: written.append(stem))

    def records(flavors: list[str], runs: list[str] | None, baseline: Path | None = None) -> bool:
        written.clear()
        args = _args(tmp_path, config="nano3", nodes=8, flavors=flavors, flavor=flavors, run=runs, baseline=baseline)
        slurm_job._configure_run(args)
        slurm_job._perform_runs("nano3_8x4", args)
        return bool(written)

    # Recording what a checkpoint holds costs a pass over every tensor, so a
    # lone measured run with nothing to diff against does not pay for it.
    assert not records(["perf", "native"], None)
    # Anything that will be read does: a flavor that exists to compare, a
    # second run to compare with, or a baseline to compare against.
    assert records(["detr", "native"], None)
    assert records(["perf", "native"], ["a", "b"])
    assert records(["perf", "native"], None, tmp_path)


def test_the_run_settings_can_be_argued_or_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    def settle(argv: list[str]) -> dict[str, str]:
        args, _ = slurm_job._parse_args(slurm_job._parser(), argv)
        slurm_job._export_settings(args)
        return {name: slurm_job.os.environ.get(name, "") for name in ("PLATFORM", "CLUSTER", "SLURM_ACCOUNT")}

    # Named on the command line, with nothing in the environment.
    for name in ("PLATFORM", "CLUSTER", "SLURM_ACCOUNT"):
        monkeypatch.delenv(name, raising=False)
    assert settle(["--platform", "gb200", "--cluster", "awscmh", "--slurm-account", "acct"]) == {
        "PLATFORM": "gb200", "CLUSTER": "awscmh", "SLURM_ACCOUNT": "acct",
    }
    assert slurm_job.platform_tag() == "gb200"
    assert slurm_job.cluster_tag() == "awscmh"

    # An environment that already carries them still works without arguments.
    monkeypatch.setenv("PLATFORM", "h100")
    monkeypatch.setenv("CLUSTER", "lyris")
    monkeypatch.setenv("SLURM_ACCOUNT", "from-env")
    assert settle([]) == {"PLATFORM": "h100", "CLUSTER": "lyris", "SLURM_ACCOUNT": "from-env"}

    # An argument wins over the environment, as every other option does.
    assert settle(["--platform", "gb200", "--slurm-account", "argued"]) == {
        "PLATFORM": "gb200", "CLUSTER": "lyris", "SLURM_ACCOUNT": "argued",
    }


def test_scoped_runs_inherit_common_options_and_override_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    candidate = "735068d5df9b450f32f33b4f42e449bf126c3683"
    args, args.extra_args = slurm_job._parse_args(
        slurm_job._parser(),
        [
            "--nodes=8",
            "--config=nano3",
            "--flavor=detr+nvrx",
            "--micro-batch-size=2",
            "--run=ref",
            "--run=cmp",
            f"--nvrx-commit={candidate}",
            "--micro-batch-size=4",
            "--run=working-tree",
            "--nvrx-commit=HEAD",
        ],
    )
    args.flavors = slurm_job.FlavorSelection(args.flavor)

    ref, cmp, working_tree = slurm_job._run_specs(args)

    assert [spec.name for spec in (ref, cmp, working_tree)] == ["ref", "cmp", "working-tree"]
    assert [spec.flavors.names for spec in (ref, cmp, working_tree)] == [
        ("detr", "nvrx"),
        ("detr", "nvrx"),
        ("detr", "nvrx"),
    ]
    assert [spec.args.nvrx_commit for spec in (ref, cmp, working_tree)] == [
        DEFAULT_NVRX_COMMIT,
        candidate,
        "HEAD",
    ]
    assert ref.args.extra_args == ["--micro-batch-size=2"]
    assert cmp.args.extra_args == ["--micro-batch-size=2", "--micro-batch-size=4"]


def test_allocation_options_must_precede_scoped_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args, args.extra_args = slurm_job._parse_args(slurm_job._parser(), ["--run=ref", "--background"])
    args.flavors = slurm_job.FlavorSelection(args.flavor)

    with pytest.raises(ValueError, match=r"allocation option.*--background"):
        slurm_job._run_specs(args)


def test_the_platform_names_its_gpus_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    # GPUs per node is a property of the hardware, so naming the platform is
    # enough and a run only states it to deviate.
    monkeypatch.setenv("PLATFORM", "gb200")
    assert slurm_job.platform_gpus_per_node() == 4  # noqa: PLR2004
    monkeypatch.setenv("PLATFORM", "h100")
    assert slurm_job.platform_gpus_per_node() == 8  # noqa: PLR2004
    monkeypatch.setenv("PLATFORM", "gb300")
    assert slurm_job.platform_gpus_per_node() == 4  # noqa: PLR2004
    monkeypatch.setenv("PLATFORM", "gb200")
    assert _validated_args(slurm_job, ["--config", "pico", "--nodes", "1"]).gpus_per_node == 4  # noqa: PLR2004
    assert _validated_args(
        slurm_job, ["--config", "pico", "--nodes", "1", "--gpus-per-node", "2"],
    ).gpus_per_node == 2  # noqa: PLR2004

    # A platform the matrix does not name cannot supply one.
    monkeypatch.setenv("PLATFORM", "unknown")
    assert slurm_job.platform_gpus_per_node() is None


def test_an_unset_platform_names_itself_not_the_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.delenv("PLATFORM", raising=False)

    # Falling back to the hostname reads as a missing baseline for a platform
    # nobody named, so the message says which variable was not set.
    with pytest.raises(RuntimeError, match=r"PLATFORM is unset"):
        slurm_job.PerfReport._platform_bandwidth_baseline(slurm_job.platform_tag())

    monkeypatch.setenv("PLATFORM", "unknown")
    with pytest.raises(RuntimeError, match=r"Known platforms: devbox, gb200, gb300, h100"):
        slurm_job.PerfReport._platform_bandwidth_baseline(slurm_job.platform_tag())

    monkeypatch.setenv("PLATFORM", "gb200")
    assert slurm_job.PerfReport._platform_bandwidth_baseline(slurm_job.platform_tag()) > 0


def test_a_ci_job_holds_the_runs_it_owns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    monkeypatch.setenv("CI_JOB_ID", "389645773")
    args = _args(
        tmp_path, config="nano3", nodes=8, name=None, flavors=["detr"],
        flavor=["detr"], run=["ref:nvrx", "cmp:native"],
    )
    slurm_job._configure_run(args)

    # The job's id leads its own directory, which a retry of the same job keeps
    # apart from this one. Its runs sit inside it, named for what they vary.
    assert slurm_job.RUN_ROOT.name == "389645773_nano3_8n-detr"
    for spec in slurm_job._run_specs(args):
        with slurm_job._in_run(spec, args):
            assert slurm_job.RUN_ROOT.parent.name == "389645773_nano3_8n-detr"
    names = []
    for spec in slurm_job._run_specs(args):
        with slurm_job._in_run(spec, args):
            names.append(slurm_job.RUN_ROOT.name)
    assert names == ["nvrx.ref", "native.cmp"]

    # A manual run has no job to belong to and says so by omission.
    monkeypatch.delenv("CI_JOB_ID")
    manual = _args(
        tmp_path, config="nano3", nodes=8, name=None, flavors=["detr"],
        flavor=["detr"], run=["ref:nvrx", "cmp:native"],
    )
    slurm_job._configure_run(manual)
    assert slurm_job.RUN_ROOT.name == "nano3_8n-detr"
    with slurm_job._in_run(slurm_job._run_specs(manual)[0], manual):
        assert slurm_job.RUN_ROOT.relative_to(tmp_path).as_posix() == "nano3_8n-detr/nvrx.ref"

    # Scoped revisions identify child directories while common options identify the allocation directory.
    candidate = "735068d5df9b450f32f33b4f42e449bf126c3683"
    scoped = _args(
        tmp_path, config="nano3", nodes=8, name=None, flavors=["detr", "nvrx"],
        flavor=["detr", "nvrx"], run=["ref", "cmp"],
    )
    scoped.run_blocks = (
        slurm_job.RunBlock("ref"),
        slurm_job.RunBlock("cmp", (f"--nvrx-commit={candidate}",)),
    )
    slurm_job._configure_run(scoped)
    assert slurm_job.RUN_ROOT.name == "nano3_8n-detr-nvrx"
    assert [slurm_job._run_dir_name(spec) for spec in slurm_job._run_specs(scoped)] == [
        "ref",
        "nvrx-735068d5.cmp",
    ]


def test_a_run_on_pinned_sources_names_only_what_it_is(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    slurm_job._configure_run(_args(tmp_path, name=None, config="nano3", nodes=8, flavors=["detr"]))
    assert slurm_job.RUN_NAME == "nano3_8n-detr"

    # The extension in the image says nothing; a commit given for it does.
    slurm_job._configure_run(
        _args(tmp_path, name=None, config="nano3", nodes=8, flavors=["detr"], nvrx_commit="c" * 40),
    )
    assert slurm_job.RUN_NAME == "nano3_8n-detr-nvrx-cccccccc"

def test_run_name_template_expands_before_paths_are_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(
        tmp_path,
        name="{flavors}_{config}_{nodes}n_mlm.{mlm_abbreviated_sha_commit}_nvrx.{nvrx_abbreviated_sha_commit}",
        megatron_commit="1" * 40,
        nvrx_commit="a" * 40,
    )

    slurm_job._configure_run(args)

    expected = "perf_ultra3fp8_64n_mlm.11111111_nvrx.aaaaaaaa"
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
    monkeypatch.setenv("CLUSTER", "cwdfw")
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
    assert '--target "${SLURMJOB_OPT_RUN_DIR}/deps/nvidia-resiliency-ext-${NVRX_COMMIT}"' in reproducer

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


def test_nvrx_head_installs_a_snapshot_without_modifying_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installer = _load_run_dependency()
    source = tmp_path / "nvrx"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    tracked = source / "tracked"
    tracked.write_text("committed\n")
    subprocess.run(["git", "add", "tracked"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=source, check=True)
    tracked.write_text("locally modified\n")
    (source / "untracked").write_text("also included\n")
    before_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True)
    before_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    target = tmp_path / "installed"
    commands: list[list[str]] = []
    real_run = subprocess.run

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        command = [str(item) for item in command]
        commands.append(command)
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=".git\n", stderr="")
        if "pip" in command:
            staging = Path(command[command.index("--target") + 1])
            snapshot = Path(command[-1])
            assert snapshot != source
            assert (snapshot / "tracked").read_text() == "locally modified\n"
            assert (snapshot / "untracked").read_text() == "also included\n"
            package = staging / "nvidia_resiliency_ext"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=str(target / "nvidia_resiliency_ext/__init__.py") + "\n",
            stderr="",
        )

    monkeypatch.setattr(installer.subprocess, "run", run)

    installer.install_nvrx("https://example.invalid/nvrx.git", "HEAD", target, source)

    after_head = real_run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True).stdout
    after_status = real_run(["git", "status", "--porcelain"], cwd=source, capture_output=True, text=True, check=True).stdout
    assert after_head == before_head
    assert after_status == before_status
    git_commands = [command for command in commands if command[0].endswith("git")]
    assert len(git_commands) == 1
    assert "rev-parse" in git_commands[0]


def test_megatron_head_does_not_modify_the_checkout(tmp_path: Path) -> None:
    source = tmp_path / "megatron"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    tracked = source / "tracked"
    tracked.write_text("committed\n")
    subprocess.run(["git", "add", "tracked"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=source, check=True)
    tracked.write_text("locally modified\n")
    before_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True)
    before_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/megatron_root.sh"), str(source)],
        env={"PATH": "/usr/bin:/bin", "MEGATRON_COMMIT": "HEAD"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert Path(result.stdout.strip()) == source.resolve()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True) == before_head
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True) == before_status


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


def test_checkpoint_cycle_extends_sample_horizon_for_large_allocations(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    config = slurm_job.TrainConfig(
        64,
        4,
        tmp_path / "checkpoints",
        tmp_path / "phase1-checkpoints",
        "ultra3fp8",
        "native",
        2,
    )

    opts = slurm_job._train_options("ultra3fp8_64x4", config)

    assert opts[opts.index("--exit-interval") + 1] == "750"
    assert opts[opts.index("--train-samples") + 1] == "192000"


@pytest.mark.parametrize(
    ("repeat", "expected"),
    [
        (1, [(1, 0), (2, 0)]),
        (3, [(1, 0), (2, 0), (2, 1), (2, 2)]),
    ],
)
def test_repeat_runs_phase1_once_then_cycles_phase2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repeat: int,
    expected: list[tuple[int, int]],
) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, repeat=repeat)
    args.flavors = slurm_job.FlavorSelection(["perf", "native"])
    calls = []
    child_envs = []

    def phase_command(_cfg: str, _args: object, _provider: str, phase: int, cycle: int = 0, **_kwargs: object):
        env = {}
        child_envs.append(env)
        calls.append((phase, cycle))
        return ["true"], env

    monkeypatch.setattr(
        slurm_job,
        "_phase_command",
        phase_command,
    )
    monkeypatch.setattr(slurm_job, "wait_for_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(slurm_job.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    slurm_job._run_routine("nano3_4x4", args, "nvrx")

    assert calls == expected
    expected_tags = ["phase1", *(f"cycle{cycle + 1}" for _phase, cycle in expected[1:])]
    assert [Path(env["NATIVE_CKPT_SERVER_LOG"]).name for env in child_envs] == [
        f"checkpoint_server.{tag}.%h.%p.log" for tag in expected_tags
    ]


def test_launcher_restart_resumes_latest_complete_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, checkpoints_per_run=3, repeat=1)
    args.flavors = slurm_job.FlavorSelection(["perf", "native"])
    slurm_job._configure_run(args)
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "16")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    run = slurm_job._run_dir("ultra3fp8_64x4")
    phase1 = run / "phase1-checkpoints/iter_0000100"
    phase1.mkdir(parents=True)
    (phase1 / ".metadata").touch()
    for iteration in (250, 500):
        checkpoint = run / f"checkpoints/iter_{iteration:07d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").touch()
    incomplete = run / "checkpoints/iter_0000750"
    incomplete.mkdir(parents=True)

    start = slurm_job._routine_start("ultra3fp8_64x4", args, args.flavors.policy)

    assert start == slurm_job.RoutineStart(
        phase=2,
        cycle=0,
        resume_iteration=500,
        checkpoints=1,
    )
    (incomplete / ".metadata").touch()
    assert slurm_job._routine_start("ultra3fp8_64x4", args, args.flavors.policy).complete


@pytest.mark.parametrize(
    "case",
    [(["perf", "native"], None, 3), (["detr"], 1, 1), (["detr"], None, 3)],
)
def test_worker_preserves_checkpoint_cycle_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[list[str], int | None, int],
) -> None:
    slurm_job = _load_slurm_job()
    flavors, checkpoints, repeat = case
    args = _args(tmp_path, checkpoints_per_run=checkpoints, repeat=repeat, flavor=flavors, job_array_segment=0)
    args.flavors = slurm_job.FlavorSelection(flavors)
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "0")
    calls = []
    monkeypatch.setattr(slurm_job, "_run_routine",
                        lambda _cfg, run_args, _provider, _label="": calls.append(
                            (run_args.checkpoints_per_run, run_args.repeat)))
    monkeypatch.setattr(slurm_job, "_manifest_batch", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(slurm_job, "_write_checkpoint_digest", lambda *_args, **_kwargs: None)
    uninstalls = []
    monkeypatch.setattr(slurm_job, "uninstall_nvrx_for_native", lambda: uninstalls.append("native"))
    monkeypatch.setattr(slurm_job, "CHECKPOINT_SERVER", SimpleNamespace(require_stopped=lambda: None))

    slurm_job._worker(args)

    assert calls == [(checkpoints, repeat)]
    assert uninstalls == (["native"] if "native" in flavors else [])
    assert args.checkpoints_per_run == checkpoints
    assert args.repeat == repeat


def test_worker_failure_requests_nonblocking_local_server_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, flavors=["perf", "native"])
    args.job = "pretrain-native"
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "0")
    args.flavors = slurm_job.FlavorSelection(args.flavors)
    monkeypatch.setattr(
        slurm_job, "_run_routine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    requests = []
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr(slurm_job, "uninstall_nvrx_for_native", lambda: None)
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(require_stopped=lambda: None, request=lambda *request: requests.append(request)),
    )

    with pytest.raises(RuntimeError, match="failed"):
        slurm_job._worker(args)

    assert requests == [("stop", 5)]


def test_worker_failure_preserves_original_error_when_local_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, flavors=["perf", "native"])
    args.job = "pretrain-native"
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "0")
    args.flavors = slurm_job.FlavorSelection(args.flavors)
    monkeypatch.setattr(
        slurm_job, "_run_routine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("workload failed")),
    )
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr(slurm_job, "uninstall_nvrx_for_native", lambda: None)
    monkeypatch.setattr(
        slurm_job,
        "CHECKPOINT_SERVER",
        SimpleNamespace(
            require_stopped=lambda: None,
            request=lambda *_args: (_ for _ in ()).throw(ConnectionError("daemon stopped")),
        ),
    )

    with pytest.raises(RuntimeError, match="workload failed"):
        slurm_job._worker(args)


def test_uninstall_nvrx_for_native_is_serialized_and_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    site = tmp_path / "site-packages"
    package = site / "nvidia_resiliency_ext"
    metadata = site / "nvidia_resiliency_ext-0.6.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    monkeypatch.setattr(slurm_job, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node.example")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    def distribution(name: str) -> object:
        assert name == "nvidia-resiliency-ext"
        if metadata.exists():
            return SimpleNamespace(_path=metadata)
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(slurm_job.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(
        slurm_job.importlib.util,
        "find_spec",
        lambda _name: (
            SimpleNamespace(submodule_search_locations=[str(package)], origin=None)
            if package.exists()
            else None
        ),
    )

    slurm_job.uninstall_nvrx_for_native()

    assert (tmp_path / ".nvrx-uninstall-node.lock").exists()
    assert not package.exists()
    assert not metadata.exists()
    assert (site / ".nvidia_resiliency_ext.ci-disabled-123-node").is_dir()
    assert (site / ".nvidia_resiliency_ext-0.6.0.dist-info.ci-disabled-123-node").is_dir()


def test_valgrind_keeps_checkpoints_in_one_training_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    args = _args(
        tmp_path,
        flavors=["detr", "vg"],
    )
    args.flavors = slurm_job.FlavorSelection(args.flavors)
    calls = []
    monkeypatch.setattr(
        slurm_job,
        "_run_routine",
        lambda _cfg, run_args, _provider, _label="": calls.append((run_args.checkpoints_per_run, run_args.repeat)),
    )

    args.flavors.policy.run_worker("nano3_8x4", args, "native")

    assert calls == [(None, 1)]


def _validated_args(slurm_job, argv: list[str]):  # noqa: ANN001
    parser = slurm_job._parser()
    args, args.extra_args = slurm_job._parse_args(parser, argv)
    args.flavors = slurm_job.FlavorSelection(args.flavor)
    slurm_job._selected_flavors(args)
    slurm_job._validate_args(parser, args)
    return args


def test_a_run_trains_on_one_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setattr(slurm_job, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(slurm_job, "WORK_ROOT", tmp_path)

    args = _args(
        tmp_path, config="nano3", nodes=8, flavors=["detr"],
        megatron_repo_url="https://example/fork.git", megatron_commit="a" * 40, nvrx_commit="b" * 40,
    )
    toolchain = slurm_job._toolchain("nano3_8x4", args)
    assert (toolchain.megatron_url, toolchain.megatron_commit) == ("https://example/fork.git", "a" * 40)

    # Both routines train on it, so a difference between them is never a
    # difference in sources; comparing two source trees compares two runs.
    environments = []
    for _variant in ("reference", "compare"):
        env = {}
        toolchain.apply(env)
        environments.append(env)
    assert environments[0] == environments[1]
    assert environments[0]["MEGATRON_COMMIT"] == "a" * 40
    assert environments[0]["PYTHONPATH"].split(":")[0] == toolchain.nvrx_target.as_posix()

    # Nothing names a second toolchain for a run to hold beside this one.
    parser = slurm_job._parser()
    assert not [action for action in parser._actions if action.dest.endswith(("_commit_compare", "_repo_url_compare"))]


def test_naming_a_second_toolchain_fails_before_the_allocation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("GPUS_PER_NODE", "4")

    # An option the parser does not know reaches Megatron, so a retired one has
    # to be caught here rather than in phase 1 with the nodes already held.
    with pytest.raises(SystemExit):
        _validated_args(slurm_job, ["--config", "nano3", "--nodes", "8", "--flavor", "detr",
            "--megatron-commit-compare", "a" * 40,
        ])
    error = capsys.readouterr().err
    assert "--megatron-commit-compare: a run trains on one toolchain" in error
    assert "--diff-digests" in error


_ASYNC_OPTS = ("--async-save", "--use-persistent-ckpt-worker", "--async-ckpt-use-cpu-shm")

def test_deterministic_drops_async_checkpointing_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["detr"])
    slurm_job._configure_run(args)

    cmd, _env = slurm_job._phase_command("nano3_8x4", args, "nvrx", 2)

    assert not [opt for opt in _ASYNC_OPTS if opt in cmd]
    assert "--deterministic-mode" in cmd


def test_perf_enables_parallel_checkpoint_optimizations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["perf"])
    slurm_job._configure_run(args)

    for provider in ("native", "nvrx"):
        phase1 = slurm_job._phase_command("nano3_8x4", args, provider, 1)[0]
        phase2 = slurm_job._phase_command("nano3_8x4", args, provider, 2)[0]

        for command in (phase1, phase2):
            assert command.count("--ckpt-fully-parallel-save-process-group=ep_dp") == 1
            assert command.count("--ckpt-fully-parallel-load-process-group=ep_dp") == 1
            assert "--ckpt-fully-parallel-load-per-rank-objects" in command
            assert "--ckpt-drop-redundant-extra-state" in command
            assert "--no-ckpt-load-validate-sharding-integrity" in command
            cache_options = [option for option in command if option.startswith("--ckpt-pg-tensors-cache-path=")]
            if provider == "native":
                assert not cache_options
            else:
                assert len(cache_options) == 1
                assert cache_options[0].endswith("/pg-tensors-cache")
        assert ("--ckpt-pg-tensors-cache-create" in phase1) == (provider == "nvrx")
        assert "--ckpt-pg-tensors-cache-create" not in phase2


def test_both_providers_export_megatron_otel_to_local_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["perf", "native"])
    slurm_job._configure_run(args)

    _cmd, native_env = slurm_job._phase_command("nano3_8x4", args, "native", 1)
    _cmd, nvrx_env = slurm_job._phase_command("nano3_8x4", args, "nvrx", 1)

    assert {name: native_env[name] for name in slurm_job._MEGATRON_OTEL_ENV_DEFAULTS} == {
        "MEGATRON_OTEL_ENABLED": "1",
        "MEGATRON_OTEL_METRICS_ENABLED": "1",
        "MEGATRON_OTEL_EXPORT_STRATEGY": "all_ranks",
        "MEGATRON_OTEL_SPAN_GROUPS": "default,load_checkpoint",
        "MEGATRON_OTEL_TRACES_ENABLED": "1",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4100",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }
    assert {name: nvrx_env[name] for name in slurm_job._MEGATRON_OTEL_ENV_DEFAULTS} == {
        name: native_env[name] for name in slurm_job._MEGATRON_OTEL_ENV_DEFAULTS
    }


def test_periodic_memory_sampling_is_a_manual_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["perf", "native"])
    slurm_job._configure_run(args)

    _cmd, default_env = slurm_job._phase_command("nano3_8x4", args, "native", 1)
    assert "NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS" not in default_env

    monkeypatch.setenv("NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS", "7")
    _cmd, overridden_env = slurm_job._phase_command("nano3_8x4", args, "native", 1)
    assert overridden_env["NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS"] == "7"
    assert "NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS" in slurm_job._RUN_ENV_NAMES


@pytest.mark.parametrize(
    "case",
    [
        ("9.0", None, "0"),
        ("9.0", "1", "1"),
        ("10.0", None, None),
        ("", None, None),
    ],
)
def test_hopper_training_uses_flash_attention_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    case: tuple[str, str | None, str | None],
) -> None:
    capability, preset, expected = case
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.delenv("NVTE_FLASH_ATTN_V4", raising=False)
    if preset is not None:
        monkeypatch.setenv("NVTE_FLASH_ATTN_V4", preset)
    monkeypatch.setattr(slurm_job, "_gpu_compute_capability", lambda: capability)
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["perf", "native"])
    slurm_job._configure_run(args)

    for provider in ("native", "nvrx"):
        _cmd, env = slurm_job._phase_command("nano3_8x4", args, provider, 1)
        assert env.get("NVTE_FLASH_ATTN_V4") == expected
        capsys.readouterr()
        slurm_job._print_pretrain_env(env, "phase1")
        dumped = [line for line in capsys.readouterr().out.splitlines() if "NVTE_FLASH_ATTN_V4" in line]
        assert dumped == ([] if expected is None else [f"[slurm_job:env] NVTE_FLASH_ATTN_V4={expected}"])


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (subprocess.CompletedProcess([], 0, stdout="9.0\n"), "9.0"),
        (subprocess.CompletedProcess([], 0, stdout="10.0\n"), "10.0"),
        (subprocess.CompletedProcess([], 9, stdout="NVIDIA-SMI has failed to communicate with the driver\n"), ""),
        (FileNotFoundError("nvidia-smi"), ""),
        (subprocess.TimeoutExpired(["nvidia-smi"], 60), ""),
    ],
)
def test_gpu_compute_capability_reads_the_first_gpu_from_the_driver(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    result: subprocess.CompletedProcess[str] | Exception,
    expected: str,
) -> None:
    slurm_job = _load_slurm_job(gpu_probe=True)
    monkeypatch.setenv("SLURM_PROCID", "0")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if isinstance(result, Exception):
            raise result
        if kwargs.get("check") and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
        return result

    monkeypatch.setattr(slurm_job.subprocess, "run", run)

    with caplog.at_level("WARNING", logger="native_ci"):
        assert slurm_job._gpu_compute_capability() == expected
    assert [command for command, _kwargs in calls] == [
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"],
    ]
    assert calls[0][1]["timeout"]
    warned = "GPU compute capability unavailable" in caplog.text
    assert warned == (expected == "")


@pytest.mark.parametrize("deterministic", [True, False])
def test_only_deterministic_runs_fill_uninitialized_memory(deterministic: bool) -> None:
    slurm_job = _load_slurm_job()
    settings = SimpleNamespace(fill_uninitialized_memory=False)
    torch = SimpleNamespace(utils=SimpleNamespace(deterministic=settings))

    slurm_job._fill_uninitialized_memory_if_deterministic(SimpleNamespace(deterministic_mode=deterministic), torch)

    assert settings.fill_uninitialized_memory is deterministic


@pytest.mark.parametrize(
    "routing",
    [
        ("native", ["perf", "native"], "0", "worker", None, "collector", 1),
        ("nvrx", ["perf"], "0", "worker", "leader:4100", "relay", 1),
        ("nvrx", ["perf"], "4", "leader", "file", "collector", 1),
    ],
)
def test_routine_validates_each_collected_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    routing: tuple[str, list[str], str, str, str | None, str, int],
) -> None:
    slurm_job = _load_slurm_job()
    provider, flavor_names, rank, host, collector, server_role, phase1_checkpoints = routing
    args = _args(tmp_path, repeat=1)
    args.flavors = slurm_job.FlavorSelection(flavor_names)
    child_envs = []
    validated = []
    converted = []
    charts = []
    lifecycle = []
    stops = []

    def phase_command(_cfg: str, _args: object, _provider: str, _phase: int, _cycle: int = 0, **_kwargs: object):
        env = {"SLURM_JOB_ID": "42", "MEGATRON_OTEL_ENABLED": "1", "MASTER_ADDR": "leader"}
        child_envs.append(env)
        return ["true"], env

    def run(*_args: object, env: dict[str, str] | None = None, **_kwargs: object) -> SimpleNamespace:
        if env is not None:
            Path(env["NATIVE_CKPT_OTEL_FILE"]).touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(slurm_job, "RUN_NAME", "otel-run")
    monkeypatch.setenv("SLURM_PROCID", rank)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: host)
    monkeypatch.setitem(
        sys.modules,
        "nvrx_checkpointing",
        SimpleNamespace(checkpoint=SimpleNamespace(ping=lambda: lifecycle.append(("ping", "")))),
    )
    monkeypatch.setattr(slurm_job, "_phase_command", phase_command)
    monkeypatch.setattr(slurm_job, "_sync_ranks", lambda tag: lifecycle.append(("sync", tag)))
    monkeypatch.setattr(slurm_job, "wait_for_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(slurm_job.otel_report, "validate_collected", lambda *values: validated.append(values))
    monkeypatch.setattr(slurm_job.otel_report, "write_process_memory_charts", lambda *values: charts.append(values))
    monkeypatch.setattr(
        slurm_job.otel_report,
        "OtelData",
        lambda paths: SimpleNamespace(paths=paths, records=[("record", {})]),
    )
    monkeypatch.setattr(slurm_job.otel_report, "write_analysis", lambda *_args: None)
    monkeypatch.setattr(slurm_job.otel_to_perfetto, "convert_records", lambda *values: converted.append(values))
    monkeypatch.setattr(slurm_job.subprocess, "run", run)
    monkeypatch.setattr(args.flavors.policy, "stop_server", lambda *values, **_kwargs: stops.append(values))

    slurm_job._run_routine("nano3_4x4", args, provider)

    assert [env["MEGATRON_OTEL_RUN_ID"] for env in child_envs] == [
        "42-otel-run-phase1-sr0-te0",
        "42-otel-run-cycle1-sr0-te0",
    ]
    assert all("NATIVE_CKPT_TRAINING_MEMORY_FILE" not in env for env in child_envs)
    assert sum(event == ("ping", "") for event in lifecycle) == 2  # noqa: PLR2004
    if provider == "native":
        expected_startup = [("ping", ""), ("sync", "phase1-otel-listeners-ready")]
    elif server_role == "collector":
        expected_startup = [
            ("ping", ""),
            ("sync", "phase1-otel-collector-ready"),
            ("sync", "phase1-otel-listeners-ready"),
        ]
    else:
        expected_startup = [
            ("sync", "phase1-otel-collector-ready"),
            ("ping", ""),
            ("sync", "phase1-otel-listeners-ready"),
        ]
    assert lifecycle[: len(expected_startup)] == expected_startup
    assert [env.get("NATIVE_CKPT_OTEL_COLLECTOR") for env in child_envs] == [collector, collector]
    assert [label.rsplit("-", 2)[-2] for _provider, label in stops] == [server_role, server_role]
    validation_calls = [
        (source.paths[0].name, run_id, actual_provider, checkpoints)
        for source, run_id, _nodes, _gpus, actual_provider, checkpoints in validated
    ]
    assert validation_calls == (
        [
            ("native_ckpt_otel.jsonl", "42-otel-run-phase1-sr0-te0", provider, phase1_checkpoints),
            ("native_ckpt_otel.jsonl", "42-otel-run-cycle1-sr0-te0", provider, 3),
        ]
        if rank == "0"
        else []
    )
    assert [output.name for _records, output in converted] == (
        ["native_ckpt_perfetto.json", "native_ckpt_perfetto.json"] if rank == "0" else []
    )
    assert len(charts) == (2 if rank == "0" else 0)


def test_otel_checkpoint_timings_are_grouped_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    args = _args(
        tmp_path,
        config="pico",
        nodes=1,
        gpus_per_node=4,
        flavors=["detr"],
        flavor=["detr"],
        run=["ref:nvrx", "cmp:native"],
    )
    batch_log = tmp_path / "job" / "logs" / "job.log"
    batch_log.parent.mkdir(parents=True)
    reports = []
    for spec in slurm_job._run_specs(args):
        root = slurm_job._argument_log(args, batch_log, spec).parent / "otel"
        for phase in ("cycle1", "phase1"):
            path = root / phase / "native_ckpt_otel.jsonl"
            path.parent.mkdir(parents=True)
            path.touch()
    monkeypatch.setattr(slurm_job.otel_report, "log_checkpoint_timings", lambda *values: reports.append(values))

    slurm_job._log_otel_checkpoint_timings(args, batch_log)

    assert [(path.parent.name, run_id, label) for path, run_id, label in reports] == [
        ("phase1", None, "ref phase1"),
        ("cycle1", None, "ref cycle1"),
        ("phase1", None, "cmp phase1"),
        ("cycle1", None, "cmp cycle1"),
    ]


def test_megatron_otel_validation_requires_every_rank(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    path = tmp_path / "native_ckpt_otel.jsonl"

    def record(rank: int) -> dict:
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "nemo.run.id", "value": {"stringValue": "run-1"}},
                            {"key": "dl.rank", "value": {"intValue": str(rank)}},
                        ],
                    },
                    "scopeSpans": [{"spans": [{"name": "megatron.train.iteration"}]}],
                },
            ],
        }

    def metric_record(rank: int, provider: str = "nvrx") -> dict:
        resource = {
            "attributes": [
                {"key": "nemo.run.id", "value": {"stringValue": "run-1"}},
                {"key": "dl.rank", "value": {"intValue": str(rank)}},
                {"key": "checkpoint.provider", "value": {"stringValue": provider}},
            ],
        }
        metrics = [
            {
                "name": f"nvrx.checkpoint.run.{name}",
                "gauge": {"dataPoints": [{"asInt": "1"}]},
            }
            for name in ("gpu_peak_bytes", "gpu_hbm_peak_bytes", "gpu_total_bytes", "cpu_rss_bytes")
        ]
        return {"resourceMetrics": [{"resource": resource, "scopeMetrics": [{"metrics": metrics}]}]}

    def megatron_metric_record(rank: int) -> dict:
        resource = {
            "attributes": [
                {"key": "nemo.run.id", "value": {"stringValue": "run-1"}},
                {"key": "dl.rank", "value": {"intValue": str(rank)}},
            ],
        }
        metrics = [
            {
                "name": name,
                "gauge": {"dataPoints": [{"asDouble": 1.0, "timeUnixNano": "10"}]},
            }
            for name in slurm_job.otel_report.MEGATRON_PERF_METRICS
        ]
        return {"resourceMetrics": [{"resource": resource, "scopeMetrics": [{"metrics": metrics}]}]}

    checkpoints = [record(rank) for rank in range(8)]
    for checkpoint in checkpoints:
        save = checkpoint["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        save["name"] = "megatron.checkpoint.save"
        save.update(
            {"startTimeUnixNano": "1", "endTimeUnixNano": "2"},
        )
        checkpoint["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
            {
                "name": "megatron.checkpoint.exposed_save",
                "startTimeUnixNano": "1",
                "endTimeUnixNano": "2",
            },
        )
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                *checkpoints,
                *(megatron_metric_record(rank) for rank in range(8)),
                *(metric_record(rank) for rank in range(8)),
            ]
        )
        + "\n",
    )
    slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "nvrx", 1)

    with pytest.raises(RuntimeError, match="checkpoint save span counts mismatch"):
        slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "nvrx", 2)

    with pytest.raises(RuntimeError, match=r"expected \[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11\], found \[0, 1, 2, 3, 4, 5, 6, 7\]"):
        slurm_job.otel_report.validate_collected(path, "run-1", 3, 4, "nvrx", 1)

    incomplete_megatron = megatron_metric_record(7)
    missing_name = incomplete_megatron["resourceMetrics"][0]["scopeMetrics"][0]["metrics"].pop()["name"]
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                *checkpoints,
                *(megatron_metric_record(rank) for rank in range(7)),
                incomplete_megatron,
                *(metric_record(rank) for rank in range(8)),
            ]
        )
        + "\n",
    )
    with pytest.raises(RuntimeError, match=rf"Megatron training metric {re.escape(missing_name)} ranks mismatch"):
        slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "nvrx", 1)

    def native_record(rank: int) -> dict:
        attrs = [
            {"key": f"nvrx.{key}", "value": {"intValue": str(value)}}
            for key, value in {
                "session_id": 0,
                "rank": rank,
                "snapshot_id": 1,
                "wr_tnsr_us": 10,
                "snapshot_us": 11,
                "total_us": 12,
                "num_tensors": 2,
                "total_d2h": 2,
                "total_tensor_bytes": 100,
                "total_written_bytes": 100,
                "avg_d2h_rq": 50,
                "avg_tnsr": 50,
            }.items()
        ]
        attrs.append({"key": "nvrx.bw_mbps", "value": {"doubleValue": 10.0}})
        spans = [
            {
                "name": "ckpt.save",
                "attributes": attrs,
                "startTimeUnixNano": "1",
                "endTimeUnixNano": "2",
            },
        ]
        if rank == 0:
            spans.append(
                {
                    "name": "ckpt.svr.durable",
                    "attributes": [
                        {"key": "nvrx.session_id", "value": {"intValue": "0"}},
                        {"key": "nvrx.snapshot_id", "value": {"intValue": "1"}},
                    ],
                    "startTimeUnixNano": "1",
                    "endTimeUnixNano": "2",
                },
            )
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "nemo.run.id", "value": {"stringValue": "run-1"}},
                        ],
                    },
                    "scopeSpans": [
                        {
                            "spans": spans,
                        },
                    ],
                },
            ],
        }

    native_spans = [native_record(rank) for rank in range(8)]
    native_records = [
        *checkpoints,
        *(megatron_metric_record(rank) for rank in range(8)),
        *(metric_record(rank, "native") for rank in range(8)),
        *native_spans,
    ]
    path.write_text("\n".join(json.dumps(item) for item in native_records) + "\n")
    slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "native", 1)

    native_spans[0]["resourceSpans"][0]["scopeSpans"][0]["spans"].pop()
    path.write_text("\n".join(json.dumps(item) for item in native_records) + "\n")
    with pytest.raises(RuntimeError, match="durable OTLP spans mismatch"):
        slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "native", 1)

    path.write_text("\n".join(json.dumps(item) for item in native_records[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="native checkpoint OTLP span ranks mismatch"):
        slurm_job.otel_report.validate_collected(path, "run-1", 2, 4, "native", 1)


def test_every_variant_is_a_flavor_and_they_mix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")

    def options(*names: str, phase: int = 2) -> list[str]:
        args = _args(tmp_path, config="nano3", nodes=8, flavors=["detr", *names])
        slurm_job._configure_run(args)
        return slurm_job._phase_command("nano3_8x4", args, "nvrx", phase)[0]

    # Each variant is reachable on its own and reaches the training command.
    assert "--async-ckpt-use-cpu-shm" in options("shm")
    assert "--async-save" in options("ipc")
    assert "--async-ckpt-use-cpu-shm" not in options("ipc")
    assert "--no-ckpt-fully-parallel-save" in options("serial")
    assert _CONSTANT_STRUCTURE not in options("no-cnst-strct")
    assert "--ckpt-drop-redundant-extra-state" in options("drop-xstate")

    # And they compose, each contributing its own part.
    both = options("shm", "pg-cache", "drop-xstate")
    assert "--async-ckpt-use-cpu-shm" in both
    assert "--ckpt-fully-parallel-load-per-rank-objects" in both
    assert "--ckpt-drop-redundant-extra-state" in both
    # The cache is filled by the phase that creates it and read by the next.
    assert "--ckpt-pg-tensors-cache-create" in options("pg-cache", phase=1)
    assert "--ckpt-pg-tensors-cache-create" not in options("pg-cache", phase=2)

    # One variant undoing another is refused, deduced from the option lists.
    with pytest.raises(ValueError, match=r"shm adds --async-ckpt-use-cpu-shm, which ipc drops"):
        slurm_job.FlavorSelection(["detr", "shm", "ipc"])
    # Including a pair the old declared groups let through: this writer needs
    # the shared-memory staging the CUDA IPC variant removes.
    with pytest.raises(ValueError, match=r"which ipc drops"):
        slurm_job.FlavorSelection(["detr", "ipc", "pt-async"])
    # A writer and the staging it agrees with still compose.
    slurm_job.FlavorSelection(["detr", "shm", "pt-async"])

def test_one_flavor_option_names_several_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("GPUS_PER_NODE", "4")

    def flavors(*argv: str) -> tuple[str, ...]:
        return _validated_args(
            slurm_job, ["--config", "nano3", "--nodes", "8", *argv],
        ).flavors.names

    joined = flavors("--flavor", "detr+shm+pg-cache")
    assert joined == ("detr", "shm", "pg-cache")
    # One option or several, the combination is the same.
    assert flavors("--flavor", "detr+shm", "--flavor", "pg-cache") == joined
    assert flavors("--flavor", "detr", "--flavor", "shm", "--flavor", "pg-cache") == joined

    with pytest.raises(SystemExit):
        flavors("--flavor", "detr+nonesuch")


def test_declared_flavors_carry_their_yaml_definition() -> None:
    slurm_job = _load_slurm_job()
    declared = slurm_job._load_yaml(slurm_job.FLAVORS_FILE)

    # Every variant that only edits options is declared, not coded.
    coded = {flavor.name for flavor in slurm_job._EXECUTION_MODES}
    assert coded == {"perf", "detr", "vg"}
    assert set(declared) | coded == set(slurm_job._FLAVOR_NAMES)
    assert all(spec.get("summary") for spec in declared.values())

    # A declaration reaches the policy as the edits it names, in scope order.
    policy = slurm_job.FlavorSelection(["detr", "no-cnst-strct", "pg-cache"]).policy
    assert slurm_job.OptionEdit("all", (), ("--ckpt-assume-constant-structure",)) in policy.option_edits
    assert [edit.scope for edit in policy.option_edits] == ["all", "all", "phase1"]

    # An unknown key is a typo, not a variant nobody implemented.
    with pytest.raises(ValueError, match="unknown keys: phase3"):
        slurm_job.DeclaredFlavor("typo", {"phase3": {"add": ["--nope"]}})


def test_a_variant_reaches_both_phases_and_the_structure_cache_holds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="nano3", nodes=8, flavors=["detr", "pt-async"])
    slurm_job._configure_run(args)

    for phase in (1, 2):
        command = slurm_job._phase_command("nano3_8x4", args, "nvrx", phase)[0]
        # Megatron's own writer requires shared-memory staging and rejects a
        # fully parallel save, so naming it settles both.
        assert "--async-strategy=pytorch-async" in command, phase
        assert "--async-ckpt-use-cpu-shm" in command, phase
        assert "--no-ckpt-fully-parallel-save" in command, phase
        assert "--ckpt-fully-parallel-save" not in command, phase
        # The structure cache is on unless a variant drops it.
        assert _CONSTANT_STRUCTURE in command, phase


def test_a_mixin_keeps_its_execution_mode_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    plain = slurm_job.FlavorSelection(["detr"])
    mixed = slurm_job.FlavorSelection(["detr", "pg-cache", "drop-xstate"])
    perf = slurm_job.FlavorSelection(["perf", "pg-cache"])

    # A variant changes the tag but not what the run costs, so the budget holds.
    assert mixed.tag != plain.tag
    budget = slurm_job._time_budget("nano3", plain, "srun", 30)
    assert slurm_job._time_budget("nano3", mixed, "srun", 30) == budget
    # The deterministic entry, not the fallback that cut two runs short at 25 minutes.
    table = slurm_job._load_yaml(slurm_job.CONFIGS_DIR / "base.yaml")["job_time_in_mins"]["srun"]
    assert budget == table["detr"]
    assert budget != table["default"]
    assert slurm_job._time_budget("nano3", perf, "srun", 30) == slurm_job._time_budget(
        "nano3", slurm_job.FlavorSelection(["perf"]), "srun", 30,
    )
    # A table naming the exact combination still wins over the mode.
    valgrind = slurm_job.FlavorSelection(["detr", "vg"])
    assert valgrind.table_keys[0] == "detr-vg"


def test_checkpoint_digest_is_diffable_and_summarised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    run, baseline = tmp_path / "run", tmp_path / "base"
    produced = run / "checkpoints/iter_0000120"
    produced.mkdir(parents=True)
    (produced / "reference.metadata.info").write_text("a.md5sum=1\nb.md5sum=2\nc.md5sum=3\n")
    # Everything the checkpoint holds, not the tensors alone.
    (produced / "common_pt_dump.info").write_text("iteration=120\n")
    recorded = baseline / "checkpoints/iter_0000120"
    recorded.mkdir(parents=True)
    # One entry agrees, one differs, one exists only in the baseline.
    (recorded / "reference.metadata.info").write_text("a.md5sum=1\nb.md5sum=CHANGED\nz.md5sum=9\n")

    slurm_job._write_checkpoint_digest(run, "stem", lambda g, n: baseline / g / n)

    summary = (tmp_path / "logs/stem_checkpoint_summary.log").read_text()
    # A recorded baseline holds the tensor manifest alone, so the common state
    # is reported as unrecorded rather than as one missing entry per key.
    assert "entries=4 baseline=3 missing=1 extra=1 changed=1 unrecorded=common" in summary
    with lzma.open(tmp_path / "logs/stem_checkpoint_digest.xz", "rt") as handle:
        digest = handle.read().splitlines()
    # Sorted, one entry per line, labelled by checkpoint and source so a diff
    # reads cleanly and two runs compare without either naming the other.
    assert digest == [
        "checkpoints/iter_0000120 common/iteration=120",
        "checkpoints/iter_0000120 tensor/a.md5sum=1",
        "checkpoints/iter_0000120 tensor/b.md5sum=2",
        "checkpoints/iter_0000120 tensor/c.md5sum=3",
    ]

    # A run with nothing to compare against still writes its own digest.
    slurm_job._write_checkpoint_digest(run, "alone")
    assert (tmp_path / "logs/alone_checkpoint_summary.log").read_text().strip() == (
        "checkpoints/iter_0000120: entries=4"
    )

    # What names where or how a run wrote is not what it computed. Runs keep
    # their own directories and checkpoint implementation, and DCP may select
    # any rank's replicated common state.
    (produced / "common_pt_dump.info").write_text(
        f"iteration=120\nargs.ckpt_provider=native\nargs.async_save=True\n"
        f"args.async_strategy=pytorch-async\nargs.async_ckpt_use_cpu_shm=True\n"
        f"args.use_persistent_ckpt_worker=True\nargs.ckpt_fully_parallel_save=False\n"
        f"args.ckpt_fully_parallel_save_deprecated=False\n"
        f"args.ckpt_load_validate_sharding_integrity=False\n"
        f"args.rank=2\nargs.local_rank=2\nargs.num_layers=52\n"
        f"args.save={produced}/checkpoints\n",
    )
    slurm_job._write_checkpoint_digest(run, "scoped")
    with lzma.open(tmp_path / "logs/scoped_checkpoint_digest.xz", "rt") as handle:
        scoped = handle.read()
    assert "common/iteration=120" in scoped
    assert "common/args.num_layers=52" in scoped
    for key in slurm_job._RUN_SCOPED_KEYS:
        assert key not in scoped
    assert "args.save" not in scoped


def test_checkpoint_digest_fails_explicitly_when_no_metadata_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    with pytest.raises(RuntimeError, match="no checkpoint metadata found"):
        slurm_job._write_checkpoint_digest(tmp_path / "run", "empty")

    assert not (tmp_path / "logs/empty_checkpoint_digest.xz").exists()


def test_existing_digests_ignore_run_scoped_entries(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    label = "checkpoints/iter_0000120"
    digests = []
    for name, value in (("ref", "False"), ("cmp", "True")):
        path = tmp_path / f"{name}.xz"
        with lzma.open(path, "wt") as handle:
            handle.write(f"{label} common/args.async_save={value}\n")
            handle.write(f"{label} common/args.rank={'0' if name == 'ref' else '2'}\n")
            handle.write(f"{label} common/args.num_layers=52\n")
            handle.write(f"{label} tensor/weight.md5sum=abc\n")
        digests.append(path)

    assert slurm_job._digest_differences(*digests) == 0


def test_a_chunk_is_compared_by_where_it_sits_not_where_it_was_listed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    # Two writers may enumerate the same chunks in their own orders. Here the
    # chunk at offset 128 is listed first by one and second by the other.
    for name, lines in (
        ("ref", ("w.chunks.0.md5sum=A\nw.chunks.0.offsets=[128, 0, 0]\n"
                "w.chunks.1.md5sum=B\nw.chunks.1.offsets=[256, 0, 0]\n")),
        ("cmp", ("w.chunks.0.md5sum=A\nw.chunks.0.offsets=[128, 0, 0]\n"
                "w.chunks.1.md5sum=B\nw.chunks.1.offsets=[256, 0, 0]\n")),
        ("rev", ("w.chunks.0.md5sum=B\nw.chunks.0.offsets=[256, 0, 0]\n"
                "w.chunks.1.md5sum=A\nw.chunks.1.offsets=[128, 0, 0]\n")),
        ("bad", ("w.chunks.0.md5sum=A\nw.chunks.0.offsets=[128, 0, 0]\n"
                "w.chunks.1.md5sum=CHANGED\nw.chunks.1.offsets=[256, 0, 0]\n")),
    ):
        iteration = tmp_path / name / "checkpoints/iter_0000050"
        iteration.mkdir(parents=True)
        (iteration / "reference.metadata.info").write_text(lines)

    def counts(left: str, right: str) -> str:
        a = slurm_job._checkpoint_entries(tmp_path / left / "checkpoints/iter_0000050")
        b = slurm_job._checkpoint_entries(tmp_path / right / "checkpoints/iter_0000050")
        return slurm_job._compare_entries(b, a)

    # Listing the same chunks in a different order is not a difference.
    assert "changed=0" in counts("ref", "cmp")
    assert "changed=0" in counts("ref", "rev")
    # A chunk whose contents really differ still is.
    assert "changed=1" in counts("ref", "bad")


def test_two_runs_compare_through_their_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    for name, tensor, common in (("a", "1", "52"), ("b", "CHANGED", "52"), ("c", "1", "9")):
        iteration = tmp_path / name / "checkpoints/iter_0000120"
        iteration.mkdir(parents=True)
        (iteration / "reference.metadata.info").write_text(f"w.md5sum={tensor}\n")
        (iteration / "common_pt_dump.info").write_text(f"args.num_layers={common}\n")
        slurm_job._write_checkpoint_digest(tmp_path / name, name)

    def diff(left: str, right: str) -> str:
        caplog.clear()
        with caplog.at_level("INFO"), pytest.raises(SystemExit) as exit_code:
            slurm_job._diff_digests(*(tmp_path / f"logs/{s}_checkpoint_digest.xz" for s in (left, right)))
        return f"{caplog.text}rc={exit_code.value.code}"

    assert "PASS: 0 differences" in diff("a", "a")
    assert "rc=0" in diff("a", "a")
    # A tensor difference and a common-state difference both count, which is
    # what a comparison of the tensor manifest alone would have missed.
    changed = diff("a", "b")
    assert "changed=1" in changed
    assert "changed tensor/w.md5sum: baseline='1' candidate='CHANGED'" in changed
    assert "FAIL: 1 differences" in diff("a", "c")
    assert "rc=1" in diff("a", "c")


def test_failure_summary_promotes_the_cause_and_digest_details(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "logs/run.log"
    log.parent.mkdir()
    log.write_text(
        "0: [slurm_job:diff] checkpoints/iter_0000075: entries=10 baseline=10 missing=0 extra=0 changed=1\n"
        "0: [slurm_job:diff] checkpoints/iter_0000075: changed common/optimizer.lr: "
        "baseline='2.025e-06' candidate='1.35e-06'\n"
        "0: RuntimeError: checkpoints differ from reference.xz: cmp (1 differences)\n"
        "0: RuntimeError: training phase 2 failed (rc=1)\n"
        "srun: error: node: task 0: Exited with exit code 1\n",
    )

    with caplog.at_level("ERROR"):
        slurm_job._report_failure_summary(log)

    assert "underlying failure summary" in caplog.text
    assert "checkpoints differ from reference.xz: cmp (1 differences)" in caplog.text
    assert "changed common/optimizer.lr: baseline='2.025e-06' candidate='1.35e-06'" in caplog.text
    assert "training phase 2 failed" not in caplog.text


def test_recover_otel_entry_generates_only_missing_memory_graphs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    cycle = tmp_path / "42_test/logs/otel/cycle1"
    cycle.mkdir(parents=True)
    source = cycle / "native_ckpt_otel.jsonl"
    source.write_text("{}\n")
    calls = []

    def load(paths: list[Path], *, tolerate_truncated_tail: bool = False) -> SimpleNamespace:
        calls.append((paths, tolerate_truncated_tail))
        return SimpleNamespace()

    def write(_data: object, output_dir: Path) -> list[Path]:
        output = output_dir / "process_memory_all.svg"
        output.write_text("partial")
        return [output]

    monkeypatch.setenv("CI_JOB_ID", "42")
    monkeypatch.setattr(slurm_job.otel_report, "OtelData", load)
    monkeypatch.setattr(slurm_job.otel_report, "write_process_memory_charts", write)

    slurm_job._recover_otel_entry([str(tmp_path)])
    slurm_job._recover_otel_entry([str(tmp_path)])

    assert calls == [([source], True)]
    assert (cycle / "process_memory_all.svg").read_text() == "partial"


def test_each_phase_records_the_sources_it_loaded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    root = tmp_path / "deps/megatron-lm-24d6cc494c08"
    root.mkdir(parents=True)
    monkeypatch.setattr(slurm_job, "megatron_root", lambda: root)
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout="24d6cc494c08540b6459810fc9a2e4846090a316\n"),
    )
    nvrx = tmp_path / "deps/nvidia-resiliency-ext-c2b970d3/nvidia_resiliency_ext/__init__.py"
    nvrx.parent.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext", SimpleNamespace(__file__=str(nvrx)))
    monkeypatch.setattr(slurm_job.importlib.metadata, "version", lambda _dist: "0.7.0.dev97+c2b970d")

    slurm_job._print_source_provenance()

    out = capsys.readouterr().out
    assert f"[native_ckpt:sources] megatron={root}@24d6cc494c08540b6459810fc9a2e4846090a316" in out
    assert f"[native_ckpt:sources] nvidia_resiliency_ext={nvrx}@0.7.0.dev97+c2b970d" in out


def test_source_provenance_reports_an_uninstalled_extension(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(slurm_job, "megatron_root", lambda: tmp_path)
    monkeypatch.setattr(slurm_job.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=""))
    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext", None)

    slurm_job._print_source_provenance()

    out = capsys.readouterr().out
    assert f"[native_ckpt:sources] megatron={tmp_path}@unknown" in out
    assert "[native_ckpt:sources] nvidia_resiliency_ext=absent" in out


def test_long_run_options_are_exported_and_override_allocation_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "cwdfw")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, checkpoints_per_run=6, repeat=2, time_minutes=240)
    slurm_job._configure_run(args)

    script, reaper, _log = slurm_job._build_job(args)
    slurm_job.sbatch_run(
        script, slurm_job.RESULTS_DIR / "scheduler.log", extra_sbatch_args=[reaper], dry_run=True,
    )
    reproducer = (tmp_path / "ultra3-64n" / "run.sbatch").read_text()

    assert "#SBATCH --time=04:00:00" in reproducer
    # The array emits one prep step plus separate task-0 and peer launcher
    # steps because task 0 must preserve its rendezvous store on node loss.
    assert reproducer.count("--time=03:55:00") == 3  # noqa: PLR2004
    assert reproducer.count("--time=00:05:00") == 1
    assert "#SBATCH --signal=TERM@300" in reproducer
    assert '#SBATCH --comment=\'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240"' in reproducer
    assert "export SLURMJOB_OPT_CHECKPOINTS_PER_RUN=6" in reproducer
    assert "export SLURMJOB_OPT_REPEAT=2" in reproducer
    assert "export SLURMJOB_OPT_TIME_MINUTES=240" in reproducer
    assert "--checkpoints-per-run" not in reproducer.split("set -Eeuo pipefail", 1)[1]
    assert reproducer.count("--kill-on-bad-exit=1") == len(("allocation-epilog", "prepare", "train"))


def test_awscmh_ultra_fp8_determinism_uses_standard_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "awscmh")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, flavors=["detr"])
    slurm_job._configure_run(args)

    script, reaper, _log = slurm_job._build_job(args)
    slurm_job.sbatch_run(
        script,
        slurm_job.RESULTS_DIR / "scheduler.log",
        extra_sbatch_args=[reaper],
        dry_run=True,
    )
    reproducer = (tmp_path / "ultra3-64n" / "run.sbatch").read_text()

    assert "#SBATCH --time=03:00:00" in reproducer
    assert "#SBATCH --partition=batch" in reproducer
    assert "batch_long" not in reproducer


def test_awscmh_ultra_fp8_perf_keeps_default_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "awscmh")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path)
    slurm_job._configure_run(args)

    script, _reaper, _log = slurm_job._build_job(args)

    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --partition=batch" in script
    assert "batch_long" not in script


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


def test_clean_run_removes_the_directories_the_runs_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(
        tmp_path, config="nano3", nodes=8, name=None, flavors=["detr"],
        flavor=["detr"], run=["ref:nvrx", "cmp:native"], clean_run=True,
    )
    # Every run writes inside the job's own directory, so removing that
    # directory takes each run's checkpoints, logs and digests with it.
    job = tmp_path / "nano3_8n-detr"
    stale = []
    for name in ("nvrx.ref", "native.cmp", "."):
        (job / name / "logs").mkdir(parents=True, exist_ok=True)
        stale.append(job / name / "logs" / "old_digest.xz")
        stale[-1].write_text("old")

    slurm_job._configure_run(args)

    assert [path for path in stale if path.exists()] == []
    for spec in slurm_job._run_specs(args):
        with slurm_job._in_run(spec, args):
            assert slurm_job.RUN_ROOT.parent == job


def test_clean_run_refuses_a_symlinked_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "logs").mkdir(parents=True)
    (tmp_path / "nano3_8n-detr").symlink_to(elsewhere)
    args = _args(
        tmp_path, config="nano3", nodes=8, name=None, flavors=["detr"],
        flavor=["detr"], run=["ref:nvrx", "cmp:native"], clean_run=True,
    )

    with pytest.raises(SystemExit, match="refuses to remove a symlink"):
        slurm_job._configure_run(args)

    assert elsewhere.is_dir()


def test_clean_run_intent_is_reproducible_from_sbatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "cwdfw")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, clean_run=True, job_array_segment=0)
    slurm_job._configure_run(args)
    script, _reaper, run_log = slurm_job._build_job(args)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
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
    monkeypatch.setenv("CLUSTER", "cwdfw")
    monkeypatch.setenv("PLATFORM", "gb200")
    monkeypatch.setenv("NVRX_COMMIT", "a" * 40)
    monkeypatch.setenv("CI_PIPELINE_ID", "98765")
    monkeypatch.setenv("CI_JOB_TOKEN", "must-not-be-written")
    args = _args(tmp_path, nvrx_commit="a" * 40, job_array_segment=0)
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
    assert "#SBATCH --signal=TERM@300" in reproducer
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
    assert 'export TRITON_CACHE_DIR="${SLURMJOB_OPT_WORK_ROOT}/caches/triton"' in reproducer
    assert 'export TRITON_CACHE_PATH="${SLURMJOB_OPT_WORK_ROOT}/caches/triton"' in reproducer
    assert 'export TORCHINDUCTOR_CACHE_DIR="${SLURMJOB_OPT_WORK_ROOT}/caches/torchinductor"' in reproducer
    assert 'export TORCH_EXTENSIONS_DIR="${SLURMJOB_OPT_WORK_ROOT}/caches/torch-extensions"' in reproducer
    assert 'export NATIVE_CKPT_SERVER_LOG="${SLURMJOB_OPT_RUN_DIR}/logs/checkpoint_server.%h.%p.log"' in reproducer
    assert 'export SLURMJOB_OPT_RUN_DIR="${SLURMJOB_OPT_WORK_DIR}/${SLURMJOB_OPT_RUN_NAME}"' in reproducer
    # Keyed by revision rather than by run, so it is exported outright instead
    # of derived from a parent.
    assert "export MEGATRON_NATIVE_BUILD_DIR=" in reproducer
    assert "/build/nvrx_checkpointing-" in reproducer
    assert 'export SLURMJOB_OPT_PYTHON_INSTALL_DIR="${SLURMJOB_OPT_WORK_ROOT}/deps/python"' in reproducer
    assert "must-not-be-written" not in reproducer
    assert reproducer.index("export NVRX_COMMIT=") < reproducer.index("srun ")
    assert "${SLURMJOB_OPT_WORK_DIR}:${SLURMJOB_OPT_WORK_DIR}" in reproducer
    assert "VCPKG_ROOT,MEGATRON_NATIVE_BUILD_DIR,NVRX_REPO_ROOT" in reproducer
    worker_step = next(line for line in reproducer.splitlines() if " --ntasks=256 " in line)
    assert "--container-writable" not in worker_step
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
    assert "scripts/megatron_root.sh" in prep_step
    assert prep_step.index("scripts/megatron_root.sh") < prep_step.index("wheel_install.py")
    assert all(name in prep_step for name in ("MEGATRON_ROOT", "MEGATRON_COMMIT", "MEGATRON_REPO_URL"))
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


def test_job_array_passes_the_worker_script_directly_to_ft_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "ocihsg")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, config="ultra3bf16", job_array_segment=16)
    slurm_job._configure_run(args)

    script, _reaper, _log = slurm_job._build_job(args)

    launcher_line = next(line for line in script.splitlines() if "ft_launcher " in line)
    worker_script = str(Path(slurm_job.__file__).resolve())
    assert worker_script in launcher_line
    assert f"ft_launcher python3 -u {worker_script}" not in launcher_line
    assert "--rdzv-conf join_timeout=12000,store_connect_wait_seconds=600" in launcher_line


def test_jobs_wider_than_one_nvlink_domain_launch_as_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()

    assert slurm_job._job_array_segment(_args(tmp_path, nodes=64)) == 16  # noqa: PLR2004
    assert slurm_job._job_array_segment(_args(tmp_path, nodes=16)) == 0
    assert slurm_job._job_array_segment(_args(tmp_path, nodes=64, job_array_segment=0)) == 0
    assert slurm_job._job_array_segment(_args(tmp_path, nodes=48, job_array_segment=8)) == 8  # noqa: PLR2004

    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "ocihsg")
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, config="ultra3bf16")
    slurm_job._configure_run(args)

    script, _reaper, _log = slurm_job._build_job(args)

    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --array=0-9%5" in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --segment=16" in script
    assert "--nnodes 64:80" in script
    assert "--ft-segment 16" in script
    assert "--ft-cycle-info-dir" in script
    assert "--kill-on-bad-exit=0" in script
    assert "--kill-on-bad-exit=1" in script
    clean = script.index('if [[ "${SLURMJOB_OPT_CLEAN_RUN:-0}"')
    control_dir = script.index('mkdir -p "${NVRX_ARRAY_DIR}/control"', clean)
    assert clean < control_dir
    assert 'while [[ ! -f "${CLEAN_DONE}" ]]' in script
    assert 'scancel "${SLURM_ARRAY_JOB_ID}_[1-${SLURM_ARRAY_TASK_MAX}]"' in script
    assert 'scancel "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true' not in script
    assert "no_restart" not in script
    assert "rc == 93" not in script

    classic, _reaper, _log = slurm_job._build_job(_args(tmp_path, config="ultra3bf16", job_array_segment=0))
    assert "#SBATCH --nodes=64" in classic
    assert "#SBATCH --array=" not in classic


@pytest.mark.parametrize("cluster", ["cwdfw", "cwpdx"])
def test_job_array_steps_request_only_segment_gpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cluster: str,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", cluster)
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(tmp_path, config="ultra3bf16", job_array_segment=16)
    slurm_job._configure_run(args)

    script, _reaper, _log = slurm_job._build_job(args)

    assert "--gpus=64" in script
    assert "--gpus=256" not in script


def test_megatron_root_reuses_prepared_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    root = tmp_path / "megatron-lm"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setenv("MEGATRON_ROOT", str(root))
    monkeypatch.setattr(slurm_job.subprocess, "check_output", lambda *_a, **_k: pytest.fail("checkout rerun"))

    assert slurm_job.megatron_root() == root.resolve()


def test_manual_training_environment_override_is_reproduced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "cwdfw")
    monkeypatch.setenv("PLATFORM", "gb200")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    args = _args(tmp_path, nodes=8, config="nano3", flavors=[])
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
    monkeypatch.setenv("CLUSTER", "cwdfw")
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
    monkeypatch.setenv("CLUSTER", "cwdfw")
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


@pytest.mark.parametrize("platform", ["gb200", "gb300"])
def test_lyris_templates_select_the_platform_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "lyris")
    monkeypatch.setenv("PLATFORM", platform)

    values = slurm_job._slurm_values(
        1, 4, 30, "partition-test", str(tmp_path / "image.sqsh"), "", 4, 4,
    )
    values.update(
        image_base=str(tmp_path / "base.sqsh"),
        image_out=str(tmp_path / "output.sqsh"),
    )

    for name in ("srun.tmpl", "enroot_import.tmpl", "image_prep.tmpl"):
        command = slurm_job.build_cmd_from_template(name, values, [])
        assert f"--partition={platform}" in command


@pytest.mark.parametrize(
    "case",
    [
        ("cwdfw", 1, None, None),
        ("lyris", 2, None, None),
        ("ocihsg", 8, None, None),
        ("ocihsg", 16, None, "16"),
        ("cwdfw", 64, None, "16"),
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


def test_shipped_clusters_have_complete_template_sets() -> None:
    expected = {"srun.tmpl", "enroot_import.tmpl", "image_prep.tmpl"}
    templates = {path.name for path in (REPO_ROOT / "ci/templates").glob("*.tmpl")}
    clusters = {name.split(".", 1)[0] for name in templates}

    assert clusters == {"awscmh", "cwdfw", "cwpdx", "lyris", "ocihsg"}
    assert all(
        {name.removeprefix(f"{cluster}.") for name in templates if name.startswith(f"{cluster}.")} == expected
        for cluster in clusters
    )


def test_template_override_must_be_cluster_prefixed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("CI_DIR", str(tmp_path))
    monkeypatch.setenv("CLUSTER", "missing")
    (tmp_path / "srun.tmpl").write_text("unprefixed")
    (tmp_path / "missing.srun.tmpl").write_text("cluster-prefixed")

    assert slurm_job._load_template("srun.tmpl") == "cluster-prefixed"


def test_missing_cluster_template_has_no_default_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("CI_DIR", str(tmp_path))
    monkeypatch.setenv("CLUSTER", "missing")
    (tmp_path / "srun.tmpl").write_text("unprefixed")

    with pytest.raises(FileNotFoundError, match="not found for cluster 'missing'"):
        slurm_job._load_template("srun.tmpl")


@pytest.mark.parametrize(
    ("cluster", "partition"),
    [("awscmh", "batch"), ("cwdfw", "interactive")],
)
def test_enroot_import_requests_runner_gpus(
    monkeypatch: pytest.MonkeyPatch,
    cluster: str,
    partition: str,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("CLUSTER", cluster)

    cmd = slurm_job.build_cmd_from_template(
        "enroot_import.tmpl",
        {"account": "account", "time_minutes": 20, "gpus_per_node": 4},
        ["enroot", "import"],
    )

    assert "--gpus-per-node=4" in cmd
    assert f"--partition={partition}" in cmd


def test_image_preparation_stack_versions_match() -> None:
    slurm_job = _load_slurm_job()
    prepare_image = (SCRIPTS_DIR / "prepare_image.py").read_text()
    prepare = (REPO_ROOT / "scripts" / "prepare.sh").read_text()
    enroot = (REPO_ROOT / "scripts" / "enroot.sh").read_text()
    dockerfile = (REPO_ROOT / "Dockerfile.ci").read_text()

    assert slurm_job.DOCKER_IMAGE_BASE == "nvcr.io/nvidia/pytorch:26.09-py3"
    assert 'base_image = image.parent / "pytorch2609.sqsh"' in prepare_image
    assert 'DOCKER_IMAGES["pytorch"]="nvcr.io/nvidia/pytorch:26.09-py3"' in prepare
    assert 'DOCKER_IMAGE_BASE="nvcr.io/nvidia/pytorch:26.09-py3"' in enroot
    assert "pytorch2609.sqsh" in enroot
    assert "FROM nvcr.io/nvidia/pytorch:26.09-py3" in dockerfile
    assert "ENV CC=clang-23 CXX=clang++-23" in dockerfile
    assert "CLANG_VERSION=23" in prepare
    assert "cuda-toolkit-13-4" in prepare
    assert "/usr/local/cuda-13.4" in prepare


def test_image_build_selects_account_compatible_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_image = _load_prepare_image()
    probed: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        if cmd[0] == "sinfo":
            return SimpleNamespace(returncode=0, stdout="batch*\ninteractive\ngb200\n", stderr="")
        probed.append(cmd)
        return SimpleNamespace(returncode=0 if "--partition=interactive" in cmd else 1, stdout="", stderr="invalid")

    monkeypatch.setattr(prepare_image.subprocess, "run", run)
    cmd = [
        "srun",
        "--account=account",
        "--partition=batch",
        "--time=00:120:00",
        "--mem=0",
        "--gpus-per-node=4",
        "enroot",
        "import",
    ]

    partition = prepare_image._valid_partition(cmd)

    assert partition == "interactive"
    assert prepare_image._set_partition(cmd, partition)[1] == "--partition=interactive"
    assert "--time=00:120:00" in probed[0]
    assert "--mem=0" in probed[0]


def test_image_build_validates_import_and_prepare_partitions_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prepare_image = _load_prepare_image()
    (tmp_path / "pytorch2609.sqsh").touch()
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path))
    built: list[tuple[str, list[str], dict[str, object]]] = []
    build_values: list[dict[str, object]] = []
    validated: list[str] = []
    runs: list[tuple[list[str], dict[str, object]]] = []

    def build(name: str, values: dict[str, object], payload: list[str], **kwargs: object) -> list[str]:
        build_values.append(values)
        built.append((name, payload, kwargs))
        partition = "interactive" if name == "image_prep.tmpl" else "backfill"
        return ["srun", f"--partition={partition}", *payload]

    def validate(cmd: list[str]) -> str:
        partition = next(arg.removeprefix("--partition=") for arg in cmd if arg.startswith("--partition="))
        validated.append(partition)
        return partition

    monkeypatch.setattr(prepare_image, "build_cmd_from_template", build)
    monkeypatch.setattr(prepare_image, "_valid_partition", validate)
    monkeypatch.setattr(prepare_image, "_remove_native_smoke_build", lambda _path: None)
    monkeypatch.setattr(prepare_image, "_publish_to_shared", lambda _local, _image: True)
    monkeypatch.setattr(prepare_image.subprocess, "run", lambda cmd, **kwargs: runs.append((cmd, kwargs)))

    prepare_image._ensure_enroot_image(tmp_path / "native.sqsh", 8)

    assert built[-1][0] == "image_prep.tmpl"
    assert build_values[-1]["time_minutes"] == 240  # noqa: PLR2004
    assert built[-1][1][-1].endswith("scripts/prepare.sh ubuntu2404 megatron test")
    prepare = (REPO_ROOT / "scripts" / "prepare.sh").read_text()
    assert '"nvidia-cudnn-frontend[cutedsl]==1.28.0"' in prepare
    assert '"${MEGATRON_ROOT}[training,dev,otel,te,ssm]"' in prepare
    assert 'export TMS_CUDA_MAJOR="${BASH_REMATCH[1]}"' in prepare
    assert "nvidia-nvshmem-cu13==3.7.2" in prepare
    assert "DEEPEP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95" in prepare
    assert 'patch -p1 < "${SCRIPT_DIR}/deepep-nvshmem.patch"' in prepare
    assert (REPO_ROOT / "scripts" / "deepep-nvshmem.patch").is_file()
    assert built[-1][2] == {}
    assert validated == ["backfill", "interactive"]
    assert "--partition=interactive" in runs[-1][0]
    assert runs[-1][1]["env"]["NVTE_WITH_NCCL_EP"] == "0"


def test_otel_cmake_replaces_the_incompatible_legacy_vcpkg_port() -> None:
    module = (REPO_ROOT / "cmake/modules/OpenTelemetry.cmake").read_text()

    legacy_remove = 'remove "opentelemetry-cpp:${VCPKG_TARGET_TRIPLET}" --recurse'
    cpp23_install = 'vcpkg_install("opentelemetry-cpp23[otlp-file]")'
    assert legacy_remove in module
    assert module.index(legacy_remove) < module.index(cpp23_install)


def test_slurm_job_consumers_import() -> None:
    subprocess.run(
        [sys.executable, "-c", "import ckpt_stress, prepare_image"],
        cwd=SCRIPTS_DIR,
        check=True,
    )


def test_auto_workloads_only_allow_perf_gate_warnings() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())

    before_script = config[".ci_base"]["before_script"]
    source_index = before_script.index("source ~/data/ci/config 2>/dev/null || source ~/data/ci 2>/dev/null || true")
    assert before_script[source_index - 1 : source_index + 2] == ["set -a", before_script[source_index], "set +a"]
    assert 'export PLATFORM="${PLATFORM_OVERRIDE:-${PLATFORM:-${CLUSTER}}}"' in before_script
    assert config[".slurm_job"]["allow_failure"] == {"exit_codes": [3]}
    after_script = "\n".join(config[".slurm_job"]["after_script"])
    assert "du -ch " in after_script
    assert "| tail -n 1 || true" in after_script
    assert "__failure_summary__ ci/ci-runs" in after_script
    assert "__recover_otel__ ci/ci-runs || true" in after_script
    assert after_script.index("__recover_otel__") < after_script.index("gzip -f")
    assert config[".slurm_job"]["artifacts"]["when"] == "always"
    assert config[".slurm_job"]["variables"] == {
        "SLURMJOB_OPT_WORK_DIR": "${CI_PROJECT_DIR}/ci/ci-runs",
        "TRITON_CACHE_DIR": "/tmp/nvrx-${CI_JOB_ID}/triton-cache",
        "TRITON_CACHE_PATH": "/tmp/nvrx-${CI_JOB_ID}/triton-cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/nvrx-${CI_JOB_ID}/torchinductor-cache",
        "TORCH_EXTENSIONS_DIR": "/tmp/nvrx-${CI_JOB_ID}/torch-extensions",
    }
    manual_optional = config[".manual_optional"]
    assert "allow_failure" not in manual_optional
    assert manual_optional["rules"][0] == {
        "if": (
            '$PIPELINE_SUITE == "nvrx" || $PIPELINE_SUITE == "large-scale" '
            '|| $PIPELINE_SUITE == "main-branch-runs"'
        ),
        "when": "never",
    }
    assert manual_optional["rules"][1] == {
        "if": '$CI_COMMIT_BRANCH == "main"',
        "when": "on_success",
    }
    assert manual_optional["rules"][2] == {
        "if": '$RUN_PIPELINE == "true" || $CI_PIPELINE_SOURCE == "merge_request_event"',
        "when": "on_success",
    }
    # A rule's allow_failure replaces the job's own, so leaving it unset keeps
    # the job's exit_codes: a played job that fails reports a failure, and only
    # the perf-drift exit is tolerated.
    assert manual_optional["rules"][3] == {"when": "manual"}
    assert "allow_failure" not in manual_optional["rules"][3]
    for suite in ("nvrx", "large_scale", "main_branch_runs"):
        policy = config[f".{suite}_pipeline"]
        assert "allow_failure" not in policy
        assert policy["rules"][-1] == {"when": "never"}
    assert ".manual_nvrx" not in config


def test_generated_pipeline_suites_and_node_tags() -> None:
    matrix = yaml.safe_load((REPO_ROOT / "ci/jobs.json").read_text())
    assert all("jobs" in platform for platform in matrix["platforms"])
    platform_jobs = {platform["name"]: platform["jobs"] for platform in matrix["platforms"]}
    assert platform_jobs["gb300"] == []

    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    workloads = {
        name: job for name, job in config.items()
        if isinstance(job, dict) and isinstance(job.get("variables"), dict) and "JOB_ARGS" in job["variables"]
    }

    for name, job in workloads.items():
        args = shlex.split(job["variables"]["JOB_ARGS"])
        nodes = int(args[args.index("--nodes") + 1]) if "--nodes" in args else 1
        platform = next(tag for tag in ("devbox", "gb200", "gb300", "h100") if f":{tag}:" in f":{name}:")
        expected_tags = [platform, f"{nodes}n"]
        assert job["tags"] == expected_tags

    assert all("pico_2x" not in name for name in workloads)
    assert all(
        ":gb300:" not in name or ":nano4_32x4" in name or ":nano4_8x4" in name
        for name in workloads
    )
    assert ".run_ckpt_stress" not in config

    nano4 = {name: job for name, job in workloads.items() if ":nano4_32x4" in name}
    assert config[".runner_gb300"]["tags"] == ["gb300"]
    assert config[".runner_gb300"]["variables"] == {
        "PLATFORM_OVERRIDE": "gb300",
        "GPUS_PER_NODE": "4",
        "WHEEL_ARCH": "aarch64",
        "NVTE_FUSED_ATTN": "0",
    }
    assert set(nano4) == {
        "[DETR]:gb300:native:nano4_32x4",
        "[PERF]:gb300:native:nano4_32x4",
        "[DETR]:gb300:nvrx:nano4_32x4",
        "[PERF]:gb300:nvrx:nano4_32x4",
    }
    for name, job in nano4.items():
        expected_suite = ".nvrx_pipeline" if ":nvrx:" in name else ".large_scale_pipeline"
        assert job["extends"] == [".slurm_job", ".runner_gb300", expected_suite]
        args = shlex.split(job["variables"]["JOB_ARGS"])
        assert args[:4] == ["--nodes", "32", "--config", "nano4"]
        assert "--megatron-commit" not in args
        assert "--flavor" in args
        assert "async" in args[args.index("--flavor") + 1]
        segment_index = args.index("--job-array-segment")
        assert args[segment_index + 1] == "8"
        run_indexes = [index for index, arg in enumerate(args) if arg.startswith("--run=")]
        assert not run_indexes or segment_index < min(run_indexes)
        if "[PERF]" in name:
            interval_index = args.index("--checkpoint-interval")
            assert args[interval_index + 1] == "125"
        else:
            assert "--lr-decay-style" not in args
            assert "--lr-warmup-samples" not in args
        assert "--cuda-graph-impl" not in args
        assert "--cuda-graph-modules" not in args
        assert "--muon-fp32-matmul-prec" not in args

    eight_node_nano4 = workloads["[DETR]:gb300:nvrx:nano4_8x4"]
    assert eight_node_nano4["extends"] == [".slurm_job", ".runner_gb300", ".nvrx_pipeline"]
    args = shlex.split(eight_node_nano4["variables"]["JOB_ARGS"])
    assert args[:4] == ["--nodes", "8", "--config", "nano4"]
    assert args[args.index("--flavor") + 1] == "detr"
    assert "--cuda-graph-impl" not in args
    assert "--cuda-graph-modules" not in args
    assert "--muon-fp32-matmul-prec" not in args
    assert args[args.index("--run=ref:nvrx") : args.index("--run=cmp:nvrx") + 1] == [
        "--run=ref:nvrx",
        "--run=cmp:nvrx",
    ]
    assert "--lr-decay-style" not in args
    assert "--lr-warmup-samples" not in args

    for name in (
        "[DETR]:gb300:native:nano4_8x4",
        "[PERF]:gb300:native:nano4_8x4",
    ):
        job = workloads[name]
        assert job["extends"] == [".slurm_job", ".runner_gb300", ".manual_optional"]
        args = shlex.split(job["variables"]["JOB_ARGS"])
        assert args[:4] == ["--nodes", "8", "--config", "nano4"]
        assert "--cuda-graph-impl" not in args
        assert "--cuda-graph-modules" not in args
        assert "--muon-fp32-matmul-prec" not in args
        if "[PERF]" in name:
            interval_index = args.index("--checkpoint-interval")
            assert args[interval_index + 1] == "25"

    nvrx = {name: job for name, job in workloads.items() if ":nvrx:" in name and ":gb200:" in name}
    assert any(":super3fp4_16x4" in name for name in nvrx)
    assert any(":ultra3bf16_64x4" in name for name in nvrx)
    assert any(":ultra3fp8_64x4" in name for name in nvrx)
    assert all(job["extends"][-1] == ".nvrx_pipeline" for job in nvrx.values())

    main_only = {
        name: job for name, job in workloads.items()
        if (
            ":h100:" in f":{name}:"
            or ":native:nano3_8x4" in name
            or "detr+vg" in shlex.split(job["variables"]["JOB_ARGS"])
        )
    }
    assert main_only
    assert all(job["extends"][-1] == ".main_branch_runs_pipeline" for job in main_only.values())

    gpt_moe = workloads["[PERF]:gb200:native:gpt_moe_1x4"]
    assert gpt_moe["extends"] == [".slurm_job", ".runner_gb200", ".main_branch_runs_pipeline"]
    assert gpt_moe["tags"] == ["gb200", "1n"]
    assert shlex.split(gpt_moe["variables"]["JOB_ARGS"]) == [
        "--nodes", "1", "--config", "gpt_moe",
        "--flavor", "perf+native",
        "--checkpoint-interval", "20",
    ]

    large = {
        name: job for name, job in workloads.items()
        if (
            ":native:super3bf16_16x4" in name
            or ":native:super3fp4_16x4" in name
            or ":native:ultra3bf16_64x4" in name
            or ":native:ultra3fp8_64x4" in name
        )
    }
    assert set(large) == {
        "[DETR]:gb200:native:super3bf16_16x4",
        "[PERF]:gb200:native:super3bf16_16x4",
        "[PERF]:gb200:native:super3fp4_16x4",
        "[PERF]:gb200:native:ultra3bf16_64x4",
        "[DETR]:gb200:native:ultra3fp8_64x4",
        "[PERF]:gb200:native:ultra3fp8_64x4",
    }
    assert all(job["extends"][-1] == ".large_scale_pipeline" for job in large.values())
    ultra64_det = shlex.split(large["[DETR]:gb200:native:ultra3fp8_64x4"]["variables"]["JOB_ARGS"])
    assert ultra64_det == [
        "--nodes", "64", "--config", "ultra3fp8",
        "--flavor", "detr", "--run=ref:nvrx", "--run=cmp:native",
    ]
    # Staging is named per entry and per flavor: the sixteen-node gate stages
    # through shared memory while the run beside it keeps the default. The
    # 64-node determinism gate stays synchronous to avoid async staging's extra
    # host-memory pressure for this large optimizer state.
    super16_det = shlex.split(large["[DETR]:gb200:native:super3bf16_16x4"]["variables"]["JOB_ARGS"])
    super16_perf = shlex.split(large["[PERF]:gb200:native:super3bf16_16x4"]["variables"]["JOB_ARGS"])
    assert "detr+shm" in super16_det
    assert "perf+async+native" in super16_perf
    ultra64_perf = shlex.split(large["[PERF]:gb200:native:ultra3fp8_64x4"]["variables"]["JOB_ARGS"])
    assert "--use-precision-aware-optimizer" not in ultra64_perf

    pico_perf = config["[PERF]:gb200:native:pico_1x4"]
    assert pico_perf["needs"] == [{"job": "wheel:aarch64", "artifacts": True}]
    assert all("wheel-publish" not in need["job"] for need in pico_perf["needs"])

    pico_detr_args = shlex.split(config["[DETR]:gb200:native:pico_1x4"]["variables"]["JOB_ARGS"])
    assert "--checkpoint-interval" not in pico_detr_args
    nano_valgrind_args = shlex.split(config["[DETR]:gb200:native:nano3_8x4:vg"]["variables"]["JOB_ARGS"])
    assert "--checkpoint-interval" not in nano_valgrind_args
    assert "--checkpoint-interval" not in shlex.split(
        config["gb200:valgrind:pico_1x4"]["variables"]["JOB_ARGS"],
    )
    assert "--checkpoint-interval" not in shlex.split(
        config["[DETR]:gb200:native:nano3_8x4"]["variables"]["JOB_ARGS"],
    )


def test_generated_jobs_do_not_override_config_or_flavor_options() -> None:
    matrix = yaml.safe_load((REPO_ROOT / "ci/jobs.json").read_text())
    assert all(
        "flavor_args" not in entry
        for platform in matrix["platforms"]
        for entry in platform.get("multinode", [])
    )

    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    forbidden = (
        "--cuda-graph-impl",
        "--cuda-graph-modules",
        "--muon-fp32-matmul-prec",
        "--lr-decay-style",
        "--lr-warmup-samples",
        "--drop-config-option",
        "--no-moe-permute-fusion",
    )
    for name, job in config.items():
        if not isinstance(job, dict) or "JOB_ARGS" not in job.get("variables", {}):
            continue
        argv = shlex.split(job["variables"]["JOB_ARGS"])
        assert all(option not in argv for option in forbidden), name


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
    monkeypatch.setattr(slurm_job, "_verify_allocation", lambda job_id, rc, **_kwargs: verified.append((job_id, rc)))

    assert slurm_job.sbatch_run("#!/bin/bash\n\necho test\n", run / "scheduler.log") == 0
    assert ["sbatch", "--parsable", "--wait", "--export=ALL", str(run / "run.sbatch")] in commands
    assert verified == [("123", 0)]
    assert (run / "job_id").read_text() == "123\n"
    assert (run / "gitlab_job_id").read_text() == "456\n"


def test_foreground_submission_reports_scheduler_status_until_job_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slurm_job = _load_slurm_job()
    waits = []

    class Process:
        def wait(self, timeout: int | None = None) -> int:
            waits.append(timeout)
            if len(waits) <= 2:
                raise subprocess.TimeoutExpired("sbatch", timeout)
            return 0

    reports = []
    keep_polling = iter((True, False))

    def report(job_id: str) -> bool:
        reports.append(job_id)
        return next(keep_polling)

    monkeypatch.setattr(slurm_job, "_report_queue_status", report)

    assert slurm_job._wait_for_submission(Process(), "123") == 0
    assert waits == [5 * 60, 5 * 60, None]
    assert reports == ["123", "123"]


def test_only_nccl_ib_proxy_segfault_gets_one_clean_allocation_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    run = tmp_path / "run"
    log = run / "logs" / "run.log"
    log.parent.mkdir(parents=True)
    (run / "partial-checkpoint").mkdir()
    monkeypatch.setattr(slurm_job, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run)
    monkeypatch.setattr(slurm_job, "RUN_NAME", "run")
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", log.parent)
    calls = []

    def submit(*_args: object, **kwargs: object) -> int:
        calls.append(kwargs)
        if len(calls) == 1:
            log.write_text("!!!!!!! Segfault encountered !!!!!!! ncclIbMultiSend sendProxyProgress")
            raise RuntimeError("allocation failed")
        assert not (run / "partial-checkpoint").exists()
        return 0

    monkeypatch.setattr(slurm_job, "sbatch_run", submit)

    assert slurm_job._sbatch_run_with_retry(
        "script", log, extra_sbatch_args=["reaper"], background=False, dry_run=False,
    ) == 0
    assert calls == [
        {"extra_sbatch_args": ["reaper"], "background": False, "dry_run": False, "job_array": False},
        {"extra_sbatch_args": ["reaper"], "job_array": False},
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "torch.OutOfMemoryError: CUDA out of memory",
        "Unexpected result nan",
        "!!!!!!! Segfault encountered !!!!!!! unrelated crash",
    ],
)
def test_oom_nan_and_unrelated_crashes_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "run.log"
    log.write_text(failure)
    calls = []
    monkeypatch.setattr(
        slurm_job,
        "sbatch_run",
        lambda *_args, **_kwargs: calls.append(True) or (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        slurm_job._sbatch_run_with_retry(
            "script", log, extra_sbatch_args=[], background=False, dry_run=False,
        )

    assert calls == [True]


def test_scheduler_status_and_accounting_are_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    outputs = iter(
        (
            SimpleNamespace(returncode=0, stdout="123|RUNNING|05:00|01:00:00|now|n[1-4]\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="accounting database unavailable"),
        ),
    )
    monkeypatch.setattr(slurm_job.subprocess, "run", lambda *_args, **_kwargs: next(outputs))

    with caplog.at_level("INFO"):
        assert not slurm_job._report_queue_status("123")
        slurm_job._report_allocation_accounting("123")

    assert "123|RUNNING|05:00" in caplog.text
    assert "accounting database unavailable" in caplog.text


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


def test_array_submission_drops_stale_peer_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    log = logs / "train.log"
    stale = [logs / "train.task1.log", logs / "train.task9.log", logs / "train.combined.log"]
    for path in stale:
        path.write_text("old run")
    monkeypatch.setattr(slurm_job, "RUN_ROOT", run)
    monkeypatch.setattr(slurm_job, "RESULTS_DIR", logs)
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="123\n", stderr=""),
    )

    assert slurm_job.sbatch_run("#!/bin/bash\n\necho test\n", log, background=True, job_array=True) == 0

    assert not any(path.exists() for path in stale)


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

    with pytest.raises(
        RuntimeError,
        match=r"Slurm job 123 finished unsuccessfully: state=CANCELLED by 456 exit=0:0 derived_exit=0:15",
    ):
        slurm_job._verify_allocation("123", 0)


def test_sacct_accepts_only_the_perf_soft_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="123|FAILED|3:0|0:0\n", stderr=""),
    )

    slurm_job._verify_allocation("123", slurm_job.SOFT_FAIL_RC)


def test_sacct_job_array_reads_task_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "123_0|COMPLETED|0:0|0:0\n"
                "123_1|COMPLETED|0:0|0:0\n"
                "123_2|CANCELLED by 456|0:0|0:0\n"
                "123_3|FAILED|1:0|0:0\n"  # drained by task 0's RDZV_CLOSED
                "123_4|NODE_FAIL|0:0|0:0\n"  # replaced infrastructure failure
            ),
            stderr="",
        ),
    )

    assert slurm_job._verify_allocation("123", 0, job_array=True) == 0


def test_sacct_job_array_preserves_task_zero_soft_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="123_0|FAILED|3:0|0:0\n123_1|CANCELLED by 456|0:0|0:0\n",
            stderr="",
        ),
    )

    assert slurm_job._verify_allocation("123", 0, job_array=True) == slurm_job.SOFT_FAIL_RC


def test_sacct_job_array_rejects_a_failed_task_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="123_0|FAILED|1:0|0:0\n123_1|COMPLETED|0:0|0:0\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=r"task 0 was not successful: state=FAILED exit=1:0"):
        slurm_job._verify_allocation("123", 0, job_array=True)


def test_sacct_job_array_without_indexed_rows_fails_on_any_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare per-task ids cannot name task 0: a clean peer must not carry the verdict."""
    slurm_job = _load_slurm_job()
    monkeypatch.setattr(
        slurm_job.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="124|FAILED|1:0|0:0\n125|COMPLETED|0:0|0:0\n126|CANCELLED|0:0|0:0\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=r"no identifiable task 0"):
        slurm_job._verify_allocation("123", 0, job_array=True)


def test_workflow_runs_merge_request_events_without_duplicate_branch_pipelines() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    rules = config["workflow"]["rules"]

    assert config["variables"]["PIPELINE_SUITE"]["options"] == [
        "default",
        "nvrx",
        "large-scale",
        "main-branch-runs",
    ]
    assert {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'} in rules
    assert not any("CI_OPEN_MERGE_REQUESTS" in rule.get("if", "") for rule in rules)
    assert {"if": '$CI_PIPELINE_SOURCE == "push" && $RUN_PIPELINE == "true"'} in rules
    assert any("pipeline" in rule.get("if", "") and rule.get("when") != "never" for rule in rules)
    assert any(rule.get("when") == "never" and "PIPELINE_SUITE !=" in rule.get("if", "") for rule in rules)
    assert rules[-1] == {"when": "never"}


def test_default_pipeline_has_manual_suite_launchers() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    launcher = config[".pipeline_launcher"]

    assert launcher["stage"] == "launch"
    assert launcher["rules"][0] == {
        "if": '$PIPELINE_SUITE != "default"',
        "when": "never",
    }
    assert launcher["rules"][1] == {
        "if": '$CI_PIPELINE_SOURCE == "pipeline" || $CI_PIPELINE_SOURCE == "parent_pipeline"',
        "when": "never",
    }
    assert launcher["rules"][2] == {
        "if": '$RUN_ALL_SUITES == "true"',
        "when": "on_success",
    }
    assert launcher["rules"][-1] == {"when": "manual", "allow_failure": True}
    assert launcher["image"] == "curlimages/curl:8.15.0"
    assert launcher["tags"] == ["perflab", "type/docker", "os/linux"]
    assert launcher["variables"]["GIT_STRATEGY"] == "none"
    assert "trigger" not in launcher
    script = launcher["script"][0]
    assert "CI_JOB_TOKEN" in script
    assert "variables[PIPELINE_SUITE]=${LAUNCH_PIPELINE_SUITE}" in script
    assert "variables[UPSTREAM_PIPELINE_SHA]" in script
    assert "variables[UPSTREAM_PIPELINE_ID]" in script
    assert "variables[UPSTREAM_WHEEL_X86_64_JOB_ID]" in script
    assert "variables[UPSTREAM_WHEEL_AARCH64_JOB_ID]" in script
    assert "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME" in script
    assert "CI_MERGE_REQUEST_SOURCE_BRANCH_SHA" in script
    assert "head -n1" not in script
    assert "sed -n '1p'" in script
    needs = {need["job"]: need["artifacts"] for need in launcher["needs"]}
    assert needs == {
        "wheel:x86_64": True,
        "wheel:aarch64": True,
        "wheel-pytest:x86_64": False,
        "wheel-pytest:aarch64": False,
        "ctest:x86_64": False,
        "ctest:aarch64": False,
    }
    assert config["launch:nvrx-pipeline"]["variables"]["LAUNCH_PIPELINE_SUITE"] == "nvrx"
    assert config["launch:large-scale-pipeline"]["variables"]["LAUNCH_PIPELINE_SUITE"] == "large-scale"
    main_runs = config["launch:main-branch-runs"]
    assert main_runs["variables"]["LAUNCH_PIPELINE_SUITE"] == "main-branch-runs"
    assert main_runs["rules"] == [
        {"if": '$PIPELINE_SUITE != "default"', "when": "never"},
        {"if": '$CI_COMMIT_BRANCH == "main"', "when": "on_success"},
        {"if": '$RUN_ALL_SUITES == "true"', "when": "on_success"},
        {"when": "manual", "allow_failure": True},
    ]
    assert config[".wheel_publish"]["rules"] == [
        {"if": '$PIPELINE_SUITE != "default"', "when": "never"},
        {"if": '$CI_COMMIT_BRANCH == "main"', "when": "on_success"},
        {"if": '$PUBLISH_WHEELS == "true"', "when": "on_success"},
        {"when": "never"},
    ]


def test_each_pipeline_suite_bundles_generated_baselines() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())

    assert config["stages"][-1] == "baseline-artifacts"
    assert "ci/ci-runs/*/logs/baselines/*" in config[".slurm_job"]["artifacts"]["paths"]
    assert config[".baseline_bundle"]["artifacts"]["paths"] == ["ci/baselines/"]
    bundles = {
        suite: config[f"baseline-artifacts:{suite}"]
        for suite in ("default", "nvrx", "large-scale", "main-branch-runs")
    }
    assert all(bundle["rules"][0]["when"] in {"always", "never"} for bundle in bundles.values())
    assert "[PERF]:gb200:native:super3bf16_16x4" in bundles["large-scale"]["dependencies"]
    assert "[PERF]:gb200:native:super3fp4_16x4" in bundles["large-scale"]["dependencies"]
    assert "[PERF]:gb200:nvrx:super3fp4_16x4" in bundles["nvrx"]["dependencies"]
    assert "[PERF]:gb200:nvrx:ultra3bf16_64x4" in bundles["nvrx"]["dependencies"]
    assert "[DETR]:gb300:nvrx:nano4_32x4" in bundles["nvrx"]["dependencies"]
    assert "[PERF]:gb300:nvrx:nano4_32x4" in bundles["nvrx"]["dependencies"]
    assert "[DETR]:gb300:nvrx:nano4_8x4" in bundles["nvrx"]["dependencies"]
    assert "[DETR]:gb300:native:nano4_8x4" in bundles["default"]["dependencies"]
    assert "[PERF]:gb300:native:nano4_8x4" in bundles["default"]["dependencies"]
    assert "[DETR]:gb300:native:nano4_32x4" in bundles["large-scale"]["dependencies"]
    assert "[PERF]:gb300:native:nano4_32x4" in bundles["large-scale"]["dependencies"]
    assert "[DETR]:gb200:native:nano3_8x4" in bundles["main-branch-runs"]["dependencies"]
    assert "[PERF]:gb200:native:nano3_8x4" in bundles["main-branch-runs"]["dependencies"]
    assert "[PERF]:h100:native:pico_1x8" in bundles["main-branch-runs"]["dependencies"]
    assert "[PERF]:gb200:native:gpt_moe_1x4" in bundles["main-branch-runs"]["dependencies"]


def test_slurm_jobs_upload_otel_and_perfetto_files() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())

    paths = config[".slurm_job"]["artifacts"]["paths"]
    assert "ci/ci-results/otel/*/*.jsonl.gz" in paths
    assert "ci/ci-runs/*/logs/otel/*/*.jsonl.gz" in paths
    assert "ci/ci-runs/*/*/logs/otel/*/*.jsonl.gz" in paths
    assert "ci/ci-results/otel/*/*perfetto.json.gz" in paths
    assert "ci/ci-runs/*/logs/otel/*/*perfetto.json.gz" in paths
    assert "ci/ci-runs/*/*/logs/otel/*/*perfetto.json.gz" in paths
    assert "ci/ci-results/otel/*/process_memory_*.svg" in paths
    assert "ci/ci-runs/*/logs/otel/*/process_memory_*.svg" in paths
    assert "ci/ci-runs/*/*/logs/otel/*/process_memory_*.svg" in paths


@pytest.mark.parametrize("status", ["success", "failed"])
def test_slurm_jobs_keep_registries_and_upload_them_only_when_failed(tmp_path: Path, status: str) -> None:
    job = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())[".slurm_job"]
    registries = [
        "ci/ci-runs/456_run/logs/registry.0.bin",
        "ci/ci-runs/456_run/logs/registry.global.bin",
        "ci/ci-runs/456_run/variant/logs/registry.1.bin",
        "ci/ci-runs/456_run/variant/logs/registry.global.bin.tmp",
    ]
    shards = [
        "ci/ci-runs/456_run/checkpoints/iter_0000010/__0_0.distcp",
        "ci/ci-runs/456_run/variant/phase1-checkpoints/iter_0000005/__1_0.distcp",
    ]
    untouched = [
        "ci/ci-runs/456_run/checkpoints/iter_0000010/.metadata",
        "ci/ci-runs/789_other/checkpoints/iter_0000010/__0_0.distcp",
    ]
    for path in (*registries, *shards, *untouched):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)
    steps = [line for line in job["after_script"] if "registries" in line or "-name '*.distcp' -delete" in line]
    assert len(steps) == 2  # noqa: PLR2004

    subprocess.run(
        ["bash", "-c", "\n".join(steps)],
        cwd=tmp_path,
        env={"CI_JOB_ID": "456", "CI_JOB_STATUS": status, "PATH": "/usr/bin:/bin"},
        check=True,
    )

    assert all((tmp_path / path).read_text() == path for path in (*registries, *untouched))
    assert [(tmp_path / path).exists() for path in shards] == [status != "success"] * len(shards)
    staging = tmp_path / "ci/ci-results/registries"
    staged = {path.relative_to(staging).as_posix(): path.read_text() for path in staging.rglob("*") if path.is_file()}
    expected = {path.removeprefix("ci/ci-runs/"): path for path in registries} if status == "failed" else {}
    assert staged == expected
    assert "ci/ci-results/registries/" in job["artifacts"]["paths"]
    assert not any("registry" in pattern for pattern in job["artifacts"]["paths"])


def test_default_native_jobs_use_wheel_artifacts() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())

    for name in (
        "[DETR]:gb200:native:pico_1x4",
        "[PERF]:gb200:native:pico_1x4",
        "[DETR]:gb300:native:nano4_8x4",
        "[PERF]:gb300:native:nano4_8x4",
    ):
        assert config[name]["needs"] == [{"job": "wheel:aarch64", "artifacts": True}]
    assert "needs" not in config["[PERF]:gb200:native:super3bf16_16x4"]


def test_wheel_test_matches_main_and_ctest_is_isolated() -> None:
    config = yaml.safe_load((REPO_ROOT / "ci/gitlab-ci-native.yml").read_text())
    build_script = "\n".join(config[".wheel_build"]["script"])
    wheel_script = "\n".join(config[".wheel_pytest"]["script"])
    ctest_script = "\n".join(config[".ctest"]["script"])

    assert "nvrx_checkpointing-linux_${WHEEL_ARCH}.txt" in build_script
    assert "dist/*.txt" in config[".wheel_build"]["artifacts"]["paths"]
    assert config["wheel:x86_64"]["variables"]["WHEEL_JOB_ID_VARIABLE"] == "UPSTREAM_WHEEL_X86_64_JOB_ID"
    assert config["wheel:aarch64"]["variables"]["WHEEL_JOB_ID_VARIABLE"] == "UPSTREAM_WHEEL_AARCH64_JOB_ID"
    assert any("--fetch-upstream-wheel" in step for step in config[".slurm_job"]["before_script"])
    assert " -m pytest tests " in wheel_script
    assert "./dev_binaries/ckpttest" in wheel_script
    assert "build-test" not in wheel_script
    assert "ckpt_stress.py" not in wheel_script
    assert config[".wheel_pytest"]["timeout"] == "30m"

    assert "ctest --test-dir build-test --output-on-failure" in ctest_script
    assert '--output-junit "${CI_PROJECT_DIR}/ci/ci-results/ctest.xml"' in ctest_script
    assert "ci/scripts/ckpt_stress.py --gpus-per-node 2" in ctest_script
    assert "PHOENIX_BUILD_BENCHMARK=ON" in ctest_script
    assert "Python3_EXECUTABLE=/opt/venv/bin/python" in ctest_script
    assert "UV_EXECUTABLE=/nonexistent/uv" in ctest_script
    assert "--target ckpttest _distcp_shim ckpt_stress_test dcp_load distcp_roundtrip" in ctest_script
    assert set(config["ctest:x86_64"]["needs"]) == {"wheel-image:x86_64", "wheel:x86_64"}
    assert set(config["ctest:aarch64"]["needs"]) == {"wheel-image:aarch64", "wheel:aarch64"}
    assert 'PHOENIX_WHEEL_VERSION="0.0.1.dev${CI_PIPELINE_IID}+g${CI_COMMIT_SHORT_SHA}"' in build_script
    publish = config["wheel-publish:pypi"]
    assert {need["job"] for need in publish["needs"]} == {
        "wheel:x86_64",
        "wheel:aarch64",
        "wheel-pytest:x86_64",
        "wheel-pytest:aarch64",
        "ctest:x86_64",
        "ctest:aarch64",
    }
    assert {need["job"] for need in publish["needs"] if need["artifacts"]} == {
        "wheel:x86_64",
        "wheel:aarch64",
    }
    publish_script = "\n".join(config[".wheel_publish"]["script"])
    assert "TWINE_USERNAME=gitlab-ci-token" in publish_script
    assert 'TWINE_PASSWORD="${CI_JOB_TOKEN}"' in publish_script
    assert "python -m twine check dist/*.whl" in publish_script
    assert "python -m twine upload" in publish_script
    assert "--skip-existing" not in publish_script
    assert "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi" in publish_script
    assert "packages/generic" not in publish_script
    assert config[".wheel_publish"]["rules"][1]["if"] == '$CI_COMMIT_BRANCH == "main"'
    assert config[".wheel_publish"]["rules"][2] == {"if": '$PUBLISH_WHEELS == "true"', "when": "on_success"}
    assert config["variables"]["PUBLISH_WHEELS"]["value"] == "false"
    assert "pyyaml twine" in (REPO_ROOT / "Dockerfile.ci").read_text()
    assert "Description-Content-Type: text/markdown" in (REPO_ROOT / "METADATA").read_text()
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    assert "PHOENIX_INSTALL_TEST_TORCH" not in cmake


def test_training_is_teed_without_a_postprocess_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_ACCOUNT", "account")
    monkeypatch.setenv("CLUSTER", "cwdfw")
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
    monkeypatch.setattr(
        slurm_job.PerfReport,
        "report",
        lambda *args, **kwargs: calls.append((args, kwargs)) or False,
    )
    deterministic_args = _args(tmp_path, flavors=["detr"], baseline={})
    deterministic_args.flavors = slurm_job.FlavorSelection(deterministic_args.flavors)

    assert not deterministic_args.flavors.policy.epilog(deterministic_args, tmp_path / "run.log")
    assert len(calls) == 1
    valgrind_args = _args(tmp_path, flavors=["vg", "detr"], baseline={})
    valgrind_args.flavors = slurm_job.FlavorSelection(valgrind_args.flavors)

    assert valgrind_args.flavors.policy.epilog(valgrind_args, tmp_path / "run.log")
    assert len(calls) == 1


def test_valgrind_epilog_uses_log_reporting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    log = tmp_path / "run.log"
    log.write_text("")
    validated = []
    monkeypatch.setattr(slurm_job.log_report, "validate_checkpoint_saves", lambda *args: validated.append(args))
    monkeypatch.setattr(slurm_job, "_write_and_compare_arguments", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        slurm_job,
        "_log_otel_checkpoint_timings",
        lambda *_args: pytest.fail("valgrind must not require OTel files"),
    )

    slurm_job._epilog(
        [
            "--nodes=1",
            "--gpus-per-node=1",
            "--config=pico",
            "--flavor=detr,vg,native",
            f"--log-file={log}",
            '--baseline-json={"metrics":{}}',
        ],
    )

    assert validated


def test_deterministic_report_reads_the_native_routines_log_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    calls = []
    monkeypatch.setattr(
        slurm_job.PerfReport,
        "report",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    baseline = {"metrics": {"bw_mbps": 115000}}
    args = _args(
        tmp_path,
        config="nano4",
        nodes=8,
        flavors=["detr"],
        run=["ref:nvrx", "cmp:native"],
        baseline=baseline,
    )
    args.flavors = slurm_job.FlavorSelection(args.flavors)
    log = tmp_path / "logs" / "job.log"

    assert args.flavors.policy.epilog(args, log)

    positional, keywords = calls[0]
    assert positional[0] == "detr"
    assert positional[2] == "detr-native"
    assert positional[4] is baseline
    assert positional[3] == tmp_path / "native.cmp" / "logs" / "nano4_8x4_cmp-routine_rank0.log"
    assert keywords == {
        "results_dir": tmp_path / "native.cmp" / "logs",
        "enforced_metrics": slurm_job.MEMORY_REGRESSION_METRICS,
    }


def test_nvrx_only_deterministic_report_uses_its_memory_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb300")
    calls = []
    nvrx_baseline = {"metrics": {"gpu_peak_bytes": 100}}
    monkeypatch.setattr(
        slurm_job.PerfReport,
        "report",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    args = _args(
        tmp_path,
        config="nano4",
        nodes=8,
        flavors=["detr", "async"],
        run=["ref:nvrx", "cmp:nvrx"],
        baseline=nvrx_baseline,
    )
    args.flavors = slurm_job.FlavorSelection(args.flavors)

    assert args.flavors.policy.epilog(args, tmp_path / "logs" / "job.log")

    positional, keywords = calls[0]
    assert positional[0] == "detr-async"
    assert positional[2] == "detr-async-nvrx"
    assert positional[4] is nvrx_baseline
    assert slurm_job._performance_baseline_stem(args) == "gb300_async-detr-nvrx_nano4_8x4"
    assert positional[3] == tmp_path / "nvrx.cmp" / "logs" / "nano4_8x4_cmp-routine_rank0.log"
    assert keywords == {
        "results_dir": tmp_path / "nvrx.cmp" / "logs",
        "enforced_metrics": slurm_job.MEMORY_REGRESSION_METRICS,
    }


def test_flavors_compose_execution_policy_independent_of_cli_order() -> None:
    slurm_job = _load_slurm_job()

    flavors = slurm_job.FlavorSelection(["vg", "detr"])

    # The name follows the command line; what the combination does does not.
    assert flavors.names == ("vg", "detr")
    assert flavors.tag == "vg-detr"
    assert slurm_job.FlavorSelection(["detr", "vg"]).tag == "detr-vg"
    # A per-flavor table names the combination, so either spelling finds it.
    assert "detr-vg" in flavors.table_keys
    assert flavors.policy.phase2_save_interval == 1
    assert flavors.policy.phase2_checkpoints == slurm_job.FlavorPolicy.phase2_checkpoints
    assert flavors.policy.checkpoint_routines == 1
    assert not flavors.policy.manages_native_server
    assert flavors.policy.install_prebuilt
    assert flavors.policy.prep_args() == ["--ensure-prebuilt"]
    assert flavors.policy.worker is None
    assert flavors.policy.server.__self__.__class__ is slurm_job.ValgrindFlavor
    assert flavors.policy.reporter is None


def test_deterministic_cadence_keeps_three_phase2_saves(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    policy = slurm_job.FlavorSelection(["detr"]).policy
    phase1 = policy.train_config("pico", _args(tmp_path), "native", 1)
    interval = policy.phase2_save_interval
    phase1_iteration = policy.phase1_train_iters
    phase2 = policy.train_config("pico", _args(tmp_path), "native", 2)
    opts = slurm_job._train_options("pico", phase2)
    exit_iteration = int(opts[opts.index("--exit-interval") + 1])
    first_save = (phase1_iteration // interval + 1) * interval

    assert phase1_iteration == 50  # noqa: PLR2004
    assert phase1.save_interval == 25  # noqa: PLR2004
    assert phase1.checkpoints == 2  # noqa: PLR2004
    assert phase2.load == phase1.save
    assert phase2.start_iteration == phase1_iteration
    assert interval == 25  # noqa: PLR2004
    assert list(range(first_save, exit_iteration + 1, interval)) == [75, 100, 125]
    # How many runs happen is --run's business, not the flavor's.
    assert policy.checkpoint_routines == 1

def test_checkpoint_interval_override_only_changes_phase2(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, checkpoint_interval=125)
    policy = slurm_job.FlavorSelection(["perf"]).policy

    phase1 = policy.train_config("ultra3bf16_16x4", args, "native", 1)
    phase2 = policy.train_config("ultra3bf16_16x4", args, "native", 2)

    assert phase1.save_interval == slurm_job._PHASE1_TRAIN_ITERS
    assert phase2.save_interval == 125  # noqa: PLR2004
    assert phase2.checkpoints == 3  # noqa: PLR2004


def test_one_iteration_deterministic_cadence_keeps_three_phase2_saves(tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    args = _args(tmp_path, checkpoint_interval=1)
    policy = slurm_job.FlavorSelection(["detr"]).policy

    phase1 = policy.train_config("pico", args, "native", 1)
    phase2 = policy.train_config("pico", args, "native", 2)
    opts = slurm_job._train_options("pico", phase2)

    assert phase1.save_interval == 25  # noqa: PLR2004
    assert phase1.checkpoints == 2  # noqa: PLR2004
    assert phase2.save_interval == 1
    assert phase2.checkpoints == 3  # noqa: PLR2004
    assert int(opts[opts.index("--exit-interval") + 1]) == 53  # noqa: PLR2004


def test_deterministic_flavor_uses_the_pipeline_wheel() -> None:
    policy = _load_slurm_job().FlavorSelection(["detr"]).policy

    assert policy.install_prebuilt
    assert policy.prep_args() == ["--ensure-prebuilt"]


def test_a_variant_that_undoes_another_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)

    # Nothing declares a conflict; it follows from one variant dropping what
    # another adds, in a scope they share.
    with pytest.raises(ValueError, match=r"adds --async-ckpt-use-cpu-shm, which ipc drops"):
        slurm_job.FlavorSelection(["shm", "ipc"])

    # A variant replacing its own option is not a contradiction.
    slurm_job.FlavorSelection(["pt-async"])

    # Execution modes are not refused. Comparing what two of them write is a
    # question the command line may ask; whether it answers is the digest's job.
    assert slurm_job.FlavorSelection(["perf", "detr"]).names == ("perf", "detr")

def test_batch_epilog_reports_from_embedded_baseline_without_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setattr(slurm_job, "_load_yaml", lambda *_: pytest.fail("batch epilog must not load YAML"))
    log = tmp_path / "run.log"
    log.write_text(
        "0: ------------------------ arguments ------------------------\n"
        " 0:   ckpt_provider ................................... native\n"
        " 0:   inference_dynamic_batching_enable_prefix_caching  False\n"
        " 0:   inference_dynamic_batching_unified_memory_level . 0\n"
        " 0:   config_logger_dir ............................... \n"
        " 0:   ckpt_pg_tensors_cache_path ...................... /run/pg-tensors-cache\n"
        " 0:   data_cache_path ................................. /run/data-cache\n"
        " 0: WARNING: unrelated output inside the argument block\n"
        "0:   micro_batch_size ................................ 1\n"
        "0: -------------------- end of arguments ---------------------\n",
    )
    monkeypatch.setattr(
        slurm_job.otel_report,
        "load_latest",
        lambda _path: _otel_data(
            _native_otel_sample(),
            megatron_metrics=_megatron_perf_metric_samples(0),
        ),
    )
    (tmp_path / "checkpoint_server.cycle1.log").write_text("")
    baseline = '{"metrics":{"bw_mbps":140000,"wr_tnsr_us":10,"total_written_bytes":100}}'
    # The epilog compares the options the run trained on as well as its
    # numbers, so record the ones this log carries.
    monkeypatch.setenv("PLATFORM", "devbox")
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    (recorded / "devbox_perf_pico_1x1_arguments.txt").write_text(
        "  ckpt_provider ................................... native\n"
        "  inference_dynamic_batching_enable_prefix_caching  False\n"
        "  inference_dynamic_batching_unified_memory_level . 0\n"
        "  config_logger_dir ...............................\n"
        "  micro_batch_size ................................ 1\n",
    )
    monkeypatch.setattr(slurm_job, "BASELINES_DIR", recorded)
    monkeypatch.setattr(slurm_job, "_log_otel_checkpoint_timings", lambda *_args: None)

    slurm_job._epilog(
        ["--nodes",
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
    arguments = next((tmp_path / "baselines").glob("*_arguments.txt")).read_text()
    assert arguments.startswith("------------------------ arguments")
    assert "\n0:" not in arguments
    assert "data_cache_path" not in arguments
    assert "ckpt_pg_tensors_cache_path" not in arguments
    assert "WARNING" not in arguments
    assert all(line == line.rstrip() for line in arguments.splitlines())
    assert "inference_dynamic_batching_enable_prefix_caching  False" in arguments
    assert "inference_dynamic_batching_unified_memory_level . 0" in arguments
    assert yaml.safe_load(next((tmp_path / "baselines").glob("*.yaml")).read_text()) == {
        "description": "auto-generated by CI",
        "source": "otlp",
        "metrics": {
            "bw_mbps": 150000,
            "wr_tnsr_us": 10,
            "total_written_bytes": 100,
            "megatron.training.memory_allocated_gb": 60,
        },
        "rank_statistics": {
            name: {"min": value, "median": value, "max": value}
            for name, value in {
                "bw_mbps": 150000,
                "wr_tnsr_us": 10,
                "total_written_bytes": 100,
                "megatron.training.memory_allocated_gb": 60.0,
            }.items()
        },
    }


def test_batch_epilog_reads_task_logs_of_an_array_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Rank 0's stream lives in the task log of whichever array task hosted it."""
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURMJOB_OPT_JOB_ARRAY_SEGMENT", "16")
    log = tmp_path / "run.log"
    log.write_text("")  # task 0's stream carries no training output this time
    (tmp_path / "run.task4.log").write_text(
        "0: ------------------------ arguments ------------------------\n"
        " 0:   ckpt_provider ................................... native\n"
        "0:   micro_batch_size ................................ 1\n"
        "0: -------------------- end of arguments ---------------------\n",
    )
    monkeypatch.setattr(slurm_job.otel_report, "load_latest", lambda _path: _otel_data(_native_otel_sample()))
    monkeypatch.setenv("PLATFORM", "devbox")
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    (recorded / "devbox_perf_pico_1x1_arguments.txt").write_text(
        "  ckpt_provider ................................... native\n"
        "  micro_batch_size ................................ 1\n",
    )
    monkeypatch.setattr(slurm_job, "BASELINES_DIR", recorded)
    monkeypatch.setattr(slurm_job, "_log_otel_checkpoint_timings", lambda *_args: None)

    slurm_job._epilog(
        ["--nodes",
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
            '{"metrics":{"bw_mbps":140000,"wr_tnsr_us":10,"total_written_bytes":100}}',
        ],
    )

    combined = (tmp_path / "run.combined.log").read_text()
    assert "end of arguments" in combined
    assert list(tmp_path.glob("*_report.json"))


def test_nvrx_perf_writes_a_baseline_for_every_reported_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(
        tmp_path,
        config="pico",
        nodes=1,
        gpus_per_node=4,
        flavor=["perf", "nvrx"],
        flavors=slurm_job.FlavorSelection(["perf", "nvrx"]),
    )
    _cfg, _flavor, stem = slurm_job._identity(args)
    log = tmp_path / "logs" / f"{stem}.log"
    log.parent.mkdir()
    log.write_text("")
    report = {
        "telemetry_source": "otlp",
        "ckpt_save_spans_valid": False,
        "metrics": {
            "ckpt_durable_ms": {"mean": 10, "higher_is_better": False, "unit": "ms"},
            "gpu_peak_bytes": {"mean": 100, "higher_is_better": False, "unit": "B"},
            "gpu_hbm_peak_bytes": {"max": 120, "higher_is_better": False, "unit": "B"},
            "training_process_peak_rss_bytes": {"max": 200, "higher_is_better": False, "unit": "B"},
        },
    }
    log.with_name(f"{stem}_report.json").write_text(json.dumps(report))

    slurm_job._write_performance_baseline(args, log.parent)

    candidate = yaml.safe_load((log.parent / "baselines" / f"{stem}.yaml").read_text())
    assert candidate["metrics"] == {
        "ckpt_durable_ms": 10,
        "gpu_peak_bytes": 100,
        "gpu_hbm_peak_bytes": 120,
        "training_process_peak_rss_bytes": 200,
    }


def test_native_perf_refuses_an_incomplete_baseline_candidate(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(
        tmp_path,
        config="pico",
        nodes=1,
        gpus_per_node=4,
        flavor=["perf", "native"],
        flavors=slurm_job.FlavorSelection(["perf", "native"]),
    )
    _cfg, _flavor, stem = slurm_job._identity(args)
    log = tmp_path / "logs" / f"{stem}.log"
    log.parent.mkdir()
    log.write_text("")
    log.with_name(f"{stem}_report.json").write_text(
        '{"telemetry_source":"otlp","ckpt_save_spans_valid":false,"metrics":{"gpu_peak_bytes":{"mean":100}}}',
    )

    with caplog.at_level("WARNING"):
        slurm_job._write_performance_baseline(args, log.parent)

    assert not list((log.parent / "baselines").glob("*.yaml"))
    assert "refusing to write an incomplete baseline candidate" in caplog.text
    assert "required metrics were absent" in caplog.text


def test_log_report_writes_a_log_baseline_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(
        tmp_path,
        config="pico",
        nodes=1,
        gpus_per_node=4,
        flavor=["perf", "native"],
        flavors=slurm_job.FlavorSelection(["perf", "native"]),
    )
    _cfg, _flavor, stem = slurm_job._identity(args)
    report = {
        "telemetry_source": "logs",
        "ckpt_save_markers_valid": True,
        "metrics": {
            "bw_mbps": {"median": 150000, "higher_is_better": True, "unit": "MB/s"},
            "wr_tnsr_us": {"median": 10, "higher_is_better": False, "unit": "us"},
            "total_written_bytes": {"mean": 100, "higher_is_better": None, "unit": "B"},
            "checkpoint_daemon_peak_rss_bytes": {"max": 200, "higher_is_better": False, "unit": "B"},
            "training_process_peak_rss_bytes": {"max": 300, "higher_is_better": False, "unit": "B"},
        },
    }
    (tmp_path / f"{stem}_report.json").write_text(json.dumps(report))

    slurm_job._write_performance_baseline(args, tmp_path)

    candidate = yaml.safe_load((tmp_path / "baselines" / f"{stem}.yaml").read_text())
    assert candidate["source"] == "logs"
    assert candidate["metrics"] == {
        "bw_mbps": 150000,
        "wr_tnsr_us": 10,
        "total_written_bytes": 100,
        "checkpoint_daemon_peak_rss_bytes": 200,
        "training_process_peak_rss_bytes": 300,
    }
    assert candidate["tolerances"] == {
        "checkpoint_daemon_peak_rss_bytes": 0.10,
        "training_process_peak_rss_bytes": 0.10,
    }


def test_every_workload_baseline_tracks_gpu_peak_memory() -> None:
    missing = []
    for path in sorted((REPO_ROOT / "ci" / "baselines").glob("*.yaml")):
        if path.name == "platform_bw_mbps.yaml":
            continue
        if "gpu_peak_bytes" not in yaml.safe_load(path.read_text()).get("metrics", {}):
            missing.append(path.name)

    assert not missing


def test_every_host_memory_baseline_uses_host_memory_tolerance() -> None:
    slurm_job = _load_slurm_job()
    for path in sorted((REPO_ROOT / "ci" / "baselines").glob("*.yaml")):
        baseline = yaml.safe_load(path.read_text())
        metrics = baseline.get("metrics", {})
        tolerances = baseline.get("tolerances", {})
        for metric in slurm_job.HOST_MEMORY_METRICS & metrics.keys():
            assert tolerances.get(metric) == slurm_job.HOST_MEMORY_TOLERANCE, f"{path.name}: {metric}"


def test_recorded_bandwidth_takes_precedence_over_the_platform_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    baseline_path = tmp_path / "workload.yaml"
    baseline_path.write_text("metrics:\n  bw_mbps: 150000\n")
    monkeypatch.setattr(slurm_job, "BASELINES_DIR", tmp_path)
    monkeypatch.setattr(slurm_job.PerfReport, "_platform_bandwidth_baseline", lambda _platform: 115000)

    assert slurm_job.PerfReport.baseline("workload")["metrics"]["bw_mbps"] == 150000  # noqa: PLR2004


def test_log_report_rejects_incomplete_checkpoint_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
    log = tmp_path / "run.log"
    log.write_text(
        "successfully saved checkpoint from iteration 100 to /run/phase1-checkpoints\n"
        "successfully saved checkpoint from iteration 250 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 500 to /run/checkpoints",
    )

    with pytest.raises(RuntimeError, match=r"found 1 phase-1 and 2 phase-2 saves; expected at least 1 and 3"):
        slurm_job._epilog(
            [
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


def _arguments_case(
    slurm_job: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    produced: str,
    recorded: str | None,
) -> bool:
    args = _args(tmp_path, config="pico", nodes=1, flavors=["perf"])
    log = tmp_path / "logs" / "run.log"
    (log.parent / "baselines").mkdir(parents=True)
    log.write_text("")
    stem = slurm_job._identity(args)[2]
    (log.parent / "baselines" / f"{stem}_arguments.txt").write_text(produced)
    baselines = tmp_path / "recorded"
    baselines.mkdir()
    if recorded is not None:
        (baselines / f"{stem}_arguments.txt").write_text(recorded)
    monkeypatch.setattr(slurm_job, "BASELINES_DIR", baselines)
    return slurm_job._compare_arguments(args, log)


def test_a_changed_training_option_does_not_pass_as_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    recorded = (
        "  recompute_granularity ........................... None\n"
        "  recompute_modules ............................... None\n"
        "  micro_batch_size ................................ 1\n"
    )
    # The run this branch performs recomputes what the recorded run held.
    changed = recorded.replace("recompute_granularity ........................... None",
                               "recompute_granularity ........................... selective")

    assert _arguments_case(slurm_job, monkeypatch, tmp_path / "same", recorded, recorded) is True
    assert _arguments_case(slurm_job, monkeypatch, tmp_path / "changed", changed, recorded) is False


def test_two_jobs_comparing_different_providers_record_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")

    def stem(compared: str) -> str:
        args = _args(tmp_path, config="pico", nodes=1, flavors=["detr"], flavor=["detr"],
                     run=["ref:nvrx", f"cmp:{compared}"])
        return slurm_job._arguments_stem(args)

    # A job reports the options of the last run it trained, and these two train
    # different providers there, so one recorded file cannot describe both.
    assert stem("native") != stem("nvrx")
    assert stem("native").endswith("-native")
    assert stem("nvrx").endswith("-nvrx")


def test_native_detr_compares_only_native_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    args = _args(
        tmp_path,
        config="pico",
        nodes=1,
        gpus_per_node=4,
        flavors=["detr"],
        flavor=["detr"],
        run=["ref:nvrx", "cmp:native"],
    )
    run_root = tmp_path / "job"
    batch_log = run_root / "logs" / "job.log"
    batch_log.parent.mkdir(parents=True)
    batch_log.write_text("")
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    monkeypatch.setattr(slurm_job, "BASELINES_DIR", recorded)

    specs = slurm_job._run_specs(args)
    for spec in specs:
        log = slurm_job._argument_log(args, batch_log, spec)
        log.parent.mkdir(parents=True)
        provider = spec.variants[0]
        block = (
            "------------------------ arguments ------------------------\n"
            f"  ckpt_provider ................................... {provider}\n"
            "  micro_batch_size ................................ 1\n"
            "-------------------- end of arguments ---------------------\n"
        )
        log.write_text(block)
        if provider == "native":
            stem = slurm_job._arguments_stem(spec.args, ())
            (recorded / f"{stem}_arguments.txt").write_text(block)

    assert slurm_job._write_and_compare_arguments(args, batch_log) is True
    for spec in specs:
        log = slurm_job._argument_log(args, batch_log, spec)
        candidate = log.parent / "baselines" / f"{slurm_job._arguments_stem(spec.args, ())}_arguments.txt"
        assert candidate.exists() is (spec.variants[0] == "native")
        if candidate.exists():
            assert "ckpt_provider ................................... native" in candidate.read_text()


def test_arguments_with_nothing_recorded_do_not_pass_uncompared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("PLATFORM", "gb200")
    arguments = "  micro_batch_size ................................ 1\n"

    # Nothing was compared, which is not the same as nothing having changed.
    assert _arguments_case(slurm_job, monkeypatch, tmp_path, arguments, None) is False

    # A file holding only the rule that delimits the block yields no options,
    # so it agrees with nothing rather than agreeing with everything.
    rule = "------------------------ arguments ------------------------\n"
    assert _arguments_case(slurm_job, monkeypatch, tmp_path / "empty-produced", rule, arguments) is False
    assert _arguments_case(slurm_job, monkeypatch, tmp_path / "empty-recorded", arguments, rule) is False


def test_the_rank_that_failed_is_named_and_shown_from_a_nested_run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    run_root = tmp_path / "job"
    run_log = run_root / "logs" / "job.log"
    run_log.parent.mkdir(parents=True)
    # srun terminates every other rank after one exits. A log reader may prefix
    # the line before what srun wrote.
    run_log.write_text(
        "2026-08-12T18:00:00Z 01O srun: error: n51: task 58: Exited with exit code 1\n"
        "2026-08-12T18:00:00Z 01O srun: error: n51: tasks 56-57,59: Terminated\n"
        "2026-08-12T18:00:00Z 01O srun: error: n37: tasks 0-3: Terminated\n",
    )
    rank_log = run_root / "native.cmp" / "logs" / "super3bf16_16x4_cmp-routine_rank58.log"
    rank_log.parent.mkdir(parents=True)
    rank_log.write_text(
        "starting\n[rank58]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.91 GiB\n",
    )

    with caplog.at_level("INFO"):
        slurm_job._report_failing_rank_logs(run_log)

    assert "rank(s) 58 exited" in caplog.text
    assert "native.cmp/logs/super3bf16_16x4_cmp-routine_rank58.log" in caplog.text
    assert "torch.OutOfMemoryError" in caplog.text
    # The ranks srun merely terminated are consequences, not the cause.
    assert "rank0" not in caplog.text


def test_ranks_srun_only_terminated_implicate_nobody() -> None:
    slurm_job = _load_slurm_job()

    assert slurm_job._failing_ranks("srun: error: n1: tasks 0-3: Terminated") == []
    assert slurm_job._failing_ranks("srun: error: n1: task 58: Exited with exit code 1") == [58]
    # A step can lose several ranks at once, named as a range and a list.
    assert slurm_job._failing_ranks(
        "srun: error: n1: tasks 56-57,59: Exited with exit code 2",
    ) == [56, 57, 59]


def test_failing_rank_log_output_is_bounded(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    run_root = tmp_path / "job"
    run_log = run_root / "logs" / "job.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text("srun: error: n1: tasks 1-3: Exited with exit code 1\n")
    for rank in (1, 2, 3):
        rank_log = run_root / "logs" / f"routine_rank{rank}.log"
        rank_log.write_text("old line\nexception tail\n")

    with caplog.at_level("INFO"):
        slurm_job._report_failing_rank_logs(run_log, rank_limit=2, tail_lines=1)

    assert "rank1] exception tail" in caplog.text
    assert "rank2] exception tail" in caplog.text
    assert "rank3]" not in caplog.text
    assert "1 additional exited rank(s)" in caplog.text


def test_log_report_requires_both_deterministic_checkpoint_routines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    _clear_run_env(monkeypatch)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
    log = tmp_path / "run.log"
    log.write_text(
        "successfully saved checkpoint from iteration 100 to /run/phase1-checkpoints\n"
        "successfully saved checkpoint from iteration 120 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 150 to /run/checkpoints\n"
        "successfully saved checkpoint from iteration 180 to /run/checkpoints",
    )

    with pytest.raises(RuntimeError, match=r"expected at least 2 and 6"):
        slurm_job._epilog(
            [
                "--nodes",
                "1",
                "--gpus-per-node",
                "1",
                "--config",
                "pico",
                "--flavor",
                "detr",
                "--checkpoints-per-run",
                "3",
                "--log-file",
                str(log),
                "--baseline-json",
                '{"metrics":{"bw_mbps":140000}}',
                "--run=ref",
                "--run=cmp",
            ],
        )


def test_otel_metrics_support_python_without_strict_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    builtin_zip = zip
    monkeypatch.setattr(slurm_job, "zip", builtin_zip, raising=False)
    samples = [_native_otel_sample(rank, bandwidth) for rank, bandwidth in enumerate((100, 100, 80, 80))]

    metrics = slurm_job.otel_report.metrics_from_data(_otel_data(*samples), 4)

    assert metrics["bw_mbps"]["median"] == 90  # noqa: PLR2004
    assert metrics["bw_socket_imbalance"]["mean"] == 1.25  # noqa: PLR2004


def test_native_perf_spans_require_the_complete_schema_and_timestamps(caplog: pytest.LogCaptureFixture) -> None:
    slurm_job = _load_slurm_job()

    sample = _native_otel_sample(rank=2, snapshot=7, write_us=1_000_000)
    assert slurm_job.otel_report.native_spans_valid([sample])
    assert slurm_job.otel_report.metrics_from_data(_otel_data(sample))["wr_tnsr_us"]["mean"] == 1_000_000

    with caplog.at_level("WARNING"):
        assert not slurm_job.otel_report.native_spans_valid([{key: value for key, value in sample.items() if key != "bw_mbps"}])
    assert "did not match the required schema" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        assert not slurm_job.otel_report.native_spans_valid([{**sample, "end_time_unix_nano": 1}])
    assert "invalid timestamps" in caplog.text


def test_native_perf_spans_warn_when_the_measured_cycle_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    slurm_job = _load_slurm_job()

    with caplog.at_level("WARNING"):
        assert not slurm_job.otel_report.native_spans_valid([])

    assert "no ckpt.save spans found" in caplog.text


def test_perf_metrics_have_no_log_fallback() -> None:
    slurm_job = _load_slurm_job()
    sample = _native_otel_sample(rank=2, bandwidth=42, write_us=20, written_bytes=200, snapshot=7)

    metrics = slurm_job.otel_report.metrics_from_data(_otel_data(sample))
    assert metrics["bw_mbps"]["mean"] == 42  # noqa: PLR2004
    assert slurm_job.otel_report.metrics_from_data(_otel_data()) == {}


def test_log_report_requires_matching_perf_markers(caplog: pytest.LogCaptureFixture) -> None:
    slurm_job = _load_slurm_job()

    for suffix in ("us", "µs"):
        one_second = _perf_span(write_us=1_000_000).replace("1000000us", f"1000000{suffix}")
        assert slurm_job.log_report._validate_daemon_markers(one_second)
        assert slurm_job.log_report._metrics("", one_second, 1)["wr_tnsr_us"]["mean"] == 1_000_000

    malformed = _perf_span(rank=2, snapshot=7).replace(" snapshot_us=11us", "")
    with caplog.at_level("WARNING"):
        assert not slurm_job.log_report._validate_daemon_markers(malformed)
    assert "did not match the required schema" in caplog.text

    caplog.clear()
    mismatched = _perf_span(rank=2, snapshot=7).replace("snapshot=7\n", "snapshot=8\n", 1)
    with caplog.at_level("WARNING"):
        assert not slurm_job.log_report._validate_daemon_markers(mismatched)
    assert "without Finished" in caplog.text
    assert "without Begin" in caplog.text


def test_perf_report_uses_logs_only_when_otel_is_disabled(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.delenv("NATIVE_CKPT_OTEL_FILE", raising=False)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
    monkeypatch.setattr(slurm_job.otel_report, "post_process", lambda *_args: pytest.fail("OTLP must be disabled"))
    log = tmp_path / "run.log"
    log.write_text(
        "[native_ckpt:phase] phase=2\n"
        "[2026-08-15 11:00:00.000000] saving checkpoint at iteration 300 to /ckpt\n"
        "[2026-08-15 11:00:20.000000] successfully saved checkpoint from iteration 300 to /ckpt\n"
        "[native_ckpt:mem] rank=0 gpu_peak_bytes=80 gpu_hbm_peak_bytes=90 "
        "gpu_total_bytes=100 cpu_rss_bytes=10\n",
    )
    (tmp_path / "checkpoint_server.cycle1.host.1.log").write_text(
        _perf_span(write_us=8, snapshot=1) + _perf_span(write_us=10, snapshot=2) + _perf_span(write_us=12, snapshot=3),
    )
    daemon = slurm_job.otel_report.metric([20], "B", higher_is_better=False)
    monkeypatch.setattr(
        slurm_job.otel_report,
        "complete_run_process_memory_peaks",
        lambda _results_dir: {"checkpoint_daemon_peak_rss_bytes": daemon},
    )
    report_path = tmp_path / "report.json"

    with caplog.at_level("INFO"):
        passed = slurm_job.PerfReport.report(
            "perf",
            "pico_1x1",
            "perf",
            log,
            {"metrics": {"bw_mbps": 150000, "wr_tnsr_us": 10, "total_written_bytes": 100}},
            report_path,
            1,
        )

    report = json.loads(report_path.read_text())
    assert passed
    assert report["telemetry_source"] == "logs"
    assert report["ckpt_save_markers_valid"]
    assert report["metrics"]["ckpt_durable_ms"]["mean"] == 20000
    assert report["metrics"]["training_process_peak_rss_bytes"]["max"] == 10  # noqa: PLR2004
    assert report["metrics"]["checkpoint_daemon_peak_rss_bytes"]["max"] == 20  # noqa: PLR2004
    assert "[no_baseline] training_process_peak_rss_bytes: max=10 B" in caplog.text
    assert "[no_baseline] checkpoint_daemon_peak_rss_bytes: max=20 B" in caplog.text
    assert ("[PASS] wr_tnsr_us: median=10 us (min=8 us, median=10 us, max=12 us); baseline=10.0 us; tolerance=5.0%") in caplog.messages


def test_perf_report_explicit_log_reporting_overrides_otel_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("NATIVE_CKPT_OTEL_FILE", str(tmp_path / "otel.jsonl"))
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
    monkeypatch.setattr(slurm_job.otel_report, "post_process", lambda *_args: pytest.fail("OTLP must be disabled"))
    monkeypatch.setattr(
        slurm_job.log_report,
        "post_process",
        lambda *_args: {"source": "logs", "metrics": {}, "validation": None},
    )

    assert slurm_job.PerfReport.report(
        "detr",
        "pico_1x1",
        "detr-nvrx",
        tmp_path / "run.log",
        {},
        tmp_path / "report.json",
        1,
        otel_reporting=False,
    )


def test_perf_report_can_enforce_only_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    metrics = {
        "bw_mbps": slurm_job.otel_report.metric([10], "MB/s", higher_is_better=True),
        "gpu_peak_bytes": slurm_job.otel_report.metric([100], "B", higher_is_better=False),
        "gpu_hbm_peak_bytes": slurm_job.otel_report.metric([120], "B", higher_is_better=False),
        "training_process_peak_rss_bytes": slurm_job.otel_report.metric([100], "B", higher_is_better=False),
        "checkpoint_daemon_peak_rss_bytes": slurm_job.otel_report.metric([200], "B", higher_is_better=False),
    }
    monkeypatch.setattr(
        slurm_job.log_report,
        "post_process",
        lambda *_args: {
            "source": "logs",
            "metrics": metrics,
            "validation": {
                "name": "ckpt_save_markers",
                "valid": False,
                "measured": 0,
                "baseline": 1,
                "report_field": "ckpt_save_markers_valid",
            },
        },
    )
    baseline = {
        "metrics": {
            "bw_mbps": 1000,
            "gpu_peak_bytes": 100,
            "gpu_hbm_peak_bytes": 120,
            "training_process_peak_rss_bytes": 100,
            "checkpoint_daemon_peak_rss_bytes": 200,
        },
    }

    assert slurm_job.PerfReport.report(
        "detr",
        "pico_1x1",
        "detr-native",
        tmp_path / "run.log",
        baseline,
        tmp_path / "report.json",
        1,
        otel_reporting=False,
        enforced_metrics=slurm_job.MEMORY_REGRESSION_METRICS,
    )

    baseline["metrics"]["gpu_peak_bytes"] = 50
    assert not slurm_job.PerfReport.report(
        "detr",
        "pico_1x1",
        "detr-native",
        tmp_path / "run.log",
        baseline,
        tmp_path / "report.json",
        1,
        otel_reporting=False,
        enforced_metrics=slurm_job.MEMORY_REGRESSION_METRICS,
    )

    baseline["metrics"]["gpu_peak_bytes"] = 100
    baseline["metrics"]["training_process_peak_rss_bytes"] = 50
    assert not slurm_job.PerfReport.report(
        "detr",
        "pico_1x1",
        "detr-native",
        tmp_path / "run.log",
        baseline,
        tmp_path / "report.json",
        1,
        otel_reporting=False,
        enforced_metrics=slurm_job.MEMORY_REGRESSION_METRICS,
    )

    baseline["metrics"]["training_process_peak_rss_bytes"] = 100
    baseline["metrics"]["gpu_hbm_peak_bytes"] = 100
    assert not slurm_job.PerfReport.report(
        "detr",
        "pico_1x1",
        "detr-native",
        tmp_path / "run.log",
        baseline,
        tmp_path / "report.json",
        1,
        otel_reporting=False,
        enforced_metrics=slurm_job.MEMORY_REGRESSION_METRICS,
    )


def test_perf_report_does_not_fall_back_when_otel_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("NATIVE_CKPT_OTEL_FILE", str(tmp_path / "otel.jsonl"))
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
    monkeypatch.setattr(slurm_job.log_report, "post_process", lambda *_args: pytest.fail("logs must not be used"))
    monkeypatch.setattr(slurm_job.otel_report, "load_latest", lambda _path: _otel_data())
    log = tmp_path / "run.log"
    log.write_text(_perf_span())

    with pytest.raises(RuntimeError, match="no perf metrics found in otlp report data"):
        slurm_job.PerfReport.report("perf", "pico_1x1", "perf-native", log, {}, tmp_path / "report.json", 1)


def test_native_perf_report_cannot_pass_on_memory_metrics_alone(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "run.log"
    report_path = tmp_path / "report.json"
    log.write_text("[native_ckpt:mem] rank=0 gpu_peak_bytes=100 cpu_rss_bytes=10\n")
    resource_metric = {
        "name": "nvrx.checkpoint.run.gpu_peak_bytes",
        "resource": {"dl.rank": 0},
        "value": 100,
    }
    monkeypatch.setattr(slurm_job.otel_report, "load_latest", lambda _path: _otel_data(resource_metrics=[resource_metric]))

    with caplog.at_level("WARNING"):
        passed = slurm_job.PerfReport.report(
            "perf-native",
            "pico_1x1",
            "perf-native",
            log,
            {"metrics": {"bw_mbps": 100, "wr_tnsr_us": 10, "total_written_bytes": 100, "gpu_peak_bytes": 100}},
            report_path,
            1,
        )

    report = yaml.safe_load(report_path.read_text())
    assert not passed
    assert not report["passed"]
    assert not report["ckpt_save_spans_valid"]
    assert report["comparison"]["bw_mbps"]["reason"] == "missing_metric"
    assert report["comparison"]["ckpt_save_spans"]["status"] == "FAIL"
    assert "required baseline metric 'wr_tnsr_us' was not measured" in caplog.text


def test_durability_pairs_async_saves_by_call_index() -> None:
    slurm_job = _load_slurm_job()
    spans = [
        _checkpoint_span("megatron.checkpoint.save", 0, 2, call_idx=7),
        _checkpoint_span("megatron.checkpoint.save", 10, 12, call_idx=8),
        _checkpoint_span("megatron.checkpoint.save.finalize", 18, 20, call_idx=[8]),
        _checkpoint_span("megatron.checkpoint.exit_finalize", 20, 25, call_idx=[7]),
    ]

    assert slurm_job.otel_report.checkpoint_durability_ms(spans) == ([25.0, 10.0], [0, 0])


def test_durability_pairs_native_saves_and_finalizers_in_fifo_order() -> None:
    slurm_job = _load_slurm_job()
    spans = [
        _checkpoint_span("megatron.checkpoint.save", 0, 2),
        _checkpoint_span("megatron.checkpoint.save", 10, 12),
        _checkpoint_span("nvrx.checkpoint.wait_previous", 18, 20, call_idx=[0]),
        _checkpoint_span("megatron.checkpoint.exit_finalize", 20, 25, call_idx=[1]),
    ]

    assert slurm_job.otel_report.checkpoint_durability_ms(spans) == ([20.0, 15.0], [0, 0])


def test_durability_warns_and_pairs_malformed_nvrx_indices_in_fifo_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    spans = [
        _checkpoint_span("megatron.checkpoint.save", 0, 2, call_idx=0),
        _checkpoint_span("megatron.checkpoint.save", 10, 12, call_idx=1),
        _checkpoint_span("megatron.checkpoint.save", 20, 22, call_idx=2),
        _checkpoint_span("megatron.checkpoint.save.finalize", 4, 5, call_idx=[0]),
        _checkpoint_span("megatron.checkpoint.save.finalize", 14, 15, call_idx=[0]),
        _checkpoint_span("megatron.checkpoint.exit_finalize", 24, 30, call_idx=[0]),
    ]

    with caplog.at_level("WARNING"):
        assert slurm_job.otel_report.checkpoint_durability_ms(spans) == ([5.0, 5.0, 10.0], [0, 0, 0])

    assert "finalizer call indices mismatch" in caplog.text
    assert "pairing in FIFO order" in caplog.text


def test_durability_aggregates_identical_index_warnings_across_ranks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    slurm_job = _load_slurm_job()
    spans = [
        _checkpoint_span("megatron.checkpoint.save", 0, 2, call_idx=0, rank=rank)
        for rank in range(16)
    ] + [
        _checkpoint_span("megatron.checkpoint.save.finalize", 4, 5, call_idx=[1], rank=rank)
        for rank in range(16)
    ]

    with caplog.at_level("WARNING"):
        slurm_job.otel_report.checkpoint_durability_ms(spans)

    assert caplog.text.count("finalizer call indices mismatch") == 1
    assert "ranks=0-15 (16 total)" in caplog.text


def test_durability_requires_async_finalizer_span() -> None:
    slurm_job = _load_slurm_job()

    with pytest.raises(RuntimeError, match="no finalizer"):
        slurm_job.otel_report.checkpoint_durability_ms(
            [_checkpoint_span("megatron.checkpoint.save", 0, 2, call_idx=7)],
        )

    with pytest.raises(RuntimeError, match="finalizer count mismatch"):
        slurm_job.otel_report.checkpoint_durability_ms(
            [
                _checkpoint_span("megatron.checkpoint.save", 0, 2, call_idx=7),
                _checkpoint_span("megatron.checkpoint.save", 10, 12, call_idx=8),
                _checkpoint_span("megatron.checkpoint.save.finalize", 4, 5, call_idx=[0]),
            ],
        )


def test_durability_is_reported_for_both_providers() -> None:
    slurm_job = _load_slurm_job()
    saves = [
        _checkpoint_span("megatron.checkpoint.save", 0, 20_000),
        _checkpoint_span("megatron.checkpoint.save", 0, 24_000, rank=4),
    ]
    without_daemon = slurm_job.otel_report.metrics_from_data(_otel_data(megatron_spans=saves))
    assert without_daemon["ckpt_durable_ms"]["mean"] == 22000.0  # noqa: PLR2004
    assert without_daemon["ckpt_durable_ms"]["rank_min"] == 20000.0  # noqa: PLR2004
    assert without_daemon["ckpt_durable_ms"]["rank_median"] == 22000.0  # noqa: PLR2004
    assert without_daemon["ckpt_durable_ms"]["rank_max"] == 24000.0  # noqa: PLR2004
    assert without_daemon["ckpt_durable_ms"]["higher_is_better"] is False
    assert "ckpt_daemon_durable_ms" not in without_daemon

    data = _otel_data(megatron_spans=saves)
    data["daemon_durable_spans"] = [_checkpoint_span("ckpt.svr.durable", 0, 11_500)]
    with_daemon = slurm_job.otel_report.metrics_from_data(data)
    assert with_daemon["ckpt_daemon_durable_ms"]["mean"] == 11500.0  # noqa: PLR2004


def test_daemon_durability_is_summarised_from_otel(caplog: pytest.LogCaptureFixture) -> None:
    slurm_job = _load_slurm_job()
    metrics = {
        "ckpt_daemon_durable_ms": slurm_job.otel_report.metric(
            [11000.0, 12000.0],
            "ms",
            higher_is_better=False,
        ),
    }

    with caplog.at_level("INFO"):
        slurm_job.PerfReport._log_daemon_durability(metrics)

    assert "daemon activate -> durable" in caplog.text
    assert "(11000.00, 12000.00)  mean 11500.00  over 2 save(s)" in caplog.text


def test_durability_summary_is_silent_without_checkpoints(caplog: pytest.LogCaptureFixture) -> None:
    """An nvrx run with no daemon must not claim a daemon measurement."""
    slurm_job = _load_slurm_job()

    with caplog.at_level("INFO"):
        slurm_job.PerfReport._log_daemon_durability({})
        slurm_job.PerfReport._log_daemon_durability(
            {"ckpt_durable_ms": slurm_job.otel_report.metric([9000.0], "ms", higher_is_better=False)},
        )

    assert "daemon activate" not in caplog.text


def test_durability_carries_a_wide_tolerance() -> None:
    """Lustre variance must not fail a job, but a change beyond the wider limit should."""
    slurm_job = _load_slurm_job()

    def compare(measured: float, tolerances: dict | None = None) -> tuple[dict, bool]:
        metrics = {"ckpt_durable_ms": slurm_job.otel_report.metric([measured], "ms", higher_is_better=False)}
        baseline = {"metrics": {"ckpt_durable_ms": 20000}, "tolerances": tolerances or {}}
        return slurm_job.PerfReport._compare_to_baseline(metrics, baseline, slurm_job.PERF_TOLERANCE)

    result, passed = compare(119000.0)
    assert passed
    assert result["ckpt_durable_ms"]["tolerance_pct"] == 500.0  # noqa: PLR2004
    assert not compare(121000.0)[1]
    # A baseline that states its own tolerance still wins.
    assert not compare(40000.0, {"ckpt_durable_ms": 0.1})[1]


def test_a_metric_missing_from_the_baseline_does_not_fail_the_job() -> None:
    slurm_job = _load_slurm_job()
    metrics = {"ckpt_durable_ms": slurm_job.otel_report.metric([20000.0], "ms", higher_is_better=False)}

    result, passed = slurm_job.PerfReport._compare_to_baseline(metrics, {"metrics": {}}, 0.1)

    assert passed
    assert result["ckpt_durable_ms"]["status"] == "no_baseline"


def test_gpu_peak_regression_fails_the_baseline_comparison() -> None:
    slurm_job = _load_slurm_job()
    metrics = {"gpu_peak_bytes": slurm_job.otel_report.metric([106], "B", higher_is_better=False)}

    result, passed = slurm_job.PerfReport._compare_to_baseline(
        metrics,
        {"metrics": {"gpu_peak_bytes": 100}},
        slurm_job.PERF_TOLERANCE,
    )

    assert not passed
    assert result["gpu_peak_bytes"]["status"] == "FAIL"
    assert result["gpu_peak_bytes"]["tolerance_pct"] == 5


@pytest.mark.parametrize(
    ("name", "measured", "higher_is_better"),
    [
        ("gpu_peak_bytes", 94, False),
        ("bw_mbps", 106, True),
    ],
)
def test_improvement_beyond_tolerance_warns_without_failing(
    name: str,
    measured: int,
    higher_is_better: bool,
) -> None:
    slurm_job = _load_slurm_job()
    metrics = {name: slurm_job.otel_report.metric([measured], "B", higher_is_better=higher_is_better)}

    result, passed = slurm_job.PerfReport._compare_to_baseline(
        metrics,
        {"metrics": {name: 100}},
        slurm_job.PERF_TOLERANCE,
    )

    assert passed
    assert result[name]["status"] == "WARN"


def test_a_required_baseline_metric_missing_from_the_run_fails_the_job(caplog: pytest.LogCaptureFixture) -> None:
    slurm_job = _load_slurm_job()
    metrics = {"gpu_peak_bytes": slurm_job.otel_report.metric([100], "B", higher_is_better=False)}

    with caplog.at_level("WARNING"):
        result, passed = slurm_job.PerfReport._compare_to_baseline(
            metrics,
            {"metrics": {"gpu_peak_bytes": 100, "wr_tnsr_us": 10}},
            0.1,
        )

    assert not passed
    assert result["wr_tnsr_us"] == {
        "status": "FAIL",
        "measured": None,
        "baseline": 10,
        "reason": "missing_metric",
    }
    assert "required baseline metric 'wr_tnsr_us' was not measured" in caplog.text


def test_perf_report_uses_the_largest_rank_hbm_peak() -> None:
    slurm_job = _load_slurm_job()
    resource_metrics = [
        {"name": f"nvrx.checkpoint.run.{name}", "resource": {"dl.rank": rank}, "value": value}
        for rank, values in ((0, (80, 90, 100)), (58, (85, 98, 100)))
        for name, value in zip(("gpu_peak_bytes", "gpu_hbm_peak_bytes", "gpu_total_bytes"), values)
    ]
    resource_metrics += [
        {"name": "nvrx.checkpoint.run.cpu_rss_bytes", "resource": {"dl.rank": rank}, "value": value}
        for rank, value in ((0, 101), (58, 104))
    ]

    metrics = slurm_job.otel_report.metrics_from_data(_otel_data(resource_metrics=resource_metrics), 4)

    assert metrics["gpu_hbm_peak_bytes"]["max"] == 98  # noqa: PLR2004
    assert "gpu_hbm_peak_pct" not in metrics
    assert metrics["cpu_rss_bytes"]["max"] == 104  # noqa: PLR2004


def test_megatron_perf_metrics_report_latest_values_across_export_ranks() -> None:
    slurm_job = _load_slurm_job()
    samples = _megatron_perf_metric_samples(0, 4)
    samples.append(
        {
            "name": "megatron.training.memory_allocated_gb",
            "resource": {"dl.rank": 0},
            "value": 999.0,
            "time_unix_nano": 1,
        },
    )

    metrics = slurm_job.otel_report.metrics_from_data(_otel_data(megatron_metrics=samples), 4)

    memory_allocated = metrics["megatron.training.memory_allocated_gb"]
    assert memory_allocated["per_rank_median"] == {0: 60.0, 4: 64.0}
    assert memory_allocated["rank_min"] == 60.0
    assert memory_allocated["rank_median"] == 62.0
    assert memory_allocated["rank_max"] == 64.0
    assert memory_allocated["median"] == 62.0
    assert slurm_job.PerfReport._statistic(
        "megatron.training.memory_allocated_gb",
        metrics["megatron.training.memory_allocated_gb"],
    ) == "max"


def test_hbm_sampler_tracks_hbm_and_training_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(slurm_job, "_process_rss_bytes", lambda _pid: 123)
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setattr(slurm_job.os, "getpid", lambda: 42)
    samples = iter(((20, 100), (5, 100)))
    devices = []
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 4,
        current_device=lambda: pytest.fail("LOCAL_RANK is a visible CUDA device"),
        mem_get_info=lambda device: devices.append(device) or next(samples),
    )
    sampler = slurm_job._PeakHbmSampler(SimpleNamespace(cuda=cuda))

    sampler._sample_once()
    sampler._sample_once()

    assert devices == [2, 2]
    assert sampler.peak_bytes == 95  # noqa: PLR2004
    assert sampler.total_bytes == 100  # noqa: PLR2004
    assert [rss for _timestamp, rss in sampler.rss_samples] == [123, 123]

    assert all(timestamp > 0 for timestamp, _rss in sampler.rss_samples)


def test_training_process_memory_is_exported_as_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    slurm_job = _load_slurm_job()
    logs_service = pytest.importorskip("opentelemetry.proto.collector.logs.v1.logs_service_pb2")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("MEGATRON_OTEL_RUN_ID", "run-1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4100")
    monkeypatch.setattr(slurm_job.socket, "gethostname", lambda: "node-a.example.com")
    monkeypatch.setattr(slurm_job.os, "getpid", lambda: 42)
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        @staticmethod
        def read() -> bytes:
            return b""

    def urlopen(request: object, *, timeout: int) -> Response:
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(slurm_job.urllib.request, "urlopen", urlopen)

    slurm_job._emit_training_process_memory([(10, 40), (20, 50)])

    http_request, timeout = requests[0]
    assert http_request.full_url == "http://127.0.0.1:4100/v1/logs"
    assert timeout == 30  # noqa: PLR2004
    payload = logs_service.ExportLogsServiceRequest.FromString(http_request.data)
    resource_logs = payload.resource_logs[0]
    resource = {attribute.key: attribute.value for attribute in resource_logs.resource.attributes}
    assert resource["service.name"].string_value == slurm_job.otel_report.TRAINING_SERVICE
    assert resource["host.name"].string_value == "node-a"
    assert resource["process.pid"].int_value == 42  # noqa: PLR2004
    records = resource_logs.scope_logs[0].log_records
    assert [record.event_name for record in records] == ["process.memory", "process.memory"]
    assert [
        next(attribute.value.int_value for attribute in record.attributes if attribute.key == "nvrx.memory.rss_bytes")
        for record in records
    ] == [40, 50]


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
        "clang-23": "/usr/bin/clang-23",
        "clang++-23": "/usr/bin/clang++-23",
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
    assert "-DCMAKE_C_COMPILER=/usr/bin/clang-23" in commands[0]
    assert "-DCMAKE_CXX_COMPILER=/usr/bin/clang++-23" in commands[0]
    assert commands[0][commands[0].index("-B") + 1] == str(build_dir)
    assert commands[1][2] == str(build_dir)


def test_wheel_install_uses_pip_target_instead_of_extracting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    wheel = tmp_path / "nvrx_checkpointing-0.0.1-cp312-abi3-linux_aarch64.whl"
    wheel.write_text("wheel")
    target = tmp_path / "deps/python"
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


def test_upstream_job_artifact_downloads_wheel_without_extracting_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    filename = "nvrx_checkpointing-0.0.1-cp312-abi3-linux_aarch64.whl"
    monkeypatch.setenv("WHEEL_ARCH", "aarch64")
    monkeypatch.setenv("UPSTREAM_WHEEL_AARCH64_JOB_ID", "12345")
    monkeypatch.setenv("CI_API_V4_URL", "https://gitlab.example/api/v4")
    monkeypatch.setenv("CI_PROJECT_ID", "381021")
    monkeypatch.setenv("CI_JOB_TOKEN", "secret")
    requests = []

    def response(request: object, **_kwargs: object) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(f"{filename}\n".encode() if str(request.full_url).endswith(".txt") else b"wheel contents")

    monkeypatch.setattr(wheel_install.urllib.request, "urlopen", response)

    downloaded = wheel_install.fetch_upstream_wheel(tmp_path)

    assert downloaded == tmp_path / filename
    assert downloaded.read_bytes() == b"wheel contents"
    assert requests[0].full_url.endswith("/jobs/12345/artifacts/dist/nvrx_checkpointing-linux_aarch64.txt")
    assert requests[1].full_url.endswith(f"/jobs/12345/artifacts/dist/{filename}")
    assert requests[0].headers["Job-token"] == "secret"


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

    assert wheel_install.install_prebuilt_wheel(tmp_path, probe=False)
    assert installed == [wheel]


def test_wheel_install_main_installs_from_explicit_dist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    installed_from: list[Path | None] = []
    monkeypatch.setattr(sys, "argv", ["wheel_install.py", "--from-dist", str(tmp_path)])
    monkeypatch.setattr(wheel_install, "install_prebuilt_wheel", lambda dist: installed_from.append(dist) or True)

    wheel_install.main()

    assert installed_from == [tmp_path]


def test_wheel_packages_megatron_modules() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    template = (REPO_ROOT / "ci/gitlab-ci-native.template.yml").read_text()

    assert "megatron@megatron" in cmake
    assert "megatron_stubs@megatron_stubs" in cmake
    assert "debug.py" in cmake
    assert "metadata_compare.py" in cmake
    assert "peers.py" in cmake
    assert "scripts@scripts" not in cmake
    assert 'python -c "import nvrx_checkpointing.debug"' in template
    assert "python -m nvrx_checkpointing.metadata_compare --help" in template


def test_source_wheel_build_prepares_ubuntu_when_clang_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_install = _load_slurm_job().wheel_install
    prepared = False

    def which(name: str) -> str | None:
        if name in ("clang-23", "clang++-23") and prepared:
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


def test_vcpkg_bootstrap_uses_and_restores_canonical_origin() -> None:
    cmake = (REPO_ROOT / "cmake" / "vcpkg.cmake").read_text()

    assert 'remote add origin "${VCPKG_AUTOCLONE_URL}"' in cmake
    assert 'fetch --depth 1 origin "${VCPKG_AUTOCLONE_COMMIT}"' in cmake
    assert "remote get-url origin" in cmake
    assert "origin1" not in cmake


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
            "NCCL_NET": "IB",
            "RANK": "0",
            "CI_JOB_TOKEN": "must-not-be-printed",
        },
        "phase1",
    )

    output = capsys.readouterr().out
    assert "[slurm_job:env] phase1 rank0 environment" in output
    assert "[slurm_job:env] PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" in output
    assert "[slurm_job:env] NCCL_IB_TIMEOUT=24" in output
    assert "[slurm_job:env] NCCL_NET=IB" in output
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
    monkeypatch.setattr(slurm_job, "_phase_command", lambda *_, **__: (["train"], {}))
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

    assert perf_opts == [
        "--cuda-graph-impl", "local",
        "--recompute-granularity", "selective", "--recompute-modules", "moe",
        "--async-save",
        "--ckpt-assume-constant-structure", "--async-ckpt-use-cpu-shm",
    ]
    assert not perf_env
    deterministic_opts = [
        "--use-fused-weighted-squared-relu",
        "--cuda-graph-impl",
        "local",
        "--cuda-graph-modules",
        "mamba",
        "attn",
        "moe_router",
        "--moe-router-fusion",
        "--moe-permute-fusion",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
    ]
    deterministic_env = {
        "MEGATRON_OTEL_ENABLED": "1",
        "NATIVE_CKPT_OTEL_FILE": "/tmp/otel.jsonl",
    }
    slurm_job.FlavorSelection(["detr"]).transform(deterministic_opts, deterministic_env)
    assert deterministic_opts == [
        "--moe-permute-fusion",
        "--attention-backend",
        "unfused",
        "--moe-router-dtype",
        "fp64",
        "--deterministic-mode",
    ]
    assert deterministic_env["PYTHONHASHSEED"] == "0"
    assert deterministic_env["NVRX_OTEL_USE_SYSTEM_RANDOM"] == "1"
    assert deterministic_env["FLA_TILELANG"] == "0"
    assert deterministic_env["MEGATRON_OTEL_ENABLED"] == "1"
    assert deterministic_env["NATIVE_CKPT_OTEL_FILE"] == "/tmp/otel.jsonl"
    assert slurm_job.FlavorSelection(["detr"]).policy.otel_reporting
    assert "TRITON_CACHE_AUTOTUNING" not in deterministic_env
    disabled_env = {"MEGATRON_OTEL_ENABLED": "0"}
    slurm_job.FlavorSelection(["detr"]).transform([], disabled_env)
    assert disabled_env["MEGATRON_OTEL_ENABLED"] == "0"
    mamba_only_opts = ["--cuda-graph-impl", "local", "--cuda-graph-modules", "mamba"]
    slurm_job.FlavorSelection(["detr"]).transform(mamba_only_opts, {})
    assert mamba_only_opts == [
        "--attention-backend",
        "unfused",
        "--moe-router-dtype",
        "fp64",
        "--deterministic-mode",
    ]
    source = (REPO_ROOT / "ci/scripts/slurm_job.py").read_text()
    assert "ProfileOptionPolicy" not in source
    assert "profile_policies" not in source


def test_valgrind_keeps_training_nan_aborts_enabled() -> None:
    slurm_job = _load_slurm_job()
    opts = [
        "--check-for-nan-in-loss-and-grad",
        "--cuda-graph-impl",
        "local",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "moe",
    ]

    env = {"NATIVE_CKPT_OTEL_FILE": "/tmp/otel.jsonl"}
    slurm_job.FlavorSelection(["vg"]).transform(opts, env)

    assert "--check-for-nan-in-loss-and-grad" in opts
    assert "--no-check-for-nan-in-loss-and-grad" not in opts
    assert opts[-2:] == ["--rerun-mode", "disabled"]
    assert {
        name: env[name]
        for name in slurm_job._VALGRIND_NATIVE_ENV
    } == slurm_job._VALGRIND_NATIVE_ENV
    assert env["NATIVE_CKPT_SERVER_AUTO_SHUTDOWN_TIMEOUT_SECONDS"] == "0"
    assert "NATIVE_CKPT_OTEL_FILE" not in env
    assert env["MEGATRON_OTEL_ENABLED"] == "0"


def test_nonzero_rank_log_is_retained_without_mirroring_stdout(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    slurm_job = _load_slurm_job()
    log = tmp_path / "rank1.log"

    assert slurm_job._tee_rc(
        [sys.executable, "-c", "print('rank-specific warning')"],
        log,
        mirror_stdout=False,
    ) == 0

    assert capfd.readouterr().out == ""
    assert log.read_text() == "rank-specific warning\n"


def test_ultra3_profiles_share_reference_architecture() -> None:
    slurm_job = _load_slurm_job()
    bf16 = slurm_job._model_config("ultra3bf16")
    fp8 = slurm_job._model_config("ultra3fp8")
    precision = {
        "fp8-format": "e4m3",
        "fp8-recipe": "mxfp8",
        "fp8-param-gather": True,
        "reuse-grad-buf-for-mxfp8-param-ag": True,
        "moe-router-padding-for-quantization": True,
    }
    expected = {
        "num-layers": 60,
        "hidden-size": 8192,
        "ffn-hidden-size": 5120,
        "mamba-num-heads": 256,
        "num-attention-heads": 64,
        "num-query-groups": 2,
        "num-experts": 512,
        "moe-latent-size": 2048,
        "moe-shared-expert-intermediate-size": 10240,
        "tensor-model-parallel-size": 8,
        "expert-model-parallel-size": 64,
        "micro-batch-size": 1,
        "ckpt-fully-parallel-save-process-group": "ep_dp",
        "ckpt-fully-parallel-load-process-group": "ep_dp",
        "use-precision-aware-optimizer": True,
        "main-grads-dtype": "bf16",
        "main-params-dtype": "fp32",
        "exp-avg-dtype": "fp32",
        "exp-avg-sq-dtype": "fp32",
        "grad-reduce-in-bf16": True,
    }

    assert bf16.pop("async-ckpt-use-cpu-shm") is True
    assert fp8["job_time_in_mins"]["srun"].pop("detr") == 180  # noqa: PLR2004
    assert bf16["job_time_in_mins"]["srun"].pop("detr") == 180  # noqa: PLR2004
    assert {key: fp8.pop(key) for key in precision} == precision
    assert fp8 == bf16
    assert {key: bf16[key] for key in expected} == expected
    main, *mtp = bf16["hybrid-layer-pattern"].split("/")
    assert len(main) == 60  # noqa: PLR2004
    assert mtp == ["*E", "*E"]


def test_nano4_matches_the_30b_gdp_reference_architecture() -> None:
    slurm_job = _load_slurm_job()
    nano4 = slurm_job._model_config("nano4")
    expected = {
        "spec": [
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "gated_delta_product_stack_spec",
        ],
        "hidden-size": 2688,
        "ffn-hidden-size": 1856,
        "num-attention-heads": 32,
        "num-query-groups": 2,
        "mamba-num-heads": 16,
        "mamba-num-groups": 16,
        "mamba-head-dim": 64,
        "mamba-state-dim": 128,
        "num-experts": 512,
        "moe-latent-size": 672,
        "moe-router-topk": 22,
        "moe-ffn-hidden-size": 1856,
        "moe-shared-expert-intermediate-size": 3712,
        "tensor-model-parallel-size": 2,
        "expert-model-parallel-size": 16,
        "pipeline-model-parallel-size": 1,
        "global-batch-size": 128,
        "optimizer": "dist_muon",
        "muon-num-ns-steps": 16,
        "muon-coefficient-type": "polar_express",
        "muon-scalar-optimizer": "lion",
        "seq-length": 8192,
        "max-position-embeddings": 8192,
        "vocab-size": 204800,
        "padded-vocab-size": 204800,
        "mtp-use-repeated-layer": True,
        "mtp-loss-scaling-factor": "0.1",
    }

    assert {key: nano4[key] for key in expected} == expected
    assert nano4["first-last-layers-bf16"] is False
    assert nano4["moe-permute-fusion"] is False
    assert nano4["no-gradient-accumulation-fusion"] is False
    assert nano4["cuda-graph-impl"] == "local"
    assert nano4["cuda-graph-modules"] == ["mamba", "attn"]
    assert nano4["sequence-parallel"] is True
    assert nano4["fine-grained-activation-offloading"] is True
    assert nano4["offload-modules"] == "moe_act"
    assert nano4["recompute-granularity"] == "selective"
    assert nano4["recompute-modules"] == "moe_act"
    optimizer_precision = {
        "use-precision-aware-optimizer",
        "main-grads-dtype",
        "grad-reduce-in-bf16",
    }
    assert optimizer_precision.isdisjoint(nano4)
    assert nano4["mock-data"] is True
    main, *mtp = nano4["hybrid-layer-pattern"].split("/")
    assert len(main) == 52  # noqa: PLR2004
    assert mtp == ["*E", "*E"]


def test_gpt_moe_covers_prepended_expert_axes_and_fp8_param_gather(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    slurm_job = _load_slurm_job()
    gpt_moe = slurm_job._model_config("gpt_moe")

    assert gpt_moe["ci"] == {"pretrain-entry": "gpt"}
    assert gpt_moe["num-layers"] == 2  # noqa: PLR2004
    assert gpt_moe["num-experts"] == 8  # noqa: PLR2004
    assert gpt_moe["moe-layer-freq"] == 1
    assert gpt_moe["moe-grouped-gemm"] is True
    assert gpt_moe["tensor-model-parallel-size"] == 2  # noqa: PLR2004
    assert gpt_moe["expert-model-parallel-size"] == 2  # noqa: PLR2004
    assert gpt_moe["expert-tensor-parallel-size"] == 2  # noqa: PLR2004
    assert gpt_moe["kv-channels"] == 32  # noqa: PLR2004
    assert gpt_moe["is-hybrid-model"] is False
    assert gpt_moe["spec"] is None
    assert gpt_moe["fp8-format"] == "hybrid"
    assert gpt_moe["fp8-recipe"] == "tensorwise"
    assert gpt_moe["fp8-param-gather"] is True

    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="gpt_moe", nodes=1, gpus_per_node=4, flavors=["perf", "native"])
    slurm_job._configure_run(args)
    command, _env = slurm_job._phase_command("gpt_moe_1x4", args, "native", 2)

    entry = command.index(slurm_job._PRETRAIN_ENTRY_CMD)
    assert command[entry + 1 : entry + 3] == ["2", "gpt"]
    assert "--ci" not in command
    assert "--is-hybrid-model" not in command
    assert "--spec" not in command
    assert "--fp8-param-gather" in command
    assert command[command.index("--moe-layer-freq") + 1] == "1"


def test_pico_trains_with_the_mixed_precision_optimizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slurm_job = _load_slurm_job()
    pico = slurm_job._model_config("pico")

    assert pico["use-distributed-optimizer"] is False
    assert pico["overlap-param-gather"] is False
    for path in sorted(slurm_job.CONFIGS_DIR.glob("*.yaml")):
        if path.stem not in ("base", "pico"):
            assert slurm_job._model_config(path.stem)["use-distributed-optimizer"] is True, path.stem

    _clear_run_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n0")
    monkeypatch.setenv("MASTER_ADDR", "n0")
    args = _args(tmp_path, config="pico", nodes=1, gpus_per_node=4, flavors=["perf", "native"])
    slurm_job._configure_run(args)
    command, _env = slurm_job._phase_command("pico_1x4", args, "native", 1)

    assert "--use-distributed-optimizer" not in command
    assert "--overlap-param-gather" not in command
    assert "--bf16" in command
    assert "--overlap-grad-reduce" in command
    assert command[command.index("--num-experts") + 1] == "32"


def test_super3_profiles_share_reference_architecture() -> None:
    slurm_job = _load_slurm_job()
    common = slurm_job._model_config("super3")
    bf16 = slurm_job._model_config("super3bf16")
    fp4 = slurm_job._model_config("super3fp4")
    precision = {
        "fp4-format": "e2m1",
        "fp4-recipe": "nvfp4",
    }
    optimizer_precision = {
        "use-precision-aware-optimizer": True,
        "main-grads-dtype": "bf16",
        "grad-reduce-in-bf16": True,
    }
    expected = {
        "num-layers": 88,
        "hidden-size": 4096,
        "ffn-hidden-size": 2688,
        "mamba-num-heads": 128,
        "num-attention-heads": 32,
        "num-query-groups": 2,
        "num-experts": 512,
        "moe-latent-size": 1024,
        "moe-shared-expert-intermediate-size": 5376,
        "expert-model-parallel-size": 64,
        "micro-batch-size": 1,
        "num-layers-at-end-in-bf16": 14,
        "async-ckpt-use-cpu-shm": True,
    }

    assert {key: fp4.pop(key) for key in precision} == precision
    assert {key: fp4.pop(key) for key in optimizer_precision} == optimizer_precision
    assert {key: bf16.pop(key) for key in optimizer_precision} == optimizer_precision
    # Both profiles need TP4 to halve the native fused-loss temporary while
    # retaining identical model topology.
    tp_key = "tensor-model-parallel-size"
    assert common.pop(tp_key) == bf16.pop(tp_key) == fp4.pop(tp_key) == 4  # noqa: PLR2004
    # The BF16 profile also recomputes its MoE activations. Attention-only graph
    # capture supports it, so both deterministic and measured runs retain it.
    recompute = {"recompute-granularity": "selective", "recompute-modules": "moe"}
    assert {key: bf16.pop(key) for key in recompute} == recompute
    graph_keys = ("cuda-graph-impl", "cuda-graph-modules")
    common_graphs = {key: common.pop(key) for key in graph_keys}
    bf16_graphs = {key: bf16.pop(key) for key in graph_keys}
    fp4_graphs = {key: fp4.pop(key) for key in graph_keys}
    assert common_graphs == {
        "cuda-graph-impl": "local",
        "cuda-graph-modules": ["mamba", "attn", "moe_router"],
    }
    assert bf16_graphs == fp4_graphs == {
        "cuda-graph-impl": "local",
        "cuda-graph-modules": ["attn"],
    }
    assert common == bf16
    assert fp4 == bf16
    assert {key: bf16[key] for key in expected} == expected
    assert len(bf16["hybrid-override-pattern"]) == 88  # noqa: PLR2004


def test_all_model_configs_use_fp32_optimizer_states() -> None:
    slurm_job = _load_slurm_job()

    for path in sorted(slurm_job.CONFIGS_DIR.glob("*.yaml")):
        if path.stem == "base":
            continue
        config = slurm_job._model_config(path.stem)
        assert config["main-params-dtype"] == "fp32", path.stem
        assert config["exp-avg-dtype"] == "fp32", path.stem
        assert config["exp-avg-sq-dtype"] == "fp32", path.stem


def test_cuda_graph_selection_is_profile_scoped() -> None:
    slurm_job = _load_slurm_job()

    pico = slurm_job._model_config("pico")
    assert pico["cuda-graph-impl"] == "local"
    assert pico["cuda-graph-modules"] == ["mamba", "attn", "moe_router"]

    ultra3bf16 = slurm_job._model_config("ultra3bf16")
    assert ultra3bf16["cuda-graph-impl"] == "local"
    assert ultra3bf16["cuda-graph-modules"] == ["mamba", "attn", "moe_router"]

    nano3 = slurm_job._model_config("nano3")
    assert nano3["cuda-graph-impl"] == "local"
    assert nano3["moe-router-force-load-balancing"] is True
    assert nano3["cuda-graph-modules"] == ["mamba", "attn"]

    nano4 = slurm_job._model_config("nano4")
    assert nano4["cuda-graph-impl"] == "local"
    assert nano4["cuda-graph-modules"] == ["mamba", "attn"]

    ultra3fp8 = slurm_job._model_config("ultra3fp8")
    assert ultra3fp8["cuda-graph-impl"] == "local"
    assert ultra3fp8["cuda-graph-modules"] == ["mamba", "attn", "moe_router"]


def test_standalone_determinism_script_invokes_megatron_directly() -> None:
    script = (REPO_ROOT / "scripts/sbatch/determinism_checkpoints.sbatch").read_text()

    assert "pretrain_mamba.py" in script
    assert "MEGATRON_ROOT" in script
    assert "ci/scripts/slurm_job.py" not in script
    assert "pretrain-native" not in script
    assert "NVRX_CHECKPOINTING_ROOT" not in script
    assert "--ckpt-provider" not in script
    assert "--async-strategy nvrx" not in script
    assert "CUDA_GRAPH_MODULES:=mamba attn moe_router" in script


def test_ultra3bf16_uses_cpu_shared_memory_for_async_checkpoints() -> None:
    slurm_job = _load_slurm_job()

    config = slurm_job._model_config("ultra3bf16")
    assert config["async-ckpt-use-cpu-shm"] is True
    flavors = slurm_job.FlavorSelection(["perf"])
    assert slurm_job._time_budget("ultra3bf16", flavors, "srun", 30) == 120  # noqa: PLR2004
