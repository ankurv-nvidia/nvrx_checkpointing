# ruff: noqa: ANN401, ARG001, C901, INP001, PLR0913, PLR0915, S101, SLF001

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_native_save_retains_only_common_checkpoint_state() -> None:
    tree = ast.parse((REPO_ROOT / "tools.py").read_text())
    retained = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "save_checkpoint"
    }
    assert retained == {"common"}


def _load_integration(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, SimpleNamespace]:
    calls = SimpleNamespace(
        empty_cache=0,
        free_overlap=0,
        native_saves=[],
        reference_saves=[],
        train=0,
        validate=0,
        warmup=0,
        gc_collect=0,
        parsed_provider=None,
        native_async=[],
        native_readers=[],
        reference_async=[],
        reference_readers=[],
    )

    fake_package = ModuleType("nvrx_checkpointing")
    fake_package.__path__ = [str(REPO_ROOT)]
    fake_tools = ModuleType("nvrx_checkpointing.tools")

    def native_save_checkpoint(*args: Any, **kwargs: Any) -> str:
        calls.native_saves.append((args, kwargs))
        return "native"

    def warmup() -> None:
        calls.warmup += 1

    fake_tools.save_checkpoint = native_save_checkpoint
    fake_tools.warmup = warmup
    fake_package.tools = fake_tools
    fake_native_async = ModuleType("nvrx_checkpointing.native_async")

    def record_native_async(name: str, *args: Any) -> str:
        calls.native_async.append((name, args))
        return f"native-{name}"

    fake_native_async.init_persistent_async_worker = lambda *args: record_native_async("init", *args)
    fake_native_async.schedule_async_save = lambda *args: record_native_async("schedule", *args)
    fake_native_async.maybe_finalize_async_save = lambda *args: record_native_async("finalize", *args)
    fake_native_async.is_empty_async_queue = lambda: record_native_async("is-empty") == "not-empty"
    fake_native_async.reset_persistent_async_worker = lambda *args: record_native_async("reset", *args)
    fake_native_async.get_filesystem_reader = lambda path, *, cache_metadata: (
        calls.native_readers.append((path, cache_metadata)) or "native-reader"
    )
    fake_package.native_async = fake_native_async

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.__path__ = []
    dist_checkpointing_package = ModuleType("megatron.core.dist_checkpointing")
    dist_checkpointing_package.__path__ = []
    strategies_package = ModuleType("megatron.core.dist_checkpointing.strategies")
    strategies_package.__path__ = []
    torch_strategy_module = ModuleType("megatron.core.dist_checkpointing.strategies.torch")

    def get_filesystem_reader(
        checkpoint_dir: str,
        cache_metadata: bool = False,
        async_strategy: str = "mcore",
    ) -> str:
        calls.reference_readers.append((checkpoint_dir, cache_metadata, async_strategy))
        return "reference-reader"

    torch_strategy_module._get_filesystem_reader = get_filesystem_reader
    strategies_package.torch = torch_strategy_module
    dist_checkpointing_package.strategies = strategies_package
    megatron_core.dist_checkpointing = dist_checkpointing_package
    optimizer_package = ModuleType("megatron.core.optimizer")
    optimizer_package.__path__ = []
    distrib_module = ModuleType("megatron.core.optimizer.distrib_optimizer")

    class FakeDistributedOptimizer:
        @classmethod
        def _build_gbuf_range_map(cls, buffer: Any) -> dict[Any, list[Any]]:
            return {(buffer.param_dtype, buffer.grad_dtype): []}

        def __init__(self, buffers: list[Any]) -> None:
            self.buffers = buffers
            self.gbuf_ranges = [self._build_gbuf_range_map(buffer) for buffer in buffers]
            self.per_bucket_numel = [{(buffer.param_dtype, buffer.grad_dtype): [10]} for buffer in buffers]
            self.per_bucket_numel_unpadded = [{(buffer.param_dtype, buffer.grad_dtype): [9]} for buffer in buffers]

        def split_state_dict_if_needed(self, state_dict: dict[Any, Any]) -> None:
            assert all(isinstance(next(iter(entry)), tuple) for entry in self.gbuf_ranges)
            dtype_map = state_dict[0]
            assert isinstance(next(iter(dtype_map)), tuple)

    distrib_module.DistributedOptimizer = FakeDistributedOptimizer
    fake_torch = SimpleNamespace(uint8="torch.uint8", cuda=SimpleNamespace())
    distrib_module.torch = fake_torch
    optimizer_package.distrib_optimizer = distrib_module
    megatron_core.optimizer = optimizer_package

    training_package = ModuleType("megatron.training")
    training_package.__path__ = []
    arguments_module = ModuleType("megatron.training.arguments")
    async_utils_module = ModuleType("megatron.training.async_utils")
    checkpointing_module = ModuleType("megatron.training.checkpointing")
    initialize_module = ModuleType("megatron.training.initialize")
    restart_module = ModuleType("megatron.training.inprocess_restart")
    training_module = ModuleType("megatron.training.training")
    current_args = SimpleNamespace(ckpt_provider="native", async_save=True, async_strategy="nvrx")

    def parse_args(extra_args_provider: Any = None, ignore_unknown_args: bool = False) -> Any:
        parser = argparse.ArgumentParser()
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parsed = parser.parse_args([])
        calls.parsed_provider = parsed.ckpt_provider
        return parsed

    def validate_args(args: Any, defaults: dict[str, Any]) -> Any:
        calls.validate += 1
        if not args.async_save:
            args.async_strategy = "mcore"
        return args

    arguments_module.parse_args = parse_args
    arguments_module.validate_args = validate_args

    def record_reference_async(name: str, *args: Any) -> str:
        calls.reference_async.append((name, args))
        return f"reference-{name}"

    async_utils_module.init_persistent_async_worker = lambda *args: record_reference_async("init", *args)
    async_utils_module.schedule_async_save = lambda *args: record_reference_async("schedule", *args)
    async_utils_module.maybe_finalize_async_save = lambda *args: record_reference_async("finalize", *args)
    async_utils_module.is_empty_async_queue = lambda: record_reference_async("is-empty") == "empty"
    async_utils_module.reset_persistent_async_worker = lambda *args: record_reference_async("reset", *args)
    checkpointing_module.schedule_async_save = async_utils_module.schedule_async_save
    checkpointing_module.is_empty_async_queue = async_utils_module.is_empty_async_queue
    initialize_module.init_persistent_async_worker = async_utils_module.init_persistent_async_worker
    restart_module.reset_persistent_async_worker = async_utils_module.reset_persistent_async_worker

    def empty_cache() -> None:
        calls.empty_cache += 1

    training_module.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty_cache))
    training_module.gc = SimpleNamespace(collect=lambda: setattr(calls, "gc_collect", calls.gc_collect + 1))
    training_module.get_args = lambda: current_args

    def reference_save(*args: Any, **kwargs: Any) -> str:
        calls.reference_saves.append((args, kwargs))
        return "reference"

    training_module.save_checkpoint = reference_save
    training_module.maybe_finalize_async_save = async_utils_module.maybe_finalize_async_save

    def save_checkpoint_and_time(
        iteration: int,
        model: list[Any],
        optimizer: Any,
        scheduler: Any,
        flops: int,
        checkpointing_context: Any,
        **kwargs: Any,
    ) -> Any:
        for chunk in model:
            if hasattr(chunk, "free_overlap_buffers"):
                chunk.free_overlap_buffers()
        training_module.torch.cuda.empty_cache()
        return training_module.save_checkpoint(
            iteration,
            model,
            optimizer,
            scheduler,
            flops,
            checkpointing_context,
            **kwargs,
        )

    training_module.save_checkpoint_and_time = save_checkpoint_and_time

    def train(*args: Any, **kwargs: Any) -> str:
        calls.train += 1
        return "trained"

    training_module.train = train
    training_package.arguments = arguments_module
    training_package.async_utils = async_utils_module
    training_package.checkpointing = checkpointing_module
    training_package.initialize = initialize_module
    training_package.inprocess_restart = restart_module
    training_package.training = training_module
    megatron.training = training_package

    modules = {
        "nvrx_checkpointing": fake_package,
        "nvrx_checkpointing.native_async": fake_native_async,
        "nvrx_checkpointing.tools": fake_tools,
        "megatron": megatron,
        "megatron.core": megatron_core,
        "megatron.core.dist_checkpointing": dist_checkpointing_package,
        "megatron.core.dist_checkpointing.strategies": strategies_package,
        "megatron.core.dist_checkpointing.strategies.torch": torch_strategy_module,
        "megatron.core.optimizer": optimizer_package,
        "megatron.core.optimizer.distrib_optimizer": distrib_module,
        "megatron.training": training_package,
        "megatron.training.arguments": arguments_module,
        "megatron.training.async_utils": async_utils_module,
        "megatron.training.checkpointing": checkpointing_module,
        "megatron.training.initialize": initialize_module,
        "megatron.training.inprocess_restart": restart_module,
        "megatron.training.training": training_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "nvrx_checkpointing.megatron_integration",
        REPO_ROOT / "megatron_integration.py",
    )
    assert spec is not None
    assert spec.loader is not None
    integration = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, integration)
    spec.loader.exec_module(integration)
    integration.install()

    calls.args = current_args
    calls.arguments_module = arguments_module
    calls.async_utils_module = async_utils_module
    calls.checkpointing_module = checkpointing_module
    calls.distrib_module = distrib_module
    calls.initialize_module = initialize_module
    calls.restart_module = restart_module
    calls.training_module = training_module
    calls.torch_strategy_module = torch_strategy_module
    return integration, calls


def test_argument_provider_composes_and_defaults_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    integration, calls = _load_integration(monkeypatch)
    parser = argparse.ArgumentParser()

    def add_existing_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--existing")
        return parser

    parser = integration.compose_extra_args_provider(add_existing_argument)(parser)
    args = parser.parse_args(["--existing", "yes", "--ckpt-provider", "nvrx"])

    assert args.existing == "yes"
    assert args.ckpt_provider == "nvrx"
    assert integration.add_arguments(parser) is parser
    parsed = calls.arguments_module.parse_args()
    assert parsed.ckpt_provider == "native"
    assert calls.parsed_provider == "native"


def test_module_launcher_installs_before_running_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    integration, _ = _load_integration(monkeypatch)
    entrypoint = tmp_path / "pretrain_gpt.py"
    entrypoint.write_text("# fake entrypoint\n")
    events: list[Any] = []
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(integration, "install", lambda: events.append("install"))
    monkeypatch.setattr(integration.runpy, "run_path", lambda path, run_name: events.append((path, run_name)))

    integration.main([str(entrypoint), "--save", "/checkpoints"])

    assert events == ["install", (str(entrypoint), "__main__")]
    assert sys.argv == [str(entrypoint), "--save", "/checkpoints"]
    assert sys.path[0] == str(tmp_path)


def test_adapter_and_resolver_pin_same_public_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    integration, _ = _load_integration(monkeypatch)
    resolver = (REPO_ROOT / "scripts/megatron_root.sh").read_text()

    assert f"COMMIT_HASH=${{MEGATRON_COMMIT:-{integration.SUPPORTED_MEGATRON_COMMIT}}}" in resolver
    assert "https://github.com/NVIDIA/Megatron-LM.git" in resolver
    assert "gitlab-master.nvidia.com/ankurv/megatron-lm" not in resolver
    assert "git -c protocol.version=1 clone" in resolver
    assert "-c protocol.version=1" in resolver
    assert '-c remote.neworigin.url="${MEGATRON_REPO_URL}"' in resolver
    assert 'diff --quiet "${COMMIT_HASH}" -- .' in resolver
    assert 'checkout --quiet "${COMMIT_HASH}" -- .' in resolver
    assert 'local staging="${MEGATRON_ROOT}.cloning-$$"' in resolver
    assert "git_retry git -c protocol.version=1 clone" in resolver
    assert "git_retry git -C" in resolver


def test_native_provider_preserves_configured_async_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Native lifecycle hooks are patched separately, so argument validation must
    # not select either external checkpoint implementation.
    _, calls = _load_integration(monkeypatch)
    calls.arguments_module.validate_args(calls.args, {})
    assert calls.validate == 1
    assert calls.args.async_save is True
    assert calls.args.async_strategy == "nvrx"


def test_reference_provider_preserves_nvrx_async_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.args.ckpt_provider = "nvrx"
    calls.arguments_module.validate_args(calls.args, {})
    assert calls.validate == 1
    assert calls.args.async_save is True
    assert calls.args.async_strategy == "nvrx"


def test_native_provider_redirects_async_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)

    calls.initialize_module.init_persistent_async_worker(3, "forkserver")
    calls.async_utils_module.schedule_async_save("request")
    calls.training_module.maybe_finalize_async_save(True, True)
    calls.checkpointing_module.is_empty_async_queue()
    calls.restart_module.reset_persistent_async_worker("nvrx")

    assert calls.native_async == [
        ("init", (3, "forkserver")),
        ("schedule", ("request",)),
        ("finalize", (True, True)),
        ("is-empty", ()),
        ("reset", ("nvrx",)),
    ]
    assert calls.reference_async == []


def test_reference_provider_keeps_original_async_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.args.ckpt_provider = "nvrx"

    calls.initialize_module.init_persistent_async_worker(3, "forkserver")
    calls.async_utils_module.schedule_async_save("request")
    calls.training_module.maybe_finalize_async_save(True, True)
    calls.checkpointing_module.is_empty_async_queue()
    calls.restart_module.reset_persistent_async_worker("nvrx")

    assert calls.native_async == []
    assert calls.reference_async == [
        ("init", (3, "forkserver")),
        ("schedule", ("request",)),
        ("finalize", (True, True)),
        ("is-empty", ()),
        ("reset", ("nvrx",)),
    ]


def test_native_provider_uses_repo_owned_load_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)

    get_reader = calls.torch_strategy_module._get_filesystem_reader
    result = get_reader("/checkpoint", cache_metadata=True, async_strategy="nvrx")

    assert result == "native-reader"
    assert calls.native_readers == [("/checkpoint", True)]
    assert calls.reference_readers == []
    assert calls.args.async_strategy == "nvrx"


def test_reference_provider_keeps_original_load_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.args.ckpt_provider = "nvrx"

    get_reader = calls.torch_strategy_module._get_filesystem_reader
    result = get_reader("/checkpoint", cache_metadata=True, async_strategy="nvrx")

    assert result == "reference-reader"
    assert calls.native_readers == []
    assert calls.reference_readers == [("/checkpoint", True, "nvrx")]


def test_optimizer_uses_legacy_string_dtype_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    buffer = SimpleNamespace(
        param_dtype="torch.bfloat16",
        grad_dtype="torch.float32",
        params=[SimpleNamespace(dtype="torch.bfloat16")],
    )
    optimizer = calls.distrib_module.DistributedOptimizer([buffer])
    expected = "dtype_param_torch:bfloat16_grad_torch:float32"
    assert list(optimizer.gbuf_ranges[0]) == [expected]
    assert list(optimizer.per_bucket_numel[0]) == [expected]
    assert list(optimizer.per_bucket_numel_unpadded[0]) == [expected]
    state_dict = {0: {expected: {}}, "buckets_coalesced": False}
    optimizer.split_state_dict_if_needed(state_dict)
    assert list(state_dict[0]) == [expected]
    assert list(optimizer.gbuf_ranges[0]) == [expected]


def test_training_dispatch_warmup_and_upstream_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)

    class ModelChunk:
        def free_overlap_buffers(self) -> None:
            calls.free_overlap += 1

    model = [ModelChunk()]
    result = calls.training_module.save_checkpoint_and_time(
        10,
        model,
        "optimizer",
        "scheduler",
        123,
        "context",
        train_data_iterator="iterator",
    )
    assert result == "native"
    assert calls.empty_cache == 1
    assert calls.free_overlap == 1
    assert len(calls.native_saves) == 1
    assert calls.native_saves[0][1]["train_data_iterator"] == "iterator"
    assert not calls.reference_saves
    assert calls.training_module.train() == "trained"
    assert calls.training_module.train() == "trained"
    assert calls.warmup == 2  # noqa: PLR2004
    calls.args.ckpt_provider = "nvrx"
    assert calls.training_module.save_checkpoint(11, model, None, None, 0) == "reference"
    assert len(calls.reference_saves) == 1


def test_fp8_gc_stays_inside_checkpoint_save(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.args.fp8 = True
    observed_during_save: list[int] = []
    original_native_save = sys.modules["nvrx_checkpointing.tools"].save_checkpoint

    def native_save(*args: Any, **kwargs: Any) -> str:
        result = original_native_save(*args, **kwargs)
        observed_during_save.append(calls.gc_collect)
        return result

    sys.modules["nvrx_checkpointing.tools"].save_checkpoint = native_save
    calls.training_module.save_checkpoint_and_time(10, [], None, None, 0, None)
    assert observed_during_save == [0]
    assert calls.gc_collect == 1
    calls.training_module.gc.collect()
    assert calls.gc_collect == 2  # noqa: PLR2004
