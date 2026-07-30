# ruff: noqa: ANN401, ARG001, C901, INP001, PLR0913, PLR0915, S101

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.__path__ = []
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

    def empty_cache() -> None:
        calls.empty_cache += 1

    training_module.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty_cache))
    training_module.gc = SimpleNamespace(collect=lambda: setattr(calls, "gc_collect", calls.gc_collect + 1))
    training_module.get_args = lambda: current_args

    def reference_save(*args: Any, **kwargs: Any) -> str:
        calls.reference_saves.append((args, kwargs))
        return "reference"

    training_module.save_checkpoint = reference_save

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
    training_package.training = training_module
    megatron.training = training_package

    modules = {
        "nvrx_checkpointing": fake_package,
        "nvrx_checkpointing.tools": fake_tools,
        "megatron": megatron,
        "megatron.core": megatron_core,
        "megatron.core.optimizer": optimizer_package,
        "megatron.core.optimizer.distrib_optimizer": distrib_module,
        "megatron.training": training_package,
        "megatron.training.arguments": arguments_module,
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
    calls.distrib_module = distrib_module
    calls.training_module = training_module
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


def test_native_validation_disables_megatron_async_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.arguments_module.validate_args(calls.args, {})
    assert calls.validate == 1
    assert calls.args.async_save is False
    assert calls.args.async_strategy == "mcore"


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
