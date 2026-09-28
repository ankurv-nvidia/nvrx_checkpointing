# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: ANN401, ARG001, ARG005, C901, FBT003, INP001, PLR0913, PLR0915, PLR0917, S101, S102

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_native_save_retains_no_function_attribute_checkpoint_state() -> None:
    tree = ast.parse((REPO_ROOT / "megatron/save.py").read_text())
    retained = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "save_checkpoint"
    }
    assert retained == set()


def test_save_keeps_megatron_imports_at_module_scope() -> None:
    tree = ast.parse((REPO_ROOT / "megatron/save.py").read_text())
    imports = [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module]
    nested_imports = [
        node.module
        for function in (node for node in tree.body if isinstance(node, ast.FunctionDef))
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("megatron", "nemo", "opentelemetry"))
    ]

    assert "megatron.training.checkpointing" in imports
    assert nested_imports == []


def _load_provider_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "nvrx_checkpointing.megatron.provider",
        REPO_ROOT / "megatron/provider.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_optional_provider_args_are_composable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _load_provider_module(monkeypatch)
    parser = argparse.ArgumentParser()

    def add_existing_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--existing")
        return parser

    parser = provider.compose_extra_args_provider(add_existing_argument)(parser)
    args = parser.parse_args(["--existing", "yes", "--ckpt-provider", "nvrx"])
    assert args.existing == "yes"
    assert args.ckpt_provider == "nvrx"
    assert provider.add_arguments(parser) is parser
    assert provider.selected_provider([]) == "native"
    assert provider.selected_provider(["--ckpt-provider=nvrx"]) == "nvrx"
    assert provider.selected_provider(["--ckpt-provider", "native"]) == "native"


def test_save_checkpoint_captures_entry_time_before_implementation() -> None:
    tree = ast.parse((REPO_ROOT / "megatron/save.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint"]
    events = []

    @contextmanager
    def checkpoint_span(name: str, iteration: int) -> Any:
        del iteration
        events.append(("begin", name))
        yield
        events.append(("end", name))

    def save_impl(*args: Any, **kwargs: Any) -> None:
        del kwargs
        events.append(("impl", args[1]))

    namespace = {
        "_checkpoint_span": checkpoint_span,
        "_save_checkpoint_impl": save_impl,
        "SaveCheckpointArgs": SimpleNamespace,
        "diagnose_checkpoint_server_failure": lambda _error: None,
        "print_rank_0": lambda _message: None,
        "time": SimpleNamespace(time_ns=lambda: 123),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "save.py", "exec"), namespace)

    namespace["save_checkpoint"](SimpleNamespace(iteration=125))

    assert events == [
        ("begin", "nvrx.checkpoint.save"),
        ("impl", 123),
        ("end", "nvrx.checkpoint.save"),
    ]


def _load_integration(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, SimpleNamespace]:
    calls = SimpleNamespace(
        init=[],
        finalize=[],
        native_saves=[],
        native_loads=[],
        strategy_loads=[],
        public_loads=[],
        train=0,
        registry_finalized=0,
        legacy_only=False,
    )

    fake_package = ModuleType("nvrx_checkpointing")
    fake_package.__path__ = [str(REPO_ROOT)]
    fake_megatron_package = ModuleType("nvrx_checkpointing.megatron")
    fake_megatron_package.__path__ = [str(REPO_ROOT / "megatron")]
    fake_save = ModuleType("nvrx_checkpointing.megatron.save")
    fake_save.SaveCheckpointArgs = SimpleNamespace
    fake_save.save_checkpoint = lambda *args, **kwargs: calls.native_saves.append((args, kwargs)) or "native"

    class Root:
        def finalize_registry(self) -> None:
            calls.registry_finalized += 1

    root = Root()
    fake_save.register_state_dict = lambda state, iteration: calls.native_loads.append((state, iteration)) or root
    fake_save._wrap_load_checkpoint = lambda fn: fn  # noqa: SLF001

    fake_save.init_persistent_async_worker = lambda *args: calls.init.append(args)
    fake_save.maybe_finalize_async_save = lambda *args, **kwargs: calls.finalize.append((args, kwargs))
    fake_megatron_package.save = fake_save
    fake_package.megatron = fake_megatron_package

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.__path__ = []
    dist_checkpointing = ModuleType("megatron.core.dist_checkpointing")
    dist_checkpointing.__path__ = []
    strategies = ModuleType("megatron.core.dist_checkpointing.strategies")
    strategies.__path__ = []
    torch_strategy = ModuleType("megatron.core.dist_checkpointing.strategies.torch")

    class TorchDistLoadShardedStrategy:
        def load(self, state: dict[str, Any], checkpoint_dir: str, async_strategy: str) -> Any:
            calls.strategy_loads.append((state, checkpoint_dir, async_strategy))
            return {"loaded": True}

    torch_strategy.TorchDistLoadShardedStrategy = TorchDistLoadShardedStrategy
    strategies.torch = torch_strategy

    def _leaf_key(value: Any) -> str:
        if isinstance(value, dict):
            for child in value.values():
                if key := _leaf_key(child):
                    return key
        if isinstance(value, list):
            for child in value:
                if key := _leaf_key(child):
                    return key
        return getattr(value, "key", "")

    def public_load(
        state: dict[str, Any],
        checkpoint_dir: str,
        sharded_strategy: Any = None,
        **kwargs: Any,
    ) -> Any:
        calls.public_loads.append(state)
        key = _leaf_key(state.get("optimizer"))
        if calls.legacy_only:
            if ".dtype_(torch." in key:
                raise KeyError(key)
            if ".dtype_dtype_param_" in key:
                legacy = next(iter(state["optimizer"][0]))
                return {"optimizer": {legacy: "loaded"}}
        return sharded_strategy.load(state, checkpoint_dir, "nvrx")

    dist_checkpointing.load = public_load
    dist_checkpointing.strategies = strategies
    megatron_core.dist_checkpointing = dist_checkpointing
    megatron.core = megatron_core

    training_package = ModuleType("megatron.training")
    training_package.__path__ = []
    initialize_module = ModuleType("megatron.training.initialize")
    restart_module = ModuleType("megatron.training.inprocess_restart")
    training_module = ModuleType("megatron.training.training")

    initialize_module.init_persistent_async_worker = lambda *args: "reference-init"
    restart_module.reset_persistent_async_worker = lambda *args: "reference-reset"
    training_module.maybe_finalize_async_save = lambda *args, **kwargs: "reference-finalize"

    def reference_save(
        iteration: int,
        model: Any,
        optimizer: Any,
        opt_param_scheduler: Any,
        num_floating_point_operations_so_far: int,
        checkpointing_context: Any = None,
        train_data_iterator: Any = None,
        preprocess_common_state_dict_fn: Any = None,
        tp_group: Any = None,
        pp_group: Any = None,
        dp_cp_group: Any = None,
        dp_group: Any = None,
        dp_gtp_remat_group: Any = None,
        expt_dp_group: Any = None,
        rng_state_key_prefix: str = "",
    ) -> str:
        return "reference"

    def train(*args: Any, **kwargs: Any) -> str:
        calls.train += 1
        return "trained"

    def reference_load_checkpoint(
        state: dict[str, Any],
        optimizer: Any,
        opt_param_scheduler: Any,
        *,
        tp_group: Any = None,
        pp_group: Any = None,
        dp_cp_group: Any = None,
        dp_group: Any = None,
        rng_state_key_prefix: str = "",
    ) -> Any:
        del optimizer, opt_param_scheduler, tp_group, pp_group, dp_cp_group, dp_group, rng_state_key_prefix
        return dist_checkpointing.load(state, "/checkpoint")

    training_module.load_checkpoint = reference_load_checkpoint
    training_module.save_checkpoint = reference_save
    training_module.train = train
    training_package.initialize = initialize_module
    training_package.inprocess_restart = restart_module
    training_package.training = training_module
    megatron.training = training_package

    modules = {
        "nvrx_checkpointing": fake_package,
        "nvrx_checkpointing.megatron": fake_megatron_package,
        "nvrx_checkpointing.megatron.save": fake_save,
        "megatron": megatron,
        "megatron.core": megatron_core,
        "megatron.core.dist_checkpointing": dist_checkpointing,
        "megatron.core.dist_checkpointing.strategies": strategies,
        "megatron.core.dist_checkpointing.strategies.torch": torch_strategy,
        "megatron.training": training_package,
        "megatron.training.initialize": initialize_module,
        "megatron.training.inprocess_restart": restart_module,
        "megatron.training.training": training_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "nvrx_checkpointing.megatron.integration",
        REPO_ROOT / "megatron/integration.py",
    )
    assert spec is not None
    assert spec.loader is not None
    integration = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, integration)
    spec.loader.exec_module(integration)
    integration.install()

    calls.dist_checkpointing = dist_checkpointing
    calls.initialize_module = initialize_module
    calls.restart_module = restart_module
    calls.training_module = training_module
    return integration, calls


def test_install_patches_only_native_call_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.initialize_module.init_persistent_async_worker(3, "forkserver")
    calls.training_module.maybe_finalize_async_save(True, terminate=True)

    assert calls.init == [(3, "forkserver")]
    assert calls.finalize == [((True,), {"terminate": True})]
    assert calls.restart_module.reset_persistent_async_worker("nvrx") == "reference-reset"
    integration_source = (REPO_ROOT / "megatron/integration.py").read_text()
    assert "megatron.training.arguments" not in integration_source
    assert "megatron.core.optimizer.distrib_optimizer" not in integration_source


def test_training_save_forwards_constructed_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    result = calls.training_module.save_checkpoint(
        10,
        "model",
        "optimizer",
        "scheduler",
        123,
        {"context": True},
        train_data_iterator="iterator",
        tp_group="tp",
        pp_group="pp",
        dp_cp_group="dp-cp",
        dp_group="dp",
        dp_gtp_remat_group="dp-gtp-remat",
        expt_dp_group="edp",
        rng_state_key_prefix="grid.",
    )

    assert result == "native"
    assert len(calls.native_saves) == 1
    (save_args,), kwargs = calls.native_saves[0]
    assert kwargs == {}
    assert vars(save_args) == {
        "iteration": 10,
        "model": "model",
        "optimizer": "optimizer",
        "opt_param_scheduler": "scheduler",
        "num_floating_point_operations_so_far": 123,
        "train_data_iterator": "iterator",
        "tp_group": "tp",
        "pp_group": "pp",
        "dp_cp_group": "dp-cp",
        "dp_group": "dp",
        "dp_gtp_remat_group": "dp-gtp-remat",
        "rng_state_key_prefix": "grid.",
    }


def test_train_does_not_register_state_without_a_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    assert calls.training_module.train() == "trained"
    assert calls.native_loads == []


def test_native_load_uses_mcore_and_registers_constructed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    state = {"model": "tensor"}
    result = calls.dist_checkpointing.load(state, "/checkpoint")

    assert result == {"loaded": True}
    assert calls.strategy_loads == [(state, "/checkpoint", "mcore")]
    assert calls.native_loads == [(state, 0)]
    assert calls.registry_finalized == 1


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


def test_old_string_optimizer_keys_load_without_optimizer_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _load_integration(monkeypatch)
    calls.legacy_only = True
    dtype_pair = (_DType("torch.bfloat16"), _DType("torch.float32"))
    leaf = SimpleNamespace(
        key=("optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.exp_avg"),
    )
    state = {"optimizer": [{dtype_pair: {"exp_avg": leaf}}]}

    result = calls.dist_checkpointing.load(state, "/old-checkpoint")

    legacy = "dtype_param_torch:bfloat16_grad_torch:float32"
    assert result == {"optimizer": {dtype_pair: "loaded"}}
    assert len(calls.public_loads) == 2  # noqa: PLR2004
    assert next(iter(calls.public_loads[0]["optimizer"][0])) == dtype_pair
    assert next(iter(calls.public_loads[1]["optimizer"][0])) == legacy
    legacy_leaf = calls.public_loads[1]["optimizer"][0][legacy]["exp_avg"]
    assert f".dtype_{legacy}." in legacy_leaf.key
    assert leaf.key.endswith(".dtype_(torch.bfloat16, torch.float32).bucket_idx_0.exp_avg")
    assert calls.native_loads == [(state, 0)]


def test_legacy_optimizer_adapter_restores_strict_load_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_package = ModuleType("nvrx_checkpointing")
    fake_package.__path__ = [str(REPO_ROOT)]
    fake_megatron_package = ModuleType("nvrx_checkpointing.megatron")
    fake_megatron_package.__path__ = [str(REPO_ROOT / "megatron")]
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing", fake_package)
    monkeypatch.setitem(sys.modules, "nvrx_checkpointing.megatron", fake_megatron_package)
    spec = importlib.util.spec_from_file_location(
        "nvrx_checkpointing.megatron.legacy_optimizer",
        REPO_ROOT / "megatron/legacy_optimizer.py",
    )
    assert spec is not None
    assert spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)

    dtype_pair = (_DType("torch.bfloat16"), _DType("torch.float32"))
    legacy = "dtype_param_torch:bfloat16_grad_torch:float32"
    result = adapter.restore_optimizer_dtype_keys(
        ({"optimizer": {legacy: "loaded"}}, {"missing"}, {"unexpected"}),
        {legacy: dtype_pair},
    )

    assert result == ({"optimizer": {dtype_pair: "loaded"}}, {"missing"}, {"unexpected"})


def test_module_launcher_installs_before_running_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
