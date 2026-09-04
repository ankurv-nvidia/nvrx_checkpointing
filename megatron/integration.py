# ruff: noqa: ANN401, PLC0415
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Minimal runtime integration for an unmodified Megatron-LM checkout.

The native provider owns checkpoint saves, its rank-local durability queue, and
load compatibility. Provider selection is intentionally kept out of this
module: callers that compare native with another provider can opt in to
:mod:`nvrx_checkpointing.megatron.provider` before installing this module.
"""

from __future__ import annotations

import inspect
import runpy
import sys
from functools import wraps
from pathlib import Path
from typing import Any

SUPPORTED_MEGATRON_COMMIT = "aac9b2db7f0d203940f6c0a7f2852a094c18c12e"

_NATIVE_SAVE_MARKER = "_nvrx_checkpointing_native_save_installed"
_NATIVE_ASYNC_MARKER = "_nvrx_checkpointing_native_async_installed"
_NATIVE_LOAD_MARKER = "_nvrx_checkpointing_native_load_installed"
_NATIVE_WARMUP_MARKER = "_nvrx_checkpointing_native_warmup_installed"

class _MCoreLoadStrategy:
    """Per-load proxy that keeps native independent of an external async reader."""

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy

    def load(
        self,
        sharded_state_dict: dict[str, Any],
        checkpoint_dir: str | Path,
        async_strategy: str = "mcore",
    ) -> Any:
        del async_strategy
        return self._strategy.load(sharded_state_dict, checkpoint_dir, "mcore")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._strategy, name)


def _mcore_load_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Wrap the strategy for one public ``dist_checkpointing.load`` call."""
    from megatron.core.dist_checkpointing.strategies.torch import (  # type: ignore[import-not-found]
        TorchDistLoadShardedStrategy,
    )

    args = list(args)
    kwargs = dict(kwargs)
    if len(args) >= 2:  # noqa: PLR2004 # checkpoint_dir, sharded_strategy
        strategy = args[1]
        if strategy is None:
            strategy = TorchDistLoadShardedStrategy()
        args[1] = _MCoreLoadStrategy(strategy)
    elif "sharded_strategy" in kwargs:
        strategy = kwargs["sharded_strategy"]
        if strategy is None:
            strategy = TorchDistLoadShardedStrategy()
        kwargs["sharded_strategy"] = _MCoreLoadStrategy(strategy)
    else:
        kwargs["sharded_strategy"] = _MCoreLoadStrategy(TorchDistLoadShardedStrategy())
    return tuple(args), kwargs


def _install_load_wrapper() -> None:
    """Use MCore's public reader and retain old optimizer compatibility."""
    from megatron.core import dist_checkpointing  # type: ignore[import-not-found]

    if getattr(dist_checkpointing, _NATIVE_LOAD_MARKER, False):
        return

    from .legacy_optimizer import adapt_legacy_optimizer_request, restore_optimizer_dtype_keys

    original_load = dist_checkpointing.load

    @wraps(original_load)
    def load(sharded_state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        load_args, load_kwargs = _mcore_load_args(args, kwargs)
        try:
            result = original_load(sharded_state_dict, *load_args, **load_kwargs)
        except BaseException as current_error:  # PyTorch CheckpointException derives from BaseException.
            if isinstance(current_error, (KeyboardInterrupt, SystemExit)):
                raise
            legacy_request, dtype_lookup = adapt_legacy_optimizer_request(sharded_state_dict)
            if not dtype_lookup:
                raise
            try:
                result = original_load(legacy_request, *load_args, **load_kwargs)
            except BaseException as legacy_error:
                if isinstance(legacy_error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise current_error from legacy_error
            result = restore_optimizer_dtype_keys(result, dtype_lookup)

        return result

    dist_checkpointing.load = load
    setattr(dist_checkpointing, _NATIVE_LOAD_MARKER, True)


def _install_async_queue_bindings() -> None:
    """Bind Megatron's training-loop callbacks to native's rank-local queue."""
    from megatron.training import initialize as initialize_module  # type: ignore[import-not-found]
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(training_module, _NATIVE_ASYNC_MARKER, False):
        return

    from . import save as native_save

    initialize_module.init_persistent_async_worker = native_save.init_persistent_async_worker
    training_module.maybe_finalize_async_save = native_save.maybe_finalize_async_save

    setattr(training_module, _NATIVE_ASYNC_MARKER, True)


def _install_save_wrapper() -> None:
    """Route Megatron's public training save binding to the native writer."""
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(training_module, _NATIVE_SAVE_MARKER, False):
        return

    original_save_checkpoint = training_module.save_checkpoint
    save_signature = inspect.signature(original_save_checkpoint)

    @wraps(original_save_checkpoint)
    def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
        from . import save as native_save

        call = save_signature.bind(*args, **kwargs)
        call.apply_defaults()
        values = call.arguments
        return native_save.save_checkpoint(
            native_save.SaveCheckpointArgs(
                iteration=values["iteration"],
                model=values["model"],
                optimizer=values["optimizer"],
                opt_param_scheduler=values["opt_param_scheduler"],
                num_floating_point_operations_so_far=values["num_floating_point_operations_so_far"],
                train_data_iterator=values.get("train_data_iterator"),
                tp_group=values.get("tp_group"),
                pp_group=values.get("pp_group"),
                dp_cp_group=values.get("dp_cp_group"),
                dp_group=values.get("dp_group"),
                rng_state_key_prefix=values.get("rng_state_key_prefix", ""),
            ),
        )

    training_module.save_checkpoint = save_checkpoint

    setattr(training_module, _NATIVE_SAVE_MARKER, True)


def _install_warmup() -> None:
    """Preserve the existing pre-training shared-memory warmup."""
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(training_module, _NATIVE_WARMUP_MARKER, False):
        return

    original_train = training_module.train

    @wraps(original_train)
    def train(*args: Any, **kwargs: Any) -> Any:
        from . import save as native_save

        native_save.warmup()
        return original_train(*args, **kwargs)

    training_module.train = train

    setattr(training_module, _NATIVE_WARMUP_MARKER, True)


def install() -> None:
    """Install native checkpointing into the current Megatron process."""
    _install_async_queue_bindings()
    _install_load_wrapper()
    _install_save_wrapper()
    _install_warmup()


def main(argv: list[str] | None = None) -> None:
    """Run a Megatron entrypoint with the native provider installed."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m nvrx_checkpointing.megatron.integration <megatron-entrypoint.py> [args ...]",
        )

    entrypoint = Path(args[0]).resolve()
    if not entrypoint.is_file():
        raise SystemExit(f"Megatron entrypoint not found: {entrypoint}")

    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *args[1:]]
    install()
    runpy.run_path(str(entrypoint), run_name="__main__")


__all__ = ["SUPPORTED_MEGATRON_COMMIT", "install", "main"]


if __name__ == "__main__":
    main()
