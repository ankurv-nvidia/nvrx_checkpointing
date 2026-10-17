# ruff: noqa: ANN401, PLC0415
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Minimal runtime integration for an unmodified Megatron-LM checkout.

The native provider owns checkpoint saves, its rank-local durability queue,
loads that do not require NVIDIA Resiliency Extension, and post-load
shared-memory registration. Provider selection is intentionally kept out of
this module: callers that compare native with another provider can opt in to
:mod:`nvrx_checkpointing.megatron.provider` before installing this module.
"""

from __future__ import annotations

import inspect
import runpy
import sys
from functools import wraps
from pathlib import Path
from typing import Any

SUPPORTED_MEGATRON_COMMIT = "16251ac1221f4814cd39350def05cefa3c36734a"

_NATIVE_SAVE_MARKER = "_nvrx_checkpointing_native_save_installed"
_NATIVE_ASYNC_MARKER = "_nvrx_checkpointing_native_async_installed"
_NATIVE_LOAD_MARKER = "_nvrx_checkpointing_native_load_installed"


def _read_metadata_uncached(strategy: Any) -> None:
    """Make a load strategy chain read checkpoint metadata directly.

    Megatron's only cached metadata reader belongs to NVIDIA Resiliency
    Extension, and a training process loads its checkpoint once.
    """
    while strategy is not None:
        if getattr(strategy, "cache_metadata", False):
            strategy.cache_metadata = False
        strategy = getattr(strategy, "base_strategy", None)


def _install_load_registration() -> None:
    """Load without NVIDIA Resiliency Extension, then register the constructed state."""
    from megatron.core import dist_checkpointing  # type: ignore[import-not-found]
    from megatron.core.dist_checkpointing.dict_utils import nested_values  # type: ignore[import-not-found]
    from megatron.core.dist_checkpointing.mapping import ShardedTensor  # type: ignore[import-not-found]
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(dist_checkpointing, _NATIVE_LOAD_MARKER, False):
        return

    from . import save as native_save

    original_load = dist_checkpointing.load

    @wraps(original_load)
    def load(sharded_state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        _read_metadata_uncached(args[1] if len(args) > 1 else kwargs.get("sharded_strategy"))
        # TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
        # replica_ids are overwritten by the fully parallel loading process. We need to preserve the original values.
        replica_ids = [(value, value.replica_id) for value in nested_values(sharded_state_dict) if isinstance(value, ShardedTensor)]
        result = original_load(sharded_state_dict, *args, **kwargs)
        for value, replica_id in replica_ids:
            value.replica_id = replica_id
        root = native_save.register_state_dict(sharded_state_dict, 0)
        root.finalize_registry()
        root.wait_for_async_completion()
        return result

    dist_checkpointing.load = load
    setattr(dist_checkpointing, _NATIVE_LOAD_MARKER, True)
    training_module.load_checkpoint = native_save._wrap_load_checkpoint(training_module.load_checkpoint)  # noqa: SLF001


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
                dp_gtp_remat_group=values.get("dp_gtp_remat_group"),
                cp_group=values.get("cp_group"),
                rng_state_key_prefix=values.get("rng_state_key_prefix", ""),
            ),
        )

    training_module.save_checkpoint = save_checkpoint

    setattr(training_module, _NATIVE_SAVE_MARKER, True)


def install() -> None:
    """Install native checkpointing into the current Megatron process."""
    _install_async_queue_bindings()
    _install_load_registration()
    _install_save_wrapper()


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
