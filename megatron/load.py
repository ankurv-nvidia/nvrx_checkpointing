# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401

"""Route filesystem checkpoint loads through the native dictionary binding."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_INSTALL_MARKER = "_nvrx_checkpointing_dictionary_load_installed"
_BOOTSTRAP_TEMPLATE: ContextVar[Any] = ContextVar("checkpoint_bootstrap_template", default=None)
_LOAD_DEFAULTS: ContextVar[dict[str, Any] | None] = ContextVar("checkpoint_load_defaults", default=None)


def _restore_optimizer_metadata(groups: dict, requested: dict, *, allow_partial_load: bool) -> None:
    for name, (keys, default) in requested.items():
        group = groups.get(name, {})
        if not allow_partial_load:
            for key in keys:
                if key != "is_dsa_indexer" and key not in group:
                    raise ValueError(f"Missing checkpoint value: optimizer.param_to_group_meta.{name}.{key}")
        group.setdefault("is_dsa_indexer", default)


def _wrap_load(function: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(function)

    @wraps(function)
    def load(*args: Any, **kwargs: Any) -> Any:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        reader = call.arguments.get("storage_reader")
        if reader is not None and not isinstance(reader, dcp.FileSystemReader):
            return function(*args, **kwargs)
        directory = getattr(reader, "path", None) if reader is not None else call.arguments.get("checkpoint_id")
        if directory is None:
            return function(*args, **kwargs)
        directory = Path(directory)
        if not (directory / ".metadata").is_file():
            return function(*args, **kwargs)
        planner = call.arguments.get("planner")
        if (
            planner is not None
            and type(planner).__module__ != "torch.distributed.checkpoint.default_planner"
            and (directory / "metadata.json").is_file()
        ):
            return function(*args, **kwargs)
        state = call.arguments["state_dict"]
        if "args" in state and state["args"] is None and (template := _BOOTSTRAP_TEMPLATE.get()) is not None:
            state["args"] = template
        optimizer = state.get("optimizer")
        groups = optimizer.get("param_to_group_meta", {}) if isinstance(optimizer, dict) else {}
        requested_groups = {
            name: (tuple(group), group["is_dsa_indexer"])
            for name, group in groups.items()
            if isinstance(group, dict) and "is_dsa_indexer" in group
        }
        missing_defaults = (_LOAD_DEFAULTS.get() or {}) | {
            f"optimizer.param_to_group_meta.{name}.is_dsa_indexer": default for name, (_, default) in requested_groups.items()
        }
        allow_partial_load = getattr(planner, "allow_partial_load", False)
        initialized = dist.is_available() and dist.is_initialized()
        checkpoint.load_checkpoint(
            directory,
            state,
            rank=dist.get_rank() if initialized else 0,
            world_size=dist.get_world_size() if initialized else 1,
            allow_partial_load=allow_partial_load,
            missing_defaults=missing_defaults,
        )
        _restore_optimizer_metadata(groups, requested_groups, allow_partial_load=allow_partial_load)
        return None

    return load


def install(checkpointing_module: ModuleType) -> None:
    """Install the dictionary load hook in Megatron's filesystem load path."""
    if not getattr(dcp, _INSTALL_MARKER, False):
        dcp.load = _wrap_load(dcp.load)
        dcp.load_state_dict = _wrap_load(dcp.load_state_dict)
        setattr(dcp, _INSTALL_MARKER, True)
    original = getattr(checkpointing_module, "_load_base_checkpoint", None)
    if original is not None and not getattr(original, _INSTALL_MARKER, False):
        signature = inspect.signature(original)

        @wraps(original)
        def load_base_checkpoint(*args: Any, **kwargs: Any) -> Any:
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            template = None
            if call.arguments.get("rank0"):
                template = call.arguments.get("args")
                if template is None and (get_args := getattr(checkpointing_module, "get_args", None)) is not None:
                    template = get_args()
            defaults = {}
            sharded_state = call.arguments.get("sharded_state_dict")
            if not call.arguments.get("rank0") and isinstance(sharded_state, dict) and "_model" in sharded_state:
                defaults["num_floating_point_operations_so_far"] = 0
                sharded_state.setdefault("num_floating_point_operations_so_far", 0)
            token = _BOOTSTRAP_TEMPLATE.set(template)
            defaults_token = _LOAD_DEFAULTS.set(defaults)
            try:
                return original(*args, **kwargs)
            finally:
                _LOAD_DEFAULTS.reset(defaults_token)
                _BOOTSTRAP_TEMPLATE.reset(token)

        setattr(load_base_checkpoint, _INSTALL_MARKER, True)
        checkpointing_module._load_base_checkpoint = load_base_checkpoint  # noqa: SLF001
