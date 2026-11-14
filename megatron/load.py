# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401

"""Route filesystem checkpoint loads through the native dictionary binding."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from copy import deepcopy
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
            state["args"] = deepcopy(template)
        initialized = dist.is_available() and dist.is_initialized()
        checkpoint.load_checkpoint(
            directory,
            state,
            rank=dist.get_rank() if initialized else 0,
            world_size=dist.get_world_size() if initialized else 1,
            allow_partial_load=getattr(planner, "allow_partial_load", False),
        )
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
            token = _BOOTSTRAP_TEMPLATE.set(call.arguments.get("args") if call.arguments.get("rank0") else None)
            try:
                return original(*args, **kwargs)
            finally:
                _BOOTSTRAP_TEMPLATE.reset(token)

        setattr(load_base_checkpoint, _INSTALL_MARKER, True)
        checkpointing_module._load_base_checkpoint = load_base_checkpoint  # noqa: SLF001
