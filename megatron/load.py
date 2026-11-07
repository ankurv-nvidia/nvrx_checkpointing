# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: ANN401

"""Restore daemon-bundled objects alongside DCP tensor shards."""

from __future__ import annotations

import inspect
import io
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, create_default_local_load_plan
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, MetadataIndex
from torch.distributed.checkpoint.planner import LoadPlanner

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.planner import LoadPlan, ReadItem

COMMON_STATE_KEY = "common_state/shard_0_1"
_INSTALL_MARKER = "_nvrx_checkpointing_bundle_load_installed"


def is_common_state_key(key: str) -> bool:
    """Identify the rank-local non-tensor bundle records."""
    return key == COMMON_STATE_KEY or key.startswith(f"{COMMON_STATE_KEY}.rank_")


def decode_common_value(value: Any) -> Any:
    """Restore one opaque object archive or captured primitive value."""
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=False) if isinstance(value, bytes) else value


def decode_common_state(state: dict[str, Any]) -> dict[str, Any]:
    """Restore opaque object archives in the captured common state."""
    return {key: decode_common_value(value) for key, value in state.items()}


def read_common_state(reader: FileSystemReader, metadata: Metadata, *, decode_objects: bool = True) -> dict[str, Any]:
    """Read the non-tensor dictionaries from their local-owner bundles."""
    state = {}
    for key in metadata.state_dict_metadata:
        if not is_common_state_key(key):
            continue
        storage = metadata.storage_data[MetadataIndex(key)]
        with (Path(reader.path) / storage.relative_path).open("rb") as stream:
            stream.seek(storage.offset)
            bundle = torch.load(io.BytesIO(stream.read(storage.length)), map_location="cpu", weights_only=False)
        if isinstance(bundle, io.BytesIO):
            bundle.seek(0)
            bundle = torch.load(bundle, map_location="cpu", weights_only=False)
        if not isinstance(bundle, list) or len(bundle) != 1 or not isinstance(bundle[0], dict):
            raise TypeError("Expected one bundled non-tensor state dictionary")
        objects = decode_common_state(bundle[0]) if decode_objects else bundle[0]
        duplicates = state.keys() & objects.keys()
        if duplicates:
            raise ValueError(f"Duplicate bundled checkpoint objects: {sorted(duplicates)}")
        state.update(objects)
    return state


class BundleFileSystemReader(FileSystemReader):
    """Expose bundled object keys while retaining physical tensor metadata."""

    objects: dict[str, Any] | None = None

    def reset(self, checkpoint_id: str | Path | None = None) -> None:
        self.objects = None
        super().reset(checkpoint_id)

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        metadata = super().read_metadata(*args, **kwargs)
        if not any(map(is_common_state_key, metadata.state_dict_metadata)) or (Path(self.path) / "metadata.json").is_file():
            return metadata
        if self.objects is None:
            self.objects = read_common_state(self, metadata, decode_objects=False)
        logical_metadata = dict(metadata.state_dict_metadata)
        for key in self.objects:
            if key in logical_metadata:
                raise ValueError(f"Bundled object '{key}' duplicates a checkpoint entry")
            logical_metadata[key] = BytesStorageMetadata()
        return replace(metadata, state_dict_metadata=logical_metadata)


class _ObjectLoadPlanner(DefaultLoadPlanner):
    """Restore captured objects and plan physical tensor reads."""

    def __init__(self, objects: dict[str, Any], planner: LoadPlanner | None = None) -> None:
        super().__init__(
            flatten_state_dict=getattr(planner, "flatten_state_dict", True),
            flatten_sharded_tensors=getattr(planner, "flatten_sharded_tensors", True),
            allow_partial_load=getattr(planner, "allow_partial_load", False),
        )
        self.objects = objects

    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)
        tensors = {}
        for key, target in self.state_dict.items():
            if key not in self.objects and isinstance(target, torch.Tensor):
                tensors[key] = target
                continue
            if key not in self.objects:
                raise RuntimeError(f"Missing bundled checkpoint object: {key}")
            value = decode_common_value(self.objects[key])
            if isinstance(target, torch.Tensor):
                if not isinstance(value, torch.Tensor) or target.shape != value.shape:
                    raise ValueError(f"Bundled tensor '{key}' does not match the destination shape")
                target.detach().copy_(value)
                continue
            if isinstance(target, bool) and isinstance(value, int):
                value = bool(value)
            if self.flatten_state_dict:
                set_element(self.original_state_dict, self.mappings[key], value)
            else:
                self.original_state_dict[key] = value
        self.state_dict = tensors

    def create_local_plan(self) -> LoadPlan:
        if self.metadata is None:
            raise RuntimeError("Checkpoint metadata is unavailable")
        return create_default_local_load_plan(self.state_dict, self.metadata, not self.allow_partial_load)


class BundleLoadPlanner(LoadPlanner):
    """Select bundled restoration after synchronized metadata loading."""

    def __init__(self, reader: BundleFileSystemReader, planner: LoadPlanner | None = None) -> None:
        self.reader = reader
        self.base_planner = planner if planner is not None else DefaultLoadPlanner()
        self.planner = self.base_planner

    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        self.planner = self.base_planner
        if self.reader.objects is not None and not any(map(is_common_state_key, state_dict)):
            self.planner = _ObjectLoadPlanner(self.reader.objects, self.base_planner)
        self.planner.set_up_planner(state_dict, metadata, is_coordinator)

    def create_local_plan(self) -> LoadPlan:
        return self.planner.create_local_plan()

    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return self.planner.create_global_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return self.planner.finish_plan(new_plan)

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        self.planner.load_bytes(read_item, value)

    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        return self.planner.resolve_tensor(read_item)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        self.planner.commit_tensor(read_item, tensor)


def _wrap_load(function: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(function)

    @wraps(function)
    def load(*args: Any, **kwargs: Any) -> Any:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        reader = call.arguments.get("storage_reader")
        if reader is None and call.arguments.get("checkpoint_id") is not None:
            reader = BundleFileSystemReader(call.arguments["checkpoint_id"])
        elif type(reader) is FileSystemReader:
            adapted_reader = BundleFileSystemReader(reader.path)
            adapted_reader.fs = reader.fs
            adapted_reader.transforms = reader.transforms
            reader = adapted_reader
        if isinstance(reader, BundleFileSystemReader):
            call.arguments["planner"] = BundleLoadPlanner(reader, call.arguments.get("planner"))
            call.arguments["storage_reader"] = reader
        return function(*call.args, **call.kwargs)

    return load


def install(checkpointing_module: ModuleType) -> None:
    """Install native-bundle handling in Megatron's DCP entry points."""
    if not getattr(dcp, _INSTALL_MARKER, False):
        dcp.load = _wrap_load(dcp.load)
        dcp.load_state_dict = _wrap_load(dcp.load_state_dict)
        dcp.FileSystemReader = BundleFileSystemReader
        setattr(dcp, _INSTALL_MARKER, True)
    checkpointing_module.FileSystemReader = dcp.FileSystemReader
