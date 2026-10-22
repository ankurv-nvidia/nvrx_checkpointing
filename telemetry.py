# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Optional OpenTelemetry tracing for native checkpoint integration."""

from __future__ import annotations

import importlib
import os
import random
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator

try:
    from opentelemetry import trace
except ImportError:
    trace = None  # type: ignore[assignment]

_MAX_CHECKPOINT_CUDA_TRANSIENT_BYTES = 1 << 20


@dataclass(frozen=True)
class _CudaMemoryState:
    allocated_bytes: int
    reserved_bytes: int
    cumulative_allocated_bytes: int
    cumulative_freed_bytes: int
    hbm_used_bytes: int
    hbm_total_bytes: int


def _capture_cuda_memory() -> _CudaMemoryState | None:
    """Capture allocator counters and whole-device HBM without synchronizing CUDA."""
    if not torch.cuda.is_initialized():
        return None
    stats = torch.cuda.memory_stats()
    required = (
        "allocated_bytes.all.current",
        "reserved_bytes.all.current",
        "allocated_bytes.all.allocated",
        "allocated_bytes.all.freed",
    )
    if any(name not in stats for name in required):
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return _CudaMemoryState(
        allocated_bytes=int(stats["allocated_bytes.all.current"]),
        reserved_bytes=int(stats["reserved_bytes.all.current"]),
        cumulative_allocated_bytes=int(stats["allocated_bytes.all.allocated"]),
        cumulative_freed_bytes=int(stats["allocated_bytes.all.freed"]),
        hbm_used_bytes=int(total_bytes - free_bytes),
        hbm_total_bytes=int(total_bytes),
    )


def _record_cuda_memory_event(
    span: object | None,
    phase: str,
    state: _CudaMemoryState | None,
    *,
    attributes: dict[str, object] | None = None,
) -> None:
    add_event = getattr(span, "add_event", None)
    if add_event is None:
        return
    values: dict[str, object] = {"nvrx.gpu.memory.phase": phase, "nvrx.gpu.memory.available": state is not None}
    if state is not None:
        values.update(
            {
                "nvrx.gpu.memory.allocated_bytes": state.allocated_bytes,
                "nvrx.gpu.memory.reserved_bytes": state.reserved_bytes,
                "nvrx.gpu.memory.cumulative_allocated_bytes": state.cumulative_allocated_bytes,
                "nvrx.gpu.memory.cumulative_freed_bytes": state.cumulative_freed_bytes,
                "nvrx.gpu.memory.hbm_used_bytes": state.hbm_used_bytes,
                "nvrx.gpu.memory.hbm_total_bytes": state.hbm_total_bytes,
            },
        )
    if attributes is not None:
        values.update(attributes)
    add_event(f"nvrx.checkpoint.save.gpu_memory.{phase}", attributes=values)


def _cuda_memory_exit_attributes(entry: _CudaMemoryState, exit_state: _CudaMemoryState) -> dict[str, object]:
    allocated_delta = exit_state.allocated_bytes - entry.allocated_bytes
    reserved_delta = exit_state.reserved_bytes - entry.reserved_bytes
    allocated_during_save = exit_state.cumulative_allocated_bytes - entry.cumulative_allocated_bytes
    freed_during_save = exit_state.cumulative_freed_bytes - entry.cumulative_freed_bytes
    hbm_used_delta = exit_state.hbm_used_bytes - entry.hbm_used_bytes
    allocator_unchanged = allocated_delta == reserved_delta == 0
    transient_activity_within_limit = (
        allocated_during_save <= _MAX_CHECKPOINT_CUDA_TRANSIENT_BYTES
        and freed_during_save <= _MAX_CHECKPOINT_CUDA_TRANSIENT_BYTES
    )
    return {
        "nvrx.gpu.memory.allocated_delta_bytes": allocated_delta,
        "nvrx.gpu.memory.reserved_delta_bytes": reserved_delta,
        "nvrx.gpu.memory.allocated_during_save_bytes": allocated_during_save,
        "nvrx.gpu.memory.freed_during_save_bytes": freed_during_save,
        "nvrx.gpu.memory.hbm_used_delta_bytes": hbm_used_delta,
        "nvrx.gpu.memory.allocator_unchanged": allocator_unchanged,
        "nvrx.gpu.memory.transient_activity_within_limit": transient_activity_within_limit,
        "nvrx.gpu.memory.transient_limit_bytes": _MAX_CHECKPOINT_CUDA_TRANSIENT_BYTES,
        "nvrx.gpu.memory.invariant_satisfied": allocator_unchanged,
        "nvrx.gpu.memory.hbm_unchanged": hbm_used_delta == 0,
    }


def _assert_cuda_memory_unchanged(entry: _CudaMemoryState | None, exit_state: _CudaMemoryState | None) -> None:
    if entry is None or exit_state is None:
        raise RuntimeError("CUDA memory counters are unavailable for the checkpoint invariant")
    details = _cuda_memory_exit_attributes(entry, exit_state)
    if details["nvrx.gpu.memory.invariant_satisfied"]:
        return
    raise RuntimeError(
        "Checkpoint save changed CUDA allocator state: "
        f"allocated_delta_bytes={details['nvrx.gpu.memory.allocated_delta_bytes']}, "
        f"reserved_delta_bytes={details['nvrx.gpu.memory.reserved_delta_bytes']}",
    )


@contextmanager
def assert_cuda_memory_unchanged(span: object | None) -> Generator[None]:
    """Report checkpoint-boundary CUDA memory and enforce allocator stability."""
    entry = _capture_cuda_memory()
    _record_cuda_memory_event(span, "entry", entry)
    if entry is None:
        _assert_cuda_memory_unchanged(entry, None)
    try:
        yield
    finally:
        exit_state = _capture_cuda_memory()
        attributes: dict[str, object] = {"nvrx.gpu.memory.invariant_required": True}
        if entry is not None and exit_state is not None:
            attributes.update(_cuda_memory_exit_attributes(entry, exit_state))
        _record_cuda_memory_event(span, "exit", exit_state, attributes=attributes)
    _assert_cuda_memory_unchanged(entry, exit_state)


def isolate_opentelemetry_randomness() -> list[str]:
    """Keep OpenTelemetry's random draws out of the application's RNG state."""
    system_random = random.SystemRandom()
    targets = {
        "opentelemetry.sdk.trace.id_generator": {"random": system_random},
        "opentelemetry.sdk.metrics._internal.exemplar.exemplar_reservoir": {
            "randrange": system_random.randrange,
        },
        # OTel <= 1.44 keeps the HTTP retry loop in each signal exporter.
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": {"random": system_random},
        "opentelemetry.exporter.otlp.proto.http.metric_exporter": {"random": system_random},
        # Newer releases share the retry loop in this common transport.
        "opentelemetry.exporter.otlp.common.http": {"random": system_random},
    }
    patched = []
    for name, replacements in targets.items():
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for attribute, replacement in replacements.items():
            if hasattr(module, attribute):
                setattr(module, attribute, replacement)
                patched.append(f"{name}.{attribute}")
    return patched


def managed_span(scope: str, name: str, **attributes: object) -> AbstractContextManager[object]:
    """Create a span through the application's provider, or an empty context."""
    if os.environ.get("MEGATRON_OTEL_ENABLED") != "1" or trace is None:
        return nullcontext()
    return trace.get_tracer(scope).start_as_current_span(name, attributes=attributes)


__all__ = ["assert_cuda_memory_unchanged", "isolate_opentelemetry_randomness", "managed_span"]
