# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import importlib.util
import random
import sys
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_telemetry() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nvrx_checkpointing._telemetry_test", REPO_ROOT / "telemetry.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_managed_span_is_empty_when_disabled_or_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()

    class UnexpectedTrace:
        @staticmethod
        def get_tracer(_scope: str) -> None:
            raise AssertionError("disabled telemetry requested a tracer")

    monkeypatch.setattr(telemetry, "trace", UnexpectedTrace)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
    with telemetry.managed_span("checkpoint", "disabled") as span:
        assert span is None

    monkeypatch.setattr(telemetry, "trace", None)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
    with telemetry.managed_span("checkpoint", "unavailable") as span:
        assert span is None


def test_opentelemetry_randomness_isolated_from_application_state(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    id_generator = SimpleNamespace(random=random)
    reservoir = SimpleNamespace(randrange=random.randrange)
    trace_exporter = SimpleNamespace(random=random)
    metric_exporter = SimpleNamespace(random=random)
    common_http = SimpleNamespace(random=random)
    modules = {
        "opentelemetry.sdk.trace.id_generator": id_generator,
        "opentelemetry.sdk.metrics._internal.exemplar.exemplar_reservoir": reservoir,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": trace_exporter,
        "opentelemetry.exporter.otlp.proto.http.metric_exporter": metric_exporter,
        "opentelemetry.exporter.otlp.common.http": common_http,
    }
    monkeypatch.setattr(telemetry.importlib, "import_module", modules.__getitem__)
    state = random.getstate()

    patched = telemetry.isolate_opentelemetry_randomness()
    id_generator.random.getrandbits(128)
    reservoir.randrange(1024)
    trace_exporter.random.uniform(0, 1)
    metric_exporter.random.uniform(0, 1)
    common_http.random.uniform(0, 1)

    assert random.getstate() == state
    assert patched == [
        "opentelemetry.sdk.trace.id_generator.random",
        "opentelemetry.sdk.metrics._internal.exemplar.exemplar_reservoir.randrange",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.random",
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.random",
        "opentelemetry.exporter.otlp.common.http.random",
    ]

    def unavailable(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(telemetry.importlib, "import_module", unavailable)
    assert telemetry.isolate_opentelemetry_randomness() == []


def test_managed_span_uses_application_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    events: list[tuple[object, ...]] = []

    class FakeTracer:
        def __init__(self, scope: str) -> None:
            self.scope = scope

        def start_as_current_span(self, name: str, *, attributes: dict[str, object]) -> AbstractContextManager[object]:
            @contextmanager
            def span() -> Iterator[object]:
                token = object()
                events.append(("begin", self.scope, name, attributes, token))
                yield token
                events.append(("end", name, token))

            return span()

    class FakeTrace:
        @staticmethod
        def get_tracer(scope: str) -> FakeTracer:
            events.append(("tracer", scope))
            return FakeTracer(scope)

    monkeypatch.setattr(telemetry, "trace", FakeTrace)
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

    with telemetry.managed_span("checkpoint", "nvrx.checkpoint.save", iteration=7) as span:
        assert span is not None

    assert [(event[1], event[2], event[3]) for event in events if event[0] == "begin"] == [
        ("checkpoint", "nvrx.checkpoint.save", {"iteration": 7}),
    ]


def test_cuda_memory_assertion_reports_unchanged_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    events: list[tuple[str, dict[str, object]]] = []

    class Span:
        @staticmethod
        def add_event(name: str, *, attributes: dict[str, object]) -> None:
            events.append((name, attributes))

    stats = {
        "allocated_bytes.all.current": 100,
        "reserved_bytes.all.current": 200,
        "allocated_bytes.all.allocated": 300,
        "allocated_bytes.all.freed": 200,
    }
    monkeypatch.setattr(telemetry.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(telemetry.torch.cuda, "memory_stats", lambda: stats)
    monkeypatch.setattr(telemetry.torch.cuda, "mem_get_info", lambda: (700, 1_000))

    with telemetry.assert_cuda_memory_unchanged(Span()):
        pass

    assert [name for name, _ in events] == [
        "nvrx.checkpoint.save.gpu_memory.entry",
        "nvrx.checkpoint.save.gpu_memory.exit",
    ]
    exit_attributes = events[1][1]
    assert exit_attributes["nvrx.gpu.memory.allocated_delta_bytes"] == 0
    assert exit_attributes["nvrx.gpu.memory.allocated_during_save_bytes"] == 0
    assert exit_attributes["nvrx.gpu.memory.allocator_unchanged"] is True
    assert exit_attributes["nvrx.gpu.memory.transient_activity_within_limit"] is True
    assert exit_attributes["nvrx.gpu.memory.invariant_satisfied"] is True
    assert exit_attributes["nvrx.gpu.memory.hbm_unchanged"] is True
    assert exit_attributes["nvrx.gpu.memory.invariant_required"] is True


def test_cuda_memory_assertion_reports_transient_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    events: list[tuple[str, dict[str, object]]] = []
    temporary_allocation_bytes = 2 << 20

    class Span:
        @staticmethod
        def add_event(name: str, *, attributes: dict[str, object]) -> None:
            events.append((name, attributes))

    stats = iter(
        [
            {
                "allocated_bytes.all.current": 100,
                "reserved_bytes.all.current": 200,
                "allocated_bytes.all.allocated": 300,
                "allocated_bytes.all.freed": 200,
            },
            {
                "allocated_bytes.all.current": 100,
                "reserved_bytes.all.current": 200,
                "allocated_bytes.all.allocated": 300 + temporary_allocation_bytes,
                "allocated_bytes.all.freed": 200 + temporary_allocation_bytes,
            },
        ],
    )
    monkeypatch.setattr(telemetry.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(telemetry.torch.cuda, "memory_stats", lambda: next(stats))
    monkeypatch.setattr(telemetry.torch.cuda, "mem_get_info", lambda: (700, 1_000))

    with telemetry.assert_cuda_memory_unchanged(Span()):
        pass

    assert events[1][1]["nvrx.gpu.memory.allocated_during_save_bytes"] == temporary_allocation_bytes
    assert events[1][1]["nvrx.gpu.memory.transient_activity_within_limit"] is False
    assert events[1][1]["nvrx.gpu.memory.invariant_satisfied"] is True


def test_cuda_memory_assertion_rejects_changed_allocator_state(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    stats = iter(
        [
            {
                "allocated_bytes.all.current": 100,
                "reserved_bytes.all.current": 200,
                "allocated_bytes.all.allocated": 300,
                "allocated_bytes.all.freed": 200,
            },
            {
                "allocated_bytes.all.current": 101,
                "reserved_bytes.all.current": 200,
                "allocated_bytes.all.allocated": 301,
                "allocated_bytes.all.freed": 200,
            },
        ],
    )
    monkeypatch.setattr(telemetry.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(telemetry.torch.cuda, "memory_stats", lambda: next(stats))
    monkeypatch.setattr(telemetry.torch.cuda, "mem_get_info", lambda: (700, 1_000))

    with pytest.raises(RuntimeError, match="allocated_delta_bytes=1"), telemetry.assert_cuda_memory_unchanged(None):
        pass


def test_cuda_memory_assertion_rejects_unavailable_counters_before_save(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _load_telemetry()
    entered = False
    monkeypatch.setattr(telemetry.torch.cuda, "is_initialized", lambda: False)

    with (
        pytest.raises(RuntimeError, match="CUDA memory counters are unavailable"),
        telemetry.assert_cuda_memory_unchanged(None),
    ):
        entered = True

    assert not entered
