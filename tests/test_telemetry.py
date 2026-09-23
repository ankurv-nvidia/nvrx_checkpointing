# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import importlib.util
import random
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_telemetry() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nvrx_checkpointing._telemetry_test", REPO_ROOT / "telemetry.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
