# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001, S101

from __future__ import annotations

import importlib.util
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
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
