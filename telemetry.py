# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Optional OpenTelemetry tracing for native checkpoint integration."""

from __future__ import annotations

import importlib
import os
import random
from contextlib import AbstractContextManager, nullcontext

try:
    from opentelemetry import trace
except ImportError:
    trace = None  # type: ignore[assignment]


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


__all__ = ["isolate_opentelemetry_randomness", "managed_span"]
