# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

_REPO = Path(__file__).resolve().parent.parent
_SNAPSHOT_ID = 125
_spec = importlib.util.spec_from_file_location("otel_to_perfetto", _REPO / "ci" / "scripts" / "otel_to_perfetto.py")
otel_to_perfetto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(otel_to_perfetto)


def _attribute(key: str, value: int | str) -> dict:
    kind = "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value) if kind == "intValue" else value}}


def _resource() -> dict:
    return {
        "attributes": [
            _attribute("service.name", "megatron-lm"),
            _attribute("service.instance.id", "rank-3"),
            _attribute("nemo.run.id", "job-cycle1-sr0-te0"),
            _attribute("host.name", "node-1"),
            _attribute("dl.rank", 3),
        ],
    }


def _write(path: Path, records: Iterable[dict]) -> None:
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def test_mixed_otlp_jsonl_converts_to_perfetto_events(tmp_path: Path) -> None:
    source = tmp_path / "native_ckpt_otel.jsonl"
    output = tmp_path / "native_ckpt_perfetto.json"
    _write(
        source,
        [
            {
                "resourceSpans": [
                    {
                        "resource": _resource(),
                        "scopeSpans": [
                            {
                                "scope": {"name": "checkpoint"},
                                "spans": [
                                    {
                                        "traceId": "trace-1",
                                        "spanId": "span-1",
                                        "name": "ckpt.save",
                                        "startTimeUnixNano": "1000",
                                        "endTimeUnixNano": "4000",
                                        "attributes": [_attribute("nvrx.snapshot_id", _SNAPSHOT_ID)],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "resourceMetrics": [
                    {
                        "resource": _resource(),
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "megatron.training.loss",
                                        "gauge": {"dataPoints": [{"timeUnixNano": "5000", "asDouble": 1.25}]},
                                    },
                                    {
                                        "name": "megatron.training.step_duration_ms",
                                        "unit": "ms",
                                        "histogram": {
                                            "dataPoints": [
                                                {"timeUnixNano": "6000", "count": "1", "sum": 8.0, "min": 8.0, "max": 8.0},
                                            ],
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "resourceLogs": [
                    {
                        "resource": _resource(),
                        "scopeLogs": [
                            {
                                "logRecords": [
                                    {
                                        "observedTimeUnixNano": "7000",
                                        "severityText": "INFO",
                                        "severityNumber": 9,
                                        "body": {"stringValue": "checkpoint complete"},
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    )

    stats = otel_to_perfetto.convert([source], output)
    converted = json.loads(output.read_text())
    events = converted["traceEvents"]

    assert stats == {"spans": 1, "metrics": 2, "logs": 1, "tracks": 1}
    assert converted["otherData"]["timeOriginUnixNano"] == "1000"
    span = next(event for event in events if event.get("name") == "ckpt.save")
    assert (span["ph"], span["ts"], span["dur"]) == ("X", 0.0, 3.0)
    assert span["args"]["otel.span_id"] == "span-1"
    assert span["args"]["nvrx.snapshot_id"] == _SNAPSHOT_ID
    gauge = next(event for event in events if event.get("name") == "megatron.training.loss")
    assert (gauge["ph"], gauge["ts"], gauge["args"]) == ("C", 4.0, {"value": 1.25})
    histogram = next(event for event in events if event.get("name") == "megatron.training.step_duration_ms (ms)")
    assert histogram["args"] == {"count": 1, "sum": 8.0, "min": 8.0, "max": 8.0}
    log = next(event for event in events if event.get("cat") == "otel.log.info")
    assert (log["ph"], log["ts"], log["args"]["body"]) == ("I", 6.0, "checkpoint complete")
    assert any(event.get("name") == "process_name" and event["args"]["name"] == "cycle1 · node-1" for event in events)
    assert any(event.get("name") == "thread_name" and event["args"]["name"] == "rank 3 · megatron-lm" for event in events)


def test_unknown_jsonl_record_fails_with_location(tmp_path: Path) -> None:
    source = tmp_path / "unknown.jsonl"
    output = tmp_path / "trace.json"
    _write(source, [{"unexpected": []}])

    with pytest.raises(ValueError, match=rf"^{re.escape(str(source))}:1: unknown OTLP record type$"):
        otel_to_perfetto.convert([source], output)


def test_conversion_parses_input_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "native_ckpt_otel.jsonl"
    output = tmp_path / "native_ckpt_perfetto.json"
    _write(
        source,
        [
            {
                "resourceSpans": [
                    {
                        "resource": _resource(),
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "traceId": "trace-1",
                                        "spanId": "span-1",
                                        "name": "checkpoint",
                                        "startTimeUnixNano": "1000",
                                        "endTimeUnixNano": "2000",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    )
    real_open = Path.open
    reads = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal reads
        if path == source:
            reads += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    otel_to_perfetto.convert([source], output)

    assert reads == 1


def test_span_rank_selects_distinct_daemon_tracks(tmp_path: Path) -> None:
    source = tmp_path / "native_ckpt_otel.jsonl"
    output = tmp_path / "native_ckpt_perfetto.json"
    resource = _resource()
    resource["attributes"] = [attribute for attribute in resource["attributes"] if attribute["key"] != "dl.rank"]
    spans = [
        {
            "traceId": f"trace-{rank}",
            "spanId": f"span-{rank}",
            "name": "ckpt.save",
            "startTimeUnixNano": "1000",
            "endTimeUnixNano": "4000",
            "attributes": [_attribute("nvrx.rank", rank)],
        }
        for rank in (3, 4)
    ]
    _write(
        source,
        [{"resourceSpans": [{"resource": resource, "scopeSpans": [{"spans": spans}]}]}],
    )

    stats = otel_to_perfetto.convert([source], output)
    thread_names = {
        event["args"]["name"]
        for event in json.loads(output.read_text())["traceEvents"]
        if event.get("name") == "thread_name"
    }

    assert stats["tracks"] == 2  # noqa: PLR2004
    assert thread_names == {"rank 3 · megatron-lm", "rank 4 · megatron-lm"}
