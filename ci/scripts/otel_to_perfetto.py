#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import TextIO

_SIGNALS = ("resourceSpans", "resourceMetrics", "resourceLogs")
_PHASE_RE = re.compile(r"(?:^|-)((?:phase|cycle)\d+)(?:-|$)")


def _value(value: dict) -> object:
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"])
    return value.get("stringValue", value.get("bytesValue"))


def _attributes(record: dict) -> dict[str, object]:
    return {item["key"]: _value(item.get("value", {})) for item in record.get("attributes", [])}


def _records(paths: Sequence[Path]) -> Iterator[tuple[Path, dict]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid OTLP JSON") from error
                if not any(signal in record for signal in _SIGNALS):
                    raise ValueError(f"{path}:{line_number}: unknown OTLP record type")
                yield path, record


def _point_time(point: dict) -> int:
    return int(point.get("timeUnixNano") or point.get("startTimeUnixNano") or 0)


def _span_timestamps(record: dict) -> Iterator[int]:
    for resource in record.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                yield int(span.get("startTimeUnixNano", 0))


def _metric_timestamps(record: dict) -> Iterator[int]:
    for resource in record.get("resourceMetrics", []):
        for scope in resource.get("scopeMetrics", []):
            for metric in scope.get("metrics", []):
                for kind in ("gauge", "sum", "histogram", "exponentialHistogram", "summary"):
                    for point in metric.get(kind, {}).get("dataPoints", []):
                        yield _point_time(point)


def _log_timestamps(record: dict) -> Iterator[int]:
    for resource in record.get("resourceLogs", []):
        for scope in resource.get("scopeLogs", []):
            for log in scope.get("logRecords", []):
                yield int(log.get("timeUnixNano") or log.get("observedTimeUnixNano") or 0)


def _timestamps(record: dict) -> Iterator[int]:
    yield from _span_timestamps(record)
    yield from _metric_timestamps(record)
    yield from _log_timestamps(record)


def _origin(records: Sequence[tuple[Path, dict]]) -> int:
    origin_ns: int | None = None
    for _, record in records:
        for timestamp in _timestamps(record):
            if timestamp > 0 and (origin_ns is None or timestamp < origin_ns):
                origin_ns = timestamp
    if origin_ns is None:
        raise ValueError("no timestamped OTLP records found")
    return origin_ns


class _Writer:
    def __init__(self, handle: TextIO, origin_ns: int) -> None:
        self._handle = handle
        self._first = True
        handle.write('{"displayTimeUnit":"ns","traceEvents":[')
        self._origin_ns = origin_ns

    def event(self, event: dict) -> None:
        if not self._first:
            self._handle.write(",")
        json.dump(event, self._handle, separators=(",", ":"))
        self._first = False

    def micros(self, timestamp_ns: int) -> float:
        return round((timestamp_ns - self._origin_ns) / 1_000, 3)

    def finish(self) -> None:
        self._handle.write(
            f'],"otherData":{{"timeOriginUnixNano":"{self._origin_ns}","source":"OpenTelemetry OTLP JSONL"}}}}\n',
        )


class _Tracks:
    def __init__(self, writer: _Writer) -> None:
        self._writer = writer
        self._processes: dict[tuple[str, str, str], int] = {}
        self._tracks: dict[tuple[tuple[str, str, str], str, str, str], tuple[int, int]] = {}
        self._next_tid: dict[int, int] = {}

    def resolve(self, path: Path, resource: dict[str, object], event_rank: object | None = None) -> tuple[int, int]:
        service = str(resource.get("service.name") or "unknown-service")
        run_id = str(resource.get("nemo.run.id") or "")
        phase_match = _PHASE_RE.search(run_id)
        phase = phase_match.group(1) if phase_match else path.parent.name
        host = str(resource.get("host.name") or service)
        process_key = phase, run_id, host
        pid = self._processes.get(process_key)
        if pid is None:
            pid = len(self._processes) + 1
            self._processes[process_key] = pid
            self._next_tid[pid] = 1
            self._writer.event({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": f"{phase} · {host}"}})
            self._writer.event({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0, "args": {"sort_index": pid}})

        rank = event_rank if event_rank is not None else resource.get("dl.rank", resource.get("nvrx.rank"))
        identity = str(rank if rank is not None else resource.get("process.pid") or resource.get("service.instance.id") or service)
        instance = str(resource.get("service.instance.id") or "")
        track_key = process_key, service, identity, instance
        track = self._tracks.get(track_key)
        if track is not None:
            return track

        tid = self._next_tid[pid]
        self._next_tid[pid] += 1
        self._tracks[track_key] = pid, tid
        name = f"rank {rank} · {service}" if rank is not None else service
        self._writer.event({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}})
        sort_index = int(rank) if rank is not None else tid
        self._writer.event({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": sort_index}})
        self._writer.event(
            {"ph": "I", "name": "OTel resource", "cat": "otel.resource", "pid": pid, "tid": tid, "ts": 0, "s": "t", "args": resource},
        )
        return pid, tid

    @property
    def count(self) -> int:
        return len(self._tracks)


def _scope_args(scope: dict) -> dict[str, object]:
    metadata = scope.get("scope", {})
    args: dict[str, object] = {}
    if name := metadata.get("name"):
        args["otel.scope.name"] = name
    if version := metadata.get("version"):
        args["otel.scope.version"] = version
    return args


def _emit_spans(
    writer: _Writer,
    tracks: _Tracks,
    path: Path,
    record: dict,
    seen: set[tuple[str, str]],
) -> int:
    count = 0
    for resource_spans in record.get("resourceSpans", []):
        resource = _attributes(resource_spans.get("resource", {}))
        service = str(resource.get("service.name") or "unknown-service")
        for scope in resource_spans.get("scopeSpans", []):
            scope_args = _scope_args(scope)
            for span in scope.get("spans", []):
                span_args = _attributes(span)
                pid, tid = tracks.resolve(path, resource, span_args.get("nvrx.rank"))
                trace_id, span_id = str(span.get("traceId") or ""), str(span.get("spanId") or "")
                if not trace_id or not span_id:
                    raise ValueError(f"{path}: span {span.get('name', '')!r} is missing traceId or spanId")
                if (trace_id, span_id) in seen:
                    continue
                seen.add((trace_id, span_id))
                start_ns = int(span.get("startTimeUnixNano", 0))
                end_ns = int(span.get("endTimeUnixNano", 0))
                if start_ns <= 0 or end_ns < start_ns:
                    raise ValueError(f"{path}: span {span_id} has invalid timestamps")
                args = {
                    "otel.trace_id": trace_id,
                    "otel.span_id": span_id,
                    "otel.parent_span_id": str(span.get("parentSpanId") or ""),
                    "otel.kind": span.get("kind", 0),
                    "otel.status.code": (span.get("status") or {}).get("code", 0),
                    **scope_args,
                    **span_args,
                }
                writer.event(
                    {
                        "ph": "X",
                        "name": str(span.get("name") or "unnamed span"),
                        "cat": f"otel.span,{service}",
                        "pid": pid,
                        "tid": tid,
                        "ts": writer.micros(start_ns),
                        "dur": round((end_ns - start_ns) / 1_000, 3),
                        "args": args,
                    },
                )
                count += 1
    return count


def _metric_name(metric: dict, point: dict) -> str:
    name = str(metric.get("name") or "unnamed metric")
    labels = _attributes(point)
    if labels:
        name += "{" + ",".join(f"{key}={labels[key]}" for key in sorted(labels)) + "}"
    if unit := metric.get("unit"):
        name += f" ({unit})"
    return name


def _number(point: dict) -> int | float:
    if "asDouble" in point:
        return float(point["asDouble"])
    if "asInt" in point:
        return int(point["asInt"])
    raise ValueError("metric data point has no numeric value")


def _aggregate_values(point: dict, keys: Sequence[str]) -> dict[str, int | float]:
    return {
        key: int(point[key]) if key == "count" else float(point[key])
        for key in keys
        if key in point
    }


def _emit_metrics(writer: _Writer, tracks: _Tracks, path: Path, record: dict) -> int:
    count = 0
    for resource_metrics in record.get("resourceMetrics", []):
        resource = _attributes(resource_metrics.get("resource", {}))
        for scope in resource_metrics.get("scopeMetrics", []):
            for metric in scope.get("metrics", []):
                kinds = [kind for kind in ("gauge", "sum", "histogram", "exponentialHistogram", "summary") if kind in metric]
                if len(kinds) != 1:
                    raise ValueError(f"{path}: metric {metric.get('name', '')!r} has unsupported data")
                kind = kinds[0]
                for point in metric[kind].get("dataPoints", []):
                    point_args = _attributes(point)
                    pid, tid = tracks.resolve(path, resource, point_args.get("nvrx.rank"))
                    timestamp = _point_time(point)
                    if timestamp <= 0:
                        raise ValueError(f"{path}: metric {metric.get('name', '')!r} has no timestamp")
                    if kind in ("gauge", "sum"):
                        values = {"value": _number(point)}
                    elif kind == "summary":
                        values = {f"p{item['quantile']}": float(item["value"]) for item in point.get("quantileValues", [])}
                        values.update(_aggregate_values(point, ("count", "sum")))
                    else:
                        values = _aggregate_values(point, ("count", "sum", "min", "max"))
                    writer.event(
                        {
                            "ph": "C",
                            "name": _metric_name(metric, point),
                            "cat": "otel.metric",
                            "pid": pid,
                            "tid": tid,
                            "ts": writer.micros(timestamp),
                            "args": values,
                        },
                    )
                    count += 1
    return count


def _emit_logs(writer: _Writer, tracks: _Tracks, path: Path, record: dict) -> int:
    count = 0
    for resource_logs in record.get("resourceLogs", []):
        resource = _attributes(resource_logs.get("resource", {}))
        for scope in resource_logs.get("scopeLogs", []):
            scope_args = _scope_args(scope)
            for log in scope.get("logRecords", []):
                log_args = _attributes(log)
                pid, tid = tracks.resolve(path, resource, log_args.get("nvrx.rank"))
                timestamp = int(log.get("timeUnixNano") or log.get("observedTimeUnixNano") or 0)
                if timestamp <= 0:
                    raise ValueError(f"{path}: log record has no timestamp")
                body = _value(log.get("body", {}))
                severity = str(log.get("severityText") or "LOG")
                args = {
                    "body": body,
                    "otel.trace_id": str(log.get("traceId") or ""),
                    "otel.span_id": str(log.get("spanId") or ""),
                    "otel.severity.number": log.get("severityNumber", 0),
                    **scope_args,
                    **log_args,
                }
                writer.event(
                    {
                        "ph": "I",
                        "name": f"{severity}: {str(body)[:120]}",
                        "cat": f"otel.log.{severity.lower()}",
                        "pid": pid,
                        "tid": tid,
                        "ts": writer.micros(timestamp),
                        "s": "t",
                        "args": args,
                    },
                )
                count += 1
    return count


def convert_records(records: Sequence[tuple[Path, dict]], output: Path) -> dict[str, int]:
    for path, record in records:
        if not any(signal in record for signal in _SIGNALS):
            raise ValueError(f"{path}: unknown OTLP record type")
    origin_ns = _origin(records)
    stats = {"spans": 0, "metrics": 0, "logs": 0, "tracks": 0}
    seen: set[tuple[str, str]] = set()
    with output.open("w", encoding="utf-8") as handle:
        writer = _Writer(handle, origin_ns)
        tracks = _Tracks(writer)
        for path, record in records:
            stats["spans"] += _emit_spans(writer, tracks, path, record, seen)
            stats["metrics"] += _emit_metrics(writer, tracks, path, record)
            stats["logs"] += _emit_logs(writer, tracks, path, record)
        stats["tracks"] = tracks.count
        writer.finish()
    return stats


def convert(paths: Sequence[Path], output: Path) -> dict[str, int]:
    return convert_records(list(_records(paths)), output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert mixed OTLP JSONL telemetry to Perfetto-compatible Chrome Trace JSON.")
    parser.add_argument("inputs", nargs="+", type=Path, help="OTLP JSONL input files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .json file")
    args = parser.parse_args(argv)
    stats = convert(args.inputs, args.output)
    print(  # noqa: T201
        f"wrote {args.output}: {stats['spans']} spans, {stats['logs']} logs, "
        f"{stats['metrics']} metric points on {stats['tracks']} tracks",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
