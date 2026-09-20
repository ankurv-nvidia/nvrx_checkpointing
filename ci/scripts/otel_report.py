#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

SAVE_SPAN = "ckpt.save"
TRAINING_SERVICE = "nvrx-training"
_ATTR_PREFIX = "nvrx."
MEGATRON_PERF_METRICS = {
    "megatron.training.memory_allocated_gb": ("GB", False),
}
REQUIRED_PERF_METRICS = frozenset(
    {"bw_mbps", "wr_tnsr_us", "total_written_bytes", *MEGATRON_PERF_METRICS},
)
logger = logging.getLogger("native_ci")
_MIN_SOCKET_RANKS = 4
_CHECKPOINT_TIMING_SPANS = {
    "megatron.checkpoint.exposed_save",
    "megatron.checkpoint.load",
}
_ANALYSIS_NAME = "native_ckpt_otel.report.json"


def _otlp_value(value: dict) -> object:
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(item) for item in value["arrayValue"].get("values", [])]
    return None


def _iter_records(
    paths: list[str | Path],
    *,
    tolerate_truncated_tail: bool = False,
) -> Iterator[tuple[Path, dict]]:
    for path in paths:
        source = Path(path)
        with source.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as error:
                    if tolerate_truncated_tail and not any(remaining.strip() for remaining in handle):
                        logger.warning("ignoring truncated final OTLP record in %s", source)
                        break
                    raise ValueError(f"{source}:{line_number}: invalid OTLP JSON") from error
                yield source, obj


def _attributes(record: dict) -> dict:
    return {attr["key"]: _otlp_value(attr["value"]) for attr in record.get("attributes", [])}


def _iter_spans(records: list[tuple[Path, dict]]) -> Iterator[dict]:
    seen: set[tuple[str, str]] = set()
    for _path, obj in records:
        for resource_spans in obj.get("resourceSpans", []):
            resource = _attributes(resource_spans.get("resource", {}))
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    identity = str(span.get("traceId", "")), str(span.get("spanId", ""))
                    if all(identity) and identity in seen:
                        continue
                    if all(identity):
                        seen.add(identity)
                    yield {
                        "name": span.get("name", ""),
                        "trace_id": identity[0],
                        "span_id": identity[1],
                        "resource": resource,
                        "attributes": _attributes(span),
                        "start_time_unix_nano": int(span.get("startTimeUnixNano", 0)),
                        "end_time_unix_nano": int(span.get("endTimeUnixNano", 0)),
                    }


def _iter_log_records(records: list[tuple[Path, dict]]) -> Iterator[dict]:
    for _path, obj in records:
        for resource_logs in obj.get("resourceLogs", []):
            resource = _attributes(resource_logs.get("resource", {}))
            for scope_logs in resource_logs.get("scopeLogs", []):
                for record in scope_logs.get("logRecords", []):
                    yield {
                        "name": record.get("eventName", ""),
                        "resource": resource,
                        "attributes": _attributes(record),
                        "time_unix_nano": int(
                            record.get("timeUnixNano", record.get("observedTimeUnixNano", 0)),
                        ),
                    }


def _iter_metric_samples(records: list[tuple[Path, dict]]) -> Iterator[dict]:
    for _path, obj in records:
        for resource_metrics in obj.get("resourceMetrics", []):
            resource = _attributes(resource_metrics.get("resource", {}))
            for scope_metrics in resource_metrics.get("scopeMetrics", []):
                for metric in scope_metrics.get("metrics", []):
                    for kind in ("gauge", "sum", "histogram"):
                        for point in metric.get(kind, {}).get("dataPoints", []):
                            count = int(point.get("count", 0)) if kind == "histogram" else 1
                            value = (
                                float(point["sum"]) / count
                                if kind == "histogram" and count
                                else point.get("asDouble", point.get("asInt"))
                            )
                            if value is not None:
                                yield {
                                    "name": metric.get("name", ""),
                                    "resource": resource,
                                    "attributes": _attributes(point),
                                    "value": float(value),
                                    "kind": kind,
                                    "count": count,
                                    "min": float(point["min"]) if "min" in point else None,
                                    "max": float(point["max"]) if "max" in point else None,
                                    "time_unix_nano": int(point.get("timeUnixNano", 0)),
                                }


class OtelData:
    def __init__(self, paths: list[str | Path], *, tolerate_truncated_tail: bool = False) -> None:
        self.paths = [Path(path) for path in paths]
        inputs = list(self.paths)
        known = set(inputs)
        for path in self.paths:
            logs = Path(f"{path}.logs")
            if logs.is_file() and logs not in known:
                inputs.append(logs)
                known.add(logs)
        self.records = list(_iter_records(inputs, tolerate_truncated_tail=tolerate_truncated_tail))
        self.spans = list(_iter_spans(self.records))
        self.log_records = list(_iter_log_records(self.records))
        self.metric_samples = list(_iter_metric_samples(self.records))


def _data(source: OtelData | list[str | Path]) -> OtelData:
    return source if isinstance(source, OtelData) else OtelData(source)


def iter_spans(source: OtelData | list[str | Path]) -> Iterator[dict]:
    yield from _data(source).spans


def iter_log_records(source: OtelData | list[str | Path]) -> Iterator[dict]:
    yield from _data(source).log_records


def iter_span_resources(source: OtelData | list[str | Path], name_prefix: str) -> Iterator[dict]:
    for span in iter_spans(source):
        if span["name"].startswith(name_prefix):
            yield span["resource"]


def iter_span_samples(source: OtelData | list[str | Path]) -> Iterator[dict]:
    for span in iter_spans(source):
        if span["name"] != SAVE_SPAN:
            continue
        sample = {
            key[len(_ATTR_PREFIX) :]: value
            for key, value in span["attributes"].items()
            if key.startswith(_ATTR_PREFIX)
        }
        if (ci_phase := span["resource"].get("nvrx.ci_phase")) is not None:
            sample["ci_phase"] = ci_phase
        if (run_id := span["resource"].get("nemo.run.id")) is not None:
            sample["run_id"] = run_id
        sample["start_time_unix_nano"] = span["start_time_unix_nano"]
        sample["end_time_unix_nano"] = span["end_time_unix_nano"]
        yield sample


def iter_metric_samples(source: OtelData | list[str | Path]) -> Iterator[dict]:
    yield from _data(source).metric_samples


def collect_megatron_checkpoint_spans(source: OtelData | list[str | Path], run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(source)
        if span["name"].startswith("megatron.checkpoint.")
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_checkpoint_spans(source: OtelData | list[str | Path], run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(source)
        if (span["name"].startswith("megatron.checkpoint.") or span["name"] == "nvrx.checkpoint.wait_previous")
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_spans(source: OtelData | list[str | Path], name: str, run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(source)
        if span["name"] == name
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_metrics(source: OtelData | list[str | Path], name_prefix: str, run_id: str | None = None) -> list[dict]:
    return [
        sample
        for sample in iter_metric_samples(source)
        if sample["name"].startswith(name_prefix)
        and (run_id is None or str(sample["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_resource_metrics(source: OtelData | list[str | Path], run_id: str | None = None) -> list[dict]:
    return collect_metrics(source, "nvrx.checkpoint.run.", run_id)


def collect_megatron_training_metrics(source: OtelData | list[str | Path], run_id: str | None = None) -> list[dict]:
    return collect_metrics(source, "megatron.training.", run_id)


def collect_process_memory(source: OtelData | list[str | Path]) -> list[dict]:
    """Collect daemon and training process-memory events from OTLP logs."""
    samples = []
    for event in iter_log_records(source):
        if event["name"] != "process.memory":
            continue
        attributes, resource = event["attributes"], event["resource"]
        host = str(resource.get("host.name", "unknown-host"))
        pid = int(resource.get("process.pid", -1))
        rank = attributes.get("nvrx.rank", resource.get("dl.rank"))
        timestamp = event["time_unix_nano"]
        if attributes.get("nvrx.memory.rss_bytes") is None or timestamp <= 0:
            continue
        samples.append(
            {
                "timestamp_ns": timestamp,
                "host": host,
                "service": str(resource.get("service.name", "unknown-service")),
                "pid": pid,
                "rank": int(rank) if rank is not None else None,
                "trigger": str(attributes.get("nvrx.memory.trigger", "")),
                **{
                    name: int(attributes.get(f"nvrx.memory.{name}", 0))
                    for name in (
                        "rss_bytes",
                        "peak_rss_bytes",
                        "anonymous_rss_bytes",
                        "file_rss_bytes",
                        "shmem_rss_bytes",
                        "locked_bytes",
                        "pinned_bytes",
                        "swap_bytes",
                    )
                },
            },
        )
    return samples


def process_memory_metrics(samples: list[dict]) -> dict:
    """Summarize checkpoint-daemon high-water RSS by node."""
    daemon: dict[str, int] = {}
    for sample in samples:
        if sample["service"] == TRAINING_SERVICE:
            continue
        daemon[sample["host"]] = max(sample["peak_rss_bytes"], daemon.get(sample["host"], 0))
    result = {}
    if daemon:
        measured = metric([daemon[host] for host in sorted(daemon)], "B", higher_is_better=False)
        measured["per_node"] = dict(sorted(daemon.items()))
        result["checkpoint_daemon_peak_rss_bytes"] = measured
    return result


def _process_memory_series(daemon_samples: list[dict], training_samples: list[dict]) -> list[tuple[str, tuple, list[dict]]]:
    series: list[tuple[str, tuple, list[dict]]] = []
    daemon_series: dict[tuple[str, int], list[dict]] = {}
    for sample in daemon_samples:
        daemon_series.setdefault((sample["host"], sample["pid"]), []).append(sample)
    for key, values in sorted(daemon_series.items()):
        series.append(("daemon", key, values))

    training_series: dict[tuple[str, int, int], list[dict]] = {}
    for sample in training_samples:
        training_series.setdefault((sample["host"], sample["rank"], sample["pid"]), []).append(sample)
    for key, values in sorted(training_series.items()):
        series.append(("training", key, values))
    return series


def _process_memory_color(role: str, index: int, count: int) -> str:
    lightness = 32 if count <= 1 else 24 + 48 * index / (count - 1)
    hue = 215 if role == "daemon" else 0
    return f"hsl({hue} 75% {lightness:.1f}%)"


def _process_memory_svg(series: list[tuple[str, tuple, list[dict]]], samples: list[dict]) -> str:
    start = min(sample["timestamp_ns"] for sample in samples)
    stop = max(sample["timestamp_ns"] for sample in samples)
    duration = max((stop - start) / 1e9, 1.0)
    maximum = max((sample["rss_bytes"] for sample in samples), default=1)

    width, plot_height = 1900, 620
    left, right, top, bottom = 90, 50, 60, 70
    plot_width = width - left - right
    legend_columns = min(6, max(1, len(series)))
    legend_rows = math.ceil(len(series) / legend_columns)
    legend_top = top + plot_height + bottom
    height = legend_top + legend_rows * 18 + 40

    def x(timestamp: int) -> float:
        return left + (((timestamp - start) / 1e9) / duration) * plot_width

    def y(value: int) -> float:
        return top + plot_height - (value / maximum) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="20">Host memory — daemon and training processes</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        py = y(value)
        elements.extend(
            (
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" stroke="#e5e7eb"/>',
                (f'<text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">{value / (1 << 30):.1f}</text>'),
            ),
        )
    for tick in range(6):
        seconds = duration * tick / 5
        px = left + plot_width * tick / 5
        elements.extend(
            (
                f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_height}" stroke="#d1d5db"/>',
                (f'<text x="{px:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{seconds:.1f}</text>'),
            ),
        )
    elements.extend(
        (
            (f'<text x="18" y="{top + plot_height / 2}" '
            f'transform="rotate(-90 18 {top + plot_height / 2})" '
            'font-family="sans-serif" font-size="12">RSS (GiB)</text>'),
            (f'<text x="{left + plot_width / 2}" y="{top + plot_height + 45}" text-anchor="middle" '
            'font-family="sans-serif" font-size="12">seconds</text>'),
        ),
    )

    role_counts = {
        role: sum(series_role == role for series_role, _key, _values in series)
        for role in ("daemon", "training")
    }
    role_indices = {"daemon": 0, "training": 0}
    legend_column_width = plot_width / legend_columns
    for index, (role, key, values) in enumerate(series):
        values.sort(key=lambda sample: sample["timestamp_ns"])
        shade = _process_memory_color(role, role_indices[role], role_counts[role])
        role_indices[role] += 1
        points = " ".join(f"{x(sample['timestamp_ns']):.1f},{y(sample['rss_bytes']):.1f}" for sample in values)
        elements.append(f'<polyline points="{points}" fill="none" stroke="{shade}" stroke-width="1.5"/>')
        if role == "daemon":
            host, pid = key
            label = f"daemon {host} pid {pid}"
        else:
            host, rank, pid = key
            label = f"training {host} rank {rank} pid {pid}"
        peak = max(sample["rss_bytes"] for sample in values) / (1 << 30)
        column, row = divmod(index, legend_rows)
        legend_x = left + column * legend_column_width
        legend_y = legend_top + row * 18
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{shade}" stroke-width="2"/>',
                (f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="10">'
                f'{html.escape(label)} max {peak:.1f} GiB</text>'),
            ),
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_process_memory_charts(source: OtelData | list[str | Path], output_dir: Path) -> list[Path]:
    """Write one RSS timeline containing every daemon and training process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("process_memory_*.svg"):
        previous.unlink()

    samples = collect_process_memory(source)
    if not samples:
        return []
    training_samples = [sample for sample in samples if sample["service"] == TRAINING_SERVICE]
    daemon_samples = [sample for sample in samples if sample["service"] != TRAINING_SERVICE]

    output = output_dir / "process_memory_all.svg"
    output.write_text(_process_memory_svg(_process_memory_series(daemon_samples, training_samples), samples))
    return [output]


def collect_samples(
    source: OtelData | list[str | Path],
    ci_phase: str | None = None,
    run_id: str | None = None,
) -> list[dict]:
    samples = list(iter_span_samples(source))
    if ci_phase is not None:
        samples = [s for s in samples if str(s.get("ci_phase")) == str(ci_phase)]
    if run_id is not None:
        samples = [s for s in samples if str(s.get("run_id")) == run_id]
    return samples


def metric(
    samples: list[float | int],
    unit: str,
    higher_is_better: bool | None,
    ranks: list[int] | None = None,
) -> dict:
    result = {
        "samples": samples,
        "ranks": ranks,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "unit": unit,
        "higher_is_better": higher_is_better,
    }
    if ranks is not None:
        per_rank: dict[int, list[float | int]] = {}
        for rank, value in zip(ranks, samples):
            per_rank.setdefault(rank, []).append(value)
        rank_medians = {rank: statistics.median(values) for rank, values in sorted(per_rank.items())}
        across_ranks = list(rank_medians.values())
        result.update(
            rank_min=min(across_ranks),
            rank_median=statistics.median(across_ranks),
            rank_max=max(across_ranks),
            per_rank_median=rank_medians,
        )
    return result


def _checkpoint_timing_run_id(spans: list[dict], path: Path, run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    run_ids = {
        str(span["resource"]["nemo.run.id"])
        for span in spans
        if span["name"] in _CHECKPOINT_TIMING_SPANS
        and span["resource"].get("nemo.run.id") is not None
    }
    if len(run_ids) != 1:
        raise RuntimeError(f"expected one checkpoint timing run ID in {path}, found {sorted(run_ids)}")
    return run_ids.pop()


def _checkpoint_timing_report(
    spans: list[dict],
    path: Path,
    run_id: str | None,
    durability: tuple[list[float], list[int]],
) -> dict:
    by_checkpoint: dict[tuple[str, int | None], dict[int, list[float]]] = {}
    started: dict[tuple[str, int | None], int] = {}
    run_id = _checkpoint_timing_run_id(spans, path, run_id)
    for span in spans:
        name = span["name"]
        if str(span["resource"].get("nemo.run.id")) != run_id:
            continue
        if name not in _CHECKPOINT_TIMING_SPANS:
            continue
        start, end = span["start_time_unix_nano"], span["end_time_unix_nano"]
        if end <= start:
            raise RuntimeError(f"checkpoint timing span {name} has invalid timestamps in {path}")
        rank = int(span["resource"]["dl.rank"])
        iteration = span["attributes"].get("megatron.iteration")
        key = name, int(iteration) if iteration is not None else None
        by_checkpoint.setdefault(key, {}).setdefault(rank, []).append((end - start) / 1e6)
        started[key] = min(started.get(key, start), start)
    if not any(name == "megatron.checkpoint.exposed_save" for name, _iteration in by_checkpoint):
        raise RuntimeError(f"no megatron.checkpoint.exposed_save spans found for run {run_id!r} in {path}")
    timings = []
    for (name, iteration), by_rank in sorted(by_checkpoint.items(), key=lambda item: started[item[0]]):
        values = [statistics.median(samples) for samples in by_rank.values()]
        label = "load-checkpoint" if name.endswith(".load") else "save-checkpoint"
        if iteration is not None:
            label += f" iteration {iteration}"
        timings.append(
            {
                "label": label,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            },
        )
    durable, ranks = durability
    durability_report = None
    if durable:
        durability_report = {
            "min": min(durable),
            "max": max(durable),
            "median": statistics.median(durable),
            "mean": statistics.mean(durable),
            "spans": len(durable),
            "ranks": len(set(ranks)),
        }
    return {"run_id": run_id, "checkpoint_timings": timings, "checkpoint_durability": durability_report}


def log_checkpoint_analysis(analysis: dict, run: str) -> None:
    logger.info(f"{run}: checkpoint timings (min, max, median ms across ranks)")
    for timing in analysis["checkpoint_timings"]:
        logger.info(
            f"    {(timing['label'] + ' ').ljust(48, '.')}"
            f": ({timing['min']:.2f}, {timing['max']:.2f}, {timing['median']:.2f})",
        )
    if durability := analysis.get("checkpoint_durability"):
        logger.info(f"{run}: checkpoint durability (min, max, median ms across ranks and saves)")
        logger.info(
            f"    {'save -> durable '.ljust(48, '.')}"
            f": ({durability['min']:.2f}, {durability['max']:.2f}, {durability['median']:.2f})  "
            f"mean {durability['mean']:.2f} over {durability['spans']} span(s) "
            f"across {durability['ranks']} rank(s)",
        )


def log_checkpoint_timings(path: Path, run_id: str | None, run: str) -> None:
    if analysis := _load_analysis(path):
        log_checkpoint_analysis(analysis, run)
        return
    source = OtelData([path])
    checkpoint_spans = collect_checkpoint_spans(source, run_id)
    durability = checkpoint_durability_ms(checkpoint_spans)
    log_checkpoint_analysis(_checkpoint_timing_report(source.spans, path, run_id, durability), run)


def load_latest(results_dir: Path) -> dict[str, list]:
    empty = {
        "native_saves": [],
        "daemon_durable_spans": [],
        "megatron_spans": [],
        "megatron_metrics": [],
        "resource_metrics": [],
        "process_memory": [],
    }
    otel_root = results_dir / "otel"
    if not otel_root.is_dir():
        return empty
    by_cycle = {int(path.name[5:]): path for path in otel_root.glob("cycle*") if path.is_dir() and path.name[5:].isdigit()}
    newest = by_cycle[max(by_cycle)] if by_cycle else otel_root
    files = sorted(newest.glob("*.jsonl"))
    if not files:
        return empty
    source = OtelData(files)
    resource_metrics = collect_resource_metrics(source)
    run_id = os.environ.get("MEGATRON_OTEL_RUN_ID")
    if run_id is None and resource_metrics:
        latest = max(resource_metrics, key=lambda sample: sample["time_unix_nano"])
        run_id = str(latest["resource"].get("nemo.run.id"))
    return {
        "native_saves": collect_samples(source, run_id=run_id),
        "daemon_durable_spans": collect_spans(source, "ckpt.svr.durable", run_id),
        "megatron_spans": collect_checkpoint_spans(source, run_id),
        "megatron_metrics": collect_megatron_training_metrics(source, run_id),
        "resource_metrics": [
            sample for sample in resource_metrics if run_id is None or str(sample["resource"].get("nemo.run.id")) == run_id
        ],
        "process_memory": collect_process_memory(source),
    }


def native_spans_valid(samples: list[dict]) -> bool:
    required = {
        "session_id",
        "rank",
        "snapshot_id",
        "bw_mbps",
        "wr_tnsr_us",
        "snapshot_us",
        "total_us",
        "num_tensors",
        "total_d2h",
        "total_tensor_bytes",
        "total_written_bytes",
        "avg_d2h_rq",
        "avg_tnsr",
        "start_time_unix_nano",
        "end_time_unix_nano",
    }
    if not samples:
        logger.warning("[slurm_job:perf] WARNING no ckpt.save spans found in measured-cycle OTLP data")
        return False
    invalid = [sample for sample in samples if not required <= sample.keys()]
    if invalid:
        missing = sorted(required - invalid[0].keys())
        logger.warning(
            f"[slurm_job:perf] WARNING {len(invalid)} ckpt.save OTLP span(s) did not match the required schema; "
            f"first span is missing {missing}",
        )
    invalid_timestamps = [sample for sample in samples if sample.get("end_time_unix_nano", 0) <= sample.get("start_time_unix_nano", 0)]
    if invalid_timestamps:
        logger.warning(
            f"[slurm_job:perf] WARNING {len(invalid_timestamps)} ckpt.save OTLP span(s) have invalid timestamps",
        )
    return not invalid and not invalid_timestamps


def _match_checkpoint_finalizers(
    saves: list[tuple[int, int | None, dict]],
    finalizers: dict[int, list[tuple[int, list[int]]]],
) -> dict[int, int]:
    finalized: dict[int, int] = {}
    mismatches: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for rank in sorted({rank for rank, _call_idx, _span in saves}):
        pending = [
            (position, call_idx, span["start_time_unix_nano"])
            for position, (save_rank, call_idx, span) in enumerate(saves)
            if save_rank == rank
        ]
        expected = [int(call_idx) for _position, call_idx, _start in pending if call_idx is not None]
        if len(set(expected)) != len(expected):
            raise RuntimeError(f"Megatron OTLP save spans have duplicate nvrx.call_idx values for rank={rank}")
        completions = sorted(finalizers.get(rank, []))
        advertised = [(call_idx, end) for end, indices in completions for call_idx in indices]
        if not advertised and not expected:
            continue
        if len(expected) == len(pending) and len(advertised) == len(pending) and {
            call_idx for call_idx, _end in advertised
        } == set(expected):
            end_by_call_idx = dict(advertised)
            finalized.update({position: end_by_call_idx[call_idx] for position, call_idx, _start in pending})
            continue
        if not advertised:
            raise RuntimeError(f"Megatron OTLP save span has no finalizer for rank={rank}")
        if len(advertised) != len(pending):
            raise RuntimeError(
                f"Megatron OTLP finalizer count mismatch for rank={rank}: expected {len(pending)}, found {len(advertised)}",
            )
        if expected:
            key = tuple(expected), tuple(call_idx for call_idx, _end in advertised)
            mismatches.setdefault(key, []).append(rank)
        for (position, _call_idx, _start), (_advertised_idx, end) in zip(
            sorted(pending, key=lambda item: item[2]),
            sorted(advertised, key=lambda item: item[1]),
        ):
            finalized[position] = end
    for (expected, found), ranks in mismatches.items():
        rendered_ranks = f"{ranks[0]}-{ranks[-1]} ({len(ranks)} total)"
        logger.warning(
            f"[slurm_job:perf] WARNING Megatron OTLP finalizer call indices mismatch for ranks={rendered_ranks}; "
            f"expected {list(expected)}, found {list(found)}; pairing in FIFO order",
        )
    return finalized


def checkpoint_durability_ms(spans: list[dict]) -> tuple[list[float], list[int]]:
    saves: list[tuple[int, int | None, dict]] = []
    finalizers: dict[int, list[tuple[int, list[int]]]] = {}
    for span in spans:
        rank = int(span["resource"].get("dl.rank", -1))
        if span["name"] == "megatron.checkpoint.save":
            call_idx = span["attributes"].get("nvrx.call_idx")
            saves.append((rank, int(call_idx) if call_idx is not None else None, span))
            continue
        if span["name"] not in {
            "megatron.checkpoint.save.finalize",
            "megatron.checkpoint.exit_finalize",
            "nvrx.checkpoint.wait_previous",
        }:
            continue
        indices = span["attributes"].get("nvrx.call_idx", [])
        if not isinstance(indices, list):
            indices = [indices]
        if indices:
            finalizers.setdefault(rank, []).append(
                (span["end_time_unix_nano"], [int(index) for index in indices]),
            )
    finalized = _match_checkpoint_finalizers(saves, finalizers)
    elapsed = []
    ranks = []
    for position, (rank, _call_idx, span) in enumerate(saves):
        end = finalized.get(position, span["end_time_unix_nano"])
        if end < span["end_time_unix_nano"] or end <= span["start_time_unix_nano"]:
            raise RuntimeError("Megatron OTLP save span has invalid timestamps")
        elapsed.append((end - span["start_time_unix_nano"]) / 1e6)
        ranks.append(rank)
    return elapsed, ranks


def _native_metrics(samples: list[dict], gpus_per_node: int | None) -> dict:
    rows = [sample for sample in samples if {"rank", "bw_mbps", "wr_tnsr_us", "total_written_bytes"} <= sample.keys()]
    ranks = [int(sample["rank"]) for sample in rows]
    bandwidth = [round(float(sample["bw_mbps"])) for sample in rows]
    write = [int(sample["wr_tnsr_us"]) for sample in rows]
    written = [int(sample["total_written_bytes"]) for sample in rows]
    if not ranks:
        return {}
    metrics = {
        "bw_mbps": metric(bandwidth, "MB/s", higher_is_better=True, ranks=ranks),
        "wr_tnsr_us": metric(write, "us", higher_is_better=False, ranks=ranks),
        "total_written_bytes": metric(written, "B", higher_is_better=None, ranks=ranks),
    }
    unique = sorted(set(ranks))
    per_rank = [statistics.mean(value for rank_value, value in zip(ranks, bandwidth) if rank_value == rank) for rank in unique]
    if gpus_per_node and len(per_rank) == gpus_per_node and len(per_rank) >= _MIN_SOCKET_RANKS and len(per_rank) % 2 == 0:
        half = len(per_rank) // 2
        means = statistics.mean(per_rank[:half]), statistics.mean(per_rank[half:])
        metrics["bw_socket_imbalance"] = metric(
            [max(means) / min(means)],
            "ratio",
            higher_is_better=None,
        )
    return metrics


def _megatron_metrics(samples: list[dict]) -> dict:
    metrics = {}
    for name, (unit, higher_is_better) in MEGATRON_PERF_METRICS.items():
        latest: dict[int, dict] = {}
        for sample in samples:
            rank = int(sample["resource"].get("dl.rank", -1))
            if sample["name"] == name and rank >= 0 and sample["time_unix_nano"] >= latest.get(rank, {}).get("time_unix_nano", -1):
                latest[rank] = sample
        if latest:
            ranks = sorted(latest)
            metrics[name] = metric(
                [latest[rank]["value"] for rank in ranks],
                unit,
                higher_is_better,
                ranks,
            )
    return metrics


def _resource_metrics(samples: list[dict]) -> dict:
    metrics = {}
    by_name: dict[str, dict[int, float]] = {}
    for sample in samples:
        rank = int(sample["resource"].get("dl.rank", -1))
        by_name.setdefault(sample["name"], {})[rank] = sample["value"]
    gpu = by_name.get("nvrx.checkpoint.run.gpu_peak_bytes", {})
    if gpu:
        metrics["gpu_peak_bytes"] = metric(
            [round(value) for value in gpu.values()],
            "B",
            higher_is_better=False,
            ranks=list(gpu),
        )
    hbm = by_name.get("nvrx.checkpoint.run.gpu_hbm_peak_bytes", {})
    total = by_name.get("nvrx.checkpoint.run.gpu_total_bytes", {})
    if hbm:
        metrics["gpu_hbm_peak_bytes"] = metric(
            [round(value) for value in hbm.values()],
            "B",
            higher_is_better=False,
            ranks=list(hbm),
        )
    hbm_pct = {rank: value * 100 / total[rank] for rank, value in hbm.items() if total.get(rank, 0) > 0}
    if hbm_pct:
        metrics["gpu_hbm_peak_pct"] = metric(
            list(hbm_pct.values()),
            "%",
            higher_is_better=False,
            ranks=list(hbm_pct),
        )
    cpu = by_name.get("nvrx.checkpoint.run.cpu_rss_bytes", {})
    if cpu:
        metrics["cpu_rss_bytes"] = metric(
            [round(value) for value in cpu.values()],
            "B",
            higher_is_better=False,
            ranks=list(cpu),
        )
    return metrics


def metrics_from_data(
    data: dict[str, list],
    gpus_per_node: int | None = None,
    durability: tuple[list[float], list[int]] | None = None,
) -> dict:
    metrics = _native_metrics(data["native_saves"], gpus_per_node)
    durable, durable_ranks = durability if durability is not None else checkpoint_durability_ms(data["megatron_spans"])
    if durable:
        metrics["ckpt_durable_ms"] = metric(
            durable,
            "ms",
            higher_is_better=False,
            ranks=durable_ranks,
        )
    daemon_durable = [
        (span["end_time_unix_nano"] - span["start_time_unix_nano"]) / 1e6
        for span in data["daemon_durable_spans"]
        if span["end_time_unix_nano"] > span["start_time_unix_nano"]
    ]
    if daemon_durable:
        metrics["ckpt_daemon_durable_ms"] = metric(
            daemon_durable,
            "ms",
            higher_is_better=False,
        )
    metrics.update(_megatron_metrics(data["megatron_metrics"]))
    metrics.update(_resource_metrics(data["resource_metrics"]))
    metrics.update(process_memory_metrics(data.get("process_memory", [])))
    return metrics


def _processed(analysis: dict, native: bool) -> dict:
    valid = bool(analysis.get("native_spans_valid")) if native else True
    return {
        "source": "otlp",
        "metrics": analysis["metrics"],
        "validation": (
            {
                "name": "ckpt_save_spans",
                "valid": valid,
                "measured": "complete" if valid else "missing_or_invalid",
                "baseline": "complete",
                "report_field": "ckpt_save_spans_valid",
            }
            if native
            else None
        ),
    }


def analysis_path(path: Path) -> Path:
    return path.with_name(_ANALYSIS_NAME)


def write_analysis(path: Path, analysis: dict) -> None:
    stat = path.stat()
    analysis = {**analysis, "otel_file": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
    analysis_path(path).write_text(json.dumps(analysis, separators=(",", ":")))


def _load_analysis(path: Path) -> dict | None:
    report = analysis_path(path)
    if not path.is_file() or not report.is_file():
        return None
    analysis = json.loads(report.read_text())
    stat = path.stat()
    return analysis if analysis.get("otel_file") == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} else None


def _complete_run_cpu_rss(results_dir: Path) -> dict | None:
    """Return each training rank's largest peak RSS across all completed phases."""
    root = results_dir / "otel"
    sources = [root / "native_ckpt_otel.jsonl", *sorted(root.glob("*/native_ckpt_otel.jsonl"))]
    per_rank: dict[int, int] = {}
    for source in sources:
        analysis = _load_analysis(source)
        measured = analysis.get("metrics", {}).get("cpu_rss_bytes") if analysis else None
        if not measured or measured.get("ranks") is None:
            continue
        ranks = measured["ranks"]
        samples = measured["samples"]
        if len(ranks) != len(samples):
            raise ValueError("cpu_rss_bytes ranks and samples differ in length")
        for rank, value in zip(ranks, samples):
            rank_id = int(rank)
            per_rank[rank_id] = max(round(value), per_rank.get(rank_id, 0))
    if not per_rank:
        return None
    ranks = sorted(per_rank)
    return metric([per_rank[rank] for rank in ranks], "B", higher_is_better=False, ranks=ranks)


def _load_latest_analysis(results_dir: Path) -> dict | None:
    root = results_dir / "otel"
    if not root.is_dir():
        return None
    by_cycle = {int(path.name[5:]): path for path in root.glob("cycle*") if path.is_dir() and path.name[5:].isdigit()}
    newest = by_cycle[max(by_cycle)] if by_cycle else root
    return _load_analysis(newest / "native_ckpt_otel.jsonl")


def complete_run_process_memory_peaks(results_dir: Path) -> dict:
    """Combine process-memory peaks from every phase's collected OTLP."""
    per_process: dict[str, int] = {}
    per_node: dict[str, int] = {}
    root = results_dir / "otel"
    sources = [root / "native_ckpt_otel.jsonl", *sorted(root.glob("*/native_ckpt_otel.jsonl"))]
    for source in sources:
        analysis = _load_analysis(source)
        metrics = analysis.get("metrics", {}) if analysis else {}
        if not analysis and source.is_file():
            metrics = process_memory_metrics(collect_process_memory([source]))
        for key, value in metrics.get("training_process_peak_rss_bytes", {}).get("per_process", {}).items():
            per_process[key] = max(int(value), per_process.get(key, 0))
        for key, value in metrics.get("checkpoint_daemon_peak_rss_bytes", {}).get("per_node", {}).items():
            per_node[key] = max(int(value), per_node.get(key, 0))
    combined = {}
    if per_process:
        measured = metric(list(per_process.values()), "B", higher_is_better=False)
        measured["per_process"] = dict(sorted(per_process.items()))
        combined["training_process_peak_rss_bytes"] = measured
    if per_node:
        measured = metric(list(per_node.values()), "B", higher_is_better=False)
        measured["per_node"] = dict(sorted(per_node.items()))
        combined["checkpoint_daemon_peak_rss_bytes"] = measured
    return combined


def post_process(results_dir: Path, gpus_per_node: int, native: bool) -> dict:
    if analysis := _load_latest_analysis(results_dir):
        processed = _processed(analysis, native)
    else:
        data = load_latest(results_dir)
        valid = native_spans_valid(data["native_saves"]) if native else True
        processed = _processed({"metrics": metrics_from_data(data, gpus_per_node), "native_spans_valid": valid}, native)
    if cpu_rss := _complete_run_cpu_rss(results_dir):
        processed["metrics"]["cpu_rss_bytes"] = cpu_rss
    processed["metrics"].update(complete_run_process_memory_peaks(results_dir))
    return processed


def validate_collected(  # noqa: C901, PLR0912, PLR0913, PLR0917
    source: Path | OtelData,
    run_id: str,
    nodes: int,
    gpus_per_node: int,
    provider: str,
    expected_checkpoints: int,
) -> dict:
    path = source.paths[0] if isinstance(source, OtelData) else source
    if not isinstance(source, OtelData) and not path.is_file():
        raise RuntimeError(f"Megatron emitted no collected OTLP file: {path}")
    data = source if isinstance(source, OtelData) else OtelData([path])
    resources = [resource for resource in iter_span_resources(data, "megatron.") if str(resource.get("nemo.run.id")) == run_id]
    ranks = {int(resource["dl.rank"]) for resource in resources if resource.get("dl.rank") is not None}
    expected = set(range(nodes * gpus_per_node))
    if ranks != expected:
        raise RuntimeError(f"Megatron OTLP exporter ranks mismatch: expected {sorted(expected)}, found {sorted(ranks)} in {path}")
    megatron_metrics = collect_megatron_training_metrics(data, run_id)
    for name in MEGATRON_PERF_METRICS:
        matching = [sample for sample in megatron_metrics if sample["name"] == name]
        metric_ranks = {int(sample["resource"].get("dl.rank", -1)) for sample in matching}
        if metric_ranks != expected:
            raise RuntimeError(
                f"Megatron training metric {name} ranks mismatch: expected {sorted(expected)}, found {sorted(metric_ranks)} in {path}",
            )
        if any(sample["time_unix_nano"] <= 0 or not math.isfinite(sample["value"]) for sample in matching):
            raise RuntimeError(f"Megatron training metric {name} has invalid OTLP points in {path}")
    resource_metrics = collect_resource_metrics(data, run_id)
    expected_metric_ranks = set(range(nodes * gpus_per_node))
    required_metrics = {
        "nvrx.checkpoint.run.gpu_peak_bytes",
        "nvrx.checkpoint.run.gpu_hbm_peak_bytes",
        "nvrx.checkpoint.run.gpu_total_bytes",
        "nvrx.checkpoint.run.cpu_rss_bytes",
    }
    for name in required_metrics:
        matching = [sample for sample in resource_metrics if sample["name"] == name]
        metric_ranks = {int(sample["resource"]["dl.rank"]) for sample in matching}
        if metric_ranks != expected_metric_ranks:
            raise RuntimeError(
                f"checkpoint resource metric {name} ranks mismatch: expected {sorted(expected_metric_ranks)}, "
                f"found {sorted(metric_ranks)} in {path}",
            )
        metric_providers = {str(sample["resource"].get("checkpoint.provider")) for sample in matching}
        if metric_providers != {provider}:
            raise RuntimeError(
                f"checkpoint resource metric {name} provider mismatch: expected {provider!r}, found {sorted(metric_providers)} in {path}",
            )
    checkpoint_spans = collect_checkpoint_spans(data, run_id)
    save_counts = {
        rank: sum(
            span["name"] == "megatron.checkpoint.save" and int(span["resource"].get("dl.rank", -1)) == rank for span in checkpoint_spans
        )
        for rank in sorted(expected)
    }
    expected_save_counts = dict.fromkeys(sorted(expected), expected_checkpoints)
    if save_counts != expected_save_counts:
        raise RuntimeError(
            f"Megatron checkpoint save span counts mismatch: expected {expected_save_counts}, "
            f"found {save_counts} for run {run_id!r} in {path}",
        )
    durability = checkpoint_durability_ms(checkpoint_spans)
    native_spans = collect_samples(data, run_id=run_id)
    valid_native_spans = native_spans_valid(native_spans) if provider == "native" else False
    if provider == "native":
        if not valid_native_spans:
            raise RuntimeError(f"native checkpoint OTLP spans are missing or invalid in {path}")
        snapshots: dict[tuple[int, int], set[int]] = {}
        for span in native_spans:
            key = int(span["session_id"]), int(span["snapshot_id"])
            snapshots.setdefault(key, set()).add(int(span["rank"]))
        incomplete = {key: found for key, found in snapshots.items() if found != expected_metric_ranks}
        if incomplete:
            key, found = next(iter(incomplete.items()))
            raise RuntimeError(
                f"native checkpoint OTLP span ranks mismatch for session/snapshot {key}: "
                f"expected {sorted(expected_metric_ranks)}, found {sorted(found)} in {path}",
            )
        durable_spans = collect_spans(data, "ckpt.svr.durable", run_id)
        durable_ids = {
            (int(span["attributes"].get("nvrx.session_id", -1)), int(span["attributes"].get("nvrx.snapshot_id", -1)))
            for span in durable_spans
            if span["end_time_unix_nano"] > span["start_time_unix_nano"]
        }
        if durable_ids != set(snapshots):
            raise RuntimeError(
                f"native checkpoint durable OTLP spans mismatch: expected {sorted(snapshots)}, found {sorted(durable_ids)} in {path}",
            )
    report_data = {
        "native_saves": native_spans,
        "daemon_durable_spans": collect_spans(data, "ckpt.svr.durable", run_id),
        "megatron_spans": checkpoint_spans,
        "megatron_metrics": megatron_metrics,
        "resource_metrics": resource_metrics,
        "process_memory": collect_process_memory(data),
    }
    return {
        "source": "otlp",
        "run_id": run_id,
        "provider": provider,
        "native_spans_valid": valid_native_spans,
        "metrics": metrics_from_data(report_data, gpus_per_node, durability),
        **_checkpoint_timing_report(data.spans, path, run_id, durability),
    }


# Entries define the unit and whether larger values are better.
_SUMMARY_METRICS = (
    ("bw_mbps", "MB/s", True),
    ("wr_tnsr_us", "us", False),
    ("snapshot_us", "us", False),
    ("total_us", "us", False),
    ("total_written_bytes", "B", None),
    ("num_tensors", "1", None),
)


def summarize(samples: list[dict]) -> dict:
    out: dict = {}
    for name, unit, higher_is_better in _SUMMARY_METRICS:
        pairs = [(int(s["rank"]), float(s[name])) for s in samples if name in s]
        if not pairs:
            continue
        values = [v for _, v in pairs]
        per_rank: dict[int, list[float]] = {}
        for rank, value in pairs:
            per_rank.setdefault(rank, []).append(value)
        out[name] = {
            "unit": unit,
            "higher_is_better": higher_is_better,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n_samples": len(values),
            "per_rank_mean": {rank: statistics.mean(vs) for rank, vs in sorted(per_rank.items())},
        }
    return out


def _expand(paths: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in paths:
        candidate = Path(pattern)
        root = Path(candidate.anchor) if candidate.is_absolute() else Path()
        glob_pattern = str(candidate.relative_to(candidate.anchor)) if candidate.is_absolute() else pattern
        matches = sorted(root.glob(glob_pattern))
        if matches:
            files.extend(map(str, matches))
        elif candidate.exists():
            files.append(pattern)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze checkpoint OTLP-JSONL telemetry.")
    parser.add_argument("paths", nargs="+", help="OTLP-JSONL span file(s) or glob(s)")
    parser.add_argument("--ci-phase", default=None, help="keep only spans whose nvrx.ci_phase resource attr matches")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    files = _expand(args.paths)
    if not files:
        print("otel_report: no input files matched", file=sys.stderr)  # noqa: T201
        return 2

    samples = collect_samples(files, ci_phase=args.ci_phase)
    if not samples:
        print("otel_report: no ckpt.save spans found", file=sys.stderr)  # noqa: T201
        return 1

    summary = summarize(samples)
    if args.json:
        print(json.dumps({"n_saves": len(samples), "metrics": summary}, indent=2))  # noqa: T201
    else:
        print(f"ckpt.save spans: {len(samples)} across ranks {sorted({int(s['rank']) for s in samples})}")  # noqa: T201
        for name, data in summary.items():
            per_rank = ", ".join(f"r{r}={v:.2f}" for r, v in data["per_rank_mean"].items())
            print(  # noqa: T201
                f"  {name:22s} mean={data['mean']:.2f} {data['unit']:5s} "
                f"[min={data['min']:.2f} max={data['max']:.2f} n={data['n_samples']}] ({per_rank})",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
