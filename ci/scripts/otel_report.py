#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
from __future__ import annotations

import argparse
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
_ATTR_PREFIX = "nvrx."
MEGATRON_PERF_METRICS = {
    "megatron.training.step_duration_ms": ("ms", False),
    "megatron.training.throughput_tflops": ("TFLOP/s/GPU", True),
    "megatron.training.tokens_per_sec": ("tokens/s", True),
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


def _iter_objects(paths: list[str]) -> Iterator[dict]:
    for path in paths:
        with Path(path).open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid OTLP JSON") from error
                yield obj


def _attributes(record: dict) -> dict:
    return {attr["key"]: _otlp_value(attr["value"]) for attr in record.get("attributes", [])}


def iter_spans(paths: list[str]) -> Iterator[dict]:
    seen: set[tuple[str, str]] = set()
    for obj in _iter_objects(paths):
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


def iter_span_resources(paths: list[str], name_prefix: str) -> Iterator[dict]:
    for span in iter_spans(paths):
        if span["name"].startswith(name_prefix):
            yield span["resource"]


def iter_span_samples(paths: list[str]) -> Iterator[dict]:
    for span in iter_spans(paths):
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


def iter_metric_samples(paths: list[str]) -> Iterator[dict]:
    for obj in _iter_objects(paths):
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


def collect_megatron_checkpoint_spans(paths: list[str], run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(paths)
        if span["name"].startswith("megatron.checkpoint.")
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_checkpoint_spans(paths: list[str], run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(paths)
        if (span["name"].startswith("megatron.checkpoint.") or span["name"] == "nvrx.checkpoint.wait_previous")
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_spans(paths: list[str], name: str, run_id: str | None = None) -> list[dict]:
    return [
        span
        for span in iter_spans(paths)
        if span["name"] == name
        and (run_id is None or str(span["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_metrics(paths: list[str], name_prefix: str, run_id: str | None = None) -> list[dict]:
    return [
        sample
        for sample in iter_metric_samples(paths)
        if sample["name"].startswith(name_prefix)
        and (run_id is None or str(sample["resource"].get("nemo.run.id")) == run_id)
    ]


def collect_resource_metrics(paths: list[str], run_id: str | None = None) -> list[dict]:
    return collect_metrics(paths, "nvrx.checkpoint.run.", run_id)


def collect_megatron_training_metrics(paths: list[str], run_id: str | None = None) -> list[dict]:
    return collect_metrics(paths, "megatron.training.", run_id)


def collect_samples(paths: list[str], ci_phase: str | None = None, run_id: str | None = None) -> list[dict]:
    samples = list(iter_span_samples(paths))
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


def log_checkpoint_timings(path: Path, run_id: str | None, run: str) -> None:
    by_checkpoint: dict[tuple[str, int | None], dict[int, list[float]]] = {}
    started: dict[tuple[str, int | None], int] = {}
    checkpoint_spans = []
    spans = list(iter_spans([str(path)]))
    run_id = _checkpoint_timing_run_id(spans, path, run_id)
    for span in spans:
        name = span["name"]
        if str(span["resource"].get("nemo.run.id")) != run_id:
            continue
        if name.startswith("megatron.checkpoint.") or name == "nvrx.checkpoint.wait_previous":
            checkpoint_spans.append(span)
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
    logger.info(f"{run}: checkpoint timings (min, max, median ms across ranks)")
    for (name, iteration), by_rank in sorted(by_checkpoint.items(), key=lambda item: started[item[0]]):
        values = [statistics.median(samples) for samples in by_rank.values()]
        label = "load-checkpoint" if name.endswith(".load") else "save-checkpoint"
        if iteration is not None:
            label += f" iteration {iteration}"
        logger.info(
            f"    {(label + ' ').ljust(48, '.')}"
            f": ({min(values):.2f}, {max(values):.2f}, {statistics.median(values):.2f})",
        )
    durable, ranks = checkpoint_durability_ms(checkpoint_spans)
    if durable:
        logger.info(f"{run}: checkpoint durability (min, max, median ms across ranks and saves)")
        logger.info(
            f"    {'save -> durable '.ljust(48, '.')}"
            f": ({min(durable):.2f}, {max(durable):.2f}, {statistics.median(durable):.2f})  "
            f"mean {statistics.mean(durable):.2f} over {len(durable)} span(s) across {len(set(ranks))} rank(s)",
        )


def load_latest(results_dir: Path) -> dict[str, list]:
    empty = {
        "native_saves": [],
        "daemon_durable_spans": [],
        "megatron_spans": [],
        "megatron_metrics": [],
        "resource_metrics": [],
    }
    otel_root = results_dir / "otel"
    if not otel_root.is_dir():
        return empty
    by_cycle = {int(path.name[5:]): path for path in otel_root.glob("cycle*") if path.is_dir() and path.name[5:].isdigit()}
    newest = by_cycle[max(by_cycle)] if by_cycle else otel_root
    files = [str(path) for path in sorted(newest.glob("*.jsonl"))]
    if not files:
        return empty
    resource_metrics = collect_resource_metrics(files)
    run_id = os.environ.get("MEGATRON_OTEL_RUN_ID")
    if run_id is None and resource_metrics:
        latest = max(resource_metrics, key=lambda sample: sample["time_unix_nano"])
        run_id = str(latest["resource"].get("nemo.run.id"))
    return {
        "native_saves": collect_samples(files, run_id=run_id),
        "daemon_durable_spans": collect_spans(files, "ckpt.svr.durable", run_id),
        "megatron_spans": collect_checkpoint_spans(files, run_id),
        "megatron_metrics": collect_megatron_training_metrics(files, run_id),
        "resource_metrics": [
            sample for sample in resource_metrics if run_id is None or str(sample["resource"].get("nemo.run.id")) == run_id
        ],
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


def metrics_from_data(data: dict[str, list], gpus_per_node: int | None = None) -> dict:
    metrics = _native_metrics(data["native_saves"], gpus_per_node)
    durable, durable_ranks = checkpoint_durability_ms(data["megatron_spans"])
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
    return metrics


def post_process(results_dir: Path, gpus_per_node: int, native: bool) -> dict:
    data = load_latest(results_dir)
    valid = native_spans_valid(data["native_saves"]) if native else True
    return {
        "source": "otlp",
        "metrics": metrics_from_data(data, gpus_per_node),
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


def validate_collected(  # noqa: C901, PLR0912, PLR0913, PLR0917
    path: Path,
    run_id: str,
    nodes: int,
    gpus_per_node: int,
    provider: str,
    expected_checkpoints: int,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"Megatron emitted no collected OTLP file: {path}")
    resources = [resource for resource in iter_span_resources([str(path)], "megatron.") if str(resource.get("nemo.run.id")) == run_id]
    ranks = {int(resource["dl.rank"]) for resource in resources if resource.get("dl.rank") is not None}
    expected = set(range(nodes * gpus_per_node))
    if ranks != expected:
        raise RuntimeError(f"Megatron OTLP exporter ranks mismatch: expected {sorted(expected)}, found {sorted(ranks)} in {path}")
    megatron_metrics = collect_megatron_training_metrics([str(path)], run_id)
    for name in MEGATRON_PERF_METRICS:
        matching = [sample for sample in megatron_metrics if sample["name"] == name]
        metric_ranks = {int(sample["resource"].get("dl.rank", -1)) for sample in matching}
        if metric_ranks != expected:
            raise RuntimeError(
                f"Megatron training metric {name} ranks mismatch: expected {sorted(expected)}, found {sorted(metric_ranks)} in {path}",
            )
        if any(sample["time_unix_nano"] <= 0 or not math.isfinite(sample["value"]) for sample in matching):
            raise RuntimeError(f"Megatron training metric {name} has invalid OTLP points in {path}")
    resource_metrics = collect_resource_metrics([str(path)], run_id)
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
    checkpoint_spans = collect_checkpoint_spans([str(path)], run_id)
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
    checkpoint_durability_ms(checkpoint_spans)
    if provider == "native":
        native_spans = collect_samples([str(path)], run_id=run_id)
        if not native_spans_valid(native_spans):
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
        durable_spans = collect_spans([str(path)], "ckpt.svr.durable", run_id)
        durable_ids = {
            (int(span["attributes"].get("nvrx.session_id", -1)), int(span["attributes"].get("nvrx.snapshot_id", -1)))
            for span in durable_spans
            if span["end_time_unix_nano"] > span["start_time_unix_nano"]
        }
        if durable_ids != set(snapshots):
            raise RuntimeError(
                f"native checkpoint durable OTLP spans mismatch: expected {sorted(snapshots)}, found {sorted(durable_ids)} in {path}",
            )
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
