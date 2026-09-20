from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

_spec = importlib.util.spec_from_file_location("otel_report", _REPO / "ci" / "scripts" / "otel_report.py")
otel_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(otel_report)

def _iv(x):
    return {"intValue": str(int(x))}


def _dv(x):
    return {"doubleValue": float(x)}


def _save_span(
    rank,
    bw,
    wr,
    twb,
    *,
    session_id=0,
    snapshot_id=1,
    num_tensors=10,
    snapshot_us=5,
    total_us=100,
    total_d2h=4,
    total_tensor_bytes=None,
    ci_phase=None,
    run_id=None,
    trace_id=None,
    span_id=None,
):
    resource_attrs = [
        {"key": "service.name", "value": {"stringValue": "nvrx-checkpoint-server"}},
        {"key": "nvrx.rank", "value": _iv(rank)},
    ]
    if ci_phase is not None:
        resource_attrs.append({"key": "nvrx.ci_phase", "value": {"stringValue": str(ci_phase)}})
    if run_id is not None:
        resource_attrs.append({"key": "nemo.run.id", "value": {"stringValue": str(run_id)}})
    total_tensor_bytes = total_tensor_bytes if total_tensor_bytes is not None else twb
    span_attrs = [
        {"key": "nvrx.session_id", "value": _iv(session_id)},
        {"key": "nvrx.rank", "value": _iv(rank)},
        {"key": "nvrx.snapshot_id", "value": _iv(snapshot_id)},
        {"key": "nvrx.bw_mbps", "value": _dv(bw)},
        {"key": "nvrx.wr_tnsr_us", "value": _iv(wr)},
        {"key": "nvrx.total_written_bytes", "value": _iv(twb)},
        {"key": "nvrx.snapshot_us", "value": _iv(snapshot_us)},
        {"key": "nvrx.total_us", "value": _iv(total_us)},
        {"key": "nvrx.num_tensors", "value": _iv(num_tensors)},
        {"key": "nvrx.total_d2h", "value": _iv(total_d2h)},
        {"key": "nvrx.total_tensor_bytes", "value": _iv(total_tensor_bytes)},
        {"key": "nvrx.avg_d2h_rq", "value": _iv(total_tensor_bytes // total_d2h)},
        {"key": "nvrx.avg_tnsr", "value": _iv(total_tensor_bytes // num_tensors)},
    ]
    span = {"name": "ckpt.save", "attributes": span_attrs}
    if trace_id is not None:
        span["traceId"] = trace_id
    if span_id is not None:
        span["spanId"] = span_id
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [{"spans": [span]}],
            }
        ]
    }


def _resource_metrics(rank: int, run_id: str = "run-1") -> dict:
    resource = {
        "attributes": [
            {"key": "nemo.run.id", "value": {"stringValue": run_id}},
            {"key": "dl.rank", "value": _iv(rank)},
        ],
    }
    metrics = [
        {"name": "nvrx.checkpoint.run.gpu_peak_bytes", "gauge": {"dataPoints": [{"asInt": "123"}]}},
        {"name": "unrelated.metric", "sum": {"dataPoints": [{"asDouble": 4.5}]}},
    ]
    return {"resourceMetrics": [{"resource": resource, "scopeMetrics": [{"metrics": metrics}]}]}


def _megatron_metrics(rank: int, run_id: str = "run-1") -> dict:
    resource = {
        "attributes": [
            {"key": "nemo.run.id", "value": {"stringValue": run_id}},
            {"key": "dl.rank", "value": _iv(rank)},
        ],
    }
    metrics = [
        {
            "name": "megatron.training.example_histogram",
            "histogram": {
                "dataPoints": [
                    {"count": "4", "sum": 80.0, "min": 10.0, "max": 30.0, "timeUnixNano": "10"},
                ],
            },
        },
    ]
    return {"resourceMetrics": [{"resource": resource, "scopeMetrics": [{"metrics": metrics}]}]}


def _write_jsonl(path: Path, objs) -> str:
    path.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return str(path)


def _memory_event(
    host: str,
    pid: int,
    rss: int,
    peak: int,
    *,
    rank: int | None = None,
    timestamp: int = 1,
    service: str = "nvrx-checkpoint-server",
) -> dict:
    resource = [
        {"key": "host.name", "value": {"stringValue": host}},
        {"key": "service.name", "value": {"stringValue": service}},
        {"key": "process.pid", "value": _iv(pid)},
    ]
    attributes = [
        {"key": "nvrx.memory.rss_bytes", "value": _iv(rss)},
        {"key": "nvrx.memory.peak_rss_bytes", "value": _iv(peak)},
        {"key": "nvrx.memory.shmem_rss_bytes", "value": _iv(rss // 2)},
        {"key": "nvrx.memory.trigger", "value": {"stringValue": "ckpt.save"}},
    ]
    if rank is not None:
        attributes.append({"key": "nvrx.rank", "value": _iv(rank)})
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": resource},
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "eventName": "process.memory",
                                "attributes": attributes,
                                "timeUnixNano": str(timestamp),
                            },
                        ],
                    },
                ],
            },
        ],
    }


def _checkpoint_timing_span(
    name: str,
    rank: int,
    duration_ms: float,
    *,
    iteration: int | None = None,
    run_id: str = "run-1",
) -> dict:
    attributes = []
    if iteration is not None:
        attributes.append({"key": "megatron.iteration", "value": _iv(iteration)})
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "nemo.run.id", "value": {"stringValue": run_id}},
                        {"key": "dl.rank", "value": _iv(rank)},
                    ],
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": name,
                                "attributes": attributes,
                                "startTimeUnixNano": "1000000",
                                "endTimeUnixNano": str(round((duration_ms + 1) * 1e6)),
                            },
                        ],
                    },
                ],
            },
        ],
    }


def test_iter_samples_extracts_per_rank(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_save_span(0, 100.0, 200, 4096), _save_span(1, 50.0, 400, 4096)])
    by_rank = {s["rank"]: s for s in otel_report.collect_samples([path])}
    assert set(by_rank) == {0, 1}
    assert by_rank[0]["bw_mbps"] == 100.0
    assert by_rank[0]["wr_tnsr_us"] == 200
    assert by_rank[0]["session_id"] == 0
    assert by_rank[0]["snapshot_id"] == 1
    assert by_rank[1]["bw_mbps"] == 50.0
    assert by_rank[1]["total_written_bytes"] == 4096


def test_process_memory_metrics_and_combined_chart(tmp_path: Path) -> None:
    previous_chart = tmp_path / "process_memory_node-a.svg"
    previous_chart.write_text("stale")
    path = _write_jsonl(tmp_path / "telemetry.jsonl", [_save_span(0, 100.0, 200, 4096)])
    _write_jsonl(
        Path(f"{path}.logs"),
        [
            _memory_event("node-a", 20, 70, 80, timestamp=10),
            _memory_event("node-a", 20, 75, 85, rank=0, timestamp=20),
            _memory_event("node-b", 21, 60, 70, timestamp=15),
            _memory_event("node-b", 21, 65, 75, rank=1, timestamp=25),
            _memory_event("node-a", 30, 40, 40, rank=0, timestamp=10, service=otel_report.TRAINING_SERVICE),
            _memory_event("node-a", 30, 50, 50, rank=0, timestamp=20, service=otel_report.TRAINING_SERVICE),
            _memory_event("node-b", 31, 35, 35, rank=1, timestamp=15, service=otel_report.TRAINING_SERVICE),
            _memory_event("node-b", 31, 45, 45, rank=1, timestamp=25, service=otel_report.TRAINING_SERVICE),
        ],
    )
    data = otel_report.OtelData([path])
    samples = otel_report.collect_process_memory(data)
    metrics = otel_report.process_memory_metrics(samples)

    assert metrics["checkpoint_daemon_peak_rss_bytes"]["max"] == 85
    outputs = otel_report.write_process_memory_charts(data, tmp_path)
    assert outputs == [tmp_path / "process_memory_all.svg"]
    assert not previous_chart.exists()
    chart = outputs[0].read_text()
    assert "daemon node-a pid 20" in chart
    assert "daemon node-b pid 21" in chart
    assert "training node-a rank 0 pid 30" in chart
    assert "training node-b rank 1 pid 31" in chart
    assert {"hsl(215 75% 24.0%)", "hsl(215 75% 72.0%)"} <= set(re.findall(r'hsl\(215 75% [^)]+\)', chart))
    assert {"hsl(0 75% 24.0%)", "hsl(0 75% 72.0%)"} <= set(re.findall(r'hsl\(0 75% [^)]+\)', chart))
    assert "shared-memory" not in chart
    assert chart.count("<polyline") == 4  # two daemon and two training processes
    assert chart.count('stroke="#d1d5db"') == 6
    assert ">0.0</text>" in chart


def test_process_memory_chart_supports_training_only(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "telemetry.jsonl",
        [_memory_event("node-a", 30, 40, 40, rank=0, timestamp=10, service=otel_report.TRAINING_SERVICE)],
    )

    outputs = otel_report.write_process_memory_charts(otel_report.OtelData([path]), tmp_path)

    assert outputs == [tmp_path / "process_memory_all.svg"]
    chart = outputs[0].read_text()
    assert "training node-a rank 0 pid 30" in chart
    assert "hsl(0 75%" in chart
    assert "hsl(215 75%" not in chart


def test_otel_data_can_recover_a_truncated_final_record(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        json.dumps(
            _memory_event("node-a", 30, 40, 40, rank=0, timestamp=10, service=otel_report.TRAINING_SERVICE),
        )
        + '\n{"resourceLogs":',
    )

    with pytest.raises(ValueError, match="invalid OTLP JSON"):
        otel_report.OtelData([path])

    data = otel_report.OtelData([path], tolerate_truncated_tail=True)
    outputs = otel_report.write_process_memory_charts(data, tmp_path)

    assert outputs == [tmp_path / "process_memory_all.svg"]
    assert "training node-a rank 0 pid 30" in outputs[0].read_text()


def test_non_save_spans_ignored(tmp_path):
    other = {"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "ckpt.finalize_wait", "attributes": []}]}]}]}
    path = _write_jsonl(tmp_path / "t.jsonl", [other, _save_span(0, 100.0, 200, 4096)])
    samples = otel_report.collect_samples([path])
    assert len(samples) == 1
    assert samples[0]["rank"] == 0


def test_iter_span_resources_selects_megatron_spans(tmp_path) -> None:
    megatron = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "nemo.run.id", "value": {"stringValue": "run-1"}},
                        {"key": "dl.rank", "value": _iv(4)},
                    ],
                },
                "scopeSpans": [{"spans": [{"name": "megatron.train.iteration"}]}],
            },
        ],
    }
    path = _write_jsonl(tmp_path / "t.jsonl", [_save_span(0, 100.0, 200, 4096), megatron])

    assert list(otel_report.iter_span_resources([path], "megatron.")) == [{"nemo.run.id": "run-1", "dl.rank": 4}]


def test_iter_span_resources_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("not-json\n")

    with pytest.raises(ValueError, match="invalid OTLP JSON"):
        list(otel_report.iter_span_resources([str(path)], "megatron."))


def test_collect_resource_metrics_selects_run_and_rank(tmp_path) -> None:
    path = _write_jsonl(tmp_path / "t.jsonl", [_resource_metrics(0), _resource_metrics(1), _resource_metrics(2, "other")])

    samples = otel_report.collect_resource_metrics([path], "run-1")

    assert [(sample["resource"]["dl.rank"], sample["value"]) for sample in samples] == [(0, 123.0), (1, 123.0)]


def test_collect_megatron_metrics_includes_histogram_mean(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "t.jsonl", [_megatron_metrics(0), _megatron_metrics(4), _megatron_metrics(8, "other")])

    samples = otel_report.collect_megatron_training_metrics([path], "run-1")

    histogram = [sample for sample in samples if sample["name"] == "megatron.training.example_histogram"]
    assert [(sample["resource"]["dl.rank"], sample["value"]) for sample in histogram] == [(0, 20.0), (4, 20.0)]
    assert histogram[0]["kind"] == "histogram"
    assert histogram[0]["count"] == 4
    assert histogram[0]["min"] == 10.0
    assert histogram[0]["max"] == 30.0


def test_summarize_means_and_per_rank(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_save_span(0, 100.0, 200, 4096), _save_span(1, 200.0, 400, 4096)])
    summary = otel_report.summarize(otel_report.collect_samples([path]))
    assert summary["bw_mbps"]["mean"] == 150.0
    assert summary["bw_mbps"]["per_rank_mean"] == {0: 100.0, 1: 200.0}
    assert summary["wr_tnsr_us"]["mean"] == 300.0
    assert summary["bw_mbps"]["higher_is_better"] is True
    assert summary["total_written_bytes"]["higher_is_better"] is None


def test_ci_phase_filter(tmp_path):
    path = _write_jsonl(
        tmp_path / "t.jsonl",
        [_save_span(0, 100.0, 200, 4096, ci_phase="1"), _save_span(0, 300.0, 200, 4096, ci_phase="2")],
    )
    samples = otel_report.collect_samples([path], ci_phase="2")
    assert len(samples) == 1
    assert samples[0]["bw_mbps"] == 300.0


def test_run_filter_deduplicates_retried_spans(tmp_path: Path) -> None:
    repeated = _save_span(0, 100.0, 200, 4096, run_id="run-1", trace_id="trace", span_id="span")
    other = _save_span(0, 300.0, 200, 4096, run_id="run-2", trace_id="other", span_id="other")
    path = _write_jsonl(tmp_path / "t.jsonl", [repeated, repeated, other])

    samples = otel_report.collect_samples([path], run_id="run-1")

    assert len(samples) == 1
    assert samples[0]["bw_mbps"] == 100.0


def test_parsed_otel_data_is_reused_by_all_collectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(_write_jsonl(tmp_path / "t.jsonl", [_save_span(0, 100.0, 200, 4096), _resource_metrics(0)]))
    real_open = Path.open
    reads = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal reads
        if path == source:
            reads += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    data = otel_report.OtelData([source])

    assert len(otel_report.collect_samples(data)) == 1
    assert len(otel_report.collect_spans(data, "ckpt.save")) == 1
    assert len(otel_report.collect_resource_metrics(data)) == 1
    assert reads == 1


def test_analysis_sidecar_avoids_epilog_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cycle = tmp_path / "otel" / "cycle1"
    cycle.mkdir(parents=True)
    source = cycle / "native_ckpt_otel.jsonl"
    source.write_text("not parsed\n")
    analysis = {
        "metrics": {},
        "native_spans_valid": True,
        "checkpoint_timings": [{"label": "save-checkpoint", "min": 1.0, "max": 3.0, "median": 2.0}],
        "checkpoint_durability": None,
    }
    otel_report.write_analysis(source, analysis)
    real_open = Path.open
    reads = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal reads
        if path == source:
            reads += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    assert otel_report.post_process(tmp_path, 4, native=True)["validation"]["valid"]
    with caplog.at_level("INFO"):
        otel_report.log_checkpoint_timings(source, None, "cycle1")
    assert "(1.00, 3.00, 2.00)" in caplog.text
    assert reads == 0

    source.write_text("changed\n")
    reads = 0
    with pytest.raises(ValueError, match="invalid OTLP JSON"):
        otel_report.post_process(tmp_path, 4, native=True)
    assert reads == 1


def test_post_process_uses_each_ranks_peak_rss_across_all_phases(tmp_path: Path) -> None:
    for phase, values in {"phase1": [100, 120], "cycle1": [130, 110]}.items():
        source = tmp_path / "otel" / phase / "native_ckpt_otel.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text("not parsed\n")
        otel_report.write_analysis(
            source,
            {
                "metrics": {
                    "cpu_rss_bytes": otel_report.metric(values, "B", higher_is_better=False, ranks=[0, 1]),
                },
                "native_spans_valid": True,
            },
        )

    measured = otel_report.post_process(tmp_path, 4, native=True)["metrics"]["cpu_rss_bytes"]

    assert measured["samples"] == [130, 120]
    assert measured["per_rank_median"] == {0: 130, 1: 120}
    assert measured["max"] == 130  # noqa: PLR2004


def test_complete_run_process_memory_peaks_reads_collected_otlp(tmp_path: Path) -> None:
    for phase, peak in {"phase1": 80, "cycle1": 95}.items():
        source = tmp_path / "otel" / phase / "native_ckpt_otel.jsonl"
        source.parent.mkdir(parents=True)
        _write_jsonl(source, [_memory_event("node-a", 20, peak - 10, peak, timestamp=10)])

    metrics = otel_report.complete_run_process_memory_peaks(tmp_path)

    daemon = metrics["checkpoint_daemon_peak_rss_bytes"]
    assert daemon["max"] == 95  # noqa: PLR2004
    assert daemon["per_node"] == {"node-a": 95}


def test_post_process_rejects_mismatched_cpu_rss_ranks_and_samples(tmp_path: Path) -> None:
    source = tmp_path / "otel" / "native_ckpt_otel.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("not parsed\n")
    otel_report.write_analysis(
        source,
        {
            "metrics": {"cpu_rss_bytes": {"ranks": [0, 1], "samples": [100]}},
            "native_spans_valid": True,
        },
    )

    with pytest.raises(ValueError, match="cpu_rss_bytes ranks and samples differ in length"):
        otel_report.post_process(tmp_path, 4, native=True)


def test_main_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_jsonl(tmp_path / "t.jsonl", [_save_span(0, 100.0, 200, 4096)])

    assert otel_report.main([path, "--json"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["n_saves"] == 1
    assert summary["metrics"]["bw_mbps"]["mean"] == 100


def test_log_checkpoint_timings_summarizes_each_phase_across_ranks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_jsonl(
        tmp_path / "t.jsonl",
        [
            _checkpoint_timing_span("megatron.checkpoint.load", 0, 40),
            _checkpoint_timing_span("megatron.checkpoint.load", 4, 60),
            _checkpoint_timing_span("megatron.checkpoint.exposed_save", 0, 10, iteration=100),
            _checkpoint_timing_span("megatron.checkpoint.exposed_save", 4, 50, iteration=100),
            _checkpoint_timing_span("megatron.checkpoint.exposed_save", 8, 100, iteration=100, run_id="other"),
        ],
    )

    with caplog.at_level("INFO"):
        otel_report.log_checkpoint_timings(Path(path), "run-1", "cycle1")

    assert "cycle1: checkpoint timings (min, max, median ms across ranks)" in caplog.text
    assert "load-checkpoint" in caplog.text
    assert "(40.00, 60.00, 50.00)" in caplog.text
    assert "save-checkpoint iteration 100" in caplog.text
    assert "(10.00, 50.00, 30.00)" in caplog.text
    assert "100.000" not in caplog.text


def test_log_checkpoint_timings_infers_the_phase_run_id(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "t.jsonl",
        [_checkpoint_timing_span("megatron.checkpoint.exposed_save", 0, 10)],
    )

    otel_report.log_checkpoint_timings(Path(path), None, "phase1")


def test_log_checkpoint_timings_requires_the_exposed_save(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "t.jsonl",
        [_checkpoint_timing_span("megatron.checkpoint.save", 0, 10)],
    )

    with pytest.raises(RuntimeError, match="expected one checkpoint timing run ID"):
        otel_report.log_checkpoint_timings(Path(path), None, "phase1")
