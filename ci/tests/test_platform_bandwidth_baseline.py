# ruff: noqa: INP001, PLR2004, S101, SLF001
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci" / "scripts"))

from slurm_job import PERF_TOLERANCE, PerfReport, otel_report


def _otel_data(rows: list[tuple[float, int]]) -> dict[str, list]:
    return {
        "native_saves": [
            {"rank": rank, "snapshot_id": 1, "bw_mbps": bandwidth, "wr_tnsr_us": write_us, "total_written_bytes": 100}
            for rank, (bandwidth, write_us) in enumerate(rows)
        ],
        "daemon_durable_spans": [],
        "megatron_spans": [],
        "megatron_metrics": [],
        "resource_metrics": [],
    }


def test_each_platform_has_one_bandwidth_floor() -> None:
    # The floors themselves move with the hardware and are recorded in
    # ci/baselines/platform_bw_mbps.yaml; what this asserts is that every
    # platform has exactly one, and that it is a usable number.
    floors = {
        name: PerfReport._platform_bandwidth_baseline(name)
        for name in ("devbox", "h100", "gb200", "gb300")
    }
    assert all(isinstance(floor, int) and floor > 0 for floor in floors.values())
    assert floors["devbox"] < floors["h100"] < floors["gb200"]
    assert floors["gb300"] >= floors["gb200"]


def test_native_bandwidth_is_reported_as_integer_mbps() -> None:
    metrics = otel_report.metrics_from_data(_otel_data([(94000.4, 10)]), 4)

    assert metrics["bw_mbps"]["mean"] == 94000
    assert metrics["bw_mbps"]["unit"] == "MB/s"
    assert "bw_gbps" not in metrics


def test_bandwidth_improvement_outside_tolerance_warns() -> None:
    baseline = {"metrics": {"bw_mbps": 140000}}
    fast = {"bw_mbps": {"mean": 160000, "higher_is_better": True, "unit": "MB/s"}}
    slow = {"bw_mbps": {"mean": 80000, "higher_is_better": True, "unit": "MB/s"}}

    fast_comparison, fast_passed = PerfReport._compare_to_baseline(fast, baseline, PERF_TOLERANCE)
    slow_comparison, slow_passed = PerfReport._compare_to_baseline(slow, baseline, PERF_TOLERANCE)

    assert fast_passed
    assert fast_comparison["bw_mbps"]["status"] == "WARN"
    assert not slow_passed
    assert slow_comparison["bw_mbps"]["status"] == "FAIL"


def test_timing_improvement_outside_tolerance_warns() -> None:
    baseline = {"metrics": {"wr_tnsr_us": 37700}}
    faster = {"wr_tnsr_us": {"mean": 34867, "higher_is_better": False, "unit": "us"}}
    slower = {"wr_tnsr_us": {"mean": 41000, "higher_is_better": False, "unit": "us"}}

    faster_comparison, faster_passed = PerfReport._compare_to_baseline(faster, baseline, PERF_TOLERANCE)
    slower_comparison, slower_passed = PerfReport._compare_to_baseline(slower, baseline, PERF_TOLERANCE)

    assert faster_passed
    assert faster_comparison["wr_tnsr_us"]["status"] == "WARN"
    assert not slower_passed
    assert slower_comparison["wr_tnsr_us"]["status"] == "FAIL"


def test_timing_baselines_use_steady_state_median() -> None:
    metrics = otel_report.metrics_from_data(_otel_data([(80000, 38000), (145000, 20500), (150000, 19500)]), 4)
    comparison, passed = PerfReport._compare_to_baseline(
        metrics,
        {"metrics": {"bw_mbps": 140000, "wr_tnsr_us": 22000}},
        PERF_TOLERANCE,
    )

    assert metrics["bw_mbps"]["mean"] < 140000 * (1 - PERF_TOLERANCE)
    assert metrics["wr_tnsr_us"]["mean"] > 22000 * (1 + PERF_TOLERANCE)
    assert passed
    assert comparison["bw_mbps"]["measured"] == 145000
    assert comparison["bw_mbps"]["statistic"] == "median"
    assert comparison["wr_tnsr_us"]["measured"] == 20500
    assert comparison["wr_tnsr_us"]["statistic"] == "median"


def test_written_bytes_pass_only_inside_the_recorded_value_or_range() -> None:
    metrics = otel_report.written_bytes_metrics(
        [(0, 1, 1000), (1, 1, 900), (0, 2, 1000), (1, 2, 900), (0, 3, 1064), (1, 3, 900)],
    )
    recorded = {
        "total_written_bytes_total": "1900+64",
        "total_written_bytes_rank0": "1000+64",
        "total_written_bytes_per_rank": "890+200",
    }

    comparison, passed = PerfReport._compare_to_baseline(metrics, {"metrics": recorded}, PERF_TOLERANCE)
    exact = {**recorded, "total_written_bytes_total": 1900, "total_written_bytes_rank0": "1000+63"}
    exact_comparison, exact_passed = PerfReport._compare_to_baseline(metrics, {"metrics": exact}, PERF_TOLERANCE)

    assert passed
    assert {name: entry["measured"] for name, entry in comparison.items()} == {
        "total_written_bytes_total": "1900+64",
        "total_written_bytes_rank0": "1000+64",
        "total_written_bytes_per_rank": "900+164",
    }
    assert not exact_passed
    assert {name: entry["status"] for name, entry in exact_comparison.items()} == {
        "total_written_bytes_total": "FAIL",
        "total_written_bytes_rank0": "FAIL",
        "total_written_bytes_per_rank": "PASS",
    }


@pytest.mark.parametrize("recorded", [1900.5, "1900-1964", "1900+", True])
def test_written_bytes_baselines_must_be_integers_or_ranges(recorded: object) -> None:
    metrics = otel_report.written_bytes_metrics([(0, 1, 1000), (1, 1, 900)])

    with pytest.raises(ValueError, match="must be an integer or <lower>"):
        PerfReport._compare_to_baseline(metrics, {"metrics": {"total_written_bytes_total": recorded}}, PERF_TOLERANCE)
