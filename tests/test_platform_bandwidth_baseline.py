# ruff: noqa: INP001, PLR2004, S101, SLF001
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "scripts"))

from slurm_job import PERF_TOLERANCE, PerfReport


def test_each_platform_has_one_bandwidth_floor() -> None:
    assert PerfReport._platform_bandwidth_baseline("devbox") == 6800
    assert PerfReport._platform_bandwidth_baseline("h100") == 36000
    assert PerfReport._platform_bandwidth_baseline("gb200") == 115000


def test_native_bandwidth_is_reported_as_integer_mbps() -> None:
    log = "[native_ckpt:cli][0][INFO][perf] Finished bw_mbps=94000.4 wr_tnsr_us=10 total_written_bytes=100"

    metrics = PerfReport._parse_metrics(log, 4)

    assert metrics["bw_mbps"]["mean"] == 94000
    assert metrics["bw_mbps"]["unit"] == "MB/s"
    assert "bw_gbps" not in metrics


def test_platform_bandwidth_is_a_floor_not_a_ceiling() -> None:
    baseline = {"metrics": {"bw_mbps": 140000}}
    fast = {"bw_mbps": {"mean": 160000, "higher_is_better": True, "unit": "MB/s"}}
    slow = {"bw_mbps": {"mean": 80000, "higher_is_better": True, "unit": "MB/s"}}

    fast_comparison, fast_passed = PerfReport._compare_to_baseline(fast, baseline, PERF_TOLERANCE)
    slow_comparison, slow_passed = PerfReport._compare_to_baseline(slow, baseline, PERF_TOLERANCE)

    assert fast_passed
    assert fast_comparison["bw_mbps"]["status"] == "PASS"
    assert not slow_passed
    assert slow_comparison["bw_mbps"]["status"] == "FAIL"


def test_lower_is_better_metric_is_a_ceiling_not_a_floor() -> None:
    baseline = {"metrics": {"wr_tnsr_us": 37700}}
    faster = {"wr_tnsr_us": {"mean": 34867, "higher_is_better": False, "unit": "us"}}
    slower = {"wr_tnsr_us": {"mean": 41000, "higher_is_better": False, "unit": "us"}}

    faster_comparison, faster_passed = PerfReport._compare_to_baseline(faster, baseline, PERF_TOLERANCE)
    slower_comparison, slower_passed = PerfReport._compare_to_baseline(slower, baseline, PERF_TOLERANCE)

    assert faster_passed
    assert faster_comparison["wr_tnsr_us"]["status"] == "PASS"
    assert not slower_passed
    assert slower_comparison["wr_tnsr_us"]["status"] == "FAIL"


def test_timing_baselines_use_steady_state_median() -> None:
    log = "\n".join(
        f"[native_ckpt:cli][0][INFO][perf] Finished bw_mbps={bw} wr_tnsr_us={write} total_written_bytes=100"
        for bw, write in ((80000, 38000), (145000, 20500), (150000, 19500))
    )
    metrics = PerfReport._parse_metrics(log, 4)
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
