from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "scripts"))

from run_job import BaselineUpdater, PerfReport


def test_cluster_metrics_override_centers_only() -> None:
    baseline = {
        "metrics": {"bw_mbps": 142000.0, "total_written_bytes": 100},
        "cluster_metrics": {"awscmh": {"bw_mbps": 104000.0}},
    }

    selected = PerfReport._select_cluster_baseline(baseline, "awscmh")

    assert selected["metrics"] == {"bw_mbps": 104000.0, "total_written_bytes": 100}
    assert baseline["metrics"]["bw_mbps"] == 142000.0
    assert PerfReport._select_cluster_baseline(baseline, "ocihsg") is baseline


def test_baseline_update_targets_cluster_override(tmp_path: Path) -> None:
    baseline_file = tmp_path / "perf.yaml"
    baseline_file.write_text(
        yaml.safe_dump(
            {
                "metrics": {"bw_mbps": 142000.0, "total_written_bytes": 100},
                "cluster_metrics": {"awscmh": {"bw_mbps": 104000.0}},
            },
            sort_keys=False,
        ),
    )
    metrics = {
        "bw_mbps": {"mean": 110000.0, "higher_is_better": True},
        "total_written_bytes": {"mean": 100, "higher_is_better": None},
    }

    BaselineUpdater.update_baseline_file(
        baseline_file,
        metrics,
        "pretrain-native",
        "pico_1x4",
        "perf",
        4,
        cluster="awscmh",
    )

    updated = yaml.safe_load(baseline_file.read_text())
    assert updated["metrics"] == {"bw_mbps": 142000.0, "total_written_bytes": 100}
    assert updated["cluster_metrics"]["awscmh"] == {"bw_mbps": 110000.0}
