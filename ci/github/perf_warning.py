#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: T201
"""Report a workload's missed baselines and changed options as a GitHub warning and job summary tables.

  perf_warning.py WORKLOAD RESULTS_DIR
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def failed_metrics(results: Path) -> list[tuple[str, dict]]:
    """The metrics whose baseline comparison failed, across the workload's perf reports."""
    failures = []
    for report in sorted(results.glob("ci/ci-results/*_report.json")):
        comparison = json.loads(report.read_text()).get("comparison") or {}
        failures += [(name, entry) for name, entry in comparison.items() if entry.get("status") == "FAIL"]
    return failures


def changed_options(results: Path) -> list[dict]:
    """The recorded Megatron options that the workload's runs trained with different values."""
    changes = []
    for record in sorted(results.glob("ci/ci-runs/**/*_arguments_differences.json")):
        changes += [change for change in json.loads(record.read_text()) if change not in changes]
    return changes


def _number(value: object) -> str:
    return f"{value:,.1f}" if isinstance(value, (int, float)) else str(value)


def _change(entry: dict) -> str:
    measured, baseline = entry.get("measured"), entry.get("baseline")
    if not isinstance(measured, (int, float)) or not isinstance(baseline, (int, float)) or not baseline:
        return "n/a"
    return f"{(measured - baseline) / baseline * 100:+.1f}%"


def describe(name: str, entry: dict) -> str:
    unit = entry.get("unit", "")
    return (
        f"{name} {entry.get('statistic', '')} {_number(entry.get('measured'))} {unit} vs baseline "
        f"{_number(entry.get('baseline'))} {unit} ({_change(entry)}, tolerance {entry.get('tolerance_pct', '?')}%)"
    ).replace("  ", " ")


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def annotation(workload: str, failures: list[tuple[str, dict]], changes: list[dict]) -> str:
    parts = []
    if failures:
        parts.append(f"{len(failures)} metric{'s' if len(failures) != 1 else ''} missed the baseline")
    if changes:
        parts.append(f"{len(changes)} option{'s differ' if len(changes) != 1 else ' differs'} from the recorded run")
    title = f"{workload}: {'; '.join(parts) or 'performance missed the baseline'}"
    details = [describe(name, entry) for name, entry in failures]
    details += [f"{change['option']}: {change['recorded']} -> {change['run']}" for change in changes]
    message = "\n".join(details) or "See the jet-results artifact."
    return f"::warning title={_escape_property(title)}::{_escape_data(message)}"


def summary(workload: str, failures: list[tuple[str, dict]], changes: list[dict]) -> str:
    lines = [f"### :warning: {workload} missed its baseline", ""]
    if failures:
        lines += [
            "| Metric | Statistic | Measured | Baseline | Change | Tolerance |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name, entry in failures:
            unit = entry.get("unit", "")
            lines.append(
                f"| `{name}` | {entry.get('statistic', '')} | {_number(entry.get('measured'))} {unit} | "
                f"{_number(entry.get('baseline'))} {unit} | {_change(entry)} | {entry.get('tolerance_pct', '?')}% |",
            )
        lines.append("")
    if changes:
        lines += ["| Option | Recorded | This run |", "| --- | --- | --- |"]
        lines += [f"| `{change['option']}` | {change['recorded']} | {change['run']} |" for change in changes]
        lines.append("")
    if not failures and not changes:
        lines.append("See the jet-results artifact.")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    workload, results = argv[0], Path(argv[1])
    failures, changes = failed_metrics(results), changed_options(results)
    print(annotation(workload, failures, changes))
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as output:
            output.write(summary(workload, failures, changes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
