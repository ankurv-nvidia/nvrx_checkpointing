#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: T201
"""Report one JET workload's checkpoint saves and baselines, or judge a run by every report.

  perf_report.py workload NAME RESULTS_DIR EXIT_CODE OUTPUT_DIR
  perf_report.py summary REPORTS_DIR
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SOFT_FAILURE = 3
PASSED, MISSED, FAILED = "passed", "missed its baselines", "failed"
SAVE_ITERATION = re.compile(r"(?:save_checkpoint for iteration|\] iteration)\s+(\d+)")
GRADED = frozenset({"PASS", "FAIL", "WARN"})
SAVE_TIMER = re.compile(r"save-checkpoint \.+: \(([\d.]+), ([\d.]+)\)")
RANK_SUFFIX = re.compile(r"_rank\d+$")


def comparisons(results: Path) -> dict[str, dict]:
    """Every graded metric's comparison with its baseline, across the workload's reports."""
    found = {}
    for report in sorted(results.glob("ci/ci-results/*_report.json")):
        comparison = json.loads(report.read_text()).get("comparison") or {}
        found.update({name: entry for name, entry in comparison.items() if entry.get("status") in GRADED})
    return found


def changed_options(results: Path) -> list[dict]:
    """The recorded Megatron options that the workload's runs trained with different values."""
    changes = []
    for record in sorted(results.glob("ci/ci-runs/**/*_arguments_differences.json")):
        changes += [change for change in json.loads(record.read_text()) if change not in changes]
    return changes


def failure_summary(results: Path) -> list[str]:
    """The causal log lines that a failed run kept."""
    return [line for path in sorted(results.glob("ci/ci-results/*_failure_summary.txt")) for line in path.read_text().splitlines()]


def checkpoint_saves(results: Path) -> list[tuple[str, int | None, float, float]]:
    """Megatron's save-checkpoint time per save, as the min and max across ranks in milliseconds."""
    runs = results / "ci" / "ci-runs"
    saves = []
    for log in sorted(runs.glob("**/logs/*.log")):
        parts = [part for part in log.relative_to(runs).parts[1:-1] if part != "logs"]
        label = "/".join([*parts, RANK_SUFFIX.sub("", log.stem)])
        iteration = None
        for line in log.read_text(errors="replace").splitlines():
            if match := SAVE_ITERATION.search(line):
                iteration = int(match[1])
            elif match := SAVE_TIMER.search(line):
                saves.append((label, iteration, float(match[1]), float(match[2])))
    return saves


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


def verdict(exit_code: int) -> str:
    if exit_code == 0:
        return PASSED
    return MISSED if exit_code == SOFT_FAILURE else FAILED


def markdown(  # noqa: PLR0913
    name: str,
    result: str,
    comparison: dict[str, dict],
    changes: list[dict],
    saves: list,
    failure: list[str],
) -> str:
    icon = {PASSED: ":white_check_mark:", MISSED: ":warning:"}.get(result, ":x:")
    lines = [f"### {icon} `{name}` {result}", ""]
    if failure:
        lines += ["```", *failure, "```", ""]
    if comparison:
        lines += [
            "| Metric | Statistic | Status | Measured | Baseline | Change | Tolerance |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for metric, entry in comparison.items():
            unit = entry.get("unit", "")
            lines.append(
                f"| `{metric}` | {entry.get('statistic', '')} | {entry.get('status', '')} | "
                f"{_number(entry.get('measured'))} {unit} | {_number(entry.get('baseline'))} {unit} | "
                f"{_change(entry)} | {entry.get('tolerance_pct', '?')}% |",
            )
    else:
        lines.append("The workload reported no metrics.")
    lines.append("")
    if changes:
        lines += ["| Option | Recorded | This run |", "| --- | --- | --- |"]
        lines += [f"| `{change['option']}` | {change['recorded']} | {change['run']} |" for change in changes]
        lines.append("")
    if saves:
        lines += ["| Run | Iteration | save-checkpoint min (ms) | max (ms) |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {label} | {iteration or '?'} | {low:,.2f} | {high:,.2f} |" for label, iteration, low, high in saves]
        lines.append("")
    return "\n".join(lines) + "\n"


def report_workload(name: str, results: Path, exit_code: int, output: Path) -> int:
    comparison, changes, saves = comparisons(results), changed_options(results), checkpoint_saves(results)
    failure = failure_summary(results)
    result = verdict(exit_code)
    failures = {metric: entry for metric, entry in comparison.items() if entry.get("status") == "FAIL"}
    text = markdown(name, result, comparison, changes, saves, failure)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.md").write_text(text)
    record = {"workload": name, "result": result, "missed": sorted(failures), "changed_options": len(changes)}
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    if result == MISSED:
        details = [describe(metric, entry) for metric, entry in failures.items()]
        details += [f"{change['option']}: {change['recorded']} -> {change['run']}" for change in changes]
        title = f"{name}: {len(failures)} metric(s) missed the baseline; {len(changes)} option(s) changed"
        print(f"::warning title={_escape_property(title)}::{_escape_data(chr(10).join(details) or 'See the report.')}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as handle:
            handle.write(text)
    print(text)
    return 0


def summarize(reports: Path) -> int:
    records = [json.loads(path.read_text()) for path in sorted(reports.glob("*.json"))]
    lines = []
    if not records:
        lines.append(":x: No JET workload reported its results.")
    else:
        failing = [record for record in records if record["result"] != PASSED]
        lines += [
            f"{len(records) - len(failing)} of {len(records)} JET workloads matched their baselines.",
            "",
            "| Workload | Result | Missed metrics | Changed options |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| `{record['workload']}` | {record['result']} | {', '.join(record['missed']) or '-'} | {record['changed_options']} |"
            for record in records
        ]
        lines.append("")
        lines += [(reports / f"{record['workload']}.md").read_text() for record in failing]
    text = "\n".join(lines) + "\n"
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary).write_text(text)
    print(text)
    if not records or any(record["result"] != PASSED for record in records):
        names = ", ".join(record["workload"] for record in records if record["result"] != PASSED) or "no reports"
        print(f"::error title=JET results need a look::{_escape_data(names)}")
        return 1
    return 0


def main(argv: list[str]) -> int:
    match argv:
        case ["workload", name, results, exit_code, output]:
            return report_workload(name, Path(results), int(exit_code), Path(output))
        case ["summary", reports]:
            return summarize(Path(reports))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
