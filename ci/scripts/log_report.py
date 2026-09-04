# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001
from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("native_ci")
_MIN_SOCKET_RANKS = 4
REQUIRED_PERF_METRICS = frozenset({"bw_mbps", "wr_tnsr_us", "total_written_bytes"})

_PERF_BEGIN_RE = re.compile(
    r"^\[native_ckpt:svr\]\[(?P<session>\d+)\]\[PERF\] Rank=(?P<rank>\d+) "
    r"Begin span=ckpt\.save started_at_ns=(?P<started_at_ns>\d+) snapshot=(?P<snapshot>\d+)$",
    re.MULTILINE,
)
_BW_RE = re.compile(
    r"^\[native_ckpt:svr\]\[(?P<session>\d+)\]\[PERF\] Rank=(?P<rank>\d+) Finished "
    r"bw_mbps=(?P<bw_mbps>\d+(?:\.\d+)?) \| Time\(us\):  "
    r"wr_tnsr_us=(?P<wr_tnsr_us>\d+)[uµ]s "
    r"snapshot_us=(?P<snapshot_us>\d+)[uµ]s total_us=(?P<total_us>\d+)[uµ]s \| "
    r"Count: num_tensors=(?P<num_tensors>\d+) total_d2h=(?P<total_d2h>\d+) \| "
    r"Bytes: total_written_bytes=(?P<total_written_bytes>\d+) "
    r"avg_d2h_rq=(?P<avg_d2h_rq>\d+) avg_tnsr=(?P<avg_tnsr>\d+) "
    r"snapshot=(?P<snapshot>\d+)$",
    re.MULTILINE,
)
_MEM_RE = re.compile(
    r"\[native_ckpt:mem\](?: rank=(\d+))? gpu_peak_bytes=(\d+)"
    r"(?: gpu_hbm_peak_bytes=(\d+) gpu_total_bytes=(\d+))? cpu_rss_bytes=(\d+)",
)
_DAEMON_DURABLE_RE = re.compile(
    r"\[native_ckpt:svr\]\[[^\]]*\]\[PERF\] Snapshot=(\d+) Durable[^\n]*?activate_to_durable_us=(\d+)",
)
_CKPT_EVENT_RE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] "
    r"(saving checkpoint at iteration|successfully saved (?:\S+ )?checkpoint from iteration)"
    r"\s+(\d+)",
)
_PHASE2_MARKER = "[native_ckpt:phase] phase=2"
_SAVE_RE = re.compile(r"successfully saved (?:async )?checkpoint from iteration\s+\d+\s+to\s+(\S+)")


def _metric(
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


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _daemon_text(results_dir: Path) -> str:
    by_cycle: dict[int, list[Path]] = {}
    for path in results_dir.glob("checkpoint_server.cycle*.log"):
        tag = path.name.split(".")[1]
        if tag.startswith("cycle") and tag[5:].isdigit():
            by_cycle.setdefault(int(tag[5:]), []).append(path)
    if not by_cycle:
        return ""
    return "\n".join(_read(path) for path in sorted(by_cycle[max(by_cycle)]))


def _validate_daemon_markers(text: str) -> bool:
    marker_lines = [line for line in text.splitlines() if "[PERF]" in line and ("Begin span=ckpt.save" in line or " Finished " in line)]
    invalid = [line for line in marker_lines if not (_PERF_BEGIN_RE.fullmatch(line) or _BW_RE.fullmatch(line))]
    begins = Counter((match["session"], int(match["rank"]), int(match["snapshot"])) for match in _PERF_BEGIN_RE.finditer(text))
    finished = Counter((match["session"], int(match["rank"]), int(match["snapshot"])) for match in _BW_RE.finditer(text))
    valid = True
    if invalid:
        logger.warning(
            f"[slurm_job:perf] WARNING {len(invalid)} ckpt.save PERF marker(s) did not match the required schema; "
            f"first mismatch: {invalid[0]}",
        )
        valid = False
    if not begins:
        logger.warning("[slurm_job:perf] WARNING no schema-valid ckpt.save Begin markers found in measured-cycle daemon logs")
        valid = False
    if not finished:
        logger.warning("[slurm_job:perf] WARNING no schema-valid ckpt.save Finished markers found in measured-cycle daemon logs")
        valid = False
    missing_finished = begins - finished
    missing_begins = finished - begins
    for label, records in (("Begin marker(s) without Finished", missing_finished), ("Finished marker(s) without Begin", missing_begins)):
        if records:
            description = ", ".join(
                f"session={session}/rank={rank}/snapshot={snapshot} x{count}" for (session, rank, snapshot), count in records.items()
            )
            logger.warning(f"[slurm_job:perf] WARNING ckpt.save {label}: {description}")
            valid = False
    return valid


def _durability_ms(text: str) -> list[float]:
    started: dict[int, str] = {}
    elapsed = []
    for match in _CKPT_EVENT_RE.finditer(text):
        timestamp, event, iteration = match[1], match[2], int(match[3])
        if event.startswith("saving"):
            started[iteration] = timestamp
            continue
        start = started.pop(iteration, None)
        if start is not None:
            ended_at = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            started_at = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            span = ended_at - started_at
            elapsed.append(round(span.total_seconds() * 1000, 2))
    return elapsed


def _rank_memory_measurements(job_log: Path) -> str:
    if job_log.parent.name != "logs":
        return ""
    records = []
    for path in sorted(job_log.parent.parent.glob("**/logs/*-routine_rank*.log")):
        records.extend(match.group(0) for match in _MEM_RE.finditer(_read(path)))
    return "\n".join(records)


def _log_checkpoint_timings(job_log: Path) -> None:
    if job_log.parent.name != "logs":
        return
    grouped: dict[str, dict[int, Path]] = {}
    for path in job_log.parent.parent.glob("**/logs/*-routine_rank*.log"):
        if match := re.fullmatch(r"(?P<label>.+)-routine_rank(?P<rank>\d+)", path.stem):
            grouped.setdefault(match["label"], {})[int(match["rank"])] = path
    for label, ranks in sorted(grouped.items()):
        last = ranks[max(ranks)]
        timers = [line.rstrip() for line in _read(last).splitlines() if "save-checkpoint" in line or "load-checkpoint" in line]
        if timers:
            logger.info(f"{label}: checkpoint timers from {last.name} (min, max ms across ranks)")
            for line in timers:
                logger.info(line)
        spans = _durability_ms(_read(ranks[min(ranks)]))
        if spans:
            samples = ", ".join(f"{value:.2f}" for value in spans)
            logger.info(f"{label}: checkpoint durability (min, max ms across saves)")
            summary = (
                f"{('    save -> durable '.ljust(52, '.'))}: "
                f"({min(spans):.2f}, {max(spans):.2f})  mean {statistics.mean(spans):.2f} "
                f" over {len(spans)} save(s) [{samples}]"
            )
            logger.info(summary)


def _metrics(text: str, daemon_text: str, gpus_per_node: int | None) -> dict:
    text = text[text.rfind(_PHASE2_MARKER) :] if _PHASE2_MARKER in text else text
    matches = list(_BW_RE.finditer(daemon_text))
    ranks = [int(match["rank"]) for match in matches]
    bandwidth = [round(float(match["bw_mbps"])) for match in matches]
    write = [int(match["wr_tnsr_us"]) for match in matches]
    written = [int(match["total_written_bytes"]) for match in matches]
    metrics: dict = {}
    if ranks:
        metrics = {
            "bw_mbps": _metric(bandwidth, "MB/s", higher_is_better=True, ranks=ranks),
            "wr_tnsr_us": _metric(write, "us", higher_is_better=False, ranks=ranks),
            "total_written_bytes": _metric(written, "B", higher_is_better=None, ranks=ranks),
        }
        unique = sorted(set(ranks))
        per_rank = [statistics.mean(value for rank_value, value in zip(ranks, bandwidth) if rank_value == rank) for rank in unique]
        if gpus_per_node and len(per_rank) == gpus_per_node and len(per_rank) >= _MIN_SOCKET_RANKS and len(per_rank) % 2 == 0:
            half = len(per_rank) // 2
            means = statistics.mean(per_rank[:half]), statistics.mean(per_rank[half:])
            metrics["bw_socket_imbalance"] = _metric(
                [max(means) / min(means)],
                "ratio",
                higher_is_better=None,
            )
    durable = _durability_ms(text)
    if durable:
        metrics["ckpt_durable_ms"] = _metric(durable, "ms", higher_is_better=False)
    daemon_durable = [int(match[2]) / 1e3 for match in _DAEMON_DURABLE_RE.finditer(daemon_text)]
    if daemon_durable:
        metrics["ckpt_daemon_durable_ms"] = _metric(daemon_durable, "ms", higher_is_better=False)
    memory = list(_MEM_RE.finditer(text))
    if memory:
        ranks = [int(match[1]) for match in memory if match[1] is not None]
        memory_ranks = ranks if len(ranks) == len(memory) else None
        metrics["gpu_peak_bytes"] = _metric(
            [int(match[2]) for match in memory],
            "B",
            higher_is_better=False,
            ranks=memory_ranks,
        )
        hbm = [(int(match[3]), int(match[4])) for match in memory if match[3] is not None and match[4] is not None]
        if hbm:
            hbm_ranks = ranks if len(ranks) == len(hbm) else None
            metrics["gpu_hbm_peak_bytes"] = _metric(
                [used for used, _total in hbm],
                "B",
                higher_is_better=False,
                ranks=hbm_ranks,
            )
            metrics["gpu_hbm_peak_pct"] = _metric(
                [used * 100 / total for used, total in hbm],
                "%",
                higher_is_better=False,
                ranks=hbm_ranks,
            )
        cpu_rss = [int(match[5]) for match in memory]
        if memory_ranks is not None:
            per_rank = {
                rank: max(value for sample_rank, value in zip(memory_ranks, cpu_rss) if sample_rank == rank)
                for rank in sorted(set(memory_ranks))
            }
            cpu_rss, memory_ranks = list(per_rank.values()), list(per_rank)
        metrics["cpu_rss_bytes"] = _metric(cpu_rss, "B", higher_is_better=False, ranks=memory_ranks)
    return metrics


def post_process(log: Path, results_dir: Path, gpus_per_node: int, native: bool) -> dict:
    text = _read(log)
    memory = _rank_memory_measurements(log)
    metric_text = f"{_MEM_RE.sub('', text)}\n{memory}" if memory else text
    daemon_text = _daemon_text(results_dir)
    valid = _validate_daemon_markers(daemon_text) if native else True
    _log_checkpoint_timings(log)
    return {
        "source": "logs",
        "metrics": _metrics(metric_text, daemon_text, gpus_per_node),
        "validation": (
            {
                "name": "ckpt_save_markers",
                "valid": valid,
                "measured": "paired" if valid else "missing_or_mismatched",
                "baseline": "paired",
                "report_field": "ckpt_save_markers_valid",
            }
            if native
            else None
        ),
    }


def validate_checkpoint_saves(log: Path, expected_phase1: int, expected_phase2: int) -> None:
    saves = _SAVE_RE.findall(log.read_text(errors="replace"))
    phase1 = sum("/phase1-checkpoints" in path for path in saves)
    phase2 = sum("/checkpoints" in path and "/phase1-checkpoints" not in path for path in saves)
    logger.info(
        f"[slurm_job:epilog] checkpoint saves: phase1={phase1}, phase2={phase2}, "
        f"expected_phase1={expected_phase1}, expected_phase2={expected_phase2}",
    )
    if phase1 < expected_phase1 or (expected_phase2 and phase2 < expected_phase2):
        raise RuntimeError(
            f"incomplete checkpoint cycle in {log}: found {phase1} phase-1 and {phase2} phase-2 saves; "
            f"expected at least {expected_phase1} and {expected_phase2}",
        )
