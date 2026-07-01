#!/usr/bin/env python3
"""Generate the native CI child pipelines from the PLATFORMS table.

Copies the shared header (everything above HEADER_MARKER) from
gitlab-ci-native.yml and appends one standalone generated-<group>.yml per GROUPS
entry, which the parent's per-group trigger jobs fire as child pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PARENT_CI = HERE / "gitlab-ci-native.yml"

HEADER_MARKER = "# >>> generate-ci.py: concrete jobs are generated below this line <<<"


# One dict per hardware platform; keys are consumed in _emit_platform / _emit_multinode.
PLATFORMS = [
    {
        "name": "devbox",
        "gpus": "1x2",
        "has_image": False,
        "multinode": [],
    },
    {
        "name": "gb200",
        "gpus": "1x4",
        "has_image": True,
        "multinode": [
            {
                "key": "2node",
                "config": "pico",
                "nodes": 2,
                "scale": "2x4",
                "with_update": True,
                "jobs": [("pretrain-native", "deterministic"), ("pretrain-nvrx", "deterministic")],
            },
            {
                "key": "nano3",
                "config": "nano3",
                "nodes": 8,
                "scale": "8x4",
                "with_update": True,
                "group": "8node-nano3",
                "timeout": "1d",
                "jobs": [
                    ("pretrain-native", "deterministic"),
                    ("pretrain-native", "perf"),
                    ("pretrain-nvrx", "deterministic"),
                    ("pretrain-nvrx", "perf"),
                    ("pretrain-native", ["deterministic", "valgrind"]),
                ],
            },
            {
                "key": "nano4",
                "config": "nano4",
                "nodes": 8,
                "scale": "8x4",
                "with_update": True,
                "group": "8node-nano4",
                "timeout": "1d",
                "jobs": [
                    ("pretrain-native", "deterministic"),
                    ("pretrain-native", "perf"),
                    ("pretrain-nvrx", "deterministic"),
                    ("pretrain-nvrx", "perf"),
                    ("pretrain-native", ["deterministic", "valgrind"]),
                ],
            },
            {
                "key": "multinode-perf",
                "config": "ultra",
                "nodes": 16,
                "scale": "16x4",
                "with_update": False,
                "group": "16node-ultra",
                "timeout": "1d",
                "jobs": [("pretrain-native", "perf")],
            },
        ],
    },
    {
        "name": "h100",
        "gpus": "1x8",
        "has_image": True,
        "multinode": [
            {
                "key": "2node",
                "config": "pico",
                "nodes": 2,
                "scale": "2x8",
                "with_update": True,
                "jobs": [("pretrain-native", "deterministic"), ("pretrain-nvrx", "deterministic")],
            },
            {
                "key": "multinode-perf",
                "config": "ultra",
                "nodes": 16,
                "scale": "16x8",
                "with_update": False,
                "group": "16node-ultra",
                "timeout": "1d",
                "jobs": [("pretrain-native", "perf")],
            },
        ],
    },
]


def _read_header() -> str:
    """Return the shared template header of PARENT_CI (everything above HEADER_MARKER)."""
    text = PARENT_CI.read_text()
    head, sep, _rest = text.partition(HEADER_MARKER)
    if not sep:
        raise SystemExit(
            f"{PARENT_CI.name}: header marker not found — expected a line:\n  {HEADER_MARKER}",
        )
    return head.rstrip() + "\n"


def _job(name: str, extends: list[str], *, job_args: str | None = None, timeout: str | None = None) -> str:
    """Render one concrete job block; ``timeout`` overrides .run_job's default 4h job timeout."""
    lines = [f"{name}:", f"  extends: [{', '.join(extends)}]"]
    if timeout is not None:
        lines.append(f"  timeout: {timeout}")
    if job_args is not None:
        lines += ["  variables:", f'    JOB_ARGS: "{job_args}"']
    return "\n".join(lines)


def _update(test_name: str, runner: str, *, timeout: str | None = None) -> str:
    """Render the update_baselines sibling for a test job."""
    lines = [
        f"{test_name}:update_baselines:",
        f"  extends: [.update_baseline_base, {runner}]",
    ]
    if timeout is not None:
        lines.append(f"  timeout: {timeout}")
    lines += [
        "  needs:",
        f"    - job: {test_name}",
        "      artifacts: true",
    ]
    return "\n".join(lines)


# One child pipeline per group; auto-vs-manual is set by each group's trigger in gitlab-ci-native.yml.
GROUPS = (
    "ckptbench",
    "native-deterministic",
    "native-perf",
    "nvrx-deterministic",
    "nvrx-perf",
    "valgrind",
    "16node-ultra",
    "8node-nano3",
    "8node-nano4",
)

# Display tags for the flavor name segment; run_job.py's _parse_job_filter maps them back.
FLAVOR_LABEL = {"deterministic": "[DETR]", "perf": "[PERF]"}


def _provider(job: str) -> str:
    """'pretrain-native' -> 'native', 'pretrain-nvrx' -> 'nvrx'."""
    return job.split("-", 1)[1]


def _pretrain_group(job: str, flavor: str) -> str:
    """Default group for a single-node or 2-node pretrain job: <provider>-<flavor>."""
    return f"{_provider(job)}-{flavor}"


def _emit_multinode(p: dict, runner: str, groups: dict[str, list[str]]) -> None:
    """Emit each multinode entry's jobs into their group (pinned "group" or provider+flavor)."""
    for entry in p.get("multinode", []):
        timeout = entry.get("timeout")
        run_seg = f"{entry['config']}_{entry['scale']}"
        for job, flavor in entry["jobs"]:
            flavors = [flavor] if isinstance(flavor, str) else list(flavor)
            # Tag by the last (most-specific) flavor; a modifier like valgrind has no label.
            tag = FLAVOR_LABEL.get(flavors[-1], flavors[-1])
            name = f"{p['name']}:{_provider(job)}:{run_seg}:{tag}"
            flavor_flags = " ".join(f"--flavor {f}" for f in flavors)
            args = f"{job} --nodes {entry['nodes']} --config {entry['config']} {flavor_flags}"
            g = entry.get("group") or _pretrain_group(job, flavors[0])
            groups[g].append(_job(name, [".run_job", runner], job_args=args, timeout=timeout))
            # valgrind (memcheck) and nvrx-perf produce no baseline -> no update sibling.
            skip_update = "valgrind" in flavors or (job == "pretrain-nvrx" and "perf" in flavors)
            if entry["with_update"] and not skip_update:
                groups[g].append(_update(name, runner, timeout=timeout))


def _emit_platform(p: dict, groups: dict[str, list[str]]) -> None:
    """Emit every job (and update sibling) for one platform into its group."""
    runner = f".runner_{p['name']}"
    name, gpus = p["name"], p["gpus"]

    ckpt = f"{name}:ckptbench:{gpus}"
    groups["ckptbench"] += [_job(ckpt, [".run_ckptbench", runner]), _update(ckpt, runner)]

    standard = (
        ("pretrain-native", "deterministic"),
        ("pretrain-native", "perf"),
        ("pretrain-nvrx", "deterministic"),
    )
    for job, flavor in standard:
        jname = f"{name}:{_provider(job)}:pico_{gpus}:{FLAVOR_LABEL.get(flavor, flavor)}"
        args = f"{job} --config pico --flavor {flavor}"
        g = _pretrain_group(job, flavor)
        groups[g] += [_job(jname, [".run_job", runner], job_args=args), _update(jname, runner)]

    # valgrind memcheck: no baseline, so no update sibling.
    vg = f"{name}:valgrind:pico_{gpus}"
    groups["valgrind"].append(
        _job(vg, [".run_job", runner], job_args="pretrain-native --config pico --flavor deterministic --flavor valgrind"),
    )
    # 2-node valgrind exercises the cross-node server merge under memcheck; multinode platforms only.
    if p.get("multinode"):
        per_node = gpus.split("x")[1]
        groups["valgrind"].append(
            _job(
                f"{name}:valgrind:pico_2x{per_node}",
                [".run_job", runner],
                job_args="pretrain-native --nodes 2 --config pico --flavor deterministic --flavor valgrind",
                timeout="1d",
            ),
        )

    # nvrx perf smoke; no baseline (emits checkpoint times, not parseable bw metrics).
    nvrx_perf = f"{name}:nvrx:pico_{gpus}:{FLAVOR_LABEL['perf']}"
    groups["nvrx-perf"].append(
        _job(nvrx_perf, [".run_job", runner], job_args="pretrain-nvrx --config pico --flavor perf"),
    )

    _emit_multinode(p, runner, groups)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    header = _read_header().rstrip() + "\n"
    groups: dict[str, list[str]] = {g: [] for g in GROUPS}
    for p in PLATFORMS:
        _emit_platform(p, groups)
    written: list[tuple[str, int]] = []
    for g in GROUPS:
        if not groups[g]:
            continue
        path = Path.cwd() / f"generated-{g}.yml"
        path.write_text(f"{header}\n{HEADER_MARKER} (group: {g})\n\n" + "\n\n".join(groups[g]) + "\n")
        written.append((path.name, len(groups[g])))
    for fname, n in written:
        logger.info(f"wrote {fname} ({n} blocks)")


if __name__ == "__main__":
    main()
