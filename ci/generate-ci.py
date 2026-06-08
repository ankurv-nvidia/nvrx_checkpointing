#!/usr/bin/env python3
"""Generate the native CI child pipelines from a PLATFORMS table.

The hand-written ``gitlab-ci-native.yml`` carries only the shared templates
(``.ci_base``, ``.run_job``, ``.run_ckptbench``, ``.manual_optional``,
``.update_baseline_base``, ``.runner_*``) plus the bootstrap jobs that run this
script and trigger its output as child pipelines. Everything above the
``HEADER_MARKER`` line in that file is the shared header; this script copies it
verbatim and appends the concrete jobs from ``PLATFORMS``, bucketed by category
(see ``GROUPS``). It writes one ``generated-<group>.yml`` per non-empty group —
each a complete standalone pipeline — into the cwd, which the
``generate-native-pipeline`` job collects as artifacts and the per-group
trigger jobs (one per ``GROUPS`` entry) fire as child pipelines.

Adding a hardware platform = appending one dict to ``PLATFORMS``. No other YAML
edit is required: the standard per-platform jobs (ckptbench, pretrain-native
deterministic/perf, pretrain-nvrx deterministic, valgrind) and any multinode
profile job-sets are all emitted from the table.

Run standalone to preview:  cd "$(mktemp -d)" && python3 .../generate-ci.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_CI = HERE / "gitlab-ci-native.yml"

# Everything in PARENT_CI above this line is the shared header (templates) copied
# verbatim into the child pipeline; the bootstrap jobs that follow it stay in the
# parent only (the child must not re-trigger itself).
HEADER_MARKER = "# >>> generate-ci.py: concrete jobs are generated below this line <<<"


# ---------------------------------------------------------------------------
# Platform table — one dict per hardware platform.
#
#   name         runner tag (.runner_<name>) and job-name segment, e.g. "gb200"
#   gpus         node x gpu scale segment for the single-node jobs, e.g. "1x4"
#   has_image    whether this platform builds an enroot image. Retained for
#                documentation: prepare_image.py now runs in .run_job.before_script
#                for every platform (a no-op on non-SLURM runners), so it no longer
#                gates job generation.
#   multinode    optional list of scaled profile job-sets (see _emit_multinode). A
#                multinode entry may pin all its jobs to one group via "group"
#                (e.g. the expensive nano/16-node sets live in their own manual group).
#
# Manual-vs-auto is NOT a per-job property here: it is decided by the GROUP a job lands
# in — each group has one trigger job in gitlab-ci-native.yml that is either automatic
# or optional-manual (.manual_optional). Every concrete job runs automatically within
# its group and keeps .run_job's exit-code gating; an opt-in group simply isn't spawned
# until its trigger is played. See the GROUPS comment.
# ---------------------------------------------------------------------------
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
            # 2-node deterministic smoke (pico model at 2 nodes). No "group", so each job
            # lands in its provider+flavor group: native:deterministic in the automatic
            # native-deterministic group, nvrx:deterministic in the manual nvrx-deterministic
            # group (it seeds the cross-provider metadata/2x4_* baseline the native job
            # compares against). Both have update_baselines siblings.
            {
                "key": "2node",
                "config": "pico",
                "nodes": 2,
                "scale": "2x4",
                "with_update": True,
                "jobs": [("pretrain-native", "deterministic"), ("pretrain-nvrx", "deterministic")],
            },
            # nano: 8 nodes (EP=32). Expensive, so all its jobs go in the dedicated manual
            # "8node-nano" group (not the automatic native-* groups). update_baselines siblings.
            {
                "key": "nano",
                "config": "nano",
                "nodes": 8,
                "scale": "8x4",
                "with_update": True,
                "group": "8node-nano",
                "jobs": [
                    ("pretrain-native", "deterministic"),
                    ("pretrain-native", "perf"),
                    ("pretrain-nvrx", "deterministic"),
                ],
            },
            # multinode-perf: ultra at 16 nodes, a single perf smoke, no update. Manual
            # "16node-ultra" group.
            {
                "key": "multinode-perf",
                "config": "ultra",
                "nodes": 16,
                "scale": "16x4",
                "with_update": False,
                "group": "16node-ultra",
                "jobs": [("pretrain-native", "perf")],
            },
        ],
    },
    {
        "name": "h100",
        "gpus": "1x8",
        "has_image": True,
        "multinode": [
            # 2-node deterministic smoke (pico at 2 nodes); native in native-deterministic
            # (auto), nvrx in nvrx-deterministic (manual, seeds metadata/2x8_*). See gb200 2node.
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


def _job(name: str, extends: list[str], *, job_args: str | None = None) -> str:
    """Render one concrete job. Every job runs automatically within its group and keeps
    .run_job's allow_failure: exit_codes:[3] gating (a crash, exit != 3, fails the pipeline);
    opt-in is controlled by the group's trigger, not the job. So no job is when: manual."""
    lines = [f"{name}:", f"  extends: [{', '.join(extends)}]"]
    if job_args is not None:
        lines += ["  variables:", f'    JOB_ARGS: "{job_args}"']
    return "\n".join(lines)


def _update(test_name: str, runner: str) -> str:
    """Render the update_baselines sibling for a test job."""
    return "\n".join(
        [
            f"{test_name}:update_baselines:",
            f"  extends: [.update_baseline_base, {runner}]",
            "  needs:",
            f"    - job: {test_name}",
            "      artifacts: true",
        ],
    )


# ---------------------------------------------------------------------------
# Job grouping. native-tests is split into one child pipeline per category so the
# parent pipeline shows separate, independently-retriggerable groups. A single-node or
# 2-node pretrain job's group is "<provider>-<flavor>"; ckptbench is its own group; the
# expensive multinode sets pin themselves to a manual group via the entry "group" key
# (nano -> "8node-nano", the 16-node perf smoke -> "16node-ultra").
#
# Whether a group runs automatically or on demand is decided ENTIRELY by its trigger job
# in gitlab-ci-native.yml — automatic for ckptbench / native-deterministic / native-perf,
# optional-manual (.manual_optional) for nvrx-deterministic / nvrx-perf / valgrind /
# 16node-ultra / 8node-nano. Concrete jobs are never when: manual (that would block an automatic group's child
# on an unplayed job and, with exit-code gating, mask nothing while stalling everything);
# they run automatically inside whichever group is spawned and keep .run_job's exit-code
# gating, so a real failure fails that child rather than hiding as an allowed warning.
#
# Each non-empty group is written as generated-<group>.yml — a complete standalone pipeline
# (shared header + that group's jobs). The trigger's display name uses the flavor-tag form
# ([DETR].native, [PERF].native, [DETR].nvrx, [PERF].nvrx) while still referencing
# generated-<group>.yml. Keep this list in sync with those trigger jobs.
# ---------------------------------------------------------------------------
GROUPS = ("ckptbench", "native-deterministic", "native-perf", "nvrx-deterministic", "nvrx-perf", "valgrind", "16node-ultra", "8node-nano")

# Short display tags for the flavor segment of job names (…:perf -> …:[PERF],
# …:deterministic -> …:[DETR]). The real flavor still drives JOB_ARGS (--flavor …) and
# the group; only the visible job name is abbreviated. run_job.py's _parse_job_filter
# maps these back so baseline auto-detection still works.
FLAVOR_LABEL = {"deterministic": "[DETR]", "perf": "[PERF]"}


def _provider(job: str) -> str:
    """'pretrain-native' -> 'native', 'pretrain-nvrx' -> 'nvrx'."""
    return job.split("-", 1)[1]


def _pretrain_group(job: str, flavor: str) -> str:
    """Default group for a single-node or 2-node pretrain job: <provider>-<flavor>."""
    return f"{_provider(job)}-{flavor}"


def _emit_multinode(p: dict, runner: str, groups: dict[str, list[str]]) -> None:
    """Bucket each multinode profile entry's jobs into their groups.

    Each entry runs a fixed --config profile at --nodes N. An entry with a "group" pins all
    its jobs to that one (manual) group — used for the expensive opt-in sets (nano, the
    16-node perf smoke). Otherwise each job lands in its provider+flavor group via
    _pretrain_group (e.g. the 2-node pair: native:deterministic in the automatic
    native-deterministic group, nvrx:deterministic in the manual nvrx-deterministic group).
    ``with_update`` adds an update_baselines sibling alongside the job.
    """
    for entry in p.get("multinode", []):
        for job, flavor in entry["jobs"]:
            name = f"{p['name']}:{_provider(job)}:{entry['scale']}:{FLAVOR_LABEL.get(flavor, flavor)}"
            args = f"{job} --nodes {entry['nodes']} --config {entry['config']} --flavor {flavor}"
            g = entry.get("group") or _pretrain_group(job, flavor)
            groups[g].append(_job(name, [".run_job", runner], job_args=args))
            if entry["with_update"]:
                groups[g].append(_update(name, runner))


def _emit_platform(p: dict, groups: dict[str, list[str]]) -> None:
    """Bucket every job (and update sibling) for one platform into its group. Opt-in vs
    automatic is the group's trigger, not the job — so nothing here is marked manual."""
    runner = f".runner_{p['name']}"
    name, gpus = p["name"], p["gpus"]

    # ckptbench — its own entry script and group.
    ckpt = f"{name}:ckptbench:{gpus}"
    groups["ckptbench"] += [_job(ckpt, [".run_ckptbench", runner]), _update(ckpt, runner)]

    # pretrain-native (deterministic, perf) + pretrain-nvrx (deterministic): each in its
    # provider+flavor group, with an update_baselines sibling.
    standard = (
        ("pretrain-native", "deterministic"),
        ("pretrain-native", "perf"),
        ("pretrain-nvrx", "deterministic"),
    )
    for job, flavor in standard:
        jname = f"{name}:{_provider(job)}:{gpus}:{FLAVOR_LABEL.get(flavor, flavor)}"
        args = f"{job} --config pico --flavor {flavor}"
        g = _pretrain_group(job, flavor)
        groups[g] += [_job(jname, [".run_job", runner], job_args=args), _update(jname, runner)]

    # valgrind — memcheck, in the manual valgrind group, no update sibling. It keeps
    # .run_job's allow_failure: exit_codes:[3], so a memcheck error or stall (exit 42 /
    # RuntimeError — anything != 3) fails its child rather than being masked.
    vg = f"{name}:valgrind:{gpus}"
    groups["valgrind"].append(
        _job(vg, [".run_job", runner], job_args="pretrain-native --config pico --flavor deterministic --flavor valgrind"),
    )

    # nvrx perf — a fast-path (flash + fusion) smoke of the nvrx provider, in the manual
    # nvrx-perf group. No update sibling: the nvrx provider emits checkpoint times, not
    # parseable bw metrics, so its run prints those for inspection.
    nvrx_perf = f"{name}:nvrx:{gpus}:{FLAVOR_LABEL['perf']}"
    groups["nvrx-perf"].append(
        _job(nvrx_perf, [".run_job", runner], job_args="pretrain-nvrx --config pico --flavor perf"),
    )

    _emit_multinode(p, runner, groups)


def main() -> None:
    header = _read_header().rstrip() + "\n"
    groups: dict[str, list[str]] = {g: [] for g in GROUPS}
    for p in PLATFORMS:
        _emit_platform(p, groups)
    # One standalone child pipeline per non-empty group, written to the cwd
    # (the CI job runs from $CI_PROJECT_DIR; artifacts: generated-*.yml collects them).
    written: list[tuple[str, int]] = []
    for g in GROUPS:
        if not groups[g]:
            continue
        path = Path.cwd() / f"generated-{g}.yml"
        path.write_text(f"{header}\n{HEADER_MARKER} (group: {g})\n\n" + "\n\n".join(groups[g]) + "\n")
        written.append((path.name, len(groups[g])))
    for fname, n in written:
        print(f"wrote {fname} ({n} blocks)")


if __name__ == "__main__":
    main()
