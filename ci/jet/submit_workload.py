#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: T201
"""Submit one JET image build or workload, stream its log, and exit with its verdict."""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

import generate_workloads
from workload_matrix import JetPlatform, jet_cluster, jet_platform

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

PIPELINE_FILE = Path(".jet-pipeline")
RESULTS_DIR = Path("jet-results")
IMAGE_ENV_FILE = Path("jet-image.env")
SOFT_FAIL_RC = 3
POLL_SECONDS = 30
TRACE_BYTES_PER_MINUTE = 1_000_000
MIRROR_BUDGET_BYTES = 3_000_000
MAX_POLL_FAILURES = 20
LOG_WAIT_SECONDS = 600
TERMINAL_STATUSES = frozenset({"success", "failed", "canceled", "skipped"})
TRACE_STATUSES = frozenset({"running", "success", "failed", "canceled"})
WORKLOAD_PHASE = 2
WHEEL_VARIABLES = ("UPSTREAM_WHEEL_X86_64_ARTIFACT_ID", "UPSTREAM_WHEEL_AARCH64_ARTIFACT_ID")
JOB_NAME = re.compile(r"^(?P<key>(?:build|recipe)/\S+) .*? \d+ \[(?P<phase>\d+) [^]]+]$")
OTHER_RANK = re.compile(r"^(?:\x1b\[[0-9;]*[A-Za-z])*[1-9]\d*: ")
SIGNIFICANT = re.compile(r"Traceback|Error|FAIL|PASS|WARNING|\[slurm_job|\[jet\]|\[native_ckpt:run\]|differences")
SOFT_FAILURE_BANNER = "JET PERFORMANCE BASELINE FAILED"
EARLIEST = datetime.min.replace(tzinfo=UTC)


class GitLabJobs(Protocol):
    """The GitLab calls that follow and cancel one JET pipeline."""

    def get_pipeline(self, pipeline_id: int) -> dict: ...

    def get_pipeline_bridges(self, pipeline_id: int) -> list[dict]: ...

    def get_all_pipeline_jobs(self, pipeline_id: int) -> Iterable[dict]: ...

    def get_job_trace(self, job_id: int) -> str: ...

    def cancel_pipeline(self, pipeline_id: int) -> None: ...


def visible(line: str) -> bool:
    """Keep rank 0, unranked, and diagnostic lines."""
    return not OTHER_RANK.match(line) or bool(SIGNIFICANT.search(line))


@dataclass
class TraceCursor:
    """Position in one GitLab job trace that has already been printed."""

    job_id: int
    offset: int = 0
    done: bool = False

    def advance(self, trace: str, *, final: bool) -> list[str]:
        """Return the new complete lines, and the unterminated tail once the job has finished."""
        end = len(trace) if final else trace.rfind("\n", self.offset) + 1
        self.done = final
        if end <= self.offset:
            return []
        chunk = trace[self.offset:end]
        self.offset = end
        lines = chunk.split("\n")
        if chunk.endswith("\n"):
            lines.pop()
        return lines


@dataclass
class FollowResult:
    """Terminal state of a submitted JET pipeline as seen through its GitLab jobs."""

    pipeline_status: str
    jobs: dict[int, dict[str, Any]] = field(default_factory=dict)
    soft_failure_banner: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.pipeline_status == "success"
            and WORKLOAD_PHASE in self.jobs
            and all(job["status"] == "success" for job in self.jobs.values())
        )


class Follower:
    """Stream one workload's JET step log into this job's log until its pipeline finishes."""

    def __init__(  # noqa: PLR0913
        self,
        gitlab: GitLabJobs,
        pipeline_id: int,
        key: str,
        *,
        out: TextIO | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.gitlab = gitlab
        self.pipeline_id = pipeline_id
        self.key = key
        self.out = out or sys.stdout
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.cursor: TraceCursor | None = None
        self.banner = False
        self.mirrored = 0
        self.budget_noted = False

    def _child_pipeline(self) -> int | None:
        for bridge in self.gitlab.get_pipeline_bridges(self.pipeline_id):
            if child := bridge.get("downstream_pipeline"):
                return int(child["id"])
        return None

    def _workload_jobs(self, child_id: int) -> dict[int, dict[str, Any]]:
        latest: dict[int, dict[str, Any]] = {}
        for job in self.gitlab.get_all_pipeline_jobs(child_id):
            match = JOB_NAME.match(str(job.get("name", "")))
            if match is None or match["key"] != self.key:
                continue
            phase = int(match["phase"])
            if phase not in latest or int(job["id"]) > int(latest[phase]["id"]):
                latest[phase] = job
        return latest

    def _print(self, message: str) -> None:
        self.out.write(f"{message}\n")
        self.out.flush()

    def _mirror(self, lines: list[str], step: dict[str, Any]) -> None:
        for line in lines:
            self.banner |= SOFT_FAILURE_BANNER in line
            if not visible(line):
                continue
            if self.mirrored >= MIRROR_BUDGET_BYTES and not SIGNIFICANT.search(line):
                if not self.budget_noted:
                    self.budget_noted = True
                    self._print(f"[jet] log budget reached; mirroring diagnostics only; full log: {step.get('web_url', '')}")
                continue
            self.mirrored += len(line) + 1
            self.out.write(f"{line}\n")
        self.out.flush()

    def _follow_step(self, step: dict[str, Any]) -> int:
        """Print new lines of the latest attempt and return the trace size that was read."""
        if self.cursor is not None and self.cursor.job_id != int(step["id"]):
            if not self.cursor.done:
                self._mirror(self.cursor.advance(self.gitlab.get_job_trace(self.cursor.job_id), final=True), step)
            self._print(f"[jet] JET retried the workload; following {step.get('web_url', step['id'])}")
            self.cursor = None
            self.banner = False
        if self.cursor is None:
            self.cursor = TraceCursor(int(step["id"]))
        if self.cursor.done:
            return 0
        trace = self.gitlab.get_job_trace(int(step["id"]))
        self._mirror(self.cursor.advance(trace, final=step["status"] in TERMINAL_STATUSES), step)
        return len(trace)

    def _poll(self, announced: set[tuple[int, str]]) -> tuple[str, dict[int, dict[str, Any]], int]:
        pipeline = self.gitlab.get_pipeline(self.pipeline_id)
        child_id = self._child_pipeline()
        jobs = self._workload_jobs(child_id) if child_id else {}
        for phase in sorted(jobs):
            job = jobs[phase]
            if (int(job["id"]), job["status"]) not in announced:
                announced.add((int(job["id"]), job["status"]))
                self._print(f"[jet] {job['name']}: {job['status']} {job.get('web_url', '')}")
        step = jobs.get(WORKLOAD_PHASE)
        trace_size = self._follow_step(step) if step is not None and step["status"] in TRACE_STATUSES else 0
        return pipeline["status"], jobs, trace_size

    def run(self) -> FollowResult:
        announced: set[tuple[int, str]] = set()
        failures = 0
        while True:
            try:
                status, jobs, trace_size = self._poll(announced)
            except Exception as error:
                failures += 1
                if failures > MAX_POLL_FAILURES:
                    raise
                self._print(f"[jet] polling JET failed ({failures}/{MAX_POLL_FAILURES}); retrying: {error}")
                self.sleep(self.poll_seconds)
                continue
            failures = 0
            if status in TERMINAL_STATUSES:
                return FollowResult(status, jobs, self.banner)
            self.sleep(max(self.poll_seconds, trace_size * 60 / TRACE_BYTES_PER_MINUTE))


def upstream_variables(environ: dict[str, str]) -> dict[str, str]:
    """Describe the GitHub Actions run with the upstream variables the JET manifests read."""
    repository = environ.get("GITHUB_REPOSITORY")
    if not repository:
        return {}
    values = {
        "UPSTREAM_PROJECT_URL": f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}",
        "UPSTREAM_PROJECT_ID": environ.get("GITHUB_REPOSITORY_ID"),
        "UPSTREAM_PROJECT_PATH": repository,
        "UPSTREAM_PIPELINE_ID": environ.get("GITHUB_RUN_ID"),
        "UPSTREAM_PIPELINE_SOURCE": environ.get("GITHUB_EVENT_NAME"),
        "UPSTREAM_COMMIT_SHA": environ.get("GITHUB_SHA"),
        "UPSTREAM_COMMIT_REF_NAME": environ.get("GITHUB_HEAD_REF") or environ.get("GITHUB_REF_NAME"),
    }
    return {name: value for name, value in values.items() if value}


def submission_variables(environ: dict[str, str]) -> dict[str, str]:
    """Variables the JET pipeline needs: the upstream run, its exact wheels, and the API mode."""
    values = upstream_variables(environ)
    values.update({name: environ[name] for name in WHEEL_VARIABLES if environ.get(name)})
    values.update(JET_API_TAG="v3", JET_SCRIPT_GENERATION_VALIDATION="error")
    return values


def manifests(
    platform: JetPlatform,
    *,
    maintainer: str,
    workload_key: str | None,
    image_tag: str | None,
) -> tuple[str, list[dict]]:
    """Return the JET key this job follows and the manifests it submits."""
    if workload_key is None:
        return f"build/{platform.name}-ci-image", [generate_workloads.build_manifest(platform, maintainer)]
    if not image_tag:
        raise ValueError(f"no built image for {platform.name}; the jet-image job must pass its tag")
    workload = generate_workloads.find_workload(platform, workload_key)
    recipe = generate_workloads.recipe_manifest(platform, workload, maintainer, image_tag=image_tag)
    return f"recipe/{workload.display_name}", [recipe]


def image_variable(platform: JetPlatform) -> str:
    return f"JET_IMAGE_TAG_{platform.name.upper()}"


def workload_log(logs: Iterable[Any], key: str, job_id: int | None) -> Any | None:  # noqa: ANN401
    """Return the JET log of one workload attempt, or the newest attempt when the job is unknown."""
    matching = [log for log in logs if log.log.get("workload", {}).get("key") == key]
    if job_id is not None:
        return next((log for log in matching if (log.log.get("ci") or {}).get("job_id") == job_id), None)
    return max(matching, key=lambda log: log.log.get("created") or EARLIEST, default=None)


def describe(log: Any) -> list[str]:  # noqa: ANN401
    """Summarize a JET log for the job log."""
    payload = log.log
    status = payload.get("status") or {}
    slurm = payload.get("slurm") or {}
    lines = [
        f"[jet] JET status: {status.get('name', 'unknown')} ({status.get('code', '?')}) {status.get('message', '')}".rstrip(),
        f"[jet] JET log: {_logs_url()}/{payload.get('id', '')}",
    ]
    if slurm:
        lines.append(
            f"[jet] Slurm job {slurm.get('job', '?')}: queued {_duration(slurm.get('queue_time'))}, "
            f"ran {_duration(slurm.get('duration'))}, state {slurm.get('state', '?')}",
        )
    return lines


def _logs_url() -> str:
    from jetclient.utils.get_url import get_backend_url

    return get_backend_url("prod").removesuffix("/api") + "/ui/logs"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes, remainder = divmod(round(float(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{remainder:02d}s"


def publish_results(log: Any, destination: Path = RESULTS_DIR) -> bool:  # noqa: ANN401
    """Extract the workload's results archive into the job artifacts; return whether it reports a soft failure."""
    archive = next((asset for asset in log.get_assets() if asset.name.endswith("_artifacts.tar.gz")), None)
    if archive is None:
        print("[jet] the workload published no results archive")
        return False
    destination.mkdir(parents=True, exist_ok=True)
    local = destination / Path(archive.name).name
    try:
        archive.download(local)
        with tarfile.open(local, "r:gz") as contents:
            contents.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as error:
        print(f"[jet] could not publish the results archive {archive.name}: {error}")
        return False
    finally:
        local.unlink(missing_ok=True)
    return any(destination.glob("ci/ci-results/*_soft_failure.json"))


def verdict(result: FollowResult, *, soft_failure: bool, log_code: str | None) -> int:
    """Return 0 for a pass, 3 for a performance soft failure, and 1 otherwise."""
    if not result.succeeded or (log_code is not None and log_code != "0"):
        return 1
    return SOFT_FAIL_RC if soft_failure or result.soft_failure_banner else 0


def wait_for_log(pipeline: Any, key: str, job_id: int | None, *, sleep: Callable[[float], None] = time.sleep) -> Any | None:  # noqa: ANN401
    """Wait for JET to index the workload's log after its pipeline finished."""
    deadline = time.monotonic() + LOG_WAIT_SECONDS
    while True:
        try:
            logs = list(pipeline.get_logs())
        except Exception as error:  # noqa: BLE001
            print(f"[jet] reading JET logs failed; retrying: {error}", flush=True)
            logs = []
        if (log := workload_log(logs, key, job_id)) is not None:
            return log
        if time.monotonic() >= deadline:
            return workload_log(logs, key, None)
        sleep(POLL_SECONDS)


def _client() -> Any:  # noqa: ANN401
    from jetclient import JETClient

    client = JETClient(gitlab_ci_token=os.environ.get("JET_GITLAB_TOKEN") or None)
    if not client.has_gitlab_ci_token:
        raise SystemExit(
            "JET needs the submitting user's GitLab token: store a personal access token with api scope in "
            "JET Secrets at jet/gitlab-personal-access-token, or provide it as the JET_GITLAB_TOKEN secret",
        )
    return client


def _gitlab(client: Any) -> GitLabJobs:  # noqa: ANN401
    from jetclient.clients.gitlab import GitlabClient
    from jetclient.services.config import GITLAB_CI_PROJECT_ID

    return GitlabClient(GITLAB_CI_PROJECT_ID, client.get_gitlab_ci_token())


def cancel(pipeline_id: int) -> None:
    _gitlab(_client()).cancel_pipeline(pipeline_id)
    print(f"[jet] canceled JET pipeline {pipeline_id}")


def _print_generation(client: Any, pipeline_id: int) -> None:  # noqa: ANN401
    details = client.workloads.get_pipeline(pipeline_id).generation_details
    if details is None:
        return
    print(f"[jet] JET generate job: {details.job_url}", flush=True)
    if details.job_trace:
        print(details.job_trace[-20000:], flush=True)


def submit(args: argparse.Namespace) -> int:
    platform = jet_platform(args.platform)
    cluster = jet_cluster(platform.name)
    maintainer = os.environ["JET_CI_MAINTAINER"]
    key, documents = manifests(
        platform,
        maintainer=maintainer,
        workload_key=args.workload,
        image_tag=os.environ.get(image_variable(platform)),
    )
    client = _client()
    pipeline = client.workloads.submit(
        documents,
        config_id=cluster.config,
        custom_config=generate_workloads.jet_custom_config(platform),
        additional_variables=submission_variables(dict(os.environ)),
        wait_for_validation=False,
    )
    PIPELINE_FILE.write_text(f"{pipeline.jet_id}\n")
    print(f"[jet] submitted {key} to {cluster.config} as JET pipeline {pipeline.jet_id}", flush=True)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as output:
            output.write(f"`{key}` was submitted to JET config `{cluster.config}`.\n")
    gitlab = _gitlab(client)

    def cancel_on_signal(signum: int, _frame: object) -> None:
        gitlab.cancel_pipeline(pipeline.jet_id)
        print(f"[jet] canceled the JET pipeline after signal {signum}", flush=True)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, cancel_on_signal)
    signal.signal(signal.SIGINT, cancel_on_signal)
    result = Follower(gitlab, pipeline.jet_id, key).run()
    PIPELINE_FILE.unlink(missing_ok=True)
    print(f"[jet] JET pipeline finished: {result.pipeline_status}", flush=True)
    if not result.jobs:
        _print_generation(client, pipeline.jet_id)
        return 1
    step_id = int(result.jobs[WORKLOAD_PHASE]["id"]) if WORKLOAD_PHASE in result.jobs else None
    log = wait_for_log(client.workloads.get_pipeline(pipeline.jet_id), key, step_id)
    log_code = None
    if log is None:
        print("[jet] JET did not index the workload's log in time", flush=True)
    else:
        print("\n".join(describe(log)), flush=True)
        log_code = str((log.log.get("status") or {}).get("code", "")) or None
    if args.workload is None:
        if not result.succeeded:
            return 1
        image = (log.log.get("output_image") or {}).get("name") if log is not None else None
        if not image:
            print("[jet] the build reported no output image", flush=True)
            return 1
        IMAGE_ENV_FILE.write_text(f"{image_variable(platform)}={image}\n")
        print(f"[jet] built image {image}", flush=True)
        return 0
    soft_failure = publish_results(log) if log is not None else False
    return verdict(result, soft_failure=soft_failure, log_code=log_code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", help="JET platform name from ci/jobs.json")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", action="store_true", help="build the platform image")
    target.add_argument("--workload", help="JET key of the workload to run")
    target.add_argument("--cancel", type=int, metavar="PIPELINE", help="cancel a submitted JET pipeline")
    args = parser.parse_args(argv)
    if args.cancel is not None:
        cancel(args.cancel)
        return 0
    if not args.platform:
        parser.error("--platform is required to submit")
    return submit(args)


if __name__ == "__main__":
    sys.exit(main())
