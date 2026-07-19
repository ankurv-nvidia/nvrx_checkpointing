#!/usr/bin/env python3
"""Set up a GitLab CI runner on a SLURM cluster login node.

All state lives in CI_DIR (default: $HOME/data/ci).
Sensitive config (token, URL) must be written to files there before running setup —
nothing is read from environment variables.

File layout under CI_DIR:
  token             Registration token (chmod 600). Written by you before running setup.
  config            Key=value settings: GITLAB_URL, PLATFORM (required; hardware platform e.g.
                    gb200/h100/h100/devbox — the GitLab runner tag that routes jobs + baseline
                    namespace), CLUSTER (required; datacenter e.g. cwdfw/ocihsg — selects the srun
                    templates), CONCURRENT, GPUS_PER_NODE, SLURM_ACCOUNT (set => this cluster runs
                    via SLURM; also the srun --account)
  srun.tmpl            (optional) per-cluster override of the job-launch srun template.
  enroot_import.tmpl   (optional) per-cluster override of the base-image import srun command.
  image_prep.tmpl      (optional) per-cluster override of the project-image build srun command.
                       Drop any of these template files here to override the repo default for
                       this cluster; if absent, run_job.py uses the default shipped in
                       ci/templates/. Write account, partition, mounts and
                       GPU-request syntax directly into the file.
  gitlab-runner     The runner binary (downloaded by setup).
  config.toml       Runner config (written by setup after registration).
  runner.pid        PID of the running process.
  runner.log        Runner log output.
  builds/           Job build directories.
  cache/            Job cache.

Subcommands:
  setup      Download binary + register + start  (run once per login node)
  start      Start the runner (after reboot or stop)
  stop       Gracefully stop the runner
  status     Show PID, uptime, and recent log lines
  cleanup    Stop, unregister from GitLab, and delete all state under CI_DIR

Usage:
  python3 setup_runner.py setup   [--ci-dir DIR] [config flags below]
  python3 setup_runner.py start   [--ci-dir DIR] [config flags below]
  python3 setup_runner.py stop    [--ci-dir DIR]
  python3 setup_runner.py status  [--ci-dir DIR]
  python3 setup_runner.py cleanup [--ci-dir DIR] [--yes]

Config flags (setup/start). Each can be set in the config file or here; a value
passed here overrides the config for this run AND is written back to the config,
so it becomes the default on the next run (CLI wins, then caches):
  --platform NAME       Hardware platform: GitLab runner tag + baseline namespace,
                        e.g. gb200, h100, h100, devbox                     (PLATFORM)
  --cluster NAME        Datacenter that selects the srun templates, e.g. cwdfw, ocihsg  (CLUSTER)
  --gitlab-url URL      GitLab base URL                                    (GITLAB_URL)
  --concurrent N        Max concurrent jobs for this runner                (CONCURRENT)
  --gpus-per-node N     GPUs per node for srun jobs on this cluster        (GPUS_PER_NODE)
  --slurm-account NAME      SLURM account; set => run via SLURM (srun --account)  (SLURM_ACCOUNT)

  Anything else cluster-specific (partition, container mounts, GPU-request syntax,
  whether GPUs are requested at all) lives in the srun.tmpl / enroot_import.tmpl /
  image_prep.tmpl override files next to this config — not in environment variables.
"""

import argparse
import os
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_CI_DIR = Path(os.environ.get("HOME")) / "data" / "ci"
DEFAULT_GITLAB_URL = "https://gitlab-master.nvidia.com"
DOWNLOAD_BASE = "https://gitlab-runner-downloads.s3.amazonaws.com"
ARCH_MAP = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


def die(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# File-based config helpers
# ---------------------------------------------------------------------------


def read_token(ci_dir: Path) -> str:
    token_file = ci_dir / "token"
    if not token_file.exists():
        die(
            f"Token file not found: {token_file}\n"
            f"  Create it first (token comes from GitLab project -> Settings -> CI/CD -> Runners -> New project runner):\n"
            f"    echo 'glrt-xxxx' > {token_file} && chmod 600 {token_file}",
        )
    token = token_file.read_text().strip()
    if not token:
        die(f"Token file is empty: {token_file}")
    return token


def read_config(ci_dir: Path) -> dict:
    """Read optional key=value config file. Returns dict with defaults filled in.

    Inline '# comment' suffixes and full-line comments are stripped, matching how
    bash `source`s this file, so e.g. `GPUS_PER_NODE=4  # note` yields just "4".
    """
    cfg = {
        "GITLAB_URL": DEFAULT_GITLAB_URL,
        "PLATFORM": "",
        "CLUSTER": "",
        "CONCURRENT": "4",
        "RUNNER_VERSION": "latest",
        "GPUS_PER_NODE": "",
        "SLURM_ACCOUNT": "",
    }
    config_file = ci_dir / "config"
    if config_file.exists():
        for raw in config_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()  # drop inline/full-line comments
            if line and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


# Config settings that can also be set on the command line. A CLI value (when
# given) overrides the config file for this run AND is written back so it becomes
# the default on the next run. Maps config key -> argparse dest.
CLI_CONFIG_KEYS = {
    "GITLAB_URL": "gitlab_url",
    "PLATFORM": "platform",
    "CLUSTER": "cluster",
    "CONCURRENT": "concurrent",
    "GPUS_PER_NODE": "gpus_per_node",
    "SLURM_ACCOUNT": "slurm_account",
}


def write_config(ci_dir: Path, updates: dict[str, str]) -> None:
    """Persist key=value `updates` into the config file (creating it if needed).

    Existing keys are updated in place — preserving any inline comment and the
    rest of the file (other keys, comments, blank lines) — and unknown keys are
    appended. This is what lets a CLI override become the default next run.
    """
    if not updates:
        return
    config_file = ci_dir / "config"
    lines = config_file.read_text().splitlines() if config_file.exists() else []
    pending = dict(updates)
    out = []
    for raw in lines:
        body = raw.split("#", 1)[0]
        key = body.split("=", 1)[0].strip() if "=" in body else ""
        if key in pending:
            comment = raw[raw.index("#") :] if "#" in raw else ""
            out.append(f"{key}={pending.pop(key)}" + (f"  {comment}" if comment else ""))
        else:
            out.append(raw)
    out.extend(f"{key}={val}" for key, val in pending.items())
    config_file.write_text("\n".join(out) + "\n")
    config_file.chmod(0o600)


# ---------------------------------------------------------------------------
# Path helpers (all under ci_dir)
# ---------------------------------------------------------------------------


def paths(ci_dir: Path) -> dict[str, Path]:
    return {
        "bin": ci_dir / "gitlab-runner",
        "config": ci_dir / "config.toml",
        "pid": ci_dir / "runner.pid",
        "log": ci_dir / "runner.log",
        "builds": ci_dir / "builds",
        "cache": ci_dir / "cache",
    }


def read_pid(ci_dir: Path) -> int | None:
    pf = paths(ci_dir)["pid"]
    try:
        return int(pf.read_text().strip()) if pf.exists() else None
    except ValueError:
        return None


def is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install(ci_dir: Path, version: str, force: bool) -> None:
    rb = paths(ci_dir)["bin"]
    if rb.is_file() and not force:
        subprocess.run([rb.as_posix(), "--version"], capture_output=True, check=True, text=True).stdout.splitlines()[0]
        return
    arch = ARCH_MAP.get(platform.machine()) or die(f"unsupported architecture {platform.machine()}")
    base = f"{DOWNLOAD_BASE}/latest/binaries" if version == "latest" else f"{DOWNLOAD_BASE}/v{version}/binaries"
    url = f"{base}/gitlab-runner-linux-{arch}"
    urllib.request.urlretrieve(url, rb)
    rb.chmod(rb.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run([rb.as_posix(), "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def register(ci_dir: Path, url: str, token: str, platform_tag: str, cluster: str, concurrent: int) -> None:
    p = paths(ci_dir)
    name = f"{platform_tag}-{cluster}-{socket.gethostname().split('.')[0]}-{os.environ.get('USER', 'runner')}"
    p["builds"].mkdir(parents=True, exist_ok=True)
    p["cache"].mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            p["bin"].as_posix(),
            "register",
            "--non-interactive",
            "--url",
            url,
            "--token",
            token,
            "--name",
            name,
            "--executor",
            "shell",
            "--shell",
            "bash",
            #        "--tag-list", tags, "--run-untagged=false", "--locked=false",
            "--config",
            p["config"].as_posix(),
            "--builds-dir",
            p["builds"].as_posix(),
            "--cache-dir",
            p["cache"].as_posix(),
        ],
        check=True,
    )
    # Patch top-level concurrent
    text = p["config"].read_text()
    text = (
        re.sub(r"^concurrent\s*=.*", f"concurrent = {concurrent}", text, flags=re.MULTILINE)
        if re.search(r"^concurrent\s*=", text, re.MULTILINE)
        else f"concurrent = {concurrent}\n\n" + text
    )
    p["config"].write_text(text)


# ---------------------------------------------------------------------------
# Start / stop / status / cleanup
# ---------------------------------------------------------------------------


def start(ci_dir: Path) -> None:
    p = paths(ci_dir)
    if not p["bin"].is_file():
        die(f"Binary not found at {p['bin']}. Run: python3 setup_runner.py setup")
    if not p["config"].is_file():
        die(f"config.toml not found at {p['config']}. Run: python3 setup_runner.py setup")
    pid = read_pid(ci_dir)
    if is_running(pid):
        return
    # Values come from the config file; main() has already persisted any CLI
    # overrides into it, so this also honors flags passed to setup/start.
    cfg = read_config(ci_dir)
    env = os.environ.copy()
    # Tell run_job.py where this runner's state lives so it finds override
    # templates (srun.tmpl / enroot_import.tmpl / image_prep.tmpl) placed next to the config.
    env["CI_DIR"] = ci_dir.as_posix()
    # Hardware platform (e.g. gb200): the GitLab runner tag + baseline-file namespace
    # that run_job.py reads via _platform_tag(). Fall back to CLUSTER for configs that
    # predate the PLATFORM/CLUSTER split.
    env["PLATFORM"] = cfg["PLATFORM"]
    # Datacenter (e.g. cwdfw): selects the per-cluster srun/enroot_import/image_prep templates.
    env["CLUSTER"] = cfg["CLUSTER"]
    # SLURM account: forwarded so run_job.py uses it as the srun --account and
    # treats this cluster as SLURM-based (its presence is the slurm-vs-inline switch).
    if cfg.get("SLURM_ACCOUNT", ""):
        env["SLURM_ACCOUNT"] = cfg["SLURM_ACCOUNT"]
        # SLURM partition: forwarded as the srun --partition; defaults to "batch".
    # GPUs per node for srun jobs.
    if cfg.get("GPUS_PER_NODE", ""):
        env["GPUS_PER_NODE"] = cfg["GPUS_PER_NODE"]
    with p["log"].open("a") as lf:
        proc = subprocess.Popen(
            [p["bin"].as_posix(), "run", "--config", p["config"].as_posix(), "--working-directory", ci_dir.as_posix()],
            stdout=lf,
            stderr=lf,
            start_new_session=True,
            env=env,
        )
    p["pid"].write_text(str(proc.pid))
    time.sleep(1)
    if is_running(proc.pid):
        pass
    else:
        die(f"Runner failed to start — check {p['log']}")


def stop(ci_dir: Path) -> None:
    pid = read_pid(ci_dir)
    if not is_running(pid):
        paths(ci_dir)["pid"].unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        time.sleep(1)
        if not is_running(pid):
            break
    if is_running(pid):
        os.kill(pid, signal.SIGKILL)
    else:
        pass
    paths(ci_dir)["pid"].unlink(missing_ok=True)


def status(ci_dir: Path, log_lines: int = 20) -> None:
    p = paths(ci_dir)
    read_pid(ci_dir)
    if p["log"].exists():
        p["log"].read_text().splitlines()[-log_lines:]


def cleanup(ci_dir: Path, yes: bool) -> None:
    if not yes:
        confirm = input(f"  Stop runner, unregister from GitLab, and delete {ci_dir}? [y/N] ")
        if confirm.lower() != "y":
            return
    stop(ci_dir)
    p = paths(ci_dir)
    if p["config"].exists():
        subprocess.run([p["bin"].as_posix(), "unregister", "--all-runners", "--config", p["config"].as_posix()], check=False)
    # Remove everything except the token file (user may want to re-register)
    for f in [p["config"], p["pid"], p["log"], p["bin"]]:
        f.unlink(missing_ok=True)
    for d in [p["builds"], p["cache"]]:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["setup", "start", "stop", "status", "cleanup"])
    p.add_argument("--ci-dir", default=DEFAULT_CI_DIR.as_posix(), help=f"Root directory for all CI state (default: {DEFAULT_CI_DIR})")
    # Settings that can come from the config file or the command line. When passed
    # here they override the config for this run AND are written back to the config
    # so they become the defaults next run. Defaults are None = "not provided".
    p.add_argument(
        "--platform",
        default=None,
        help="Hardware platform: GitLab runner tag + baseline namespace, e.g. gb200/h100/h100/devbox (persisted)",
    )
    p.add_argument("--cluster", default=None, help="Datacenter that selects the srun templates, e.g. cwdfw/ocihsg (persisted)")
    p.add_argument("--gitlab-url", default=None, dest="gitlab_url", help="GitLab base URL (persisted to config)")
    p.add_argument("--concurrent", default=None, help="Max concurrent jobs for this runner (persisted to config)")
    p.add_argument("--gpus-per-node", default=None, dest="gpus_per_node", help="GPUs per node for srun jobs (persisted to config)")
    p.add_argument(
        "--slurm-account",
        default=None,
        dest="slurm_account",
        help="SLURM account for srun --account; set => this cluster runs via SLURM (persisted to config)",
    )
    p.add_argument("--force-reinstall", action="store_true", help="Re-download binary even if already present")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt for cleanup")
    p.add_argument("--log-lines", type=int, default=20)
    args = p.parse_args()

    ci_dir = Path(args.ci_dir)
    ci_dir.mkdir(parents=True, exist_ok=True)

    # Collect CLI-provided settings (those actually passed) and persist them into
    # the config file for setup/start, so the merged values are cached for next run.
    overrides = {key: str(getattr(args, dest)) for key, dest in CLI_CONFIG_KEYS.items() if getattr(args, dest) is not None}
    if overrides and args.command in ("setup", "start"):
        write_config(ci_dir, overrides)

    # Read config after persisting, so cfg already reflects any CLI overrides.
    cfg = read_config(ci_dir)
    cluster = cfg["CLUSTER"]
    if not cluster:
        die("CLUSTER is not set. Add 'CLUSTER=<name>' to ${env:HOME}/data/ci/config or pass --cluster <name>.")
    if not cfg["PLATFORM"]:
        die("PLATFORM is not set. Add 'PLATFORM=<name>' to ${env:HOME}/data/ci/config or pass --platform <name>.")

    if args.command == "setup":
        token = read_token(ci_dir)
        install(ci_dir, cfg["RUNNER_VERSION"], args.force_reinstall)
        register(ci_dir, cfg["GITLAB_URL"], token, cfg["PLATFORM"], cluster, int(cfg["CONCURRENT"]))
        start(ci_dir)

    elif args.command == "start":
        start(ci_dir)
    elif args.command == "stop":
        stop(ci_dir)
    elif args.command == "status":
        status(ci_dir, args.log_lines)
    elif args.command == "cleanup":
        cleanup(ci_dir, args.yes)


if __name__ == "__main__":
    main()
