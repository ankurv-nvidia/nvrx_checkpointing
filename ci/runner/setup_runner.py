#!/usr/bin/env python3
"""Set up a GitLab CI runner on a SLURM cluster login node.

All state lives in CI_DIR (default: $HOME/data/ci).
Sensitive config (token, URL) must be written to files there before running setup —
nothing is read from environment variables.

File layout under CI_DIR:
  token           Registration token (chmod 600). Written by you before running setup.
  config          Key=value settings: GITLAB_URL, CLUSTER (required), CONCURRENT, SLURM_PARTITION (optional).
  gitlab-runner   The runner binary (downloaded by setup).
  config.toml     Runner config (written by setup after registration).
  runner.pid      PID of the running process.
  runner.log      Runner log output.
  builds/         Job build directories.
  cache/          Job cache.

Subcommands:
  setup      Download binary + register + start  (run once per login node)
  start      Start the runner (after reboot or stop)
  stop       Gracefully stop the runner
  status     Show PID, uptime, and recent log lines
  cleanup    Stop, unregister from GitLab, and delete all state under CI_DIR

Usage:
  python3 setup_runner.py setup   [--ci-dir DIR] [--cluster NAME] [--slurm-partition PARTITION] [--no-slurm-gpu-flags]
  python3 setup_runner.py start   [--ci-dir DIR] [--slurm-partition PARTITION] [--no-slurm-gpu-flags]
  python3 setup_runner.py stop    [--ci-dir DIR]
  python3 setup_runner.py status  [--ci-dir DIR]
  python3 setup_runner.py cleanup [--ci-dir DIR] [--yes]

  --no-slurm-gpu-flags  Pass this on runners whose SLURM does not support --gpus/--gpus-per-node.
                        Sets SLURM_GPU_FLAGS_ENABLED=0 in the runner environment so run_job.py
                        omits those flags and the job runs with all GPUs the scheduler allocates.
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
    """Read optional key=value config file. Returns dict with defaults filled in."""
    cfg = {"GITLAB_URL": DEFAULT_GITLAB_URL, "CLUSTER": "", "CONCURRENT": "4", "RUNNER_VERSION": "latest", "SLURM_PARTITION": ""}
    config_file = ci_dir / "config"
    if config_file.exists():
        for line in [line.strip() for line in config_file.read_text().splitlines()]:
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


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


def register(ci_dir: Path, url: str, token: str, cluster: str, extra_tags: str, concurrent: int) -> str:  # noqa: PLR0913
    p = paths(ci_dir)
    slurm_tag = f"slurm-{cluster}"
    ",".join(filter(None, [slurm_tag, "slurm", "shell", extra_tags]))
    name = f"{cluster}-{socket.gethostname().split('.')[0]}-{os.environ.get('USER', 'runner')}"
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
    return slurm_tag


# ---------------------------------------------------------------------------
# Start / stop / status / cleanup
# ---------------------------------------------------------------------------


def start(ci_dir: Path, slurm_partition: str = "", gpu_flags_enabled: bool = True) -> None:
    p = paths(ci_dir)
    if not p["bin"].is_file():
        die(f"Binary not found at {p['bin']}. Run: python3 setup_runner.py setup")
    if not p["config"].is_file():
        die(f"config.toml not found at {p['config']}. Run: python3 setup_runner.py setup")
    pid = read_pid(ci_dir)
    if is_running(pid):
        return
    cfg = read_config(ci_dir)
    partition = slurm_partition or cfg.get("SLURM_PARTITION", "")
    env = os.environ.copy()
    if partition:
        env["SLURM_PARTITION"] = partition
    if not gpu_flags_enabled:
        env["SLURM_GPU_FLAGS_ENABLED"] = "0"
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
    p.add_argument("--cluster", default="", help="Cluster tag suffix, e.g. h200, gb200 (required if not set in config file)")
    p.add_argument("--slurm-partition", default="", help="SLURM partition for srun jobs (overrides SLURM_PARTITION in config file)")
    p.add_argument(
        "--no-slurm-gpu-flags",
        action="store_true",
        dest="no_slurm_gpu_flags",
        help="Omit --gpus/--gpus-per-node from srun commands (for runners where the scheduler allocates all GPUs automatically)",
    )
    p.add_argument("--tags", default="", help="Extra comma-separated runner tags")
    p.add_argument("--force-reinstall", action="store_true", help="Re-download binary even if already present")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt for cleanup")
    p.add_argument("--log-lines", type=int, default=20)
    args = p.parse_args()

    ci_dir = Path(args.ci_dir)
    ci_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_config(ci_dir)
    cluster = args.cluster or cfg["CLUSTER"]
    if not cluster:
        die("CLUSTER is not set. Add 'CLUSTER=<name>' to ${env:HOME}/data/ci/config or pass --cluster <name>.")

    if args.command == "setup":
        token = read_token(ci_dir)
        install(ci_dir, cfg["RUNNER_VERSION"], args.force_reinstall)
        register(ci_dir, cfg["GITLAB_URL"], token, cluster, args.tags, int(cfg["CONCURRENT"]))
        start(ci_dir, args.slurm_partition, gpu_flags_enabled=not args.no_slurm_gpu_flags)

    elif args.command == "start":
        start(ci_dir, args.slurm_partition, gpu_flags_enabled=not args.no_slurm_gpu_flags)
    elif args.command == "stop":
        stop(ci_dir)
    elif args.command == "status":
        status(ci_dir, args.log_lines)
    elif args.command == "cleanup":
        cleanup(ci_dir, args.yes)


if __name__ == "__main__":
    main()
