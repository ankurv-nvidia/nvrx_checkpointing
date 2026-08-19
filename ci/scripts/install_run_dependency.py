#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Install a selected NVIDIA Resiliency Extension checkout into an isolated run target."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _git() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git is not installed or not found in PATH")
    return executable


def _source_requirement(repo_url: str, commit: str | None, repo_root: Path | None) -> str:
    if repo_root is None:
        vcs_url = repo_url if repo_url.startswith("git+") else f"git+{repo_url}"
        return f"{vcs_url}@{commit}"

    root = repo_root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    clone_url = repo_url.removeprefix("git+")
    with (root.parent / f".{root.name}.clone.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cloned = False
        if not root.exists():
            staging = root.with_name(f".{root.name}.cloning-{os.getpid()}")
            shutil.rmtree(staging, ignore_errors=True)
            try:
                subprocess.run([_git(), "clone", "--quiet", clone_url, str(staging)], check=True)
                staging.replace(root)
                cloned = True
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if not (root / ".git").exists():
            raise SystemExit(f"--repo-root is not a Git checkout: {root}")
        if commit:
            found = (
                subprocess.run(
                    [_git(), "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                    check=False,
                ).returncode
                == 0
            )
            if not found:
                subprocess.run([_git(), "-C", str(root), "fetch", "--quiet", clone_url, commit], check=True)
            if cloned:
                subprocess.run([_git(), "-C", str(root), "checkout", "--quiet", "--detach", commit], check=True)
    return f"git+{root.as_uri()}@{commit}" if commit else str(root)


def _head_root(repo_root: Path | None) -> Path:
    if repo_root is None:
        raise SystemExit("--commit HEAD requires --repo-root")
    root = repo_root.expanduser().resolve()
    valid = root.is_dir() and subprocess.run(
        [_git(), "-C", str(root), "rev-parse", "--git-dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not valid:
        raise SystemExit(f"--commit HEAD requires an existing Git checkout at --repo-root: {root}")
    return root


@contextlib.contextmanager
def _snapshot(root: Path | None, target: Path) -> Generator[Path | None, None, None]:
    """Yields a disposable source tree for a HEAD working checkout."""
    if root is None:
        yield None
        return
    snapshot = target.with_name(f".{target.name}.source-{os.getpid()}")
    shutil.rmtree(snapshot, ignore_errors=True)
    try:
        shutil.copytree(root, snapshot, ignore=shutil.ignore_patterns(".git"))
        yield snapshot
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


def install_nvrx(repo_url: str, commit: str | None, target: Path, repo_root: Path | None = None) -> None:
    """Install without dependencies, then verify this target wins over the image package."""
    worktree = commit == "HEAD"
    if commit and not worktree and not _COMMIT_RE.fullmatch(commit):
        raise SystemExit("--commit must be HEAD or a 7-40 character hexadecimal commit")
    if not commit and repo_root is None:
        raise SystemExit("--commit or --repo-root is required")
    root = _head_root(repo_root) if worktree else None

    target = target.expanduser().resolve()
    marker = target / ".nvrx-commit"
    source_commit = None if worktree else commit
    if source_commit and marker.is_file() and marker.read_text().strip() == source_commit:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.installing-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        with _snapshot(root, target) as snapshot:
            requirement = str(snapshot) if snapshot else _source_requirement(repo_url, source_commit, repo_root)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--no-deps",
                    "--upgrade",
                    "--force-reinstall",
                    "--target",
                    str(staging),
                    requirement,
                ],
                check=True,
            )
        if source_commit:
            (staging / marker.name).write_text(f"{source_commit}\n")
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{target}{os.pathsep}{prior}" if prior else str(target)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nvidia_resiliency_ext as p; print(p.__file__)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    imported = Path(proc.stdout.strip()).resolve()
    if not imported.is_relative_to(target):
        raise RuntimeError(f"NVIDIA Resiliency Extension resolved to {imported}, not the requested target {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    install_nvrx(args.repo_url, args.commit, args.target, args.repo_root)


if __name__ == "__main__":
    main()
