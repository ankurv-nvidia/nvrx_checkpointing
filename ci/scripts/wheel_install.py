#!/usr/bin/env python3
"""Install prebuilt native binaries from a CI perf wheel into the checkout.

The wheel jobs publish the perf-flavor wheel two ways: as job artifacts
(``dist/*.whl``, consumed by same-pipeline ``needs:``) and to the project's
generic package registry keyed by commit SHA. The registry copy exists for
the run_job matrix: those jobs run in child pipelines that cannot ``needs:``
the parent's wheel jobs without serializing behind them, so they probe the
registry for a wheel matching the current commit + CPU arch and fall back
to the import-triggered source build when none is published yet.

Installation extracts ``checkpoint.abi3.so`` and ``checkpoint_server`` from
the wheel to the repo root as regular files; the root ``__init__.py`` treats
artifacts that resolve outside the auto-build dir as externally provided and
skips its CMake build.

CLI:
  wheel_install.py --from-dist DIR   install from a local artifacts dir
  wheel_install.py --registry-url    print the registry URL for this commit+arch
"""

from __future__ import annotations

import argparse
import http.client
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("native_ci")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files the perf wheel carries that the checkout needs; everything else in
# the wheel (the python scripts) already lives in the checkout.
_WHEEL_BINARIES = ("checkpoint.abi3.so", "checkpoint_server")

# Generic package registry coordinates. The upload side (the wheel job's
# curl) uses the same URL via --registry-url so the two sides can't drift.
_REGISTRY_PACKAGE = "ci-wheels"
_FETCH_TIMEOUT_S = 120


def wheel_arch() -> str:
    """Target CPU arch for the wheel filename.

    WHEEL_ARCH is set per runner platform in the CI YAML (the slurm launcher
    may not share the compute nodes' arch, so probing the local machine is
    only a fallback for local runs).
    """
    return os.environ.get("WHEEL_ARCH") or platform.machine()


def registry_wheel_url() -> str | None:
    """Registry URL of this commit's perf wheel, or None outside CI."""
    api = os.environ.get("CI_API_V4_URL")
    project = os.environ.get("CI_PROJECT_ID")
    sha = os.environ.get("CI_COMMIT_SHA")
    if not (api and project and sha):
        return None
    return f"{api}/projects/{project}/packages/generic/{_REGISTRY_PACKAGE}/{sha}/nvrx_checkpointing-linux_{wheel_arch()}.whl"


def _find_local_wheel(dist_dir: Path) -> Path | None:
    """Newest wheel in dist_dir whose platform tag matches this arch."""
    if not dist_dir.is_dir():
        return None
    tag = f"linux_{wheel_arch()}"
    wheels = sorted((p for p in dist_dir.glob("*.whl") if tag in p.name), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def _fetch_registry_wheel(dest_dir: Path) -> Path | None:
    """Download this commit's wheel from the package registry, if published."""
    url = registry_wheel_url()
    token = os.environ.get("CI_JOB_TOKEN")
    if not (url and token):
        return None
    request = urllib.request.Request(url, headers={"JOB-TOKEN": token})
    dest = dest_dir / url.rsplit("/", 1)[-1]
    try:
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as resp, dest.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        level = logging.INFO if e.code == http.HTTPStatus.NOT_FOUND else logging.WARNING
        logger.log(level, f"[wheel] no registry wheel at {url} (HTTP {e.code})")
        return None
    except (OSError, http.client.HTTPException) as e:
        logger.warning(f"[wheel] registry fetch failed: {e}")
        return None
    return dest


def _remove_extracted() -> None:
    """Remove wheel-extracted binaries; dev symlinks into a build tree stay."""
    for name in _WHEEL_BINARIES:
        path = REPO_ROOT / name
        if not path.is_symlink():
            path.unlink(missing_ok=True)


def _extract_binaries(whl: Path) -> None:
    """Extract the native binaries from the wheel to the repo root."""
    with zipfile.ZipFile(whl) as zf:
        for name in _WHEEL_BINARIES:
            member = f"nvrx_checkpointing/{name}"
            target = REPO_ROOT / name
            # Unlink first: a leftover dev symlink would otherwise route the
            # write into its build-tree target.
            target.unlink(missing_ok=True)
            with zf.open(member) as src:
                target.write_bytes(src.read())
            target.chmod(0o755)


def _binaries_load() -> bool:
    """Check the extracted binaries' dynamic deps resolve on this host.

    The wheel is built on the CI builder image; a bare-metal runner with an
    older glibc/libstdc++ can't run it, in which case the caller falls back
    to the source build. Meaningful only where the binaries actually run —
    slurm launchers skip it because the workers run inside a container.
    """
    proc = subprocess.run(["ldd", str(REPO_ROOT / "checkpoint_server")], capture_output=True, text=True, check=False)  # noqa: S607
    if proc.returncode != 0 or "not found" in proc.stdout:
        logger.warning(f"[wheel] prebuilt binaries do not load here:\n{proc.stdout}{proc.stderr}")
        return False
    return True


def install_prebuilt_wheel(dist_dir: Path | None = None, *, fetch_remote: bool = True, probe: bool = True) -> bool:
    """Resolve and install a perf wheel; False means build from source."""
    try:
        with tempfile.TemporaryDirectory(prefix="ci-wheel-") as tmp:
            whl = _find_local_wheel(dist_dir or REPO_ROOT / "dist")
            if whl is None and fetch_remote:
                whl = _fetch_registry_wheel(Path(tmp))
            if whl is None:
                logger.info("[wheel] no prebuilt wheel found — building from source")
                # Drop regular-file binaries a previous install may have left
                # so the auto-build guard can't latch onto stale ones.
                _remove_extracted()
                return False
            _extract_binaries(whl)
    except (OSError, KeyError, zipfile.BadZipFile) as e:
        logger.warning(f"[wheel] install failed ({e}) — building from source")
        _remove_extracted()
        return False
    if probe and not _binaries_load():
        _remove_extracted()
        return False
    logger.info(f"[wheel] installed prebuilt binaries from {whl.name}")
    return True


def _binaries_installed() -> bool:
    """True when both binaries sit at the repo root as regular files."""
    return all((REPO_ROOT / name).is_file() and not (REPO_ROOT / name).is_symlink() for name in _WHEEL_BINARIES)


def build_and_install_from_source() -> None:
    """Build the wheel from the checkout and install its binaries.

    The srun prep step runs this for flavors whose build config the CI wheel
    cannot match (deterministic runs the plain build, valgrind needs
    PHOENIX_BUILD_FOR_VALGRIND): the flavor's build env is already exported
    (CMake's DetectVar reads it), the build runs on the machine that executes
    the workload (so -march=native stays), and the run step then starts from
    installed binaries exactly like a perf-flavor job.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is not installed or not found in PATH.")
    build_dir = REPO_ROOT / "build" / "wheel-prep"
    subprocess.run(
        [
            cmake,
            "-B",
            str(build_dir),
            "-S",
            str(REPO_ROOT),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DPHOENIX_BUILD_WHEEL=ON",
            "-DPHOENIX_BUILD_BENCHMARK=OFF",
            "-DPHOENIX_BUILD_TESTS=OFF",
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build_dir), "-j16", "--target", "nvrx_checkpointing_whl"], check=True)
    whl = _find_local_wheel(build_dir)
    if whl is None:
        raise RuntimeError(f"wheel build produced no wheel in {build_dir}")
    _extract_binaries(whl)
    logger.info(f"[wheel] built and installed {whl.name} from source")


def ensure_prebuilt() -> None:
    """Guarantee installed binaries for a perf-flavor run.

    The launcher normally installs the published wheel before submitting the
    srun, so this is a fast no-op in the prep step; the fallbacks cover runs
    outside a full pipeline (manual branches, wheel jobs skipped).
    """
    if _binaries_installed():
        logger.info("[wheel] prebuilt binaries already installed")
        return
    if install_prebuilt_wheel():
        return
    logger.warning("[wheel] no published wheel for this commit — building from source")
    build_and_install_from_source()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-dist", type=Path, default=None, help="Install from this local artifacts dir (no registry fetch)")
    p.add_argument("--registry-url", action="store_true", help="Print the registry wheel URL for this commit+arch and exit")
    p.add_argument(
        "--ensure-prebuilt",
        action="store_true",
        help="Prep step, perf flavor: no-op when binaries are installed; else fetch the published wheel, else build from source",
    )
    p.add_argument(
        "--build-from-source",
        action="store_true",
        help="Prep step, other flavors: build the wheel from the checkout with the ambient build env and install it",
    )
    p.add_argument(
        "--define",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Export KEY=VALUE before acting — carries the flavor build env into the source build on argv",
    )
    args = p.parse_args()
    for define in args.define:
        key, _, value = define.partition("=")
        os.environ[key] = value
    if args.registry_url:
        url = registry_wheel_url()
        if url is None:
            raise SystemExit("CI_API_V4_URL / CI_PROJECT_ID / CI_COMMIT_SHA not set")
        sys.stdout.write(f"{url}\n")
        return
    if args.ensure_prebuilt:
        ensure_prebuilt()
        return
    if args.build_from_source:
        build_and_install_from_source()
        return
    if not install_prebuilt_wheel(args.from_dist, fetch_remote=args.from_dist is None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
