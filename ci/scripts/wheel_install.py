#!/usr/bin/env python3
"""Install an nvrx_checkpointing wheel for CI and Slurm runs.

CI pipelines consume ``dist/*.whl`` from the exact parent wheel-build job.
Main publishes its tested wheels to the project PyPI registry for users.
Slurm runs without wheel artifacts fall back to an import-triggered source
build.

Slurm runs use ``pip install --target`` below the named run directory, which
keeps the package visible on every node without modifying the source checkout
or the container image.

CLI:
  wheel_install.py --from-dist DIR   install from a local artifacts dir
  wheel_install.py --fetch-upstream-wheel
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("native_ci")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_WHEEL_BINARIES = ("checkpoint.abi3.so", "checkpoint_server")

_FETCH_TIMEOUT_S = 120


def wheel_arch() -> str:
    """Target CPU arch for the wheel filename.

    WHEEL_ARCH is set per runner platform in the CI YAML (the slurm launcher
    may not share the compute nodes' arch, so probing the local machine is
    only a fallback for local runs).
    """
    return os.environ.get("WHEEL_ARCH") or platform.machine()


def _find_local_wheel(dist_dir: Path) -> Path | None:
    """Newest wheel in dist_dir whose platform tag matches this arch."""
    if not dist_dir.is_dir():
        return None
    tag = f"linux_{wheel_arch()}"
    wheels = sorted((p for p in dist_dir.glob("*.whl") if tag in p.name), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def fetch_upstream_wheel(dest_dir: Path | None = None) -> Path:
    """Download this architecture's wheel from an exact upstream CI job."""
    job_var = f"UPSTREAM_WHEEL_{wheel_arch().upper().replace('-', '_')}_JOB_ID"
    job_id = os.environ.get(job_var, "")
    api = os.environ.get("CI_API_V4_URL")
    project = os.environ.get("CI_PROJECT_ID")
    token = os.environ.get("CI_JOB_TOKEN")
    if not (job_id.isdigit() and api and project and token):
        raise RuntimeError(f"{job_var}, CI_API_V4_URL, CI_PROJECT_ID, and CI_JOB_TOKEN are required")
    root = f"{api}/projects/{project}/jobs/{job_id}/artifacts"
    headers = {"JOB-TOKEN": token}
    pointer = f"dist/nvrx_checkpointing-linux_{wheel_arch()}.txt"
    request = urllib.request.Request(f"{root}/{urllib.parse.quote(pointer, safe='/')}", headers=headers)
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:
        filename = response.read().decode().strip()
    if Path(filename).name != filename or not filename.endswith(f"-linux_{wheel_arch()}.whl"):
        raise RuntimeError(f"upstream wheel job {job_id} returned invalid filename {filename!r}")
    target_dir = dest_dir or REPO_ROOT / "dist"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    wheel_path = urllib.parse.quote(f"dist/{filename}", safe="/")
    request = urllib.request.Request(f"{root}/{wheel_path}", headers=headers)
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    logger.info(f"[wheel] downloaded {target.name} from upstream job {job_id}")
    return target


def _install_dir() -> Path | None:
    value = os.environ.get("SLURMJOB_OPT_PYTHON_INSTALL_DIR")
    return Path(value).expanduser().resolve() if value else None


def _source_revision() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()  # noqa: S607


def _package_root() -> Path:
    if target := _install_dir():
        return target / "nvrx_checkpointing"
    distribution = importlib.metadata.distribution("nvrx-checkpointing")
    return Path(distribution.locate_file("nvrx_checkpointing")).resolve()


def _remove_installed() -> None:
    if target := _install_dir():
        shutil.rmtree(target, ignore_errors=True)


def _pip_target_options(whl: Path) -> list[str]:
    python_tag, abi_tag, platform_tag = whl.stem.rsplit("-", 3)[-3:]
    version_tag = python_tag.removeprefix("cp")
    major, minor = version_tag[:1], version_tag[1:]
    if not python_tag.startswith("cp") or not major.isdigit() or not minor.isdigit():
        return ["--platform", f"linux_{wheel_arch()}"]
    return [
        "--implementation",
        "cp",
        "--python-version",
        f"{major}.{minor}",
        "--abi",
        abi_tag,
        "--platform",
        platform_tag,
    ]


def _pip_install(whl: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-deps",
        "--upgrade",
        "--force-reinstall",
    ]
    target = _install_dir()
    if target is None:
        subprocess.run([*command, str(whl)], check=True)
        return
    staging = target.with_name(f".{target.name}.installing-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                *command,
                "--only-binary=:all:",
                *_pip_target_options(whl),
                "--target",
                str(staging),
                str(whl),
            ],
            check=True,
        )
        (staging / ".nvrx-checkpointing-commit").write_text(f"{_source_revision()}\n")
        shutil.rmtree(target, ignore_errors=True)
        staging.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _binaries_load() -> bool:
    """Check the installed binaries' dynamic deps resolve on this host.

    The wheel is built on the CI builder image; a bare-metal runner with an
    older glibc/libstdc++ can't run it, in which case the caller falls back
    to the source build. Meaningful only where the binaries actually run —
    slurm launchers skip it because the workers run inside a container.
    """
    proc = subprocess.run(["ldd", str(_package_root() / "checkpoint_server")], capture_output=True, text=True, check=False)  # noqa: S607
    if proc.returncode != 0 or "not found" in proc.stdout:
        logger.warning(f"[wheel] prebuilt binaries do not load here:\n{proc.stdout}{proc.stderr}")
        return False
    return True


def install_prebuilt_wheel(dist_dir: Path | None = None, *, probe: bool = True) -> bool:
    """Resolve and install a perf wheel; False means build from source."""
    if _install_dir() is not None and _binaries_installed():
        logger.info("[wheel] requested nvrx_checkpointing wheel is already installed")
        return True
    try:
        whl = _find_local_wheel(dist_dir or REPO_ROOT / "dist")
        if whl is None:
            logger.info("[wheel] no prebuilt wheel found — building from source")
            _remove_installed()
            return False
        _pip_install(whl)
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning(f"[wheel] install failed ({e}) — building from source")
        _remove_installed()
        return False
    if probe and not _binaries_load():
        _remove_installed()
        return False
    logger.info(f"[wheel] installed {whl.name} with pip")
    return True


def _binaries_installed() -> bool:
    """True when the selected checkout's wheel is installed."""
    target = _install_dir()
    if target is None:
        return False
    marker = target / ".nvrx-checkpointing-commit"
    if not marker.is_file() or marker.read_text().strip() != _source_revision():
        return False
    root = target / "nvrx_checkpointing"
    return all((root / name).is_file() for name in _WHEEL_BINARIES)


def _ensure_source_toolchain() -> None:
    if shutil.which("clang-23") and shutil.which("clang++-23"):
        return
    prepare = REPO_ROOT / "scripts" / "prepare.sh"
    logger.warning("[wheel] clang-23 is unavailable — preparing the Ubuntu build environment")
    subprocess.run(["/bin/bash", str(prepare), "ubuntu2404"], check=True)
    if not shutil.which("clang-23") or not shutil.which("clang++-23"):
        raise RuntimeError("scripts/prepare.sh ubuntu2404 did not install clang-23")


def build_and_install_from_source() -> None:
    """Build the wheel from the checkout and install the package.

    The srun prep step runs this when a flavor explicitly requests a local
    build. Its build env is already exported, and the run step then starts from
    installed binaries exactly like a prebuilt-wheel job.
    """
    _ensure_source_toolchain()
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is not installed or not found in PATH.")
    cc, cxx = shutil.which("clang-23"), shutil.which("clang++-23")
    build_dir = Path(os.environ.get("MEGATRON_NATIVE_BUILD_DIR", REPO_ROOT / "build" / "wheel-prep")).expanduser().resolve()
    vcpkg_root = os.environ.get("VCPKG_ROOT")
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
            "-DVCPKG_BOOTSTRAP_OPTIONS=-skipDependencyChecks",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            *([f"-DVCPKG_ROOT={vcpkg_root}"] if vcpkg_root else []),
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build_dir), "-j16", "--target", "nvrx_checkpointing_whl"], check=True)
    whl = _find_local_wheel(build_dir)
    if whl is None:
        raise RuntimeError(f"wheel build produced no wheel in {build_dir}")
    _pip_install(whl)
    logger.info(f"[wheel] built and installed {whl.name} from source with pip")


def ensure_prebuilt() -> None:
    """Guarantee installed binaries for a perf-flavor run.

    The launcher normally installs the pipeline wheel artifact before
    submitting the srun. Runs without an artifact build from source.
    """
    if _binaries_installed():
        logger.info("[wheel] prebuilt binaries already installed")
        return
    if install_prebuilt_wheel():
        return
    logger.warning("[wheel] no wheel artifact for this commit — building from source")
    build_and_install_from_source()


def report_install() -> None:
    """Validate and report the installed distribution and source revision."""
    try:
        distribution = importlib.metadata.distribution("nvrx-checkpointing")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("nvrx_checkpointing is not pip-installed") from error
    package_root = _package_root()
    spec = importlib.util.find_spec("nvrx_checkpointing")
    if spec is None or spec.origin is None:
        raise RuntimeError("nvrx_checkpointing is not importable")
    if Path(spec.origin).resolve() != package_root / "__init__.py":
        raise RuntimeError(f"nvrx_checkpointing resolves to {spec.origin}, not pip installation {package_root}")
    binaries = tuple(package_root / name for name in _WHEEL_BINARIES)
    if missing := [str(path) for path in binaries if not path.is_file()]:
        raise RuntimeError(f"nvrx_checkpointing native binaries are missing: {', '.join(missing)}")
    logger.info(
        f"[wheel] nvrx_checkpointing distribution={distribution.version} commit={_source_revision()} module={spec.origin} "
        f"shlib={binaries[0]} server={binaries[1]}",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-dist", type=Path, default=None, help="Install from this local artifacts dir (no registry fetch)")
    p.add_argument(
        "--fetch-upstream-wheel",
        action="store_true",
        help="Download this architecture's wheel artifact from the upstream CI wheel job",
    )
    p.add_argument(
        "--ensure-prebuilt",
        action="store_true",
        help="Prep step, perf flavor: use installed or local wheel binaries, else build from source",
    )
    p.add_argument(
        "--build-from-source",
        action="store_true",
        help="Prep step, other flavors: build the wheel from the checkout with the ambient build env and install it",
    )
    p.add_argument("--report-install", action="store_true", help="Report selected checkout, commit, and native binary paths")
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
    if args.fetch_upstream_wheel:
        fetch_upstream_wheel()
        return
    if args.ensure_prebuilt:
        ensure_prebuilt()
        return
    if args.build_from_source:
        build_and_install_from_source()
        return
    if args.report_install:
        report_install()
        return
    if not install_prebuilt_wheel(args.from_dist):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
