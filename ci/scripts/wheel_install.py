#!/usr/bin/env python3
"""Install an nvrx_checkpointing wheel for CI and Slurm runs.

The wheel jobs publish the perf-flavor wheel two ways: as job artifacts
(``dist/*.whl``, consumed by same-pipeline ``needs:``) and to the project's
generic package registry keyed by commit SHA. Native perf jobs wait for the
publish jobs, then install the registry wheel matching their commit and CPU
architecture. Other Slurm jobs can probe the same registry and fall back to
an import-triggered source build when no matching wheel exists.

Slurm runs use ``pip install --target`` below the named run directory, which
keeps the package visible on every node without modifying the source checkout
or the container image.

CLI:
  wheel_install.py --from-dist DIR   install from a local artifacts dir
  wheel_install.py --registry-url    print the registry URL for this commit+arch
"""

from __future__ import annotations

import argparse
import http.client
import importlib.metadata
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("native_ci")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_WHEEL_BINARIES = ("checkpoint.abi3.so", "checkpoint_server")

# Generic package registry coordinates. The upload side (the wheel job's
# curl) uses the same URL via --registry-url so the two sides can't drift.
_REGISTRY_PACKAGE = "ci-wheels"
_FETCH_TIMEOUT_S = 120


def _registry_file_url(filename: str) -> str | None:
    api = os.environ.get("CI_API_V4_URL")
    project = os.environ.get("CI_PROJECT_ID")
    sha = os.environ.get("CI_COMMIT_SHA")
    if not (api and project and sha):
        return None
    filename = urllib.parse.quote(filename, safe="")
    return f"{api}/projects/{project}/packages/generic/{_REGISTRY_PACKAGE}/{sha}/{filename}"


def wheel_arch() -> str:
    """Target CPU arch for the wheel filename.

    WHEEL_ARCH is set per runner platform in the CI YAML (the slurm launcher
    may not share the compute nodes' arch, so probing the local machine is
    only a fallback for local runs).
    """
    return os.environ.get("WHEEL_ARCH") or platform.machine()


def registry_wheel_url(filename: str | None = None) -> str | None:
    """Registry URL for a wheel, or its canonical-filename pointer."""
    return _registry_file_url(filename or f"nvrx_checkpointing-linux_{wheel_arch()}.txt")


def _find_local_wheel(dist_dir: Path) -> Path | None:
    """Newest wheel in dist_dir whose platform tag matches this arch."""
    if not dist_dir.is_dir():
        return None
    tag = f"linux_{wheel_arch()}"
    wheels = sorted((p for p in dist_dir.glob("*.whl") if tag in p.name), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def _fetch_registry_wheel(dest_dir: Path) -> Path | None:
    """Download this commit's wheel from the package registry, if published."""
    pointer_url = registry_wheel_url()
    token = os.environ.get("CI_JOB_TOKEN")
    if not (pointer_url and token):
        return None
    headers = {"JOB-TOKEN": token}
    result = None
    try:
        request = urllib.request.Request(pointer_url, headers=headers)
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as resp:
            filename = resp.read().decode().strip()
        if Path(filename).name != filename or not filename.endswith(f"-linux_{wheel_arch()}.whl"):
            logger.warning(f"[wheel] invalid registry wheel filename: {filename!r}")
        elif url := registry_wheel_url(filename):
            dest = dest_dir / filename
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as resp, dest.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            result = dest
    except urllib.error.HTTPError as e:
        level = logging.INFO if e.code == http.HTTPStatus.NOT_FOUND else logging.WARNING
        logger.log(level, f"[wheel] no registry wheel at {pointer_url} (HTTP {e.code})")
    except (OSError, ValueError, UnicodeDecodeError, http.client.HTTPException) as e:
        logger.warning(f"[wheel] registry fetch failed: {e}")
    return result


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


def install_prebuilt_wheel(dist_dir: Path | None = None, *, fetch_remote: bool = True, probe: bool = True) -> bool:
    """Resolve and install a perf wheel; False means build from source."""
    if _install_dir() is not None and _binaries_installed():
        logger.info("[wheel] requested nvrx_checkpointing wheel is already installed")
        return True
    try:
        with tempfile.TemporaryDirectory(prefix="ci-wheel-") as tmp:
            whl = _find_local_wheel(dist_dir or REPO_ROOT / "dist")
            if whl is None and fetch_remote:
                whl = _fetch_registry_wheel(Path(tmp))
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
    if shutil.which("clang-22") and shutil.which("clang++-22"):
        return
    prepare = REPO_ROOT / "scripts" / "prepare.sh"
    logger.warning("[wheel] clang-22 is unavailable — preparing the Ubuntu build environment")
    subprocess.run(["/bin/bash", str(prepare), "ubuntu2404"], check=True)
    if not shutil.which("clang-22") or not shutil.which("clang++-22"):
        raise RuntimeError("scripts/prepare.sh ubuntu2404 did not install clang-22")


def build_and_install_from_source() -> None:
    """Build the wheel from the checkout and install the package.

    The srun prep step runs this for flavors whose build config the CI wheel
    cannot match (deterministic runs the plain build, valgrind needs
    PHOENIX_BUILD_FOR_VALGRIND): the flavor's build env is already exported
    (CMake's DetectVar reads it), the build runs on the machine that executes
    the workload (so -march=native stays), and the run step then starts from
    installed binaries exactly like a perf-flavor job.
    """
    _ensure_source_toolchain()
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is not installed or not found in PATH.")
    cc, cxx = shutil.which("clang-22"), shutil.which("clang++-22")
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
        "--registry-url",
        nargs="?",
        const="",
        metavar="WHEEL",
        help="Print the registry pointer URL, or the URL for canonical WHEEL, and exit",
    )
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
    if args.registry_url is not None:
        url = registry_wheel_url(args.registry_url or None)
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
    if args.report_install:
        report_install()
        return
    if not install_prebuilt_wheel(args.from_dist, fetch_remote=args.from_dist is None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
