import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SHLIB_PATH = SCRIPT_DIR / "checkpoint.abi3.so"
SERVER_PATH = SCRIPT_DIR / "checkpoint_server"
CURRENT_HASH = None


def try_get_rank() -> int | None:
    rank: int | None = None
    rank = int(os.environ.get("RANK")) if os.environ.get("RANK") else None
    try:
        import torch  # noqa: F401, PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        if dist.is_initialized():
            rank = dist.get_rank()
    except ImportError:
        pass
    return rank


def get_native_build_dir() -> Path:
    if os.environ.get("MEGATRON_NATIVE_BUILD_DIR"):
        return Path(os.environ["MEGATRON_NATIVE_BUILD_DIR"]).resolve()
    return SCRIPT_DIR / "build"


def _is_git_repo() -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],  # noqa: S607
            capture_output=True,
            check=False,
            cwd=SCRIPT_DIR,
        ).returncode
        == 0
    )


def _compute_source_hash() -> str:
    ls = subprocess.run(
        ["git", "ls-files"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=SCRIPT_DIR,
    )
    files = sorted(ls.stdout.splitlines())
    if not files:
        raise RuntimeError(
            f"No files found in git repository to hash for {SCRIPT_DIR}.",
        )
    h = hashlib.sha256()
    for f in files:
        h.update((SCRIPT_DIR / f).read_bytes())
    return h.hexdigest()


def build_native_extensions() -> None:
    if not hasattr(build_native_extensions, "CURRENT_HASH"):
        build_native_extensions.CURRENT_HASH: str | None = None  # type: ignore[attr-defined]
    if build_native_extensions.CURRENT_HASH is not None:
        return

    native_build_dir = get_native_build_dir()
    if SHLIB_PATH.is_file():
        # resolve() maps a dev symlink to its build-tree target and a prebuilt
        # regular file (e.g. extracted from a wheel artifact) to itself, so
        # both forms of an externally provided extension skip the auto-build.
        shlib_realpath = SHLIB_PATH.resolve()
        if not shlib_realpath.is_relative_to(native_build_dir.resolve()):
            sys.stderr.write(
                f"[nvrx_checkpointing] Native extension path {shlib_realpath} "
                f"is not within auto-build dir {native_build_dir}, skipping auto-build.\n",
            )
            return

    rank = try_get_rank()
    hash_file = native_build_dir / "source.hash"
    build_native_extensions.CURRENT_HASH = _compute_source_hash()
    # A matching hash only counts with the artifacts present — the repo-root
    # files can disappear independently of the build tree (e.g. a cleanup of
    # extracted wheel binaries), and rebuilding is what restores them.
    if (
        hash_file.is_file()
        and hash_file.read_text().strip() == build_native_extensions.CURRENT_HASH
        and SHLIB_PATH.is_file()
        and SERVER_PATH.is_file()
    ):
        return

    if rank is not None and rank != 0:
        sys.stderr.write(f"[nvrx_checkpointing] src_hash={CURRENT_HASH} rank={rank}: waiting for rank 0 to complete native build...\n")
        while True:
            if hash_file.is_file() and hash_file.read_text().strip() == build_native_extensions.CURRENT_HASH:
                break
            time.sleep(1)
        sys.stderr.write(f"[nvrx_checkpointing] src_hash={CURRENT_HASH} rank={rank}: native build complete.\n")
        return

    cmake_exe = shutil.which("cmake")
    if cmake_exe is None:
        raise RuntimeError("CMake is not installed or not found in PATH.")
    native_build_dir.mkdir(parents=True, exist_ok=True)
    (native_build_dir / "CMakeCache.txt").unlink(missing_ok=True)
    configure_cmd = [
        cmake_exe,
        "-B",
        native_build_dir.as_posix(),
        "-S",
        SCRIPT_DIR.as_posix(),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "--debug-trycompile",
    ]
    sys.stderr.write(f"configure_cmd: {' '.join(configure_cmd)}\n")
    subprocess.run(
        configure_cmd,
        check=True,
    )
    subprocess.run([cmake_exe, "--build", str(native_build_dir), "-j16"], check=True)
    sys.stderr.write(f"[nvrx_checkpointing] src_hash={CURRENT_HASH} rank={rank}: native build complete.\n")
    # Write hash file to signal to waiting ranks that the build is complete.
    hash_file.write_text(build_native_extensions.CURRENT_HASH)


if _is_git_repo():
    build_native_extensions()

if not SHLIB_PATH.is_file():
    raise RuntimeError("[nvrx_checkpointing] extensions were not built successfully.")
if not SERVER_PATH.is_file():
    raise RuntimeError(
        "[nvrx_checkpointing] checkpoint_server was not built successfully.",
    )
