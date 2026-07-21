#!/usr/bin/env python3
"""Build the enroot .sqsh image for native CI when it is missing.

Runs in every workload job's before_script (see gitlab-ci-native.yml). Fast no-op
when the shared image already exists, and a no-op on inline/non-SLURM runners
(which use no image). Shares the template/image-path helpers with run_job.py.

Concurrency model: each job builds the image into its own per-job path (under
$CI_PROJECT_DIR — every GitLab job/runner slot gets a distinct build directory),
then moves it to the shared image path only if that path is still absent. So
concurrent jobs on one cold machine never write the same scratch file (no
trampling), and the first to finish populates the shared path that the rest reuse.
Duplicate builds are possible on a cold machine but tolerated — the steady state is
the shared image already present, so nearly every job is the fast no-op above.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_job import (  # companion script, shares run_job's build helpers
    DOCKER_IMAGE_BASE,
    ENROOT_SCRIPTS_DIR,
    REPO_ROOT,
    build_cmd_from_template,
    detect_local_gpu_count,
    ensure_vcpkg_root,
    image_path,
    runner_gpus_per_node,
    runner_uses_slurm,
)


def _remove_native_smoke_build(build_dir: Path) -> None:
    """Remove only native artifacts produced by the image-preparation smoke test."""
    build_root = build_dir.resolve()
    artifacts = [REPO_ROOT / "checkpoint_server", *REPO_ROOT.glob("checkpoint*.so")]
    for artifact in artifacts:
        if artifact.is_symlink() and artifact.resolve(strict=False).is_relative_to(build_root):
            artifact.unlink()
    if build_dir.exists():
        shutil.rmtree(build_dir)


def _ci_build_path(image: Path) -> Path:
    """Per-job path the image is built into before being published to the shared path.

    Under $CI_PROJECT_DIR — every GitLab job (concurrent runner slot) has its own build
    directory, so concurrent builders write distinct files and never trample each other's
    in-progress image. Off-CI (no $CI_PROJECT_DIR) it falls back to a pid-suffixed name in
    the image directory, which is still per-process-unique.
    """
    ci = os.environ.get("CI_PROJECT_DIR", "").strip()
    if ci:
        return Path(ci) / f"{image.name}.building"
    return image.with_name(f"{image.name}.{os.getpid()}.building")


def _publish_to_shared(local: Path, image: Path) -> bool:
    """Move a freshly-built image from the per-job path to the shared path, only if absent.

    Returns True if this job published the shared image, False if a concurrent job already
    did (the local build is then discarded — duplicate work, but the shared image is reused).
    The move stages through a unique name on the shared image's filesystem so the final swap
    is an atomic rename even when the build path is on a different filesystem.
    """
    if image.exists():
        local.unlink(missing_ok=True)
        return False
    staged = image.with_name(f"{image.name}.{os.getpid()}.publish")
    shutil.move(str(local), str(staged))  # same FS: rename; cross FS: copy+delete onto the shared FS
    if image.exists():  # a peer published while we were moving
        staged.unlink(missing_ok=True)
        return False
    os.replace(staged, image)  # atomic on the shared FS (benign last-writer-wins if a peer races here)
    return True


def _ensure_enroot_image(image: Path, gpus_per_node: int) -> None:
    """Run enroot_import.tmpl then image_prep.tmpl to build the project enroot image.

    The base PyTorch image is imported (via enroot_import.tmpl) only when absent.
    The project image is built (via image_prep.tmpl) into a per-job path, then
    promoted to the shared path only after a clean exit — a failed build never
    leaves a partial sqsh for later jobs to reuse.
    """
    base_image = image.parent / "pytorch2604.sqsh"
    local = _ci_build_path(image)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.unlink(missing_ok=True)  # clear any leftover from a previously failed build
    sys.stderr.write(f"[prepare-image] image not found: {image} — building into per-job path {local} (this may take ~30 min)\n")

    account = os.environ.get("SLURM_ACCOUNT", "").strip()
    if not account:
        raise RuntimeError("SLURM_ACCOUNT is not set — required for SLURM image build")
    container_root = local.parent
    srun_values: dict[str, object] = {
        "account": account,
        "home": Path.home().as_posix(),
        "root": container_root.as_posix(),
        "image_base": base_image.as_posix(),
        "image_out": local.as_posix(),
        "gpus_per_node": gpus_per_node,
    }

    if not base_image.exists():
        local_base_image = local.parent / base_image.name
        local_base_image.unlink(missing_ok=True)  # clear any leftover from a previously failed build
        sys.stderr.write(f"[prepare-image] base image not found: {base_image} — importing from registry\n")
        import_cmd = build_cmd_from_template(
            "enroot_import.tmpl",
            {**srun_values, "time_minutes": 20},
            ["enroot", "import", "--output", local_base_image.as_posix(), f"docker://{DOCKER_IMAGE_BASE}"],
        )
        sys.stderr.write(f"[prepare-image] running import command: {' '.join(import_cmd)}\n")
        subprocess.run(import_cmd, check=True)
        shutil.move(str(local_base_image), str(base_image))  # move to final location after successful import

    prepare_sh = (ENROOT_SCRIPTS_DIR / "prepare.sh").as_posix()
    prep_cmd = build_cmd_from_template(
        "image_prep.tmpl",
        # Building DeepEP's HybridEP CUDA kernels for three arches (sm90/sm100/sm120)
        # roughly doubles the image-build time over the toolchain+megatron+flash-attn
        # baseline, which overran the previous 60-minute window.
        {**srun_values, "time_minutes": 120},
        ["bash", "-c", f"bash {prepare_sh} ubuntu2404 megatron test"],
    )
    # scripts/run.sh intentionally enables the low-memory Valgrind build for its
    # image smoke test. Keep that build out of the workload's default build/
    # directory; otherwise a cold-image job silently reuses its reduced SHM
    # constants while cached-image jobs build the normal configuration.
    native_smoke_build = container_root / ".image-prep-native-build"
    _remove_native_smoke_build(native_smoke_build)
    prep_env = os.environ.copy()
    prep_env["MEGATRON_NATIVE_BUILD_DIR"] = native_smoke_build.as_posix()
    sys.stderr.write(f"[prepare-image] running prepare command: {' '.join(prep_cmd)}\n")
    try:
        subprocess.run(prep_cmd, check=True, env=prep_env)  # raises on failure, before the publish below
    finally:
        _remove_native_smoke_build(native_smoke_build)

    if _publish_to_shared(local, image):
        sys.stderr.write(f"[prepare-image] published built image to shared path: {image}\n")
    else:
        sys.stderr.write(f"[prepare-image] shared image already present (a concurrent job published it); reusing {image}\n")


def main() -> None:
    ensure_vcpkg_root()
    if not runner_uses_slurm():
        sys.stderr.write("[prepare-image] non-SLURM runner — no enroot image needed; nothing to do\n")
        return
    image = image_path()
    if image.exists():
        sys.stderr.write(f"[prepare-image] image already present, nothing to build: {image}\n")
        return
    _ensure_enroot_image(image, runner_gpus_per_node() or detect_local_gpu_count())


if __name__ == "__main__":
    main()
