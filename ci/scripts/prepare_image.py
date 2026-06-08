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
    _detect_local_gpu_count,
    _ensure_vcpkg_root,
    _image_path,
    _render_template,
    _runner_gpus_per_node,
    _runner_uses_slurm,
)


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
    """Render image_build.tmpl (per-cluster override in CI_DIR, else repo default) and run it.

    The default imports the base PyTorch image if missing, runs prepare.sh inside it, then
    saves the result to image_out — mirroring scripts/enroot.sh. The build targets a per-job
    path (see _ci_build_path) instead of a shared scratch file, so concurrent jobs on one
    machine never clobber each other's in-progress image; the finished image is then moved to
    the shared path only if still missing (_publish_to_shared). enroot saves the container at
    srun teardown even when prepare.sh fails inside it, so the build writes to the per-job
    path and is promoted to the shared path only after a clean exit — a failed build never
    leaves a partial sqsh that later jobs would reuse as if prebuilt.
    """
    base_image = image.parent / "pytorch2604.sqsh"
    local = _ci_build_path(image)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.unlink(missing_ok=True)  # clear any leftover from a previously failed build
    sys.stderr.write(f"[prepare-image] image not found: {image} — building into per-job path {local} (this may take ~30 min)\n")
    script = _render_template(
        "image_build.tmpl",
        {
            "home": Path.home().as_posix(),
            "image_base": base_image.as_posix(),
            "image_out": local.as_posix(),
            "docker_image": DOCKER_IMAGE_BASE,
            "prepare_sh": (ENROOT_SCRIPTS_DIR / "prepare.sh").as_posix(),
            "gpus_per_node": gpus_per_node,
        },
    )
    bash = shutil.which("bash") or "bash"
    subprocess.run([bash, "-c", script], check=True)  # raises on failure, before the publish below
    if _publish_to_shared(local, image):
        sys.stderr.write(f"[prepare-image] published built image to shared path: {image}\n")
    else:
        sys.stderr.write(f"[prepare-image] shared image already present (a concurrent job published it); reusing {image}\n")


def main() -> None:
    _ensure_vcpkg_root()
    if not _runner_uses_slurm():
        sys.stderr.write("[prepare-image] non-SLURM runner — no enroot image needed; nothing to do\n")
        return
    image = _image_path()
    if image.exists():
        sys.stderr.write(f"[prepare-image] image already present, nothing to build: {image}\n")
        return
    _ensure_enroot_image(image, _runner_gpus_per_node() or _detect_local_gpu_count())


if __name__ == "__main__":
    main()
