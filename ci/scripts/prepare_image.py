#!/usr/bin/env python3
"""Build the enroot .sqsh image for native CI when it is missing.

Its own CI stage (see gitlab-ci-native.yml): SLURM jobs ``needs:`` it instead of
building or stalling on the image inline. Fast no-op when the image already
exists, and a no-op on inline/non-SLURM runners (which use no image). Shares the
template/image-path helpers with run_job.py.
"""

from __future__ import annotations

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


def _ensure_enroot_image(image: Path, gpus_per_node: int) -> None:
    """Render image_build.tmpl (per-cluster override in CI_DIR, else repo default) and run it.

    The default imports the base PyTorch image if missing, runs prepare.sh inside it, then
    saves the result — mirroring scripts/enroot.sh.
    """
    base_image = image.parent / "pytorch2604.sqsh"
    sys.stderr.write(f"[prepare-image] enroot image not found: {image} — building now (this may take ~30 min)\n")
    script = _render_template(
        "image_build.tmpl",
        {
            "home": Path.home().as_posix(),
            "image_base": base_image.as_posix(),
            "image_out": image.as_posix(),
            "docker_image": DOCKER_IMAGE_BASE,
            "prepare_sh": (ENROOT_SCRIPTS_DIR / "prepare.sh").as_posix(),
            "gpus_per_node": gpus_per_node,
        },
    )
    bash = shutil.which("bash") or "bash"
    subprocess.run([bash, "-c", script], check=True)


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
    sys.stderr.write(f"[prepare-image] built {image}\n")


if __name__ == "__main__":
    main()
