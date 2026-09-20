from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load_otel_report():
    spec = importlib.util.spec_from_file_location("otel_report", _REPO / "ci" / "scripts" / "otel_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_ckptbench() -> str | None:
    env = os.environ.get("NATIVE_CKPT_CKPTBENCH_BIN")
    if env and os.path.exists(env):
        return env
    return shutil.which("ckptbench")


def _find_registry() -> str | None:
    for path in sorted((_REPO / "metadata").glob("*/registry.bin")):
        return str(path)
    return None


def test_daemon_mode_collects_single_file(tmp_path):
    try:
        import pytest
    except ImportError:
        return

    ckptbench = _find_ckptbench()
    registry = _find_registry()
    if not ckptbench:
        pytest.skip("ckptbench binary not found (set NATIVE_CKPT_CKPTBENCH_BIN)")
    if not registry:
        pytest.skip("no state registry fixture found under metadata/")

    out = tmp_path / "collected.jsonl"
    env = {
        **os.environ,
        "NATIVE_CKPT_OTEL_EXPORT": "daemon",
        "NATIVE_CKPT_OTEL_FILE": str(out),
    }
    result = subprocess.run(
        [ckptbench, "--stateregistry", registry, "--nodes", "2"],
        env=env,
        timeout=300,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ckptbench failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert out.exists(), "ckptbench ran successfully but produced no collected OTLP file"

    otel_report = _load_otel_report()
    samples = otel_report.collect_samples([str(out)])
    ranks = {int(s["rank"]) for s in samples}
    assert samples, "no ckpt.save spans in the collected file"
    assert len(ranks) >= 2, f"expected >=2 ranks in the single collected file, got {ranks}"
