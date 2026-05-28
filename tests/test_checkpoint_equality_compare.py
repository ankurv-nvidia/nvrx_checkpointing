"""Unit tests for ``scripts/checkpoint_equality_compare.py``.

Builds two synthetic checkpoint directories with ``dcp.save`` (tensors) plus
``torch.save`` (common.pt) and exercises the comparator over the spec
scenarios:

  - equal directories                         → exit 0
  - missing-on-native key in common.pt        → exit 0 + stderr warning
  - missing-on-native tensor                  → exit 1 (strict tensor policy)
  - extra-on-native key in common.pt / tensor → exit 1
  - value mismatch (tensor or common.pt)      → exit 1

Single-process, CPU-only. ``dcp.save`` falls back to local I/O when
``torch.distributed`` is uninitialized and emits one benign warning per
invocation.
"""

from __future__ import annotations

import argparse
import enum
import importlib.util
import types
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp


class _RunMode(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"


# `scripts/` is a sibling directory of `tests/`, not a Python package, so we
# load `checkpoint_equality_compare.py` directly via `importlib`. Going via
# `from megatron.native.scripts...` would trigger the CMake autobuild in
# `megatron/native/__init__.py`, which the C++ tests use but a pure-Python
# unit test should sidestep.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "checkpoint_equality_compare.py"
_spec = importlib.util.spec_from_file_location("checkpoint_equality_compare", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main


def _make_checkpoint(d: Path, tensors: dict[str, torch.Tensor], common: dict) -> None:
    """Drop a (tensors + common.pt) checkpoint into ``d``."""
    d.mkdir(parents=True, exist_ok=True)
    dcp.save(tensors, checkpoint_id=str(d))
    torch.save(common, d / "common.pt")


@pytest.fixture
def base_tensors() -> dict[str, torch.Tensor]:
    return {
        "layer.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer.1.weight": torch.arange(4, dtype=torch.float64).reshape(2, 2),
    }


@pytest.fixture
def base_common() -> dict:
    return {
        "iteration": 5,
        "checkpoint_version": 3.0,
        "is_master": True,
        "args": argparse.Namespace(
            seed=42,
            lr=1e-3,
            nested=types.SimpleNamespace(beta=0.9, names=("a", "b")),
        ),
        "history": [1, 2, 3],
        "opt_param_scheduler": {"step": 7, "min_lr": 0.0},
    }


def test_equal_directories_pass(tmp_path, base_tensors, base_common) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, base_common)
    _make_checkpoint(nvrx, base_tensors, base_common)

    assert main(native, nvrx) == 0


def test_missing_on_native_warns(tmp_path, base_tensors, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, base_common)
    _make_checkpoint(nvrx, base_tensors, {**base_common, "extra_only_in_nvrx": 99})

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc == 0
    assert "missing on native side: common.pt:extra_only_in_nvrx" in err


def test_extra_on_native_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "extra_only_in_native": 99})
    _make_checkpoint(nvrx, base_tensors, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "extra on native side: common.pt:extra_only_in_native" in err


def test_common_value_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "iteration": 999})
    _make_checkpoint(nvrx, base_tensors, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "value mismatch at common.pt:iteration" in err


def test_namespace_inner_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    """A divergence buried inside an ``argparse.Namespace`` must still fail."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    native_common = {
        **base_common,
        "args": argparse.Namespace(
            seed=42,
            lr=1e-3,
            nested=types.SimpleNamespace(beta=0.5, names=("a", "b")),  # beta differs
        ),
    }
    _make_checkpoint(native, base_tensors, native_common)
    _make_checkpoint(nvrx, base_tensors, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "common.pt:args:nested:beta" in err


def test_tensor_value_mismatch_fails(tmp_path, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, {"w": torch.zeros(2, 3, dtype=torch.float32)}, base_common)
    _make_checkpoint(nvrx, {"w": torch.ones(2, 3, dtype=torch.float32)}, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "tensor value mismatch: tensor:w" in err


def test_tensor_extra_on_native_fails(tmp_path, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(
        native,
        {"a": torch.zeros(2, 3), "b": torch.zeros(2)},
        base_common,
    )
    _make_checkpoint(nvrx, {"a": torch.zeros(2, 3)}, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "extra on native side: tensor:b" in err


def test_tensor_missing_on_native_fails(tmp_path, base_common, capsys) -> None:
    """Tensors must match in both directions — no binding-gap carve-out."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, {"a": torch.zeros(2, 3)}, base_common)
    _make_checkpoint(
        nvrx,
        {"a": torch.zeros(2, 3), "b": torch.zeros(2)},
        base_common,
    )

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "FAIL: missing on native side: tensor:b" in err


def test_common_type_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    """A dict on one side vs. a list at the same path is a structural failure."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "history": {"a": 1}})
    _make_checkpoint(nvrx, base_tensors, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "type mismatch at common.pt:history" in err


def test_common_list_length_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "history": [1, 2]})
    _make_checkpoint(nvrx, base_tensors, base_common)  # history = [1, 2, 3]

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "length mismatch at common.pt:history" in err


def test_namespace_vs_dict_type_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    """nvrx may carry an `argparse.Namespace` where native has a plain dict."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "args": {"seed": 42}})
    _make_checkpoint(nvrx, base_tensors, base_common)

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "type mismatch at common.pt:args" in err


def test_complex_leaf_equal_passes(tmp_path, base_tensors, base_common) -> None:
    """Matching enum leaves take the repr-equality fallback and pass silently."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    common = {**base_common, "mode": _RunMode.TRAIN}
    _make_checkpoint(native, base_tensors, common)
    _make_checkpoint(nvrx, base_tensors, common)

    assert main(native, nvrx) == 0


def test_complex_leaf_mismatch_fails(tmp_path, base_tensors, base_common, capsys) -> None:
    """Diverging enum leaves fall through to the repr-equality fallback and fail."""
    native = tmp_path / "native"
    nvrx = tmp_path / "nvrx"
    _make_checkpoint(native, base_tensors, {**base_common, "mode": _RunMode.EVAL})
    _make_checkpoint(nvrx, base_tensors, {**base_common, "mode": _RunMode.TRAIN})

    rc = main(native, nvrx)
    err = capsys.readouterr().err

    assert rc != 0
    assert "complex-type value mismatch at common.pt:mode" in err
