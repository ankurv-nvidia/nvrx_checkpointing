"""Tier 3 — DCP load round-trip via the parity shim.

Drives the C++ writer end-to-end through `_distcp_shim` (one `write_rank_shard`
per rank plus a single `write_metadata`), then loads the resulting checkpoint
directory back with `torch.distributed.checkpoint.load` and asserts every
tensor matches the original. This is the gold-standard format-compat test —
if it passes, our `.distcp` + `.metadata` output is a real DCP checkpoint
under the loader's eyes.

The torch.distributed-disabled warning from `dcp.load` is normal: we're a
single process, so it falls back to local I/O.
"""

from __future__ import annotations

# sys.path + MEGATRON_NATIVE_BUILD_DIR opt-out are wired by `pytest_discover_tests`.
import io
import json

import _distcp_shim
import torch
import torch.distributed.checkpoint as dcp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

SUPPORTED_DTYPES = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return bytes(t.contiguous().detach().cpu().view(torch.uint8).numpy().tobytes())


@st.composite
def _tensor(draw) -> torch.Tensor:
    ndim = draw(st.integers(min_value=1, max_value=4))
    shape = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(ndim))
    dtype = draw(st.sampled_from(SUPPORTED_DTYPES))
    n = 1
    for d in shape:
        n *= d
    base = torch.arange(n)
    if dtype.is_floating_point:
        return base.to(torch.float64).to(dtype).reshape(shape)
    return base.to(dtype).reshape(shape)


@st.composite
def _state_dict(draw) -> dict[str, torch.Tensor]:
    n = draw(st.integers(min_value=1, max_value=4))
    return {f"layer.{i}.weight": draw(_tensor()) for i in range(n)}


def _write_single_rank(out_dir, state_dict, objects=()) -> None:
    """Write `state_dict` as rank 0 into `out_dir` (one __0_0.distcp + .metadata)."""
    entries = [(fqn, _dtype_name(t.dtype), tuple(int(d) for d in t.shape), _tensor_bytes(t)) for fqn, t in state_dict.items()]
    _distcp_shim.write_rank_shard(str(out_dir), 0, entries, list(objects))

    # write_metadata reconstructs (file_offset, length) from (dtype, sharddim,
    # data_size) on the C++ side — see ComputeShardArchiveSize. We just hand
    # over the per-tensor sharding plan.
    tensors = []
    for fqn, t in state_dict.items():
        full_shape = [int(d) for d in t.shape]
        sharddim = [(0, int(d)) for d in t.shape]
        tensors.append((fqn, _dtype_name(t.dtype), full_shape, [(0, sharddim)]))
    _distcp_shim.write_metadata(str(out_dir), tensors, list(objects))


def _serialize_common_state(common_state) -> bytes:
    buffer = io.BytesIO()
    torch.save([common_state], buffer)
    return buffer.getvalue()


def _load_common_state(checkpoint_dir):
    key = "common_state/shard_0_1"
    loaded = {key: io.BytesIO()}
    dcp.load(loaded, checkpoint_id=str(checkpoint_dir))
    value = loaded[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False)
    return value[0]


def test_smoke_single_float32_tensor_dcp_load(tmp_path) -> None:
    state_dict = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    _write_single_rank(tmp_path, state_dict)

    loaded = {fqn: torch.zeros_like(t) for fqn, t in state_dict.items()}
    dcp.load(loaded, checkpoint_id=str(tmp_path))

    assert torch.equal(loaded["weight"], state_dict["weight"])


def test_multi_tensor_mixed_dtype_dcp_load(tmp_path) -> None:
    state_dict = {
        "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "b": torch.arange(6, dtype=torch.int64).reshape(2, 3),
        "c": torch.arange(8, dtype=torch.uint8).reshape(2, 2, 2),
        "nested.layer.weight": torch.arange(4, dtype=torch.float64).reshape(2, 2),
    }
    _write_single_rank(tmp_path, state_dict)

    loaded = {fqn: torch.zeros_like(t) for fqn, t in state_dict.items()}
    dcp.load(loaded, checkpoint_id=str(tmp_path))

    for fqn, expected in state_dict.items():
        assert torch.equal(loaded[fqn], expected), f"mismatch for {fqn!r}"
    assert not (tmp_path / "metadata.json").exists()


def test_common_state_is_embedded_and_fresh_for_each_checkpoint(tmp_path) -> None:
    key = "common_state/shard_0_1"
    checkpoints = []
    for iteration in (1, 2):
        checkpoint_dir = tmp_path / f"iter_{iteration:07d}"
        common_state = {"iteration": iteration, "scheduler": {"num_steps": iteration * 10}}
        objects = [(key, _serialize_common_state(common_state))]
        _write_single_rank(checkpoint_dir, {"weight": torch.tensor([iteration])}, objects)
        checkpoints.append((checkpoint_dir, common_state))

    for checkpoint_dir, expected in checkpoints:
        assert _load_common_state(checkpoint_dir) == expected
        assert not (checkpoint_dir / "common.pt").exists()
        assert json.loads((checkpoint_dir / "metadata.json").read_text()) == {
            "sharded_backend": "torch_dist",
            "sharded_backend_version": 1,
        }


@given(state_dict=_state_dict())
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dcp_load_roundtrip(state_dict, tmp_path_factory) -> None:
    out_dir = tmp_path_factory.mktemp("dcpload")
    _write_single_rank(out_dir, state_dict)

    loaded = {fqn: torch.zeros_like(t) for fqn, t in state_dict.items()}
    dcp.load(loaded, checkpoint_id=str(out_dir))

    for fqn, expected in state_dict.items():
        assert torch.equal(loaded[fqn], expected), f"mismatch for {fqn!r}"
