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
import os

import _distcp_shim
import pytest
import torch
import torch.distributed.checkpoint as dcp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch.distributed.checkpoint.metadata import TensorProperties

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
        properties_state = TensorProperties.create_from_tensor(t).__getstate__()
        tensors.append((fqn, _dtype_name(t.dtype), full_shape, [(0, sharddim, properties_state)]))
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


_PROPERTY_STATE_SIZES = sorted({5, len(TensorProperties().__getstate__())})


def _write_tensor_chunks(out_dir, shape, chunks, properties_states):
    shards = []
    for rank, ((tensor, ranges), properties_state) in enumerate(zip(chunks, properties_states, strict=True)):
        _distcp_shim.write_rank_shard(
            str(out_dir),
            rank,
            [("weight", _dtype_name(tensor.dtype), tuple(end - start for start, end in ranges), tensor.numpy().tobytes())],
        )
        shards.append((rank, ranges, properties_state))
    _distcp_shim.write_metadata(str(out_dir), [("weight", _dtype_name(chunks[0][0].dtype), shape, shards)])
    return dcp.FileSystemReader(out_dir).read_metadata().state_dict_metadata["weight"]


@pytest.mark.parametrize("state_size", _PROPERTY_STATE_SIZES)
@pytest.mark.parametrize("shape,strides", [((2, 3), (3, 1)), ((1, 3), (19, 1)), ((3, 1), (1, 19)), ((1, 1), (31, 17))])
def test_tensor_properties_preserve_local_chunk_strides(tmp_path, state_size, shape, strides):
    chunks = []
    properties_states = []
    for rank in range(2):
        tensor = torch.empty_strided(shape, strides, dtype=torch.float32)
        tensor.copy_(torch.arange(tensor.numel()).reshape(shape) + rank * 100)
        chunks.append((tensor, [(0, shape[0]), (rank * shape[1], (rank + 1) * shape[1])]))
        properties_states.append(TensorProperties.create_from_tensor(tensor).__getstate__()[:state_size])

    global_shape = (shape[0], shape[1] * 2)
    metadata = _write_tensor_chunks(tmp_path, global_shape, chunks, properties_states)
    expected_properties = TensorProperties()
    expected_properties.__setstate__(properties_states[0])
    assert metadata.properties == expected_properties
    loaded = {"weight": torch.zeros(global_shape)}
    dcp.load(loaded, checkpoint_id=tmp_path)
    torch.testing.assert_close(loaded["weight"], torch.cat([tensor for tensor, _ranges in chunks], dim=1))


@pytest.mark.parametrize("state_size", _PROPERTY_STATE_SIZES)
def test_prepended_expert_axis_keeps_optional_strides(tmp_path, state_size):
    chunks = []
    properties_states = []
    for rank in range(2):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 100
        chunks.append((tensor, [(rank, rank + 1), (0, 2), (0, 3)]))
        properties_states.append(TensorProperties(dtype=tensor.dtype).__getstate__()[:state_size])

    metadata = _write_tensor_chunks(tmp_path, (2, 2, 3), chunks, properties_states)
    assert getattr(metadata.properties, "strides", None) is None
    loaded = {"weight": torch.zeros(2, 2, 3)}
    dcp.load(loaded, checkpoint_id=tmp_path)
    torch.testing.assert_close(loaded["weight"], torch.stack([tensor for tensor, _ranges in chunks]))


@pytest.mark.parametrize("state_size", _PROPERTY_STATE_SIZES)
def test_scalar_properties_preserve_empty_strides(tmp_path, state_size):
    scalar = torch.tensor(7.0)
    properties_state = TensorProperties.create_from_tensor(scalar).__getstate__()[:state_size]
    metadata = _write_tensor_chunks(tmp_path, (1,), [(scalar, [(0, 1)])], [properties_state])
    expected_properties = TensorProperties()
    expected_properties.__setstate__(properties_state)
    assert metadata.properties == expected_properties
    loaded = {"weight": torch.zeros(1)}
    dcp.load(loaded, checkpoint_id=tmp_path)
    torch.testing.assert_close(loaded["weight"], scalar.reshape(1))


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1", reason="requires checkpoint-server IPC support")
def test_registered_tensor_properties_match_dcp_sources(tmp_path):
    from nvrx_checkpointing import checkpoint
    from nvrx_checkpointing.megatron.mapping import ShardedTensor

    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    session = 4000
    root = checkpoint.model_state.create_root(session, 0, 1, ["localhost"], 1)
    try:
        for iteration in (1, 2):
            current = local + iteration
            scalar = torch.tensor(7.0 + iteration)
            singleton = torch.full((1, 3), float(iteration))
            irregular_experts = [
                ShardedTensor.from_rank_offsets("experts.irregular", current + rank, (0, rank, 2), prepend_axis_num=1) for rank in range(2)
            ]
            for shard in irregular_experts:
                shard.axis_fragmentations = None
            state = {
                "dense.weight": [ShardedTensor.from_rank_offsets("dense.weight", current + rank, (1, rank, 2)) for rank in range(2)],
                "experts.weight": [
                    ShardedTensor.from_rank_offsets("experts.weight", current + rank, (0, rank, 2), prepend_axis_num=1) for rank in range(2)
                ],
                "experts.irregular": irregular_experts,
                "scalar": scalar,
                "singleton": singleton,
            }
            checkpoint_dir = tmp_path / str(iteration)
            root.register_state_dict(state)
            root.save(iteration, checkpoint_dir)
            checkpoint.wait_for_ckpt_finalize(session, iteration)
            metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
            for key, tensor in (("dense.weight", current), ("experts.irregular", current), ("scalar", scalar), ("singleton", singleton)):
                expected = TensorProperties.create_from_tensor(tensor)
                assert metadata[key].properties == expected
            assert getattr(metadata["experts.weight"].properties, "strides", None) is None
            loaded = {
                "dense.weight": torch.zeros(2, 6),
                "experts.weight": torch.zeros(2, 2, 3),
                "experts.irregular": torch.zeros(2, 2, 3),
                "scalar": torch.zeros(1),
                "singleton": torch.zeros(1, 3),
            }
            dcp.load(loaded, checkpoint_id=checkpoint_dir)
            torch.testing.assert_close(loaded["dense.weight"], torch.cat([current, current + 1], dim=1))
            torch.testing.assert_close(loaded["experts.weight"], torch.stack([current, current + 1]))
            torch.testing.assert_close(loaded["experts.irregular"], torch.stack([current, current + 1]))
            torch.testing.assert_close(loaded["scalar"], scalar.reshape(1))
            torch.testing.assert_close(loaded["singleton"], singleton)
    finally:
        checkpoint.svc_spindown()


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
