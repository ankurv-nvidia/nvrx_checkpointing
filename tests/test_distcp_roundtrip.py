"""Tier-8 per-tensor torch.load round-trip via the parity shim.

For each `(fqn, sharddim, offset, length)` in the shim's returned `writeLog`,
slice the rank file at `[offset:offset+length]` and `torch.load` it back into
a tensor; assert that round-trips to the input tensor with `torch.equal`.

What this validates (orthogonal to tier 7's "byte parity with dcp.save"):
    a) each per-tensor ZIP in `__<rank>_0.distcp` is complete and parseable
       in isolation,
    b) the bytes inside that ZIP decode back to the right dtype/shape/values,
    c) the offsets in `writeLog` actually slice ZIP boundaries — getting any
       of them wrong corrupts the next `torch.load` call.

This test does NOT depend on `dcp.save` as a reference, so it survives the
step-9 cutover when tier 7 is deleted.
"""

from __future__ import annotations

from io import BytesIO

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# sys.path + MEGATRON_NATIVE_BUILD_DIR opt-out are wired by `pytest_discover_tests`.
import _distcp_shim


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


@given(state_dict=_state_dict())
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_per_tensor_torch_load_roundtrip(state_dict, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("roundtrip")
    entries = [(fqn, _dtype_name(t.dtype), tuple(int(d) for d in t.shape), _tensor_bytes(t)) for fqn, t in state_dict.items()]
    write_log = _distcp_shim.write_rank_shard(str(out_dir), 0, entries)

    assert len(write_log) == len(state_dict), f"writeLog has {len(write_log)} entries; expected {len(state_dict)}"

    file_bytes = (out_dir / "__0_0.distcp").read_bytes()
    expected_items = list(state_dict.items())

    for i, (fqn, _sharddim, offset, length) in enumerate(write_log):
        expected_fqn, expected_tensor = expected_items[i]
        assert fqn == expected_fqn, f"writeLog order diverged at index {i}: got {fqn!r}, expected {expected_fqn!r}"
        loaded = torch.load(BytesIO(file_bytes[offset : offset + length]), weights_only=True)
        assert torch.equal(loaded, expected_tensor), (
            f"per-tensor round-trip mismatch for {fqn!r}: expected {expected_tensor!r}, got {loaded!r}"
        )


def test_smoke_single_float32_tensor_roundtrip(tmp_path):
    state_dict = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    entries = [(fqn, _dtype_name(t.dtype), tuple(int(d) for d in t.shape), _tensor_bytes(t)) for fqn, t in state_dict.items()]
    write_log = _distcp_shim.write_rank_shard(str(tmp_path), 0, entries)

    assert len(write_log) == 1
    file_bytes = (tmp_path / "__0_0.distcp").read_bytes()
    _fqn, _sharddim, offset, length = write_log[0]
    loaded = torch.load(BytesIO(file_bytes[offset : offset + length]), weights_only=True)
    assert torch.equal(loaded, state_dict["weight"])


def test_single_tensor_archive_is_byte_identical_to_torch_save(tmp_path):
    tensor = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    entries = [("weight", "float32", tuple(tensor.shape), _tensor_bytes(tensor))]

    _distcp_shim.write_rank_shard(str(tmp_path), 0, entries)
    native_bytes = (tmp_path / "__0_0.distcp").read_bytes()

    reference = BytesIO()
    torch.save(tensor, reference)

    # This covers more than loadability: timestamp normalization, the captured
    # miniz CRC used by serialization_id, and all ZIP headers must match too.
    assert native_bytes == reference.getvalue()
