# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

import argparse
import enum
import io
import math
import signal
import struct
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import _distcp_shim
import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

if TYPE_CHECKING:
    from pathlib import Path


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _string(value: str) -> bytes:
    return b"\x05" + _text(value)


def _integer(value: int) -> bytes:
    return b"\x01" + struct.pack("<q", value)


def _floating(value: float) -> bytes:
    return b"\x03" + struct.pack("<d", value)


def _dictionary(items: list[tuple[str, bytes]], *, namespace: bool = False) -> bytes:
    return bytes([10 if namespace else 9]) + struct.pack("<I", len(items)) + b"".join(_string(key) + value for key, value in items)


def _object(value: bytes) -> bytes:
    return b"NVRXOBJ1" + value


def _schema(
    records: list[tuple[str, int, bytes]],
    items: list[tuple[str, tuple[int, ...] | None, int]] | None = None,
) -> bytes:
    schema = (b"NVRXNST1" if items is None else b"NVRXNST2") + struct.pack("<I", len(records))
    schema += b"".join(_text(key) + bytes([kind]) for key, kind, _ in records)
    if items is not None:
        schema += struct.pack("<I", len(items))
        for key, offsets, cost in items:
            schema += _text(key) + bytes([offsets is not None])
            if offsets is not None:
                schema += struct.pack("<I", len(offsets)) + b"".join(struct.pack("<Q", offset) for offset in offsets)
            schema += struct.pack("<Q", cost)
    return schema


def _write_native(
    path: Path,
    records: list[tuple[str, int, bytes]],
    tensors: dict[str, torch.Tensor] | None = None,
    plans: dict[int, list[tuple[str, tuple[int, ...] | None, int]]] | None = None,
) -> None:
    objects = [(key, data) for key, _, data in records]
    if plans is None:
        objects.append(("__nvrx_native_state_rank_0__", _schema(records)))
    else:
        for rank, items in plans.items():
            local = [
                record
                for record in records
                if record[0].startswith(f"__nvrx_native_dict_rank_{rank}__.")
                or (rank == 0 and not record[0].startswith("__nvrx_native_dict_rank_"))
            ]
            objects.append((f"__nvrx_native_state_rank_{rank}__", _schema(local, items)))
    entries = []
    metadata = []
    for key, tensor in (tensors or {}).items():
        shape = list(tensor.shape)
        entries.append((key, str(tensor.dtype).removeprefix("torch."), shape, tensor.numpy().tobytes()))
        metadata.append((key, str(tensor.dtype).removeprefix("torch."), shape, [(0, [(0, dimension) for dimension in shape])]))
    _distcp_shim.write_rank_shard(str(path), 0, entries, objects)
    _distcp_shim.write_metadata(str(path), metadata, objects)


@pytest.mark.parametrize(
    ("kind", "payload", "expected"),
    [
        (1, struct.pack("<q", -(1 << 63)), -(1 << 63)),
        (2, struct.pack("<Q", (1 << 64) - 1), (1 << 64) - 1),
        (3, struct.pack("<d", math.inf), math.inf),
        (4, struct.pack("<Q", 1), True),
    ],
)
def test_raw_scalars_preserve_their_full_range(tmp_path: Path, kind: int, payload: bytes, expected: object) -> None:
    _write_native(tmp_path, [("value", kind, payload)])
    target = {"value": None}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["value"] == expected
    assert type(target["value"]) is type(expected)
    inspected = checkpoint.read_common_state(tmp_path)
    assert inspected["value"] == expected
    assert type(inspected["value"]) is type(expected)


def test_native_values_preserve_nulls_empty_containers_and_binary_data(tmp_path: Path) -> None:
    binary = bytes(range(256))
    payload = _object(
        _dictionary(
            [
                ("none", b"\x00"),
                ("empty_list", b"\x07\x00\x00\x00\x00"),
                ("empty_tuple", b"\x08\x00\x00\x00\x00"),
                ("empty_dict", b"\x09\x00\x00\x00\x00"),
                ("λ", b"\x06" + struct.pack("<Q", len(binary)) + binary),
            ],
        ),
    )
    _write_native(tmp_path, [("values", 10, payload)])
    target = {"values": None}

    checkpoint.load_checkpoint(tmp_path, target)

    expected = {"none": None, "empty_list": [], "empty_tuple": (), "empty_dict": {}, "λ": binary}
    assert target["values"] == expected
    assert checkpoint.read_common_state(tmp_path)["values"] == expected


def test_native_binary_blob_preserves_a_complete_torch_archive(tmp_path: Path) -> None:
    archive = io.BytesIO()
    torch.save({"iteration": 7}, archive)
    raw = archive.getvalue()
    _write_native(tmp_path, [("payload", 10, _object(b"\x06" + struct.pack("<Q", len(raw)) + raw))])
    target = {"payload": b""}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target == {"payload": raw}
    assert checkpoint.read_common_state(tmp_path)["payload"] == raw


def test_native_numpy_rng_keeps_float64_gaussian_precision(tmp_path: Path) -> None:
    keys = np.array([0, 1, (1 << 32) - 1], dtype=np.uint32)
    array = b"\x0b" + _text("uint32") + struct.pack("<IQQ", 1, 3, keys.nbytes) + keys.tobytes()
    gaussian = math.nextafter(1.0, 2.0)
    state = _object(b"\x08" + struct.pack("<I", 5) + _string("MT19937") + array + _integer(2) + _integer(1) + _floating(gaussian))
    _write_native(tmp_path, [("rng", 10, state)])
    target = {"rng": None}

    checkpoint.load_checkpoint(tmp_path, target)

    restored = target["rng"]
    assert restored[0] == "MT19937"
    np.testing.assert_array_equal(restored[1], keys)
    assert restored[1].dtype == keys.dtype
    assert restored[2:] == (2, 1, gaussian)
    assert struct.pack("<d", restored[-1]) == struct.pack("<d", gaussian)


def test_native_generator_restores_state_and_destination_identity(tmp_path: Path) -> None:
    expected = torch.Generator(device="cpu").manual_seed(1234)
    state = expected.get_state()
    raw_state = state.numpy().tobytes()
    payload = _object(b"\x10" + _text("cpu") + struct.pack("<Q", len(raw_state)) + raw_state)
    _write_native(tmp_path, [("generator", 10, payload)])
    destination = torch.Generator(device="cpu").manual_seed(7)
    target = {"generator": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["generator"] is destination
    assert torch.equal(destination.get_state(), state)
    torch.testing.assert_close(torch.rand(8, generator=destination), torch.rand(8, generator=expected))
    inspected = checkpoint.read_common_state(tmp_path)["generator"]
    assert isinstance(inspected, torch.Generator)
    assert torch.equal(inspected.get_state(), state)


@pytest.mark.parametrize(
    ("kind", "payload", "message"),
    [
        (1, b"\x00" * 7, "eight|8"),
        (4, struct.pack("<Q", 2), "boolean"),
        (10, b"bad magic", "native|magic"),
        (10, _object(_integer(7)) + b"extra", "trailing"),
        (10, _object(_string("abc"))[:-1], "truncated"),
        (10, _object(b"\xff"), "unsupported|kind"),
    ],
)
def test_native_corruption_is_rejected(tmp_path: Path, kind: int, payload: bytes, message: str) -> None:
    _write_native(tmp_path, [("value", kind, payload)])

    with pytest.raises((RuntimeError, ValueError), match=f"(?i){message}"):
        checkpoint.load_checkpoint(tmp_path, {"value": None})


def test_load_restores_native_values_without_python_pickle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arguments = _object(_dictionary([("num_layers", _integer(12)), ("title", _string("λ"))], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments), ("iteration", 1, struct.pack("<q", 50))], {"weight": torch.ones(2, 3)})
    weight = torch.zeros(2, 3)
    target = {"args": None, "iteration": None, "weight": weight}

    def reject_pickle(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Native checkpoint restoration called the Python pickle loader")

    monkeypatch.setattr(torch, "load", reject_pickle)
    monkeypatch.setattr(dcp, "load", reject_pickle)
    monkeypatch.setattr(dcp, "load_state_dict", reject_pickle)
    checkpoint.load_checkpoint(tmp_path, target)

    assert vars(target["args"]) == {"num_layers": 12, "title": "λ"}
    assert target["iteration"] == 50
    assert target["weight"] is weight
    torch.testing.assert_close(weight, torch.ones(2, 3))


def test_native_load_keeps_tensor_and_optimizer_aliases(tmp_path: Path) -> None:
    metadata = _dictionary(
        [
            (
                "weight",
                _dictionary(
                    [
                        ("lr", _floating(0.002)),
                        ("step", _integer(7)),
                        ("betas", b"\x08" + struct.pack("<I", 2) + _floating(0.8) + _floating(0.95)),
                    ],
                ),
            ),
        ],
    )
    _write_native(
        tmp_path,
        [("__nvrx_native_dict_rank_0__.optimizer.param_to_group_meta", 20, _object(metadata)), ("iteration", 1, struct.pack("<q", 7))],
        {"model.weight": torch.ones(2, 3), "optimizer.state.weight.exp_avg": torch.full((2, 3), 2.0)},
    )
    weight = torch.zeros(2, 3, requires_grad=True)
    average = torch.zeros(2, 3)
    group = {"lr": 1.0, "step": 0, "betas": (0.1, 0.2), "old": 9}
    optimizer = {"state": {"weight": {"exp_avg": average}}, "param_to_group_meta": {"weight": group}}
    raw_optimizer = optimizer.copy()
    target = {"model": {"weight": weight}, "optimizer": optimizer, "iteration": 0}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["model"]["weight"] is weight
    assert target["optimizer"]["state"]["weight"]["exp_avg"] is average
    assert raw_optimizer["param_to_group_meta"]["weight"] is group
    assert group == {"lr": 0.002, "step": 7, "betas": (0.8, 0.95)}
    torch.testing.assert_close(weight, torch.ones(2, 3))
    torch.testing.assert_close(average, torch.full((2, 3), 2.0))


def test_shared_walk_preserves_mixed_container_and_tensor_aliases(tmp_path: Path) -> None:
    _write_native(
        tmp_path,
        [("items.0.step", 1, struct.pack("<q", 17))],
        {"items.0.weight": torch.arange(3.0), "items.1.options.weight": torch.ones(2)},
    )
    weight = torch.zeros(3)
    options = SimpleNamespace(weight=torch.zeros(2))
    group = {"step": 0, "weight": weight}
    pair = (group, SimpleNamespace(options=options))
    target = {"items": pair}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["items"] is pair
    assert pair[0] is group
    assert pair[0]["weight"] is weight
    assert pair[0]["step"] == 17
    assert pair[1].options is options
    torch.testing.assert_close(weight, torch.arange(3.0))
    torch.testing.assert_close(options.weight, torch.ones(2))


def test_load_request_scan_does_not_acquire_tensor_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_native(tmp_path, [], {"weight": torch.arange(3.0)})
    destination = torch.zeros(3)
    address = destination.data_ptr()

    def reject_payload(*_args: object, **_kwargs: object) -> None:
        pytest.fail("load request scan acquired destination tensor storage")

    monkeypatch.setattr(torch.Tensor, "__dlpack__", reject_payload)
    target = {"weight": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["weight"] is destination
    assert destination.data_ptr() == address
    torch.testing.assert_close(destination, torch.arange(3.0))


def test_full_load_keeps_live_runtime_arguments_unchanged(tmp_path: Path) -> None:
    nested = _dictionary(
        [("options", b"\x07" + struct.pack("<I", 2) + _integer(2) + _integer(3)), ("groups", _dictionary([("lr", _floating(0.002))]))],
        namespace=True,
    )
    arguments = _object(_dictionary([("rank", _integer(0)), ("train_iters", _integer(50)), ("nested", nested)], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments)])
    options, groups = [1], {"lr": 1.0}
    nested_target = SimpleNamespace(options=options, groups=groups)
    runtime = argparse.Namespace(rank=3, train_iters=125, nested=nested_target)
    target = {"args": runtime}

    checkpoint.load_checkpoint(tmp_path, target)

    checkpoint_args = target["args"]
    assert checkpoint_args is not runtime
    assert vars(checkpoint_args) == {"rank": 0, "train_iters": 50, "nested": SimpleNamespace(options=[2, 3], groups={"lr": 0.002})}
    assert type(checkpoint_args.nested) is SimpleNamespace
    assert runtime.rank == 3
    assert runtime.train_iters == 125
    assert runtime.nested is nested_target
    assert runtime.nested.options is options
    assert runtime.nested.groups is groups
    assert options == [1]
    assert groups == {"lr": 1.0}


def test_peer_dictionaries_merge_expert_keys(tmp_path: Path) -> None:
    _write_native(
        tmp_path,
        [
            (
                "__nvrx_native_dict_rank_0__.optimizer.groups",
                20,
                _object(_dictionary([("expert0", _floating(0.01)), ("shared", _integer(7))])),
            ),
            (
                "__nvrx_native_dict_rank_1__.optimizer.groups",
                20,
                _object(_dictionary([("expert4", _floating(0.02)), ("shared", _integer(7))])),
            ),
        ],
    )
    groups = {"expert0": 1.0, "expert4": 1.0, "shared": 0}
    target = {"optimizer": {"groups": groups}}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["optimizer"]["groups"] is groups
    assert groups == {"expert0": 0.01, "expert4": 0.02, "shared": 7}


def test_conflicting_peer_dictionary_values_fail(tmp_path: Path) -> None:
    _write_native(
        tmp_path,
        [
            (f"__nvrx_native_dict_rank_{rank}__.optimizer.groups", 20, _object(_dictionary([("shared", _integer(rank))])))
            for rank in range(2)
        ],
    )

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)conflicting.*dictionary"):
        checkpoint.load_checkpoint(tmp_path, {"optimizer": {"groups": {"shared": 0}}})


def test_native_containers_restore_size_and_empty_values(tmp_path: Path) -> None:
    options = _dictionary(
        [("values", b"\x07" + struct.pack("<I", 3) + _integer(1) + _integer(2) + b"\x00"), ("empty", b"\x08\x00\x00\x00\x00")],
    )
    _write_native(tmp_path, [("__nvrx_native_dict_rank_0__.options", 20, _object(options))])
    values = [9]
    options_target = {"values": values, "empty": (1,), "removed": True}
    target = {"options": options_target}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["options"] is options_target
    assert target["options"]["values"] is values
    assert target == {"options": {"values": [1, 2, None], "empty": ()}}


def test_native_namespace_preserves_destination_class(tmp_path: Path) -> None:
    payload = _object(
        _dictionary(
            [("values", b"\x07" + struct.pack("<I", 2) + _integer(2) + _integer(3)), ("child", _dictionary([("step", _integer(9))]))],
            namespace=True,
        ),
    )
    _write_native(tmp_path, [("options", 10, payload)])
    target_namespace = SimpleNamespace(values=[1], child={"step": 0})
    values, child = target_namespace.values, target_namespace.child
    target = {"options": target_namespace}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["options"] is target_namespace
    assert type(target_namespace) is SimpleNamespace
    assert target_namespace.values is values
    assert target_namespace.child is child
    assert target_namespace == SimpleNamespace(values=[2, 3], child={"step": 9})


def test_native_enum_uses_supplied_destination_type(tmp_path: Path) -> None:
    class Mode(enum.Enum):
        SLOW = "slow"
        FAST = "fast"

    _write_native(tmp_path, [("mode", 10, _object(b"\x0d" + _string("fast")))])
    target = {"mode": Mode.SLOW}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["mode"] is Mode.FAST
    assert checkpoint.read_common_state(tmp_path)["mode"] == "fast"


def test_missing_unknown_enum_template_is_rejected(tmp_path: Path) -> None:
    _write_native(tmp_path, [("other_enum", 10, _object(b"\x0d" + _integer(1)))])

    with pytest.raises((RuntimeError, TypeError, ValueError), match=r"(?i)destination enum"):
        checkpoint.load_checkpoint(tmp_path, {"other_enum": None})


def test_subset_load_ignores_unrequested_enum_type(tmp_path: Path) -> None:
    _write_native(tmp_path, [("mode", 10, _object(b"\x0d" + _integer(1))), ("iteration", 1, struct.pack("<q", 7))])
    target = {"iteration": 0}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target == {"iteration": 7}


def test_known_signal_enum_restores_without_destination_template(tmp_path: Path) -> None:
    arguments = _object(_dictionary([("exit_signal", b"\x0d" + _integer(signal.SIGTERM.value))], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments)])
    target = {"args": None}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["args"].exit_signal is signal.SIGTERM
    assert type(target["args"].exit_signal) is signal.Signals


@pytest.mark.parametrize("scope", [None, "none"])
def test_bootstrap_recovers_known_enums_before_argument_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str | None,
) -> None:
    enums = ModuleType("megatron.core.transformer.enums")

    class AttnBackend(enum.Enum):
        unfused = 3

    class InferenceCudaGraphScope(enum.Enum):
        none = 1

    enums.AttnBackend = AttnBackend
    enums.InferenceCudaGraphScope = InferenceCudaGraphScope
    packages = {name: ModuleType(name) for name in ("megatron", "megatron.core", "megatron.core.transformer")}
    packages[enums.__name__] = enums
    for name, module in packages.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(packages[parent], child, module)
    arguments = _object(
        _dictionary([("attention_backend", b"\x0d" + _integer(3)), ("inference_cuda_graph_scope", b"\x0d" + _integer(1))], namespace=True),
    )
    _write_native(tmp_path, [("args", 10, arguments)])
    runtime = argparse.Namespace(attention_backend="unfused", inference_cuda_graph_scope=scope)
    target = {"args": runtime}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["args"].attention_backend is AttnBackend.unfused
    assert target["args"].inference_cuda_graph_scope is InferenceCudaGraphScope.none
    assert runtime.attention_backend == "unfused"
    assert runtime.inference_cuda_graph_scope == scope


def test_native_empty_tensor_blob_preserves_destination_identity(tmp_path: Path) -> None:
    payload = _object(b"\x0f" + _text("float32") + struct.pack("<IQQ", 2, 0, 3))
    _write_native(tmp_path, [("empty", 10, payload)])
    destination = torch.empty(0, 3, requires_grad=True)
    target = {"empty": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["empty"] is destination
    assert tuple(destination.shape) == (0, 3)


def test_native_tuple_recovers_torch_size_destination(tmp_path: Path) -> None:
    _write_native(tmp_path, [("shape", 10, _object(b"\x08" + struct.pack("<I", 2) + _integer(2) + _integer(3)))])
    target = {"shape": torch.Size((1, 1))}

    checkpoint.load_checkpoint(tmp_path, target)

    assert type(target["shape"]) is torch.Size
    assert target["shape"] == torch.Size((2, 3))


@pytest.mark.parametrize(
    ("destination", "captured"),
    [(np.float32(0), 0.125), (np.int16(0), -7), (np.uint64(0), (1 << 64) - 1), (np.bool_(), True)],
)
def test_native_primitive_recovers_numpy_scalar_dtype(tmp_path: Path, destination: np.generic, captured: float) -> None:
    if isinstance(captured, bool):
        value = b"\x04" + bytes([captured])
    elif isinstance(captured, float):
        value = _floating(captured)
    elif captured > (1 << 63) - 1:
        value = b"\x02" + struct.pack("<Q", captured)
    else:
        value = _integer(captured)
    _write_native(tmp_path, [("value", 10, _object(value))])
    target = {"value": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert type(target["value"]) is type(destination)
    assert target["value"].dtype == destination.dtype
    assert target["value"] == captured


@pytest.mark.parametrize(("destination", "captured"), [(np.str_("x"), "abcdef"), (np.bytes_(b"x"), b"abcdef")])
def test_native_numpy_string_preserves_value_beyond_placeholder_width(
    tmp_path: Path,
    destination: np.generic,
    captured: str | bytes,
) -> None:
    value = _string(captured) if isinstance(captured, str) else b"\x06" + struct.pack("<Q", len(captured)) + captured
    _write_native(tmp_path, [("value", 10, _object(value))])
    target = {"value": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert type(target["value"]) is type(destination)
    assert target["value"] == captured
    assert len(target["value"]) == 6


@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_native_numpy_array_preserves_destination_storage(tmp_path: Path, shape: tuple[int, ...]) -> None:
    expected = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    array = b"\x0b" + _text("float64") + struct.pack("<I", len(shape))
    array += b"".join(struct.pack("<Q", dimension) for dimension in shape)
    array += struct.pack("<Q", expected.nbytes) + expected.tobytes()
    _write_native(tmp_path, [("array", 10, _object(array))])
    destination = np.zeros(shape, dtype=expected.dtype)
    target = {"array": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["array"] is destination
    np.testing.assert_array_equal(destination, expected)


def test_native_dtype_and_device_values_restore(tmp_path: Path) -> None:
    _write_native(tmp_path, [("dtype", 10, _object(b"\x0c" + _text("float64"))), ("device", 10, _object(b"\x0e" + _text("cpu")))])
    target = {"dtype": torch.float32, "device": torch.device("cpu")}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target == {"dtype": torch.float64, "device": torch.device("cpu")}


def test_missing_destination_key_is_rejected(tmp_path: Path) -> None:
    _write_native(tmp_path, [("iteration", 1, struct.pack("<q", 7))])

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)missing.*optimizer.step"):
        checkpoint.load_checkpoint(tmp_path, {"optimizer": {"step": 0}})


def test_partial_load_retains_missing_destination_values(tmp_path: Path) -> None:
    _write_native(tmp_path, [("iteration", 1, struct.pack("<q", 7))])
    target = {"iteration": 0, "optimizer": {"step": 91}}

    checkpoint.load_checkpoint(tmp_path, target, allow_partial_load=True)

    assert target == {"iteration": 7, "optimizer": {"step": 91}}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_stock_dcp_checkpoint_restores_values_and_tensor_aliases(tmp_path: Path, dtype: torch.dtype) -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(dtype)
    arguments = argparse.Namespace(num_layers=12, tokens=[1, 2, 3])
    dcp.save({"model": {"weight": weight}, "iteration": 7, "args": arguments, "enabled": True}, checkpoint_id=tmp_path)
    destination = torch.zeros_like(weight, requires_grad=dtype.is_floating_point)
    target = {"model": {"weight": destination}, "iteration": 0, "args": None, "enabled": False}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["model"]["weight"] is destination
    torch.testing.assert_close(destination, weight)
    assert target["iteration"] == 7
    assert target["enabled"] is True
    assert vars(target["args"]) == vars(arguments)


@pytest.mark.parametrize("shape", [(), (0, 3)])
def test_stock_dcp_scalar_and_empty_tensors_preserve_destination_identity(tmp_path: Path, shape: tuple[int, ...]) -> None:
    expected = torch.full(shape, 3.5)
    dcp.save({"weight": expected}, checkpoint_id=tmp_path)
    destination = torch.zeros_like(expected, requires_grad=True)
    target = {"weight": destination}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target["weight"] is destination
    assert tuple(destination.shape) == shape
    torch.testing.assert_close(destination, expected)


def test_stock_dcp_load_copies_into_noncontiguous_destination(tmp_path: Path) -> None:
    expected = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    dcp.save({"weight": expected}, checkpoint_id=tmp_path)
    backing = torch.zeros(2, 3)
    destination = backing.t()

    checkpoint.load_checkpoint(tmp_path, {"weight": destination})

    torch.testing.assert_close(destination, expected)
    torch.testing.assert_close(backing, expected.t())


def test_stock_subset_load_ignores_unrequested_object_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dcp.save({"weight": expected, "unrequested": argparse.Namespace(value=7)}, checkpoint_id=tmp_path)
    destination = torch.zeros_like(expected)

    def reject_pickle(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Unrequested checkpoint objects do not need deserialization")

    monkeypatch.setattr(torch, "load", reject_pickle)

    checkpoint.load_checkpoint(tmp_path, {"weight": destination})

    torch.testing.assert_close(destination, expected)


@pytest.mark.parametrize("destination_dtype", [torch.float32, torch.float64])
def test_stock_dcp_load_copies_noncontiguous_source_with_destination_conversion(tmp_path: Path, destination_dtype: torch.dtype) -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    assert not expected.is_contiguous()
    dcp.save({"weight": expected}, checkpoint_id=tmp_path)
    destination = torch.zeros(expected.shape, dtype=destination_dtype)

    checkpoint.load_checkpoint(tmp_path, {"weight": destination})

    assert destination.dtype is destination_dtype
    torch.testing.assert_close(destination, expected.to(destination_dtype))


@pytest.mark.parametrize("payload", [b"\x80abcdefg", b"PK\x03\x04data"])
def test_unmarked_binary_blob_uses_destination_bytes_type(tmp_path: Path, payload: bytes) -> None:
    objects = [("payload", payload)]
    _distcp_shim.write_rank_shard(str(tmp_path), 0, [], objects)
    _distcp_shim.write_metadata(str(tmp_path), [], objects)
    target = {"payload": b""}

    checkpoint.load_checkpoint(tmp_path, target)

    assert target == {"payload": payload}


def _reference_descriptions(
    world_size: int,
    prefix_count: int,
) -> tuple[dict, dict, list[torch.Generator]]:
    local_plans = []
    generators = []
    for rank in range(world_size):
        generator = torch.Generator(device="cpu").manual_seed(100 + rank)
        generators.append(generator)
        state = {f"prefix_{index}": torch.zeros(index + 1) for index in range(prefix_count)}
        state["objects"] = {
            "generator": generator,
            "pure_list": [rank, 9],
            "tuple": (rank, "value"),
            "nested": [{"key": rank}, [rank]],
            "peer": {"shared": 7, f"expert_{rank}": rank + 10},
        }
        planner = DefaultSavePlanner()
        planner.set_up_planner(state, is_coordinator=rank == 0)
        local_plans.append(planner.create_local_plan())
    descriptions = {
        rank: [
            (item.index.fqn, tuple(item.index.offset) if item.index.offset is not None else None, item.tensor_storage_size() or 1)
            for item in plan.items
        ]
        for rank, plan in enumerate(local_plans)
    }
    planned, _metadata = DefaultSavePlanner().create_global_plan(local_plans)
    sources = {
        (item.index.fqn, tuple(item.index.offset) if item.index.offset is not None else None): rank
        for rank, plan in enumerate(planned)
        for item in plan.items
    }
    return descriptions, sources, generators


@pytest.mark.parametrize("world_size", [2, 4])
def test_native_dictionary_load_uses_recorded_dcp_sources(tmp_path: Path, world_size: int) -> None:
    chosen = []
    for prefix_count in (1, world_size):
        descriptions, expected, generators = _reference_descriptions(world_size, prefix_count)
        records = []
        for rank, generator in enumerate(generators):
            state = generator.get_state().numpy().tobytes()
            value = _dictionary(
                [
                    ("generator", b"\x10" + _text("cpu") + struct.pack("<Q", len(state)) + state),
                    ("pure_list", b"\x07" + struct.pack("<I", 2) + _integer(rank) + _integer(9)),
                    ("tuple", b"\x08" + struct.pack("<I", 2) + _integer(rank) + _string("value")),
                    (
                        "nested",
                        b"\x07"
                        + struct.pack("<I", 2)
                        + _dictionary([("key", _integer(rank))])
                        + b"\x07"
                        + struct.pack("<I", 1)
                        + _integer(rank),
                    ),
                    ("peer", _dictionary([("shared", _integer(7)), (f"expert_{rank}", _integer(rank + 10))])),
                ],
            )
            records.append((f"__nvrx_native_dict_rank_{rank}__.objects", 20, _object(value)))
        directory = tmp_path / str(prefix_count)
        tensors = {f"prefix_{index}": torch.zeros(index + 1) for index in range(prefix_count)}
        _write_native(directory, records, tensors, descriptions)
        destination_generator = torch.Generator(device="cpu")
        target = {"objects": {"generator": destination_generator, "pure_list": [], "tuple": (), "nested": [{"key": 0}, []], "peer": {}}}

        checkpoint.load_checkpoint(directory, target)

        objects = target["objects"]
        source = expected[("objects.generator", None)]
        chosen.append(source)
        assert objects["generator"] is destination_generator
        assert torch.equal(objects["generator"].get_state(), generators[source].get_state())
        assert objects["pure_list"] == [expected[("objects.pure_list", None)], 9]
        assert objects["tuple"] == (expected[("objects.tuple", None)], "value")
        assert objects["nested"] == [{"key": expected[("objects.nested.0.key", None)]}, [expected[("objects.nested.1", None)]]]
        assert objects["peer"] == {"shared": 7, **{f"expert_{rank}": rank + 10 for rank in range(world_size)}}
        inspected = checkpoint.read_common_state(directory)
        assert torch.equal(inspected["objects"]["generator"].get_state(), generators[source].get_state())
    assert len(set(chosen)) == 2


def test_conflicting_dictionary_without_logical_description_is_rejected(tmp_path: Path) -> None:
    records = [(f"__nvrx_native_dict_rank_{rank}__.objects", 20, _object(_dictionary([("value", _integer(rank))]))) for rank in range(2)]
    _write_native(tmp_path, records, plans={0: [("other", None, 1)], 1: [("other", None, 1)]})

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)missing logical source description"):
        checkpoint.load_checkpoint(tmp_path, {"objects": {"value": 0}})


def test_missing_selected_dictionary_rank_is_rejected(tmp_path: Path) -> None:
    _write_native(
        tmp_path,
        [("__nvrx_native_dict_rank_0__.objects", 20, _object(_dictionary([("value", _integer(1))])))],
        plans={0: [], 2: [("objects.value", None, 1)]},
    )

    with pytest.raises((RuntimeError, ValueError), match=r"(?i)missing selected.*dictionary source"):
        checkpoint.load_checkpoint(tmp_path, {"objects": {"value": 0}})
