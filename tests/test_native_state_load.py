# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101, SLF001

from __future__ import annotations

import argparse
import enum
import importlib.util
import math
import signal
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import _distcp_shim
import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType


def _module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_load(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).parents[1] / "megatron"
    codec = _module(monkeypatch, "nvrx_checkpointing.megatron.native_state", root / "native_state.py")
    load = _module(monkeypatch, "nvrx_checkpointing.megatron.native_load_test", root / "load.py")
    for name in ("load", "load_state_dict", "FileSystemReader"):
        monkeypatch.setattr(dcp, name, getattr(dcp, name))
    monkeypatch.setattr(dcp, load._INSTALL_MARKER, False, raising=False)
    load.install(ModuleType("megatron.training.checkpointing"))
    return codec, load


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
def test_raw_scalars_preserve_their_full_range(
    native_load: tuple[ModuleType, ModuleType],
    kind: int,
    payload: bytes,
    expected: object,
) -> None:
    codec, _ = native_load

    assert codec.decode_native_value(kind, payload) == expected
    assert type(codec.decode_native_value(kind, payload)) is type(expected)


def test_native_values_preserve_nulls_empty_containers_and_binary_data(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
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

    restored = codec.decode_native_value(10, payload)

    assert restored == {"none": None, "empty_list": [], "empty_tuple": (), "empty_dict": {}, "λ": binary}


def test_native_numpy_rng_keeps_float64_gaussian_precision(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    keys = np.array([0, 1, (1 << 32) - 1], dtype=np.uint32)
    array = b"\x0b" + _text("uint32") + struct.pack("<IQQ", 1, 3, keys.nbytes) + keys.tobytes()
    gaussian = math.nextafter(1.0, 2.0)
    state = _object(b"\x08" + struct.pack("<I", 5) + _string("MT19937") + array + _integer(2) + _integer(1) + _floating(gaussian))

    restored = codec.decode_native_value(10, state)

    assert restored[0] == "MT19937"
    np.testing.assert_array_equal(restored[1], keys)
    assert restored[2:] == (2, 1, gaussian)


def test_native_generator_restores_state_and_destination_identity(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    expected = torch.Generator(device="cpu").manual_seed(1234)
    state = expected.get_state()
    raw_state = state.numpy().tobytes()
    payload = _object(b"\x10" + _text("cpu") + struct.pack("<Q", len(raw_state)) + raw_state)
    expected_values = torch.rand(8, generator=expected)
    captured = codec.decode_native_value(10, payload)
    destination = torch.Generator(device="cpu").manual_seed(7)

    restored = codec.restore_native_value(captured, destination)

    assert restored is destination
    assert torch.equal(restored.get_state(), state)
    torch.testing.assert_close(torch.rand(8, generator=restored), expected_values)
    inspected = codec.materialize_native_value(captured)
    assert isinstance(inspected, torch.Generator)
    assert torch.equal(inspected.get_state(), state)
    torch.testing.assert_close(torch.rand(8, generator=inspected), expected_values)


@pytest.mark.parametrize(
    ("kind", "payload", "message"),
    [
        (1, b"\x00" * 7, "eight bytes"),
        (4, struct.pack("<Q", 2), "boolean"),
        (10, b"bad magic", "Unsupported native"),
        (10, _object(_integer(7)) + b"extra", "Trailing bytes"),
        (10, _object(_string("abc"))[:-1], "Truncated"),
        (10, _object(b"\xff"), "Unsupported native"),
    ],
)
def test_native_corruption_is_rejected(
    native_load: tuple[ModuleType, ModuleType],
    kind: int,
    payload: bytes,
    message: str,
) -> None:
    codec, _ = native_load

    with pytest.raises(ValueError, match=message):
        codec.decode_native_value(kind, payload)


def test_load_restores_native_values_without_pickle(
    native_load: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _load = native_load
    arguments = _object(_dictionary([("num_layers", _integer(12)), ("title", _string("λ"))], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments), ("iteration", 1, struct.pack("<q", 50))])
    target = {"args": None, "iteration": None}

    def reject_pickle(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Native objects do not use the pickle loader")

    monkeypatch.setattr(torch, "load", reject_pickle)
    dcp.load(target, checkpoint_id=tmp_path)

    assert vars(target["args"]) == {"num_layers": 12, "title": "λ"}
    assert target["iteration"] == 50


def test_native_load_keeps_tensor_and_optimizer_aliases(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
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
    weight, average = torch.zeros(2, 3), torch.zeros(2, 3)
    group = {"lr": 1.0, "step": 0, "betas": (0.1, 0.2), "old": 9}
    optimizer = {"state": {"weight": {"exp_avg": average}}, "param_to_group_meta": {"weight": group}}
    raw_optimizer = optimizer.copy()
    target = {"model": {"weight": weight}, "optimizer": optimizer, "iteration": 0}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["model"]["weight"] is weight
    assert target["optimizer"]["state"]["weight"]["exp_avg"] is average
    assert raw_optimizer["param_to_group_meta"]["weight"] is group
    assert group == {"lr": 0.002, "step": 7, "betas": (0.8, 0.95)}
    torch.testing.assert_close(weight, torch.ones(2, 3))
    torch.testing.assert_close(average, torch.full((2, 3), 2.0))


def test_full_load_keeps_live_runtime_arguments_unchanged(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
    nested = _dictionary(
        [("options", b"\x07" + struct.pack("<I", 2) + _integer(2) + _integer(3)), ("groups", _dictionary([("lr", _floating(0.002))]))],
        namespace=True,
    )
    arguments = _object(
        _dictionary([("rank", _integer(0)), ("train_iters", _integer(50)), ("nested", nested)], namespace=True),
    )
    _write_native(tmp_path, [("args", 10, arguments)])
    options, groups = [1], {"lr": 1.0}
    nested_target = SimpleNamespace(options=options, groups=groups)
    runtime = argparse.Namespace(rank=3, train_iters=125, nested=nested_target)
    target = {"args": runtime}

    dcp.load(target, checkpoint_id=tmp_path)

    checkpoint_args = target["args"]
    assert checkpoint_args is not runtime
    assert checkpoint_args.rank == 0
    assert checkpoint_args.train_iters == 50
    assert checkpoint_args.nested.options == [2, 3]
    assert checkpoint_args.nested.groups == {"lr": 0.002}
    assert type(checkpoint_args.nested) is SimpleNamespace
    assert runtime.rank == 3
    assert runtime.train_iters == 125
    assert runtime.nested is nested_target
    assert runtime.nested.options is options
    assert runtime.nested.groups is groups
    assert options == [1]
    assert groups == {"lr": 1.0}


def test_peer_dictionaries_merge_expert_keys(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
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

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["optimizer"]["groups"] is groups
    assert groups == {"expert0": 0.01, "expert4": 0.02, "shared": 7}


def test_conflicting_peer_dictionary_values_fail(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
    _write_native(
        tmp_path,
        [
            (f"__nvrx_native_dict_rank_{rank}__.optimizer.groups", 20, _object(_dictionary([("shared", _integer(rank))])))
            for rank in range(2)
        ],
    )

    with pytest.raises(CheckpointException, match="Conflicting replicated dictionary value"):
        dcp.load({"optimizer": {"groups": {"shared": 0}}}, checkpoint_id=tmp_path)


def test_native_containers_restore_size_and_empty_values(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
    options = _dictionary(
        [("values", b"\x07" + struct.pack("<I", 3) + _integer(1) + _integer(2) + b"\x00"), ("empty", b"\x08\x00\x00\x00\x00")],
    )
    _write_native(tmp_path, [("__nvrx_native_dict_rank_0__.options", 20, _object(options))])
    values = [9]
    options_target = {"values": values, "empty": (1,), "removed": True}
    target = {"options": options_target}

    dcp.load(target, checkpoint_id=tmp_path)

    assert target["options"] is options_target
    assert target["options"]["values"] is values
    assert target == {"options": {"values": [1, 2, None], "empty": ()}}


def test_bootstrap_uses_runtime_enum_template(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    _, load = native_load

    class Mode(enum.Enum):
        SLOW = "slow"
        FAST = "fast"

    arguments = _object(_dictionary([("mode", b"\x0d" + _string("fast")), ("layers", _integer(12))], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments)])
    checkpointing = ModuleType("native_checkpoint_bootstrap_test")

    def load_base_checkpoint(_directory: Path, args: argparse.Namespace, rank0: bool = False) -> dict:
        assert args.mode is Mode.SLOW
        assert rank0
        target = {"args": None}
        dcp.load(target, checkpoint_id=tmp_path)
        return target

    checkpointing._load_base_checkpoint = load_base_checkpoint
    load.install(checkpointing)
    arguments_target = argparse.Namespace(mode=Mode.SLOW, layers=4)

    target = checkpointing._load_base_checkpoint(tmp_path, arguments_target, rank0=True)

    assert target["args"].mode is Mode.FAST
    assert target["args"].layers == 12
    assert arguments_target.mode is Mode.SLOW
    assert arguments_target.layers == 4


def test_native_namespace_preserves_destination_class(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    target = SimpleNamespace(values=[1], child={"step": 0})
    values, child = target.values, target.child
    captured = argparse.Namespace(values=[2, 3], child={"step": 9})

    restored = codec.restore_native_value(captured, target)

    assert restored is target
    assert type(restored) is SimpleNamespace
    assert restored.values is values
    assert restored.child is child
    assert restored == SimpleNamespace(values=[2, 3], child={"step": 9})


@pytest.mark.parametrize("scope", [None, "none"])
def test_bootstrap_recovers_known_enums_before_argument_validation(
    native_load: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scope: str | None,
) -> None:
    _, load = native_load
    enums = ModuleType("megatron.core.transformer.enums")

    class AttnBackend(enum.Enum):
        unfused = 3

    class InferenceCudaGraphScope(enum.Enum):
        none = 1

    enums.AttnBackend = AttnBackend
    enums.InferenceCudaGraphScope = InferenceCudaGraphScope
    monkeypatch.setitem(sys.modules, enums.__name__, enums)
    arguments = _object(
        _dictionary(
            [("attention_backend", b"\x0d" + _integer(3)), ("inference_cuda_graph_scope", b"\x0d" + _integer(1))],
            namespace=True,
        ),
    )
    _write_native(tmp_path, [("args", 10, arguments)])
    checkpointing = ModuleType("native_prevalidation_bootstrap_test")

    def load_base_checkpoint(_directory: Path, args: argparse.Namespace, rank0: bool = False) -> dict:
        assert args.inference_cuda_graph_scope == scope
        assert rank0
        target = {"args": None}
        dcp.load(target, checkpoint_id=tmp_path)
        return target

    checkpointing._load_base_checkpoint = load_base_checkpoint
    load.install(checkpointing)
    template = argparse.Namespace(attention_backend="unfused", inference_cuda_graph_scope=scope)

    target = checkpointing._load_base_checkpoint(tmp_path, template, rank0=True)

    assert target["args"].attention_backend is AttnBackend.unfused
    assert target["args"].inference_cuda_graph_scope is InferenceCudaGraphScope.none
    assert template.attention_backend == "unfused"
    assert template.inference_cuda_graph_scope == scope


def test_missing_unknown_enum_template_is_rejected(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load

    with pytest.raises(TypeError, match="Missing destination enum type"):
        codec.restore_native_value(codec.NativeEnum(1), None, "args.other_enum")


def test_known_signal_enum_restores_in_bootstrap_and_inspection(
    native_load: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    codec, _ = native_load
    arguments = _object(_dictionary([("exit_signal", b"\x0d" + _integer(signal.SIGTERM.value))], namespace=True))
    _write_native(tmp_path, [("args", 10, arguments)])
    reader = dcp.FileSystemReader(tmp_path)
    captured = codec.read_native_state(reader, reader.read_metadata())["args"]

    inspected = codec.restore_native_value(captured, argparse.Namespace(), "args")
    target = {"args": None}
    dcp.load(target, checkpoint_id=tmp_path)

    assert inspected.exit_signal is signal.SIGTERM
    assert type(inspected.exit_signal) is signal.Signals
    assert target["args"].exit_signal is signal.SIGTERM
    assert type(target["args"].exit_signal) is signal.Signals


def test_native_tuple_recovers_torch_size_destination(native_load: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    del native_load
    _write_native(tmp_path, [("shape", 10, _object(b"\x08" + struct.pack("<I", 2) + _integer(2) + _integer(3)))])
    target = {"shape": torch.Size((1, 1))}

    dcp.load(target, checkpoint_id=tmp_path)

    assert type(target["shape"]) is torch.Size
    assert target["shape"] == torch.Size((2, 3))


@pytest.mark.parametrize(
    ("destination", "captured"),
    [(np.float32(0), 0.125), (np.int16(0), -7), (np.uint64(0), (1 << 64) - 1), (np.bool_(), True)],
)
def test_native_primitive_recovers_numpy_scalar_dtype(
    native_load: tuple[ModuleType, ModuleType],
    tmp_path: Path,
    destination: np.generic,
    captured: float,
) -> None:
    del native_load
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

    dcp.load(target, checkpoint_id=tmp_path)

    assert type(target["value"]) is type(destination)
    assert target["value"].dtype == destination.dtype
    assert target["value"] == captured


@pytest.mark.parametrize(("destination", "captured"), [(np.str_("x"), "abcdef"), (np.bytes_(b"x"), b"abcdef")])
def test_native_numpy_string_preserves_value_beyond_placeholder_width(
    native_load: tuple[ModuleType, ModuleType],
    tmp_path: Path,
    destination: np.generic,
    captured: str | bytes,
) -> None:
    del native_load
    value = _string(captured) if isinstance(captured, str) else b"\x06" + struct.pack("<Q", len(captured)) + captured
    _write_native(tmp_path, [("value", 10, _object(value))])
    target = {"value": destination}

    dcp.load(target, checkpoint_id=tmp_path)

    assert type(target["value"]) is type(destination)
    assert target["value"] == captured
    assert len(target["value"]) == 6


def _logical_shard(rank: int) -> torch.Tensor:
    offset, size = (0, 7) if rank % 2 == 0 else (7, 4)
    local = torch.zeros(size)

    def create_write_items(key: str, value: torch.Tensor) -> list[WriteItem]:
        return [
            WriteItem(
                index=MetadataIndex(key, [offset]),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(offsets=torch.Size([offset]), sizes=value.size()),
                    properties=TensorProperties.create_from_tensor(value),
                    size=torch.Size([11]),
                ),
            ),
        ]

    local.__create_write_items__ = create_write_items
    return local


def _reference_descriptions(
    world_size: int,
    prefix_count: int,
    *,
    sharded: bool = False,
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
        if sharded:
            state["uneven_shard"] = _logical_shard(rank)
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
@pytest.mark.parametrize("prefix_count", [1, 4])
def test_native_source_replay_matches_stock_dcp_planner(
    native_load: tuple[ModuleType, ModuleType],
    world_size: int,
    prefix_count: int,
) -> None:
    codec, _ = native_load
    descriptions, expected, _generators = _reference_descriptions(world_size, prefix_count, sharded=True)

    sources = codec.native_source_ranks(descriptions)

    assert sources == expected
    assert ("objects.pure_list", None) in sources
    assert ("objects.tuple", None) in sources
    assert ("objects.nested.0.key", None) in sources
    assert ("objects.nested.1", None) in sources
    assert all(cost == 44 for items in descriptions.values() for key, _offset, cost in items if key == "uneven_shard")


def test_native_schema_retains_object_and_scalar_tensor_addresses(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    items = [("tensor", (), 4), ("object", None, 1), ("shard", (7,), 44)]

    schema = codec.decode_native_schema(_schema([], items))

    assert schema.version == 2
    assert schema.write_items == items


@pytest.mark.parametrize("world_size", [2, 4])
def test_native_dictionary_load_uses_recorded_dcp_sources(
    native_load: tuple[ModuleType, ModuleType],
    tmp_path: Path,
    world_size: int,
) -> None:
    codec, _ = native_load
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
        target = {
            "objects": {"generator": destination_generator, "pure_list": [], "tuple": (), "nested": [{"key": 0}, []], "peer": {}},
        }

        dcp.load(target, checkpoint_id=directory)

        objects = target["objects"]
        source = expected[("objects.generator", None)]
        chosen.append(source)
        assert objects["generator"] is destination_generator
        assert torch.equal(objects["generator"].get_state(), generators[source].get_state())
        assert objects["pure_list"] == [expected[("objects.pure_list", None)], 9]
        assert objects["tuple"] == (expected[("objects.tuple", None)], "value")
        assert objects["nested"] == [{"key": expected[("objects.nested.0.key", None)]}, [expected[("objects.nested.1", None)]]]
        assert objects["peer"] == {"shared": 7, **{f"expert_{rank}": rank + 10 for rank in range(world_size)}}
        reader = dcp.FileSystemReader(directory)
        inspected = codec.read_native_state(reader, reader.read_metadata())
        assert inspected["objects"]["generator"].state == generators[source].get_state().numpy().tobytes()
    assert len(set(chosen)) == 2


def test_conflicting_dictionary_without_logical_description_is_rejected(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    records = codec.NativeRecords()
    records.write_items = {0: [("other", None, 1)], 1: [("other", None, 1)]}
    for rank in range(2):
        records[f"__nvrx_native_dict_rank_{rank}__.objects"] = (20, _object(_dictionary([("value", _integer(rank))])))

    with pytest.raises(ValueError, match="Missing logical source description"):
        codec.decode_native_records(records)


def test_missing_selected_dictionary_rank_is_rejected(native_load: tuple[ModuleType, ModuleType]) -> None:
    codec, _ = native_load
    records = codec.NativeRecords()
    records.write_items = {2: [("objects.value", None, 1)]}
    records["__nvrx_native_dict_rank_0__.objects"] = (20, _object(_dictionary([("value", _integer(1))])))

    with pytest.raises(ValueError, match="Missing selected native dictionary source"):
        codec.decode_native_records(records)
