# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

import os
import random
import struct
from argparse import Namespace
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from nvrx_checkpointing import checkpoint
from nvrx_checkpointing.megatron import load as native_load
from nvrx_checkpointing.megatron.native_state import decode_native_schema, is_native_state, read_native_state, restore_native_value
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, MetadataIndex

if TYPE_CHECKING:
    from pathlib import Path

_requires_server = pytest.mark.skipif(
    os.environ.get("MEGATRON_NATIVE_RUN_SERVER_TESTS") != "1",
    reason="requires checkpoint-server IPC support",
)


def _root() -> tuple[int, object]:
    session = random.randrange(1, 1 << 31)  # noqa: S311
    return session, checkpoint.model_state.create_root(session, 0, 1, ["localhost"], 1)


@pytest.mark.parametrize(
    ("initial", "later"),
    [
        ("register_native_state_dict", "register_state_dict"),
        ("register_state_dict", "register_native_state_dict"),
    ],
)
def test_registration_rejects_api_changes(initial: str, later: str) -> None:
    _, root = _root()
    getattr(root, initial)({"step": 1})
    root.wait_for_async_completion()
    getattr(root, later)({"step": 2})

    with pytest.raises(RuntimeError, match="registration API changed"):
        root.wait_for_async_completion()


@pytest.mark.parametrize(
    "value",
    [np.arange(6, dtype=np.dtype(">u4"))[::-1], np.array(2.5), torch.Generator(device="cpu").manual_seed(481)],
    ids=["noncontiguous-array", "zero-dimensional-array", "generator"],
)
def test_native_registration_accepts_arrays_and_generators(value: object) -> None:
    _, root = _root()
    root.register_native_state_dict({"value": value})

    root.wait_for_async_completion()


@pytest.mark.parametrize("value", [object(), lambda: None, np.longdouble("1.234567890123456789")])
def test_native_registration_rejects_unsupported_objects(value: object) -> None:
    _, root = _root()
    root.register_native_state_dict({"value": value})

    with pytest.raises(ValueError, match="Unsupported native checkpoint value type"):
        root.wait_for_async_completion()


def test_native_registration_rejects_recursive_containers() -> None:
    value = []
    value.append(value)
    _, root = _root()
    root.register_native_state_dict({"value": value})

    with pytest.raises(ValueError, match="recursive container"):
        root.wait_for_async_completion()


@pytest.mark.parametrize("depth", [100, 101])
def test_native_registration_enforces_the_decoder_depth_limit(depth: int) -> None:
    value = None
    for _ in range(depth):
        value = [value]
    _, root = _root()
    root.register_native_state_dict({"value": value})

    if depth > 100:
        with pytest.raises(ValueError, match="nesting exceeds the supported depth"):
            root.wait_for_async_completion()
    else:
        root.wait_for_async_completion()


@pytest.mark.parametrize("prefix", ["__nvrx_native_state_rank_", "__nvrx_native_dict_rank_"])
def test_native_registration_rejects_reserved_addresses(prefix: str) -> None:
    _, root = _root()
    root.register_native_state_dict({f"{prefix}0__": 1})

    with pytest.raises(ValueError, match="reserved metadata address"):
        root.wait_for_async_completion()


def test_native_registration_rejects_ambiguous_addresses() -> None:
    _, root = _root()
    root.register_native_state_dict({"model.weight": torch.ones(1), "model": {"weight": torch.zeros(1)}})

    with pytest.raises(ValueError, match="addresses overlap"):
        root.wait_for_async_completion()


@pytest.mark.integration
@_requires_server
def test_prepared_registration_persists_independent_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("load", "load_state_dict", "FileSystemReader"):
        monkeypatch.setattr(dcp, name, getattr(dcp, name))
    monkeypatch.setattr(dcp, native_load._INSTALL_MARKER, False, raising=False)  # noqa: SLF001
    native_load.install(ModuleType("megatron.training.checkpointing"))
    session, root = _root()
    try:
        for iteration in (1, 2):
            values = {
                "enabled": iteration == 1,
                "maximum": (1 << 64) - iteration,
                "minimum": -(1 << 63) + iteration - 1,
                "floating": -0.0 if iteration == 1 else 2.5,
            }
            weight = torch.arange(4, dtype=torch.float32) + iteration
            root.register_state_dict({"model": {"weight": weight}, "values": values})
            if iteration == 1:
                root.finalize_registry()
            directory = tmp_path / str(iteration)
            root.save(iteration, directory)
            checkpoint.wait_for_ckpt_finalize(session, iteration)
            metadata = dcp.FileSystemReader(directory).read_metadata()

            assert not is_native_state(metadata)
            assert (directory / "metadata.json").is_file()
            assert set(metadata.state_dict_metadata) == {"model.weight", *(f"values.{key}" for key in values)}
            for name, kind, code in (
                ("enabled", "bool", "Q"),
                ("maximum", "uint64", "Q"),
                ("minimum", "int64", "q"),
                ("floating", "float64", "d"),
            ):
                key = f"values.{name}"
                entry = metadata.state_dict_metadata[key]
                assert isinstance(entry, BytesStorageMetadata)
                assert entry.nvrx_scalar_type == kind
                location = metadata.storage_data[MetadataIndex(key)]
                assert location.length == 8
                with (directory / location.relative_path).open("rb") as stream:
                    stream.seek(location.offset)
                    assert stream.read(location.length) == struct.pack(f"<{code}", values[name])

            target = {
                "model": {"weight": torch.zeros_like(weight)},
                "values": {"enabled": False, "maximum": 0, "minimum": 0, "floating": 0.0},
            }
            tensor_destination = target["model"]["weight"]
            dcp.load(target, checkpoint_id=directory)

            assert target["model"]["weight"] is tensor_destination
            torch.testing.assert_close(tensor_destination, weight, rtol=0, atol=0)
            for name, expected in values.items():
                assert type(target["values"][name]) is type(expected)
                assert target["values"][name] == expected
            assert struct.pack("<d", target["values"]["floating"]) == struct.pack("<d", values["floating"])
    finally:
        checkpoint.svc_spindown()


@pytest.mark.integration
@_requires_server
def test_native_values_round_trip_without_torch_archives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python_rng = random.Random(17)  # noqa: S311
    python_rng.gauss(0, 1)
    python_state = python_rng.getstate()
    numpy_rng = np.random.RandomState(19)
    numpy_rng.normal(size=1)
    numpy_state = numpy_rng.get_state()
    generator = torch.Generator(device="cpu").manual_seed(481)
    torch.randn(3, generator=generator)
    generator_state = generator.get_state().clone()
    array = np.arange(12, dtype=np.dtype(">u4")).reshape(3, 4)[:, ::-1]
    session, root = _root()

    def reject_archive(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("native registration called torch.save")

    monkeypatch.setattr(torch, "save", reject_archive)
    state = {
        "step": 50,
        "maximum": (1 << 64) - 1,
        "minimum": -(1 << 63),
        "negative_zero": -0.0,
        "enabled": True,
        "nothing": None,
        "args": Namespace(text="λ\x00😀", empty_tuple=(), empty_list=[], empty_dict={}, optional=None),
        "mapping": {(1, "x"): ([], {}, None), "binary": bytes(range(256))},
        "array": array,
        "zero_dimensional_array": np.array(2.5),
        "python_rng": python_state,
        "numpy_rng": numpy_state,
        "generator": generator,
        "dtype": torch.float64,
        "device": torch.device("cpu"),
        "empty": torch.empty(0, 3),
    }
    try:
        root.register_native_state_dict(state)
        root.finalize_registry()
        root.save(1, tmp_path)
        checkpoint.wait_for_ckpt_finalize(session, 1)
        reader = dcp.FileSystemReader(tmp_path)
        metadata = reader.read_metadata()
        assert is_native_state(metadata)
        restored = read_native_state(reader, metadata)

        for key in ("step", "maximum", "minimum", "enabled", "nothing", "mapping", "python_rng", "dtype", "device"):
            assert restored[key] == state[key]
        assert struct.pack("<d", restored["negative_zero"]) == struct.pack("<d", -0.0)
        assert vars(restored["args"]) == vars(state["args"])
        np.testing.assert_array_equal(restored["array"], array)
        assert restored["array"].dtype == array.dtype
        np.testing.assert_array_equal(restored["zero_dimensional_array"], state["zero_dimensional_array"])
        assert restored["zero_dimensional_array"].shape == ()
        assert restored["numpy_rng"][0] == numpy_state[0]
        np.testing.assert_array_equal(restored["numpy_rng"][1], numpy_state[1])
        assert restored["numpy_rng"][2:] == numpy_state[2:]
        assert restored["generator"].device == "cpu"
        assert restored["generator"].state == generator_state.numpy().tobytes()
        destination_generator = torch.Generator(device="cpu").manual_seed(0)
        assert restore_native_value(restored["generator"], destination_generator) is destination_generator
        torch.testing.assert_close(destination_generator.get_state(), generator_state, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.rand(5, generator=destination_generator),
            torch.rand(5, generator=generator),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(restored["empty"], state["empty"])
        for key in ("step", "maximum", "minimum", "negative_zero", "enabled"):
            index = next(index for index in metadata.storage_data if index.fqn == key)
            assert metadata.storage_data[index].length == 8
        assert not (tmp_path / "metadata.json").exists()
    finally:
        checkpoint.svc_spindown()


@pytest.mark.integration
@_requires_server
def test_native_dictionary_and_sequence_growth_refreshes_payloads(tmp_path: Path) -> None:
    session, root = _root()
    try:
        for iteration, length in enumerate((0, 1024, 2), 1):
            state = {"values": {str(i): [i, None] for i in range(length)}, "sequence": tuple(range(length))}
            root.register_native_state_dict(state)
            if iteration == 1:
                root.finalize_registry()
            directory = tmp_path / str(iteration)
            root.save(iteration, directory)
            checkpoint.wait_for_ckpt_finalize(session, iteration)
            reader = dcp.FileSystemReader(directory)
            assert read_native_state(reader, reader.read_metadata()) == state
    finally:
        checkpoint.svc_spindown()


@pytest.mark.integration
@_requires_server
def test_native_tensor_addresses_retain_dotted_parent_prefix(tmp_path: Path) -> None:
    session, root = _root()
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    try:
        root.register_native_state_dict({"model": {"decoder.layers.0.weight": tensor}, "items": [tensor + 1]})
        root.finalize_registry()
        root.save(1, tmp_path)
        checkpoint.wait_for_ckpt_finalize(session, 1)
        metadata = dcp.FileSystemReader(tmp_path).read_metadata()
        names = set(metadata.state_dict_metadata)
        assert "model.decoder.layers.0.weight" in names
        assert "items.0" in names
        target = {"model.decoder.layers.0.weight": torch.zeros_like(tensor), "items.0": torch.zeros_like(tensor)}
        dcp.load(target, checkpoint_id=tmp_path, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
        torch.testing.assert_close(target["model.decoder.layers.0.weight"], tensor)
        torch.testing.assert_close(target["items.0"], tensor + 1)
    finally:
        checkpoint.svc_spindown()


@pytest.mark.integration
@_requires_server
def test_native_schema_tracks_stock_dcp_item_order_and_cost(tmp_path: Path) -> None:
    session, root = _root()
    state = {
        "metadata": {"late": 3, "early": None, "empty": {}},
        "args": Namespace(step=5),
        "objects": [1, 2, (3, 4)],
        "mapped": [{"step": 1}, []],
        "weights": {"z.weight": torch.ones(2, 3), "a.weight": torch.empty(0)},
        "mixed": [torch.ones(1), "scalar", [torch.ones(3)]],
        "tuple": (torch.ones(1), 4),
    }
    reference = dcp.DefaultSavePlanner()
    reference.set_up_planner(state, is_coordinator=True)
    expected = [
        (item.index.fqn, tuple(item.index.offset) if item.index.offset is not None else None, item.tensor_storage_size() or 1)
        for item in reference.create_local_plan().items
    ]
    try:
        root.register_native_state_dict(state)
        root.finalize_registry()
        root.save(1, tmp_path)
        checkpoint.wait_for_ckpt_finalize(session, 1)
        metadata = dcp.FileSystemReader(tmp_path).read_metadata()
        index = next(index for index in metadata.storage_data if index.fqn == "__nvrx_native_state_rank_0__")
        location = metadata.storage_data[index]
        with (tmp_path / location.relative_path).open("rb") as stream:
            stream.seek(location.offset)
            schema = decode_native_schema(stream.read(location.length))

        assert schema.version == 2
        assert schema.write_items == expected
    finally:
        checkpoint.svc_spindown()
