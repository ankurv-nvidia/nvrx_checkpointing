"""Unit tests for ``scripts/distcp_tree.py``.

Builds small distcp checkpoints two ways: a two-rank ``.metadata`` assembled
from torch's own DCP classes, with real ``torch.save`` blob payloads in the
``__<rank>_0.distcp`` files, and a single-process ``dcp.save``. Checks the
per-rank sizes, blob decoding, the payload embedded in the page, the page's
JavaScript data layer (when node is installed), and that the script runs
without torch and never executes pickled code.

Single-process and CPU-only.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib.util
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

# Load by path: importing through the package triggers its CMake autobuild.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "distcp_tree.py"
_spec = importlib.util.spec_from_file_location("distcp_tree", _SCRIPT_PATH)
distcp_tree = importlib.util.module_from_spec(_spec)
# Dataclass creation looks the defining module up by name.
sys.modules[_spec.name] = distcp_tree
_spec.loader.exec_module(distcp_tree)

_COMMON_PADDING = 300
_NODE_DRIVER = r"""
const fs = require("fs");
const path = require("path");
const dir = process.argv[2];
const DT = require(path.join(dir, "page.js"));
(async () => {
  const read = (name) => fs.readFileSync(path.join(dir, name), "utf8");
  const {doc, arrays} = await DT.decodePayload(read("doc.b64"), read("bin.b64"));
  const model = DT.buildModel(doc, arrays);
  const keys = DT.buildTree(model, "keys");
  const ev = DT.evaluate(model, keys, DT.rankEntries(model, 0), DT.rankEntries(model, 1), [], false);
  const optim = DT.buildTree(model, "optim");
  const evOptim = DT.evaluate(model, optim, null, null, [], false);
  let node = optim.root;
  for (const name of ["model", "layers", "0", "weight"]) node = optim.children[node].find((c) => optim.name[c] === name);
  const blobs = DT.evaluate(model, keys, null, null, [DT.makeFilter("kind", "select", "blob")], false);
  const onRank1 = DT.evaluate(model, keys, DT.rankEntries(model, 1), null, [], true);
  console.log(JSON.stringify({
    rootAll: ev.all[keys.root],
    rootA: ev.a[keys.root],
    rootB: ev.b[keys.root],
    rankTotals: Array.from(DT.rankTotals(model, keys, ev, keys.root)),
    optimRoot: evOptim.all[optim.root],
    weightChildren: optim.children[node].map((c) => optim.name[c]),
    blobLeaves: blobs.matchedLeaves,
    rank1Leaves: onRank1.matchedLeaves,
    badFilter: DT.makeFilter("all", "size", ">1XB").error !== undefined,
  }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""


class _Payload:
    """Pickles as a shell command call that decoding must leave inert."""

    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, (self.command,))


@dataclass
class _Checkpoint:
    root: Path
    sentinel: Path
    archive: dict[str, int]
    stored: dict[str, int]

    def rank_bytes(self, *, decoded: bool) -> dict[int, int]:
        blobs = self.archive if decoded else self.stored
        rank0 = 24 + 160 + 48 + blobs["common_state/shard_0_1"] + blobs["train_state.iteration/shard_0_1"] + blobs["untrusted/shard_0_1"]
        rank1 = 24 + 48 + 4 + blobs["rng_state/shard_0.1_1.2"]
        return {0: rank0, 1: rank1}


@pytest.fixture
def two_rank_ckpt(tmp_path: Path) -> _Checkpoint:
    """Two ranks: TP-sharded tensors, a rank-0-only tensor, a 0-dim tensor and four blobs."""
    root = tmp_path / "iter_0000007"
    root.mkdir()
    sentinel = tmp_path / "executed"
    cursors: dict[int, int] = {}
    storage: dict[MetadataIndex, _StorageInfo] = {}
    archive: dict[str, int] = {}
    stored: dict[str, int] = {}

    def put(rank: int, index: MetadataIndex, payload: bytes) -> None:
        path = f"__{rank}_0.distcp"
        offset = cursors.get(rank, 0)
        with (root / path).open("ab") as stream:
            stream.write(payload)
        cursors[rank] = offset + len(payload)
        storage[index] = _StorageInfo(path, offset, len(payload))

    def tensor(fqn: str, dtype: torch.dtype, size: tuple[int, ...], chunks: list[tuple]) -> TensorStorageMetadata:
        metas = []
        for rank, offsets, sizes in chunks:
            metas.append(ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes)))
            put(rank, MetadataIndex(fqn, torch.Size(offsets)), b"\0" * 32)
        return TensorStorageMetadata(TensorProperties(dtype=dtype), torch.Size(size), metas)

    def blob(fqn: str, rank: int, value: object, padding: int = 0) -> BytesStorageMetadata:
        buffer = io.BytesIO()
        torch.save([value], buffer)
        data = buffer.getvalue()
        archive[fqn] = len(data)
        stored[fqn] = len(data) + padding
        put(rank, MetadataIndex(fqn), data + b"\0" * padding)
        return BytesStorageMetadata()

    sharded = [(0, (0, 0), (2, 6)), (1, (2, 0), (2, 6))]
    common = {
        "iteration": 7,
        "args": argparse.Namespace(tensor_model_parallel_size=2, lr=0.5),
        "rng": torch.arange(5, dtype=torch.int64),
        "np": np.arange(3, dtype=np.uint32),
        "flag": True,
        "name": "nvrx",
    }
    state = {
        "model.layers.0.weight": tensor("model.layers.0.weight", torch.bfloat16, (4, 6), sharded),
        "model.embed.weight": tensor("model.embed.weight", torch.float32, (10, 4), [(0, (0, 0), (10, 4))]),
        "optimizer.state.exp_avg.model.layers.0.weight": tensor(
            "optimizer.state.exp_avg.model.layers.0.weight", torch.float32, (4, 6), sharded,
        ),
        "optimizer.state.step": tensor("optimizer.state.step", torch.float32, (), [(1, (), ())]),
        "common_state/shard_0_1": blob("common_state/shard_0_1", 0, common, padding=_COMMON_PADDING),
        "rng_state/shard_0.1_1.2": blob(
            "rng_state/shard_0.1_1.2", 1, [{"random_rng_state": (3, (1, 2), None), "torch_rng_state": torch.zeros(16, dtype=torch.uint8)}],
        ),
        "train_state.iteration/shard_0_1": blob("train_state.iteration/shard_0_1", 0, 12),
        "untrusted/shard_0_1": blob("untrusted/shard_0_1", 0, _Payload(f"touch {sentinel}")),
    }
    with (root / ".metadata").open("wb") as stream:
        pickle.dump(Metadata(state_dict_metadata=state, storage_data=storage), stream)
    return _Checkpoint(root, sentinel, archive, stored)


def _rank_bytes(report: object) -> dict[int, int]:
    totals: dict[int, int] = {}
    for leaf in report.leaves:
        for rank, slot in leaf.ranks.items():
            totals[rank] = totals.get(rank, 0) + slot[0]
    return totals


def _payload_blocks(html_text: str) -> tuple[str, str]:
    blocks = dict(re.findall(r'<script id="([a-z-]+)" type="application/octet-stream">([^<]*)</script>', html_text))
    return blocks["doc-data"], blocks["bin-data"]


def test_split_fqn_keeps_shard_suffixes_and_dtype_tokens() -> None:
    assert distcp_tree.split_fqn("decoder.layers.mlp.linear_fc1.weight") == ["decoder", "layers", "mlp", "linear_fc1", "weight"]
    assert distcp_tree.split_fqn("rng_state/shard_0.1_1.2") == ["rng_state", "shard_0.1_1.2"]
    assert distcp_tree.split_fqn("a.b._extra_state/shard_0.0_2.8") == ["a", "b", "_extra_state", "shard_0.0_2.8"]
    assert distcp_tree.split_fqn("opt.dtype_torch.bfloat16.bucket_idx_0") == ["opt", "dtype_torch.bfloat16", "bucket_idx_0"]
    assert distcp_tree.split_fqn("opt.dtype_(torch.bfloat16, torch.float32).exp_avg") == [
        "opt",
        "dtype_(torch.bfloat16, torch.float32)",
        "exp_avg",
    ]


def test_optimizer_paths_nest_states_under_their_parameter() -> None:
    fqns = [
        "m.w",
        "optimizer.state.exp_avg.m.w",
        "chained_0.optimizer.state.exp_avg_sq.m.w",
        "optimizer.state.step",
        "optimizer.distributed.bucket_idx_0.param",
    ]
    assert distcp_tree.optimizer_paths(fqns) == {
        0: ["m", "w", "(param)"],
        1: ["m", "w", "exp_avg"],
        2: ["m", "w", "chained_0.exp_avg_sq"],
    }


def test_metadata_only_report(two_rank_ckpt: _Checkpoint) -> None:
    report = distcp_tree.analyze(two_rank_ckpt.root)
    leaves = {leaf.fqn: leaf for leaf in report.leaves}

    assert (report.ranks, report.files, report.expanded, report.warnings) == (2, 2, False, [])
    weight = leaves["model.layers.0.weight"]
    assert (weight.kind, weight.dtype, weight.shape, weight.nbytes) == ("tensor", "bfloat16", (4, 6), 48)
    assert {rank: slot[0] for rank, slot in weight.ranks.items()} == {0: 24, 1: 24}
    assert weight.layout() == f"2 {chr(0xD7)} [2, 6]"
    assert {rank: slot[0] for rank, slot in leaves["model.embed.weight"].ranks.items()} == {0: 160}
    step = leaves["optimizer.state.step"]
    assert (step.kind, step.dtype, step.shape, step.nbytes, list(step.ranks)) == ("scalar", "float32", (), 4, [1])
    common = leaves["common_state/shard_0_1"]
    assert (common.kind, common.nbytes, common.blob) == ("blob", two_rank_ckpt.stored["common_state/shard_0_1"], None)
    assert _rank_bytes(report) == two_rank_ckpt.rank_bytes(decoded=False)


def test_expand_blobs_decodes_values_and_drops_slot_padding(two_rank_ckpt: _Checkpoint) -> None:
    report = distcp_tree.analyze(two_rank_ckpt.root, expand_blobs=True)
    leaves = {leaf.fqn: leaf for leaf in report.leaves}

    assert report.warnings == []
    common = leaves["common_state/shard_0_1"]
    assert common.nbytes == two_rank_ckpt.archive["common_state/shard_0_1"]
    assert common.blob["len"] - common.blob["arch"] == _COMMON_PADDING
    assert (common.blob["k"], common.blob["l"], common.blob["w"]) == ("dict", 6, 1)
    children = {child["n"]: child for child in common.blob["c"]}
    assert children["iteration"] == {"n": "iteration", "k": "scalar", "t": "int", "b": 8, "v": "7"}
    assert (children["flag"]["v"], children["name"]["v"]) == ("True", "'nvrx'")
    args = children["args"]
    assert (args["k"], args["t"]) == ("object", "Namespace")
    assert {child["n"]: child["v"] for child in args["c"]} == {"tensor_model_parallel_size": "2", "lr": "0.5"}
    assert (children["rng"]["k"], children["rng"]["d"], tuple(children["rng"]["s"]), children["rng"]["b"]) == ("tensor", "int64", (5,), 40)
    assert (children["np"]["k"], children["np"]["d"], tuple(children["np"]["s"]), children["np"]["b"]) == ("ndarray", "uint32", (3,), 12)

    iteration = leaves["train_state.iteration/shard_0_1"]
    assert (iteration.kind, iteration.blob["t"], iteration.blob["v"]) == ("scalar", "int", "12")
    rng = leaves["rng_state/shard_0.1_1.2"].blob
    assert rng["k"] == "list"
    torch_rng = {child["n"]: child for child in rng["c"][0]["c"]}["torch_rng_state"]
    assert (torch_rng["d"], tuple(torch_rng["s"])) == ("uint8", (16,))
    assert _rank_bytes(report) == two_rank_ckpt.rank_bytes(decoded=True)


def test_decoding_never_executes_pickled_code(two_rank_ckpt: _Checkpoint) -> None:
    report = distcp_tree.analyze(two_rank_ckpt.root, expand_blobs=True)
    untrusted = {leaf.fqn: leaf for leaf in report.leaves}["untrusted/shard_0_1"]

    assert not two_rank_ckpt.sentinel.exists()
    assert untrusted.blob["v"] == f"system('touch {two_rank_ckpt.sentinel}')"


def test_storage_outside_rank_files_is_unattributed(tmp_path: Path) -> None:
    index = MetadataIndex("w", torch.Size((0,)))
    chunk = ChunkStorageMetadata(torch.Size((0,)), torch.Size((8,)))
    meta = TensorStorageMetadata(TensorProperties(dtype=torch.float16), torch.Size((8,)), [chunk])
    with (tmp_path / ".metadata").open("wb") as stream:
        pickle.dump(Metadata(state_dict_metadata={"w": meta}, storage_data={index: _StorageInfo("data.bin", 0, 16)}), stream)
    report = distcp_tree.analyze(tmp_path)

    assert report.ranks == 0
    assert report.leaves[0].ranks == {distcp_tree.UNKNOWN_RANK: [16, 1, (8,), True]}
    assert report.warnings == ["1 storage entries in files without a __<rank>_<n>.distcp name"]


def test_dcp_save_checkpoint(tmp_path: Path) -> None:
    common = io.BytesIO()
    torch.save([{"iteration": 3}], common)
    common.seek(0)
    dcp.save({"layer": {"weight": torch.ones(2, 3), "step": 5}, "common_state/shard_0_1": common}, checkpoint_id=str(tmp_path))
    report = distcp_tree.analyze(tmp_path, expand_blobs=True)
    leaves = {leaf.fqn: leaf for leaf in report.leaves}

    assert (report.ranks, report.warnings) == (1, [])
    assert (leaves["layer.weight"].kind, leaves["layer.weight"].nbytes, list(leaves["layer.weight"].ranks)) == ("tensor", 24, [0])
    assert (leaves["layer.step"].kind, leaves["layer.step"].blob["v"]) == ("scalar", "5")
    common = leaves["common_state/shard_0_1"].blob
    assert (common["io"], common["w"]) == (1, 1)
    assert {child["n"]: child["v"] for child in common["c"]} == {"iteration": "3"}


def test_page_embeds_the_report(two_rank_ckpt: _Checkpoint) -> None:
    report = distcp_tree.analyze(two_rank_ckpt.root)
    doc, binary = distcp_tree.encode_report(report, title="iter <7> & co")
    html_text = distcp_tree.render_html(doc, binary)

    assert "<title>iter &lt;7&gt; &amp; co</title>" in html_text
    doc_text, bin_text = _payload_blocks(html_text)
    embedded = json.loads(gzip.decompress(base64.b64decode(doc_text)))
    raw = gzip.decompress(base64.b64decode(bin_text))
    arrays = {
        name: np.frombuffer(raw, dtype="<f8" if kind == "f64" else "<u4", count=length, offset=offset)
        for name, (offset, length, kind) in embedded["bin"].items()
    }
    assert (embedded["options"]["unique"], embedded["options"]["values"], embedded["inputLabel"]) == (False, False, ".metadata")
    assert embedded["summary"]["totalBytes"] == sum(two_rank_ckpt.rank_bytes(decoded=False).values())
    assert arrays["entBytes"].sum() == embedded["summary"]["totalBytes"]
    assert len(arrays["leafPtr"]) == len(embedded["leaves"]) + 1
    assert arrays["chunkPtr"][-1] == embedded["summary"]["chunks"] == 6
    by_fqn = {leaf["f"]: leaf for leaf in embedded["leaves"]}
    assert by_fqn["optimizer.state.exp_avg.model.layers.0.weight"]["a"] == ["model", "layers", "0", "weight", "exp_avg"]
    assert by_fqn["model.layers.0.weight"]["a"] == ["model", "layers", "0", "weight", "(param)"]


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_page_data_layer_matches_the_report(two_rank_ckpt: _Checkpoint, tmp_path: Path) -> None:
    report = distcp_tree.analyze(two_rank_ckpt.root, expand_blobs=True)
    doc, binary = distcp_tree.encode_report(report, title="t")
    html_text = distcp_tree.render_html(doc, binary)
    doc_text, bin_text = _payload_blocks(html_text)
    (tmp_path / "page.js").write_text(re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)[-1], encoding="utf-8")
    (tmp_path / "doc.b64").write_text(doc_text, encoding="ascii")
    (tmp_path / "bin.b64").write_text(bin_text, encoding="ascii")
    (tmp_path / "driver.js").write_text(_NODE_DRIVER, encoding="utf-8")
    result = subprocess.run([_NODE, str(tmp_path / "driver.js"), str(tmp_path)], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)

    expected = two_rank_ckpt.rank_bytes(decoded=True)
    assert values["rankTotals"] == [expected[0], expected[1]]
    assert (values["rootAll"], values["rootA"], values["rootB"]) == (sum(expected.values()), expected[0], expected[1])
    assert values["optimRoot"] == sum(expected.values())
    assert values["weightChildren"] == ["(param)", "exp_avg"]
    assert values["blobLeaves"] == 2
    assert values["rank1Leaves"] == 4
    assert values["badFilter"]


def test_script_runs_without_torch_or_numpy(two_rank_ckpt: _Checkpoint, tmp_path: Path) -> None:
    output = tmp_path / "page.html"
    blocker = (
        "import runpy, sys; sys.modules.update(torch=None, numpy=None); "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    command = [sys.executable, "-I", "-c", blocker, str(_SCRIPT_PATH), str(two_rank_ckpt.root), "--expand-blobs", "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert not two_rank_ckpt.sentinel.exists()


def test_main_reports_a_missing_metadata_file(tmp_path: Path) -> None:
    assert distcp_tree.main([str(tmp_path), "-o", str(tmp_path / "page.html")]) == 2
    assert not (tmp_path / "page.html").exists()
