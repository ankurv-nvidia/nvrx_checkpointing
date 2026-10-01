"""Unit tests for ``scripts/registry_tree.py``.

Writes small registries in the checkpoint daemon's serialization (rank
topology, then the flattened state tree) for per-rank and global views of the
same two-rank state. Checks per-rank and unique sizes, scalar and blob
decoding, input selection, the page payload, the page's JavaScript data layer
(when node is installed), and that the script runs without torch or numpy.

Single-process and CPU-only.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib.util
import io
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Load by path: importing through the package triggers its CMake autobuild.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "registry_tree.py"
_spec = importlib.util.spec_from_file_location("registry_tree", _SCRIPT_PATH)
registry_tree = importlib.util.module_from_spec(_spec)
# Dataclass creation looks the defining module up by name.
sys.modules[_spec.name] = registry_tree
_spec.loader.exec_module(registry_tree)
page = registry_tree.page

_BF16, _F32, _I64 = 2, 3, 12
_DICT, _SCALAR, _BLOB, _TENSOR_SHARD, _SHARDED_TENSOR = 1, 2, 3, 4, 5
_INT64_BITS = 0b10100
_DOUBLE_BITS = 0b11100
_LR = struct.unpack("<Q", struct.pack("<d", 0.25))[0]
_REMOVED = 0xFFFFFFFF
_PEER = "node-a:4100"
_NODE = shutil.which("node")
_NODE_DRIVER = r"""
const fs = require("fs");
const path = require("path");
const dir = process.argv[2];
const DT = require(path.join(dir, "page.js"));
(async () => {
  const read = (name) => fs.readFileSync(path.join(dir, name), "utf8");
  const {doc, arrays} = await DT.decodePayload(read("doc.b64"), read("bin.b64"));
  const model = DT.buildModel(doc, arrays);
  const tree = DT.buildTree(model, "keys");
  const entA = DT.rankEntries(model, 0);
  const entB = DT.rankEntries(model, 1);
  const ev = DT.evaluate(model, tree, entA, entB, [], false);
  const ctx = {model, tree, ev, entA, entB};
  const node = tree.leafNode[model.leaves.findIndex((leaf) => leaf.f === "train.iteration")];
  console.log(JSON.stringify({
    rootAll: ev.all[tree.root],
    rootUnique: ev.unique[tree.root],
    rankTotals: Array.from(DT.rankTotals(model, tree, ev, tree.root)),
    iteration: [DT.cellValue(ctx, "ashape", node), DT.cellValue(ctx, "bshape", node), DT.cellValue(ctx, "value", node)],
  }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""


def _dim(dims: tuple[int, ...]) -> bytes:
    return struct.pack("<5Q", *dims, *[0] * (4 - len(dims)), len(dims))


def _ranges(ranges: list[tuple[int, int]]) -> bytes:
    flat = [value for pair in ranges for value in pair]
    return struct.pack("<9Q", *flat, *[0] * (8 - len(flat)), len(ranges))


class _Tree:
    """Serializes a hand-built state tree the way the checkpoint daemon does."""

    def __init__(self, rank: int, world: int = 2) -> None:
        self.rank = rank
        self.world = world
        self.nodes: list[list] = [[0, 0, "", _DICT, []]]
        self.dicts = {"": 0}

    def _parent(self, fqn: str) -> tuple[str, int]:
        *folders, name = fqn.split(".")
        parent, path = 0, ""
        for folder in folders:
            path = f"{path}.{folder}" if path else folder
            if path not in self.dicts:
                self.dicts[path] = self._append(folder, parent, _DICT, [])
            parent = self.dicts[path]
        return name, parent

    def _append(self, name: str, parent: int, kind: int, payload: object) -> int:
        handle = len(self.nodes)
        self.nodes.append([handle, parent, name, kind, payload])
        self.nodes[parent][4].append(handle)
        return handle

    def add(self, fqn: str, kind: int, payload: object) -> None:
        name, parent = self._parent(fqn)
        self._append(name, parent, kind, payload)

    def removed(self) -> None:
        self.nodes.append([_REMOVED, _REMOVED, "", 0, None])

    def write(self, path: Path) -> None:
        out = bytearray(struct.pack("<IIII", self.rank, self.world, 2, 1))
        out += struct.pack("<I", len(_PEER)) + _PEER.encode()
        out += struct.pack("<I", len(self.nodes))
        for handle, parent, name, kind, payload in self.nodes:
            out += struct.pack("<BBHII", kind, len(name), 0, handle, parent) + name.encode()
            if kind == _DICT:
                out += struct.pack(f"<I{len(payload)}I", len(payload), *payload)
            elif kind == _SCALAR:
                out += struct.pack("<B7xQ", *payload)
            elif kind == _BLOB:
                out += struct.pack("<I", len(payload)) + payload
            elif kind == _TENSOR_SHARD:
                dtype, fulldim, ranges = payload
                out += _ranges(ranges) + _dim(fulldim) + struct.pack("<B7xQ4sI", dtype, 0, b"cuda", 0)
            elif kind == _SHARDED_TENSOR:
                dtype, fulldim, shards = payload
                out += struct.pack("<IB", len(shards), dtype) + _dim(fulldim)
                out += b"".join(_ranges(ranges) + struct.pack("<4sIQII", b"cuda", 0, 0, rank, 0) for rank, ranges in shards)
        path.write_bytes(bytes(out))


def _torch_blob(value: object, padding: int) -> bytes:
    buffer = io.BytesIO()
    torch.save([value], buffer)
    return buffer.getvalue() + b"\0" * padding


def _per_rank(common: bytes) -> list[_Tree]:
    trees = []
    for rank in range(2):
        tree = _Tree(rank)
        rows = [(2 * rank, 2 * rank + 2), (0, 6)]
        tree.add("model.layers.0.weight", _TENSOR_SHARD, (_BF16, (4, 6), rows))
        for layer in range(2):
            tree.add("model.experts.fc1.weight", _TENSOR_SHARD, (_F32, (2, 8, 4), [(layer, layer + 1), (4 * rank, 4 * rank + 4), (0, 4)]))
        tree.add("model.norm.weight", _TENSOR_SHARD, (_BF16, (6,), [(0, 6)]))
        tree.add("optimizer.state.exp_avg.model.layers.0.weight", _TENSOR_SHARD, (_F32, (4, 6), rows))
        if rank == 0:
            tree.add("model.embed.weight", _TENSOR_SHARD, (_I64, (10, 4), [(0, 10), (0, 4)]))
            tree.add("common_state/shard_0_1", _BLOB, common)
        tree.add(f"rng_state/shard_0.{rank}_1.2", _BLOB, b"x" * (100 + rank))
        tree.add("train.iteration", _SCALAR, (_INT64_BITS, 7 + rank))
        tree.add("train.lr", _SCALAR, (_DOUBLE_BITS, _LR))
        trees.append(tree)
    return trees


def _global() -> _Tree:
    tree = _Tree(0)
    tree.add("model.layers.0.weight", _SHARDED_TENSOR, (_BF16, (4, 6), [(r, [(2 * r, 2 * r + 2), (0, 6)]) for r in range(2)]))
    experts = [(r, [(layer, layer + 1), (4 * r, 4 * r + 4), (0, 4)]) for r in range(2) for layer in range(2)]
    tree.add("model.experts.fc1.weight", _SHARDED_TENSOR, (_F32, (2, 8, 4), experts))
    tree.add("model.norm.weight", _SHARDED_TENSOR, (_BF16, (6,), [(r, [(0, 6)]) for r in range(2)]))
    exp_avg = [(r, [(2 * r, 2 * r + 2), (0, 6)]) for r in range(2)]
    tree.add("optimizer.state.exp_avg.model.layers.0.weight", _SHARDED_TENSOR, (_F32, (4, 6), exp_avg))
    tree.add("model.embed.weight", _SHARDED_TENSOR, (_I64, (10, 4), [(0, [(0, 10), (0, 4)])]))
    tree.removed()
    tree.add("common_state/shard_0_1", _BLOB, b"")
    tree.add("rng_state/shard_0.0_1.2", _BLOB, b"")
    tree.add("train.iteration", _SCALAR, (_INT64_BITS, 7))
    tree.add("train.lr", _SCALAR, (_DOUBLE_BITS, _LR))
    return tree


def _write_per_rank(directory: Path, common: bytes = b"common") -> list[_Tree]:
    directory.mkdir(parents=True, exist_ok=True)
    trees = _per_rank(common)
    for tree in trees:
        tree.write(directory / f"registry.{tree.rank}.bin")
    return trees


def _tensor_view(report: object) -> dict[str, tuple[int, dict[int, int]]]:
    return {leaf.fqn: (leaf.unique, {rank: slot[0] for rank, slot in leaf.ranks.items()}) for leaf in report.leaves if leaf.dtype}


def _tensor_bytes(report: object) -> dict[int, int]:
    totals: dict[int, int] = {}
    for _, ranks in _tensor_view(report).values():
        for rank, nbytes in ranks.items():
            totals[rank] = totals.get(rank, 0) + nbytes
    return totals


def test_per_rank_registries(tmp_path: Path) -> None:
    common = _torch_blob({"iteration": 7}, padding=64)
    _write_per_rank(tmp_path, common)
    report = registry_tree.analyze([tmp_path])
    leaves = {leaf.fqn: leaf for leaf in report.leaves}

    assert (report.ranks, report.files, report.warnings, report.hosts, report.ranks_per_host) == (2, 2, [], [_PEER], 2)
    assert _tensor_bytes(report) == {0: 532, 1: 212}
    fc1 = leaves["model.experts.fc1.weight"]
    assert (fc1.dtype, fc1.shape, fc1.chunk_count(), fc1.unique, fc1.layout()) == ("float32", (2, 8, 4), 4, 256, f"4 {chr(0xD7)} [1, 4, 4]")
    assert (leaves["model.norm.weight"].nbytes, leaves["model.norm.weight"].unique) == (24, 12)
    iteration = leaves["train.iteration"]
    assert (iteration.kind, iteration.nbytes, iteration.unique, iteration.rank_labels) == ("scalar", 16, 8, {0: "= 7", 1: "= 8"})
    assert iteration.blob == {"k": "scalar", "t": "int64", "b": 8, "v": "2 distinct values across ranks"}
    assert leaves["train.lr"].blob["v"] == "0.25"
    rng = leaves["rng_state/shard_0.1_1.2"]
    assert ({rank: slot[0] for rank, slot in rng.ranks.items()}, rng.blob) == ({1: 101}, None)
    assert leaves["common_state/shard_0_1"].nbytes == len(common)


def test_global_registry(tmp_path: Path) -> None:
    _global().write(tmp_path / "registry.bin")
    report = registry_tree.analyze([tmp_path / "registry.bin"])
    leaves = {leaf.fqn: leaf for leaf in report.leaves}

    assert (report.ranks, report.files) == (2, 1)
    assert _tensor_bytes(report) == {0: 532, 1: 212}
    assert leaves["model.norm.weight"].unique == 12
    common = leaves["common_state/shard_0_1"]
    assert (common.nbytes, list(common.ranks)) == (0, [page.UNKNOWN_RANK])
    assert report.warnings == [
        "the global registry's 2 blobs hold no bytes and no rank; per-rank registries hold their sizes",
        "the global registry's 2 scalars carry rank 0's values and no rank",
    ]
    assert report.notes[-1].startswith("This is the global registry")


def test_per_rank_and_global_registries_agree(tmp_path: Path) -> None:
    _write_per_rank(tmp_path / "per_rank")
    _global().write(tmp_path / "registry.bin")

    assert _tensor_view(registry_tree.analyze([tmp_path / "per_rank"])) == _tensor_view(registry_tree.analyze([tmp_path / "registry.bin"]))


def test_expand_blobs_decodes_inline_payloads(tmp_path: Path) -> None:
    common = _torch_blob({"iteration": 7, "args": argparse.Namespace(lr=0.5)}, padding=64)
    _write_per_rank(tmp_path, common)
    leaves = {leaf.fqn: leaf for leaf in registry_tree.analyze([tmp_path], expand_blobs=True).leaves}

    blob = leaves["common_state/shard_0_1"]
    assert blob.nbytes == len(common)
    assert (blob.blob["len"] - blob.blob["arch"], blob.blob["k"]) == (64, "dict")
    assert {child["n"] for child in blob.blob["c"]} == {"iteration", "args"}
    assert leaves["rng_state/shard_0.0_1.2"].blob["k"] == "bytes"


def test_input_selection_and_errors(tmp_path: Path) -> None:
    trees = _write_per_rank(tmp_path)
    _global().write(tmp_path / "registry.bin")
    assert registry_tree.analyze([tmp_path]).warnings == ["ignored 1 global registry file(s) because per-rank registries were given"]

    partial = tmp_path / "partial"
    partial.mkdir()
    trees[1].write(partial / "registry.rank1.bin")
    assert registry_tree.analyze([partial]).warnings == ["per-rank registries cover 1 of 2 ranks"]

    trees[1].write(partial / "registry.1.bin")
    with pytest.raises(ValueError, match="both hold rank 1"):
        registry_tree.analyze([partial])
    with pytest.raises(ValueError, match="one global registry"):
        registry_tree.analyze([tmp_path / "registry.bin", tmp_path / "registry.bin"])

    truncated = tmp_path / "truncated" / "registry.bin"
    truncated.parent.mkdir()
    truncated.write_bytes((tmp_path / "registry.bin").read_bytes()[:100])
    with pytest.raises(ValueError, match="truncated"):
        registry_tree.analyze([truncated])

    empty = tmp_path / "empty"
    empty.mkdir()
    assert registry_tree.main([str(empty), "-o", str(tmp_path / "page.html")]) == 2
    assert not (tmp_path / "page.html").exists()


def test_page_payload(tmp_path: Path) -> None:
    _write_per_rank(tmp_path, b"common")
    report = registry_tree.analyze([tmp_path])
    doc, binary = page.encode_report(report, title="regs <2>")
    items = {item["f"]: item for item in doc["leaves"]}

    assert (doc["options"]["unique"], doc["options"]["values"]) == (True, True)
    assert (doc["inputLabel"], doc["hosts"], doc["ranksPerHost"]) == ("Registries", [_PEER], 2)
    assert doc["summary"]["uniqueBytes"] == 732 + len(b"common") + 100 + 101 + 8 + 8
    assert items["train.iteration"]["st"] == "64-bit scalar in per-rank registries"
    assert {"= 7", "= 8", "= 0.25"} <= set(doc["shapes"])
    assert "<title>regs &lt;2&gt;</title>" in page.render_html(doc, binary)


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_page_data_layer_counts_replicas_once_in_unique(tmp_path: Path) -> None:
    _write_per_rank(tmp_path / "regs", b"common")
    report = registry_tree.analyze([tmp_path / "regs"])
    doc, binary = page.encode_report(report, title="t")
    html_text = page.render_html(doc, binary)
    blocks = dict(re.findall(r'<script id="([a-z-]+)" type="application/octet-stream">([^<]*)</script>', html_text))
    (tmp_path / "page.js").write_text(re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)[-1], encoding="utf-8")
    (tmp_path / "doc.b64").write_text(blocks["doc-data"], encoding="ascii")
    (tmp_path / "bin.b64").write_text(blocks["bin-data"], encoding="ascii")
    (tmp_path / "driver.js").write_text(_NODE_DRIVER, encoding="utf-8")
    result = subprocess.run([_NODE, str(tmp_path / "driver.js"), str(tmp_path)], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)

    rank0 = 532 + len(b"common") + 100 + 16
    rank1 = 212 + 101 + 16
    assert values["rankTotals"] == [rank0, rank1]
    assert (values["rootAll"], values["rootUnique"]) == (rank0 + rank1, doc["summary"]["uniqueBytes"])
    assert values["iteration"] == ["= 7", "= 8", "2 distinct values across ranks"]
    assert json.loads(gzip.decompress(base64.b64decode(blocks["doc-data"])))["notes"] == report.notes


def test_script_runs_without_torch_or_numpy(tmp_path: Path) -> None:
    _write_per_rank(tmp_path / "regs", _torch_blob({"iteration": 7}, padding=0))
    output = tmp_path / "page.html"
    blocker = (
        "import runpy, sys; sys.modules.update(torch=None, numpy=None); "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    command = [sys.executable, "-I", "-c", blocker, str(_SCRIPT_PATH), str(tmp_path / "regs"), "--expand-blobs", "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
