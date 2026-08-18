# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, S101, SLF001

"""Unit tests for the ordered-peer adapter used by tools.py.

Covers the three launch shapes the native checkpoint writer runs under:

- plain srun with one Slurm task per rank (CI jobs, nano.sh),
- ft_launcher/torchrun inside a single allocation, one Slurm task per node,
- Slurm job arrays whose tasks join one ft_launcher rendezvous, where no
  single allocation covers the world and peers come from the launcher's
  cycle-info record (NVRX_CURRENT_CYCLE_INFO).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SLURM_ENV_VARS = (
    "SLURM_JOB_NODELIST",
    "SLURM_TASKS_PER_NODE",
    "SLURM_NTASKS",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "NVRX_CURRENT_CYCLE_INFO",
)


def _load_peers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the dependency-free peers adapter in isolation."""
    for name in _SLURM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("peers_test", REPO_ROOT / "peers.py")
    assert spec is not None
    assert spec.loader is not None
    peers = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, peers)
    spec.loader.exec_module(peers)
    # Most compute containers do not contain scontrol. Keep parser tests
    # deterministic; the scheduler-backed path has its own test below.
    monkeypatch.setattr(peers.shutil, "which", lambda _name: None)
    return peers


def test_expand_slurm_nodelist(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    assert peers._expand_slurm_nodelist("node1") == ["node1"]
    assert peers._expand_slurm_nodelist("node[1-3]") == ["node1", "node2", "node3"]
    assert peers._expand_slurm_nodelist("node[001,003,005-006]") == ["node001", "node003", "node005", "node006"]
    assert peers._expand_slurm_nodelist("node[1-2],other[4-5]") == ["node1", "node2", "other4", "other5"]
    assert peers._expand_slurm_nodelist("rack[1-2]node[01-02]-ib") == [
        "rack1node01-ib",
        "rack1node02-ib",
        "rack2node01-ib",
        "rack2node02-ib",
    ]


def test_expand_slurm_nodelist_prefers_scontrol(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    commands = []
    monkeypatch.setattr(peers.shutil, "which", lambda _name: "/usr/bin/scontrol")
    monkeypatch.setattr(
        peers.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0, stdout="scheduler-a\nscheduler-b\n"),
    )

    assert peers._expand_slurm_nodelist("scheduler[1-2]") == ["scheduler-a", "scheduler-b"]
    assert commands == [["/usr/bin/scontrol", "show", "hostnames", "scheduler[1-2]"]]


def test_expand_slurm_nodelist_falls_back_when_scontrol_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setattr(peers.shutil, "which", lambda _name: "/usr/bin/scontrol")
    monkeypatch.setattr(
        peers.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    assert peers._expand_slurm_nodelist("fallback[01-02]") == ["fallback01", "fallback02"]


def test_expand_rank_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    expand = peers.NvrxCyclePeerResolver._expand_rank_ranges
    assert expand("0-3") == [0, 1, 2, 3]
    assert expand("0-15,48-63,32-47,16-31") == (
        list(range(16)) + list(range(48, 64)) + list(range(32, 48)) + list(range(16, 32))
    )
    assert expand("5") == [5]


def test_expand_tasks_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    assert peers.SlurmPeerResolver("", "4", 0, 0)._tasks_per_node() == [4]
    assert peers.SlurmPeerResolver("", "2(x3),1", 0, 0)._tasks_per_node() == [2, 2, 2, 1]


def _get_slurm_peers(peers: ModuleType, world_size: int) -> object | None:
    resolver = peers.SlurmPeerResolver.from_environment()
    return resolver.resolve(world_size) if resolver else None


def test_get_slurm_peers_one_task_per_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "4(x4)")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert _get_slurm_peers(peers, 16) == peers.OrderedPeers(("node1", "node2", "node3", "node4"), 4)


def test_get_slurm_peers_launcher_one_task_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """ft_launcher steps run one Slurm task per node; LOCAL_WORLD_SIZE gives ranks per node."""
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1(x4)")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert _get_slurm_peers(peers, 16) == peers.OrderedPeers(("node1", "node2", "node3", "node4"), 4)


def test_get_slurm_peers_partial_allocation_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job-array task sees only its own nodes; that cannot describe a larger world."""
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-16]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1(x16)")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert _get_slurm_peers(peers, 256) is None


def test_get_slurm_peers_missing_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    assert _get_slurm_peers(peers, 16) is None
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    assert _get_slurm_peers(peers, 16) is None


def test_get_slurm_peers_rejects_inconsistent_tasks_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "4(x3)")
    monkeypatch.setenv("SLURM_NTASKS", "12")
    with pytest.raises(RuntimeError, match="SLURM_TASKS_PER_NODE"):
        _get_slurm_peers(peers, 12)


def test_get_slurm_peers_rejects_nonuniform_rank_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "2(x3),1")
    monkeypatch.setenv("SLURM_NTASKS", "7")
    with pytest.raises(RuntimeError, match="non-uniform"):
        _get_slurm_peers(peers, 7)


def _write_cycle_info(tmp_path: Path, nodes: list[str], ranks: list[int], cycle: int = 0) -> Path:
    """Write a cycle-info file the way the launcher's reporter formats it."""
    import json  # noqa: PLC0415

    def compress_nodes(addrs: list[str]) -> str:
        prefixes: dict[str, list[int]] = {}
        for addr in addrs:
            match = __import__("re").match(r"(.*?)(\d+)$", addr)
            prefixes.setdefault(match.group(1), []).append(int(match.group(2)))
        parts = []
        for prefix in sorted(prefixes):
            nums = prefixes[prefix]
            parts.append(f"{prefix}[{nums[0]:02d}-{nums[-1]:02d}]" if len(nums) > 1 else f"{prefix}{nums[0]:02d}")
        return ",".join(parts)

    def compress_ranks(values: list[int]) -> str:
        ranges, start, end = [], values[0], values[0]
        for value in values[1:]:
            if value == end + 1:
                end = value
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = value
        ranges.append(f"{start}-{end}" if start != end else str(start))
        return ",".join(ranges)

    path = tmp_path / f"cycle_info.100.0.{cycle}"
    path.write_text(json.dumps({
        "cycle_number": cycle,
        "active_nodes": compress_nodes(nodes),
        "active_ranks": compress_ranks(ranks),
        "standby_nodes": "",
    }))
    return path


def test_detect_peers_from_cycle_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A job-array task's allocation covers 16 of 64 nodes; the cycle file holds the world."""
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "nvl72d005-T[01-16]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1(x16)")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_RANK", "17")
    monkeypatch.setattr(peers.socket, "gethostname", lambda: "nvl72d148-T02")
    nodes = [f"nvl72d005-T{i:02d}" for i in range(1, 17)]
    nodes += [f"nvl72d136-T{i:02d}" for i in range(1, 17)]
    nodes += [f"nvl72d146-T{i:02d}" for i in range(1, 17)]
    nodes += [f"nvl72d148-T{i:02d}" for i in range(1, 17)]
    ranks = list(range(16)) + list(range(48, 64)) + list(range(32, 48)) + list(range(16, 32))
    monkeypatch.setenv("NVRX_CURRENT_CYCLE_INFO", str(_write_cycle_info(tmp_path, nodes, ranks)))

    result = peers.detect_environment_peers(17 * 4, 256)

    expected = [""] * 64
    for node, rank in zip(nodes, ranks):
        expected[rank] = node
    assert result == peers.OrderedPeers(tuple(expected), 4)
    assert result.peers[17] == "nvl72d148-T02"  # group rank 17 maps to this process's host
    assert result.checkpoint_args() == {"peers": expected, "ranks_per_peer": 4}


def test_detect_peers_waits_for_the_current_launcher_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    peers = _load_peers(monkeypatch)
    nodes = [f"node{i:02d}" for i in range(4)]
    stale = _write_cycle_info(tmp_path, nodes, list(range(4)), cycle=0)
    current = _write_cycle_info(tmp_path, nodes, list(range(4)), cycle=1)
    link = tmp_path / "cycle_info.100.current"
    link.symlink_to(stale.name)
    monkeypatch.setenv("NVRX_CURRENT_CYCLE_INFO", str(link))
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("GROUP_RANK", "0")
    monkeypatch.setattr(peers.socket, "gethostname", lambda: "node00")

    def publish_current(_seconds: float) -> None:
        link.unlink()
        link.symlink_to(current.name)

    monkeypatch.setattr(peers.time, "sleep", publish_current)

    assert peers.detect_environment_peers(0, 8) == peers.OrderedPeers(tuple(nodes), 2)


def test_detect_peers_cycle_info_requires_matching_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    nodes = [f"nvl72d005-T{i:02d}" for i in range(1, 17)]
    monkeypatch.setenv("NVRX_CURRENT_CYCLE_INFO", str(_write_cycle_info(tmp_path, nodes, list(range(16)))))
    with pytest.raises(RuntimeError, match="does not match world size 256"):
        peers.detect_environment_peers(0, 256)


def test_detect_peers_cycle_info_flags_rank_host_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_RANK", "3")
    monkeypatch.setattr(peers.socket, "gethostname", lambda: "elsewhere-T01")
    nodes = [f"nvl72d005-T{i:02d}" for i in range(1, 17)]
    monkeypatch.setenv("NVRX_CURRENT_CYCLE_INFO", str(_write_cycle_info(tmp_path, nodes, list(range(16)))))
    with pytest.raises(RuntimeError, match="group rank 3"):
        peers.detect_environment_peers(12, 64)


def test_detect_peers_cycle_info_flags_inconsistent_group_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_RANK", "3")
    nodes = [f"node{i}" for i in range(16)]
    monkeypatch.setenv("NVRX_CURRENT_CYCLE_INFO", str(_write_cycle_info(tmp_path, nodes, list(range(16)))))
    with pytest.raises(RuntimeError, match="does not match global rank"):
        peers.detect_environment_peers(8, 64)


def test_detect_peers_without_cycle_info_or_covering_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-16]")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1(x16)")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match="--ft-cycle-info-dir"):
        peers.detect_environment_peers(0, 256)


def test_detect_peers_rejects_invalid_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = _load_peers(monkeypatch)
    with pytest.raises(ValueError, match="rank=4, world_size=4"):
        peers.detect_environment_peers(4, 4)
