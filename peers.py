# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Resolve rank-ordered checkpoint peers from Slurm and NVRX metadata."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrderedPeers:
    """Checkpoint hosts ordered by launcher group rank.

    ``peers[i]`` hosts the contiguous global-rank range beginning at
    ``i * ranks_per_peer``. In particular, ``peers[0]`` hosts global rank 0.
    """

    peers: tuple[str, ...]
    ranks_per_peer: int

    def __post_init__(self) -> None:
        if not self.peers:
            raise ValueError("ordered peers cannot be empty")
        if any(not peer for peer in self.peers) or len(set(self.peers)) != len(self.peers):
            raise ValueError(f"ordered peers contain empty or duplicate hosts: {self.peers}")
        if self.ranks_per_peer <= 0:
            raise ValueError("ranks_per_peer must be positive")

    def checkpoint_args(self) -> dict[str, list[str] | int]:
        """Convert at the boundary to the extension's keyword arguments."""
        return {"peers": list(self.peers), "ranks_per_peer": self.ranks_per_peer}


def _split_hostlist(nodelistspec: str) -> list[str]:
    """Split a Slurm hostlist on commas outside bracket expressions."""
    tokens, start, depth = [], 0, 0
    for index, char in enumerate(nodelistspec):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise RuntimeError(f"invalid Slurm nodelist {nodelistspec!r}: unmatched ']'")
        elif char == "," and depth == 0:
            tokens.append(nodelistspec[start:index])
            start = index + 1
    if depth:
        raise RuntimeError(f"invalid Slurm nodelist {nodelistspec!r}: unmatched '['")
    tokens.append(nodelistspec[start:])
    if any(not token for token in tokens):
        raise RuntimeError(f"invalid Slurm nodelist {nodelistspec!r}: empty host expression")
    return tokens


def _expand_host_token(token: str) -> list[str]:
    """Expand one host expression, including suffixes and repeated brackets."""
    match = re.search(r"\[([^\]]+)\]", token)
    if match is None:
        return [token]
    prefix, ranges, suffix = token[:match.start()], match.group(1), token[match.end():]
    values = []
    for part in ranges.split(","):
        part_match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if part_match is None:
            raise RuntimeError(f"invalid Slurm host range {part!r} in {token!r}")
        first, last = part_match.group(1), part_match.group(2) or part_match.group(1)
        if int(last) < int(first):
            raise RuntimeError(f"descending Slurm host range {part!r} in {token!r}")
        width = max(len(first), len(last))
        values.extend(str(value).zfill(width) for value in range(int(first), int(last) + 1))
    return [host for value in values for host in _expand_host_token(f"{prefix}{value}{suffix}")]


def _expand_slurm_nodelist_fallback(nodelistspec: str) -> list[str]:
    """Pure-Python hostlist expansion for containers without Slurm tools."""
    return [host for token in _split_hostlist(nodelistspec) for host in _expand_host_token(token)]


def _expand_slurm_nodelist(nodelistspec: str) -> list[str]:
    """Expand a Slurm nodelist, preferring Slurm's own parser when installed.

    Compute containers do not always contain ``scontrol``, so the fallback
    covers the numeric hostlist grammar used by the supported clusters.
    """
    if scontrol := shutil.which("scontrol"):
        proc = None
        try:
            proc = subprocess.run(
                [scontrol, "show", "hostnames", nodelistspec],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None and proc.returncode == 0 and (hosts := proc.stdout.split()):
            return hosts
    return _expand_slurm_nodelist_fallback(nodelistspec)


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from None


@dataclass(frozen=True)
class SlurmPeerResolver:
    """Resolve peers from a Slurm allocation that covers the whole world."""

    nodelist_spec: str
    tasks_per_node_spec: str
    task_count: int
    local_world_size: int

    @classmethod
    def from_environment(cls) -> SlurmPeerResolver | None:
        nodelist_spec = os.environ.get("SLURM_JOB_NODELIST")
        tasks_per_node_spec = os.environ.get("SLURM_TASKS_PER_NODE")
        if not nodelist_spec or not tasks_per_node_spec:
            return None
        return cls(
            nodelist_spec,
            tasks_per_node_spec,
            _env_int("SLURM_NTASKS"),
            _env_int("LOCAL_WORLD_SIZE"),
        )

    def _tasks_per_node(self) -> list[int]:
        result = []
        for token in self.tasks_per_node_spec.split(","):
            match = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", token.strip())
            if match is None:
                raise RuntimeError(f"invalid SLURM_TASKS_PER_NODE value {self.tasks_per_node_spec!r}")
            count, repeat = int(match.group(1)), int(match.group(2) or "1")
            if count <= 0 or repeat <= 0:
                raise RuntimeError(f"invalid SLURM_TASKS_PER_NODE value {self.tasks_per_node_spec!r}")
            result.extend([count] * repeat)
        return result

    def _uniform_ranks_per_peer(self, tasks_per_node: list[int]) -> int:
        if len(set(tasks_per_node)) != 1:
            raise RuntimeError(
                f"SLURM_TASKS_PER_NODE {self.tasks_per_node_spec!r} is non-uniform; "
                "native checkpoint peers require one rank count per node",
            )
        return tasks_per_node[0]

    def resolve(self, world_size: int) -> OrderedPeers | None:
        """Return peers only when this allocation describes the full world."""
        nodes = _expand_slurm_nodelist(self.nodelist_spec)
        tasks_per_node = self._tasks_per_node()
        if len(nodes) != len(tasks_per_node):
            raise RuntimeError(
                f"SLURM_JOB_NODELIST {self.nodelist_spec!r} expands to {len(nodes)} nodes, but "
                f"SLURM_TASKS_PER_NODE {self.tasks_per_node_spec!r} specifies {len(tasks_per_node)} entries.",
            )
        if self.task_count == world_size:
            if sum(tasks_per_node) != world_size:
                raise RuntimeError(
                    f"SLURM_TASKS_PER_NODE {self.tasks_per_node_spec!r} expands to {sum(tasks_per_node)} ranks, "
                    f"expected {world_size}",
                )
            return OrderedPeers(tuple(nodes), self._uniform_ranks_per_peer(tasks_per_node))
        one_task_per_node = self.task_count == len(nodes) and all(tasks == 1 for tasks in tasks_per_node)
        if one_task_per_node and self.local_world_size > 0 and len(nodes) * self.local_world_size == world_size:
            return OrderedPeers(tuple(nodes), self.local_world_size)
        return None


@dataclass(frozen=True)
class NvrxCyclePeerResolver:
    """Resolve peers from the current NVRX launcher cycle."""

    path: Path
    expected_cycle: int
    local_world_size: int
    group_rank: int | None
    hostname: str

    @classmethod
    def from_environment(cls) -> NvrxCyclePeerResolver | None:
        path = os.environ.get("NVRX_CURRENT_CYCLE_INFO")
        if not path:
            return None
        group_rank = _env_int("GROUP_RANK") if "GROUP_RANK" in os.environ else None
        return cls(
            Path(path),
            _env_int("TORCHELASTIC_RESTART_COUNT"),
            _env_int("LOCAL_WORLD_SIZE"),
            group_rank,
            socket.gethostname(),
        )

    @staticmethod
    def _expand_rank_ranges(ranges: str) -> list[int]:
        result = []
        for part in ranges.split(","):
            token = part.strip()
            if not token:
                continue
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
            if match is None:
                raise RuntimeError(f"invalid launcher rank range {token!r}")
            start, end = int(match.group(1)), int(match.group(2) or match.group(1))
            if end < start:
                raise RuntimeError(f"descending launcher rank range {token!r}")
            result.extend(range(start, end + 1))
        return result

    def _read(self, timeout_s: float = 60) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with self.path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):  # noqa: PERF203
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"launcher cycle-info file {self.path} did not become readable") from None
                time.sleep(1)
            else:
                if not isinstance(data, dict):
                    raise TypeError(f"launcher cycle-info file {self.path} does not contain a JSON object")
                cycle = data.get("cycle_number")
                if not isinstance(cycle, int):
                    raise TypeError(f"launcher cycle-info file {self.path} does not contain an integer cycle_number")
                if cycle == self.expected_cycle:
                    return data
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"launcher cycle-info file {self.path} remained at cycle {cycle}; "
                        f"expected cycle {self.expected_cycle}",
                    )
                time.sleep(1)

    def _parse(self, data: dict, world_size: int) -> OrderedPeers:
        active_nodes, active_ranks = data.get("active_nodes"), data.get("active_ranks")
        if not isinstance(active_nodes, str) or not isinstance(active_ranks, str):
            raise TypeError(
                f"launcher cycle-info file {self.path} must contain string active_nodes and active_ranks",
            )
        nodes = _expand_slurm_nodelist(active_nodes)
        ranks = self._expand_rank_ranges(active_ranks)
        if not nodes or len(nodes) != len(ranks):
            raise RuntimeError(
                f"launcher cycle-info file {self.path} holds {len(nodes)} active nodes but {len(ranks)} active ranks",
            )
        node_by_rank = dict(zip(ranks, nodes, strict=True))
        if len(node_by_rank) != len(nodes) or sorted(node_by_rank) != list(range(len(nodes))):
            raise RuntimeError(f"launcher cycle-info file {self.path} has invalid active ranks: {ranks}")
        peers = tuple(node_by_rank[rank] for rank in range(len(nodes)))
        if self.local_world_size <= 0 or len(peers) * self.local_world_size != world_size:
            raise RuntimeError(
                f"launcher cycle-info lists {len(peers)} active nodes with {self.local_world_size} ranks each, "
                f"which does not match world size {world_size}",
            )
        return OrderedPeers(peers, self.local_world_size)

    def resolve(self, rank: int, world_size: int) -> OrderedPeers:
        ordered_peers = self._parse(self._read(), world_size)
        expected_group_rank = rank // ordered_peers.ranks_per_peer
        group_rank = expected_group_rank if self.group_rank is None else self.group_rank
        if group_rank != expected_group_rank:
            raise RuntimeError(
                f"launcher GROUP_RANK {group_rank} does not match global rank {rank} "
                f"with {ordered_peers.ranks_per_peer} ranks per node",
            )
        if not 0 <= group_rank < len(ordered_peers.peers):
            raise RuntimeError(
                f"launcher group rank {group_rank} is outside its {len(ordered_peers.peers)} active nodes",
            )
        if ordered_peers.peers[group_rank].split(".")[0] != self.hostname.split(".")[0]:
            raise RuntimeError(
                f"launcher cycle-info maps group rank {group_rank} to {ordered_peers.peers[group_rank]!r} "
                f"but this process runs on {self.hostname!r}",
            )
        return ordered_peers


def detect_environment_peers(rank: int, world_size: int) -> OrderedPeers | None:
    """Resolve scheduler/launcher peers, or return ``None`` for a local run."""
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid distributed rank/world size: rank={rank}, world_size={world_size}")
    slurm = SlurmPeerResolver.from_environment()
    if slurm and (ordered_peers := slurm.resolve(world_size)):
        return ordered_peers
    cycle = NvrxCyclePeerResolver.from_environment()
    if cycle:
        return cycle.resolve(rank, world_size)
    if slurm or "SLURM_JOB_NODELIST" in os.environ:
        raise RuntimeError(
            "The Slurm allocation does not describe the whole world and NVRX_CURRENT_CYCLE_INFO is not set; "
            "job-array runs must pass --ft-cycle-info-dir to ft_launcher",
        )
    return None
