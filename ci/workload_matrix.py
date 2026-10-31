# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001
"""Typed access to the platforms in ci/jobs.json and the clusters that run them."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Architecture = Literal["x86_64", "aarch64"]


JOBS_FILE = Path(__file__).with_name("jobs.json")
PLATFORMS = json.loads(JOBS_FILE.read_text())["platforms"]
CLUSTERS_VARIABLE = "JET_CLUSTERS"
CLUSTERS_EXAMPLE = "ci/jet/clusters.example.json"
CLUSTER_FIELDS = frozenset({"config", "launcher", "cache", "segment", "custom_config"})
RUNTIME_CACHE_MOUNT = "/nvrx-ci-cache"


@dataclass(frozen=True)
class JetPlatform:
    """Hardware and workload limits for one platform baseline namespace."""

    name: str
    profile: str
    architecture: Architecture
    gpus_per_node: int
    max_nodes: int | None = None
    max_timeout_seconds: int | None = None
    use_regular_time_budget: bool = False
    distributed_timeout_minutes: int | None = None
    distributed_timeout_min_nodes: int | None = None
    compute_capability: str = ""


@dataclass(frozen=True)
class JetCluster:
    """The JET cluster config that one platform's workloads run on.

    The launcher name also names the config's executor environment, the cache
    is the cluster directory that keeps the platform's prepared runtime, and
    the segment is the most nodes Slurm places within one NVLink domain.
    """

    config: str
    launcher: str | None = None
    cache: str | None = None
    segment: int | None = None
    custom_config: dict = field(default_factory=dict)

    @property
    def runtime_cache_dir(self) -> str | None:
        return RUNTIME_CACHE_MOUNT if self.cache else None


def jet_cluster(name: str) -> JetCluster:
    """Return one platform's cluster from the repository variable that maps platforms to clusters.

    The variable holds a JSON object keyed by platform name, shaped like the
    checked-in example, so a platform moves clusters without a code change.
    """
    text = os.environ.get(CLUSTERS_VARIABLE, "").strip()
    if not text:
        raise RuntimeError(f"{CLUSTERS_VARIABLE} is unset; set it to a JSON object shaped like {CLUSTERS_EXAMPLE}")
    try:
        clusters = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{CLUSTERS_VARIABLE} is not valid JSON: {error}") from error
    if not isinstance(clusters, dict):
        raise TypeError(f"{CLUSTERS_VARIABLE} must be a JSON object keyed by platform name")
    entry = clusters.get(name)
    if entry is None:
        listed = ", ".join(sorted(clusters)) or "nothing"
        raise RuntimeError(f"{CLUSTERS_VARIABLE} has no entry for platform {name!r}; it lists {listed}")
    if not isinstance(entry, dict):
        raise TypeError(f"{CLUSTERS_VARIABLE} entry {name!r} must be a JSON object")
    if unknown := sorted(set(entry) - CLUSTER_FIELDS):
        raise RuntimeError(f"{CLUSTERS_VARIABLE} entry {name!r} has unknown fields: {', '.join(unknown)}")
    if not entry.get("config"):
        raise RuntimeError(f"{CLUSTERS_VARIABLE} entry {name!r} names no JET config")
    if entry.get("cache") and not entry.get("launcher"):
        raise RuntimeError(f"{CLUSTERS_VARIABLE} entry {name!r} sets a cache without the launcher that mounts it")
    segment = entry.get("segment")
    if segment is not None and (not isinstance(segment, int) or isinstance(segment, bool) or segment < 1):
        raise TypeError(f"{CLUSTERS_VARIABLE} entry {name!r} segment must be a positive node count")
    return JetCluster(
        config=entry["config"],
        launcher=entry.get("launcher"),
        cache=entry.get("cache"),
        segment=segment,
        custom_config=entry.get("custom_config", {}),
    )


def platform_profile(name: str) -> dict:
    """Return the ci/jobs.json entry whose workloads a platform runs."""
    platform = next(platform for platform in PLATFORMS if platform["name"] == name)
    profile_name = platform.get("workload_profile", name)
    return next(candidate for candidate in PLATFORMS if candidate["name"] == profile_name)


def jet_platform(name: str) -> JetPlatform:
    """Return JET platform metadata by baseline namespace."""
    platform = next(platform for platform in PLATFORMS if platform["name"] == name)
    metadata = platform["jet"]
    _nodes, gpus_per_node = (int(value) for value in platform["gpus"].split("x"))
    return JetPlatform(
        name=name,
        profile=platform.get("workload_profile", name),
        architecture=platform["architecture"],
        gpus_per_node=gpus_per_node,
        max_nodes=metadata.get("max_nodes"),
        max_timeout_seconds=(
            metadata["max_timeout_minutes"] * 60 if metadata.get("max_timeout_minutes") else None
        ),
        use_regular_time_budget=metadata.get("use_regular_time_budget", False),
        distributed_timeout_minutes=metadata.get("distributed_timeout_minutes"),
        distributed_timeout_min_nodes=metadata.get("distributed_timeout_min_nodes"),
        compute_capability=metadata["compute_capability"],
    )


JET_PLATFORMS = tuple(jet_platform(platform["name"]) for platform in PLATFORMS if "jet" in platform)
