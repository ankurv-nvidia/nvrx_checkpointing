import os
import re
import sys
from pathlib import Path

import torch

from . import checkpoint, mapping


def _expand_slurm_nodelist(nodelistspec: str) -> list[str]:
    """Expand a SLURM nodelist specification into a list of hostnames.

    Handles formats like node1, node[1-3], node[001,003,005-007],
    and comma-separated groups like node[1-3],other[4-5].
    """

    def expand_range(prefix: str, suffix: str, part: str) -> list[str]:
        return (
            [f"{prefix}{str(n).zfill(len(s))}{suffix}" for s, e in [part.split("-", 1)] for n in range(int(s), int(e) + 1)]
            if "-" in part
            else [f"{prefix}{part}{suffix}"]
        )

    def expand_token(token: str) -> list[str]:
        return [
            h
            for m in [re.match(r"(.*?)\[([^\]]+)\](.*)", token)]
            if m
            for prefix, ranges, suffix in [m.groups()]
            for part in ranges.split(",")
            for h in expand_range(prefix, suffix, part)
        ] or [token]

    return [h for token in re.findall(r"[^,\[]+(?:\[[^\]]+\])?", nodelistspec) for h in expand_token(token)]


def _expand_tasks_per_node(tasks_per_node_spec: str) -> list[int]:
    """Expand a SLURM_TASKS_PER_NODE specification into a per-node task count list.

    Handles the SLURM format where consecutive equal counts are compressed, e.g.
    "2(x3),1" -> [2, 2, 2, 1].
    """
    result = []
    for token in tasks_per_node_spec.split(","):
        m = re.match(r"^(\d+)\(x(\d+)\)$", token.strip())
        if m:
            count, repeat = int(m.group(1)), int(m.group(2))
            result.extend([count] * repeat)
        else:
            result.append(int(token.strip()))
    return result


def get_slurm_nodelist(rank: int, world_size: int) -> list[str]:
    if "SLURM_JOB_NODELIST" not in os.environ:
        raise RuntimeError("SLURM_JOB_NODELIST environment variable not found")
    if "SLURM_TASKS_PER_NODE" not in os.environ:
        raise RuntimeError("SLURM_TASKS_PER_NODE environment variable not found")
    if "SLURM_NTASKS" not in os.environ:
        raise RuntimeError("SLURM_NTASKS environment variable not found")
    if int(os.environ["SLURM_NTASKS"]) != world_size:
        raise RuntimeError(f"SLURM_NTASKS {os.environ['SLURM_NTASKS']} does not match world size {world_size}")
    nodelistspec = os.environ["SLURM_JOB_NODELIST"]
    ntasks_per_node_spec = os.environ.get("SLURM_TASKS_PER_NODE")
    expanded_nodelist = _expand_slurm_nodelist(nodelistspec)
    ntasks_per_node = _expand_tasks_per_node(ntasks_per_node_spec)
    if len(expanded_nodelist) != len(ntasks_per_node):
        raise RuntimeError(
            f"SLURM_JOB_NODELIST {nodelistspec!r} expands to "
            f"{len(expanded_nodelist)} nodes, but SLURM_TASKS_PER_NODE {ntasks_per_node_spec!r} "
            f"specifies {len(ntasks_per_node)} entries.",
        )

    rank_expanded_nodelist = [node for node, ntasks in zip(expanded_nodelist, ntasks_per_node, strict=False) for _ in range(ntasks)]
    sys.stdout.write(
        f"SLURM_JOB_NODELIST: {nodelistspec}\n"
        f"SLURM_TASKS_PER_NODE: {ntasks_per_node_spec}\n"
        f"Expanded SLURM_JOB_NODELIST: {', '.join(expanded_nodelist)}\n"
        f"Expanded SLURM_TASKS_PER_NODE: {', '.join(map(str, ntasks_per_node))}\n"
        f"Rank-expanded SLURM_JOB_NODELIST: {', '.join(rank_expanded_nodelist)}\n"
        f"World size = {world_size}\n"
        f"Rank = {rank}\n",
    )
    if len(rank_expanded_nodelist) != world_size:
        raise RuntimeError(f"SLURM_JOB_NODELIST {nodelistspec!r} expands to {len(rank_expanded_nodelist)} ranks, expected {world_size}.")

    return {"peers": expanded_nodelist, "ranks_per_peer": ntasks_per_node[0]}


def detect_peers(rank: int, world_size: int) -> dict[str, list[str] | int]:
    if "SLURM_JOB_NODELIST" in os.environ:
        return get_slurm_nodelist(rank, world_size)
    if world_size == 1:
        return {"peers": ["localhost"], "ranks_per_peer": 1}
    if torch.distributed.get_backend() == "nccl" and world_size <= torch.cuda.device_count():
        return {"peers": ["localhost"], "ranks_per_peer": world_size}
    raise RuntimeError("Unable to detect peers for distributed run")


def register_state_dict(sharded_state_dict: mapping.ShardedStateDict) -> checkpoint.model_state:
    # TODO_CACHE_STATEDICT
    if not hasattr(register_state_dict, "cache"):
        register_state_dict.cache: checkpoint.model_state | None = None  # type: ignore[attr-defined]
    if register_state_dict.cache is not None:
        return register_state_dict.cache
    # Re-register every save: handlers cache `tnsr.data_ptr()` at registration time
    # (torch_bindings.cpp:Convert) and never refresh it. If any tensor's storage has
    # been reallocated since the previous save, the cached pointer is stale and the
    # D2H copy reads freed memory. Most visible on small `bucket_idx_2` tail shards
    # whose backing buffers Megatron may rebuild between optimizer steps.
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    root = checkpoint.model_state.create_root(rank, world_size, **detect_peers(rank, world_size))
    root.register_state_dict(sharded_state_dict)
    root.finalize_registry()
    # TODO_CACHE_STATEDICT
    # register_state_dict.cache = root
    return root


def save(sharded_state_dict: mapping.ShardedStateDict, iteration: int, checkpoint_dir: Path) -> None:
    root = register_state_dict(sharded_state_dict)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    root.save_async(iteration, Path(checkpoint_dir))
    root.wait_for_deferred_save()


__all__ = ["register_state_dict", "save"]
