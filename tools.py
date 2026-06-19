import os
import re
import sys
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.serialization import _CONTENT_METADATA_KEY
from megatron.core.dist_checkpointing.state_dict_utils import save_preprocess
from megatron.core.dist_checkpointing.utils import extract_nonpersistent, extract_sharded_base
from megatron.core.msc_utils import open_file
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training.checkpointing import (
    _build_sharded_state_dict_metadata,
    generate_state_dict,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_rng_state,
)
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0, unwrap_model

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

    rank_expanded_nodelist = [node for node, ntasks in zip(expanded_nodelist, ntasks_per_node) for _ in range(ntasks)]  # noqa: B905
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
    """Register the preprocessed sharded state dict with a native root, once.

    The root is cached and reused for every save: its handlers read each tensor's storage
    live, so only the sharded structure has to stay constant. Registration is collective,
    so all ranks must reach the first (registering) save together — the training loop
    drives saves in lockstep, so they do.
    """
    if register_state_dict.root is None:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        root = checkpoint.model_state.create_root(rank, world_size, **detect_peers(rank, world_size))
        root.register_state_dict(sharded_state_dict)
        root.finalize_registry()
        register_state_dict.root = root
    return register_state_dict.root


register_state_dict.root = None


def warmup() -> None:
    """Warm up the pinned buffers used for D2H transfers in checkpointing to avoid first save penalty"""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    checkpoint.model_state.warmup(rank, world_size, **detect_peers(rank, world_size))


def save(common_state: dict, iteration: int, checkpoint_dir: Path) -> None:
    """Write common.pt and the registered sharded shards for one checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        torch.save(common_state, checkpoint_dir / "common.pt")
        # metadata.json marker so the standard torch_dist loader recognizes these shards.
        save_config(CheckpointingConfig(sharded_backend="torch_dist"), str(checkpoint_dir))
    register_state_dict.root.save_async(iteration, checkpoint_dir)
    register_state_dict.root.wait_for_deferred_save()


def save_checkpoint(
    iteration: int,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far: int,
    train_data_iterator=None,
) -> None:
    """Save a synchronous torch_dist checkpoint through the native writer.

    The first save builds the full sharded state dict and registers it; later saves reuse
    that registration — model/optimizer tensors are read live, the frozen sharded objects
    (RNG, distributed-optimizer state) are re-emitted from their registration-time bytes,
    and only the plain common scalars are refreshed — so generate_state_dict and
    re-registration are skipped entirely.

    TODO: omits work the generic save_checkpoint does, to add back as needed: dataloader /
    Energon resume, one_logger/wandb hooks, fault tolerance, save_retain cleanup, ModelOpt.
    """
    args = get_args()
    common = save_checkpoint.common
    if common is None:
        model = unwrap_model(model)
        metadata = _build_sharded_state_dict_metadata(args)
        rng_state = get_rng_state("torch_dist", mpu.get_tensor_model_parallel_group(), mpu.get_pipeline_model_parallel_group())
        rerun_state = get_rerun_state_machine().state_dict(data_iterator=train_data_iterator, ckpt_format=args.ckpt_format)
        state_dict = generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=dict(metadata=metadata),
            model_sd_kwargs=dict(metadata=metadata),
            rerun_state=rerun_state,
        )
        state_dict["num_floating_point_operations_so_far"] = num_floating_point_operations_so_far
        # validate_access_integrity off: nvrx runs its own cross-rank validation on the same dict.
        sharded_part, common = save_preprocess(state_dict, validate_access_integrity=False)
        register_state_dict(sharded_part)
        if torch.distributed.get_rank() == 0:
            # content_metadata lets load pick the matching optimizer sharding type; only rank 0
            # writes common.pt. dp_cp_group is dropped because it is not serializable.
            common[_CONTENT_METADATA_KEY] = {k: v for k, v in metadata.items() if k != "dp_cp_group"}
        save_checkpoint.common = common
    else:
        common["iteration"] = iteration
        common["num_floating_point_operations_so_far"] = num_floating_point_operations_so_far
        if "opt_param_scheduler" in common:
            common["opt_param_scheduler"] = opt_param_scheduler.state_dict()
        if "rerun_state_machine" in common:
            # The rerun ShardedObject is frozen in the root, but its plain fields (e.g.
            # current_iteration) advance; re-extract the common portion to keep them current.
            rerun_state = get_rerun_state_machine().state_dict(data_iterator=train_data_iterator, ckpt_format=args.ckpt_format)
            _, persistent = extract_nonpersistent(rerun_state)
            _, common["rerun_state_machine"] = extract_sharded_base(persistent)

    print_rank_0(f"saving checkpoint at iteration {iteration:7d} to {args.save} in torch_dist format")
    save(common, iteration, Path(get_checkpoint_name(args.save, iteration, return_base_dir=True)))

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open_file(get_checkpoint_tracker_filename(args.save), "w") as f:
            f.write(str(iteration))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(f"  successfully saved checkpoint from iteration {iteration:7d} to {args.save}")


save_checkpoint.common = None


__all__ = ["register_state_dict", "save", "save_checkpoint", "warmup"]
