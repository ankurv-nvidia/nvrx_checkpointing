import io
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.serialization import _CONTENT_METADATA_KEY
from megatron.core.dist_checkpointing.state_dict_utils import save_preprocess
from megatron.core.dist_checkpointing.utils import extract_nonpersistent, extract_sharded_base
from megatron.core.msc_utils import maybe_msc
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import unwrap_model
from megatron.training.checkpointing import (
    _build_sharded_state_dict_metadata,
    generate_state_dict,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_rng_state,
)
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0

from . import checkpoint, mapping
from .ckpt_diag import diagnose_checkpoint_server_failure
from .peers import OrderedPeers, detect_environment_peers

_DCP_COMMON_STATE_KEY = "common_state/shard_0_1"


def detect_peers(rank: int, world_size: int) -> OrderedPeers:
    ordered_peers = detect_environment_peers(rank, world_size)
    if ordered_peers is not None:
        if cycle_info := os.environ.get("NVRX_CURRENT_CYCLE_INFO"):
            print_rank_0(
                f"[native_ckpt] ordered peers from launcher cycle info {cycle_info}: "
                f"{len(ordered_peers.peers)} nodes x {ordered_peers.ranks_per_peer} ranks per node",
            )
        return ordered_peers
    if world_size == 1:
        return OrderedPeers(("localhost",), 1)
    if torch.distributed.get_backend() == "nccl" and world_size <= torch.cuda.device_count():
        return OrderedPeers(("localhost",), world_size)
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
        ordered_peers = detect_peers(rank, world_size)
        root = checkpoint.model_state.create_root(rank, world_size, **ordered_peers.checkpoint_args())
        root.register_state_dict(sharded_state_dict)
        if rank == 0:
            register_state_dict.snapshot_state = {_DCP_COMMON_STATE_KEY: b""}
            root.register_state_dict(register_state_dict.snapshot_state)
        root.finalize_registry()
        register_state_dict.root = root
    return register_state_dict.root


register_state_dict.root = None
register_state_dict.snapshot_state = None


def _refresh_optimizer_state(model, optimizer, metadata: dict, common: dict) -> list:  # noqa: ANN001
    """Re-extract the optimizer's tensor and common state for a warm save.

    The bf16 Adam moments (exp_avg/exp_avg_sq) are transient get_unscaled_state().float() copies the
    optimizer never writes to again, so the root's once-registered handlers re-save the first-save
    values. Re-extracting the sharded tensors each warm save and re-pointing the registered handlers
    at them keeps the saved moments current — what regenerating the full state dict does for nvrx —
    while skipping the collective re-registration. Model/master-param ShardedTensors alias live
    storage, so refreshing them re-points to the same buffer (a no-op); refreshing all of them lets
    the native side assert a 1:1 node correspondence.

    The cached common optimizer dictionary also contains mutable scalars such as each parameter
    group's learning rate. Replace it from the same extraction so later checkpoints do not retain
    the first warm save's values.
    """
    model = unwrap_model(model)
    state_dict = {}
    for i in range(len(model)):
        key = "model" if len(model) == 1 else f"model{i}"
        state_dict[key] = model[i].sharded_state_dict(metadata=metadata)
    state_dict["optimizer"] = optimizer.sharded_state_dict(state_dict, metadata=metadata)
    sharded_part, fresh_common = save_preprocess(state_dict, validate_access_integrity=False)
    common["optimizer"] = fresh_common["optimizer"]
    return [leaf for leaf in nested_values(sharded_part) if isinstance(leaf, ShardedTensor)]


def warmup() -> None:
    """Warm up the pinned buffers used for D2H transfers in checkpointing to avoid first save penalty"""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ordered_peers = detect_peers(rank, world_size)
    checkpoint.model_state.warmup(rank, world_size, **ordered_peers.checkpoint_args())


def save(common_state: dict, iteration: int, checkpoint_dir: Path) -> None:
    """Queue common state and registered tensor shards for daemon-side DCP output."""
    if torch.distributed.get_rank() == 0:
        buffer = io.BytesIO()
        # Match Megatron's current ShardedObject representation: DCP stores a
        # torch.save archive containing a one-element list with the common dict.
        torch.save([common_state], buffer)
        if register_state_dict.snapshot_state is None:
            raise RuntimeError("common-state snapshot dictionary was not registered")
        register_state_dict.snapshot_state[_DCP_COMMON_STATE_KEY] = buffer.getvalue()
    register_state_dict.root.save_async(iteration, checkpoint_dir)
    register_state_dict.root.wait_for_deferred_save()


# The native writer registers every rank under session id 0 (cipc::SessionId{0}); the finalize
# wait targets that same session.
_SESSION_ID = 0


def _wait_for_ckpt_finalize(session_id: int, iteration: int) -> None:
    """Block until the node-local daemon reports the snapshot for ``iteration`` durable.

    Kept module-level (picklable) and self-contained: it runs in the async caller's worker process
    — a fork of the trainer — where it opens its own daemon connection and relies on no pre-fork
    state.
    """
    checkpoint.wait_for_ckpt_finalize(session_id, iteration)


def _timestamp() -> str:
    """Megatron stamps its checkpoint lines with this, so ours read the same in a job log."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # noqa: DTZ005


def _print_rank_0_inline(message: str) -> None:
    """Print and flush a rank-zero message fragment without ending its line."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, end="", flush=True)  # noqa: T201


def _parallel_ranks() -> str:
    """Megatron's `[ t x/y, p a/b ]` suffix, naming the writer's tensor and pipeline rank."""
    return (
        f"[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, "
        f"p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ]"
    )


def _write_checkpoint_tracker(args, iteration: int) -> None:  # noqa: ANN001
    """Advance the tracker file so the loader resolves ``iteration`` as the latest checkpoint."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with maybe_msc.open(get_checkpoint_tracker_filename(args.save), "w") as f:
            f.write(str(iteration))


def _finalize_checkpoint(args, iteration: int) -> None:  # noqa: ANN001
    """Chain a staged checkpoint's daemon-durability wait into the native async queue.

    Every save schedules a wait so the next ``save_checkpoint`` can join it before taking another
    snapshot. With ``--async-save`` the tracker update is also deferred until every rank's wait
    returns. The sync path retains its historical optimistic tracker update and barrier, while the
    queued wait propagates daemon errors and provides the next save's chain point.
    """
    from .native_async import AsyncRequest, schedule_async_save  # noqa: PLC0415

    async_save = getattr(args, "async_save", False)
    finalize_fns = []
    if async_save:

        def finalize_fn() -> None:
            _write_checkpoint_tracker(args, iteration)
            print_rank_0(
                f"  [{_timestamp()}] successfully saved "
                f"checkpoint from iteration {iteration:7d} to {args.save} {_parallel_ranks()}",
            )

        finalize_fns.append(finalize_fn)

    schedule_async_save(
        AsyncRequest(
            async_fn=_wait_for_ckpt_finalize,
            async_fn_args=(_SESSION_ID, iteration),
            finalize_fns=finalize_fns,
        ),
    )

    if async_save:
        print_rank_0(
            f"  [{_timestamp()}] scheduled "
            f"an async checkpoint save at iteration {iteration:7d} to {args.save}",
        )
        return

    _write_checkpoint_tracker(args, iteration)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(
        f"  [{_timestamp()}] successfully saved "
        f"checkpoint from iteration {iteration:7d} to {args.save} {_parallel_ranks()}",
    )


def save_checkpoint(  # noqa: PLR0913
    iteration: int,
    model,  # noqa: ANN001
    optimizer,  # noqa: ANN001
    opt_param_scheduler,  # noqa: ANN001
    num_floating_point_operations_so_far: int,
    train_data_iterator=None,  # noqa: ANN001
) -> None:
    if register_state_dict.root is not None:
        register_state_dict.root.startedAt = time.time_ns()
    try:
        # The preceding call returned after transferring its snapshot to the daemon. Join its
        # durability request before reading any live model or optimizer state for this snapshot.
        from .native_async import is_previous_save_in_progress, wait_for_previous_save  # noqa: PLC0415

        serialize_start = None
        if is_previous_save_in_progress():
            serialize_start = time.monotonic()
            _print_rank_0_inline(
                f"  [{_timestamp()}] WARNING: checkpoint save at iteration {iteration:7d} "
                "waiting for prev save ... ",
            )
        wait_for_previous_save()
        if serialize_start is not None:
            serialized_milliseconds = (time.monotonic() - serialize_start) * 1000
            print_rank_0(f"Took {serialized_milliseconds:.3f} ms")
        _save_checkpoint_impl(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            train_data_iterator=train_data_iterator,
        )
    except Exception as e:
        diagnose_checkpoint_server_failure(e)
        print_rank_0(f"Error saving checkpoint at iteration {iteration}: {e}")
        raise


def _save_checkpoint_impl(  # noqa: PLR0913
    iteration: int,
    model,  # noqa: ANN001
    optimizer,  # noqa: ANN001
    opt_param_scheduler,  # noqa: ANN001
    num_floating_point_operations_so_far: int,
    train_data_iterator=None,  # noqa: ANN001
) -> None:
    """Save a synchronous torch_dist checkpoint through the native writer.

    The first save builds the full sharded state dict and registers it; later saves reuse
    that registration — model weights and master params are read live through the frozen
    handles, the optimizer ShardedTensors are re-extracted and refreshed each save (their
    bf16 moment storage is a transient copy that would otherwise go stale), and the plain
    common scalars are refreshed — so the collective re-registration is skipped.

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
            optim_sd_kwargs={"metadata": metadata},
            model_sd_kwargs={"metadata": metadata},
            rerun_state=rerun_state,
        )
        state_dict["num_floating_point_operations_so_far"] = num_floating_point_operations_so_far
        # validate_access_integrity off: nvrx runs its own cross-rank validation on the same dict.
        sharded_part, common = save_preprocess(state_dict, validate_access_integrity=False)
        register_state_dict(sharded_part)
        if torch.distributed.get_rank() == 0:
            # content_metadata lets load pick the matching optimizer sharding type; only rank 0
            # serializes common state. dp_cp_group is dropped because it is not serializable.
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
        # Re-point the registered ShardedTensor handlers at freshly-extracted tensors so the root
        # re-reads current data instead of the transient bf16 moment copies frozen at the first save.
        fresh = _refresh_optimizer_state(model, optimizer, _build_sharded_state_dict_metadata(args), common)
        register_state_dict.root.refresh_sharded_tensors(fresh)

    # The writer emits torch_dist whatever --ckpt-format asks for, so name what it writes.
    print_rank_0(
        f"[{_timestamp()}] saving checkpoint "
        f"at iteration {iteration:7d} to {args.save} in torch_dist format",
    )
    save(common, iteration, Path(get_checkpoint_name(args.save, iteration, return_base_dir=True)))
    _finalize_checkpoint(args, iteration)


save_checkpoint.common = None


__all__ = ["register_state_dict", "save", "save_checkpoint", "warmup"]
