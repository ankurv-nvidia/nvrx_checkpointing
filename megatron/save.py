# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Native checkpoint save integration for Megatron-LM."""

from __future__ import annotations

import inspect
import io
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from megatron.core import mpu  # type: ignore[import-not-found]
from megatron.core.dist_checkpointing.state_dict_utils import save_preprocess  # type: ignore[import-not-found]
from megatron.core.msc_utils import maybe_msc  # type: ignore[import-not-found]
from megatron.core.rerun_state_machine import get_rerun_state_machine  # type: ignore[import-not-found]
from megatron.core.utils import unwrap_model  # type: ignore[import-not-found]
from megatron.training.checkpointing import (  # type: ignore[import-not-found]
    _build_sharded_state_dict_metadata,
    generate_state_dict,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_rng_state,
    maybe_save_dataloader_state,
)
from megatron.training.global_vars import get_args  # type: ignore[import-not-found]
from megatron.training.utils import print_rank_0  # type: ignore[import-not-found]

from .. import checkpoint  # noqa: TID252
from ..ckpt_diag import diagnose_checkpoint_server_failure  # noqa: TID252
from ..peers import OrderedPeers, detect_environment_peers  # noqa: TID252
from ..telemetry import managed_span  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

_DCP_COMMON_STATE_KEY = "common_state/shard_0_1"
_CONTENT_METADATA_KEY = "content_metadata"
_RERUN_STATE_KEY = "rerun_state_machine"
_RNG_STATE_KEY = "rng_state"
_SERIALIZED_SHARDED_OBJECT_ATTR = "_nvrx_serialized_blob"
_COMMON_STATE_PADDING = 512
_COMMON_STATE_ALIGNMENT = 1024
_root = None
_common_state_capacity: int | None = None
_sharded_object_capacities: dict[str, int] = {}


@dataclass
class SaveCheckpointArgs:
    """State passed from Megatron's save wrapper to the native provider."""

    iteration: int
    model: object
    optimizer: object | None
    opt_param_scheduler: object | None
    num_floating_point_operations_so_far: int
    train_data_iterator: object | None = None
    tp_group: object | None = None
    pp_group: object | None = None
    dp_cp_group: object | None = None
    dp_group: object | None = None
    dp_gtp_remat_group: object | None = None
    cp_group: object | None = None
    rng_state_key_prefix: str = ""


@dataclass
class _PendingSave:
    future: Future[None]
    finalize_fn: Callable[[], None]


_pending: _PendingSave | None = None
_executor: ThreadPoolExecutor | None = None
_LOAD_CONTEXT: dict[str, object] | None = None


class _SingletonProcessGroup:
    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def size() -> int:
        return 1


_SINGLETON_PROCESS_GROUP = _SingletonProcessGroup()


def _wrap_load_checkpoint(load_checkpoint: Callable) -> Callable:
    """Expose Megatron's role-specific groups while its DCP load is running."""

    load_signature = inspect.signature(load_checkpoint)

    @wraps(load_checkpoint)
    def wrapped(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        global _LOAD_CONTEXT  # noqa: PLW0603
        previous = _LOAD_CONTEXT
        call = load_signature.bind(*args, **kwargs)
        call.apply_defaults()
        values = call.arguments
        _LOAD_CONTEXT = {name: values.get(name) for name in ("tp_group", "pp_group", "dp_cp_group", "dp_group", "rng_state_key_prefix")}
        try:
            return load_checkpoint(*args, **kwargs)
        finally:
            _LOAD_CONTEXT = previous

    return wrapped


def _checkpoint_span(name: str, iteration: int) -> AbstractContextManager[object]:
    """Create a checkpoint span with the iteration and native snapshot ID."""
    return managed_span(
        "checkpoint",
        name,
        is_goodput_span=True,
        **{"megatron.iteration": iteration, "nvrx.snapshot_id": iteration},
    )


def detect_peers(rank: int, world_size: int) -> OrderedPeers:
    """Resolve the ordered checkpoint peers without distributed communication."""
    ordered_peers = detect_environment_peers(rank, world_size)
    if ordered_peers is not None:
        return ordered_peers
    if world_size == 1:
        return OrderedPeers(("localhost",), 1)
    if torch.distributed.get_backend() == "nccl" and world_size <= torch.cuda.device_count():
        return OrderedPeers(("localhost",), world_size)
    raise RuntimeError("Unable to detect peers for distributed run")


def _snapshot_blob_capacity(serialized_size: int) -> int:
    padded_size = serialized_size + _COMMON_STATE_PADDING
    return ((padded_size + _COMMON_STATE_ALIGNMENT - 1) // _COMMON_STATE_ALIGNMENT) * _COMMON_STATE_ALIGNMENT


def _prepare_rng_sharded_object_blobs(value: object) -> None:
    """Serialize variable-size RNG ShardedObjects into a stable slot."""
    if isinstance(value, dict):
        for child in value.values():
            _prepare_rng_sharded_object_blobs(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _prepare_rng_sharded_object_blobs(child)
        return
    if type(value).__name__ != "ShardedObject" or not hasattr(value, "data") or not hasattr(value, "unique_key"):
        return

    # TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
    replica_id = getattr(value, "replica_id", 0)
    if (isinstance(replica_id, int) and replica_id != 0) or (
        not isinstance(replica_id, int) and any(component != 0 for component in replica_id)
    ):
        return

    buffer = io.BytesIO()
    torch.save([value.data], buffer)
    serialized = buffer.getvalue()
    unique_key = str(value.unique_key)
    capacity = _sharded_object_capacities.setdefault(unique_key, _snapshot_blob_capacity(len(serialized)))
    if len(serialized) > capacity:
        raise RuntimeError(
            f"sharded object '{unique_key}' serialized to {len(serialized)} bytes, exceeding its registered {capacity}-byte capacity",
        )
    setattr(value, _SERIALIZED_SHARDED_OBJECT_ATTR, serialized.ljust(capacity, b"\0"))


def _prepare_registration_state_dict(state_dict: dict, iteration: int) -> dict:
    """Convert Megatron's generated state into the representation native saves."""
    args = state_dict.get("args")
    needs_rerun_state = _RERUN_STATE_KEY not in state_dict
    needs_rng_state = _RNG_STATE_KEY not in state_dict and args is not None and not getattr(args, "no_save_rng", False)
    if needs_rerun_state or needs_rng_state:
        state_dict = dict(state_dict)
    if needs_rerun_state:
        state_dict[_RERUN_STATE_KEY] = get_rerun_state_machine().state_dict(
            data_iterator=None,
            ckpt_format="torch_dist",
            force=True,
        )
    if needs_rng_state:
        if iteration == 0:
            if not getattr(args, "no_load_rng", False):
                warnings.warn(
                    "load registration omitted rng_state without no_load_rng; synthesizing placeholder state",
                    RuntimeWarning,
                    stacklevel=2,
                )
            context = _LOAD_CONTEXT or {}
            state_dict[_RNG_STATE_KEY] = get_rng_state(
                "torch_dist",
                context.get("tp_group"),
                context.get("pp_group"),
                dp_cp_group=context.get("dp_cp_group") or _SINGLETON_PROCESS_GROUP,
                dp_group=context.get("dp_group") or _SINGLETON_PROCESS_GROUP,
                key_prefix=str(context.get("rng_state_key_prefix") or ""),
            )
        else:
            state_dict[_RNG_STATE_KEY] = get_rng_state(
                "torch_dist",
                mpu.get_tensor_model_parallel_group(),
                mpu.get_pipeline_model_parallel_group(),
            )
    sharded_part, common = save_preprocess(state_dict, validate_access_integrity=False)
    _prepare_rng_sharded_object_blobs(sharded_part.get(_RNG_STATE_KEY))
    snapshot_state = {}
    if torch.distributed.get_rank() == 0:
        with _checkpoint_span("nvrx.checkpoint.register_state_dict.serialize_common_state", iteration):
            global _common_state_capacity  # noqa: PLW0603
            buffer = io.BytesIO()
            # TODO_BG_SERIALIZE: Measure and move this GIL-held archive creation.  # noqa: FIX002, TD002, TD003, TD004
            torch.save([common], buffer)
            common_state = buffer.getvalue()
            if _common_state_capacity is None:
                _common_state_capacity = _snapshot_blob_capacity(len(common_state))
            if len(common_state) > _common_state_capacity:
                raise RuntimeError(
                    f"common state serialized to {len(common_state)} bytes, exceeding its "
                    f"registered {_common_state_capacity}-byte capacity",
                )
            snapshot_state[_DCP_COMMON_STATE_KEY] = common_state.ljust(_common_state_capacity, b"\0")
    prepared_state = dict(sharded_part)
    prepared_state.update(snapshot_state)
    return prepared_state


def _get_executor() -> ThreadPoolExecutor:
    """Return the single worker that waits for native durability."""
    global _executor  # noqa: PLW0603
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-ckpt-finalize")
    return _executor


def init_persistent_async_worker(rank: int, mp_mode: str = "spawn") -> None:
    """Satisfy Megatron's async-worker lifecycle hook; native owns no process worker."""
    del rank, mp_mode


def maybe_finalize_async_save(blocking: bool = False, terminate: bool = False) -> None:
    """Finalize the completed native save or wait for it when requested."""
    global _executor, _pending  # noqa: PLW0603
    request = _pending
    if request is not None and (blocking or request.future.done()):
        error = request.future.exception()
        if error is not None:
            raise RuntimeError("native async checkpoint failed during background execution") from error
        _pending = None
        try:
            request.finalize_fn()
        except Exception as error:
            raise RuntimeError("native async checkpoint failed during finalization") from error

    if terminate and _executor is not None:
        if _pending is not None:
            raise RuntimeError("cannot terminate native async worker with a pending request")
        _executor.shutdown(wait=True)
        _executor = None


def register_state_dict(state_dict: dict, iteration: int) -> checkpoint.model_state:
    """Register the current state dictionary on the process-global state tree."""
    global _root  # noqa: PLW0603
    if _root is None:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        ordered_peers = detect_peers(rank, world_size)
        _root = checkpoint.model_state.create_root(0, rank, world_size, **ordered_peers.checkpoint_args())
    _root.register_state_dict(_prepare_registration_state_dict(state_dict, iteration))
    return _root


# The writer and finalizer use the same fixed session.
_SESSION_ID = 0


def _timestamp() -> str:
    """Format checkpoint log timestamps like Megatron."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def _write_checkpoint_tracker(args, iteration: int) -> None:  # noqa: ANN001
    """Publish ``iteration`` as the latest completed checkpoint."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with maybe_msc.open(get_checkpoint_tracker_filename(args.save), "w") as f:
            f.write(str(iteration))


def _save_dataloader_state(save_args: SaveCheckpointArgs, args: object) -> None:
    """Save loader state with the process groups supported by Megatron."""
    groups = {
        "tp_group": save_args.tp_group,
        "pp_group": save_args.pp_group,
        "dp_group": save_args.dp_gtp_remat_group if save_args.dp_gtp_remat_group is not None else save_args.dp_group,
        "cp_group": save_args.cp_group,
    }
    parameters = inspect.signature(maybe_save_dataloader_state).parameters
    if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        groups = {name: group for name, group in groups.items() if name in parameters}
    maybe_save_dataloader_state(
        save_args.train_data_iterator,
        save_args.iteration,
        getattr(args, "dataloader_save", None),
        **groups,
    )


def save_checkpoint(save_args: SaveCheckpointArgs) -> None:
    """Save through native checkpointing and diagnose server failures."""
    started_at_ns = time.time_ns()
    with _checkpoint_span("nvrx.checkpoint.save", save_args.iteration):
        try:
            _save_checkpoint_impl(save_args, started_at_ns)
        except Exception as e:
            diagnose_checkpoint_server_failure(e)
            print_rank_0(f"Error saving checkpoint at iteration {save_args.iteration}: {e}")
            raise


def _save_checkpoint_impl(save_args: SaveCheckpointArgs, started_at_ns: int) -> None:  # noqa: PLR0915
    """Build the current Megatron state and submit one native checkpoint.

    The load path or first save creates the process-global state tree. Every save
    rescans the complete dictionary and refreshes the registered values and tensor pointers.

    Dataloader state is written as Megatron's per-iteration sidecar file. Logging
    hooks, fault tolerance, retention cleanup, and ModelOpt state are outside this
    provider's state contract.
    """
    global _pending  # noqa: PLW0603
    iteration = save_args.iteration
    with _checkpoint_span("nvrx.checkpoint.prepare", iteration):
        with _checkpoint_span("nvrx.checkpoint.wait_previous", iteration):
            wait_started_at = None
            if _pending is not None and not _pending.future.done():
                wait_started_at = time.monotonic()
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(  # noqa: T201
                        f"  [{_timestamp()}] WARNING: checkpoint save at iteration {iteration:7d} waiting for prev save ... ",
                        end="",
                        flush=True,
                    )
            maybe_finalize_async_save(blocking=True)
            if wait_started_at is not None:
                wait_milliseconds = (time.monotonic() - wait_started_at) * 1000
                print_rank_0(f"Took {wait_milliseconds:.3f} ms")

        with _checkpoint_span("megatron.checkpoint.save.state_dict", iteration):
            args = get_args()
            model = unwrap_model(save_args.model)
            metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=save_args.dp_cp_group)
            if args.use_distributed_optimizer:
                print_rank_0(
                    f"Storing distributed optimizer sharded state of type {metadata['distrib_optim_sharding_type']}",
                )
            rng_state = get_rng_state(
                "torch_dist",
                save_args.tp_group or mpu.get_tensor_model_parallel_group(),
                save_args.pp_group or mpu.get_pipeline_model_parallel_group(),
                dp_cp_group=save_args.dp_cp_group,
                dp_group=save_args.dp_group,
                key_prefix=save_args.rng_state_key_prefix,
            )
            rerun_state = get_rerun_state_machine().state_dict(
                data_iterator=save_args.train_data_iterator,
                ckpt_format="torch_dist",
            )
            state_dict = generate_state_dict(
                args,
                model,
                save_args.optimizer,
                save_args.opt_param_scheduler,
                rng_state,
                iteration=iteration,
                optim_sd_kwargs={"metadata": metadata},
                model_sd_kwargs={"metadata": metadata},
                rerun_state=rerun_state,
            )
            state_dict["num_floating_point_operations_so_far"] = save_args.num_floating_point_operations_so_far
            if torch.distributed.get_rank() == 0:
                state_dict[_CONTENT_METADATA_KEY] = {key: value for key, value in metadata.items() if key != "dp_cp_group"}

            # Native checkpoints always use the torch_dist format.
            print_rank_0(
                f"[{_timestamp()}] saving checkpoint at iteration {iteration:7d} to {args.save} in torch_dist format",
            )
            checkpoint_dir = Path(get_checkpoint_name(args.save, iteration, return_base_dir=True))

    with _checkpoint_span("megatron.checkpoint.save.io_write", iteration):
        # Loader state lands before snapshot submission, so the tracker never names an iteration without it.
        with _checkpoint_span("megatron.checkpoint.save.dataloader_state", iteration):
            _save_dataloader_state(save_args, args)
        with _checkpoint_span("nvrx.checkpoint.cpp.save_async", iteration):
            with _checkpoint_span("nvrx.checkpoint.register_state_dict", iteration):
                root = register_state_dict(state_dict, iteration)
            root.startedAt = started_at_ns
            root.save_async(iteration, checkpoint_dir)

        if not getattr(args, "async_save", False):
            # A deferred client send must run before the blocking server wait.
            with _checkpoint_span("nvrx.checkpoint.cpp.wait_for_async_completion", iteration):
                root.wait_for_async_completion()
            with _checkpoint_span("nvrx.checkpoint.enqueue_finalize", iteration):
                checkpoint.wait_for_ckpt_finalize(_SESSION_ID, iteration)
                _write_checkpoint_tracker(args, iteration)
                print_rank_0(
                    f"  [{_timestamp()}] successfully saved checkpoint from iteration {iteration:7d} to {args.save}",
                )
        else:

            def finalize_fn() -> None:
                _write_checkpoint_tracker(args, iteration)
                print_rank_0(
                    f"  [{_timestamp()}] successfully saved checkpoint from iteration {iteration:7d} to {args.save}",
                )

            with _checkpoint_span("nvrx.checkpoint.cpp.wait_for_async_completion", iteration):
                root.wait_for_async_completion()
            with _checkpoint_span("nvrx.checkpoint.enqueue_finalize", iteration):
                future = _get_executor().submit(checkpoint.wait_for_ckpt_finalize, _SESSION_ID, iteration)
                _pending = _PendingSave(future, finalize_fn)
                print_rank_0(
                    f"  [{_timestamp()}] scheduled an async checkpoint save at iteration {iteration:7d} to {args.save}",
                )


__all__ = [
    "SaveCheckpointArgs",
    "init_persistent_async_worker",
    "maybe_finalize_async_save",
    "register_state_dict",
    "save_checkpoint",
]
