# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Native checkpoint save integration for Megatron-LM."""

from __future__ import annotations

import copy
import io
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span  # type: ignore[import-not-found]

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
)
from megatron.training.global_vars import get_args  # type: ignore[import-not-found]
from megatron.training.utils import print_rank_0  # type: ignore[import-not-found]

from .. import checkpoint  # noqa: TID252
from ..ckpt_diag import diagnose_checkpoint_server_failure  # noqa: TID252
from ..peers import OrderedPeers, detect_environment_peers  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

_DCP_COMMON_STATE_KEY = "common_state/shard_0_1"
_CONTENT_METADATA_KEY = "content_metadata"
_root = None


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
    rng_state_key_prefix: str = ""


@dataclass
class _PendingSave:
    future: Future[None]
    finalize_fn: Callable[[], None]


_pending: _PendingSave | None = None
_executor: ThreadPoolExecutor | None = None


def _checkpoint_span(name: str, iteration: int) -> AbstractContextManager[object]:
    """Create a checkpoint span with the iteration and native snapshot ID."""
    return _otel_managed_span(
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


def warmup() -> None:
    """Initialize native save resources before training begins."""
    _get_executor()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    ordered_peers = detect_peers(rank, world_size)
    checkpoint.model_state.warmup(
        rank,
        world_size,
        **ordered_peers.checkpoint_args(),
    )


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


def _save_checkpoint_impl(save_args: SaveCheckpointArgs, started_at_ns: int) -> None:  # noqa: C901, PLR0915
    """Build the current Megatron state and submit one native checkpoint.

    The first save registers the state tree. Later saves merge the newly built state
    dictionary into the cached Python dictionary before refreshing the registered nodes.

    Dataloader resume, logging hooks, fault tolerance, retention cleanup, and
    ModelOpt state are outside this provider's state contract.
    """
    global _pending, _root  # noqa: PLW0603
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
            # Native validates exact 1:1 refresh correspondence after the first traversal.
            sharded_part, common = save_preprocess(state_dict, validate_access_integrity=False)
            if "args" in common:
                # Saved arguments describe the synchronous reader used for native loads.
                saved_args = copy.copy(common["args"])
                saved_args.async_strategy = "mcore"
                common["args"] = saved_args
            rank = torch.distributed.get_rank()
            if rank == 0:
                # Rank 0 stores portable optimizer-sharding metadata without the process group.
                common[_CONTENT_METADATA_KEY] = {key: value for key, value in metadata.items() if key != "dp_cp_group"}

            # Native checkpoints always use the torch_dist format.
            print_rank_0(
                f"[{_timestamp()}] saving checkpoint at iteration {iteration:7d} to {args.save} in torch_dist format",
            )
            checkpoint_dir = Path(get_checkpoint_name(args.save, iteration, return_base_dir=True))

    with _checkpoint_span("megatron.checkpoint.save.io_write", iteration):
        with _checkpoint_span("nvrx.checkpoint.cpp.save_async", iteration):
            root = _root
            first_registration = root is None
            if root is None:
                world_size = torch.distributed.get_world_size()
                ordered_peers = detect_peers(rank, world_size)
                root = checkpoint.model_state.create_root(rank, world_size, **ordered_peers.checkpoint_args())
            root.startedAt = started_at_ns
            with _checkpoint_span("nvrx.checkpoint.serialize_common_state", iteration):
                snapshot_state = {}
                if rank == 0:
                    buffer = io.BytesIO()
                    # TODO_BG_SERIALIZE: Measure and move this GIL-held archive creation.  # noqa: FIX002, TD002, TD003, TD004
                    torch.save([common], buffer)
                    snapshot_state[_DCP_COMMON_STATE_KEY] = buffer.getvalue()
            refreshed_state = dict(sharded_part)
            refreshed_state.update(snapshot_state)
            root.register_state_dict(refreshed_state)
            if first_registration:
                root.finalize_registry()
                _root = root
            root.save_async(iteration, checkpoint_dir)

        if not getattr(args, "async_save", False):
            # A deferred client send must run before the blocking server wait.
            with _checkpoint_span("nvrx.checkpoint.cpp.wait_for_deferred_save", iteration):
                root.wait_for_deferred_save()
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

            with _checkpoint_span("nvrx.checkpoint.enqueue_finalize", iteration):
                future = _get_executor().submit(checkpoint.wait_for_ckpt_finalize, _SESSION_ID, iteration)
                _pending = _PendingSave(future, finalize_fn)
                print_rank_0(
                    f"  [{_timestamp()}] scheduled an async checkpoint save at iteration {iteration:7d} to {args.save}",
                )
            with _checkpoint_span("nvrx.checkpoint.cpp.wait_for_deferred_save", iteration):
                root.wait_for_deferred_save()


__all__ = [
    "SaveCheckpointArgs",
    "init_persistent_async_worker",
    "maybe_finalize_async_save",
    "save_checkpoint",
    "warmup",
]
