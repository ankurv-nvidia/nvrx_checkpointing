# ruff: noqa: ANN401, C901, PLC0415, SLF001
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Runtime integration for an unmodified Megatron-LM checkout.

The supported Megatron commit is the public parent of the historical
``nvrx_checkpointing`` patch.  This module relocates that patch's Python
integration points here: argument registration, checkpoint dispatch, warmup,
and the distributed-optimizer dtype-key compatibility required to load old
checkpoints.

``install`` only replaces Python objects in the current process.  It never
writes to the Megatron-LM source tree.
"""

from __future__ import annotations

import runpy
import sys
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

SUPPORTED_MEGATRON_COMMIT = "354b5d2d06ebf2c529688db69ce5773932226381"

_NATIVE_PROVIDER = "native"
_REFERENCE_PROVIDER = "nvrx"
_SUPPORTED_PROVIDERS = (_NATIVE_PROVIDER, _REFERENCE_PROVIDER)
_INSTALL_MARKER = "_nvrx_checkpointing_integration_installed"
_NATIVE_ASYNC_MARKER = "_nvrx_checkpointing_native_async_installed"
_NATIVE_LOAD_MARKER = "_nvrx_checkpointing_native_load_installed"


def add_arguments(parser: Any) -> Any:
    """Add the checkpoint-provider selector to Megatron's parser."""
    if any("--ckpt-provider" in action.option_strings for action in parser._actions):
        return parser
    parser.add_argument(
        "--ckpt-provider",
        choices=_SUPPORTED_PROVIDERS,
        default=_NATIVE_PROVIDER,
        help="Checkpoint provider to use (native nvrx_checkpointing writer or Megatron's nvrx reference writer).",
    )
    return parser


def compose_extra_args_provider(provider: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    """Compose another Megatron argument provider with :func:`add_arguments`."""

    def combined(parser: Any) -> Any:
        if provider is not None:
            parser = provider(parser)
        return add_arguments(parser)

    return combined


def _dtype_param_grad_key(param_dtype: Any, grad_dtype: Any) -> str:
    """Return the dtype key used by checkpoints from the historical patched commit."""
    return f"dtype_param_{param_dtype}_grad_{grad_dtype}".replace(".", ":")


def _is_dtype_pair(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2  # noqa: PLR2004


def _stringify_dtype_keys(mapping: dict[Any, Any]) -> dict[Any, Any]:
    return {_dtype_param_grad_key(*key) if _is_dtype_pair(key) else key: value for key, value in mapping.items()}


def _dtype_key_lookup(optimizer: Any, torch_module: Any) -> dict[str, tuple[Any, Any]]:
    """Build string-to-tuple aliases for every dtype pair relevant to an optimizer."""
    lookup: dict[str, tuple[Any, Any]] = {}
    for buffer in getattr(optimizer, "buffers", ()):
        pairs = [(buffer.param_dtype, buffer.grad_dtype)]
        if getattr(buffer, "params", None):
            pairs.append((buffer.params[0].dtype, buffer.grad_dtype))
        pairs.append((torch_module.uint8, buffer.grad_dtype))
        for pair in pairs:
            lookup[_dtype_param_grad_key(*pair)] = pair
    return lookup


def _tupleize_dtype_keys(mapping: dict[Any, Any], lookup: dict[str, tuple[Any, Any]]) -> dict[Any, Any]:
    return {lookup.get(key, key) if isinstance(key, str) else key: value for key, value in mapping.items()}


def _patch_distributed_optimizer() -> None:
    """Preserve the patched commit's optimizer key format without editing Megatron."""
    from megatron.core.optimizer import distrib_optimizer as distrib_module  # type: ignore[import-not-found]

    optimizer_cls = distrib_module.DistributedOptimizer
    if getattr(optimizer_cls, _INSTALL_MARKER, False):
        return

    original_build_range_map = optimizer_cls.__dict__["_build_gbuf_range_map"].__func__
    original_init = optimizer_cls.__init__
    original_split_state_dict = optimizer_cls.split_state_dict_if_needed

    @classmethod
    @wraps(original_build_range_map)
    def build_gbuf_range_map(cls: type, param_and_grad_buffer: Any) -> dict[Any, Any]:
        return _stringify_dtype_keys(original_build_range_map(cls, param_and_grad_buffer))

    @wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        for attr in ("per_bucket_numel", "per_bucket_numel_unpadded"):
            if hasattr(self, attr):
                setattr(self, attr, [_stringify_dtype_keys(entry) for entry in getattr(self, attr)])

    @wraps(original_split_state_dict)
    def split_state_dict_if_needed(self: Any, state_dict: dict[Any, Any]) -> None:
        """Let upstream's FP8 compatibility helper operate on the retained string keys."""
        lookup = _dtype_key_lookup(self, distrib_module.torch)
        saved_gbuf_ranges = self.gbuf_ranges
        self.gbuf_ranges = [_tupleize_dtype_keys(entry, lookup) for entry in saved_gbuf_ranges]
        try:
            for key, value in list(state_dict.items()):
                if key != "buckets_coalesced" and isinstance(value, dict):
                    state_dict[key] = _tupleize_dtype_keys(value, lookup)
            original_split_state_dict(self, state_dict)
        finally:
            for key, value in list(state_dict.items()):
                if key != "buckets_coalesced" and isinstance(value, dict):
                    state_dict[key] = _stringify_dtype_keys(value)
            self.gbuf_ranges = saved_gbuf_ranges

    optimizer_cls._build_gbuf_range_map = build_gbuf_range_map
    optimizer_cls.__init__ = init
    optimizer_cls.split_state_dict_if_needed = split_state_dict_if_needed
    setattr(optimizer_cls, _INSTALL_MARKER, True)


def _patch_argument_registration() -> None:
    """Register the ``--ckpt-provider`` argument on Megatron's parser.

    The native provider keeps both ``--async-save`` and the configured ``async_strategy`` intact.
    Its async lifecycle is redirected separately to the repo-owned implementation, so the argument
    is never used to resolve an NVRx or MCore checkpoint implementation for native saves.
    """
    from megatron.training import arguments as arguments_module  # type: ignore[import-not-found]

    if getattr(arguments_module, _INSTALL_MARKER, False):
        return
    original_parse_args = arguments_module.parse_args

    @wraps(original_parse_args)
    def parse_args(extra_args_provider: Callable[[Any], Any] | None = None, ignore_unknown_args: bool = False) -> Any:
        return original_parse_args(compose_extra_args_provider(extra_args_provider), ignore_unknown_args)

    arguments_module.parse_args = parse_args
    setattr(arguments_module, _INSTALL_MARKER, True)


def _patch_native_async_lifecycle() -> None:
    """Redirect native-provider async hooks without selecting an external strategy."""
    from megatron.training import async_utils as async_utils_module  # type: ignore[import-not-found]
    from megatron.training import checkpointing as checkpointing_module  # type: ignore[import-not-found]
    from megatron.training import initialize as initialize_module  # type: ignore[import-not-found]
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(async_utils_module, _NATIVE_ASYNC_MARKER, False):
        return

    from . import native_async

    original_init = async_utils_module.init_persistent_async_worker
    original_schedule = async_utils_module.schedule_async_save
    original_finalize = async_utils_module.maybe_finalize_async_save
    original_is_empty = async_utils_module.is_empty_async_queue
    original_reset = async_utils_module.reset_persistent_async_worker

    def is_native() -> bool:
        return getattr(training_module.get_args(), "ckpt_provider", None) == _NATIVE_PROVIDER

    @wraps(original_init)
    def init_persistent_async_worker(rank: int, mp_mode: str = "spawn") -> Any:
        if is_native():
            return native_async.init_persistent_async_worker(rank, mp_mode)
        return original_init(rank, mp_mode)

    @wraps(original_schedule)
    def schedule_async_save(request: Any) -> Any:
        if is_native():
            return native_async.schedule_async_save(request)
        return original_schedule(request)

    @wraps(original_finalize)
    def maybe_finalize_async_save(blocking: bool = False, terminate: bool = False) -> Any:
        if is_native():
            return native_async.maybe_finalize_async_save(blocking, terminate)
        return original_finalize(blocking, terminate)

    @wraps(original_is_empty)
    def is_empty_async_queue() -> bool:
        if is_native():
            return native_async.is_empty_async_queue()
        return original_is_empty()

    @wraps(original_reset)
    def reset_persistent_async_worker(async_strategy: str) -> Any:
        if is_native():
            return native_async.reset_persistent_async_worker(async_strategy)
        return original_reset(async_strategy)

    async_utils_module.init_persistent_async_worker = init_persistent_async_worker
    async_utils_module.schedule_async_save = schedule_async_save
    async_utils_module.maybe_finalize_async_save = maybe_finalize_async_save
    async_utils_module.is_empty_async_queue = is_empty_async_queue
    async_utils_module.reset_persistent_async_worker = reset_persistent_async_worker

    # These modules imported the hooks by name before install() ran, so replace
    # their local bindings as well as the defining module's attributes.
    initialize_module.init_persistent_async_worker = init_persistent_async_worker
    training_module.maybe_finalize_async_save = maybe_finalize_async_save
    checkpointing_module.schedule_async_save = schedule_async_save
    checkpointing_module.is_empty_async_queue = is_empty_async_queue

    try:
        from megatron.training import inprocess_restart as restart_module  # type: ignore[import-not-found]
    except ImportError:
        restart_module = None
    if restart_module is not None:
        restart_module.reset_persistent_async_worker = reset_persistent_async_worker

    setattr(async_utils_module, _NATIVE_ASYNC_MARKER, True)


def _patch_native_load_reader() -> None:
    """Bypass external async-strategy reader resolution for native loads."""
    from megatron.core.dist_checkpointing.strategies import torch as torch_strategy_module  # type: ignore[import-not-found]
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(torch_strategy_module, _NATIVE_LOAD_MARKER, False):
        return

    from . import native_async

    original_get_filesystem_reader = torch_strategy_module._get_filesystem_reader

    @wraps(original_get_filesystem_reader)
    def get_filesystem_reader(
        checkpoint_dir: str | Path,
        cache_metadata: bool = False,
        async_strategy: str = "mcore",
    ) -> Any:
        if getattr(training_module.get_args(), "ckpt_provider", None) == _NATIVE_PROVIDER:
            return native_async.get_filesystem_reader(
                checkpoint_dir,
                cache_metadata=cache_metadata,
            )
        return original_get_filesystem_reader(
            checkpoint_dir,
            cache_metadata=cache_metadata,
            async_strategy=async_strategy,
        )

    torch_strategy_module._get_filesystem_reader = get_filesystem_reader
    setattr(torch_strategy_module, _NATIVE_LOAD_MARKER, True)


def _patch_training() -> None:
    """Install provider dispatch and pre-training warmup in Megatron's training module."""
    from megatron.training import training as training_module  # type: ignore[import-not-found]

    if getattr(training_module, _INSTALL_MARKER, False):
        return

    original_save_checkpoint = training_module.save_checkpoint
    original_save_checkpoint_and_time = training_module.save_checkpoint_and_time
    original_train = training_module.train
    fp8_gc_collect: Callable[[], Any] | None = None

    @wraps(original_save_checkpoint)
    def save_checkpoint(
        iteration: int,
        model: Any,
        optimizer: Any,
        opt_param_scheduler: Any,
        num_floating_point_operations_so_far: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal fp8_gc_collect
        megatron_args = training_module.get_args()
        if getattr(megatron_args, "ckpt_provider", None) != _NATIVE_PROVIDER:
            result = original_save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                *args,
                **kwargs,
            )
        else:
            from . import tools as native_tools

            result = native_tools.save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                train_data_iterator=kwargs.get("train_data_iterator"),
            )
        if fp8_gc_collect is not None:
            fp8_gc_collect()
        return result

    @wraps(original_save_checkpoint_and_time)
    def save_checkpoint_and_time(*args: Any, **kwargs: Any) -> Any:
        nonlocal fp8_gc_collect
        provider = getattr(training_module.get_args(), "ckpt_provider", None)
        if provider not in _SUPPORTED_PROVIDERS:
            return original_save_checkpoint_and_time(*args, **kwargs)
        real_gc_collect = training_module.gc.collect

        def no_gc() -> None:
            return None

        # The historical patch moved FP8 cleanup before the checkpoint timer
        # stops.  Let the save wrapper perform it there and suppress upstream's
        # later call, preserving both memory behavior and timing baselines.
        if getattr(training_module.get_args(), "fp8", False):
            fp8_gc_collect = real_gc_collect
            training_module.gc.collect = no_gc
        try:
            return original_save_checkpoint_and_time(*args, **kwargs)
        finally:
            training_module.gc.collect = real_gc_collect
            fp8_gc_collect = None

    @wraps(original_train)
    def train(*args: Any, **kwargs: Any) -> Any:
        provider = getattr(training_module.get_args(), "ckpt_provider", "")
        if _NATIVE_PROVIDER in provider:
            from . import tools as native_tools

            native_tools.warmup()
        return original_train(*args, **kwargs)

    training_module.save_checkpoint = save_checkpoint
    training_module.save_checkpoint_and_time = save_checkpoint_and_time
    training_module.train = train
    setattr(training_module, _INSTALL_MARKER, True)


def install() -> None:
    """Install the supported Megatron integration into the current Python process."""
    _patch_argument_registration()
    _patch_distributed_optimizer()
    _patch_native_async_lifecycle()
    _patch_native_load_reader()
    _patch_training()


def main(argv: list[str] | None = None) -> None:
    """Run a Megatron entrypoint after installing the integration in this process."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m nvrx_checkpointing.megatron_integration <megatron-entrypoint.py> [args ...]",
        )

    entrypoint = Path(args[0]).resolve()
    if not entrypoint.is_file():
        raise SystemExit(f"Megatron entrypoint not found: {entrypoint}")

    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *args[1:]]
    install()
    runpy.run_path(str(entrypoint), run_name="__main__")


__all__ = [
    "SUPPORTED_MEGATRON_COMMIT",
    "add_arguments",
    "compose_extra_args_provider",
    "install",
    "main",
]


if __name__ == "__main__":
    main()
