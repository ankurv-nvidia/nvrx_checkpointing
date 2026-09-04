# ruff: noqa: ANN401, SLF001
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Optional provider-selection arguments for comparison launchers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

NATIVE_PROVIDER = "native"
REFERENCE_PROVIDER = "nvrx"
SUPPORTED_PROVIDERS = (NATIVE_PROVIDER, REFERENCE_PROVIDER)


def add_arguments(parser: Any) -> Any:
    """Add the comparison-only checkpoint-provider selector."""
    if any("--ckpt-provider" in action.option_strings for action in parser._actions):
        return parser
    parser.add_argument(
        "--ckpt-provider",
        choices=SUPPORTED_PROVIDERS,
        default=NATIVE_PROVIDER,
        help="Checkpoint provider to use for this comparison run.",
    )
    return parser


def compose_extra_args_provider(provider: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    """Compose an existing Megatron argument provider with the selector."""

    def combined(parser: Any) -> Any:
        if provider is not None:
            parser = provider(parser)
        return add_arguments(parser)

    return combined


def selected_provider(argv: Sequence[str] | None = None) -> str:
    """Read only ``--ckpt-provider`` without importing or patching Megatron."""
    args = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(args):
        if arg.startswith("--ckpt-provider="):
            provider = arg.partition("=")[2]
            break
        if arg == "--ckpt-provider" and index + 1 < len(args):
            provider = args[index + 1]
            break
    else:
        return NATIVE_PROVIDER
    if provider not in SUPPORTED_PROVIDERS:
        choices = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"unknown checkpoint provider {provider!r}; choose {choices}")
    return provider


__all__ = [
    "NATIVE_PROVIDER",
    "REFERENCE_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "add_arguments",
    "compose_extra_args_provider",
    "selected_provider",
]
