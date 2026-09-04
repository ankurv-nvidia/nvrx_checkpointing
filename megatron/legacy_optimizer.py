# ruff: noqa: ANN401
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Load-only compatibility for private-fork optimizer dtype keys.

The old fork stringified ``(param_dtype, grad_dtype)`` mapping keys. Public
Megatron keeps the tuples. Native saves now follow public Megatron and this
module supplies aliases only after a normal load of an old checkpoint fails.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _is_dtype_pair(key: object) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 2  # noqa: PLR2004
        and all(str(dtype).startswith("torch.") for dtype in key)
    )


def _legacy_dtype_key(dtype_pair: tuple[Any, Any]) -> str:
    param_dtype, grad_dtype = dtype_pair
    return f"dtype_param_{param_dtype}_grad_{grad_dtype}".replace(".", ":")


def _legacy_fqn(key: str, aliases: Mapping[str, tuple[Any, Any]]) -> str:
    for legacy_key, dtype_pair in aliases.items():
        current = f".dtype_{dtype_pair}"
        if current in key:
            return key.replace(current, f".dtype_{legacy_key}")
    return key


def _collect_dtype_aliases(value: Any, aliases: dict[str, tuple[Any, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_dtype_pair(key):
                aliases[_legacy_dtype_key(key)] = key
            _collect_dtype_aliases(child, aliases)
    elif isinstance(value, list):
        for child in value:
            _collect_dtype_aliases(child, aliases)


def _legacy_request(value: Any, aliases: Mapping[str, tuple[Any, Any]]) -> Any:
    if isinstance(value, dict):
        return {
            _legacy_dtype_key(key) if _is_dtype_pair(key) else key: _legacy_request(child, aliases)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_legacy_request(child, aliases) for child in value]
    if isinstance(value, tuple):
        return tuple(_legacy_request(child, aliases) for child in value)
    key = getattr(value, "key", None)
    if isinstance(key, str):
        legacy_key = _legacy_fqn(key, aliases)
        if legacy_key != key:
            value = copy.copy(value)
            value.key = legacy_key
    return value


def adapt_legacy_optimizer_request(
    sharded_state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Return an old-format optimizer request without mutating Megatron state."""
    optimizer = sharded_state_dict.get("optimizer")
    aliases: dict[str, tuple[Any, Any]] = {}
    _collect_dtype_aliases(optimizer, aliases)
    if not aliases:
        return sharded_state_dict, aliases
    request = dict(sharded_state_dict)
    request["optimizer"] = _legacy_request(optimizer, aliases)
    return request, aliases


def _restore(value: Any, aliases: Mapping[str, tuple[Any, Any]]) -> Any:
    if isinstance(value, dict):
        return {aliases.get(key, key): _restore(child, aliases) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore(child, aliases) for child in value]
    if isinstance(value, tuple):
        return tuple(_restore(child, aliases) for child in value)
    return value


def _restore_state_dict(state_dict: Any, aliases: Mapping[str, tuple[Any, Any]]) -> Any:
    if not isinstance(state_dict, dict) or "optimizer" not in state_dict:
        return state_dict
    result = dict(state_dict)
    result["optimizer"] = _restore(state_dict["optimizer"], aliases)
    return result


def restore_optimizer_dtype_keys(state_dict: Any, aliases: Mapping[str, tuple[Any, Any]]) -> Any:
    """Restore public-Megatron tuple keys in a successfully loaded result."""
    if isinstance(state_dict, tuple) and state_dict and isinstance(state_dict[0], dict):
        return (_restore_state_dict(state_dict[0], aliases), *state_dict[1:])
    return _restore_state_dict(state_dict, aliases)


__all__ = ["adapt_legacy_optimizer_request", "restore_optimizer_dtype_keys"]
