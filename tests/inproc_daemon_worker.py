# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: INP001, T201
"""Training-process driver for native daemon lifetime integration tests."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
from pathlib import Path

ROOT = None
RETAINED_ROOTS = []
SIGNALS = []


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def _children() -> list[int]:
    children = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            children.update(int(pid) for pid in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return sorted(children)


def _check_training_signal() -> None:
    signal.raise_signal(signal.SIGTERM)
    if SIGNALS != [signal.SIGTERM]:
        raise RuntimeError("training signal handler did not receive SIGTERM")


def _destroy(*, spindown: bool) -> None:
    global ROOT  # noqa: PLW0603

    if spindown:
        import checkpoint  # noqa: PLC0415

        checkpoint.svc_spindown()
    ROOT = None
    RETAINED_ROOTS.clear()
    gc.collect()


def _command(command: dict, args: argparse.Namespace) -> bool:
    global ROOT  # noqa: PLW0603

    import checkpoint  # noqa: PLC0415

    action = command["action"]
    if action == "close":
        ROOT.close()
    elif action == "new_root":
        RETAINED_ROOTS.append(ROOT)
        ROOT = checkpoint.model_state.create_root(args.session, args.rank, args.world, ["localhost"], args.world)
    elif action == "close_first":
        RETAINED_ROOTS[0].close()
    elif action in {"destroy", "spindown_destroy"}:
        _destroy(spindown=action == "spindown_destroy")
    elif action == "stderr":
        print("TRAINING_STDERR_IS_USABLE", file=sys.stderr, flush=True)
    elif action == "signal":
        _check_training_signal()
    elif action in {"save", "save_exit"}:
        import torch  # noqa: PLC0415

        ROOT.register_state_dict({"weight": torch.arange(8, dtype=torch.float32)})
        ROOT.save_async(1, Path(command["path"]))
        if action == "save_exit":
            _emit("exiting", children=_children())
            return True
        ROOT.wait_for_async_completion()
    elif action == "exit":
        _emit("exiting", children=_children())
        return True
    else:
        raise ValueError(f"Unknown training command: {action}")
    _emit(action, children=_children())
    return False


def main() -> None:
    global ROOT  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--session", type=int, default=7301)
    parser.add_argument("--ping-only", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda number, _frame: SIGNALS.append(number))

    import checkpoint  # noqa: PLC0415

    if args.ping_only:
        checkpoint.ping()
    else:
        ROOT = checkpoint.model_state.create_root(args.session, args.rank, args.world, ["localhost"], args.world)
    _emit("ready", pid=os.getpid(), children=_children())

    for line in sys.stdin:
        if _command(json.loads(line), args):
            return


if __name__ == "__main__":
    main()
