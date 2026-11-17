# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: INP001
"""Finalize one synchronized JET workload from its single teardown task."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from run_workload import coordinated_finalize


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("workload_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    workload_argv = args.workload_args
    if workload_argv[:1] == ["--"]:
        workload_argv = workload_argv[1:]
    return coordinated_finalize(workload_argv, Path(args.assets_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
