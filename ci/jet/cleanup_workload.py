#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Remove one finalized JET run after verifying its artifact archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("workload_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    workload_argv = args.workload_args
    if workload_argv[:1] == ["--"]:
        workload_argv = workload_argv[1:]
    run_workload.cleanup_finalized_run(workload_argv, args.assets_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
