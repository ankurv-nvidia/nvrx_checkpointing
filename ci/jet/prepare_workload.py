#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Prepare one JET workload in its existing Slurm allocation."""

from __future__ import annotations

import logging
import sys

import runtime


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = runtime.parse_args(sys.argv[1:])
    runtime.prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
