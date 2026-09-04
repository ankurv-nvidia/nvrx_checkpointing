# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint import Metadata, SavePlan


@dataclass
class MCoreMetadata(Metadata):
    """Metadata with mcore specific data."""

    mcore_data: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class MCoreSavePlan(SavePlan):
    """SavePlan with MCore specific data."""

    mcore_data: dict[str, dict[str, Any]] | None = None
