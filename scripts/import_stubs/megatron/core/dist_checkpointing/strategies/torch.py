from dataclasses import dataclass

from torch.distributed.checkpoint import Metadata, SavePlan


@dataclass
class MCoreMetadata(Metadata):
    """Metadata with mcore specific data."""

    # holds data related to flattened_range
    # TODO: remove when flattened_range is properly removed
    mcore_data: dict[str, dict[str, any]] | None = None  # Mcore related data about each tensor

@dataclass(frozen=True)
class MCoreSavePlan(SavePlan):
    """SavePlan with MCore specific data."""

    mcore_data: dict[str, dict[str, any]] | None = None  # Mcore related data about each tensor

