"""M.M.M Make Mincraft Mode: a deterministic Fabric mod builder."""

from .importer import (
    ExistingProjectImportError,
    ExistingProjectReport,
    inspect_existing_project_archive,
)
from .pipeline import MinecraftModPipeline, PipelineResult
from .planner import HeuristicPlanner, LocalTransformersPlanner
from .spec import ArenaSpec, BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal

__all__ = [
    "ArenaSpec",
    "BossSpec",
    "ContentSpec",
    "ExistingProjectImportError",
    "ExistingProjectReport",
    "HeuristicPlanner",
    "LocalTransformersPlanner",
    "MinecraftModPipeline",
    "ModSpec",
    "PipelineResult",
    "PlatformLock",
    "Proposal",
    "inspect_existing_project_archive",
]

__version__ = "0.1.0"
