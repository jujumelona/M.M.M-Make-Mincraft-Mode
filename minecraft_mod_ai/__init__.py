"""M.M.M Make Mincraft Mode: a deterministic Fabric mod builder."""

from .api import ChatReply, ModAISession, supported_minecraft_versions
from .importer import (
    ExistingProjectImportError,
    ExistingProjectReport,
    inspect_existing_project_archive,
)
from .pipeline import MinecraftModPipeline, PipelineResult
from .planner import (
    HeuristicPlanner,
    LocalTransformersPlanner,
    OpenAICompatiblePlanner,
)
from .spec import ArenaSpec, BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal

__all__ = [
    "ArenaSpec",
    "BossSpec",
    "ChatReply",
    "ContentSpec",
    "ExistingProjectImportError",
    "ExistingProjectReport",
    "HeuristicPlanner",
    "LocalTransformersPlanner",
    "ModAISession",
    "MinecraftModPipeline",
    "ModSpec",
    "PipelineResult",
    "PlatformLock",
    "Proposal",
    "OpenAICompatiblePlanner",
    "inspect_existing_project_archive",
    "supported_minecraft_versions",
]

__version__ = "0.1.0"
