"""M.M.M Make Mincraft Mode: role-routed multimodal Fabric production tools."""

from .api import ChatReply, ModAISession, supported_minecraft_versions
from .external_mcp import ExternalMCPRegistry
from .game_design import GameDesignPlanner
from .importer import (
    ExistingProjectImportError,
    ExistingProjectReport,
    inspect_existing_project_archive,
)
from .model_adapters import ModelBackendError, ModelConfigurationError
from .model_registry import ModelRegistry
from .model_router import ModelRouter
from .pipeline import MinecraftModPipeline, PipelineResult
from .planner import HeuristicPlanner, OpenAICompatiblePlanner
from .production_tools import ProductionToolService
from .rag_index import ProjectRAGIndex
from .routed_planner import RoutedPlanner
from .spec import ArenaSpec, BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal
from .training import TrainingTraceStore
from .world_compiler import compile_world_ir

__all__ = [
    "ArenaSpec",
    "BossSpec",
    "ChatReply",
    "ContentSpec",
    "ExistingProjectImportError",
    "ExistingProjectReport",
    "ExternalMCPRegistry",
    "GameDesignPlanner",
    "HeuristicPlanner",
    "MinecraftModPipeline",
    "ModelBackendError",
    "ModelConfigurationError",
    "ModelRegistry",
    "ModelRouter",
    "ModAISession",
    "ModSpec",
    "OpenAICompatiblePlanner",
    "PipelineResult",
    "PlatformLock",
    "ProductionToolService",
    "ProjectRAGIndex",
    "Proposal",
    "RoutedPlanner",
    "TrainingTraceStore",
    "compile_world_ir",
    "inspect_existing_project_archive",
    "supported_minecraft_versions",
]

__version__ = "0.3.0"
