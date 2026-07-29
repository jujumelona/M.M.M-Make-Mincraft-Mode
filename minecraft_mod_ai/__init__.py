"""M.M.M Make Mincraft Mode: role-routed multimodal Fabric production tools."""

from .api import ChatReply, CompleteChatReply, CompleteModAISession, ModAISession, supported_minecraft_versions
from .complete_orchestrator import CompleteExecutionOptions, CompletePipelineResult, CompleteProductionOrchestrator
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import AudioRequest, AssetRequest, CompleteProposal, ProductionModule
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
    "CompleteChatReply",
    "CompleteExecutionOptions",
    "CompleteGameDesignPlanner",
    "CompleteModAISession",
    "CompletePipelineResult",
    "CompleteProductionOrchestrator",
    "CompleteProposal",
    "ContentSpec",
    "AssetRequest",
    "AudioRequest",
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
    "ProductionModule",
    "ProductionToolService",
    "ProjectRAGIndex",
    "Proposal",
    "RoutedPlanner",
    "TrainingTraceStore",
    "compile_world_ir",
    "inspect_existing_project_archive",
    "supported_minecraft_versions",
]

__version__ = "0.4.0"
