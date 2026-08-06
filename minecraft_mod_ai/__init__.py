"""M.M.M Make Mincraft Mode: scalable multimodal Fabric mod production tools."""

# Install the verified toolchain before generator and runner users are imported.
from . import runner as _runner_module
from . import spec as _spec_module
from .toolchain_contract import install as _install_toolchain_contract

_install_toolchain_contract(_spec_module, _runner_module)

# Install the corrected legacy-boss source contract before pipeline imports.
from . import validator as _validator_module
from .validator_boss_contract import install as _install_validator_boss_contract

_install_validator_boss_contract(_validator_module)

from .api import (
    ChatReply,
    CompleteChatReply,
    CompleteModAISession,
    ModAISession,
    supported_minecraft_versions,
)
from .complete_orchestrator import (
    CompleteExecutionOptions,
    CompletePipelineResult,
    CompleteProductionOrchestrator,
)
from .complete_orchestrator_support import CompleteProductionError
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import (
    AudioRequest,
    AssetRequest,
    CompleteProposal,
    ProductionModule,
)
from .ecosystem_discovery import EcosystemDiscoveryClient
from .external_mcp import ExternalMCPRegistry
from .game_design import GameDesignPlanner
from .importer import (
    ExistingProjectImportError,
    ExistingProjectReport,
    inspect_existing_project_archive,
)
from .mod_development_methods import (
    ModDevelopmentMethod,
    mod_development_method_catalog,
    resolve_mod_development_methods,
)
from .model_adapters import ModelBackendError, ModelConfigurationError
from .model_registry import ModelRegistry
from .model_router import ModelRouter
from .pipeline import PipelineResult
from .planner import HeuristicPlanner, OpenAICompatiblePlanner
from .production_tools import ProductionToolService
from .project_index import ProjectIndex
from .rag_index import ProjectRAGIndex
from .routed_planner import RoutedPlanner
from .scale_policy import ScalePolicy, ScalePolicyError
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_pipeline import ScalableMinecraftModPipeline
from .spec import (
    ArenaSpec,
    BossSpec,
    ContentSpec,
    ModSpec,
    PlatformLock,
    Proposal,
)
from .training import TrainingTraceStore
from .technology_radar import (
    assess_technology_compatibility,
    build_technology_radar,
    compute_voice_language_intersection,
    technology_research_routes,
)

MinecraftModPipeline = ScalableMinecraftModPipeline

__all__ = [
    "ArenaSpec",
    "AssetRequest",
    "AudioRequest",
    "BossSpec",
    "ChatReply",
    "CompleteChatReply",
    "CompleteExecutionOptions",
    "CompleteGameDesignPlanner",
    "CompleteModAISession",
    "CompletePipelineResult",
    "CompleteProductionError",
    "CompleteProductionOrchestrator",
    "CompleteProposal",
    "ContentSpec",
    "EcosystemDiscoveryClient",
    "ExistingProjectImportError",
    "ExistingProjectReport",
    "ExternalMCPRegistry",
    "GameDesignPlanner",
    "HeuristicPlanner",
    "MinecraftModPipeline",
    "ModDevelopmentMethod",
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
    "ProjectIndex",
    "ProjectRAGIndex",
    "Proposal",
    "RoutedPlanner",
    "ScalePolicy",
    "ScalePolicyError",
    "ScalableFabricProjectGenerator",
    "ScalableMinecraftModPipeline",
    "TrainingTraceStore",
    "assess_technology_compatibility",
    "build_technology_radar",
    "compute_voice_language_intersection",
    "inspect_existing_project_archive",
    "mod_development_method_catalog",
    "resolve_mod_development_methods",
    "supported_minecraft_versions",
    "technology_research_routes",
]

__version__ = "0.8.0"
