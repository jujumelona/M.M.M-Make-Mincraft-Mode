"""M.M.M Make Mincraft Mode: scalable multimodal Minecraft mod production tools."""

from .coder_tool_route_integrity_contract import install as install_coder_tool_route_integrity
from .mcp_transport_pool import install_agent_mcp_transport_pool
from .runtime_bootstrap import initialize_runtime
from .runtime_preflight import run_runtime_preflight

initialize_runtime()
install_agent_mcp_transport_pool()

# The progress-aware retrieval loop is intentionally installed late by the runtime
# bootstrap. Recompose the live causal frontier after that late owner so writable
# coder turns retain the complete authorized mutation surface behind per-turn gates.
from . import causal_tool_frontier_contract as _causal_tool_frontier_contract
from . import model_router as _model_router
from . import small_model_max_agent_contract as _small_model_max_agent_contract

install_coder_tool_route_integrity(
    model_router_module=_model_router,
    small_model_module=_small_model_max_agent_contract,
    causal_module=_causal_tool_frontier_contract,
)

# Run a model-free structural smoke test only after the final tool-loop composition
# is installed. Any Python/wrapper/causal regression now fails during package import,
# before a production run spends time loading multi-gigabyte model weights.
run_runtime_preflight()

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
from .complete_spec import AssetRequest, CompleteProposal, ProductionModule
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
from .preference_training import (
    PreferenceCandidate,
    PreferenceTraceError,
    PreferenceTraceStore,
)
from .production_contract import (
    ProductionContractCompilation,
    compile_production_contract,
    evaluate_quality_contract,
    quality_contract_summary,
    quality_unresolved,
)
from .production_tools import ProductionToolService
from .project_index import ProjectIndex
from .rag_index import ProjectRAGIndex
from .routed_planner import RoutedPlanner
from .scale_policy import ScalePolicy, ScalePolicyError
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_pipeline import ScalableMinecraftModPipeline
from .spec import BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal
from .technology_radar import (
    assess_technology_compatibility,
    build_technology_radar,
    technology_research_routes,
)
from .training import TrainingTraceStore

MinecraftModPipeline = ScalableMinecraftModPipeline

__all__ = [
    "AssetRequest",
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
    "PreferenceCandidate",
    "PreferenceTraceError",
    "PreferenceTraceStore",
    "ProductionContractCompilation",
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
    "compile_production_contract",
    "evaluate_quality_contract",
    "inspect_existing_project_archive",
    "mod_development_method_catalog",
    "quality_contract_summary",
    "quality_unresolved",
    "resolve_mod_development_methods",
    "supported_minecraft_versions",
    "technology_research_routes",
]

__version__ = "0.8.0"
