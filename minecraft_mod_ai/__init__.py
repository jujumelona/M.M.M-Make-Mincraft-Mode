"""M.M.M Make Mincraft Mode: scalable multimodal Minecraft mod production tools."""

import os

from .hardware_concurrency_installation import install as install_hardware_concurrency
from .runtime_bootstrap import initialize_runtime
from .runtime_finalization import finalize_runtime
from .source_observation_budget_installation import install as install_source_observation_budget
from .source_set_boundary_installation import install as install_source_set_boundary
from .versioned_reference_context_installation import install as install_versioned_reference_context


def _configure_default_llama_parallelism() -> None:
    """Use the reviewed scheduler maximum unless server capacity is explicit.

    The application scheduler is already bounded to eight concurrent llama requests.
    A positive ``MMM_LLAMA_PARALLEL`` value remains authoritative and is clamped to
    that reviewed maximum. Missing, invalid, and server-auto values keep the full
    application-side width so independent small-model tasks are not serialized.
    """

    if os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip():
        return
    raw_server_parallel = os.environ.get("MMM_LLAMA_PARALLEL", "").strip()
    try:
        explicit_server_parallel = int(raw_server_parallel)
    except ValueError:
        explicit_server_parallel = 0
    if explicit_server_parallel > 0:
        active = max(1, min(8, explicit_server_parallel))
    else:
        active = 8
    os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = str(active)


_configure_default_llama_parallelism()
install_hardware_concurrency()
initialize_runtime()
from . import java_lsp as _java_lsp

install_source_set_boundary(_java_lsp)
finalize_runtime()
install_versioned_reference_context()

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
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_pipeline import ScalableMinecraftModPipeline
from .scale_policy import ScalePolicy, ScalePolicyError
from .spec import BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal
from .technology_radar import (
    assess_technology_compatibility,
    build_technology_radar,
    technology_research_routes,
)

install_source_observation_budget()

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
    "HeuristicPlanner",
    "MinecraftModPipeline",
    "ModAISession",
    "ModDevelopmentMethod",
    "ModSpec",
    "ModelBackendError",
    "ModelConfigurationError",
    "ModelRegistry",
    "ModelRouter",
    "OpenAICompatiblePlanner",
    "PipelineResult",
    "PlatformLock",
    "ProductionContractCompilation",
    "ProductionModule",
    "ProductionToolService",
    "ProjectIndex",
    "ProjectRAGIndex",
    "Proposal",
    "RoutedPlanner",
    "ScalableFabricProjectGenerator",
    "ScalableMinecraftModPipeline",
    "ScalePolicy",
    "ScalePolicyError",
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