"""M.M.M Make Mincraft Mode: scalable multimodal Minecraft mod production tools."""

from .api import (
    CompleteChatReply,
    CompleteModAISession,
    supported_minecraft_versions,
)
from .complete_orchestrator import (
    CompleteExecutionOptions,
    CompletePipelineResult,
    CompleteProductionOrchestrator,
)
from .complete_orchestrator_support import CompleteProductionError
from .complete_planner import CompleteGameDesignPlanner
from .authored_plan import AuthoredPlan
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
from .scalable_generator import ScalableFabricProjectGenerator
from .scale_policy import ScalePolicy, ScalePolicyError
from .spec import BossSpec, ContentSpec, ModSpec, PlatformLock, Proposal
from .technology_radar import (
    assess_technology_compatibility,
    build_technology_radar,
    technology_research_routes,
)

__all__ = [
    "AssetRequest", "AuthoredPlan", "BossSpec", "CompleteChatReply",
    "CompleteExecutionOptions", "CompleteGameDesignPlanner", "CompleteModAISession",
    "CompletePipelineResult", "CompleteProductionError", "CompleteProductionOrchestrator",
    "CompleteProposal", "ContentSpec", "EcosystemDiscoveryClient", "ExistingProjectImportError",
    "ExistingProjectReport", "ExternalMCPRegistry", "ModDevelopmentMethod", "ModSpec",
    "ModelBackendError", "ModelConfigurationError", "ModelRegistry", "ModelRouter",
    "PlatformLock", "ProductionContractCompilation", "ProductionModule",
    "ProductionToolService", "ProjectIndex", "ProjectRAGIndex", "Proposal",
    "ScalableFabricProjectGenerator", "ScalePolicy", "ScalePolicyError",
    "assess_technology_compatibility", "build_technology_radar",
    "compile_production_contract", "evaluate_quality_contract", "inspect_existing_project_archive",
    "mod_development_method_catalog", "quality_contract_summary", "quality_unresolved",
    "resolve_mod_development_methods", "supported_minecraft_versions", "technology_research_routes",
]

__version__ = "0.8.0"



# Installer-backed cross-cutting contracts have exactly one package-level owner.
# Runtime contracts are source-owned. Importing the package must not mutate or
# rebind production functions through an installer/bootstrap phase.
