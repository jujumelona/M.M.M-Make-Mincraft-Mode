"""M.M.M Make Mincraft Mode: scalable multimodal Minecraft mod production tools."""

from .custom_checkpoint_performance_installation import install as install_checkpoint_performance
from .execution_feedback_semantic_convergence_installation import (
    install as install_execution_feedback_semantic_convergence,
)
from .generation_accuracy_contract import (
    assert_inner_installed as assert_generation_accuracy_inner,
)
from .generation_accuracy_contract import (
    assert_outer_installed as assert_generation_accuracy_outer,
)
from .generation_accuracy_contract import install_inner as install_generation_accuracy_inner
from .generation_accuracy_contract import install_outer as install_generation_accuracy_outer
from .generation_verifier_fallback_installation import install as install_generation_verifier_fallback
from .hardware_concurrency_installation import install as install_hardware_concurrency
from .java_toolchain_separation_installation import install as install_java_toolchain_separation
from . import jdtls_bootstrap as _jdtls_bootstrap
from .runtime_bootstrap import initialize_runtime
from .runtime_finalization import finalize_runtime
from .source_observation_budget_installation import install as install_source_observation_budget
from .source_set_boundary_installation import install as install_source_set_boundary
from .task_template_catalog import RUNTIME_TEMPLATE_ROOT
from .template_contract_validation import runtime_consumer_roots, validate_catalog
from .verification_pending_result_installation import install as install_verification_pending_result
from .versioned_reference_context_installation import install as install_versioned_reference_context


def _validate_runtime_template_authority() -> None:
    """Fail before model decode when any runtime template is orphaned or structurally invalid."""
    validate_catalog(
        RUNTIME_TEMPLATE_ROOT,
        consumer_roots=runtime_consumer_roots(),
    )


# MMM_LLAMA_ACTIVE_PARALLEL describes proven live server capacity.  Importing the
# package must never synthesize that receipt from a desired/default server width.
install_hardware_concurrency()
initialize_runtime()
# A generation/quality MCP child inherits the target-selected MMM_JAVA_VERSION.
# Colab setup may have run before that value existed, so ensure the exact project
# JDK again at fresh process bootstrap.  When no target Java version is present,
# this is a no-op.  java_lsp remains the single canonical resolver/validator.
_jdtls_bootstrap._ensure_project_jdk()
_validate_runtime_template_authority()
from . import custom_module_generator as _custom_module_generator
from . import execution_feedback_replan_contract as _execution_feedback_replan_contract
from . import java_lsp as _java_lsp
from . import model_router as _model_router

install_checkpoint_performance(_custom_module_generator)
# JDT LS may run on a tooling JDK that differs from the project JDK. Pin the
# Buildship/Gradle daemon to the already-resolved project JDK before any source-set
# or generation verifier wrapper can start Java diagnostics.
install_java_toolchain_separation(_java_lsp)
install_source_set_boundary(_java_lsp)
# Install the accuracy verifier before runtime finalization. The atomic coder slicer is
# finalized later and therefore calls through this boundary once for every obligation.
install_generation_accuracy_inner(_model_router)
assert_generation_accuracy_inner(_model_router)
# Execution feedback must own deterministic base-project diagnostics and its retry
# termination rule before runtime_finalization installs the durable feedback wrapper.
install_execution_feedback_semantic_convergence(_execution_feedback_replan_contract)
finalize_runtime()
# The generation verifier is finalized above. Wrap only its infrastructure-unavailable
# path with a real pinned Gradle build; source failures remain ordinary verifier FAILs.
install_generation_verifier_fallback()
# Explicit pause/resume callers receive a structured pending receipt when every host
# verifier is unavailable. The normal generate() method remains production fail-closed.
install_verification_pending_result(_custom_module_generator)
# The outer normalizer runs after atomic aggregation so multi-obligation text summaries
# retain the fixed {"summary": ...} contract consumed by CustomModuleGenerator.
install_generation_accuracy_outer(_model_router)
assert_generation_accuracy_outer(_model_router)
install_versioned_reference_context()

from . import complete_orchestrator as _complete_orchestrator
from .prepared_project_resume_integrity import install as install_prepared_project_resume_integrity

install_prepared_project_resume_integrity(_complete_orchestrator)

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
    "AssetRequest", "AuthoredPlan", "BossSpec", "ChatReply", "CompleteChatReply",
    "CompleteExecutionOptions", "CompleteGameDesignPlanner", "CompleteModAISession",
    "CompletePipelineResult", "CompleteProductionError", "CompleteProductionOrchestrator",
    "CompleteProposal", "ContentSpec", "EcosystemDiscoveryClient", "ExistingProjectImportError",
    "ExistingProjectReport", "ExternalMCPRegistry", "HeuristicPlanner", "MinecraftModPipeline",
    "ModAISession", "ModDevelopmentMethod", "ModSpec", "ModelBackendError",
    "ModelConfigurationError", "ModelRegistry", "ModelRouter", "OpenAICompatiblePlanner",
    "PipelineResult", "PlatformLock", "ProductionContractCompilation", "ProductionModule",
    "ProductionToolService", "ProjectIndex", "ProjectRAGIndex", "Proposal", "RoutedPlanner",
    "ScalableFabricProjectGenerator", "ScalableMinecraftModPipeline", "ScalePolicy",
    "ScalePolicyError", "assess_technology_compatibility", "build_technology_radar",
    "compile_production_contract", "evaluate_quality_contract", "inspect_existing_project_archive",
    "mod_development_method_catalog", "quality_contract_summary", "quality_unresolved",
    "resolve_mod_development_methods", "supported_minecraft_versions", "technology_research_routes",
]

__version__ = "0.8.0"
