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

# Apply the mod-only scope to every complete planner path before public API import.
from . import complete_planner as _complete_planner_module
from . import complete_spec as _complete_spec_module
from .mod_scope_contract import install as _install_mod_scope_contract

_install_mod_scope_contract(_complete_spec_module, _complete_planner_module)

# Resource classification is installed before the orchestrator is imported.
from . import work_graph as _work_graph_module
from .work_graph_mutation_contract import install as _install_work_graph_mutation_contract

_install_work_graph_mutation_contract(_work_graph_module)

# Local GPU-backed text roles must participate in the same GPU exclusion contract
# as image/speech roles. Remote model roles remain independently concurrent.
from . import model_registry as _model_registry_module
from .gpu_resource_contract import install as _install_gpu_resource_contract

_install_gpu_resource_contract(_model_registry_module)

# Reuse expensive local model runtimes across bounded workflow calls. This is
# installed before the orchestrator imports generator functions so its asset-shard
# GPU session wrapper is the function the orchestrator binds.
from .model_runtime_performance import install as _install_model_runtime_performance

_install_model_runtime_performance()

# Increase llama.cpp logical prompt batching while retaining the existing physical
# microbatch ceiling and deterministic sampling parameters.
from .llama_runtime_tuning import install as _install_llama_runtime_tuning

_install_llama_runtime_tuning()

# On hosts with enough free VRAM, keep FLUX.2 Klein fully resident for the whole
# asset shard; otherwise retain its documented CPU-offload path. The cached pipeline
# is parked back on CPU before the local LLM can reacquire the GPU.
from .image_runtime_residency import install as _install_image_runtime_residency

_install_image_runtime_residency()

# Install isolated custom-generation staging plus the short, hash-guarded live
# project commit contract before public API users construct an orchestrator.
from . import complete_orchestrator as _complete_orchestrator_module
from . import custom_module_generator as _custom_module_generator_module
from . import source_patch as _source_patch_module
from . import performance_final_contract as _performance_final_contract_module
from .performance_final_tuning import install as _install_performance_final_tuning
from .performance_final_contract import install as _install_performance_final_contract

_install_performance_final_tuning(_performance_final_contract_module)
_install_performance_final_contract(
    _complete_orchestrator_module,
    _custom_module_generator_module,
    _source_patch_module,
)

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
from .production_contract import (
    ProductionContractCompilation,
    compile_production_contract,
    evaluate_quality_contract,
    quality_contract_summary,
    quality_unresolved,
)
from .project_index import ProjectIndex
from .rag_index import ProjectRAGIndex
from .routed_planner import RoutedPlanner
from .scale_policy import ScalePolicy, ScalePolicyError
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_pipeline import ScalableMinecraftModPipeline
from .spec import (
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
    "ProductionContractCompilation",
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
    "compute_voice_language_intersection",
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
