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

# Apply the mod-only scope and target-aware planning research before public API import.
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

# When the native llama-server binary is available, benchmark the real first
# workflow prompt against non-speculative and MTP variants. Only exact-output
# matches are eligible, and the hardware/model/server-fingerprinted winner is
# reused on subsequent runs.
from . import llama_server_autotune as _llama_server_autotune_module
from .llama_server_autotune import install as _install_llama_server_autotune

_install_llama_server_autotune()

# Let llama.cpp choose target/draft layer placement for the actual host rather than
# requiring every layer to fit in VRAM before the benchmark can run.
from .llama_server_hardware_policy import install as _install_llama_server_hardware_policy

_install_llama_server_hardware_policy(_llama_server_autotune_module)

# Start the selected GGUF download as soon as the model profile is resolved, overlap
# independent research/discovery I/O, and preserve deterministic result ordering.
# Local one-slot GPU decoding remains serialized; this contract only overlaps work
# that can execute independently without competing for that decode slot.
from . import colab_mtp_server as _colab_mtp_server_module
from .parallel_runtime_contract import install as _install_parallel_runtime_contract

_install_parallel_runtime_contract(
    complete_planner_module=_complete_planner_module,
    model_registry_module=_model_registry_module,
    llama_server_autotune_module=_llama_server_autotune_module,
    colab_mtp_server_module=_colab_mtp_server_module,
)

# In the Colab notebook MODEL_PROFILE already exists when minecraft_mod_ai is first
# imported in cell 3. Resolve that selected profile immediately so the asynchronous
# GGUF fetch overlaps existing-input preparation and later setup checks.
from .colab_prefetch_bootstrap import start as _start_colab_prefetch

_start_colab_prefetch(_model_registry_module)

# On hosts with enough free VRAM, keep FLUX.2 Klein fully resident for the whole
# asset shard; otherwise retain its documented CPU-offload path. The cached pipeline
# is parked back on CPU before the local LLM can reacquire the GPU.
from .image_runtime_residency import install as _install_image_runtime_residency

_install_image_runtime_residency()

# Verification is staged from cheap deterministic/JDT checks to Gradle/GameTest.
# Preserve Gradle incremental/build-cache state, reuse only exact-input evidence in
# process, and fail GameTest when generated-namespace resources did not load.
from . import java_lsp as _java_lsp_module
from . import repair_engine as _repair_engine_module
from . import validation_execution_contract as _validation_execution_contract_module
from .validation_execution_contract import install as _install_validation_execution_contract

_install_validation_execution_contract(
    _runner_module,
    _java_lsp_module,
    _repair_engine_module,
)

# JDT LS returns diagnostics as URI -> list[diagnostic]. Make the fail-fast repair
# gate consume that exact shape instead of silently iterating URI strings.
from .validation_diagnostic_contract import install as _install_validation_diagnostic_contract

_install_validation_diagnostic_contract(_validation_execution_contract_module)

# Generated content registrars are connected through a bounded compile-time tree.
# This avoids runtime classpath enumeration while keeping the root registrar bounded.
from . import extended_content_generator as _extended_content_generator_module
from .extended_registration_contract import install as _install_extended_registration_contract

_install_extended_registration_contract(_extended_content_generator_module)

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

# Public sessions no longer interpret the historical default 1.20.1 parameter as a
# global target. Omitted target means auto-selection; Revise preserves the imported
# project's target unless the user explicitly requests a migration.
from . import api as _api_module
from . import plan_render as _plan_render_module
from .platform_api_contract import install as _install_platform_api_contract

_install_platform_api_contract(_api_module, _plan_render_module)

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
