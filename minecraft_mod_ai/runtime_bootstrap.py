from __future__ import annotations

"""Single, ordered runtime contract bootstrap.

MMM still uses contract installers for cross-cutting runtime policies, but package
initialization has one integration point and each runtime installer is applied once.
Installers that previously re-entered other bootstraps are invoked explicitly here
in their final dependency order.
"""

from threading import RLock

_BOOTSTRAP_LOCK = RLock()
_INITIALIZED = False


def initialize_runtime() -> None:
    """Install package-wide runtime contracts exactly once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _BOOTSTRAP_LOCK:
        if _INITIALIZED:
            return
        _install_runtime_contracts()
        _INITIALIZED = True


def runtime_initialized() -> bool:
    return _INITIALIZED


def _install_runtime_contracts() -> None:
    # Core runner/spec and validation contracts must precede downstream imports.
    from . import runner as runner_module
    from . import spec as spec_module
    from .toolchain_contract import install as install_toolchain

    install_toolchain(spec_module, runner_module)

    from . import validator as validator_module
    from .validator_boss_contract import install as install_validator_boss

    install_validator_boss(validator_module)

    # Planning scope is fixed before orchestrator/API modules bind planner functions.
    from . import complete_planner as complete_planner_module
    from . import complete_spec as complete_spec_module
    from .mod_scope_contract import install as install_mod_scope

    install_mod_scope(complete_spec_module, complete_planner_module)

    from . import work_graph as work_graph_module
    from .work_graph_mutation_contract import install as install_work_graph_mutation

    install_work_graph_mutation(work_graph_module)

    # Local model resource ownership and native llama policy.
    from . import model_registry as model_registry_module
    from .gpu_resource_contract import install as install_gpu_resource

    install_gpu_resource(model_registry_module)

    from .model_runtime_performance import install as install_model_runtime_performance

    install_model_runtime_performance()

    from . import llama_server_autotune as llama_autotune_module
    from . import llama_server_hardware_policy as llama_hardware_module
    from .llama_server_hardware_policy import install as install_llama_hardware

    install_llama_hardware(llama_autotune_module)

    from .parallel_runtime_contract import install as install_parallel_runtime

    install_parallel_runtime(
        complete_planner_module=complete_planner_module,
        model_registry_module=model_registry_module,
        llama_server_autotune_module=llama_autotune_module,
    )

    from .colab_prefetch_bootstrap import start as start_colab_prefetch

    start_colab_prefetch(model_registry_module)

    from .image_runtime_residency import install as install_image_runtime_residency

    install_image_runtime_residency()

    # Validation/JDT layers.
    from . import java_lsp as java_lsp_module
    from . import repair_engine as repair_module
    from . import validation_execution_contract as validation_module
    from .validation_execution_contract import install as install_validation_execution

    install_validation_execution(runner_module, java_lsp_module, repair_module)

    from .validation_diagnostic_contract import install as install_validation_diagnostics
    from .java_lsp_process_safety_contract import install as install_java_lsp_process_safety

    install_validation_diagnostics(validation_module)
    install_java_lsp_process_safety(java_lsp_module)

    # Deterministic generation/index primitives.
    from . import extended_content_generator as extended_content_module
    from .extended_registration_contract import install as install_extended_registration

    install_extended_registration(extended_content_module)

    from . import project_index as project_index_module
    from .project_index_manifest_efficiency_contract import (
        install as install_project_index_manifest_efficiency,
    )

    install_project_index_manifest_efficiency(project_index_module)

    # Orchestrator and custom-generation concurrency policies.
    from . import complete_orchestrator as orchestrator_module
    from . import custom_module_generator as custom_generator_module
    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module
    from .llama_server_efficiency_contract import install as install_llama_efficiency
    from .performance_final_tuning import install as install_performance_tuning
    from .project_manifest_hash_efficiency_contract import (
        install as install_manifest_hash_efficiency,
    )
    from .performance_final_contract import install as install_performance_contract

    install_llama_efficiency(llama_autotune_module, llama_hardware_module)
    install_performance_tuning(performance_module)
    install_manifest_hash_efficiency(orchestrator_module, project_index_module)
    install_performance_contract(
        orchestrator_module,
        custom_generator_module,
        source_patch_module,
    )

    # Runtime target binding precedes live-target execution overlays.
    from . import mineflayer_bridge as mineflayer_module
    from . import runtime_manager as runtime_module
    from .platform_runtime_contract import install as install_platform_runtime

    install_platform_runtime(
        orchestrator_module=orchestrator_module,
        runtime_manager_module=runtime_module,
        mineflayer_module=mineflayer_module,
    )

    # Safety and platform-aware planning/generation.
    from . import external_mcp_router as external_mcp_router_module
    from .external_mcp_bridge_safety_contract import (
        install as install_external_mcp_bridge_safety,
    )
    from .proposal_deserialization_contract import install as install_proposal_deserialization

    install_external_mcp_bridge_safety(external_mcp_router_module)
    install_proposal_deserialization(spec_module, complete_spec_module)

    from . import generator as generator_module
    from .platform_generation_contract import install as install_platform_generation
    from .platform_validation_contract import install as install_platform_validation

    install_platform_generation(generator_module)
    install_platform_validation(validator_module)

    from . import central_research as central_research_module
    from . import game_design as game_design_module
    from . import platform_central_ai_contract as platform_central_ai_module
    from . import platform_resolver as platform_resolver_module
    from . import retrieval as retrieval_module
    from . import technology_radar as technology_module
    from .platform_central_ai_contract import install as install_platform_central_ai
    from .platform_planning_contract import install as install_platform_planning
    from .platform_selection_efficiency_contract import (
        install as install_platform_selection_efficiency,
    )

    install_platform_planning(
        game_design_module=game_design_module,
        complete_planner_module=complete_planner_module,
        central_research_module=central_research_module,
        retrieval_module=retrieval_module,
        technology_module=technology_module,
    )
    install_platform_selection_efficiency(
        resolver_module=platform_resolver_module,
        central_contract_module=platform_central_ai_module,
    )
    install_platform_central_ai(
        game_design_module=game_design_module,
        complete_planner_module=complete_planner_module,
    )

    from .platform_custom_coder_contract import install as install_platform_custom_coder
    from .platform_repair_target_contract import install as install_platform_repair
    from .platform_live_execution_contract import install as install_live_execution

    install_platform_custom_coder(custom_generator_module)
    install_platform_repair(repair_module)
    install_live_execution(orchestrator_module)

    from . import geckolib_generator as geckolib_module
    from . import system_pack_generator as system_module
    from .platform_specialized_generator_contract import (
        install as install_specialized_generator_guards,
    )

    install_specialized_generator_guards(
        system_module=system_module,
        geckolib_module=geckolib_module,
        orchestrator_module=orchestrator_module,
    )

    from . import production_contract as production_contract_module
    from . import system_templates_common as system_templates_module
    from .system_quality_contract import install as install_system_quality

    install_system_quality(
        templates_module=system_templates_module,
        system_module=system_module,
        production_contract_module=production_contract_module,
        runtime_module=runtime_module,
        orchestrator_module=orchestrator_module,
    )

    # Final architecture contracts are expanded here instead of entering a nested
    # final_architecture bootstrap. Ordering matches the previous authoritative path.
    from . import agentic_optimization_contract as agentic_module
    from . import atomic_requirement_contract as atomic_module
    from . import planner_json_runtime_contract as planner_json_runtime_module
    from . import planner_pagination_safety_contract as planner_pagination_module
    from . import quality_evidence as quality_module
    from .atomic_efficiency_contract import install as install_atomic_efficiency
    from .atomic_evidence_routing_contract import install as install_atomic_routes
    from .atomic_execution_policy_contract import install as install_atomic_execution
    from .atomic_planner_policy_contract import install as install_atomic_planner_policy
    from .atomic_playtest_evidence_contract import install as install_atomic_playtest
    from .atomic_quality_binding_contract import install as install_atomic_quality
    from .atomic_requirement_contract import install as install_atomic_requirements
    from .build_input_scope_contract import install as install_build_input_scope
    from .clean_room_verification_contract import install as install_clean_room
    from .custom_generation_search_contract import install as install_custom_generation_search
    from .orchestrator_jdt_gate_contract import install as install_orchestrator_jdt_gate
    from .planner_json_runtime_contract import install as install_planner_json_runtime
    from .planner_module_identity_contract import install as install_planner_module_identity
    from .planner_outline_identity_contract import install as install_planner_outline_identity
    from .planner_pagination_safety_contract import install as install_planner_pagination_safety
    from .planner_parser_safety_contract import install as install_planner_parser_safety
    from .planner_strict_json_contract import install as install_planner_strict_json
    from .repair_diagnostics_contract import install as install_repair_diagnostics
    from .repair_memory_budget_contract import install as install_repair_memory_budget
    from .required_gate_compatibility_contract import install as install_gate_compatibility
    from .scheduler_fairness_contract import install as install_scheduler_fairness
    from .semantic_reviewer_role_contract import install as install_reviewer_role
    from .visual_acceptance_scope_contract import install as install_visual_scope
    from .work_graph_state_transition_contract import (
        install as install_work_graph_state_transitions,
    )

    install_build_input_scope(validation_module)
    install_atomic_efficiency(atomic_module)
    install_atomic_routes(atomic_module, production_contract_module)
    install_reviewer_role(atomic_module)
    install_atomic_requirements(complete_planner_module, orchestrator_module)
    install_atomic_planner_policy(atomic_module, complete_planner_module)
    install_atomic_execution(atomic_module, orchestrator_module)
    install_atomic_quality(atomic_module, quality_module, orchestrator_module)
    install_atomic_playtest(atomic_module, quality_module, orchestrator_module)
    install_repair_diagnostics(repair_module, validation_module)
    install_orchestrator_jdt_gate(orchestrator_module)
    install_clean_room(orchestrator_module, quality_module, validation_module)
    install_planner_json_runtime(complete_planner_module)
    install_planner_strict_json(planner_json_runtime_module)
    install_planner_parser_safety(complete_planner_module)
    install_planner_module_identity(complete_planner_module)
    install_planner_pagination_safety(complete_planner_module)
    install_planner_outline_identity(planner_pagination_module)
    install_work_graph_state_transitions(work_graph_module)
    agentic_module.install(
        complete_planner_module=complete_planner_module,
        repair_module=repair_module,
        work_graph_module=work_graph_module,
    )
    install_custom_generation_search(custom_generator_module)
    install_repair_memory_budget(agentic_module)
    install_scheduler_fairness(work_graph_module)
    install_visual_scope(orchestrator_module)
    install_gate_compatibility(orchestrator_module)

    # Late safety wrappers are installed once, after every method-replacing policy.
    from .scheduler_parallel_safety_contract import (
        install as install_scheduler_parallel_safety,
    )
    from .scheduler_claim_fencing_contract import install as install_scheduler_claim_fencing

    install_scheduler_parallel_safety(
        work_graph_module=work_graph_module,
        orchestrator_module=orchestrator_module,
    )
    install_scheduler_claim_fencing(
        work_graph_module=work_graph_module,
        orchestrator_module=orchestrator_module,
    )

    from . import production_tools as production_tools_module
    from .production_tool_parallel_contract import (
        install as install_production_tool_parallel_safety,
    )

    install_production_tool_parallel_safety(production_tools_module)

    # Validation and deterministic result wrappers are last by design.
    from .runner_parallel_validation_contract import (
        install as install_runner_parallel_validation,
    )

    install_runner_parallel_validation(
        runner_module=runner_module,
        validation_module=validation_module,
    )

    from . import audio_generator as audio_generator_module
    from .parallel_result_determinism_contract import (
        install as install_parallel_result_determinism,
    )

    install_parallel_result_determinism(
        audio_generator_module=audio_generator_module,
        orchestrator_module=orchestrator_module,
    )

    # MCP and public API target binding are final authority boundaries.
    from . import mcp_tools as mcp_tools_module
    from .platform_mcp_contract import install as install_platform_mcp

    install_platform_mcp(mcp_tools_module, production_tools_module)

    from . import api as api_module
    from . import plan_render as plan_render_module
    from .platform_api_contract import install as install_platform_api

    install_platform_api(api_module, plan_render_module)


__all__ = ["initialize_runtime", "runtime_initialized"]
