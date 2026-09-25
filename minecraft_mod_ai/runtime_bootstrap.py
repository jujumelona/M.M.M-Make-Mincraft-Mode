from __future__ import annotations

"""Direct, one-time runtime contract bootstrap.

The legacy runtime composer/finalizer stack is intentionally not restored.  This
module is the single explicit owner that installs the still-live contract modules
in deterministic stage order.
"""

from threading import RLock

_LOCK = RLock()
_BOOTSTRAPPED = False


def bootstrap_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _LOCK:
        if _BOOTSTRAPPED:
            return
        _install_core_contracts()
        _install_model_runtime_contracts()
        _install_validation_contracts()
        _install_generation_contracts()
        _install_platform_contracts()
        _install_planner_contracts()
        _install_architecture_contracts()
        _install_late_safety_contracts()
        _install_public_boundary_contracts()
        _install_post_bootstrap_contracts()
        _BOOTSTRAPPED = True


def initialize_runtime() -> None:
    bootstrap_runtime()


def runtime_initialized() -> bool:
    return _BOOTSTRAPPED


def _install_core_contracts() -> None:
    from . import runner, spec, work_graph
    from .hardware_concurrency import publish_effective_concurrency
    from .runner_lock_contract import install as install_runner_lock
    from .toolchain_contract import install as install_toolchain
    from .work_graph_mutation_contract import install as install_work_graph_mutation

    publish_effective_concurrency()
    install_toolchain(spec, runner)
    install_runner_lock(runner)
    install_work_graph_mutation(work_graph)


def _install_model_runtime_contracts() -> None:
    from . import (
        complete_orchestrator_services,
        llama_server_autotune,
        llama_server_hardware_policy,
        llama_server_runtime_tuning,
        llama_stream_efficiency_contract,
        model_context_budget,
        model_registry,
        model_router,
    )
    from .colab_gpu_handoff_contract import install as install_gpu_handoff
    from .colab_prefetch_bootstrap import start as start_colab_prefetch
    from .gpu_resource_contract import install as install_gpu_resource
    from .llama_completion_liveness_contract import install as install_completion_liveness
    from .llama_context_safety_contract import install as install_context_safety
    from .llama_generation_budget import install as install_llama_generation_budget
    from .llama_stream_efficiency_contract import install as install_llama_stream_efficiency
    from .llama_tuning_pipeline import install_native_llama_tuning_pipeline
    from .managed_llama_reuse_contract import install as install_managed_llama_reuse
    from .model_adapters import llama_cpp_adapter
    from .model_output_atomicity_contract import install as install_model_output_atomicity
    from .model_runtime_performance import install as install_model_runtime_performance
    from .qwen_agent_family_contract import install as install_qwen_agent_family

    install_gpu_resource(model_registry)
    install_model_runtime_performance()
    install_gpu_handoff(
        services_module=complete_orchestrator_services,
        model_router_module=model_router,
    )
    install_native_llama_tuning_pipeline(
        autotune=llama_server_autotune,
        hardware_policy=llama_server_hardware_policy,
        runtime_tuning=llama_server_runtime_tuning,
    )
    install_managed_llama_reuse()
    install_llama_generation_budget(llama_server_hardware_policy)
    install_llama_stream_efficiency(llama_server_hardware_policy)
    install_completion_liveness(llama_stream_efficiency_contract, llama_cpp_adapter)
    install_context_safety(model_context_budget)
    install_model_output_atomicity()
    install_qwen_agent_family()
    start_colab_prefetch(model_registry)


def _install_validation_contracts() -> None:
    from . import java_lsp, repair_engine, runner, validation_execution_contract
    from .java_lsp_process_safety_contract import install as install_java_lsp_process_safety
    from .research_validation_fingerprint_performance import harden as harden_validation_fingerprints
    from .validation_execution_contract import install as install_validation_execution

    install_validation_execution(runner, java_lsp, repair_engine)
    install_java_lsp_process_safety(java_lsp)
    harden_validation_fingerprints(validation_execution_contract)


def _install_generation_contracts() -> None:
    from . import (
        complete_orchestrator,
        custom_module_generator,
        extended_content_generator,
        performance_final_contract,
        project_index,
        source_patch,
    )
    from .deterministic_minecraft_content_contract import install as install_deterministic_minecraft_content
    from .extended_registration_contract import install as install_extended_registration
    from .performance_final_contract import install as install_performance_contract
    from .performance_final_tuning import install as install_performance_tuning
    from .project_index_manifest_efficiency_contract import install as install_project_index_manifest_efficiency
    from .project_manifest_hash_efficiency_contract import install as install_manifest_hash_efficiency

    install_extended_registration(extended_content_generator)
    install_deterministic_minecraft_content(extended_content_generator)
    install_project_index_manifest_efficiency(project_index)
    install_performance_tuning(performance_final_contract)
    install_manifest_hash_efficiency(complete_orchestrator, project_index)
    install_performance_contract(complete_orchestrator, custom_module_generator, source_patch)


def _install_platform_contracts() -> None:
    from . import (
        complete_orchestrator,
        complete_planner,
        complete_spec,
        geckolib_generator,
        generator,
        mineflayer_bridge,
        production_contract,
        repair_engine,
        retrieval,
        runtime_manager,
        spec,
        system_pack_generator,
        system_templates_common,
        technology_radar,
        validator,
    )
    from .minecraft_domain_correctness_contract import install as install_minecraft_domain_correctness
    from .mod_scope_contract import install as install_mod_scope
    from .platform_generation_contract import install as install_platform_generation
    from .platform_live_execution_contract import install as install_live_execution
    from .platform_live_rag_contract import install as install_platform_live_rag
    from .platform_repair_target_contract import install as install_platform_repair
    from .platform_runtime_contract import install as install_platform_runtime
    from .platform_specialized_generator_contract import install as install_specialized_generator_guards
    from .platform_technology_contract import install as install_platform_technology
    from .platform_validation_contract import install as install_platform_validation
    from .proposal_deserialization_contract import install as install_proposal_deserialization
    from .system_quality_contract import install as install_system_quality

    install_platform_runtime(
        orchestrator_module=complete_orchestrator,
        runtime_manager_module=runtime_manager,
        mineflayer_module=mineflayer_bridge,
    )
    install_proposal_deserialization(spec, complete_spec)
    install_platform_generation(generator)
    install_platform_validation(validator)
    install_platform_live_rag(retrieval_module=retrieval)
    install_platform_technology(technology_radar)
    install_mod_scope(complete_spec, complete_planner)
    install_platform_repair(repair_engine)
    install_live_execution(complete_orchestrator)
    install_minecraft_domain_correctness()
    install_specialized_generator_guards(
        system_module=system_pack_generator,
        geckolib_module=geckolib_generator,
        orchestrator_module=complete_orchestrator,
    )
    install_system_quality(
        templates_module=system_templates_common,
        system_module=system_pack_generator,
        production_contract_module=production_contract,
        runtime_module=runtime_manager,
        orchestrator_module=complete_orchestrator,
    )


def _install_planner_contracts() -> None:
    from . import (
        agentic_optimization_contract,
        complete_orchestrator,
        complete_orchestrator_services,
        model_router,
        resource_asset_production,
    )
    from .agentic_search_efficiency_contract import install as install_agentic_search_efficiency
    from .asset_resume_efficiency_contract import install as install_asset_resume_efficiency
    from .colab_gpu_handoff_contract import install as install_asset_gpu_handoff

    install_agentic_search_efficiency(agentic_optimization_contract)
    install_asset_resume_efficiency(resource_asset_production)
    install_asset_gpu_handoff(
        services_module=resource_asset_production,
        model_router_module=model_router,
    )
    canonical_generate_assets = resource_asset_production.generate_assets
    complete_orchestrator_services.generate_assets = canonical_generate_assets
    complete_orchestrator.generate_assets = canonical_generate_assets


def _install_architecture_contracts() -> None:
    from . import (
        agentic_optimization_contract,
        atomic_requirement_contract,
        complete_orchestrator,
        custom_module_generator,
        production_contract,
        quality_evidence,
        repair_engine,
        validation_execution_contract,
        work_graph,
    )
    from .atomic_efficiency_contract import install as install_atomic_efficiency
    from .atomic_evidence_routing_contract import install as install_atomic_routes
    from .atomic_playtest_evidence_contract import install as install_atomic_playtest
    from .atomic_quality_binding_contract import install as install_atomic_quality
    from .build_input_scope_contract import install as install_build_input_scope
    from .clean_room_verification_contract import install as install_clean_room
    from .coder_max_efficiency_contract import install_coder_max_efficiency
    from .custom_generation_search_contract import install as install_custom_generation_search
    from .repair_diagnostics_contract import install as install_repair_diagnostics
    from .repair_memory_budget_contract import install as install_repair_memory_budget
    from .required_gate_compatibility_contract import install as install_gate_compatibility
    from .semantic_reviewer_role_contract import install as install_reviewer_role
    from .visual_acceptance_scope_contract import install as install_visual_scope
    from .work_graph_state_transition_contract import install as install_work_graph_state_transitions

    install_build_input_scope(validation_execution_contract)
    install_atomic_efficiency(atomic_requirement_contract)
    install_atomic_routes(atomic_requirement_contract, production_contract)
    install_reviewer_role(atomic_requirement_contract)
    install_atomic_quality(atomic_requirement_contract, quality_evidence, complete_orchestrator)
    install_atomic_playtest(atomic_requirement_contract, quality_evidence, complete_orchestrator)
    install_repair_diagnostics(repair_engine)
    install_clean_room(complete_orchestrator, quality_evidence, validation_execution_contract)
    install_work_graph_state_transitions(work_graph)
    agentic_optimization_contract.install(
        repair_module=repair_engine,
        work_graph_module=work_graph,
    )
    install_custom_generation_search(custom_module_generator)
    install_coder_max_efficiency()
    install_repair_memory_budget(agentic_optimization_contract)
    install_visual_scope(complete_orchestrator)
    install_gate_compatibility(complete_orchestrator)


def _install_late_safety_contracts() -> None:
    from . import (
        complete_orchestrator,
        model_router,
        production_tools,
        runner,
        scheduler_parallel_safety_contract,
        validation_execution_contract,
        work_graph,
    )
    from .llama_parallel_runtime_contract import install as install_llama_parallel_runtime
    from .parallel_result_determinism_contract import install as install_parallel_result_determinism
    from .production_tool_parallel_contract import install as install_production_tool_parallel_safety
    from .runner_parallel_validation_contract import install as install_runner_parallel_validation
    from .scheduler_parallel_safety_contract import install as install_scheduler_parallel_safety

    install_scheduler_parallel_safety(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_llama_parallel_runtime(model_router, scheduler_parallel_safety_contract)
    install_production_tool_parallel_safety(production_tools)
    install_runner_parallel_validation(
        runner_module=runner,
        validation_module=validation_execution_contract,
    )
    install_parallel_result_determinism(orchestrator_module=complete_orchestrator)


def _install_public_boundary_contracts() -> None:
    from . import mcp_tools, production_tools
    from .platform_mcp_contract import install as install_platform_mcp
    from .platform_release_contract import install as install_platform_release

    install_platform_mcp(mcp_tools, production_tools)
    install_platform_release(mcp_tools)


def _install_post_bootstrap_contracts() -> None:
    from . import (
        agentic_optimization_contract,
        model_router,
        production_tools,
        repair_engine,
        small_model_max_agent_contract,
        work_graph,
    )
    from .active_repair_verifier_contract import install as install_active_repair_verifier
    from .adaptive_retrieval_contract import install as install_adaptive_retrieval
    from .agent_security_contract import install as install_agent_security
    from .minecraft_mcp_evidence_contract import install as install_minecraft_mcp_evidence
    from .research_bottleneck_runtime import install as install_research_bottleneck_runtime
    from .small_model_hybrid_search_contract import install as install_small_model_hybrid_search
    from .small_model_relation_index_contract import install as install_small_model_relation_index
    from .small_model_research_extensions_contract import install as install_small_model_research_extensions
    from .temporary_skill_contract import install as install_temporary_skill
    from .unified_trajectory_memory_contract import install as install_unified_trajectory_memory

    install_agent_security(
        pre_design_rag_module=None,
        agentic_research_module=None,
        model_router_module=model_router,
    )
    small_model_max_agent_contract._install_repair_context(
        repair_engine,
        agentic_optimization_contract,
    )
    install_small_model_relation_index(production_tools)
    install_small_model_hybrid_search(production_tools)
    install_temporary_skill(
        model_router_module=model_router,
        work_graph_module=work_graph,
        repair_module=repair_engine,
    )
    install_active_repair_verifier(agentic_optimization_contract)
    install_minecraft_mcp_evidence()
    install_research_bottleneck_runtime()
    install_small_model_research_extensions()
    install_unified_trajectory_memory()
    install_adaptive_retrieval(model_router)


__all__ = ["bootstrap_runtime", "initialize_runtime", "runtime_initialized"]
