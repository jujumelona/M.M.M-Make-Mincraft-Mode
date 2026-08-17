from __future__ import annotations

"""Ordered package runtime bootstrap with one explicit integration path."""

from threading import RLock

_BOOTSTRAP_LOCK = RLock()
_INITIALIZED = False


def initialize_runtime() -> None:
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
    from .reuse_asset_upgrade_contract import install_postbootstrap, install_prebootstrap

    install_prebootstrap()
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
    install_postbootstrap()


def _install_core_contracts() -> None:
    from . import runner, spec, validator, work_graph
    from .runner_lock_contract import install as install_runner_lock
    from .toolchain_contract import install as install_toolchain
    from .validator_boss_contract import install as install_validator_boss
    from .work_graph_mutation_contract import install as install_work_graph_mutation

    install_toolchain(spec, runner)
    install_runner_lock(runner)
    install_validator_boss(validator)
    install_work_graph_mutation(work_graph)


def _install_model_runtime_contracts() -> None:
    from . import (
        complete_orchestrator_services,
        llama_server_autotune,
        llama_server_hardware_policy,
        llama_server_runtime_tuning,
        model_registry,
        model_router,
    )
    from .colab_gpu_handoff_contract import install as install_gpu_handoff
    from .colab_prefetch_bootstrap import start as start_colab_prefetch
    from .gpu_resource_contract import install as install_gpu_resource
    from .image_runtime_residency import install as install_image_runtime_residency
    from .llama_stream_efficiency_contract import install as install_llama_stream_efficiency
    from .llama_tuning_pipeline import install_native_llama_tuning_pipeline
    from .model_runtime_performance import install as install_model_runtime_performance
    from .parallel_runtime_contract import install as install_parallel_runtime

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
    install_llama_stream_efficiency(llama_server_hardware_policy)
    install_parallel_runtime(
        model_registry_module=model_registry,
        llama_server_autotune_module=llama_server_autotune,
    )
    start_colab_prefetch(model_registry)
    install_image_runtime_residency()


def _install_validation_contracts() -> None:
    from . import java_lsp, repair_engine, runner, validation_execution_contract
    from .java_lsp_process_safety_contract import install as install_java_lsp_process_safety
    from .validation_diagnostic_contract import install as install_validation_diagnostics
    from .validation_execution_contract import install as install_validation_execution

    install_validation_execution(runner, java_lsp, repair_engine)
    install_validation_diagnostics(validation_execution_contract)
    install_java_lsp_process_safety(java_lsp)


def _install_generation_contracts() -> None:
    from . import (
        complete_orchestrator,
        custom_module_generator,
        extended_content_generator,
        performance_final_contract,
        project_index,
        source_patch,
    )
    from .extended_registration_contract import install as install_extended_registration
    from .performance_final_contract import install as install_performance_contract
    from .performance_final_tuning import install as install_performance_tuning
    from .project_index_execution_reuse_contract import (
        install as install_project_index_execution_reuse,
    )
    from .project_index_manifest_efficiency_contract import (
        install as install_project_index_manifest_efficiency,
    )
    from .project_manifest_hash_efficiency_contract import (
        install as install_manifest_hash_efficiency,
    )

    install_extended_registration(extended_content_generator)
    install_project_index_manifest_efficiency(project_index)
    install_project_index_execution_reuse(complete_orchestrator)
    install_performance_tuning(performance_final_contract)
    install_manifest_hash_efficiency(complete_orchestrator, project_index)
    install_performance_contract(
        complete_orchestrator,
        custom_module_generator,
        source_patch,
    )


def _install_platform_contracts() -> None:
    from . import (
        agentic_pre_design_rag,
        agentic_research_game_design,
        central_intelligence_amplifier,
        central_research,
        complete_orchestrator,
        complete_planner,
        complete_spec,
        custom_module_generator,
        game_design,
        geckolib_generator,
        generator,
        mineflayer_bridge,
        minecraft_knowledge_contract,
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
    from .mod_scope_contract import install as install_mod_scope
    from .platform_central_ai_contract import install as install_platform_central_ai
    from .platform_custom_coder_contract import install as install_platform_custom_coder
    from .platform_generation_contract import install as install_platform_generation
    from .platform_live_execution_contract import install as install_live_execution
    from .platform_live_rag_contract import install as install_platform_live_rag
    from .platform_planning_contract import install as install_platform_planning
    from .platform_repair_target_contract import install as install_platform_repair
    from .platform_runtime_contract import install as install_platform_runtime
    from .platform_specialized_generator_contract import (
        install as install_specialized_generator_guards,
    )
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
    install_platform_planning(
        game_design_module=game_design,
        complete_planner_module=complete_planner,
        central_research_module=central_research,
    )
    install_platform_live_rag(retrieval_module=retrieval)
    install_platform_technology(technology_radar)
    central_intelligence_amplifier.install_parallel_core(agentic_research_game_design)
    agentic_pre_design_rag.harden_pre_design_research(agentic_research_game_design)
    central_intelligence_amplifier.install(agentic_research_game_design)
    minecraft_knowledge_contract.install(agentic_research_game_design, complete_planner)
    agentic_research_game_design.bind_game_design_planner(game_design)
    install_platform_central_ai(
        game_design_module=game_design,
        complete_planner_module=complete_planner,
    )
    install_mod_scope(complete_spec, complete_planner)
    install_platform_custom_coder(custom_module_generator)
    install_platform_repair(repair_engine)
    install_live_execution(complete_orchestrator)
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
    """Install planner-independent efficiency policies only."""
    from . import agentic_optimization_contract, complete_orchestrator_services, work_graph
    from .agentic_search_efficiency_contract import (
        install as install_agentic_search_efficiency,
    )
    from .asset_resume_efficiency_contract import install as install_asset_resume_efficiency
    from .execution_efficiency_contract import install as install_execution_efficiency

    install_agentic_search_efficiency(agentic_optimization_contract)
    install_asset_resume_efficiency(complete_orchestrator_services)
    install_execution_efficiency(work_graph_module=work_graph)


def _install_architecture_contracts() -> None:
    from . import (
        agentic_optimization_contract,
        atomic_requirement_contract,
        complete_orchestrator,
        complete_planner,
        custom_module_generator,
        production_contract,
        quality_evidence,
        repair_engine,
        validation_execution_contract,
        work_graph,
    )
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
    from .repair_diagnostics_contract import install as install_repair_diagnostics
    from .repair_memory_budget_contract import install as install_repair_memory_budget
    from .required_gate_compatibility_contract import install as install_gate_compatibility
    from .semantic_reviewer_role_contract import install as install_reviewer_role
    from .visual_acceptance_scope_contract import install as install_visual_scope
    from .work_graph_state_transition_contract import (
        install as install_work_graph_state_transitions,
    )

    install_build_input_scope(validation_execution_contract)
    install_atomic_efficiency(atomic_requirement_contract)
    install_atomic_routes(atomic_requirement_contract, production_contract)
    install_reviewer_role(atomic_requirement_contract)
    install_atomic_requirements(complete_planner, complete_orchestrator)
    install_atomic_planner_policy(atomic_requirement_contract, complete_planner)
    install_atomic_execution(atomic_requirement_contract, complete_orchestrator)
    install_atomic_quality(
        atomic_requirement_contract,
        quality_evidence,
        complete_orchestrator,
    )
    install_atomic_playtest(
        atomic_requirement_contract,
        quality_evidence,
        complete_orchestrator,
    )
    install_repair_diagnostics(repair_engine, validation_execution_contract)
    install_orchestrator_jdt_gate(complete_orchestrator)
    install_clean_room(
        complete_orchestrator,
        quality_evidence,
        validation_execution_contract,
    )
    install_work_graph_state_transitions(work_graph)
    agentic_optimization_contract.install(
        complete_planner_module=complete_planner,
        repair_module=repair_engine,
        work_graph_module=work_graph,
    )
    install_custom_generation_search(custom_module_generator)
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
    from .max_efficiency_runtime_contract import (
        enhance_runtime as enhance_max_efficiency_runtime,
    )
    from .parallel_result_determinism_contract import (
        install as install_parallel_result_determinism,
    )
    from .production_tool_parallel_contract import (
        install as install_production_tool_parallel_safety,
    )
    from .runner_parallel_validation_contract import (
        install as install_runner_parallel_validation,
    )
    from .scheduler_claim_fencing_contract import install as install_scheduler_claim_fencing
    from .scheduler_connection_reuse_contract import (
        install as install_scheduler_connection_reuse,
    )
    from .scheduler_parallel_safety_contract import (
        install as install_scheduler_parallel_safety,
    )

    install_scheduler_parallel_safety(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_scheduler_connection_reuse(work_graph)
    install_llama_parallel_runtime(model_router, scheduler_parallel_safety_contract)
    enhance_max_efficiency_runtime(work_graph_module=work_graph)
    install_scheduler_claim_fencing(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
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
        agentic_pre_design_rag,
        agentic_research_game_design,
        model_router,
        production_tools,
        repair_engine,
        runtime_regression_fixes,
        work_graph,
    )
    from .active_repair_verifier_contract import install as install_active_repair_verifier
    from .agent_security_contract import install as install_agent_security
    from .causal_tool_frontier_contract import install as install_causal_tool_frontier
    from .long_run_resilience_contract import install as install_long_run_resilience
    from .minecraft_mcp_evidence_contract import install as install_minecraft_mcp_evidence
    from .research_bottleneck_runtime import install as install_research_bottleneck_runtime
    from .small_model_compacting_adapter import CompactingAdapter
    from .small_model_hybrid_search_contract import install as install_small_model_hybrid_search
    from .small_model_max_agent_contract import install as install_small_model_max_agent
    from .small_model_relation_index_contract import install as install_small_model_relation_index
    from .small_model_tool_guard_contract import install as install_small_model_tool_guard
    from .temporary_skill_contract import install as install_temporary_skill

    install_agent_security(
        pre_design_rag_module=agentic_pre_design_rag,
        agentic_research_module=agentic_research_game_design,
        model_router_module=model_router,
    )
    install_small_model_max_agent(
        model_router_module=model_router,
        pre_design_rag_module=agentic_pre_design_rag,
        production_tools_module=production_tools,
        repair_module=repair_engine,
        optimization_module=agentic_optimization_contract,
    )
    install_small_model_tool_guard(install_small_model_max_agent)
    install_causal_tool_frontier(install_small_model_max_agent)
    install_small_model_relation_index(production_tools)
    install_small_model_hybrid_search(production_tools)
    install_temporary_skill(
        model_router_module=model_router,
        work_graph_module=work_graph,
        repair_module=repair_engine,
    )
    install_active_repair_verifier(agentic_optimization_contract)

    current_tool_loop = model_router.ModelRouter._generate_with_tools
    if not getattr(current_tool_loop, "_mmm_lossless_context_compaction", False):

        def _generate_with_compaction(self, *, adapter, request, runtime, stage, role):
            return current_tool_loop(
                self,
                adapter=CompactingAdapter(adapter),
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )

        _generate_with_compaction._mmm_lossless_context_compaction = True
        _generate_with_compaction.__wrapped__ = current_tool_loop
        model_router.ModelRouter._generate_with_tools = _generate_with_compaction

    install_minecraft_mcp_evidence()
    install_research_bottleneck_runtime()
    runtime_regression_fixes.install()
    install_long_run_resilience()
