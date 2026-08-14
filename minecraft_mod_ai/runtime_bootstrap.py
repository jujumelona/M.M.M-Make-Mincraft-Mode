from __future__ import annotations

"""Ordered package runtime bootstrap with one explicit integration path.

Cross-cutting contracts stay isolated and testable, while package initialization
owns their composition. No installer re-enters another package bootstrap and no
runtime contract is intentionally installed twice.
"""

import os
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
    """Compose runtime policies in dependency order."""
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


def _install_core_contracts() -> None:
    """Install base spec, validation, runner-lock and work-graph semantics."""
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
    """Install local model ownership and native llama runtime policy once."""
    from . import (
        complete_orchestrator_services,
        complete_planner,
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
        complete_planner_module=complete_planner,
        model_registry_module=model_registry,
        llama_server_autotune_module=llama_server_autotune,
    )
    start_colab_prefetch(model_registry)
    install_image_runtime_residency()


def _install_validation_contracts() -> None:
    """Install JDT/Gradle validation behavior before orchestrator binding."""
    from . import java_lsp, repair_engine, runner, validation_execution_contract
    from .java_lsp_process_safety_contract import install as install_java_lsp_process_safety
    from .validation_diagnostic_contract import install as install_validation_diagnostics
    from .validation_execution_contract import install as install_validation_execution

    install_validation_execution(runner, java_lsp, repair_engine)
    install_validation_diagnostics(validation_execution_contract)
    install_java_lsp_process_safety(java_lsp)


def _install_generation_contracts() -> None:
    """Install deterministic generation, indexing and project mutation policies."""
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
    from .project_index_execution_reuse_contract import install as install_project_index_execution_reuse
    from .project_index_manifest_efficiency_contract import install as install_project_index_manifest_efficiency
    from .project_manifest_hash_efficiency_contract import install as install_manifest_hash_efficiency

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
    """Bind planning, generation, runtime and quality to one Minecraft target."""
    from . import (
        agentic_pre_design_rag,
        agentic_research_game_design,
        central_intelligence_amplifier,
        central_research,
        complete_orchestrator,
        complete_planner,
        complete_spec,
        custom_module_generator,
        ecosystem_discovery,
        game_design,
        geckolib_generator,
        generator,
        mineflayer_bridge,
        minecraft_knowledge_contract,
        platform_central_ai_contract,
        platform_planning_contract,
        platform_resolver,
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
    from .colab_auto_platform_contract import install as install_colab_auto_platform
    from .mod_scope_contract import install as install_mod_scope
    from .parallel_platform_rag_contract import install as install_parallel_platform_rag
    from .platform_central_ai_contract import install as install_platform_central_ai
    from .platform_custom_coder_contract import install as install_platform_custom_coder
    from .platform_ecosystem_contract import install as install_platform_ecosystem
    from .platform_generation_contract import install as install_platform_generation
    from .platform_live_execution_contract import install as install_live_execution
    from .platform_live_rag_contract import install as install_platform_live_rag
    from .platform_planning_contract import install as install_platform_planning
    from .platform_prompt_contract import install as install_platform_prompts
    from .platform_repair_target_contract import install as install_platform_repair
    from .platform_runtime_contract import install as install_platform_runtime
    from .platform_selection_efficiency_contract import install as install_platform_selection_efficiency
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
    install_platform_planning(
        game_design_module=game_design,
        complete_planner_module=complete_planner,
        central_research_module=central_research,
        retrieval_module=retrieval,
        technology_module=technology_radar,
    )
    install_platform_live_rag(
        retrieval_module=retrieval,
        platform_planning_module=platform_planning_contract,
    )
    install_platform_technology(technology_radar)
    install_platform_ecosystem(ecosystem_discovery, complete_planner)
    install_platform_prompts(complete_planner)
    install_platform_selection_efficiency(
        resolver_module=platform_resolver,
        central_contract_module=platform_central_ai_contract,
    )
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
    install_parallel_platform_rag(
        complete_planner_module=complete_planner,
        central_module=central_research,
        retrieval_module=retrieval,
    )
    if os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        install_colab_auto_platform(game_design)
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
    """Compose planner parsing, pagination, resume and efficiency policies explicitly."""
    from . import (
        agentic_optimization_contract,
        audio_generator,
        complete_orchestrator,
        complete_orchestrator_services,
        complete_planner,
        planner_incremental_repair_contract,
        planner_json_runtime_contract,
        planner_pagination_safety_contract,
        work_graph,
    )
    from .agentic_search_efficiency_contract import install as install_agentic_search_efficiency
    from .asset_resume_efficiency_contract import install as install_asset_resume_efficiency
    from .audio_resume_efficiency_contract import install as install_audio_resume_efficiency
    from .execution_efficiency_contract import install as install_execution_efficiency
    from .planner_checkpoint_journal_contract import install as install_checkpoint_journal
    from .planner_incremental_repair_contract import install as install_incremental_repair
    from .planner_incremental_resume_contract import install as install_incremental_resume
    from .planner_json_runtime_contract import install as install_planner_json_runtime
    from .planner_module_identity_contract import install as install_planner_module_identity
    from .planner_outline_identity_contract import install as install_planner_outline_identity
    from .planner_outline_prompt_contract import install as install_planner_outline_prompt
    from .planner_pagination_safety_contract import install as install_planner_pagination_safety
    from .planner_parser_safety_contract import install as install_planner_parser_safety
    from .planner_production_page_contract import install as install_planner_production_page
    from .planner_strict_json_contract import install as install_planner_strict_json
    from .production_stream_efficiency_contract import install as install_production_stream_efficiency

    install_planner_json_runtime(complete_planner)
    install_planner_strict_json(planner_json_runtime_contract)
    install_planner_outline_prompt(planner_json_runtime_contract)
    install_incremental_repair(planner_json_runtime_contract)
    install_checkpoint_journal(planner_incremental_repair_contract)
    install_agentic_search_efficiency(agentic_optimization_contract)
    install_asset_resume_efficiency(complete_orchestrator_services)
    install_audio_resume_efficiency(audio_generator)
    complete_orchestrator.synthesize_audio_files = audio_generator.synthesize_audio_files
    install_production_stream_efficiency(complete_planner)
    install_execution_efficiency(work_graph_module=work_graph)
    install_incremental_resume(planner_incremental_repair_contract)
    install_planner_parser_safety(complete_planner)
    install_planner_module_identity(complete_planner)
    install_planner_pagination_safety(complete_planner)
    install_planner_production_page(complete_planner)
    install_planner_outline_identity(planner_pagination_safety_contract)


def _install_architecture_contracts() -> None:
    """Install deterministic-control and narrow-agent architecture policies once."""
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
    from .work_graph_state_transition_contract import install as install_work_graph_state_transitions

    install_build_input_scope(validation_execution_contract)
    install_atomic_efficiency(atomic_requirement_contract)
    install_atomic_routes(atomic_requirement_contract, production_contract)
    install_reviewer_role(atomic_requirement_contract)
    install_atomic_requirements(complete_planner, complete_orchestrator)
    install_atomic_planner_policy(atomic_requirement_contract, complete_planner)
    install_atomic_execution(atomic_requirement_contract, complete_orchestrator)
    install_atomic_quality(atomic_requirement_contract, quality_evidence, complete_orchestrator)
    install_atomic_playtest(atomic_requirement_contract, quality_evidence, complete_orchestrator)
    install_repair_diagnostics(repair_engine, validation_execution_contract)
    install_orchestrator_jdt_gate(complete_orchestrator)
    install_clean_room(complete_orchestrator, quality_evidence, validation_execution_contract)
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
    """Apply wrappers that must sit outside all method-replacing policy layers."""
    from . import (
        audio_generator,
        complete_orchestrator,
        model_router,
        production_tools,
        runner,
        scheduler_parallel_safety_contract,
        validation_execution_contract,
        work_graph,
    )
    from .llama_parallel_runtime_contract import install as install_llama_parallel_runtime
    from .max_efficiency_runtime_contract import enhance_runtime as enhance_max_efficiency_runtime
    from .parallel_result_determinism_contract import install as install_parallel_result_determinism
    from .production_tool_parallel_contract import install as install_production_tool_parallel_safety
    from .runner_parallel_validation_contract import install as install_runner_parallel_validation
    from .scheduler_claim_fencing_contract import install as install_scheduler_claim_fencing
    from .scheduler_parallel_safety_contract import install as install_scheduler_parallel_safety
    from .scheduler_poll_efficiency_contract import install as install_scheduler_poll_efficiency

    install_scheduler_parallel_safety(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_scheduler_poll_efficiency(work_graph)
    install_llama_parallel_runtime(model_router, scheduler_parallel_safety_contract)
    enhance_max_efficiency_runtime(
        work_graph_module=work_graph,
        scheduler_module=scheduler_parallel_safety_contract,
    )
    install_scheduler_claim_fencing(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_production_tool_parallel_safety(production_tools)
    install_runner_parallel_validation(runner_module=runner, validation_module=validation_execution_contract)
    install_parallel_result_determinism(
        orchestrator_module=complete_orchestrator,
        audio_generator_module=audio_generator,
    )


def _install_public_boundary_contracts() -> None:
    """Install the live MCP/release boundary owners last."""
    from . import mcp_tools, production_tools
    from .platform_mcp_contract import install as install_platform_mcp
    from .platform_release_contract import install as install_platform_release

    install_platform_mcp(mcp_tools, production_tools)
    install_platform_release(mcp_tools)


def _install_post_bootstrap_contracts() -> None:
    """Install wrappers that must observe the fully composed runtime."""
    from . import (
        agentic_optimization_contract,
        agentic_pre_design_rag,
        agentic_research_game_design,
        central_research,
        ecosystem_discovery,
        model_router,
        production_tools,
        repair_engine,
        research_coordinator,
        work_graph,
    )
    from .active_repair_verifier_contract import install as install_active_repair_verifier
    from .agent_security_contract import install as install_agent_security
    from .bottleneck_elimination_contract import install as install_bottleneck_elimination
    from .causal_tool_frontier_contract import install as install_causal_tool_frontier
    from .long_run_resilience_contract import install as install_long_run_resilience
    from .minecraft_mcp_evidence_contract import install as install_minecraft_mcp_evidence
    from .planning_stall_guard_contract import install as install_planning_stall_guard
    from .research_bottleneck_runtime import install as install_research_bottleneck_runtime
    from .small_model_compacting_adapter import CompactingAdapter
    from .small_model_hybrid_search_contract import install as install_small_model_hybrid_search
    from .small_model_max_agent_contract import install as install_small_model_max_agent
    from .small_model_relation_index_contract import install as install_small_model_relation_index
    from .small_model_research_contract import install as install_small_model_research
    from .small_model_tool_guard_contract import install as install_small_model_tool_guard
    from .temporary_skill_contract import install as install_temporary_skill

    if not hasattr(central_research, "_bounded_text"):
        def _full_research_text(value: str, *, field: str = "research text") -> str:
            del field
            return value

        central_research._bounded_text = _full_research_text

    install_small_model_research()
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
        def _generate_with_compaction(
            self,
            *,
            adapter,
            request,
            runtime,
            stage,
            role,
        ):
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

    research_coordinator.discover_seed_bundle = ecosystem_discovery.discover_seed_bundle
    install_minecraft_mcp_evidence()
    install_bottleneck_elimination()
    install_research_bottleneck_runtime()
    install_long_run_resilience()
    install_planning_stall_guard()
