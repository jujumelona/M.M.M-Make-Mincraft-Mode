from __future__ import annotations

"""Ordered package runtime bootstrap with one explicit integration path.

Cross-cutting contracts stay isolated and testable, while package initialization
owns their composition. No installer re-enters another package bootstrap and no
runtime contract is intentionally installed twice.
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


def _install_core_contracts() -> None:
    """Install base spec, validation, planning-scope and work-graph semantics."""
    from . import complete_planner, complete_spec, runner, spec, validator, work_graph
    from .mod_scope_contract import install as install_mod_scope
    from .runner_lock_contract import install as install_runner_lock
    from .toolchain_contract import install as install_toolchain
    from .validator_boss_contract import install as install_validator_boss
    from .work_graph_mutation_contract import install as install_work_graph_mutation

    install_toolchain(spec, runner)
    install_runner_lock(runner)
    install_validator_boss(validator)
    install_mod_scope(complete_spec, complete_planner)
    install_work_graph_mutation(work_graph)


def _install_model_runtime_contracts() -> None:
    """Install local model ownership and native llama runtime policy once."""
    from . import (
        complete_planner,
        llama_server_autotune,
        llama_server_hardware_policy,
        llama_server_runtime_tuning,
        model_registry,
    )
    from .colab_prefetch_bootstrap import start as start_colab_prefetch
    from .gpu_resource_contract import install as install_gpu_resource
    from .image_runtime_residency import install as install_image_runtime_residency
    from .llama_cache_reuse_efficiency_contract import install as install_llama_cache_reuse
    from .llama_server_efficiency_contract import install as install_llama_efficiency
    from .llama_server_hardware_policy import install as install_llama_hardware
    from .llama_server_runtime_tuning import install as install_llama_runtime_tuning
    from .llama_stream_efficiency_contract import install as install_llama_stream_efficiency
    from .model_runtime_performance import install as install_model_runtime_performance
    from .parallel_runtime_contract import install as install_parallel_runtime

    install_gpu_resource(model_registry)
    install_model_runtime_performance()
    install_llama_hardware(llama_server_autotune)
    install_llama_efficiency(llama_server_autotune, llama_server_hardware_policy)
    install_llama_runtime_tuning(llama_server_autotune)
    install_llama_cache_reuse(
        llama_server_autotune,
        llama_server_hardware_policy,
        llama_server_runtime_tuning,
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
    """Bind planning, generation, runtime and quality to one Minecraft target."""
    from . import (
        central_research,
        complete_orchestrator,
        complete_planner,
        complete_spec,
        custom_module_generator,
        external_mcp_router,
        game_design,
        geckolib_generator,
        generator,
        mineflayer_bridge,
        platform_central_ai_contract,
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
    from .external_mcp_bridge_safety_contract import (
        install as install_external_mcp_bridge_safety,
    )
    from .platform_central_ai_contract import install as install_platform_central_ai
    from .platform_custom_coder_contract import install as install_platform_custom_coder
    from .platform_generation_contract import install as install_platform_generation
    from .platform_live_execution_contract import install as install_live_execution
    from .platform_planning_contract import install as install_platform_planning
    from .platform_repair_target_contract import install as install_platform_repair
    from .platform_runtime_contract import install as install_platform_runtime
    from .platform_selection_efficiency_contract import (
        install as install_platform_selection_efficiency,
    )
    from .platform_specialized_generator_contract import (
        install as install_specialized_generator_guards,
    )
    from .platform_validation_contract import install as install_platform_validation
    from .proposal_deserialization_contract import install as install_proposal_deserialization
    from .system_quality_contract import install as install_system_quality

    install_platform_runtime(
        orchestrator_module=complete_orchestrator,
        runtime_manager_module=runtime_manager,
        mineflayer_module=mineflayer_bridge,
    )
    install_external_mcp_bridge_safety(external_mcp_router)
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
    install_platform_selection_efficiency(
        resolver_module=platform_resolver,
        central_contract_module=platform_central_ai_contract,
    )
    install_platform_central_ai(
        game_design_module=game_design,
        complete_planner_module=complete_planner,
    )
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
    from .production_stream_resume_contract import install as install_production_stream_resume

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
    install_production_stream_resume(complete_planner)
    install_execution_efficiency(
        complete_planner_module=complete_planner,
        work_graph_module=work_graph,
    )
    install_incremental_resume(planner_incremental_repair_contract)

    install_planner_parser_safety(complete_planner)
    install_planner_module_identity(complete_planner)
    install_planner_pagination_safety(complete_planner)
    # Pagination safety replaces _expand_one_production_batch. Adaptive page width is
    # the final authority for that method and therefore installs immediately after it.
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
    from .scheduler_fairness_contract import install as install_scheduler_fairness
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
    install_scheduler_fairness(work_graph)
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
    from .scheduler_parallel_safety_contract import (
        install as install_scheduler_parallel_safety,
    )
    from .scheduler_poll_efficiency_contract import (
        install as install_scheduler_poll_efficiency,
    )

    install_scheduler_parallel_safety(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_scheduler_poll_efficiency(work_graph)
    install_llama_parallel_runtime(model_router, scheduler_parallel_safety_contract)
    install_scheduler_claim_fencing(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
    install_production_tool_parallel_safety(production_tools)
    install_runner_parallel_validation(
        runner_module=runner,
        validation_module=validation_execution_contract,
    )
    install_parallel_result_determinism(
        audio_generator_module=audio_generator,
        orchestrator_module=complete_orchestrator,
    )


def _install_public_boundary_contracts() -> None:
    """Bind MCP and Python API surfaces after the runtime implementation is final."""
    from . import (
        api,
        complete_orchestrator,
        complete_planner,
        custom_module_generator,
        external_mcp_router,
        mcp_tools,
        minecraft_mcp_repair_batch_contract,
        minecraft_mcp_runtime_helper_contract,
        plan_render,
        production_tools,
        repair_engine,
        runtime_manager,
        skill_catalog,
    )
    from .external_mcp_target_validation_contract import (
        install as install_mcp_target_validation,
    )
    from .mcp_repair_diagnostic_shape_contract import (
        install as install_mcp_repair_diagnostic_shape,
    )
    from .minecraft_mcp_federation_contract import install as install_mcp_federation
    from .minecraft_mcp_repair_batch_contract import install as install_mcp_repair_batch
    from .minecraft_mcp_runtime_contract import install as install_mcp_runtime
    from .minecraft_mcp_runtime_helper_contract import install as install_runtime_helpers
    from .platform_api_contract import install as install_platform_api
    from .platform_mcp_contract import install as install_platform_mcp
    from .platform_release_contract import install as install_platform_release
    from .platform_skill_policy_contract import install as install_skill_policy
    from .runtime_helper_json_deadline_contract import (
        install as install_runtime_helper_json_deadline,
    )

    install_platform_mcp(mcp_tools, production_tools)
    install_platform_release(mcp_tools)
    install_platform_api(api, plan_render)

    # These policies used to be nested inside platform_api_contract. Keep their
    # original public-boundary timing while making composition explicit and unique.
    install_runtime_helpers(runtime_manager)
    install_runtime_helper_json_deadline(minecraft_mcp_runtime_helper_contract)
    install_mcp_target_validation(external_mcp_router)
    install_mcp_runtime(complete_orchestrator)
    install_mcp_federation(
        complete_planner_module=complete_planner,
        custom_module_generator_module=custom_module_generator,
        repair_engine_module=repair_engine,
        mcp_tools_module=mcp_tools,
    )
    install_mcp_repair_batch(repair_engine)
    install_mcp_repair_diagnostic_shape(minecraft_mcp_repair_batch_contract)
    install_skill_policy(skill_catalog)


__all__ = ["initialize_runtime", "runtime_initialized"]
