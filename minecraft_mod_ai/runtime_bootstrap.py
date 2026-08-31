from __future__ import annotations

"""Ordered package runtime bootstrap with one explicit integration path."""

from threading import RLock

from . import runtime_contract_composer as _contract_composer

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
    from .reuse_asset_upgrade_contract import (
        install_postbootstrap,
        install_prebootstrap,
    )
    from .runtime_wrapper_integrity import verify_installed_wrappers

    _contract_composer.compose_contract_stages(
        owner_name="package-runtime-bootstrap",
        state_owner=_contract_composer,
        stages=(
            _contract_composer.ContractStage("prebootstrap", install_prebootstrap),
            _contract_composer.ContractStage("core", _install_core_contracts),
            _contract_composer.ContractStage(
                "model-runtime", _install_model_runtime_contracts
            ),
            _contract_composer.ContractStage("validation", _install_validation_contracts),
            _contract_composer.ContractStage("generation", _install_generation_contracts),
            _contract_composer.ContractStage("platform", _install_platform_contracts),
            _contract_composer.ContractStage("planner", _install_planner_contracts),
            _contract_composer.ContractStage(
                "architecture", _install_architecture_contracts
            ),
            _contract_composer.ContractStage("late-safety", _install_late_safety_contracts),
            _contract_composer.ContractStage(
                "public-boundary", _install_public_boundary_contracts
            ),
            _contract_composer.ContractStage(
                "post-bootstrap", _install_post_bootstrap_contracts
            ),
            _contract_composer.ContractStage("postbootstrap", install_postbootstrap),
            _contract_composer.ContractStage("integrity", verify_installed_wrappers),
        ),
    )


def _install_core_contracts() -> None:
    from . import runner, spec, work_graph
    from .runner_lock_contract import install as install_runner_lock
    from .toolchain_contract import install as install_toolchain
    from .work_graph_mutation_contract import install as install_work_graph_mutation

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
    from .forced_tool_execution_contract import install as install_forced_tool_execution
    from .gpu_resource_contract import install as install_gpu_resource
    from .llama_completion_liveness_contract import (
        install as install_completion_liveness,
    )
    from .llama_context_safety_contract import install as install_context_safety
    from .llama_generation_budget import install as install_llama_generation_budget
    from .llama_stream_efficiency_contract import (
        install as install_llama_stream_efficiency,
    )
    from .llama_tuning_pipeline import install_native_llama_tuning_pipeline
    from .model_adapters import llama_cpp_adapter, openai_compatible
    from .model_runtime_performance import install as install_model_runtime_performance

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
    install_llama_generation_budget(llama_server_hardware_policy)
    install_llama_stream_efficiency(llama_server_hardware_policy)
    install_completion_liveness(llama_stream_efficiency_contract, llama_cpp_adapter)
    install_context_safety(model_context_budget)
    install_forced_tool_execution(
        openai_compatible_module=openai_compatible,
        llama_cpp_module=llama_cpp_adapter,
    )
    start_colab_prefetch(model_registry)


def _install_validation_contracts() -> None:
    from . import java_lsp, repair_engine, runner, validation_execution_contract
    from .java_lsp_process_safety_contract import (
        install as install_java_lsp_process_safety,
    )
    from .research_validation_fingerprint_performance import (
        harden as harden_validation_fingerprints,
    )
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
    from .deterministic_minecraft_content_contract import (
        install as install_deterministic_minecraft_content,
    )
    from .extended_registration_contract import install as install_extended_registration
    from .performance_final_contract import install as install_performance_contract
    from .performance_final_tuning import install as install_performance_tuning
    from .project_index_manifest_efficiency_contract import (
        install as install_project_index_manifest_efficiency,
    )
    from .project_manifest_hash_efficiency_contract import (
        install as install_manifest_hash_efficiency,
    )

    install_extended_registration(extended_content_generator)
    install_deterministic_minecraft_content(extended_content_generator)
    install_project_index_manifest_efficiency(project_index)
    install_performance_tuning(performance_final_contract)
    install_manifest_hash_efficiency(complete_orchestrator, project_index)
    install_performance_contract(
        complete_orchestrator,
        custom_module_generator,
        source_patch,
    )


def _install_platform_contracts() -> None:
    from . import (
        central_research,
        complete_orchestrator,
        complete_planner,
        complete_spec,
        custom_module_generator,
        game_design,
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
    from .minecraft_domain_correctness_contract import (
        install as install_minecraft_domain_correctness,
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
    from .proposal_deserialization_contract import (
        install as install_proposal_deserialization,
    )
    from .source_patch_precondition_contract import (
        install as install_source_patch_preconditions,
    )
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
    install_platform_central_ai(
        game_design_module=game_design,
        complete_planner_module=complete_planner,
    )
    install_mod_scope(complete_spec, complete_planner)
    install_platform_custom_coder(custom_module_generator)
    install_source_patch_preconditions(custom_module_generator)
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
    """Install planner-independent efficiency policies only."""
    from . import agentic_optimization_contract, complete_orchestrator_services
    from .agentic_search_efficiency_contract import (
        install as install_agentic_search_efficiency,
    )
    from .asset_resume_efficiency_contract import (
        install as install_asset_resume_efficiency,
    )

    install_agentic_search_efficiency(agentic_optimization_contract)
    install_asset_resume_efficiency(complete_orchestrator_services)


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
    from .atomic_playtest_evidence_contract import install as install_atomic_playtest
    from .atomic_quality_binding_contract import install as install_atomic_quality
    from .build_input_scope_contract import install as install_build_input_scope
    from .clean_room_verification_contract import install as install_clean_room
    from .coder_max_efficiency_contract import install_coder_max_efficiency
    from .custom_generation_search_contract import (
        install as install_custom_generation_search,
    )
    from .repair_diagnostics_contract import install as install_repair_diagnostics
    from .repair_memory_budget_contract import install as install_repair_memory_budget
    from .required_gate_compatibility_contract import (
        install as install_gate_compatibility,
    )
    from .semantic_reviewer_role_contract import install as install_reviewer_role
    from .visual_acceptance_scope_contract import install as install_visual_scope
    from .work_graph_state_transition_contract import (
        install as install_work_graph_state_transitions,
    )

    install_build_input_scope(validation_execution_contract)
    install_atomic_efficiency(atomic_requirement_contract)
    install_atomic_routes(atomic_requirement_contract, production_contract)
    install_reviewer_role(atomic_requirement_contract)
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
    install_repair_diagnostics(repair_engine)
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
    from .llama_parallel_runtime_contract import (
        install as install_llama_parallel_runtime,
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
    from .scheduler_claim_fencing_contract import (
        install as install_scheduler_claim_fencing,
    )
    from .scheduler_parallel_safety_contract import (
        install as install_scheduler_parallel_safety,
    )

    install_scheduler_parallel_safety(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
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
        qwen_agent_family_contract,
        repair_engine,
        work_graph,
    )
    from .active_repair_verifier_contract import (
        install as install_active_repair_verifier,
    )
    from .adaptive_retrieval_contract import install as install_adaptive_retrieval
    from .agent_security_contract import install as install_agent_security
    from .long_run_resilience_contract import install as install_long_run_resilience
    from .minecraft_mcp_evidence_contract import (
        install as install_minecraft_mcp_evidence,
    )
    from .pre_design_external_source_contract import (
        install as install_pre_design_external_source,
    )
    from .research_bottleneck_runtime import (
        install as install_research_bottleneck_runtime,
    )
    from .small_model_hybrid_search_contract import (
        install as install_small_model_hybrid_search,
    )
    from .small_model_max_agent_contract import install as install_small_model_max_agent
    from .small_model_relation_index_contract import (
        install as install_small_model_relation_index,
    )
    from .small_model_research_extensions_contract import (
        install as install_small_model_research_extensions,
    )
    from .small_model_retrieval_efficiency_contract import (
        install as install_small_model_retrieval_efficiency,
    )
    from .temporary_skill_contract import install as install_temporary_skill
    from .unified_trajectory_memory_contract import (
        install as install_unified_trajectory_memory,
    )

    # The base pre-design retriever owns the approved query set and local evidence;
    # this layer performs the missing bounded external search -> source-body acquisition.
    install_pre_design_external_source(agentic_pre_design_rag)
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
    install_small_model_relation_index(production_tools)
    install_small_model_hybrid_search(production_tools)
    install_small_model_retrieval_efficiency()
    install_temporary_skill(
        model_router_module=model_router,
        work_graph_module=work_graph,
        repair_module=repair_engine,
    )
    install_active_repair_verifier(agentic_optimization_contract)
    install_minecraft_mcp_evidence()
    install_research_bottleneck_runtime()
    install_long_run_resilience()
    qwen_agent_family_contract.install()
    install_small_model_research_extensions()
    install_unified_trajectory_memory()
    install_adaptive_retrieval(model_router)
