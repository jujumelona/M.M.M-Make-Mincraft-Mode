from __future__ import annotations

"""Single late-finalization owner for the fully composed runtime.

Pre-design retrieval is a direct host-owned pipeline and is intentionally absent from
this runtime mutation phase.
"""

import threading

_FINALIZE_LOCK = threading.RLock()
_FINALIZED = False
_FINALIZING = False


def finalize_runtime() -> None:
    global _FINALIZED, _FINALIZING
    if _FINALIZED:
        return
    with _FINALIZE_LOCK:
        if _FINALIZED:
            return
        if _FINALIZING:
            raise RuntimeError(
                "runtime finalization was re-entered or a prior late-finalization "
                "attempt failed after partial mutation; restart the process before retrying"
            )
        _FINALIZING = True

        from . import (
            agent_capability_context,
            agent_tool_runtime,
            central_intelligence_amplifier,
            complete_orchestrator,
            complete_orchestrator_support,
            complete_spec,
            custom_module_generator,
            execution_feedback_replan_contract,
            external_agent_bridge,
            external_mcp_router,
            external_procedural_skill_contract,
            host_grounding,
            llama_server_autotune,
            llama_server_runtime_tuning,
            llama_tuning_pipeline,
            mcp_transport_pool,
            model_router,
            model_tool_aliases,
            parallel_runtime_contract,
            planner_template_schema,
            production_contract,
            production_tools,
            progress_aware_tool_loop,
            quality_evidence,
            reference_source_research,
            repository_grounding,
            research_rag_performance,
            retrieval_cpu_budget_contract,
            small_model_hybrid_search_contract,
            small_model_max_agent_contract,
            work_graph,
        )
        from .adaptive_retrieval_contract import (
            _install_repository_grounding as install_repository_grounding,
        )
        from .agent_observation_determinism import install as install_observation_determinism
        from .agent_routing_intent_contract import install as install_routing_intent
        from .authored_scope_research_contract import install as install_authored_scope_research
        from .central_atomic_generation_contract import install as install_central_atomic_generation
        from .coder_mutation_authority_contract import assert_installed as assert_coder_mutation_authority
        from .coder_mutation_authority_contract import install as install_coder_mutation_authority
        from .context_budget_preflight import run_context_budget_preflight
        from .deep_design_execution_contract import install as install_deep_design_execution
        from .design_resolution_provenance_contract import install_design_resolution_provenance_contract
        from .direct_task_mutation_authority_contract import install as install_direct_task_mutation_authority
        from .evidence_first_pipeline_contract import install as install_evidence_first_pipeline
        from .evidence_obligation_contract import install_evidence_obligation_contract
        from .execution_feedback_exception_scope_contract import install as install_execution_feedback_exception_scope
        from .execution_feedback_owner_precision_contract import install as install_execution_feedback_owner_precision
        from .external_mcp_binding_concurrency_contract import install as install_external_mcp_binding_concurrency
        from .external_mcp_binding_contract import install as install_external_mcp_binding
        from .fabric_immutable_rebind_contract import install as install_fabric_immutable_rebind
        from .generation_boundary_reconciliation import install as install_generation_boundary_reconciliation
        from .generation_concurrency_safety import install as install_generation_safety
        from .implementation_kind_boundary_contract import install as install_implementation_kind_boundary
        from .immutable_platform_execution_contract import install as install_immutable_platform_execution
        from .llama_finish_reason_contract import install as install_llama_finish_reason
        from .llama_mtp_cache_policy import install as install_llama_mtp_cache_policy
        from .llama_native_context_authority_contract import install as install_llama_native_context_authority
        from .llama_server_response_resilience import install as install_llama_server_response_resilience
        from .mcp_child_trace_contract import install as install_mcp_child_trace
        from .mcp_schema_integrity_contract import install as install_mcp_schema_integrity
        from .mcp_transport_pool import install_agent_mcp_transport_pool
        from .model_adapters import llama_cpp_adapter
        from .model_output_atomicity_contract import assert_installed as assert_model_output_atomicity
        from .model_output_atomicity_contract import install as install_model_output_atomicity
        from .model_prefetch_resilience import install as install_prefetch_resilience
        from .model_tool_alias_permission_policy import install as install_model_tool_alias_permissions
        from .mutation_authority_final_guard import assert_installed as assert_mutation_authority_final_guard
        from .mutation_authority_final_guard import install as install_mutation_authority_final_guard
        from .planir_mutation_authority_contract import install as install_planir_mutation_authority
        from .planner_design_readiness_contract import install as install_planner_design_readiness
        from .procedural_skill_identity_contract import install as install_procedural_skill_identity
        from .production_boundary_contract import install_production_boundary_contract
        from .quality_public_acceptance_view_contract import install as install_quality_public_acceptance_view
        from .reference_query_parallelism_contract import install as install_reference_query_parallelism
        from .requirement_branch_scope_contract import install_requirement_branch_scope_contract
        from .retrieval_model_residency import install as install_retrieval_residency
        from .runtime_hot_path_contract import assert_installed as assert_runtime_hot_paths
        from .runtime_hot_path_contract import install as install_runtime_hot_paths
        from .runtime_live_path_preflight import run_runtime_live_path_preflight
        from .runtime_preflight import run_runtime_preflight
        from .runtime_regression_reconciliation import install as install_runtime_regression_reconciliation
        from .runtime_wrapper_integrity import verify_installed_wrappers
        from .small_model_atomic_coder_execution import assert_installed as assert_small_model_atomic_coder
        from .small_model_atomic_coder_execution import install as install_small_model_atomic_coder
        from .small_model_task_capsule_contract import assert_installed as assert_small_model_task_capsule
        from .small_model_task_capsule_contract import install as install_small_model_task_capsule
        from .small_model_write_scope_enforcement import assert_installed as assert_small_model_write_scope
        from .small_model_write_scope_enforcement import install as install_small_model_write_scope
        from .source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA
        from .task_artifact_contract import install_task_artifact_contract
        from .tool_schema_ownership_contract import install as install_tool_schema_ownership
        from .tool_validation_surface_contract import install as install_tool_validation_surface
        from .verifier_receipt_truth_contract import install as install_verifier_receipt_truth
        from .work_graph_receipt_integrity_contract import install as install_work_graph_receipt_integrity

        install_agent_mcp_transport_pool()
        install_mcp_schema_integrity(agent_tool_runtime, external_agent_bridge, external_mcp_router)
        install_external_mcp_binding(external_agent_bridge, external_mcp_router)
        install_external_mcp_binding_concurrency(external_agent_bridge)
        install_runtime_hot_paths(mcp_transport_pool_module=mcp_transport_pool, external_mcp_router_module=external_mcp_router, research_rag_performance_module=research_rag_performance)
        install_mcp_child_trace(mcp_transport_pool)
        install_prefetch_resilience(parallel_runtime_module=parallel_runtime_contract)
        install_observation_determinism(agent_tool_runtime_module=agent_tool_runtime)
        install_procedural_skill_identity(external_procedural_skill_contract)
        install_tool_schema_ownership(agent_tool_runtime, expected_parameters={"apply_source_edit": SOURCE_EDIT_SCHEMA})
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        install_planir_mutation_authority(progress_aware_tool_loop)
        install_retrieval_residency(model_router_module=model_router)

        retrieval_cpu_budget_contract._install_live_hybrid_budget(small_model_hybrid_search_contract)
        retrieval_cpu_budget_contract._install_production_tool_budget(production_tools)
        if not retrieval_cpu_budget_contract._dense_opted_in():
            repository_grounding._explore_with_degraded_fallback = retrieval_cpu_budget_contract._lexical_repository_exploration
        install_repository_grounding()

        install_model_tool_alias_permissions(agent_capability_context, model_tool_aliases)
        install_llama_mtp_cache_policy(llama_server_autotune, llama_server_runtime_tuning)
        install_llama_native_context_authority(llama_server_autotune, llama_tuning_pipeline)
        install_tool_validation_surface()
        install_llama_finish_reason(llama_cpp_adapter)
        install_llama_server_response_resilience(llama_cpp_adapter)
        install_work_graph_receipt_integrity(work_graph)
        install_verifier_receipt_truth(work_graph)
        install_reference_query_parallelism(reference_source_research)
        install_central_atomic_generation(central_intelligence_amplifier)

        install_evidence_first_pipeline()
        install_planner_design_readiness()
        install_deep_design_execution()
        install_evidence_obligation_contract()
        install_authored_scope_research()
        install_requirement_branch_scope_contract()
        install_task_artifact_contract()
        install_design_resolution_provenance_contract()
        install_production_boundary_contract()
        install_quality_public_acceptance_view(production_contract, quality_evidence)
        install_implementation_kind_boundary(complete_spec_module=complete_spec, support_module=complete_orchestrator_support, orchestrator_module=complete_orchestrator, template_module=planner_template_schema)
        install_execution_feedback_exception_scope(execution_feedback_replan_contract)
        install_execution_feedback_owner_precision(execution_feedback_replan_contract)
        execution_feedback_replan_contract.install(orchestrator_module=complete_orchestrator, work_graph_module=work_graph)

        install_immutable_platform_execution()
        install_fabric_immutable_rebind()
        install_runtime_regression_reconciliation()
        install_generation_boundary_reconciliation()

        install_small_model_task_capsule()
        assert_small_model_task_capsule()
        install_coder_mutation_authority()
        assert_coder_mutation_authority()
        install_small_model_write_scope(custom_module_generator_module=custom_module_generator, host_grounding_module=host_grounding)
        assert_small_model_write_scope(custom_module_generator_module=custom_module_generator, host_grounding_module=host_grounding)
        install_small_model_atomic_coder(
            custom_module_generator_module=custom_module_generator,
            model_router_module=model_router,
        )
        assert_small_model_atomic_coder(
            custom_module_generator_module=custom_module_generator,
            model_router_module=model_router,
        )

        install_model_output_atomicity(model_router_module=model_router)
        assert_model_output_atomicity(model_router_module=model_router)

        # These two contracts are intentionally finalized last. The direct-task bridge
        # carries the task capsule authority out-of-band, and the final guard freezes
        # that exact target before any retrieval evidence can merge into run state.
        install_direct_task_mutation_authority(
            custom_module_generator_module=custom_module_generator,
            loop_module=progress_aware_tool_loop,
        )
        install_mutation_authority_final_guard(progress_aware_tool_loop)
        assert_mutation_authority_final_guard(progress_aware_tool_loop)

        assert_runtime_hot_paths(mcp_transport_pool_module=mcp_transport_pool, external_mcp_router_module=external_mcp_router, research_rag_performance_module=research_rag_performance)
        verify_installed_wrappers()
        run_context_budget_preflight()
        run_runtime_live_path_preflight()
        run_runtime_preflight()

        _FINALIZED = True
        _FINALIZING = False


__all__ = ["finalize_runtime"]