from __future__ import annotations

"""Single late-finalization owner for the fully composed runtime.

``runtime_bootstrap.initialize_runtime`` installs the ordered core/late/post-bootstrap
contracts. Integrations that intentionally happen only *after* that sequence live here.
Request semantics and retrieval-query planning are host-scoped text calls owned by
``planning_authority`` and therefore are deliberately absent from this mutation phase.

Late finalization is process-lifetime composition just like bootstrap. If any installer
fails after mutating a callable, replaying the prefix is not transactionally safe. A
failed or recursively re-entered finalization therefore poisons this process; a clean
process restart is the only supported retry boundary.
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
                "attempt failed after partial mutation; restart the process before "
                "retrying"
            )
        _FINALIZING = True

        from . import (
            agent_capability_context,
            agent_tool_runtime,
            agentic_pre_design_rag,
            complete_orchestrator,
            complete_orchestrator_support,
            complete_spec,
            execution_feedback_replan_contract,
            external_agent_bridge,
            external_mcp_router,
            external_procedural_skill_contract,
            llama_server_autotune,
            llama_server_runtime_tuning,
            mcp_transport_pool,
            model_router,
            model_tool_aliases,
            parallel_runtime_contract,
            planner_template_schema,
            production_contract,
            quality_evidence,
            repository_grounding,
            research_rag_performance,
            small_model_max_agent_contract,
            work_graph,
        )
        from .agent_observation_determinism import (
            install as install_observation_determinism,
        )
        from .agent_routing_intent_contract import install as install_routing_intent
        from .authored_scope_research_contract import install as install_authored_scope_research
        from .context_budget_preflight import run_context_budget_preflight
        from .deep_design_execution_contract import install as install_deep_design_execution
        from .design_resolution_provenance_contract import (
            install_design_resolution_provenance_contract,
        )
        from .evidence_first_pipeline_contract import (
            install as install_evidence_first_pipeline,
        )
        from .evidence_obligation_contract import install_evidence_obligation_contract
        from .evidence_task_receipt_contract import (
            install as install_evidence_task_receipts,
        )
        from .execution_feedback_exception_scope_contract import (
            install as install_execution_feedback_exception_scope,
        )
        from .execution_feedback_owner_precision_contract import (
            install as install_execution_feedback_owner_precision,
        )
        from .external_mcp_binding_concurrency_contract import (
            install as install_external_mcp_binding_concurrency,
        )
        from .external_mcp_binding_contract import (
            install as install_external_mcp_binding,
        )
        from .generation_concurrency_safety import install as install_generation_safety
        from .implementation_kind_boundary_contract import (
            install as install_implementation_kind_boundary,
        )
        from .llama_finish_reason_contract import install as install_llama_finish_reason
        from .llama_mtp_cache_policy import install as install_llama_mtp_cache_policy
        from .llama_server_response_resilience import (
            install as install_llama_server_response_resilience,
        )
        from .mcp_schema_integrity_contract import (
            install as install_mcp_schema_integrity,
        )
        from .mcp_transport_pool import install_agent_mcp_transport_pool
        from .model_adapters import llama_cpp_adapter
        from .model_prefetch_resilience import install as install_prefetch_resilience
        from .model_tool_alias_permission_policy import (
            install as install_model_tool_alias_permissions,
        )
        from .planner_design_readiness_contract import (
            install as install_planner_design_readiness,
        )
        from .planner_graph_integrity_contract import (
            install as install_planner_graph_integrity,
        )
        from .planner_requirement_traceability_contract import (
            install as install_planner_requirement_traceability,
        )
        from .prefill_calibration_strictness_contract import (
            install as install_prefill_calibration_strictness,
        )
        from .procedural_skill_identity_contract import (
            install as install_procedural_skill_identity,
        )
        from .production_boundary_contract import install_production_boundary_contract
        from .quality_public_acceptance_view_contract import (
            install as install_quality_public_acceptance_view,
        )
        from .requirement_branch_scope_contract import install_requirement_branch_scope_contract
        from .research_grounded_rag_contract import install as install_research_grounded_rag
        from .retrieval_cpu_budget_contract import (
            install as install_retrieval_cpu_budget,
        )
        from .retrieval_model_residency import install as install_retrieval_residency
        from .runtime_hot_path_contract import (
            assert_installed as assert_runtime_hot_paths,
        )
        from .runtime_hot_path_contract import (
            install as install_runtime_hot_paths,
        )
        from .runtime_live_path_preflight import run_runtime_live_path_preflight
        from .runtime_preflight import run_runtime_preflight
        from .runtime_wrapper_integrity import verify_installed_wrappers
        from .source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA
        from .target_grounding_contract import install_target_grounding_contract
        from .task_artifact_contract import install_task_artifact_contract
        from .tool_schema_ownership_contract import (
            install as install_tool_schema_ownership,
        )
        from .tool_validation_surface_contract import (
            install as install_tool_validation_surface,
        )
        from .work_graph_receipt_integrity_contract import (
            install as install_work_graph_receipt_integrity,
        )

        install_agent_mcp_transport_pool()
        install_mcp_schema_integrity(
            agent_tool_runtime,
            external_agent_bridge,
            external_mcp_router,
        )
        install_external_mcp_binding(external_agent_bridge, external_mcp_router)
        install_external_mcp_binding_concurrency(external_agent_bridge)
        install_runtime_hot_paths(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        install_prefetch_resilience(parallel_runtime_module=parallel_runtime_contract)
        install_observation_determinism(agent_tool_runtime_module=agent_tool_runtime)
        install_procedural_skill_identity(external_procedural_skill_contract)
        install_tool_schema_ownership(
            agent_tool_runtime,
            expected_parameters={"apply_source_edit": SOURCE_EDIT_SCHEMA},
        )
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        install_retrieval_residency(model_router_module=model_router)
        install_retrieval_cpu_budget(repository_grounding, agentic_pre_design_rag)
        install_research_grounded_rag(agentic_pre_design_rag)
        install_model_tool_alias_permissions(agent_capability_context, model_tool_aliases)
        install_llama_mtp_cache_policy(llama_server_autotune, llama_server_runtime_tuning)
        install_tool_validation_surface()
        install_llama_finish_reason(llama_cpp_adapter)
        install_prefill_calibration_strictness(llama_cpp_adapter)
        install_llama_server_response_resilience(llama_cpp_adapter)
        install_work_graph_receipt_integrity(work_graph)

        install_evidence_first_pipeline()
        install_evidence_task_receipts()
        # Request semantics and retrieval-query planning are explicit calls from
        # GameDesignPlanner via planning_authority. Do not rebind them here.
        install_planner_graph_integrity()
        install_planner_design_readiness()
        install_planner_requirement_traceability()
        install_deep_design_execution()
        install_evidence_obligation_contract()
        # This legacy integration still supplies central research/knowledge-plan views;
        # its request-builder wrapper is no longer on the GameDesignPlanner live path.
        install_authored_scope_research()
        install_target_grounding_contract()
        install_requirement_branch_scope_contract()
        install_task_artifact_contract()
        install_design_resolution_provenance_contract()
        install_production_boundary_contract()
        install_quality_public_acceptance_view(production_contract, quality_evidence)
        install_implementation_kind_boundary(
            complete_spec_module=complete_spec,
            support_module=complete_orchestrator_support,
            orchestrator_module=complete_orchestrator,
            template_module=planner_template_schema,
        )
        install_execution_feedback_exception_scope(execution_feedback_replan_contract)
        install_execution_feedback_owner_precision(execution_feedback_replan_contract)
        execution_feedback_replan_contract.install(
            orchestrator_module=complete_orchestrator,
            work_graph_module=work_graph,
        )

        assert_runtime_hot_paths(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        verify_installed_wrappers()
        run_context_budget_preflight()
        run_runtime_live_path_preflight()
        run_runtime_preflight()

        _FINALIZED = True
        _FINALIZING = False


__all__ = ["finalize_runtime"]
