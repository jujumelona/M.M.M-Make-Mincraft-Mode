from __future__ import annotations

"""Single late-finalization owner for the fully composed runtime.

``runtime_bootstrap.initialize_runtime`` installs the ordered core/late/post-bootstrap
contracts. A few integrations must intentionally happen only *after* that sequence:
MCP transport pooling replaces the final AgentToolRuntime session owner, coder route
integrity must wrap the progress-aware loop that bootstrap installs late, and the
structured-intent projection must be the *outermost* routing-query owner so older
user-only wrappers cannot reintroduce tail truncation. Keeping those operations here
prevents package ``__init__`` from becoming a second ad-hoc bootstrap graph.

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

        from . import agent_capability_context
        from . import agent_tool_runtime
        from . import agentic_pre_design_rag
        from . import causal_tool_frontier_contract
        from . import external_agent_bridge
        from . import external_mcp_router
        from . import external_procedural_skill_contract
        from . import llama_server_autotune
        from . import llama_server_hardware_policy
        from . import llama_server_runtime_tuning
        from . import mcp_transport_pool
        from . import model_router
        from . import model_tool_aliases
        from . import parallel_runtime_contract
        from . import repository_grounding
        from . import research_rag_performance
        from . import small_model_max_agent_contract
        from .agent_observation_determinism import install as install_observation_determinism
        from .agent_routing_intent_contract import install as install_routing_intent
        from .causal_stale_tool_recovery_contract import install as install_stale_tool_recovery
        from .coder_tool_route_integrity_contract import install as install_route_integrity
        from .complete_orchestrator import CompleteProductionOrchestrator
        from .completion_boundary_work_recovery import install as install_completion_boundary_work_recovery
        from .context_budget_preflight import run_context_budget_preflight
        from .evidence_first_pipeline_contract import install as install_evidence_first_pipeline
        from .external_mcp_binding_concurrency_contract import (
            install as install_external_mcp_binding_concurrency,
        )
        from .external_mcp_binding_contract import install as install_external_mcp_binding
        from .forced_tool_execution_contract import install as install_forced_tool_execution
        from .generation_concurrency_safety import install as install_generation_safety
        from .llama_finish_reason_contract import install as install_llama_finish_reason
        from .llama_length_resilience import install as install_llama_length_resilience
        from .llama_mtp_cache_policy import install as install_llama_mtp_cache_policy
        from .llama_server_response_resilience import (
            install as install_llama_server_response_resilience,
        )
        from .llama_unbounded_generation import install as install_llama_unbounded_generation
        from .mcp_schema_integrity_contract import install as install_mcp_schema_integrity
        from .mcp_transport_pool import install_agent_mcp_transport_pool
        from .model_adapters import llama_cpp_adapter, openai_compatible
        from .model_prefetch_resilience import install as install_prefetch_resilience
        from .model_tool_alias_permission_policy import install as install_model_tool_alias_permissions
        from .procedural_skill_identity_contract import install as install_procedural_skill_identity
        from .qwen_enum_recovery_contract import install as install_qwen_enum_recovery
        from .retrieval_cpu_budget_contract import install as install_retrieval_cpu_budget
        from .retrieval_model_residency import install as install_retrieval_residency
        from .runtime_hot_path_contract import (
            assert_installed as assert_runtime_hot_paths,
            install as install_runtime_hot_paths,
        )
        from .runtime_live_path_preflight import run_runtime_live_path_preflight
        from .runtime_preflight import run_runtime_preflight
        from .runtime_wrapper_integrity import verify_installed_wrappers
        from .small_model_compacting_adapter import install as install_small_model_compaction
        from .source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA
        from .tool_schema_ownership_contract import install as install_tool_schema_ownership
        from .tool_validation_surface_contract import install as install_tool_validation_surface

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
        install_route_integrity(
            model_router_module=model_router,
            small_model_module=small_model_max_agent_contract,
            causal_module=causal_tool_frontier_contract,
        )
        install_stale_tool_recovery(causal_tool_frontier_contract)
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        install_completion_boundary_work_recovery(CompleteProductionOrchestrator)
        install_forced_tool_execution(
            openai_compatible_module=openai_compatible,
            llama_cpp_module=llama_cpp_adapter,
        )
        install_retrieval_residency(model_router_module=model_router)
        install_retrieval_cpu_budget(repository_grounding, agentic_pre_design_rag)
        install_small_model_compaction(model_router)
        install_model_tool_alias_permissions(agent_capability_context, model_tool_aliases)
        install_llama_mtp_cache_policy(llama_server_autotune, llama_server_runtime_tuning)
        install_llama_unbounded_generation(llama_server_hardware_policy)
        install_tool_validation_surface()
        install_qwen_enum_recovery(llama_cpp_adapter)
        install_llama_finish_reason(llama_cpp_adapter)
        install_llama_server_response_resilience(llama_cpp_adapter)
        install_llama_length_resilience(llama_cpp_adapter)
        assert_runtime_hot_paths(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        verify_installed_wrappers()
        run_context_budget_preflight()
        run_runtime_live_path_preflight()
        run_runtime_preflight()
        install_evidence_first_pipeline()
        _FINALIZED = True
        _FINALIZING = False


__all__ = ["finalize_runtime"]
