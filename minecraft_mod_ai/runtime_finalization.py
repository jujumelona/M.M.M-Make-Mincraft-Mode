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
        # Intentionally cleared only after every installer and preflight succeeds.
        # Any exception leaves the process poisoned so already-applied wrappers cannot
        # be replayed on top of themselves by a later import/finalization attempt.
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
        from . import small_model_execution_extensions_contract
        from . import small_model_max_agent_contract
        from .agent_observation_determinism import install as install_observation_determinism
        from .agent_routing_intent_contract import install as install_routing_intent
        from .bounded_source_edit_contract import (
            BOUNDED_SOURCE_EDIT_SCHEMA,
            install as install_bounded_source_edit,
        )
        from .causal_stale_tool_recovery_contract import install as install_stale_tool_recovery
        from .coder_tool_route_integrity_contract import install as install_route_integrity
        from .complete_orchestrator import CompleteProductionOrchestrator
        from .completion_boundary_work_recovery import install as install_completion_boundary_work_recovery
        from .context_budget_preflight import run_context_budget_preflight
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
        from .llama_tool_output_budget import install as install_llama_tool_output_budget
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
        from .source_edit_scalar_protocol_contract import (
            SOURCE_EDIT_SCHEMA,
            install as install_scalar_source_edit_protocol,
        )
        from .tool_schema_ownership_contract import install as install_tool_schema_ownership
        from .tool_validation_surface_contract import install as install_tool_validation_surface

        # Order is semantic. Route integrity must come after bootstrap because adaptive
        # retrieval replaces ModelRouter._generate_with_tools late. Structured intent
        # comes after route integrity because route integrity also installs a legacy
        # user-only query wrapper; the structured projection must own the final call.
        install_agent_mcp_transport_pool()
        # Raw tools/list is the source contract for both the first-party MCP server and
        # reviewed external providers. Reject duplicate names and malformed/non-object
        # input schemas before any model-facing layer can normalize them into a weaker
        # permissive shape. Then bind external schema discovery to the exact provider
        # that executes it so route failover cannot swap schema A for provider B.
        install_mcp_schema_integrity(
            agent_tool_runtime,
            external_agent_bridge,
            external_mcp_router,
        )
        install_external_mcp_binding(external_agent_bridge, external_mcp_router)
        # The bridge is shared across concurrent model agents. Keep one reviewed schema
        # owner per exact stage/target/access/provider scope so a simultaneous schema
        # refresh cannot replace the provider contract another request already saw.
        install_external_mcp_binding_concurrency(external_agent_bridge)
        # Remove event-loop blocking queue backpressure, unrelated-provider global
        # serialization, and query-time semantic-LSH reconciliation only after the
        # safety/ownership contracts above have established the execution boundaries.
        install_runtime_hot_paths(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        install_prefetch_resilience(parallel_runtime_module=parallel_runtime_contract)
        install_observation_determinism(agent_tool_runtime_module=agent_tool_runtime)
        # Source files are repository state, not model output pages. Replace the old
        # complete-file model contract with bounded exact-span edits before any runtime
        # instance can cache the generation schema. The host still materializes the
        # canonical SHA-bound transactional patch immediately before execution.
        install_bounded_source_edit(agent_tool_runtime)
        # The small-model extension historically exposed one nested array<object> of
        # up to dozens of edits. Besides encouraging oversized actions, that shape is
        # fragile on tagged Qwen tool transports. Freeze the final model-facing ACI to
        # one scalar exact edit per turn before any runtime instance caches schemas.
        install_scalar_source_edit_protocol(
            small_model_execution_extensions_contract,
            agent_tool_runtime,
        )
        # Procedural skill IDs are content commitments. Persistent JSONL is mutable, so
        # reject rows whose current content no longer hashes to their declared skill_id
        # before dependency composition can collapse identities in a dict.
        install_procedural_skill_identity(external_procedural_skill_contract)
        # Schema producers may be layered, but ownership is singular at the final
        # runtime boundary. Validate the composed surface before AgentToolRuntime can
        # cache it or dispatch by name, and pin the two source-write projections to
        # their final model-facing contracts so wrapper order cannot silently regress
        # them to an older host/raw schema.
        install_tool_schema_ownership(
            agent_tool_runtime,
            expected_parameters={
                "apply_source_patch": BOUNDED_SOURCE_EDIT_SCHEMA,
                "apply_source_edit": SOURCE_EDIT_SCHEMA,
            },
        )
        install_route_integrity(
            model_router_module=model_router,
            small_model_module=small_model_max_agent_contract,
            causal_module=causal_tool_frontier_contract,
        )
        # A stale but authorized action is parser-valid yet not executable on a later
        # causal frontier. Consume it before the core loop and force one legal frontier
        # correction rather than manufacturing a failed tool observation and crashing
        # when the model repeats the stale action.
        install_stale_tool_recovery(causal_tool_frontier_contract)
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        # Durable generation nodes own output-exhaustion recovery. Retry that exact
        # persisted node once instead of issuing a hidden second model completion.
        install_completion_boundary_work_recovery(CompleteProductionOrchestrator)
        # Exact host-required calls have one policy owner across transports. Remote
        # endpoints use native required-tool forcing; local llama.cpp renders one
        # visible Jinja schema as required, returns raw markup through its managed
        # pure-content parser, and performs bounded semantic validation on the host.
        install_forced_tool_execution(
            openai_compatible_module=openai_compatible,
            llama_cpp_module=llama_cpp_adapter,
        )
        # RAG build/search may call embed/rerank repeatedly. Install this before the
        # structural preflight so a fresh process cannot silently regress to per-call
        # CPU model construction.
        install_retrieval_residency(model_router_module=model_router)
        # Baseline repository observation and pre-design code search are host-local
        # lexical/graph work. Do not silently load CPU embedding/reranker models merely
        # because the registry exposes them; dense escalation is explicit opt-in.
        install_retrieval_cpu_budget(repository_grounding, agentic_pre_design_rag)
        # Bootstrap installs an early compactor, but adaptive retrieval later replaces
        # the tool loop instead of delegating through that old callable. Re-bind the
        # compactor here, after every loop owner, so it is on the executable path.
        install_small_model_compaction(model_router)
        # Alias ACIs share one reviewed permission namespace with their canonical tool.
        # Normalize before the already-composed Skill permission stack so late wrapper
        # rebinding cannot make an alias report a narrower authorization set.
        install_model_tool_alias_permissions(agent_capability_context, model_tool_aliases)
        # MTP-capable llama.cpp profiles already reuse prompt state through the native
        # prompt cache. Drop the duplicate RAM arena only for those profiles, while
        # keeping the generic bounded cache reservation for non-MTP launches.
        install_llama_mtp_cache_policy(llama_server_autotune, llama_server_runtime_tuning)
        # Ordinary top-level llama.cpp turns may use the native unlimited prediction
        # policy. Registry-declared Qwen T4/MTP pages retain their explicit bounds.
        install_llama_unbounded_generation(llama_server_hardware_policy)
        # Tool turns are semantic actions, not long-form generation. Install this after
        # every unbounded/profile wrapper so a large RAG context cannot let a tool JSON
        # decode consume the remaining model context before the action closes.
        install_llama_tool_output_budget(llama_server_hardware_policy)
        # Parse stale but host-authorized tool names against the complete authorized
        # surface; execution remains restricted by the per-turn causal visibility gate.
        install_tool_validation_surface()
        # Qwen tagged string parameters occasionally include harmless JSON quoting or
        # formatting drift. Canonicalize only uniquely equivalent enum spellings here.
        # Semantic mismatches remain typed parser failures; causal stale recovery is the
        # single owner of any model-action re-synchronization.
        install_qwen_enum_recovery(llama_cpp_adapter)
        # llama.cpp reports both output-cap exhaustion and context pressure as
        # finish_reason='length'. Classify them before resilience wrappers are bound so
        # only genuine context pressure triggers observation compaction/retry.
        install_llama_finish_reason(llama_cpp_adapter)
        # A transient local inference failure occurs before any semantic turn reaches
        # ModelRouter, so exactly one transport retry cannot duplicate a tool action.
        # Install this inside length recovery: 5xx/connection recovery happens first,
        # while genuine finish_reason=length still follows the compaction path below.
        install_llama_server_response_resilience(llama_cpp_adapter)
        # A remaining context-pressure length stop recovers once by compacting tool
        # observations. Output-cap exhaustion is not retried in the adapter; the durable
        # work-node owner above decides whether that persisted generation node retries.
        install_llama_length_resilience(llama_cpp_adapter)
        # Bootstrap's integrity stage runs before these late finalization wrappers are
        # installed. Re-audit the fully composed runtime here so a narrowed late wrapper
        # fails before any retrieval/model work instead of surfacing during generation.
        assert_runtime_hot_paths(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        verify_installed_wrappers()
        # Exercise the exact first assistant + parallel 48 KiB tool-observation shape
        # that previously reached the server boundary before a second assistant turn.
        run_context_budget_preflight()
        # Marker inheritance cannot prove that a wrapper executes. Verify the concrete
        # code-object order of the final ModelRouter path before any model is loaded.
        run_runtime_live_path_preflight()
        run_runtime_preflight()
        _FINALIZED = True
        _FINALIZING = False


__all__ = ["finalize_runtime"]
