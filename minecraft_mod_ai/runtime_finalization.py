from __future__ import annotations

"""Single late-finalization owner for the fully composed runtime.

``runtime_bootstrap.initialize_runtime`` installs the ordered core/late/post-bootstrap
contracts. A few integrations must intentionally happen only *after* that sequence:
MCP transport pooling replaces the final AgentToolRuntime session owner, coder route
integrity must wrap the progress-aware loop that bootstrap installs late, and the
structured-intent projection must be the *outermost* routing-query owner so older
user-only wrappers cannot reintroduce tail truncation. Keeping those operations here
prevents package ``__init__`` from becoming a second ad-hoc bootstrap graph.
"""

import threading

_FINALIZE_LOCK = threading.RLock()
_FINALIZED = False


def finalize_runtime() -> None:
    global _FINALIZED
    if _FINALIZED:
        return
    with _FINALIZE_LOCK:
        if _FINALIZED:
            return

        from . import agent_tool_runtime
        from . import causal_tool_frontier_contract
        from . import llama_server_autotune
        from . import llama_server_hardware_policy
        from . import llama_server_runtime_tuning
        from . import model_router
        from . import parallel_runtime_contract
        from . import small_model_max_agent_contract
        from .agent_observation_determinism import install as install_observation_determinism
        from .agent_routing_intent_contract import install as install_routing_intent
        from .coder_tool_route_integrity_contract import install as install_route_integrity
        from .context_budget_preflight import run_context_budget_preflight
        from .forced_tool_execution_contract import install as install_forced_tool_execution
        from .generation_concurrency_safety import install as install_generation_safety
        from .llama_length_resilience import install as install_llama_length_resilience
        from .llama_mtp_cache_policy import install as install_llama_mtp_cache_policy
        from .llama_unbounded_generation import install as install_llama_unbounded_generation
        from .mcp_transport_pool import install_agent_mcp_transport_pool
        from .model_adapters import llama_cpp_adapter, openai_compatible
        from .model_prefetch_resilience import install as install_prefetch_resilience
        from .retrieval_model_residency import install as install_retrieval_residency
        from .runtime_live_path_preflight import run_runtime_live_path_preflight
        from .runtime_preflight import run_runtime_preflight
        from .runtime_wrapper_integrity import verify_installed_wrappers
        from .small_model_compacting_adapter import install as install_small_model_compaction

        # Order is semantic. Route integrity must come after bootstrap because adaptive
        # retrieval replaces ModelRouter._generate_with_tools late. Structured intent
        # comes after route integrity because route integrity also installs a legacy
        # user-only query wrapper; the structured projection must own the final call.
        install_agent_mcp_transport_pool()
        install_prefetch_resilience(parallel_runtime_module=parallel_runtime_contract)
        install_observation_determinism(agent_tool_runtime_module=agent_tool_runtime)
        install_route_integrity(
            model_router_module=model_router,
            small_model_module=small_model_max_agent_contract,
            causal_module=causal_tool_frontier_contract,
        )
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        # Local llama.cpp tool-choice validation is adapter-owned. Only remote
        # OpenAI-compatible transport still needs this late exact-tool wrapper.
        install_forced_tool_execution(openai_compatible_module=openai_compatible)
        # RAG build/search may call embed/rerank repeatedly. Install this before the
        # structural preflight so a fresh process cannot silently regress to per-call
        # CPU model construction.
        install_retrieval_residency(model_router_module=model_router)
        # Bootstrap installs an early compactor, but adaptive retrieval later replaces
        # the tool loop instead of delegating through that old callable. Re-bind the
        # compactor here, after every loop owner, so it is on the executable path.
        install_small_model_compaction(model_router)
        # MTP-capable llama.cpp profiles already reuse prompt state through the native
        # prompt cache. Drop the duplicate RAM arena only for those profiles, while
        # keeping the generic bounded cache reservation for non-MTP launches.
        install_llama_mtp_cache_policy(llama_server_autotune, llama_server_runtime_tuning)
        # llama.cpp n_predict/max_tokens=-1 is its native unlimited policy. Do not
        # truncate ordinary source-edit/tool turns at a registry max_new_tokens value.
        # Registry-declared Qwen T4/MTP pages retain their explicit bounded policy.
        install_llama_unbounded_generation(llama_server_hardware_policy)
        # A remaining finish_reason='length' is then context pressure. Recover once by
        # compacting large observations; never invent another output-token ceiling.
        install_llama_length_resilience(llama_cpp_adapter)
        # Bootstrap's integrity stage runs before these late finalization wrappers are
        # installed. Re-audit the fully composed runtime here so a narrowed late wrapper
        # fails before any retrieval/model work instead of surfacing during generation.
        verify_installed_wrappers()
        # Exercise the exact first assistant + parallel 48 KiB tool-observation shape
        # that previously reached the server boundary before a second assistant turn.
        run_context_budget_preflight()
        # Marker inheritance cannot prove that a wrapper executes. Verify the concrete
        # code-object order of the final ModelRouter path before any model is loaded.
        run_runtime_live_path_preflight()
        run_runtime_preflight()
        _FINALIZED = True


__all__ = ["finalize_runtime"]
