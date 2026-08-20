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

        from . import causal_tool_frontier_contract
        from . import model_router
        from . import small_model_max_agent_contract
        from .agent_routing_intent_contract import install as install_routing_intent
        from .coder_tool_route_integrity_contract import install as install_route_integrity
        from .generation_concurrency_safety import install as install_generation_safety
        from .mcp_transport_pool import install_agent_mcp_transport_pool
        from .runtime_preflight import run_runtime_preflight

        # Order is semantic. Route integrity must come after bootstrap because adaptive
        # retrieval replaces ModelRouter._generate_with_tools late. Structured intent
        # comes after route integrity because route integrity also installs a legacy
        # user-only query wrapper; the structured projection must own the final call.
        install_agent_mcp_transport_pool()
        install_route_integrity(
            model_router_module=model_router,
            small_model_module=small_model_max_agent_contract,
            causal_module=causal_tool_frontier_contract,
        )
        install_routing_intent(small_model_module=small_model_max_agent_contract)
        install_generation_safety()
        run_runtime_preflight()
        _FINALIZED = True


__all__ = ["finalize_runtime"]
