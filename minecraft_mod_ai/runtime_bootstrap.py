from __future__ import annotations

"""Single owner for still-live installer-backed runtime contracts.

Source-owned contracts must not be rebound here. This bootstrap exists only for
cross-cutting contracts that still expose an install() API and therefore otherwise
depend on import/test order.
"""

from threading import RLock

_LOCK = RLock()
_INITIALIZED = False


def initialize_runtime() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK:
        if _INITIALIZED:
            return

        from . import (
            complete_orchestrator,
            custom_module_generator,
            extended_content_generator,
            external_mcp_router,
            mcp_transport_pool,
            performance_final_contract,
            project_index,
            research_rag_performance,
            source_patch,
        )
        from .custom_generation_search_contract import install as install_custom_search
        from .extended_registration_contract import install as install_extended_registration
        from .mcp_child_trace_contract import install as install_mcp_child_trace
        from .performance_final_contract import install as install_performance_contract
        from .performance_final_tuning import install as install_performance_tuning
        from .project_index_manifest_efficiency_contract import (
            install as install_project_index_manifest,
        )
        from .project_manifest_hash_efficiency_contract import (
            install as install_project_manifest_hash,
        )
        from .runtime_hot_path_contract import install as install_runtime_hot_path
        from .small_model_retrieval_efficiency_contract import (
            install as install_selective_retrieval,
        )

        install_runtime_hot_path(
            mcp_transport_pool_module=mcp_transport_pool,
            external_mcp_router_module=external_mcp_router,
            research_rag_performance_module=research_rag_performance,
        )
        install_mcp_child_trace(mcp_transport_pool)

        install_extended_registration(extended_content_generator)
        install_project_index_manifest(project_index)
        install_performance_tuning(performance_final_contract)
        install_project_manifest_hash(complete_orchestrator, project_index)
        install_performance_contract(
            complete_orchestrator,
            custom_module_generator,
            source_patch,
        )

        install_custom_search(custom_module_generator)
        install_selective_retrieval()

        _INITIALIZED = True


def runtime_initialized() -> bool:
    return _INITIALIZED


__all__ = ["initialize_runtime", "runtime_initialized"]
