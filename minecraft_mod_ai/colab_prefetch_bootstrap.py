from __future__ import annotations

import os
from typing import Any


def start(model_registry_module: Any) -> None:
    """Install late runtime contracts, then overlap selected-model prefetch with setup."""

    # The hardware-policy wrapper is already installed before this bootstrap is
    # imported. Add the request-mode router now so every later local llama call uses
    # baseline decoding for structured JSON and only verified MTP for free text/code.
    from . import llama_server_hardware_policy as hardware_policy_module
    from .colab_llama_request_routing_contract import install as install_request_routing

    install_request_routing(hardware_policy_module)

    # The old asset wrapper evicts the resident llama process before FLUX starts.
    # Serialize that eviction with the same re-entrant GPU lock used by text/image
    # generation so another executor cannot kill an active LLM request.
    from . import complete_orchestrator_services as services_module
    from . import model_router as model_router_module
    from .colab_gpu_handoff_contract import install as install_gpu_handoff

    install_gpu_handoff(
        services_module=services_module,
        model_router_module=model_router_module,
    )

    # parallel_runtime_contract is installed after the platform planner and used to
    # capture the old generic research function. Rebind that late overlay to the
    # exact selected PlatformLock while retaining its I/O overlap.
    from . import complete_planner as complete_planner_module
    from . import central_research as central_module
    from . import retrieval as retrieval_module
    from .parallel_platform_rag_contract import install as install_parallel_platform_rag

    install_parallel_platform_rag(
        complete_planner_module=complete_planner_module,
        central_module=central_module,
        retrieval_module=retrieval_module,
    )

    managed_colab = bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())
    if not managed_colab:
        return

    # The historical notebook passed minecraft_version="1.20.1" only because the
    # pre-routing constructor required a value. Ignore that implementation placeholder
    # in managed Colab; an actual version written by the user in the prompt remains a
    # normal resolver constraint.
    from . import game_design as game_design_module
    from .colab_auto_platform_contract import install as install_colab_auto_platform

    install_colab_auto_platform(game_design_module)

    # A seed page carries at most twelve independent ecosystem routes. Start all
    # twelve by default in Colab instead of leaving four queued behind an 8-worker
    # pool. Explicit user/runtime overrides remain authoritative.
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", "12")
    os.environ.setdefault("MMM_RESEARCH_WORKERS", "8")

    try:
        import __main__

        profile_name = str(getattr(__main__, "MODEL_PROFILE", "")).strip()
    except Exception:
        return
    if not profile_name:
        return

    try:
        registry = model_registry_module.ModelRegistry()
        registry.load_profile(profile_name)
    except Exception:
        # Setup/profile validation remains authoritative in the notebook. Early
        # prefetch is opportunistic and must not replace its explicit diagnostics.
        return


__all__ = ["start"]
