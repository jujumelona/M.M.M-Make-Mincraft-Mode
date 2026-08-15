from __future__ import annotations

"""One late bootstrap entry for research-backed MMM bottleneck elimination."""


def install() -> None:
    from . import (
        agentic_pre_design_rag,
        agentic_research_game_design,
        bottleneck_elimination_contract,
        centroid_vector_rag,
        central_intelligence_amplifier,
        central_research,
        complete_spec,
        external_mcp_router,
        rag_index,
        research_adaptive_rag_routing,
        runner,
        small_model_max_agent_contract,
        trajectory_memory,
        work_graph,
    )
    from . import research_rag_performance as rag_performance
    from . import validation_execution_contract
    from .research_adaptive_provider_routing import (
        harden as harden_adaptive_providers,
    )
    from .research_cpu_retrieval_performance import harden as harden_cpu_retrieval
    from .research_gradle_performance import harden as harden_gradle
    from .research_memory_performance import harden as harden_memory
    from .research_rag_amortized_runtime import harden as harden_rag_amortized
    from .research_rag_performance import harden as harden_rag
    from .research_synthesis_performance import harden as harden_synthesis
    from .research_validation_fingerprint_performance import (
        harden as harden_validation_fingerprints,
    )
    from .runtime_hotpath_consolidation import harden as harden_hotpath
    from .small_model_adaptive_compute import harden as harden_adaptive_compute
    from .small_model_concurrency_budget import harden as harden_model_concurrency
    from .work_graph_hash_performance import harden as harden_work_graph_hashes

    # Package bootstrap remains the only install/composition owner. This hardener
    # only adjusts residual admission around contracts already installed immediately
    # before this research performance phase.
    harden_hotpath(
        bottleneck_elimination_contract,
        central_research,
        external_mcp_router,
    )

    harden_rag(rag_index, centroid_vector_rag)
    harden_rag_amortized(rag_index, rag_performance)
    harden_synthesis(agentic_pre_design_rag)
    research_adaptive_rag_routing.harden(
        agentic_pre_design_rag,
        small_model_max_agent_contract,
    )
    harden_adaptive_providers(
        agentic_research_game_design,
        research_adaptive_rag_routing,
    )
    harden_adaptive_compute(
        agentic_research_game_design,
        central_intelligence_amplifier,
    )
    harden_model_concurrency(
        agentic_research_game_design,
        central_intelligence_amplifier,
    )
    indexed_append, indexed_relevant = harden_memory(trajectory_memory)

    # temporary_skill_contract imported these functions by value before this late
    # bootstrap entry. Rebind only its module globals; do not add another wrapper.
    try:
        from . import temporary_skill_contract as temporary
        temporary.append_trajectory = indexed_append
        temporary.relevant_trajectories = indexed_relevant
    except Exception:
        pass

    harden_cpu_retrieval()
    harden_gradle(runner)
    harden_validation_fingerprints(validation_execution_contract)
    harden_work_graph_hashes(work_graph, complete_spec)


__all__ = ["install"]
