from __future__ import annotations

"""One late bootstrap entry for research-backed MMM bottleneck elimination."""


def install() -> None:
    from . import centroid_vector_rag, rag_index, runner, trajectory_memory
    from .research_cpu_retrieval_performance import harden as harden_cpu_retrieval
    from .research_gradle_performance import harden as harden_gradle
    from .research_memory_performance import harden as harden_memory
    from .research_rag_performance import harden as harden_rag

    harden_rag(rag_index, centroid_vector_rag)
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


__all__ = ["install"]
