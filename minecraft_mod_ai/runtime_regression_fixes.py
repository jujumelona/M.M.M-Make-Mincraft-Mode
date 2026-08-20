from __future__ import annotations

"""Temporary research-RAG integration pending native ownership migration."""

_INSTALLED = False


def _fix_research_hotpaths() -> None:
    from . import centroid_vector_rag, rag_index, research_rag_performance

    research_rag_performance.harden(rag_index, centroid_vector_rag)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _fix_research_hotpaths()
    _INSTALLED = True


__all__ = ["install"]
