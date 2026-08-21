from __future__ import annotations

"""Keep baseline and live code retrieval cheap unless dense work is explicit.

The default Colab/T4 profile runs Qwen retrieval models on CPU. Exact source anchors,
lexical ranking, dependency-graph expansion and procedural alignment are already
available without loading those models. Dense retrieval remains available only when an
operator explicitly sets ``MMM_RAG_ENABLE_CPU_DENSE=1``.
"""

import os
from typing import Any

_MARKER = "_mmm_retrieval_cpu_budget_v1"
_HYBRID_MARKER = "_mmm_small_model_hybrid_code_rag"
_HYBRID_BUDGET_MARKER = "_mmm_cpu_dense_hybrid_guard_v1"
_DENSE_OPT_IN = "MMM_RAG_ENABLE_CPU_DENSE"


def _dense_opted_in() -> bool:
    return os.environ.get(_DENSE_OPT_IN, "").strip() == "1"


def _lexical_repository_exploration(
    explorer: Any,
    query: str,
    *,
    diagnostics: tuple[str, ...],
    line_budget: int,
    degraded: list[str],
    lane: str,
):
    """Run deterministic local grounding without loading retrieval models."""

    del degraded, lane
    return explorer.explore(
        query,
        diagnostic_paths=diagnostics,
        line_budget=line_budget,
        semantic=False,
        rerank=False,
    )


def _lexical_pre_design_owner(current: Any) -> Any:
    """Strip only known dense code-RAG wrappers and retain the lexical base owner."""

    cursor = current
    seen: set[int] = set()
    while callable(cursor) and id(cursor) not in seen:
        seen.add(id(cursor))
        wrapped = getattr(cursor, "__wrapped__", None)
        if not callable(wrapped):
            break
        if bool(getattr(cursor, _HYBRID_MARKER, False)):
            cursor = wrapped
            continue
        if bool(getattr(wrapped, _HYBRID_MARKER, False)):
            cursor = wrapped
            continue
        break
    return cursor


def _install_live_hybrid_budget(hybrid_module: Any) -> None:
    """Prevent live ``search_code_rag`` from escalating into CPU dense work.

    The hybrid search wrapper consults ``_modes`` at call time and later uses
    ``adapt_query_vector`` for its centroid fallback. Guard both lookup points instead
    of bypassing the wrapper, so lexical caching, relation expansion, receipts and
    correction metadata remain intact.
    """

    if bool(getattr(hybrid_module, _HYBRID_BUDGET_MARKER, False)):
        return

    original_modes = hybrid_module._modes
    original_adapt_query_vector = hybrid_module.adapt_query_vector

    def budgeted_modes(
        route: str,
        caller_semantic: bool,
        caller_rerank: bool,
    ):
        if _dense_opted_in():
            return original_modes(route, caller_semantic, caller_rerank)
        labels = {
            "dependency": "lexical+relations",
            "global": "lexical+global-relations",
        }
        return ((False, False, labels.get(route, "lexical")),)

    def budgeted_adapt_query_vector(router: Any, query: str, texts: Any):
        if _dense_opted_in():
            return original_adapt_query_vector(router, query, texts)
        return []

    hybrid_module._modes = budgeted_modes
    hybrid_module.adapt_query_vector = budgeted_adapt_query_vector
    setattr(hybrid_module, _HYBRID_BUDGET_MARKER, True)


def install(repository_grounding_module: Any, pre_design_module: Any) -> None:
    if bool(getattr(repository_grounding_module, _MARKER, False)):
        return

    from . import small_model_hybrid_search_contract

    if not _dense_opted_in():
        repository_grounding_module._explore_with_degraded_fallback = (
            _lexical_repository_exploration
        )
        lexical_owner = _lexical_pre_design_owner(pre_design_module._search_code_index)
        pre_design_module._search_code_index = lexical_owner

    _install_live_hybrid_budget(small_model_hybrid_search_contract)
    setattr(repository_grounding_module, _MARKER, True)
    setattr(pre_design_module, _MARKER, True)


__all__ = ["install"]
