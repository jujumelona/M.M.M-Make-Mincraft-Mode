from __future__ import annotations

"""Keep baseline repository retrieval cheap unless dense work is explicitly requested.

The default Colab/T4 profile runs Qwen retrieval models on CPU. Baseline repository
observation already has exact source anchors, lexical ranking, dependency-graph expansion
and procedural alignment; silently adding embedding/reranking there can turn a small
project into minutes of CPU work before the coder receives its first turn.

Pre-design code RAG follows the same policy: the host-owned lexical/graph search is the
default. Operators that intentionally want CPU dense escalation can opt in with
``MMM_RAG_ENABLE_CPU_DENSE=1``.
"""

import os
from typing import Any

_MARKER = "_mmm_retrieval_cpu_budget_v1"
_DENSE_OPT_IN = "MMM_RAG_ENABLE_CPU_DENSE"
_HYBRID_MARKER = "_mmm_small_model_hybrid_code_rag"


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
    """Strip only the known dense code-RAG wrappers and retain the lexical base owner."""

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
        # The demand-driven wrapper is intentionally outside the hybrid wrapper and
        # inherits its marker through functools.wraps. If the next callable is still
        # marked hybrid, unwrap this layer as well; stop once the lexical base is next.
        if bool(getattr(wrapped, _HYBRID_MARKER, False)):
            cursor = wrapped
            continue
        break
    return cursor


def install(repository_grounding_module: Any, pre_design_module: Any) -> None:
    if bool(getattr(repository_grounding_module, _MARKER, False)):
        return

    if not _dense_opted_in():
        repository_grounding_module._explore_with_degraded_fallback = (
            _lexical_repository_exploration
        )
        lexical_owner = _lexical_pre_design_owner(pre_design_module._search_code_index)
        pre_design_module._search_code_index = lexical_owner

    setattr(repository_grounding_module, _MARKER, True)
    setattr(pre_design_module, _MARKER, True)


__all__ = ["install"]
