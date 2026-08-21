from __future__ import annotations

"""Keep baseline and live code retrieval cheap unless dense work is explicit.

The default Colab/T4 profile runs Qwen retrieval models on CPU. Exact source anchors,
lexical ranking, dependency-graph expansion and procedural alignment are already
available without loading those models. Dense retrieval remains available only when an
operator explicitly sets ``MMM_RAG_ENABLE_CPU_DENSE=1``.

This policy is revalidated against the *current executable owners* every time install
runs. Module-level installation markers are not sufficient because later runtime
composition can legitimately replace a search wrapper while leaving an old marker
behind.
"""

import os
from functools import wraps
from typing import Any, Sequence

_MARKER = "_mmm_retrieval_cpu_budget_v2"
_HYBRID_MARKER = "_mmm_small_model_hybrid_code_rag"
_HYBRID_BUDGET_MARKER = "_mmm_cpu_dense_hybrid_guard_v2"
_PRODUCTION_BUDGET_MARKER = "_mmm_cpu_dense_production_guard_v1"
_DENSE_OPT_IN = "MMM_RAG_ENABLE_CPU_DENSE"


def _dense_opted_in() -> bool:
    return os.environ.get(_DENSE_OPT_IN, "").strip() == "1"


def require_dense_retrieval_device(
    device: Any,
    *,
    role: str,
    model_id: str,
    backend: str,
) -> None:
    """Fail closed at the model-loader boundary for implicit CPU dense retrieval.

    Higher-level routing guards are scheduling optimizations, not a security boundary:
    pre-design/research code can legitimately reach an adapter through a different
    owner. Every dense retrieval model loader therefore rechecks the same explicit
    opt-in immediately before importing or constructing heavyweight model state.
    """

    selected = str(device or "cpu").strip().casefold()
    if not selected.startswith("cpu") or _dense_opted_in():
        return

    from .model_adapters.base import ModelConfigurationError

    raise ModelConfigurationError(
        f"CPU dense retrieval {backend} loading is disabled for role={role!r}, "
        f"model={model_id!r}. Set {_DENSE_OPT_IN}=1 to opt in explicitly."
    )


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

    Check function ownership rather than a sticky module marker. If another composer
    replaced ``_modes`` or centroid adaptation after an earlier install, this function
    binds the policy to the new executable owner instead of incorrectly returning early.
    """

    current_modes = hybrid_module._modes
    current_adapt = hybrid_module.adapt_query_vector
    modes_guarded = bool(getattr(current_modes, _HYBRID_BUDGET_MARKER, False))
    adapt_guarded = bool(getattr(current_adapt, _HYBRID_BUDGET_MARKER, False))
    if modes_guarded and adapt_guarded:
        return

    if not modes_guarded:
        original_modes = current_modes

        @wraps(original_modes)
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

        setattr(budgeted_modes, _HYBRID_BUDGET_MARKER, True)
        hybrid_module._modes = budgeted_modes

    if not adapt_guarded:
        original_adapt_query_vector = current_adapt

        @wraps(original_adapt_query_vector)
        def budgeted_adapt_query_vector(
            router: Any,
            query: str,
            hit_texts: Sequence[str],
            *,
            alpha: float = 0.65,
        ) -> list[float]:
            if _dense_opted_in():
                return original_adapt_query_vector(
                    router,
                    query,
                    hit_texts,
                    alpha=alpha,
                )
            return []

        setattr(budgeted_adapt_query_vector, _HYBRID_BUDGET_MARKER, True)
        hybrid_module.adapt_query_vector = budgeted_adapt_query_vector


def _install_production_tool_budget(production_tools_module: Any) -> None:
    """Make the ProductionToolService boundary fail closed to lexical CPU retrieval."""

    cls = production_tools_module.ProductionToolService

    current_search = cls.search_code_rag
    if not bool(getattr(current_search, _PRODUCTION_BUDGET_MARKER, False)):

        @wraps(current_search)
        def search_budgeted(
            self: Any,
            query: str,
            *,
            index_path: str = "rag/project-index.json",
            limit: int = 8,
            semantic: bool = False,
            rerank: bool = False,
            required_metadata: dict[str, Any] | None = None,
        ):
            dense = _dense_opted_in()
            return current_search(
                self,
                query,
                index_path=index_path,
                limit=limit,
                semantic=bool(semantic and dense),
                rerank=bool(rerank and dense),
                required_metadata=required_metadata,
            )

        setattr(search_budgeted, _PRODUCTION_BUDGET_MARKER, True)
        cls.search_code_rag = search_budgeted

    current_index = cls.index_project_rag
    if not bool(getattr(current_index, _PRODUCTION_BUDGET_MARKER, False)):

        @wraps(current_index)
        def index_budgeted(
            self: Any,
            roots: Any,
            *,
            index_path: str = "rag/project-index.json",
            metadata: dict[str, Any],
            semantic: bool = False,
        ):
            return current_index(
                self,
                roots,
                index_path=index_path,
                metadata=metadata,
                semantic=bool(semantic and _dense_opted_in()),
            )

        setattr(index_budgeted, _PRODUCTION_BUDGET_MARKER, True)
        cls.index_project_rag = index_budgeted


def install(repository_grounding_module: Any, pre_design_module: Any) -> None:
    from . import production_tools, small_model_hybrid_search_contract

    # These are intentionally checked on every call. Runtime composition can replace a
    # callable after an earlier policy install, and a stale module marker must never be
    # interpreted as proof that the live path is still guarded.
    _install_live_hybrid_budget(small_model_hybrid_search_contract)
    _install_production_tool_budget(production_tools)

    if not _dense_opted_in():
        repository_grounding_module._explore_with_degraded_fallback = (
            _lexical_repository_exploration
        )
        pre_design_module._search_code_index = _lexical_pre_design_owner(
            pre_design_module._search_code_index
        )

    setattr(repository_grounding_module, _MARKER, True)
    setattr(pre_design_module, _MARKER, True)


__all__ = [
    "_dense_opted_in",
    "_install_live_hybrid_budget",
    "_install_production_tool_budget",
    "install",
    "require_dense_retrieval_device",
]
