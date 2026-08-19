from __future__ import annotations

from functools import wraps
from typing import Any, Mapping, Sequence

_EXPLORER_MARKER = "__mmm_capability_aware_explorer_v1__"
_SCALING_MARKER = "__mmm_budget_aware_test_time_scaling_v1__"
_RESEARCH_SEED_MARKER = "__mmm_positive_research_graph_seed_v1__"
_RESEARCH_METRIC_MARKER = "__mmm_complete_research_metric_vector_v1__"


def harden_adaptive_execution() -> None:
    """Make optional retrieval/scaling depend on real runtime capabilities.

    Selective retrieval must not assume every router proxy exposes embedding or
    reranking, automatic test-time scaling must respect the native decode budget,
    and research graph expansion must not promote zero-relevance global symbols.
    """
    _harden_repository_explorer()
    _harden_research_code_context()
    _harden_test_time_scaling()


def _harden_repository_explorer() -> None:
    from .repository_explorer import RepositoryExplorer

    current = RepositoryExplorer.explore
    if getattr(current, _EXPLORER_MARKER, False):
        return

    @wraps(current)
    def explore(self: Any, query: str, *args: Any, **kwargs: Any):
        router = getattr(self, "router", None)
        if not _has_callable(router, "embed"):
            kwargs["semantic"] = False
        if not _has_callable(router, "rerank"):
            kwargs["rerank"] = False
        return current(self, query, *args, **kwargs)

    setattr(explore, _EXPLORER_MARKER, True)
    RepositoryExplorer.explore = explore


def _harden_research_code_context() -> None:
    from . import research_code_context as research

    current_filter = research.ResearchCodeContext._semantic_symbol_filter
    if not getattr(current_filter, _RESEARCH_SEED_MARKER, False):

        @wraps(current_filter)
        def semantic_symbol_filter(
            self: Any,
            query: str,
            symbols: Sequence[Any],
            *,
            limit: int,
        ) -> list[Any]:
            if not symbols:
                return []
            router = getattr(self, "router", None)
            rerank = getattr(router, "rerank", None)
            if callable(rerank):
                candidates = list(symbols)
                texts = [
                    research._join_query(
                        symbol.signature,
                        symbol.path,
                        self._symbol_text(symbol)[:3000],
                    )
                    for symbol in candidates
                ]
                try:
                    raw_scores = rerank(query, texts)
                    if len(raw_scores) == len(candidates):
                        scored = [
                            (float(score), symbol)
                            for score, symbol in zip(raw_scores, candidates, strict=True)
                        ]
                        positive = [item for item in scored if item[0] > 0.0]
                        if positive:
                            positive.sort(
                                key=lambda item: (
                                    -item[0],
                                    item[1].path,
                                    item[1].start_line,
                                )
                            )
                            return [symbol for _score, symbol in positive[:limit]]
                except Exception:
                    pass
            return current_filter(self, query, symbols, limit=limit)

        setattr(semantic_symbol_filter, _RESEARCH_SEED_MARKER, True)
        research.ResearchCodeContext._semantic_symbol_filter = semantic_symbol_filter

    current_metrics = research._retrieval_metrics
    if not getattr(current_metrics, _RESEARCH_METRIC_MARKER, False):

        @wraps(current_metrics)
        def retrieval_metrics(
            query: str,
            text: str,
            *,
            path: str,
            symbols: Sequence[str],
            graph_hop: int | None,
            quality: Any,
            target_plan: str,
            example_plan: str,
        ) -> dict[str, float]:
            metrics = dict(
                current_metrics(
                    query,
                    text,
                    path=path,
                    symbols=symbols,
                    graph_hop=graph_hop,
                    quality=quality,
                    target_plan=target_plan,
                    example_plan=example_plan,
                )
            )
            metrics.setdefault(
                "plan_alignment",
                research._semantic_similarity(target_plan or query, example_plan or text),
            )
            return metrics

        setattr(retrieval_metrics, _RESEARCH_METRIC_MARKER, True)
        research._retrieval_metrics = retrieval_metrics

    current_weights = research._adaptive_weights
    if not getattr(current_weights, _RESEARCH_METRIC_MARKER, False):

        @wraps(current_weights)
        def adaptive_weights(
            query: str,
            metrics: Mapping[str, float] | None = None,
        ) -> dict[str, float]:
            weights = dict(current_weights(query, metrics))
            if "plan_alignment" not in weights:
                # Keep one complete complementary metric vector after any earlier
                # runtime optimization wrapper. Renormalize rather than changing
                # the absolute candidate-score scale.
                weights["plan_alignment"] = max(weights.values(), default=0.1)
                total = sum(max(0.0, float(value)) for value in weights.values())
                if total > 0:
                    weights = {
                        key: max(0.0, float(value)) / total
                        for key, value in weights.items()
                    }
            return weights

        setattr(adaptive_weights, _RESEARCH_METRIC_MARKER, True)
        research._adaptive_weights = adaptive_weights


def _harden_test_time_scaling() -> None:
    from . import inference_time_scaling

    current = inference_time_scaling._scaling_mode
    if getattr(current, _SCALING_MARKER, False):
        return

    @wraps(current)
    def scaling_mode() -> str:
        mode = str(current())
        if mode != "auto":
            return mode
        try:
            from .max_efficiency_runtime_contract import _active_parallelism

            slots = int(_active_parallelism())
        except Exception:
            slots = 1
        return "auto" if slots > 1 else "off"

    setattr(scaling_mode, _SCALING_MARKER, True)
    inference_time_scaling._scaling_mode = scaling_mode


def _has_callable(owner: Any, name: str) -> bool:
    if owner is None:
        return False
    try:
        value = getattr(owner, name)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return callable(value)


__all__ = ["harden_adaptive_execution"]
