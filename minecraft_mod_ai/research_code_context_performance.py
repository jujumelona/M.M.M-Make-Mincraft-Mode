from __future__ import annotations

"""Narrow hot-path fixes for the canonical repository-research owner.

Do not duplicate retrieval, graph, or scoring implementations here. Reuse their
results while pruning irrelevant roots, bounding partial-graph evidence, folding the
accidental ninth metric back into the declared eight-signal fusion, and making exact
draft evolution reach a host fixed point after one successful pass.
"""

from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_research_code_context_performance_v1"


def _install_entry_pruning(module: Any) -> None:
    cls = module.ResearchCodeContext
    current = cls._entry_points
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def entry_points(self: Any, query: str):
        selected = current(self, query)
        query_tokens = module._tokens(query)
        if not query_tokens or len(selected) <= 1:
            return selected
        relevant = []
        for symbol in selected:
            unit = self.units.get(symbol.path)
            tokens = module._tokens(symbol.name) | module._tokens(symbol.signature)
            tokens |= module._tokens(symbol.path)
            if unit is not None:
                tokens |= module._tokens(unit.package)
                tokens |= module._tokens(unit.imports)
            if query_tokens & tokens:
                relevant.append(symbol)
        # A semantic-only query may legitimately have no literal root match. Keep
        # canonical ranking then; otherwise avoid zero-literal graph roots.
        return relevant or selected

    setattr(entry_points, _MARKER, True)
    entry_points._mmm_semantic_entry_filter_v1 = True  # type: ignore[attr-defined]
    entry_points.__wrapped__ = current  # type: ignore[attr-defined]
    cls._entry_points = entry_points


def _install_graph_cap(module: Any) -> None:
    cls = module.ResearchCodeContext
    current = cls._expand_partial_graph
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def expand_partial_graph(self: Any, entries: Any, *, query: str = ""):
        # Keep the canonical graph implementation as the only traversal owner. The
        # repository-research evidence surface is limited to two hops so a broad
        # third-hop tail cannot enter scoring/context.
        return [
            (symbol, hop)
            for symbol, hop in current(self, entries, query=query)
            if hop <= 2
        ]

    setattr(expand_partial_graph, _MARKER, True)
    expand_partial_graph._mmm_two_hop_graph_v1 = True  # type: ignore[attr-defined]
    expand_partial_graph.__wrapped__ = current  # type: ignore[attr-defined]
    cls._expand_partial_graph = expand_partial_graph


def _install_eight_metric_fusion(module: Any) -> None:
    current_metrics = module._retrieval_metrics
    if not getattr(current_metrics, _MARKER, False):

        @wraps(current_metrics)
        def retrieval_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            metrics = dict(current_metrics(*args, **kwargs))
            alignment = float(metrics.pop("plan_alignment", 0.0))
            if alignment:
                # PERC plan alignment remains represented as a structural refinement
                # rather than a second independently weighted semantic signal.
                structure = float(metrics.get("structure", 0.0))
                metrics["structure"] = min(1.0, 0.80 * structure + 0.20 * alignment)
            return metrics

        setattr(retrieval_metrics, _MARKER, True)
        retrieval_metrics.__wrapped__ = current_metrics  # type: ignore[attr-defined]
        module._retrieval_metrics = retrieval_metrics

    current_weights = module._adaptive_weights
    if getattr(current_weights, _MARKER, False):
        return

    @wraps(current_weights)
    def adaptive_weights(
        query: str,
        metrics: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        weights = dict(current_weights(query, metrics))
        alignment_weight = float(weights.pop("plan_alignment", 0.0))
        weights["quality"] = float(weights.get("quality", 0.0)) + alignment_weight
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {key: value / total for key, value in weights.items()}

    setattr(adaptive_weights, _MARKER, True)
    adaptive_weights.__wrapped__ = current_weights  # type: ignore[attr-defined]
    module._adaptive_weights = adaptive_weights


def _install_generation_fixed_point(module: Any) -> None:
    cls = module.ResearchCodeContext
    current = cls.evolve_from_generation
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def evolve_from_generation(self: Any, text: str):
        key = module._sha(str(text))
        seen = getattr(self, "_mmm_generation_evolution_seen", None)
        if not isinstance(seen, set):
            seen = set()
            self._mmm_generation_evolution_seen = seen
        if key in seen:
            violations = self.monitor.validate_model_output(text)
            return (self.bundle(), violations) if violations else (None, ())
        result = current(self, text)
        seen.add(key)
        return result

    setattr(evolve_from_generation, _MARKER, True)
    evolve_from_generation._mmm_generation_fixed_point_v1 = True  # type: ignore[attr-defined]
    evolve_from_generation.__wrapped__ = current  # type: ignore[attr-defined]
    cls.evolve_from_generation = evolve_from_generation


def harden(module: Any) -> None:
    _install_entry_pruning(module)
    _install_graph_cap(module)
    _install_eight_metric_fusion(module)
    _install_generation_fixed_point(module)


__all__ = ["harden"]