from __future__ import annotations

import re
from functools import wraps
from typing import Any, Mapping

from .retrieval_adaptation import adapt_query_vector, extract_hit_texts

_SYMBOL = re.compile(r"\b(?:[A-Z][A-Za-z0-9_]{2,}|[a-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+|[A-Za-z0-9_./-]+\.(?:java|json|gradle|kts))\b")


def _route(query: str) -> str:
    value = query.casefold()
    if any(marker in value for marker in ("minecraft version", "fabric api version", "mapping", "yarn", "signature", "1.20.1")):
        return "exact_version"
    if any(marker in value for marker in ("dependency", "depends", "call chain", "caller", "callee", "import", "extends", "implements", "의존", "호출", "연결")):
        return "dependency"
    if any(marker in value for marker in ("entire project", "whole project", "architecture", "overview", "global", "전체 구조", "프로젝트 전체")):
        return "global"
    if _SYMBOL.search(query):
        return "exact_symbol"
    return "semantic"


def _expanded(query: str, route: str, *, retry: bool = False) -> str:
    if route == "dependency":
        suffix = "related dependency call chain imports callers callees"
    elif route == "global":
        suffix = "entire project architecture related modules dependencies"
    elif route == "exact_symbol":
        suffix = "exact identifier declaration reference usage" if retry else ""
    elif route == "semantic":
        suffix = "implementation symbol class method resource" if retry else ""
    else:
        suffix = ""
    return query if not suffix else query + "\n" + suffix


def _quality(result: Mapping[str, Any]) -> tuple[bool, float, float, int]:
    receipt = result.get("receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    try:
        coverage = float(receipt.get("coverage_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0
    try:
        relevance = float(receipt.get("relevance_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        relevance = 0.0
    try:
        count = int(receipt.get("result_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    usable = count > 0 and coverage >= 0.50 and relevance > 0.0
    return usable, coverage, relevance, count


def _modes(route: str, caller_semantic: bool, caller_rerank: bool):
    if route in {"exact_version", "exact_symbol"}:
        return (
            (False, True, "lexical+rerank"),
            (True, True, "semantic+rerank"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    if route == "dependency":
        return (
            (True, True, "semantic+rerank+relations"),
            (False, True, "lexical+rerank+relations"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    if route == "global":
        return (
            (False, True, "lexical+rerank+global-relations"),
            (True, True, "semantic+rerank+global-relations"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    return (
        (True, True, "semantic+rerank"),
        (False, True, "lexical+rerank"),
        (caller_semantic, caller_rerank, "caller-fallback"),
    )


def _centroid_terms(router: Any, query: str, result: Mapping[str, Any]) -> str:
    texts = extract_hit_texts(result)
    vector = adapt_query_vector(router, query, texts)
    if not vector or not texts:
        return ""
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for text in texts:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.$:/-]{2,96}", text):
            lowered = token.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            try:
                embedded = router.embed(token)
                values = [float(item) for item in embedded]
            except Exception:
                continue
            width = min(len(vector), len(values))
            if not width:
                continue
            dot = sum(vector[index] * values[index] for index in range(width))
            candidates.append((dot, token))
            if len(candidates) >= 96:
                break
        if len(candidates) >= 96:
            break
    candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    return " ".join(token for _score, token in candidates[:8])


def install(production_tools_module: Any) -> None:
    cls = production_tools_module.ProductionToolService
    current = cls.search_code_rag
    if getattr(current, "_mmm_task_routed_code_search", False):
        return

    @wraps(current)
    def searched(
        self: Any,
        query: str,
        *,
        index_path: str = "rag/project-index.json",
        limit: int = 8,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ):
        route = _route(query)
        errors: list[str] = []
        best: dict[str, Any] | None = None
        best_score = (-1, -1.0, -1.0)
        router = getattr(self, "router", None)

        for retry in (False, True):
            routed_query = _expanded(query, route, retry=retry)
            for use_semantic, use_rerank, mode in _modes(route, semantic, rerank):
                try:
                    result = dict(
                        current(
                            self,
                            routed_query,
                            index_path=index_path,
                            limit=limit,
                            semantic=use_semantic,
                            rerank=use_rerank,
                            required_metadata=required_metadata,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{mode}:{type(exc).__name__}:{str(exc)[:240]}")
                    continue
                usable, coverage, relevance, count = _quality(result)
                score = (count, coverage, relevance)
                if score > best_score:
                    best = result
                    best_score = score
                    best["retrieval_mode"] = mode
                    best["expanded_query"] = routed_query
                if usable:
                    result["query"] = query
                    result["expanded_query"] = routed_query
                    result["task_route"] = route
                    result["retrieval_mode"] = mode
                    result["retrieval_retry"] = retry
                    if errors:
                        result["fallback_errors"] = errors
                    return result
            if retry:
                break

        if best is not None and router is not None and route in {"semantic", "dependency", "global"}:
            try:
                terms = _centroid_terms(router, query, best)
                if terms:
                    adapted_query = query + "\nlocal-adaptation: " + terms
                    adapted = dict(
                        current(
                            self,
                            adapted_query,
                            index_path=index_path,
                            limit=limit,
                            semantic=True,
                            rerank=True,
                            required_metadata=required_metadata,
                        )
                    )
                    usable, coverage, relevance, count = _quality(adapted)
                    adapted["query"] = query
                    adapted["expanded_query"] = adapted_query
                    adapted["task_route"] = route
                    adapted["retrieval_mode"] = "centroid-adapted-semantic+rerank"
                    adapted["retrieval_retry"] = True
                    adapted["centroid_adaptation"] = True
                    if usable or (count, coverage, relevance) > best_score:
                        return adapted
            except Exception as exc:
                errors.append(f"centroid-adaptation:{type(exc).__name__}:{str(exc)[:240]}")

        if best is not None:
            best["query"] = query
            best["task_route"] = route
            best["retrieval_retry"] = True
            best["retrieval_quality_warning"] = "coverage_or_relevance_below_target"
            best["centroid_adaptation"] = False
            if errors:
                best["fallback_errors"] = errors
            return best
        raise RuntimeError("Code RAG failed all routed modes: " + " | ".join(errors))

    searched._mmm_task_routed_code_search = True  # type: ignore[attr-defined]
    searched.__wrapped__ = current  # type: ignore[attr-defined]
    cls.search_code_rag = searched


__all__ = ["install"]
