from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from .centroid_vector_rag import direct_centroid_vector_search
from .model_router import ModelRouter
from .retrieval_adaptation import (
    _embedding_rows,
    adapt_query_vector,
    extract_hit_texts,
)

_SYMBOL = re.compile(r"\b(?:[A-Z][A-Za-z0-9_]{2,}|[a-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+|[A-Za-z0-9_./-]+\.(?:java|json|gradle|kts))\b")
_MC_VERSION = re.compile(r"(?<![0-9])(?:1\.)?[0-9]{1,2}(?:\.[0-9]{1,3}){1,2}(?![0-9])")
_SEARCH_CACHE_LOCK = RLock()
_SEARCH_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_SEARCH_CACHE_LIMIT = 256


def _route(query: str) -> str:
    value = query.casefold()
    if any(marker in value for marker in ("minecraft version", "fabric api version", "mapping", "yarn", "signature")) or _MC_VERSION.search(value):
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
            (False, False, "lexical"),
            (False, True, "lexical+rerank"),
            (True, True, "semantic+rerank"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    if route == "dependency":
        return (
            (False, False, "lexical+relations"),
            (False, True, "lexical+rerank+relations"),
            (True, True, "semantic+rerank+relations"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    if route == "global":
        return (
            (False, False, "lexical+global-relations"),
            (False, True, "lexical+rerank+global-relations"),
            (True, True, "semantic+rerank+global-relations"),
            (caller_semantic, caller_rerank, "caller-fallback"),
        )
    # Generic repair queries are the hottest path on CPU-backed Colab profiles.
    # Exact lexical evidence is cheap and often already sufficient; do not load the
    # 0.6B reranker/embedding models merely because the query lacks an explicit
    # symbol. Escalate only after objective retrieval quality remains weak.
    return (
        (False, False, "lexical"),
        (False, True, "lexical+rerank"),
        (True, True, "semantic+rerank"),
        (caller_semantic, caller_rerank, "caller-fallback"),
    )


def _centroid_terms(
    router: Any,
    query: str,
    result: Mapping[str, Any],
    *,
    texts: Sequence[str] | None = None,
    vector: Sequence[float] | None = None,
) -> str:
    """Choose centroid-nearest terms, reusing an already-computed q1 when available."""

    resolved_texts = texts if texts is not None else extract_hit_texts(result)
    resolved_vector = (
        vector
        if vector is not None
        else adapt_query_vector(router, query, resolved_texts)
    )
    if not resolved_vector or not resolved_texts:
        return ""

    tokens: list[str] = []
    seen: set[str] = set()
    for text in resolved_texts:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.$:/-]{2,96}", text):
            lowered = token.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            tokens.append(token)
            if len(tokens) >= 96:
                break
        if len(tokens) >= 96:
            break

    rows = _embedding_rows(router, tokens)
    candidates: list[tuple[float, str]] = []
    for token, values in zip(tokens, rows, strict=False):
        width = min(len(resolved_vector), len(values))
        if not width:
            continue
        dot = sum(resolved_vector[index] * values[index] for index in range(width))
        candidates.append((dot, token))
    candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    return " ".join(token for _score, token in candidates[:8])


def _resolve_index_target(service: Any, index_path: str) -> Path:
    resolve_fn = getattr(service, "_resolve", None)
    if callable(resolve_fn):
        try:
            target = resolve_fn(index_path, allow_root=True)
        except TypeError:
            target = resolve_fn(index_path)
    else:
        target = Path(index_path).expanduser().resolve()
    if target.is_dir():
        canonical = target / "rag" / "project-index.json"
        if canonical.is_file():
            return canonical
        return target
    if not target.exists() and callable(resolve_fn):
        try:
            canonical = resolve_fn("rag/project-index.json", allow_root=True)
        except TypeError:
            canonical = resolve_fn("rag/project-index.json")
        if canonical.is_file():
            return canonical
    return target


def _search_cache_key(
    service: Any,
    *,
    query: str,
    index_path: str,
    limit: int,
    semantic: bool,
    rerank: bool,
    required_metadata: dict[str, Any] | None,
) -> tuple[Any, ...]:
    target = _resolve_index_target(service, index_path)
    if target.exists():
        stat = target.stat()
        size = int(stat.st_size)
        mtime = int(stat.st_mtime_ns)
        ino = int(getattr(stat, "st_ino", 0) or 0)
    else:
        size, mtime, ino = 0, 0, 0
    metadata_key = json.dumps(
        required_metadata or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return (
        str(target),
        size,
        mtime,
        ino,
        str(getattr(service, "profile", "")),
        query,
        int(limit),
        bool(semantic),
        bool(rerank),
        metadata_key,
    )


def _search_cache_get(key: tuple[Any, ...]) -> dict[str, Any] | None:
    with _SEARCH_CACHE_LOCK:
        result = _SEARCH_CACHE.get(key)
        if result is None:
            return None
        copied = copy.deepcopy(result)
    copied["search_reused"] = True
    copied["search_reuse_reason"] = "exact_index_snapshot_and_query"
    return copied


def _search_cache_put(key: tuple[Any, ...], result: Mapping[str, Any]) -> None:
    with _SEARCH_CACHE_LOCK:
        if len(_SEARCH_CACHE) >= _SEARCH_CACHE_LIMIT and key not in _SEARCH_CACHE:
            _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
        _SEARCH_CACHE[key] = copy.deepcopy(dict(result))


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
        completed_attempts: set[tuple[str, bool, bool]] = set()

        for retry in (False, True):
            routed_query = _expanded(query, route, retry=retry)
            seen_modes: set[tuple[bool, bool]] = set()
            for use_semantic, use_rerank, mode in _modes(route, semantic, rerank):
                mode_key = (bool(use_semantic), bool(use_rerank))
                if mode_key in seen_modes:
                    continue
                seen_modes.add(mode_key)
                attempt_key = (routed_query, *mode_key)
                if attempt_key in completed_attempts:
                    continue
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
                completed_attempts.add(attempt_key)
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

        # Training-free low-coverage adaptation:
        # q0 -> first-pass top-K hit embeddings -> local centroid -> q1 blend ->
        # DIRECT q1 cosine retrieval against stored chunk embeddings -> rerank.
        if best is not None and route in {"semantic", "dependency", "global"}:
            router: Any | None = getattr(self, "router", None)
            if router is None:
                try:
                    router = ModelRouter(profile=self.profile)
                except Exception as exc:
                    errors.append(f"centroid-router:{type(exc).__name__}:{str(exc)[:240]}")
            if router is not None:
                centroid_texts: Sequence[str] | None = None
                q1_vector: Sequence[float] | None = None
                try:
                    centroid_texts = extract_hit_texts(best)
                    q1_vector = adapt_query_vector(router, query, centroid_texts)
                    if q1_vector:
                        direct = direct_centroid_vector_search(
                            _resolve_index_target(self, index_path),
                            query=query,
                            q1_vector=q1_vector,
                            router=router,
                            limit=limit,
                            required_metadata=required_metadata,
                        )
                        if direct is not None:
                            direct = dict(direct)
                            usable, coverage, relevance, count = _quality(direct)
                            direct["query"] = query
                            direct["expanded_query"] = query
                            direct["task_route"] = route
                            direct["retrieval_retry"] = True
                            direct["centroid_adaptation"] = True
                            direct["centroid_vector_direct"] = True
                            if errors:
                                direct["fallback_errors"] = list(errors)
                            if usable or (count, coverage, relevance) > best_score:
                                return direct
                except Exception as exc:
                    errors.append(f"centroid-vector:{type(exc).__name__}:{str(exc)[:240]}")

                # Compatibility fallback for semantic indexes that cannot expose
                # stored vectors directly. Reuse q1 from the direct attempt instead
                # of re-embedding the same query and same first-pass hits.
                try:
                    terms = _centroid_terms(
                        router,
                        query,
                        best,
                        texts=centroid_texts,
                        vector=q1_vector,
                    )
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
                        adapted["retrieval_mode"] = "centroid-adapted-semantic+rerank-fallback"
                        adapted["retrieval_retry"] = True
                        adapted["centroid_adaptation"] = True
                        adapted["centroid_vector_direct"] = False
                        if usable or (count, coverage, relevance) > best_score:
                            return adapted
                except Exception as exc:
                    errors.append(f"centroid-text-fallback:{type(exc).__name__}:{str(exc)[:240]}")

        if best is not None:
            best["query"] = query
            best["task_route"] = route
            best["retrieval_retry"] = True
            best["retrieval_quality_warning"] = "coverage_or_relevance_below_target"
            best["centroid_adaptation"] = False
            best["centroid_vector_direct"] = False
            if errors:
                best["fallback_errors"] = errors
            return best
        return {
            "schema_version": "mmm/code-rag-result-v1",
            "query": query,
            "hits": [],
            "task_route": route,
            "retrieval_mode": "empty_fallback",
            "retrieval_retry": False,
            "fallback_errors": errors,
            "receipt": {
                "query": query,
                "total_candidates": 0,
                "status": "NOT_FOUND",
                "warnings": errors,
            },
        }

    @wraps(searched)
    def cached_search(
        self: Any,
        query: str,
        *,
        index_path: str = "rag/project-index.json",
        limit: int = 8,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ):
        key = _search_cache_key(
            self,
            query=query,
            index_path=index_path,
            limit=limit,
            semantic=semantic,
            rerank=rerank,
            required_metadata=required_metadata,
        )
        cached = _search_cache_get(key)
        if cached is not None:
            return cached
        result = searched(
            self,
            query,
            index_path=index_path,
            limit=limit,
            semantic=semantic,
            rerank=rerank,
            required_metadata=required_metadata,
        )
        _search_cache_put(key, result)
        return result

    cached_search._mmm_task_routed_code_search = True  # type: ignore[attr-defined]
    cached_search._mmm_snapshot_search_reuse = True  # type: ignore[attr-defined]
    cached_search.__wrapped__ = searched  # type: ignore[attr-defined]
    cls.search_code_rag = cached_search


__all__ = ["install"]