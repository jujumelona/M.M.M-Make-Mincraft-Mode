from __future__ import annotations

from functools import wraps
from typing import Any


def _query(value: str) -> str:
    lowered = value.casefold()
    if any(marker in lowered for marker in ("minecraft version", "fabric api version", "mapping", "yarn", "signature")):
        return value
    return value + "\nrelated dependency call chain"


def install(production_tools_module: Any) -> None:
    cls = production_tools_module.ProductionToolService
    current = cls.search_code_rag
    if getattr(current, "_mmm_hybrid_dependency_search", False):
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
        expanded = _query(query)
        errors: list[str] = []
        for use_semantic, use_rerank, mode in (
            (True, True, "semantic+rerank+relations"),
            (False, True, "lexical+rerank+relations"),
            (semantic, rerank, "caller-fallback"),
        ):
            try:
                result = dict(
                    current(
                        self,
                        expanded,
                        index_path=index_path,
                        limit=limit,
                        semantic=use_semantic,
                        rerank=use_rerank,
                        required_metadata=required_metadata,
                    )
                )
                result["query"] = query
                result["expanded_query"] = expanded
                result["retrieval_mode"] = mode
                if errors:
                    result["fallback_errors"] = errors
                return result
            except Exception as exc:
                errors.append(f"{mode}:{type(exc).__name__}:{str(exc)[:240]}")
        raise RuntimeError("Code RAG failed all hybrid modes: " + " | ".join(errors))

    searched._mmm_hybrid_dependency_search = True  # type: ignore[attr-defined]
    searched.__wrapped__ = current  # type: ignore[attr-defined]
    cls.search_code_rag = searched


__all__ = ["install"]
