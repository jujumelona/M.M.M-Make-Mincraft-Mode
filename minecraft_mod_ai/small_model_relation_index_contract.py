from __future__ import annotations

from functools import wraps
from typing import Any, Sequence

from .small_model_rag_relations import derive_relations


def install(production_tools_module: Any) -> None:
    cls = production_tools_module.ProductionToolService
    current = cls.index_project_rag
    if getattr(current, "_mmm_dependency_relations", False):
        return
    fallback = getattr(current, "__wrapped__", None)

    @wraps(current)
    def indexed(
        self: Any,
        roots: Sequence[str],
        *,
        index_path: str = "rag/project-index.json",
        metadata: dict[str, Any],
        semantic: bool = False,
    ):
        enriched = dict(metadata)
        relations = derive_relations([self._existing_path(root) for root in roots])
        if relations:
            enriched["relations"] = relations
            enriched["relation_count"] = len(relations)
        try:
            return current(
                self,
                roots,
                index_path=index_path,
                metadata=enriched,
                semantic=semantic,
            )
        except Exception:
            repair_like = bool(enriched.get("source_commit")) and str(enriched.get("license", "")) == "project-local"
            if not callable(fallback) or not repair_like:
                raise
            return fallback(
                self,
                roots,
                index_path=index_path,
                metadata=enriched,
                semantic=False,
            )

    indexed._mmm_dependency_relations = True  # type: ignore[attr-defined]
    indexed.__wrapped__ = current  # type: ignore[attr-defined]
    cls.index_project_rag = indexed


__all__ = ["install"]
