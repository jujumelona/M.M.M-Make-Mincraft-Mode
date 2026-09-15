"""Read current project source when no separately published RAG index exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .platform_catalog import adapter_from_project
from .project_index import ProjectIndex


def search_workspace_source(
    root: Path, query: str, *, limit: int, required_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Source search requires a query and a limit from 1 to 100.")
    adapter = adapter_from_project(root)
    metadata = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "java_version": adapter.java_version,
        "mapping_namespace": adapter.mappings_kind or "official",
    }
    # Unknown metadata must not be invented to satisfy a caller's evidence filter.
    matches = all(str(metadata.get(key, "")) == str(value)
                  for key, value in (required_metadata or {}).items())
    context = ProjectIndex(root).select(query=query, byte_budget=16 * 1024) if matches else {}
    hits = [
        {
            "source_path": row["path"], "path": row["path"],
            "text": row["content"], "sha256": row["sha256"],
            "truncated": row["truncated"], "metadata": metadata,
        }
        for row in context.get("files", [])[:limit]
    ]
    return {
        "schema_version": "mmm/code-rag-result-v1", "query": query, "hits": hits,
        "source_backend": "current_project_source",
        "receipt": {"status": "FOUND" if hits else "NOT_FOUND", "result_count": len(hits)},
    }
