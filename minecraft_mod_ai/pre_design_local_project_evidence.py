from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .rag_index import ProjectRAGIndex

_SCHEMA_VERSION = "mmm/pre-design-local-project-evidence-v1"
_QUERY_SCHEMA_VERSION = "mmm/pre-design-local-project-query-v1"


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _existing_project_index() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("rag/project-index.json"))
    workspace = os.environ.get("MMM_WORKSPACE", "").strip()
    if workspace:
        candidates.append(Path(workspace).expanduser() / "rag/project-index.json")

    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _search_index(index_path: Path | None, query: str) -> dict[str, Any]:
    if index_path is None:
        return {
            "schema_version": _QUERY_SCHEMA_VERSION,
            "status": "not_indexed",
            "hits": [],
        }
    try:
        result = ProjectRAGIndex(index_path).search_with_receipt(
            query,
            semantic=False,
            rerank=False,
        )
        return {
            "schema_version": _QUERY_SCHEMA_VERSION,
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
        }
    except Exception as exc:
        return {
            "schema_version": _QUERY_SCHEMA_VERSION,
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_local_project_evidence(
    research_brief: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect only current-project/code evidence for target-neutral pre-design.

    Official documentation is intentionally excluded: the canonical official evidence
    lane owns that corpus. This function never performs donor, ecosystem or remote
    repository discovery.
    """

    raw_domains = research_brief.get("domains", [])
    domains = [item for item in raw_domains if isinstance(item, Mapping)] if isinstance(raw_domains, list) else []
    index_path = _existing_project_index()
    rendered_domains: list[dict[str, Any]] = []
    query_count = 0

    for domain in domains:
        domain_id = str(domain.get("domain_id", "")).strip()
        raw_queries = domain.get("queries", [])
        queries = raw_queries if isinstance(raw_queries, list) else []
        rendered_queries: list[dict[str, Any]] = []
        for raw_query in queries:
            query = str(raw_query).strip()
            if not query:
                continue
            query_count += 1
            rendered_queries.append(
                {
                    "query": query,
                    "query_sha256": _sha256_text(query),
                    "code_rag": _search_index(index_path, query),
                }
            )
        rendered_domains.append(
            {
                "domain_id": domain_id,
                "queries": rendered_queries,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "available" if index_path is not None else "not_indexed",
        "domain_count": len(rendered_domains),
        "query_count": query_count,
        "code_index_status": "available" if index_path is not None else "not_indexed",
        "code_index_path": str(index_path) if index_path is not None else "",
        "domains": rendered_domains,
    }
    payload["research_sha256"] = _sha256(payload)
    return payload


__all__ = ["collect_local_project_evidence"]
