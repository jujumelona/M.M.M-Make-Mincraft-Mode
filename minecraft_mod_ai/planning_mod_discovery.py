"""Host-owned catalog queries and receipts, separate from API/source evidence.

Catalog search is keyword retrieval (Modrinth /search), not an instruction-following
RAG endpoint. Build a query bundle from the approved capability, requirement wording,
research objective, and original task context; union hits by catalog identity. A zero-hit
path is retrieval evidence, never proof that no candidate exists.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

CATALOG_PROVIDERS = frozenset({"curseforge", "modrinth"})
_CATALOG_QUERY_CHARS = 420


def _query_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()[:_CATALOG_QUERY_CHARS]


def discovery_context(discovery: Mapping[str, Any]) -> str:
    """Bound model context, retaining the complete receipt in the planning state."""
    candidates = discovery.get("candidates") or []
    return json.dumps({
        "status": discovery.get("status"), "total_candidates": len(candidates),
        "candidate_preview": [{**row, "description": row.get("description", "")[:500]}
                              for row in candidates[:4]],
        "instruction": "Preview only; unlisted candidates are not rejected. Compatibility and license must be verified.",
    }, ensure_ascii=False)


def catalog_queries(
    state: Mapping[str, Any],
    research: Mapping[str, Any],
    *,
    prompt: str = "",
) -> list[str]:
    """Compile a recall-oriented catalog query bundle without replacing task context.

    ``semantic_capability`` is useful as one retrieval facet, but it is not allowed to
    become the entire search universe. Requirement wording, research intent and the
    original user prompt remain independent query paths so a narrow compiled label can
    fail without erasing the broader task.
    """
    requirement = next((row for row in state.get("decisions", [])
                        if isinstance(row, Mapping)
                        and row.get("decision_type") == "requirement"
                        and row.get("requirement_id") == research.get("requirement_ref")), {})
    capability = str(requirement.get("semantic_capability") or "")
    parts = [" ".join(re.findall(r"[\w]+", part.replace("_", " ")))
             for part in capability.split(".")]
    parts = [part for part in parts if part]

    candidates: list[str] = []
    if parts:
        # Relax conjunctive feature names so a narrow phrase cannot hide the ecosystem.
        candidates.extend([" ".join(parts), *parts])
    for value in (
        requirement.get("statement"),
        research.get("objective"),
        research.get("information_needed"),
        prompt,
    ):
        query = _query_text(value)
        if query:
            candidates.append(query)

    if candidates:
        return list(dict.fromkeys(candidates))

    # Reference/repository questions without a compiled capability retain their own
    # queries; do not borrow another requirement's identity or invent a mod name.
    return list(dict.fromkeys(_query_text(q) for q in research.get("queries", [])
                              if _query_text(q)))


def discovery_receipt(domain_id: str, grounded: Mapping[str, Any]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    for query in grounded.get("queries", []):
        for provider, receipt in (query.get("provider_receipts") or {}).items():
            if provider in CATALOG_PROVIDERS and isinstance(receipt, Mapping):
                attempts.append({"query": query.get("query", ""), "provider": provider,
                                 "status": receipt.get("status", "unknown"),
                                 "result_count": receipt.get("result_count", 0),
                                 "search_requests": receipt.get("search_requests", 0),
                                 "error": receipt.get("error", "")})
        for record in query.get("evidence_records", []):
            source_id = str(record.get("source_id") or "")
            if source_id.split(":", 1)[0] not in CATALOG_PROVIDERS:
                continue
            metadata = record.get("metadata") or {}
            candidate = candidates.setdefault(source_id, {
                "source_id": source_id, "name": record.get("title", ""),
                "url": record.get("url", ""),
                "source_url": metadata.get("source_url", ""),
                "versions": metadata.get("versions", []),
                "loaders": metadata.get("loaders", []),
                "license": metadata.get("license", None),
                "description": str(record.get("content") or "")[:1200],
                "queries": [], "compatibility": "not_verified",
                "reuse_authority": "verification_required",
            })
            text = str(query.get("query") or "")
            if text not in candidate["queries"]:
                candidate["queries"].append(text)
    # An unavailable optional provider does not invalidate a successful public catalog.
    searched = any(row["status"] == "available" and row["search_requests"] > 0
                   for row in attempts)
    lost_hits = any(row["result_count"] > 0 for row in attempts) and not candidates
    status = ("candidates_found" if candidates else "candidate_projection_failed" if lost_hits
              else "no_results" if searched else "catalog_unavailable")
    # Running a search to completion is not the same as satisfying candidate discovery.
    # Zero hits are a failed retrieval path and must not authorize downstream planning.
    complete = bool(candidates)
    return {"research_ref": domain_id, "status": status, "attempts": attempts,
            "candidates": list(candidates.values()), "complete": complete,
            "corrective_retrieval_required": bool(searched and not candidates and not lost_hits)}
