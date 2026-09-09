"""Host-owned catalog queries and receipts, separate from API/source evidence.

Catalog search is keyword retrieval (Modrinth /search), not an instruction-following
RAG endpoint. Decompose the approved capability into focused and broader queries;
union their hits by catalog identity. No model call or hardcoded mod name is needed.
"""
from __future__ import annotations

import re
import json
from collections.abc import Mapping
from typing import Any

CATALOG_PROVIDERS = frozenset({"curseforge", "modrinth"})


def discovery_context(discovery: Mapping[str, Any]) -> str:
    """Bound model context, retaining the complete receipt in the planning state."""
    candidates = discovery.get("candidates") or []
    return json.dumps({
        "status": discovery.get("status"), "total_candidates": len(candidates),
        "candidate_preview": [{**row, "description": row.get("description", "")[:500]}
                              for row in candidates[:4]],
        "instruction": "Preview only; unlisted candidates are not rejected. Compatibility and license must be verified.",
    }, ensure_ascii=False)


def catalog_queries(state: Mapping[str, Any], research: Mapping[str, Any]) -> list[str]:
    requirement = next((row for row in state.get("decisions", [])
                        if isinstance(row, Mapping)
                        and row.get("decision_type") == "requirement"
                        and row.get("requirement_id") == research.get("requirement_ref")), {})
    capability = str(requirement.get("semantic_capability") or "")
    parts = [" ".join(re.findall(r"[\w]+", part.replace("_", " ")))
             for part in capability.split(".")]
    parts = [part for part in parts if part]
    if parts:
        # Relax conjunctive feature names so a narrow phrase cannot hide the ecosystem.
        return list(dict.fromkeys([" ".join(parts), *parts]))
    # Reference/repository questions without a compiled capability retain their own
    # queries; do not borrow another requirement's identity or invent a mod name.
    return list(dict.fromkeys(str(q).strip() for q in research.get("queries", [])
                              if str(q).strip()))


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
    return {"research_ref": domain_id, "status": status, "attempts": attempts,
            "candidates": list(candidates.values()),
            "complete": bool(candidates) or (searched and not lost_hits)}
