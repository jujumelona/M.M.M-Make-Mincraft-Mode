"""Task-wide retrieval and requirement-local evidence; neither authorizes source reuse.

Query expansion preserves the original information need (Manning et al., IR chapter 9).
Lexical matches are inspectable retrieval evidence, never a semantic implementation proof.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_STOP = frozenset(["the", "and", "for", "with", "from", "that", "this", "into", "can", "will", "are", "has", "have", "after", "before", "through", "to", "of", "in", "on", "by", "as", "an", "is", "be", "it", "players", "player", "minecraft", "fabric", "forge", "neoforge", "mod", "mods", "implementation", "concrete", "patterns", "support", "artifacts", "useful", "find", "options", "source", "code", "api", "correctly", "requirement", "systems", "system", "feature"])


def fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def terms(value: Any) -> list[str]:
    return list(dict.fromkeys(word for word in re.findall(
        r"[a-z0-9]+|[가-힣]{2,}", str(value or "").casefold(),
    ) if len(word) > 2 and word not in _STOP))


def requirement_for(state: Mapping[str, Any], research: Mapping[str, Any]) -> Mapping[str, Any]:
    return next((row for row in state.get("decisions", [])
                 if isinstance(row, Mapping) and row.get("decision_type") == "requirement"
                 and row.get("requirement_id") == research.get("requirement_ref")), {})


def query_context(state: Mapping[str, Any], research: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    """Keep full task/requirement bindings independently of provider query strings."""
    return {
        "original_task": str(state.get("original_prompt") or prompt),
        "requirement": dict(requirement_for(state, research)),
        "sibling_requirements": [dict(row) for row in state.get("decisions", [])
                                 if isinstance(row, Mapping)
                                 and row.get("decision_type") == "requirement"
                                 and row.get("requirement_id") != research.get("requirement_ref")],
    }


def expansion_queries(context: Mapping[str, Any]) -> list[str]:
    """Finite authored vocabulary expansion, not repeated retries or invented mod names."""
    rows = [context.get("requirement", {}), *context.get("sibling_requirements", [])]
    queries = []
    for row in rows:
        parts = [" ".join(terms(part)) for part in str(row.get("semantic_capability") or "").split(".")]
        queries.extend([" ".join(parts), *parts, *terms(row.get("statement"))])
    queries.extend(terms(context.get("original_task")))
    return list(dict.fromkeys(q for q in queries if q))


def global_grounded_pool(grounded_domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Union catalog/repository bodies across requirements without dropping provenance."""
    queries: list[dict[str, Any]] = []
    for domain_id, grounded in grounded_domains.items():
        for raw in grounded.get("queries", []):
            records = [dict(record) for record in raw.get("evidence_records", [])
                       if str(record.get("source_id") or "").split(":", 1)[0]
                       in {"modrinth", "curseforge", "github"}]
            if records:
                queries.append({**raw, "origin_domain_id": domain_id, "evidence_records": records})
    return {"queries": queries}


def requirement_candidate_trace(
    requirement: Mapping[str, Any], pool: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every candidate against this requirement's own authored facets.

    No sibling intent or generic research instructions enter the relevance vocabulary.
    Missing vocabulary is unresolved, rather than evidence that a candidate is irrelevant.
    """
    capability = str(requirement.get("semantic_capability") or "")
    facets = [[word] for word in terms(capability)]
    if not facets:
        facets = [[word] for word in terms(requirement.get("statement"))]
    candidates: dict[str, dict[str, Any]] = {}
    for query in pool.get("queries", []):
        for record in query.get("evidence_records", []):
            source_id = str(record.get("source_id") or "")
            candidate = candidates.setdefault(source_id, {
                "source_id": source_id, "requirement_ref": requirement.get("requirement_id"),
                "requirement_sha256": fingerprint(requirement), "origin_domains": [],
                "query_sha256": [], "evidence": [], "matched_facets": [],
                "source_reuse_authority": "verification_required",
            })
            origin = query.get("origin_domain_id")
            if origin and origin not in candidate["origin_domains"]:
                candidate["origin_domains"].append(origin)
            query_sha = query.get("query_sha256") or fingerprint(query.get("query", ""))
            if query_sha not in candidate["query_sha256"]:
                candidate["query_sha256"].append(query_sha)
            content = str(record.get("content") or "")
            # Preserve exact source chunks and hashes, including evidence beyond previews.
            for chunk in re.split(r"(?:\r?\n){2,}|(?<=[.!?])\s+", content):
                words = set(terms(chunk))
                matched = [index for index, facet in enumerate(facets) if words.intersection(facet)]
                if not matched:
                    continue
                evidence = {"exact_excerpt": chunk, "content_sha256": "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            "source_url": record.get("url", ""), "matched_facets": matched}
                if evidence not in candidate["evidence"]:
                    candidate["evidence"].append(evidence)
                candidate["matched_facets"] = sorted(set(candidate["matched_facets"]) | set(matched))
    covered: set[int] = set()
    for candidate in candidates.values():
        covered.update(candidate["matched_facets"])
        candidate["status"] = "lexical_evidence" if candidate["evidence"] else "unresolved_relevance"
    missing = [facet for index, facet in enumerate(facets) if index not in covered]
    return {"requirement_ref": requirement.get("requirement_id"), "facets": facets,
            "candidates": list(candidates.values()), "missing_facets": missing,
            "coverage_complete": bool(candidates) and bool(facets) and not missing,
            "semantic_implementation_proof": False, "pool_sha256": fingerprint(pool)}


def assert_candidate_research_complete(state: Mapping[str, Any]) -> None:
    """Recompute admission at the detailed-planner boundary, including restored states."""
    for research in state.get("research_queue", []):
        if not isinstance(research, Mapping) or not research.get("requirement_ref"):
            continue
        if not set(research.get("source_kinds") or []).intersection({"repository", "existing_mods"}):
            continue
        discovery = research.get("mod_discovery") or {}
        pool = state.get("task_candidate_pool") or {}
        trace = requirement_candidate_trace(requirement_for(state, research), pool)
        saved_trace = research.get("candidate_trace") or {}
        if (research.get("status") != "complete" or research.get("research_state") != "COMPLETE"
                or not discovery.get("complete") or not discovery.get("candidates")
                or not trace["coverage_complete"] or saved_trace != trace):
            raise ValueError(
                "PLANNING_IMPLEMENTATION_RESEARCH_BLOCKED: candidate evidence is missing or stale for "
                + str(research.get("requirement_ref"))
            )
