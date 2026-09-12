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
    """Relax within authored topic anchors; never turn prose into singleton searches."""
    rows = [context.get("requirement", {}), *context.get("sibling_requirements", [])]
    queries = []
    for row in rows:
        parts = terms(row.get("semantic_capability"))
        if not parts:
            continue
        anchor = parts[0]
        queries.extend([" ".join(parts), anchor])
        queries.extend(f"{anchor} {word}" for word in terms(row.get("statement"))
                       if word != anchor)
    return list(dict.fromkeys(q for q in queries if q))


def global_grounded_pool(grounded_domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Union catalog/repository bodies across requirements without dropping provenance."""
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for domain_id, grounded in grounded_domains.items():
        for raw in grounded.get("queries", []):
            for record in raw.get("evidence_records", []):
                source_id = str(record.get("source_id") or "")
                if source_id.split(":", 1)[0] not in {"modrinth", "curseforge", "github"}:
                    continue
                body_sha = hashlib.sha256(str(record.get("content") or "").encode("utf-8")).hexdigest()
                row = sources.setdefault((source_id, body_sha), {
                    "query": raw.get("query", ""), "query_sha256": raw.get("query_sha256", ""),
                    "origin_domain_ids": [], "retrieval_queries": [], "evidence_records": [dict(record)],
                })
                for origin in raw.get("origin_domain_ids") or [raw.get("origin_domain_id") or domain_id]:
                    if origin not in row["origin_domain_ids"]:
                        row["origin_domain_ids"].append(origin)
                for query_text in raw.get("retrieval_queries") or [raw.get("query", "")]:
                    if query_text not in row["retrieval_queries"]:
                        row["retrieval_queries"].append(query_text)
    return {"schema_version": "mmm/task-candidate-pool-v2", "queries": list(sources.values())}


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
            for origin in query.get("origin_domain_ids") or [query.get("origin_domain_id")]:
                if origin and origin not in candidate["origin_domains"]:
                    candidate["origin_domains"].append(origin)
            for query_text in query.get("retrieval_queries") or [query.get("query", "")]:
                query_sha = fingerprint(query_text)
                if query_sha not in candidate["query_sha256"]:
                    candidate["query_sha256"].append(query_sha)
            content = str(record.get("content") or "")
            body_sha = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
            offset = 0
            # Preserve exact source chunks and hashes, including evidence beyond previews.
            for chunk in re.split(r"(?:\r?\n){2,}|(?<=[.!?])\s+", content):
                start = content.find(chunk, offset)
                offset = start + len(chunk)
                words = set(terms(chunk))
                matched = [index for index, facet in enumerate(facets) if words.intersection(facet)]
                if not matched:
                    continue
                evidence = {"char_start": start, "char_end": offset, "content_sha256": body_sha,
                            "matched_facets": matched}
                if evidence not in candidate["evidence"]:
                    candidate["evidence"].append(evidence)
                candidate["matched_facets"] = sorted(set(candidate["matched_facets"]) | set(matched))
    covered: set[int] = set()
    for candidate in candidates.values():
        # Tokens spread over unrelated candidates cannot form a supported requirement.
        coherent = bool(facets) and len(candidate["matched_facets"]) == len(facets)
        if coherent:
            covered.update(candidate["matched_facets"])
        candidate["status"] = "lexical_evidence" if coherent else "unresolved_relevance"
    missing = [facet for index, facet in enumerate(facets) if index not in covered]
    return {"requirement_ref": requirement.get("requirement_id"), "facets": facets,
            "candidates": list(candidates.values()), "missing_facets": missing,
            "lexical_coverage_complete": bool(candidates) and bool(facets) and not missing,
            "coverage_complete": False,
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
        from .planning_semantic_research import validate_semantic_review
        review = validate_semantic_review(requirement_for(state, research), pool,
                                          saved_trace.get("semantic_review") or {})
        trace["semantic_review"] = review
        trace["coverage_complete"] = review["complete"]
        if (research.get("status") != "complete" or research.get("research_state") != "COMPLETE"
                or not discovery.get("complete") or not discovery.get("candidates")
                or not trace["coverage_complete"] or saved_trace != trace):
            raise ValueError(
                "PLANNING_IMPLEMENTATION_RESEARCH_BLOCKED: candidate evidence is missing or stale for "
                + str(research.get("requirement_ref"))
            )
