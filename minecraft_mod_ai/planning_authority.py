from __future__ import annotations

"""Small-model request and retrieval-planning authority.

The host owns source text, source offsets, stable requirement IDs, grounding validation,
request-graph integrity, and the active planning scope. Planning has one model-owned
operation and one host-owned operation:

1. one constrained batch that maps all authored clauses to approved semantic leaves;
2. the host deterministically maps the frozen graph to dependency hints and public
   retrieval queries without a second model turn.

This module deliberately does not perform clause-by-clause leaf discovery, leaf-by-leaf
Given/When/Then generation, or requirement-by-requirement query generation. Those O(N)
model-call paths duplicate the batch authorities in ``semantic_requirement_authority`` and
``authored_scope_research_contract`` and are especially harmful for small local models.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from . import authored_scope_research_contract as _retrieval
from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard
from . import semantic_requirement_authority as _semantic


def _compile_semantic_catalog(prompt: str, router: Any | None) -> dict[str, Any]:
    """Compile the whole authored request in one semantic model turn."""

    catalog = dict(_semantic.build_approved_requirement_catalog(prompt, router=router))
    if router is None:
        return catalog

    audit = dict(catalog.get("semantic_audit") or {})
    audit.update(
        {
            "normal_model_turns": 1,
            "semantic_model_turns": 1,
            "semantic_discovery_model_turns": 1,
            "semantic_detail_model_turns": 0,
            "max_repair_turns": 0,
            "generation_policy": "single_pass_constrained_batch",
            "semantic_generation_protocol": "all_clauses_one_structured_batch",
            "max_clauses_per_model_turn": len(_semantic._clause_records(prompt)),
            "max_semantic_leaves_per_detail_turn": 0,
            "model_generated_planning_json": False,
            "source_clause_index_owner": "host",
            "source_anchor_owner": "host",
            "source_grounding_owner": "host",
        }
    )
    catalog["semantic_audit"] = audit
    catalog["catalog_sha256"] = ""
    catalog["catalog_sha256"] = _evidence._hash_without(catalog, "catalog_sha256")
    _semantic.validate_approved_requirement_catalog(catalog, prompt=prompt)
    return catalog


def _enrich_retrieval_plan(
    prompt: str,
    catalog: Mapping[str, Any],
    router: Any | None,
) -> dict[str, Any]:
    """Enrich every frozen requirement through host-owned retrieval planning."""

    if router is None:
        return dict(catalog)

    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return dict(catalog)

    payload = _retrieval._call_retrieval_planner(router, prompt, requirements)
    plan = _retrieval._normalize_retrieval_plan(prompt, requirements, payload)

    enriched = deepcopy(dict(catalog))
    enriched_requirements: list[dict[str, Any]] = []
    edges: list[list[str]] = []
    for raw in enriched["requirements"]:
        item = dict(raw)
        requirement_id = str(item.get("requirement_id") or "")
        planned = plan[requirement_id]
        item["depends_on"] = list(planned["depends_on"])
        item["search_queries"] = list(planned["search_queries"])
        edges.extend([[dependency, requirement_id] for dependency in item["depends_on"]])
        enriched_requirements.append(item)

    enriched["requirements"] = enriched_requirements
    enriched["requirement_graph"] = {
        "node_ids": [str(item["requirement_id"]) for item in enriched_requirements],
        "edges": edges,
    }

    audit = dict(enriched.get("semantic_audit") or {})
    semantic_turns = int(audit.get("semantic_model_turns") or 1)
    audit.update(
        {
            "normal_model_turns": semantic_turns,
            "retrieval_model_turns": 0,
            "retrieval_query_planning": "host_deterministic_all_requirements",
            "max_requirements_per_query_turn": len(requirements),
            "model_owned_requirement_ids": False,
            "model_generated_planning_json": False,
            "dependency_edge_count": len(edges),
        }
    )
    enriched["semantic_audit"] = audit
    enriched["catalog_sha256"] = ""
    enriched["catalog_sha256"] = _evidence._hash_without(enriched, "catalog_sha256")
    _semantic.validate_approved_requirement_catalog(enriched, prompt=prompt)
    return enriched


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None,
) -> dict[str, Any]:
    """Compile request meaning and retrieval intent before any design/RAG execution.

    With a model router, normal planning performs exactly one bounded semantic model turn
    regardless of requirement count. Retrieval planning is deterministic host work.
    """

    return _enrich_retrieval_plan(
        prompt,
        _compile_semantic_catalog(prompt, router),
        router,
    )


@contextmanager
def authoritative_request_scope(
    prompt: str,
    catalog: Mapping[str, Any],
) -> Iterator[None]:
    """Expose one immutable request catalog to downstream read-only planning helpers."""

    token = _guard._ACTIVE_REQUEST_CATALOG.set((prompt, deepcopy(dict(catalog))))
    try:
        yield
    finally:
        _guard._ACTIVE_REQUEST_CATALOG.reset(token)


__all__ = [
    "authoritative_request_scope",
    "build_authoritative_request_catalog",
]
