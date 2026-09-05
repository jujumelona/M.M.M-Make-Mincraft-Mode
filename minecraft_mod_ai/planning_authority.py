from __future__ import annotations

"""Authoritative request planning and host-owned causal dependency enrichment."""

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from . import authored_scope_research_contract as _retrieval
from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard
from . import semantic_requirement_authority as _semantic
from .minecraft_template_catalog import selected_predecessor_capabilities
from .root_cause_trace import emit_root_cause, trace_scope

_STATE_TOKEN = re.compile(r"[A-Za-z0-9_]+|[가-힣]+", re.UNICODE)
_STATE_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "from",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "with",
        "for",
        "by",
        "can",
        "may",
        "player",
        "players",
        "game",
        "mode",
        "feature",
        "mechanic",
        "requested",
        "behavior",
        "outcome",
        "state",
        "exists",
        "플레이어",
        "게임",
        "모드",
        "상태",
    }
)
# Lexical state inference is a secondary authored-evidence path.  The primary path for
# known Minecraft capabilities is the host feature model below.
_PRODUCER_STEMS = frozenset(
    {
        "acquir",
        "arriv",
        "assembl",
        "build",
        "collect",
        "construct",
        "creat",
        "discover",
        "establish",
        "generat",
        "obtain",
        "open",
        "spawn",
        "travel",
        "unlock",
    }
)


def _stem_state_token(token: str) -> str:
    value = token.casefold().strip("_")
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("es"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _state_terms(value: Any) -> frozenset[str]:
    terms: set[str] = set()
    for raw in _STATE_TOKEN.findall(str(value or "")):
        token = _stem_state_token(raw)
        if len(token) <= 1 or token in _STATE_STOP:
            continue
        terms.add(token)
    return frozenset(terms)


def _source_start(item: Mapping[str, Any]) -> int:
    span = item.get("source_span")
    if isinstance(span, Mapping) and type(span.get("char_start")) is int:
        return int(span["char_start"])
    return 2**63 - 1


def _declared_dependencies(item: Mapping[str, Any], requirement_id: str) -> list[str]:
    result: list[str] = []
    raw = item.get("depends_on")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result.extend(
            str(value).strip()
            for value in raw
            if str(value).strip() and str(value).strip() != requirement_id
        )
    unlock = item.get("unlock_policy")
    if isinstance(unlock, Mapping):
        refs = unlock.get("required_requirement_refs")
        if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes, bytearray)):
            result.extend(
                str(value).strip()
                for value in refs
                if str(value).strip() and str(value).strip() != requirement_id
            )
    return list(dict.fromkeys(result))


def _observable_text(item: Mapping[str, Any], field: str) -> str:
    behavior = item.get("observable_behavior")
    if not isinstance(behavior, Mapping):
        return ""
    return str(behavior.get(field) or "").strip()


def _feature_model_dependencies(
    requirements: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Derive safe edges only between capabilities that the user actually selected.

    Ontology/default and domain integration edges never synthesize a missing feature.  A
    predecessor is bound only when exactly one explicit requirement in this request owns
    that capability; ambiguity intentionally yields no inferred edge.
    """

    selected = [
        str(item.get("capability") or "").strip().casefold()
        for item in requirements
        if isinstance(item, Mapping) and str(item.get("capability") or "").strip()
    ]
    ids_by_capability: dict[str, list[str]] = {}
    for item in requirements:
        if not isinstance(item, Mapping):
            continue
        capability = str(item.get("capability") or "").strip().casefold()
        requirement_id = str(item.get("requirement_id") or "").strip()
        if capability and requirement_id:
            ids_by_capability.setdefault(capability, []).append(requirement_id)

    result: dict[str, list[str]] = {}
    provenance: list[dict[str, Any]] = []
    for item in requirements:
        if not isinstance(item, Mapping):
            continue
        requirement_id = str(item.get("requirement_id") or "").strip()
        capability = str(item.get("capability") or "").strip().casefold()
        if not requirement_id or not capability:
            continue
        dependencies: list[str] = []
        for predecessor_capability in selected_predecessor_capabilities(
            capability,
            selected,
        ):
            candidates = [
                candidate
                for candidate in ids_by_capability.get(predecessor_capability, ())
                if candidate != requirement_id
            ]
            if len(candidates) != 1:
                if candidates:
                    provenance.append(
                        {
                            "requirement_id": requirement_id,
                            "capability": capability,
                            "predecessor_capability": predecessor_capability,
                            "method": "host_feature_model_ambiguous_predecessor_skip",
                            "candidate_requirement_refs": candidates,
                        }
                    )
                continue
            dependency = candidates[0]
            if dependency not in dependencies:
                dependencies.append(dependency)
                provenance.append(
                    {
                        "dependency": dependency,
                        "requirement_id": requirement_id,
                        "capability": capability,
                        "predecessor_capability": predecessor_capability,
                        "method": "host_feature_model_selected_predecessor",
                    }
                )
        result[requirement_id] = dependencies
    return result, provenance


def _observable_state_dependencies(
    requirements: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Conservatively infer authored producer->consumer edges from observable states."""

    ordered = sorted(
        (dict(item) for item in requirements if isinstance(item, Mapping)),
        key=lambda item: (_source_start(item), str(item.get("requirement_id") or "")),
    )
    result: dict[str, list[str]] = {}
    provenance: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []

    for item in ordered:
        requirement_id = str(item.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        dependencies: list[str] = []
        given_terms = _state_terms(_observable_text(item, "given"))
        candidates: list[tuple[tuple[int, int], str, tuple[str, ...]]] = []
        if given_terms:
            for producer in previous:
                producer_id = str(producer.get("requirement_id") or "").strip()
                if not producer_id or producer_id == requirement_id:
                    continue
                then_text = _observable_text(producer, "then")
                then_terms = _state_terms(then_text)
                shared = tuple(sorted(given_terms & then_terms))
                if not shared:
                    continue
                producer_actions = then_terms & _PRODUCER_STEMS
                if not producer_actions:
                    continue
                candidates.append(((len(shared), len(producer_actions)), producer_id, shared))
        if candidates:
            best_score = max(score for score, _producer, _shared in candidates)
            best = [
                (producer, shared)
                for score, producer, shared in candidates
                if score == best_score
            ]
            if len(best) == 1:
                producer_id, shared = best[0]
                dependencies.append(producer_id)
                producer = next(
                    previous_item
                    for previous_item in previous
                    if str(previous_item.get("requirement_id") or "").strip() == producer_id
                )
                provenance.append(
                    {
                        "dependency": producer_id,
                        "requirement_id": requirement_id,
                        "method": "host_observable_state_producer",
                        "shared_state_terms": list(shared),
                        "producer_then": _observable_text(producer, "then"),
                        "consumer_given": _observable_text(item, "given"),
                    }
                )
        result[requirement_id] = dependencies
        previous.append(item)
    return result, provenance


def _host_causal_dependencies(
    requirements: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Combine declared, feature-model, and strict authored-state dependency evidence."""

    known = {
        str(item.get("requirement_id") or "").strip()
        for item in requirements
        if isinstance(item, Mapping) and str(item.get("requirement_id") or "").strip()
    }
    feature_dependencies, feature_provenance = _feature_model_dependencies(requirements)
    state_dependencies, state_provenance = _observable_state_dependencies(requirements)
    result: dict[str, list[str]] = {}
    for item in requirements:
        if not isinstance(item, Mapping):
            continue
        requirement_id = str(item.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        result[requirement_id] = list(
            dict.fromkeys(
                dependency
                for dependency in (
                    *_declared_dependencies(item, requirement_id),
                    *feature_dependencies.get(requirement_id, ()),
                    *state_dependencies.get(requirement_id, ()),
                )
                if dependency in known and dependency != requirement_id
            )
        )
    return result, [*feature_provenance, *state_provenance]


def _compile_semantic_catalog(prompt: str, router: Any | None) -> dict[str, Any]:
    """Compile authored clauses through the bounded canonical semantic helper."""

    from .semantic_batching_contract import build_bounded_requirement_catalog

    catalog = dict(build_bounded_requirement_catalog(prompt, router=router))
    if router is None:
        return catalog

    audit = dict(catalog.get("semantic_audit") or {})
    batch_size = audit.get("semantic_batch_size")
    batch_count = audit.get("semantic_batch_count")
    if type(batch_size) is not int or batch_size <= 0:
        raise _evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_AUDIT: semantic batch size was lost before planning finalization."
        )
    if type(batch_count) is not int or batch_count <= 0:
        raise _evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_AUDIT: semantic batch count was lost before planning finalization."
        )

    audit.update(
        {
            "normal_model_turns": batch_count,
            "semantic_model_turns": batch_count,
            "semantic_discovery_model_turns": batch_count,
            "semantic_detail_model_turns": 0,
            "max_repair_turns": 0,
            "generation_policy": "bounded_host_catalog_classification_batches",
            "semantic_generation_protocol": "bounded_host_catalog_classification_batches",
            "max_clauses_per_model_turn": batch_size,
            "max_semantic_leaves_per_detail_turn": 0,
            "model_generated_planning_json": False,
            "global_dependency_reconciliation": "host_feature_model_then_authored_state_dag",
            "source_clause_index_owner": "host",
            "source_anchor_owner": "host",
            "source_grounding_owner": "host",
            "capability_id_owner": "host_catalog",
            "dependency_owner": "host",
            "implementation_architecture_owner": "host",
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
    """Enrich frozen requirements with host-owned retrieval and dependency planning."""

    if router is None:
        return dict(catalog)

    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return dict(catalog)

    # The retrieval helper may use model embeddings/ranking internally, but requirement
    # IDs, dependency edges and query ownership remain host-controlled after validation.
    payload = _retrieval._call_retrieval_planner(router, prompt, requirements)
    emit_root_cause(
        "dependency_and_retrieval_plan_candidate",
        stage="planning",
        operation="enrich_retrieval_plan",
        gate="causal_dependency_graph",
        result="START",
        details={"requirements": requirements, "candidate": payload},
    )
    plan = _retrieval._normalize_retrieval_plan(prompt, requirements, payload)

    inferred, dependency_provenance = _host_causal_dependencies(requirements)
    known = set(plan)
    dependency_map: dict[str, tuple[str, ...]] = {}
    for requirement_id, planned in plan.items():
        # Dependency ownership is host-only.  Retrieval output is allowed to contribute
        # search queries but not causal edges.
        merged = list(
            dict.fromkeys(
                str(dependency)
                for dependency in inferred.get(requirement_id, ())
                if str(dependency) in known and str(dependency) != requirement_id
            )
        )
        planned["depends_on"] = merged
        dependency_map[requirement_id] = tuple(merged)

    _retrieval._validate_dependency_dag(dependency_map, known)

    enriched = deepcopy(dict(catalog))
    enriched_requirements: list[dict[str, Any]] = []
    edges: list[list[str]] = []
    for raw in enriched["requirements"]:
        item = dict(raw)
        requirement_id = str(item.get("requirement_id") or "")
        planned = plan[requirement_id]
        item["depends_on"] = list(planned["depends_on"])
        item["search_queries"] = list(planned["search_queries"])
        unlock = dict(item.get("unlock_policy") or {})
        unlock["required_requirement_refs"] = list(planned["depends_on"])
        unlock["required_capabilities"] = [
            str(next(
                requirement.get("capability")
                for requirement in enriched["requirements"]
                if isinstance(requirement, Mapping)
                and str(requirement.get("requirement_id") or "") == dependency
            ))
            for dependency in planned["depends_on"]
        ]
        item["unlock_policy"] = unlock
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
            "retrieval_query_planning": "host_validated_all_requirements",
            "max_requirements_per_query_turn": len(requirements),
            "model_owned_requirement_ids": False,
            "model_owned_dependency_edges": False,
            "model_generated_planning_json": False,
            "dependency_edge_count": len(edges),
            "dependency_derivation": "host_feature_model_plus_strict_authored_state_producers",
            "host_inferred_dependency_edge_count": len(dependency_provenance),
        }
    )
    enriched["semantic_audit"] = audit
    enriched["dependency_provenance"] = dependency_provenance
    enriched["catalog_sha256"] = ""
    enriched["catalog_sha256"] = _evidence._hash_without(enriched, "catalog_sha256")
    _semantic.validate_approved_requirement_catalog(enriched, prompt=prompt)
    emit_root_cause(
        "dependency_and_retrieval_plan_approved",
        stage="planning",
        operation="enrich_retrieval_plan",
        gate="causal_dependency_graph",
        result="PASS",
        details={
            "plan": plan,
            "edges": edges,
            "dependency_provenance": dependency_provenance,
            "catalog": enriched,
        },
    )
    return enriched


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None,
) -> dict[str, Any]:
    """Compile request meaning and retrieval intent before design/RAG execution."""

    with trace_scope("planner"):
        emit_root_cause(
            "pipeline_boundary_start",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="START",
            details={"prompt": prompt, "router": type(router).__name__ if router else None},
        )
        try:
            catalog = _enrich_retrieval_plan(
                prompt,
                _compile_semantic_catalog(prompt, router),
                router,
            )
        except BaseException as exc:
            emit_root_cause(
                "pipeline_boundary_failure",
                stage="planning",
                operation="build_authoritative_request_catalog",
                gate="planner",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                exc=exc,
            )
            raise
        emit_root_cause(
            "pipeline_boundary_result",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="PASS",
            details={"catalog": catalog},
        )
        return catalog


@contextmanager
def authoritative_request_scope(
    prompt: str,
    catalog: Mapping[str, Any],
) -> Iterator[None]:
    """Expose one immutable request catalog to downstream read-only helpers."""

    token = _guard._ACTIVE_REQUEST_CATALOG.set((prompt, deepcopy(dict(catalog))))
    try:
        yield
    finally:
        _guard._ACTIVE_REQUEST_CATALOG.reset(token)


__all__ = [
    "authoritative_request_scope",
    "build_authoritative_request_catalog",
]
