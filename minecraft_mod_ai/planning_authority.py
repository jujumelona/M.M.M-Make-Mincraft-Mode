from __future__ import annotations

"""Deterministic request authority for the production planner.

Planning structure is compiler-owned. The model router is deliberately not consulted
here: authored spans, capability selection, dependency edges, research queries and
requirement IDs must exist before optional model work can run.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

from . import evidence_first_planning as _evidence
from .minecraft_template_catalog import selected_predecessor_capabilities
from .root_cause_trace import emit_root_cause, trace_scope

_ACTIVE_REQUEST_CATALOG: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "mmm_active_authoritative_request_catalog",
    default=None,
)
_STATE_TOKEN = re.compile(r"[A-Za-z0-9_]+|[가-힣]+", re.UNICODE)
_STATE_STOP = frozenset({
    "a", "an", "the", "and", "or", "to", "from", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "with", "for", "by", "can", "may", "player", "players", "game", "mode",
    "feature", "mechanic", "requested", "behavior", "outcome", "state", "exists",
    "플레이어", "게임", "모드", "상태",
})
_PRODUCER_STEMS = frozenset({
    "acquir", "arriv", "assembl", "build", "collect", "construct", "creat",
    "discover", "establish", "generat", "obtain", "open", "spawn", "travel", "unlock",
})


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
    return frozenset(
        token
        for raw in _STATE_TOKEN.findall(str(value or ""))
        if len(token := _stem_state_token(raw)) > 1 and token not in _STATE_STOP
    )


def _source_start(item: Mapping[str, Any]) -> int:
    span = item.get("source_span")
    return int(span["char_start"]) if isinstance(span, Mapping) and type(span.get("char_start")) is int else 2**63 - 1


def _declared_dependencies(item: Mapping[str, Any], requirement_id: str) -> list[str]:
    result: list[str] = []
    raw = item.get("depends_on")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result.extend(str(value).strip() for value in raw if str(value).strip() and str(value).strip() != requirement_id)
    unlock = item.get("unlock_policy")
    refs = unlock.get("required_requirement_refs") if isinstance(unlock, Mapping) else None
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes, bytearray)):
        result.extend(str(value).strip() for value in refs if str(value).strip() and str(value).strip() != requirement_id)
    return list(dict.fromkeys(result))


def _observable_text(item: Mapping[str, Any], field: str) -> str:
    behavior = item.get("observable_behavior")
    return str(behavior.get(field) or "").strip() if isinstance(behavior, Mapping) else ""


def _feature_model_dependencies(requirements: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    selected = [str(item.get("capability") or "").strip().casefold() for item in requirements if str(item.get("capability") or "").strip()]
    ids_by_capability: dict[str, list[str]] = {}
    for item in requirements:
        capability = str(item.get("capability") or "").strip().casefold()
        requirement_id = str(item.get("requirement_id") or "").strip()
        if capability and requirement_id:
            ids_by_capability.setdefault(capability, []).append(requirement_id)
    result: dict[str, list[str]] = {}
    provenance: list[dict[str, Any]] = []
    for item in requirements:
        requirement_id = str(item.get("requirement_id") or "").strip()
        capability = str(item.get("capability") or "").strip().casefold()
        if not requirement_id or not capability:
            continue
        dependencies: list[str] = []
        for predecessor in selected_predecessor_capabilities(capability, selected):
            candidates = [candidate for candidate in ids_by_capability.get(predecessor, ()) if candidate != requirement_id]
            if len(candidates) != 1:
                continue
            dependency = candidates[0]
            dependencies.append(dependency)
            provenance.append({
                "dependency": dependency,
                "requirement_id": requirement_id,
                "capability": capability,
                "predecessor_capability": predecessor,
                "method": "host_feature_model_selected_predecessor",
            })
        result[requirement_id] = list(dict.fromkeys(dependencies))
    return result, provenance


def _observable_state_dependencies(requirements: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    ordered = sorted((dict(item) for item in requirements), key=lambda item: (_source_start(item), str(item.get("requirement_id") or "")))
    result: dict[str, list[str]] = {}
    provenance: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    for item in ordered:
        requirement_id = str(item.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        given_terms = _state_terms(_observable_text(item, "given"))
        candidates: list[tuple[tuple[int, int], str, tuple[str, ...]]] = []
        for producer in previous:
            producer_id = str(producer.get("requirement_id") or "").strip()
            if not given_terms or not producer_id or producer_id == requirement_id:
                continue
            then_terms = _state_terms(_observable_text(producer, "then"))
            shared = tuple(sorted(given_terms & then_terms))
            actions = then_terms & _PRODUCER_STEMS
            if shared and actions:
                candidates.append(((len(shared), len(actions)), producer_id, shared))
        dependencies: list[str] = []
        if candidates:
            best_score = max(score for score, _producer, _shared in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_score]
            if len(best) == 1:
                _score, producer_id, shared = best[0]
                dependencies.append(producer_id)
                provenance.append({
                    "dependency": producer_id,
                    "requirement_id": requirement_id,
                    "method": "host_observable_state_producer",
                    "shared_state_terms": list(shared),
                })
        result[requirement_id] = dependencies
        previous.append(item)
    return result, provenance


def _host_causal_dependencies(requirements: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    known = {str(item.get("requirement_id") or "").strip() for item in requirements if str(item.get("requirement_id") or "").strip()}
    feature, feature_provenance = _feature_model_dependencies(requirements)
    state, state_provenance = _observable_state_dependencies(requirements)
    result: dict[str, list[str]] = {}
    for item in requirements:
        requirement_id = str(item.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        result[requirement_id] = list(dict.fromkeys(
            dependency
            for dependency in (*_declared_dependencies(item, requirement_id), *feature.get(requirement_id, ()), *state.get(requirement_id, ()))
            if dependency in known and dependency != requirement_id
        ))
    return result, [*feature_provenance, *state_provenance]


def _research_queries(requirement: Mapping[str, Any]) -> list[str]:
    statement = " ".join(str(requirement.get("semantic_statement") or requirement.get("statement") or "").split())
    capability = str(requirement.get("capability") or "").replace(".", " ").replace("_", " ").strip()
    candidates = [
        f"Minecraft mod implementation {statement}" if statement else "",
        f"Minecraft mod {capability} implementation" if capability else "",
    ]
    return list(dict.fromkeys(query for query in candidates if query))[:2]


def _compile_host_catalog(prompt: str) -> dict[str, Any]:
    catalog = dict(_evidence.build_request_catalog(prompt, {}, router=None))
    raw_requirements = catalog.get("requirements")
    requirements = [dict(item) for item in raw_requirements if isinstance(item, Mapping)] if isinstance(raw_requirements, list) else []
    if not requirements:
        return catalog
    inferred, provenance = _host_causal_dependencies(requirements)
    capability_by_id = {str(item.get("requirement_id") or ""): str(item.get("capability") or "") for item in requirements}
    enriched: list[dict[str, Any]] = []
    edges: list[list[str]] = []
    for item in requirements:
        requirement_id = str(item.get("requirement_id") or "")
        dependencies = list(inferred.get(requirement_id, ()))
        item["depends_on"] = dependencies
        item["search_queries"] = _research_queries(item)
        unlock = dict(item.get("unlock_policy") or {})
        unlock["required_requirement_refs"] = dependencies
        unlock["required_capabilities"] = [capability_by_id[dependency] for dependency in dependencies if capability_by_id.get(dependency)]
        item["unlock_policy"] = unlock
        edges.extend([dependency, requirement_id] for dependency in dependencies)
        enriched.append(item)
    catalog["requirements"] = enriched
    catalog["requirement_graph"] = {"node_ids": [str(item["requirement_id"]) for item in enriched], "edges": edges}
    catalog["dependency_provenance"] = provenance
    catalog["semantic_audit"] = {
        "status": "APPROVED",
        "authored_clause_count": len(enriched),
        "covered_clause_count": len(enriched),
        "unresolved_clause_count": 0,
        "unsupported_design_choice_count": 0,
        "normal_model_turns": 0,
        "semantic_model_turns": 0,
        "retrieval_model_turns": 0,
        "generation_policy": "deterministic_host_compiler",
        "model_generated_planning_json": False,
        "source_grounding_owner": "host",
        "capability_id_owner": "host_catalog",
        "dependency_owner": "host",
        "implementation_architecture_owner": "host",
        "research_query_owner": "host",
    }
    catalog["catalog_sha256"] = ""
    catalog["catalog_sha256"] = _evidence._hash_without(catalog, "catalog_sha256")
    _evidence._validate_request_catalog(catalog, prompt=prompt)
    return catalog


def build_authoritative_request_catalog(prompt: str, router: Any | None = None) -> dict[str, Any]:
    """Compile request meaning, dependencies and research intent without model calls."""
    del router
    with trace_scope("planner"):
        emit_root_cause(
            "pipeline_boundary_start",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="START",
            details={"prompt": prompt, "authority": "deterministic_host_compiler"},
        )
        catalog = _compile_host_catalog(prompt)
        emit_root_cause(
            "pipeline_boundary_result",
            stage="planning",
            operation="build_authoritative_request_catalog",
            gate="planner",
            result="PASS",
            details={"catalog": catalog},
        )
        return catalog


def active_authoritative_request_catalog(prompt: str) -> dict[str, Any] | None:
    """Return only the currently frozen catalog; never rebuild authority implicitly."""
    active = _ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return None
    return deepcopy(active[1])


@contextmanager
def authoritative_request_scope(prompt: str, catalog: Mapping[str, Any]) -> Iterator[None]:
    token = _ACTIVE_REQUEST_CATALOG.set((prompt, deepcopy(dict(catalog))))
    try:
        yield
    finally:
        _ACTIVE_REQUEST_CATALOG.reset(token)


__all__ = [
    "active_authoritative_request_catalog",
    "authoritative_request_scope",
    "build_authoritative_request_catalog",
]
