from __future__ import annotations

"""Plan*RAG-style evidence-obligation DAG bound to approved requirements.

This module removes the legacy "whole request + implementation/dependencies/assets/license/tests"
retrieval authority. Every approved requirement is expanded into independent evidence obligations;
retrieval, correction and coverage are evaluated per obligation rather than per document count.
"""

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

from . import agentic_research_game_design as _agentic
from . import central_research as _central
from . import evidence_request_guard as _guard
from . import parallel_runtime_contract as _parallel
from . import retrieval as _retrieval

_INSTALLED = False
_LOCK = threading.RLock()
_ACTIVE_BY_PROMPT: dict[str, dict[str, Any]] = {}
_QUERY_META: dict[str, dict[str, Any]] = {}
_MIXED_SUFFIX = "minecraft java mod implementation dependencies assets license tests"
_TOKEN = re.compile(r"[a-z0-9_]+|[가-힣]{2,}", re.IGNORECASE)
_STOP = frozenset(
    {
        "minecraft", "java", "mod", "fabric", "api", "official", "documentation",
        "implementation", "implement", "target", "compatibility", "dependency",
        "dependencies", "closure", "license", "provenance", "validation", "runtime",
        "behavior", "source", "code", "reusable", "testing", "gametest", "loader",
        "mapping", "mappings", "project", "evidence", "requirement", "player", "players",
        "can", "the", "a", "an", "to", "of", "for", "with", "and", "or", "in", "on",
    }
)

_OBLIGATIONS: tuple[dict[str, Any], ...] = (
    {
        "kind": "reusable_implementation",
        "evidence_kind": "local_project",
        "providers": ("project_rag", "github"),
        "query": "{capability} reusable implementation source code {statement}",
        "depends_on": (),
        "retrieval_required": True,
    },
    {
        "kind": "target_compatibility",
        "evidence_kind": "compatibility",
        "providers": ("official_docs",),
        "query": "Minecraft Fabric target compatibility loader mappings {capability}",
        "depends_on": (),
        "retrieval_required": True,
    },
    {
        "kind": "implementation_api",
        "evidence_kind": "minecraft_api",
        "providers": ("official_docs", "project_rag", "github"),
        "query": "{capability} Minecraft Fabric API {statement}",
        "depends_on": ("target_compatibility",),
        "retrieval_required": True,
    },
    {
        "kind": "dependency_closure",
        "evidence_kind": "dependency",
        "providers": ("official_docs", "modrinth", "github"),
        "query": "Minecraft Fabric dependency closure {capability} {statement}",
        "depends_on": ("target_compatibility",),
        "retrieval_required": True,
    },
    {
        "kind": "license_provenance",
        "evidence_kind": "license",
        "providers": ("project_rag", "github", "modrinth"),
        "query": "{capability} reusable implementation license provenance {statement}",
        "depends_on": ("reusable_implementation",),
        "retrieval_required": True,
    },
    {
        "kind": "validation_mechanism",
        "evidence_kind": "testing",
        "providers": ("official_docs", "runtime"),
        "query": "Minecraft Fabric GameTest validation {capability} {statement}",
        "depends_on": ("implementation_api",),
        "retrieval_required": True,
    },
    {
        "kind": "asset_requirement",
        "evidence_kind": "visual_reference",
        "providers": (),
        "query": "",
        "depends_on": (),
        "retrieval_required": False,
    },
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_id(value: str, prefix: str = "obl") -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{(slug[:42] or prefix)}_{digest}"[:63]


def _anchors(value: str) -> tuple[str, ...]:
    result: list[str] = []
    for match in _TOKEN.finditer(value.casefold()):
        token = match.group(0)
        if token in _STOP or len(token) < 3:
            continue
        if token not in result:
            result.append(token)
    return tuple(result[:16])


def _remember(prompt: str, catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != "mmm/approved-requirement-graph-v1":
        return
    with _LOCK:
        _ACTIVE_BY_PROMPT[_sha(prompt)] = dict(catalog)
        while len(_ACTIVE_BY_PROMPT) > 128:
            _ACTIVE_BY_PROMPT.pop(next(iter(_ACTIVE_BY_PROMPT)))


def _catalog_for(prompt: str) -> dict[str, Any] | None:
    with _LOCK:
        value = _ACTIVE_BY_PROMPT.get(_sha(prompt))
        return dict(value) if isinstance(value, Mapping) else None


def build_evidence_obligation_brief(
    prompt: str,
    catalog: Mapping[str, Any],
    game_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise _central.SpecValidationError(
            "Approved requirement graph has no requirements for evidence routing."
        )

    nodes: list[dict[str, Any]] = []
    node_id: dict[tuple[str, str], str] = {}
    normalized_requirements: list[tuple[str, str, str]] = []
    for raw in requirements:
        if not isinstance(raw, Mapping):
            continue
        req_id = str(raw.get("requirement_id") or "").strip()
        capability = str(raw.get("capability") or "").strip()
        statement = str(
            raw.get("semantic_statement") or raw.get("statement") or capability
        ).strip()
        if not req_id or not capability or not statement:
            raise _central.SpecValidationError(
                "Approved requirement is missing stable identity/capability/statement."
            )
        normalized_requirements.append((req_id, capability, statement))
        for spec in _OBLIGATIONS:
            kind = str(spec["kind"])
            oid = _safe_id(f"{req_id}:{kind}")
            node_id[(req_id, kind)] = oid
            nodes.append(
                {
                    "obligation_id": oid,
                    "requirement_id": req_id,
                    "capability": capability,
                    "kind": kind,
                    "evidence_kind": spec["evidence_kind"],
                    "semantic_statement": statement,
                    "retrieval_required": bool(spec["retrieval_required"]),
                    "status": (
                        "PENDING_RETRIEVAL"
                        if spec["retrieval_required"]
                        else "PENDING_DESIGN_RESOLUTION"
                    ),
                    "depends_on": [],
                }
            )

    for node in nodes:
        spec = next(item for item in _OBLIGATIONS if item["kind"] == node["kind"])
        node["depends_on"] = [
            node_id[(node["requirement_id"], parent)] for parent in spec["depends_on"]
        ]

    by_oid = {node["obligation_id"]: node for node in nodes}
    domains: list[Any] = []
    bindings: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not node["retrieval_required"]:
            continue
        spec = next(item for item in _OBLIGATIONS if item["kind"] == node["kind"])
        query = " ".join(
            str(spec["query"]).format(
                capability=node["capability"],
                statement=node["semantic_statement"],
            ).split()
        )
        if _MIXED_SUFFIX in query.casefold():
            raise _central.SpecValidationError(
                "Atomic query reintroduced the mixed legacy retrieval suffix."
            )
        domain_id = _safe_id(node["obligation_id"], prefix="obl")
        parents = []
        for parent_oid in node["depends_on"]:
            parent = by_oid[parent_oid]
            if parent["retrieval_required"]:
                parents.append(_safe_id(parent_oid, prefix="obl"))
        domain = _central.ResearchDomain(
            domain_id=domain_id,
            objective=(
                f"Fulfill {node['kind']} evidence for capability {node['capability']}."
            ),
            requirements=(node["semantic_statement"],),
            evidence_kinds=(node["evidence_kind"],),
            queries=(query,),
            providers=tuple(spec["providers"]),
            depends_on=tuple(parents),
        )
        domain = _central._augment_domain_routes(domain)
        domains.append(domain)
        meta = {
            "obligation_id": node["obligation_id"],
            "requirement_id": node["requirement_id"],
            "capability": node["capability"],
            "kind": node["kind"],
            "evidence_kind": node["evidence_kind"],
            "query": query,
            "anchors": list(_anchors(f"{node['capability']} {node['semantic_statement']}")),
        }
        bindings[domain_id] = meta
        with _LOCK:
            _QUERY_META[query] = dict(meta)

    _central._validate_domain_graph(tuple(domains))
    payload: dict[str, Any] = {
        "schema_version": "mmm/central-research-brief-v2",
        "summary": "Atomic evidence obligations for the approved requirement graph.",
        "origin": "approved_requirement_graph",
        "approved_requirement_catalog_sha256": catalog.get("catalog_sha256", ""),
        "domains": [domain.to_dict() for domain in domains],
        "unresolved_questions": [],
        "evidence_obligation_dag": {
            "schema_version": "mmm/evidence-obligation-dag-v1",
            "nodes": nodes,
            "edges": [
                {"from": parent, "to": node["obligation_id"]}
                for node in nodes
                for parent in node["depends_on"]
            ],
        },
        "obligation_bindings": bindings,
        "routing_policy": (
            "One capability × one evidence obligation × one atomic query. "
            "The authored prompt is provenance, never a catch-all retrieval query."
        ),
        "scale_policy": (
            "No project-wide obligation cap; each obligation remains independently unresolved "
            "until its own evidence contract is fulfilled."
        ),
    }
    selection = dict(game_design or {}).get("_platform_selection")
    if isinstance(selection, Mapping):
        target = selection.get("target")
        if isinstance(target, Mapping):
            payload["_mmm_platform_target"] = dict(target)
    payload["brief_sha256"] = _central._sha256(_central.canonical_json(payload))
    return payload


def _normalize(
    original: Any,
    prompt: str,
    game_design: dict[str, Any],
    candidate: Any | None,
) -> dict[str, Any]:
    if candidate is not None:
        return original(prompt, game_design, candidate)
    catalog = _catalog_for(prompt)
    if catalog is None:
        return original(prompt, game_design, candidate)
    return build_evidence_obligation_brief(prompt, catalog, game_design)


def _coverage_query_plan(original: Any, central_module: Any, domains: list[Any]):
    if not domains or not all(str(domain.domain_id).startswith("obl_") for domain in domains):
        return original(central_module, domains)
    query_criteria: dict[str, tuple[str, ...]] = {}
    domain_queries: dict[str, list[str]] = {}
    domain_criteria: dict[str, list[str]] = {}
    for domain in domains:
        queries = list(dict.fromkeys(str(q).strip() for q in domain.queries if str(q).strip()))
        if len(queries) != 1:
            raise central_module.SpecValidationError(
                f"Atomic obligation {domain.domain_id} must have exactly one query."
            )
        query = queries[0]
        criterion = f"obligation:{domain.domain_id}"
        query_criteria[query] = (criterion,)
        domain_queries[domain.domain_id] = [query]
        domain_criteria[domain.domain_id] = [criterion]
    return query_criteria, domain_queries, domain_criteria


def _hit_text(receipt: Any) -> str:
    return " ".join(
        f"{getattr(hit, 'title', '')} {getattr(hit, 'excerpt', '')}"
        for hit in getattr(receipt, "hits", ())
    ).casefold()


def _obligation_satisfied(meta: Mapping[str, Any], receipt: Any) -> bool:
    hits = tuple(getattr(receipt, "hits", ()))
    if not hits:
        return False
    kind = str(meta.get("kind", ""))
    ids = {str(getattr(hit, "document_id", "")) for hit in hits}
    expected = {
        "target_compatibility": {"fabric-project-creation", "fabric-mod-json"},
        "dependency_closure": {
            "fabric-project-creation", "fabric-mod-json", "fabric-building"
        },
        "validation_mechanism": {"fabric-automatic-testing"},
    }
    if kind in expected:
        return bool(ids & expected[kind])
    if kind == "implementation_api":
        text = _hit_text(receipt)
        anchors = [str(value).casefold() for value in meta.get("anchors", [])]
        return bool(anchors) and any(anchor in text for anchor in anchors)
    return str(getattr(receipt, "quality", "")).casefold() == "strong"


def _correction_queries(meta: Mapping[str, Any], receipt: Any) -> tuple[str, ...]:
    anchors = " ".join(str(value) for value in meta.get("anchors", [])[:8]).strip()
    prefix = f"{anchors} " if anchors else ""
    capability = str(meta.get("capability", ""))
    kind = str(meta.get("kind", ""))
    templates = {
        "implementation_api": f"{prefix}{capability} Fabric API official symbols",
        "target_compatibility": (
            f"Minecraft {getattr(receipt, 'minecraft_version', '')} "
            f"{getattr(receipt, 'loader', '')} loader mappings compatibility"
        ),
        "dependency_closure": f"{prefix}{capability} Fabric dependency Gradle metadata",
        "validation_mechanism": f"{prefix}{capability} Fabric GameTest runtime validation",
    }
    query = " ".join(templates.get(kind, f"{prefix}{capability} official evidence").split())
    return (query,) if query else ()


def _strict_retrieve(original: Any, self: Any, query: str, **kwargs: Any):
    receipt = original(self, query, **kwargs)
    with _LOCK:
        meta = dict(_QUERY_META.get(query, {}))
    if not meta:
        return receipt
    if _obligation_satisfied(meta, receipt):
        return replace(
            receipt,
            quality="strong",
            coverage=1.0,
            correction_required=False,
            correction_queries=(),
        )
    corrections = _correction_queries(meta, receipt)
    with _LOCK:
        for correction in corrections:
            _QUERY_META[correction] = dict(meta)
    return replace(
        receipt,
        quality="weak",
        coverage=0.0,
        correction_required=True,
        correction_queries=corrections,
    )


def _attach_coverage(
    original: Any,
    graph: dict[str, Any],
    *,
    query_criteria: Mapping[str, tuple[str, ...]],
    domain_criteria: Mapping[str, list[str]],
) -> dict[str, Any]:
    atomic = bool(domain_criteria) and all(
        all(str(item).startswith("obligation:") for item in values)
        for values in domain_criteria.values()
    )
    if not atomic:
        return original(
            graph,
            query_criteria=query_criteria,
            domain_criteria=domain_criteria,
        )

    unresolved: list[str] = []
    required_total = 0
    covered_total = 0
    for raw_domain in graph.get("domains", []):
        if not isinstance(raw_domain, dict):
            continue
        domain_id = str(raw_domain.get("domain_id", ""))
        required = list(domain_criteria.get(domain_id, ()))
        if not required:
            continue
        required_total += len(required)
        covered: set[str] = set()
        for result in raw_domain.get("queries", []):
            if not isinstance(result, Mapping):
                continue
            primary = result.get("primary")
            if not isinstance(primary, Mapping):
                continue
            criteria = query_criteria.get(str(primary.get("query", "")), ())
            receipts = [primary] + [
                value
                for value in result.get("corrections", [])
                if isinstance(value, Mapping)
            ]
            if any(
                str(value.get("quality", "")).casefold() == "strong"
                and bool(value.get("hits"))
                and float(value.get("coverage", 0.0) or 0.0) >= 1.0
                for value in receipts
            ):
                covered.update(criteria)
        covered_total += len(covered)
        uncovered = [value for value in required if value not in covered]
        raw_domain["coverage"] = {
            "required": required,
            "covered": [value for value in required if value in covered],
            "uncovered": uncovered,
            "fulfilled_obligations": len(covered),
            "required_obligations": len(required),
            "ratio": len(covered) / len(required) if required else 1.0,
            "complete": not uncovered,
        }
        raw_domain["strategy"] = "atomic_obligation_corrective"
        if uncovered:
            unresolved.append(domain_id)

    graph["unresolved_official_domains"] = unresolved
    graph["coverage"] = {
        "fulfilled_obligations": covered_total,
        "required_obligations": required_total,
        "ratio": covered_total / required_total if required_total else 1.0,
        "complete": covered_total == required_total,
    }
    graph["coverage_policy"] = (
        "coverage = fulfilled evidence obligations / required evidence obligations; "
        "generic document count and lexical/family hit signals are not coverage."
    )
    return graph


def install_evidence_obligation_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    current_builder = _guard.build_authoritative_request_catalog
    if not getattr(current_builder, "_mmm_evidence_obligation_context", False):
        @wraps(current_builder)
        def build_with_context(prompt: str, router: Any | None = None):
            catalog = current_builder(prompt, router=router)
            if isinstance(catalog, Mapping):
                _remember(prompt, catalog)
            return catalog

        build_with_context._mmm_evidence_obligation_context = True
        _guard.build_authoritative_request_catalog = build_with_context

    original_normalize = _central.normalize_research_brief
    if not getattr(original_normalize, "_mmm_approved_obligation_dag", False):
        @wraps(original_normalize)
        def normalize(prompt: str, game_design: dict[str, Any], candidate: Any | None = None):
            return _normalize(original_normalize, prompt, game_design, candidate)

        normalize._mmm_approved_obligation_dag = True
        _central.normalize_research_brief = normalize
        _agentic.normalize_research_brief = normalize

    original_plan = _parallel._coverage_query_plan
    if not getattr(original_plan, "_mmm_atomic_obligation_queries", False):
        @wraps(original_plan)
        def coverage_query_plan(central_module: Any, domains: list[Any]):
            return _coverage_query_plan(original_plan, central_module, domains)

        coverage_query_plan._mmm_atomic_obligation_queries = True
        _parallel._coverage_query_plan = coverage_query_plan

    original_attach = _parallel._attach_coverage_status
    if not getattr(original_attach, "_mmm_atomic_obligation_coverage", False):
        @wraps(original_attach)
        def attach(graph: dict[str, Any], *, query_criteria, domain_criteria):
            return _attach_coverage(
                original_attach,
                graph,
                query_criteria=query_criteria,
                domain_criteria=domain_criteria,
            )

        attach._mmm_atomic_obligation_coverage = True
        _parallel._attach_coverage_status = attach

    original_retrieve = _retrieval.OfficialCorpusIndex.retrieve
    if not getattr(original_retrieve, "_mmm_obligation_evaluator", False):
        @wraps(original_retrieve)
        def retrieve(self: Any, query: str, **kwargs: Any):
            return _strict_retrieve(original_retrieve, self, query, **kwargs)

        retrieve._mmm_obligation_evaluator = True
        _retrieval.OfficialCorpusIndex.retrieve = retrieve

    _INSTALLED = True


__all__ = ["build_evidence_obligation_brief", "install_evidence_obligation_contract"]
