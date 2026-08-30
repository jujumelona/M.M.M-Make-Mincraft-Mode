from __future__ import annotations

"""Capability-level joint platform/reuse optimisation.

This module upgrades platform selection from "a compatible project exists" to an
implementation plan.  Every executable target is scored from the work that remains
after verified same-project, MMM-registry, host/API, pinned source-transplant, adapt,
and fresh options are considered.  Counts are diagnostic only; selection minimises
expected implementation + verification work and preserves donor provenance.
"""

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any

from . import platform_optimizer as _platform
from .component_registry import (
    VerifiedComponent,
    find_verified_component,
    load_verified_components,
)
from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import (
    PlatformAdapter,
    _emit_discovery_log,
    adapter_for_target,
    discover_target_keys,
)
from .proof_level import ProofLevel
from .residual_generation_contract import ResidualGenerationContract
from .reuse_artifacts import ReusableArtifactBundle, bundle_proof_allows_reuse
from .reuse_discovery import discover_repositories_for_graph
from .source_transplant import (
    DonorSlice,
    inspect_repository_slice,
)

REUSE_MODES = (
    "same_project",
    "mmm_verified",
    "library",
    "source_transplant",
    "adapt",
    "fresh",
)
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_CAPABILITY_KEYS = frozenset({
    "capabilities", "systems", "features", "requirements", "behaviors", "services",
    "subsystems", "modules", "actions", "operations",
})
from .canonical_capability_ontology import (
    canonical_domain_map as _canonical_domain_map,
)
from .canonical_capability_ontology import (
    resolve_capabilities_from_phrase,
    search_queries_for_capability,
)

_CAPABILITY_HINTS = _canonical_domain_map()
_PROMPT_CAPABILITY_WORDS = frozenset(_CAPABILITY_HINTS)
_PRE_RETRIEVAL_PLAN_SCHEMA = "mmm/pre-retrieval-semantic-plan-v1"
_DEFAULT_TARGET_CANDIDATE_LIMIT = 8
_MAX_TARGET_CANDIDATE_LIMIT = 32


def _capability_graph_limit() -> int:
    """Return an optional operator quota; zero keeps logical project scale unbounded."""

    raw = os.environ.get("MMM_REUSE_CAPABILITY_LIMIT", "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_REUSE_CAPABILITY_LIMIT must be a non-negative integer.") from exc
    if value < 0:
        raise ValueError("MMM_REUSE_CAPABILITY_LIMIT must be a non-negative integer.")
    return value


def _target_candidate_limit() -> int:
    """Bound live provider work without changing the semantic project scope."""

    raw = os.environ.get(
        "MMM_PLATFORM_CANDIDATE_LIMIT",
        str(_DEFAULT_TARGET_CANDIDATE_LIMIT),
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_TARGET_CANDIDATE_LIMIT
    return max(1, min(value, _MAX_TARGET_CANDIDATE_LIMIT))


@dataclass(frozen=True)
class CapabilityGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]
    search_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    source_plan_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "mmm/capability-graph-v1",
            "nodes": list(self.nodes),
            "edges": [{"from": left, "to": right} for left, right in self.edges],
            "sources": [{"capability": cap, "source": source} for cap, source in self.sources],
            "search_terms": [
                {"capability": cap, "terms": list(terms)}
                for cap, terms in self.search_terms
            ],
        }
        if self.source_plan_sha256:
            payload["source_plan_sha256"] = self.source_plan_sha256
        return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical_json(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _plan_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["plan_sha256"] = ""
    return _sha256(payload)


def _requirement_capabilities(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    raw = requirement.get("provides")
    values: Iterable[Any]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = raw
    else:
        values = (requirement.get("capability"),)
    return tuple(
        dict.fromkeys(
            capability
            for item in values
            if (capability := _capability_id(item))
        )
    )


def _planned_work_id(requirement_id: str) -> str:
    digest = hashlib.sha256(requirement_id.encode("utf-8")).hexdigest()[:12]
    return f"intent_{digest}"


def compile_pre_retrieval_plan(
    prompt: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze semantic work before any platform or donor lookup can run.

    This is deliberately target- and reuse-neutral.  It records what must be built,
    why it is required, its authored dependencies, and how it will be accepted.  A
    later discovery pass may choose retain/adapt/fresh for a work item, but it cannot
    add, rename, merge, or remove semantic work.
    """

    catalog = design.get("_evidence_request_catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError(
            "Authoritative request catalog must be frozen before retrieval planning."
        )
    from .evidence_first_planning import _validate_request_catalog

    _validate_request_catalog(catalog, prompt=prompt)
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Pre-retrieval semantic planning requires request requirements.")

    # This call is still planning: it turns the already-approved requirement graph
    # into immutable search boundaries.  No provider/client is constructed here.
    graph = decompose_capability_graph(prompt, design=design)
    requirement_ids = [
        str(item.get("requirement_id") or item.get("id") or "").strip()
        for item in requirements
        if isinstance(item, Mapping)
    ]
    if len(requirement_ids) != len(requirements) or any(not item for item in requirement_ids):
        raise ValueError("Pre-retrieval requirements must have stable identifiers.")
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("Pre-retrieval requirement identifiers must be unique.")
    work_by_requirement = {
        requirement_id: _planned_work_id(requirement_id)
        for requirement_id in requirement_ids
    }

    planned_work: list[dict[str, Any]] = []
    for raw in requirements:
        assert isinstance(raw, Mapping)
        requirement_id = str(raw.get("requirement_id") or raw.get("id") or "").strip()
        dependencies_raw = raw.get("depends_on", ())
        dependencies = (
            tuple(str(item).strip() for item in dependencies_raw if str(item).strip())
            if isinstance(dependencies_raw, Sequence)
            and not isinstance(dependencies_raw, (str, bytes, bytearray))
            else ()
        )
        unknown = sorted(set(dependencies) - set(work_by_requirement))
        if unknown:
            raise ValueError(
                f"Pre-retrieval requirement {requirement_id} has unknown dependencies: {unknown}."
            )
        capabilities = _requirement_capabilities(raw)
        if not capabilities:
            raise ValueError(
                f"Pre-retrieval requirement {requirement_id} has no semantic capability."
            )
        acceptance_raw = raw.get("acceptance", ())
        acceptance = (
            [str(item).strip() for item in acceptance_raw if str(item).strip()]
            if isinstance(acceptance_raw, Sequence)
            and not isinstance(acceptance_raw, (str, bytes, bytearray))
            else []
        )
        planned_work.append(
            {
                "work_id": work_by_requirement[requirement_id],
                "requirement_ref": requirement_id,
                "objective": str(
                    raw.get("semantic_statement")
                    or raw.get("statement")
                    or raw.get("capability")
                    or ""
                ).strip(),
                "capabilities": list(capabilities),
                "depends_on": [work_by_requirement[item] for item in dependencies],
                "acceptance": acceptance,
                "discovery_disposition": "unresolved_until_verified_reuse_analysis",
            }
        )

    plan: dict[str, Any] = {
        "schema_version": _PRE_RETRIEVAL_PLAN_SCHEMA,
        "prompt_sha256": _sha256(prompt),
        "prompt_char_length": len(prompt),
        "request_catalog_sha256": str(catalog.get("catalog_sha256") or ""),
        "purpose": str(catalog.get("purpose") or design.get("pitch") or prompt).strip(),
        "planned_work": planned_work,
        "capability_graph": graph.to_dict(),
        "authority": {
            "semantic_scope": "approved_request_catalog",
            "work_identity": "host_frozen_before_retrieval",
            "retrieval_authority": False,
            "allowed_post_retrieval_changes": [
                "retain_verified",
                "adapt_verified",
                "fresh_required",
            ],
        },
        "plan_sha256": "",
    }
    plan["plan_sha256"] = _plan_hash(plan)
    validate_pre_retrieval_plan(plan, prompt=prompt, design=design)
    return plan


def validate_pre_retrieval_plan(
    plan: Mapping[str, Any],
    *,
    prompt: str,
    design: Mapping[str, Any],
) -> None:
    if plan.get("schema_version") != _PRE_RETRIEVAL_PLAN_SCHEMA:
        raise ValueError("Unsupported pre-retrieval semantic plan schema.")
    if plan.get("plan_sha256") != _plan_hash(plan):
        raise ValueError("Pre-retrieval semantic plan hash mismatch.")
    if plan.get("prompt_sha256") != _sha256(prompt) or plan.get(
        "prompt_char_length"
    ) != len(prompt):
        raise ValueError("Pre-retrieval semantic plan is stale for the request.")
    catalog = design.get("_evidence_request_catalog")
    if not isinstance(catalog, Mapping) or plan.get("request_catalog_sha256") != catalog.get(
        "catalog_sha256"
    ):
        raise ValueError("Pre-retrieval semantic plan is not bound to the request catalog.")

    requirements = catalog.get("requirements")
    work = plan.get("planned_work")
    if not isinstance(requirements, list) or not isinstance(work, list):
        raise TypeError("Pre-retrieval semantic plan has no planned work catalog.")
    requirement_ids = {
        str(item.get("requirement_id") or item.get("id") or "")
        for item in requirements
        if isinstance(item, Mapping)
    }
    work_ids: set[str] = set()
    work_requirements: set[str] = set()
    planned_capabilities: list[str] = []
    for item in work:
        if not isinstance(item, Mapping):
            raise TypeError("Pre-retrieval planned work must be an object.")
        work_id = str(item.get("work_id") or "")
        requirement_ref = str(item.get("requirement_ref") or "")
        if not work_id or work_id in work_ids:
            raise ValueError("Pre-retrieval planned work identifiers are invalid or duplicated.")
        work_ids.add(work_id)
        work_requirements.add(requirement_ref)
        planned_capabilities.extend(
            capability
            for raw in item.get("capabilities", ())
            if (capability := _capability_id(raw))
        )
    if work_requirements != requirement_ids or len(work) != len(requirement_ids):
        raise ValueError("Pre-retrieval plan does not cover every request requirement exactly once.")
    for item in work:
        dependencies = tuple(str(value) for value in item.get("depends_on", ()))
        if str(item.get("work_id")) in dependencies or any(
            dependency not in work_ids for dependency in dependencies
        ):
            raise ValueError("Pre-retrieval planned work dependency graph is invalid.")

    graph = plan.get("capability_graph")
    if not isinstance(graph, Mapping):
        raise TypeError("Pre-retrieval semantic plan has no capability graph.")
    nodes = tuple(str(item) for item in graph.get("nodes", ()))
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("Pre-retrieval capability nodes are empty or duplicated.")
    if set(nodes) != set(planned_capabilities):
        raise ValueError("Pre-retrieval capability graph drifted from planned work.")
    for edge in graph.get("edges", ()):
        if not isinstance(edge, Mapping) or edge.get("from") not in nodes or edge.get("to") not in nodes:
            raise ValueError("Pre-retrieval capability graph contains an invalid edge.")
    sources = {
        str(item.get("capability"))
        for item in graph.get("sources", ())
        if isinstance(item, Mapping)
    }
    terms = {
        str(item.get("capability"))
        for item in graph.get("search_terms", ())
        if isinstance(item, Mapping) and item.get("terms")
    }
    if sources != set(nodes) or terms != set(nodes):
        raise ValueError(
            "Pre-retrieval plan must bind every capability to provenance and search intent."
        )


def _graph_from_pre_retrieval_plan(
    plan: Mapping[str, Any],
    *,
    prompt: str,
    design: Mapping[str, Any],
) -> CapabilityGraph:
    validate_pre_retrieval_plan(plan, prompt=prompt, design=design)
    raw = plan["capability_graph"]
    assert isinstance(raw, Mapping)
    edges = tuple(
        (str(item["from"]), str(item["to"]))
        for item in raw.get("edges", ())
        if isinstance(item, Mapping)
    )
    sources = tuple(
        (str(item["capability"]), str(item["source"]))
        for item in raw.get("sources", ())
        if isinstance(item, Mapping)
    )
    search_terms = tuple(
        (
            str(item["capability"]),
            tuple(str(term) for term in item.get("terms", ()) if str(term).strip()),
        )
        for item in raw.get("search_terms", ())
        if isinstance(item, Mapping)
    )
    return CapabilityGraph(
        nodes=tuple(str(item) for item in raw.get("nodes", ())),
        edges=edges,
        sources=sources,
        search_terms=search_terms,
        source_plan_sha256=str(plan.get("plan_sha256") or ""),
    )


def decompose_capability_graph(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    semantic_router: Any = None,
) -> CapabilityGraph:
    """Locate reusable feature boundaries without using whole-mod topical similarity.

    Dotted/structured capability IDs from semantic design are preserved. Composite
    system labels are expanded into reusable behavioral boundaries. Prompt fallback
    intentionally recognizes only capability-bearing words; arbitrary theme/title
    tokens never become donor-search queries.
    """

    if isinstance(design, Mapping):
        frozen_plan = design.get("_pre_retrieval_plan")
        if isinstance(frozen_plan, Mapping):
            return _graph_from_pre_retrieval_plan(
                frozen_plan,
                prompt=prompt,
                design=design,
            )

    ordered: list[str] = []
    edges: list[tuple[str, str]] = []
    sources: dict[str, str] = {}
    search_terms: dict[str, list[str]] = {}
    seen: set[str] = set()
    limit = _capability_graph_limit()

    def at_limit() -> bool:
        return limit > 0 and len(ordered) >= limit

    def add(
        raw: Any,
        source: str,
        parent: str = "",
        *,
        expand: bool = True,
    ) -> str:
        value = _capability_id(raw)
        if not value:
            return ""
        expanded = _expand_capability(value) if expand else (value,)
        anchor = expanded[0] if expanded else value
        for node in expanded or (value,):
            if node not in seen and not at_limit():
                seen.add(node)
                ordered.append(node)
                sources[node] = source
            if parent and parent != node and parent in seen and node in seen:
                edge = (parent, node)
                if edge not in edges:
                    edges.append(edge)
        return anchor

    def register_search_terms(
        capability: str,
        values: Iterable[Any],
        *,
        include_predefined: bool = True,
    ) -> None:
        if not capability:
            return
        bucket = search_terms.setdefault(capability, [])
        if include_predefined:
            for predefined in search_queries_for_capability(capability):
                if predefined.casefold() not in {item.casefold() for item in bucket}:
                    bucket.append(predefined)
        for raw in values:
            value = " ".join(str(raw or "").split())
            if value and value.casefold() not in {item.casefold() for item in bucket}:
                bucket.append(value[:512])

    def walk(value: Any, source: str, parent: str = "", depth: int = 0) -> None:
        if at_limit():
            return
        if isinstance(value, str):
            add(value, source, parent)
            return
        if isinstance(value, Mapping):
            identity = value.get("id") or value.get("capability") or value.get("kind") or value.get("name")
            local_parent = parent
            if identity:
                local_parent = add(identity, source, parent) or parent
            for key, child in value.items():
                if str(key).casefold() in _CAPABILITY_KEYS:
                    walk(child, f"{source}.{key}", local_parent, depth + 1)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                walk(item, source, parent, depth + 1)

    catalog_used = False
    from .requirement_catalog import build_requirement_catalog
    ev_catalog = design.get("_evidence_request_catalog") if isinstance(design, Mapping) else None
    if ev_catalog and isinstance(ev_catalog.get("requirements"), Sequence):
        req_catalog = build_requirement_catalog(prompt, evidence_request_catalog=ev_catalog)
        capabilities_by_requirement: dict[str, tuple[str, ...]] = {}
        for req in req_catalog.requirements:
            source = f"evidence_request_catalog.{req.id}"
            bound: list[str] = []
            semantic = " ".join(str(req.normalized_statement or req.statement).split())
            original = " ".join(str(req.original_span or req.statement).split())
            for cap in req.provides:
                # An approved evidence catalog is the semantic authority. Never
                # re-expand or rename its capability identity through the ontology.
                anchor = add(cap, source, expand=False)
                if not anchor:
                    continue
                cap_words = anchor.replace(".", " ").replace("_", " ")
                register_search_terms(
                    anchor,
                    (
                        f"{cap_words} implementation {semantic}",
                        f"{cap_words} source {original}",
                        f"{cap_words} reusable implementation",
                    ),
                    include_predefined=False,
                )
                bound.append(anchor)
                catalog_used = True
            capabilities_by_requirement[req.id] = tuple(dict.fromkeys(bound))

        if not catalog_used and req_catalog.capabilities:
            for c_spec in req_catalog.capabilities:
                anchor = add(c_spec.id, "evidence_request_catalog.capability", expand=False)
                if anchor:
                    cap_words = anchor.replace(".", " ").replace("_", " ")
                    register_search_terms(
                        anchor,
                        (
                            f"{cap_words} reusable implementation",
                            f"{cap_words} source code",
                            f"{cap_words} minecraft implementation",
                        ),
                        include_predefined=False,
                    )
                    catalog_used = True

        if catalog_used:
            # Preserve only authored gameplay dependencies. Semantic derivation is
            # provenance, not an implementation dependency, and ontology defaults
            # are forbidden after requirement approval.
            raw_requirements = ev_catalog.get("requirements")
            if isinstance(raw_requirements, Sequence) and not isinstance(
                raw_requirements, (str, bytes, bytearray)
            ):
                for index, raw_requirement in enumerate(raw_requirements, 1):
                    if not isinstance(raw_requirement, Mapping):
                        continue
                    child_id = str(
                        raw_requirement.get("requirement_id")
                        or raw_requirement.get("id")
                        or f"REQ-{index:03d}"
                    )
                    raw_dependencies = raw_requirement.get("depends_on", ())
                    if not isinstance(raw_dependencies, Sequence) or isinstance(
                        raw_dependencies, (str, bytes, bytearray)
                    ):
                        continue
                    for parent_id in raw_dependencies:
                        for child_cap in capabilities_by_requirement.get(child_id, ()):
                            for parent_cap in capabilities_by_requirement.get(str(parent_id), ()):
                                if child_cap != parent_cap and (child_cap, parent_cap) not in edges:
                                    edges.append((child_cap, parent_cap))

            # Hard authority barrier: no raw-prompt resolver, semantic inference,
            # opaque fallback node, predefined ontology search query, or default
            # dependency may execute below this point.
            return CapabilityGraph(
                nodes=tuple(ordered),
                edges=tuple(edges),
                sources=tuple((node, sources[node]) for node in ordered),
                search_terms=tuple(
                    (node, tuple(search_terms.get(node, (node.replace(".", " "),))))
                    for node in ordered
                ),
            )

    if not catalog_used and isinstance(design, Mapping):
        for key, value in design.items():
            if str(key).casefold() in _CAPABILITY_KEYS:
                walk(value, f"design.{key}")
    if not catalog_used:
        for kind in module_kinds:
            add(kind, "module_kind")

    # Merge every explicit prompt requirement not already covered by the design or
    # EvidenceRequestCatalog.  Catalog presence cannot suppress a sibling clause.
    from .canonical_capability_ontology import (
        resolve_capabilities_from_phrase_structured,
    )
    from .capability_semantic_inference import enrich_resolution_with_semantic_inference

    # Capture whether a host-authored design/catalog/module already owns the
    # requirement scope before prompt supplementation begins.  Unknown raw prompt
    # text may never enlarge an authoritative graph unless semantic inference
    # supplies an explicit proposal.
    authoritative_scope = bool(ordered)
    prompt_res = resolve_capabilities_from_phrase_structured(str(prompt or ""))
    enriched_res = enrich_resolution_with_semantic_inference(prompt_res, router=semantic_router)
    selected_nodes = list(enriched_res.nodes)
    if authoritative_scope:
        prompt_words = {
            token.casefold()
            for token in _TOKEN.findall(str(prompt or ""))
        }

        def structured_design_covers_span(capability: str, span: str) -> bool:
            if not _is_explicit_capability_identifier(capability):
                return False
            capability_words = {
                token.casefold()
                for token in _TOKEN.findall(capability.replace(".", " ").replace("_", " "))
            }
            span_words = {
                token.casefold() for token in _TOKEN.findall(span)
            }
            return bool(
                capability_words
                and span_words
                and capability_words <= prompt_words
                and span_words <= capability_words
            )

        required_by_span: dict[str, list[Any]] = {}
        for node in enriched_res.nodes:
            if not node.is_required:
                continue
            # Deterministic unresolved placeholders are bookkeeping only.  A
            # design-scoped graph accepts prompt supplements only when they are
            # canonical explicit capabilities or router-evidenced proposals.
            if node.origin not in {"explicit", "provisional_inferred"}:
                continue
            if catalog_used and node.origin != "explicit":
                continue
            required_by_span.setdefault(node.source_span, []).append(node)

        missing_spans: set[str] = set()
        for span, roots in required_by_span.items():
            # A design may carry an explicit structured ID while the prompt
            # spells it as words.  Treat that exact lossless spelling as
            # covered before ontology expansion, without hiding other spans.
            authored_id = _capability_id(span)
            covered = any(
                item == authored_id
                or structured_design_covers_span(item, span)
                for item in ordered
            )
            if covered:
                register_search_terms(authored_id, (span,))
            for root in roots:
                if covered:
                    break
                capability = _capability_id(root.capability_id)
                expanded = _expand_capability(capability) if capability else ()
                matched = [item for item in (capability, *expanded) if item in seen]
                if matched:
                    covered = True
                    for item in matched:
                        register_search_terms(item, (span,))
            if not covered:
                missing_spans.add(span)
        selected_nodes = [
            node for node in enriched_res.nodes if node.source_span in missing_spans
        ]

    for node in selected_nodes:
        anchor = add(node.capability_id, f"prompt_resolution.{node.origin}")
        if anchor:
            register_search_terms(anchor, (node.source_span or anchor,))
    for u, v in enriched_res.edges:
        if u in seen and v in seen and (u, v) not in edges:
            edges.append((u, v))

    if not authoritative_scope:
        inferred_spans = {
            node.source_span
            for node in enriched_res.nodes
            if node.origin == "provisional_inferred" and node.source_span
        }
        for span in prompt_res.unresolved_spans:
            clean_span = str(span or "").strip()
            if not clean_span or clean_span in inferred_spans:
                continue
            # Preserve one unknown authored requirement as one opaque node.  Do
            # not hallucinate primary/state/logic children or dependencies.
            opaque_id = (
                "provisional:semantic_"
                + hashlib.sha256(clean_span.encode("utf-8")).hexdigest()[:12]
            )
            anchor = add(opaque_id, "prompt_resolution.provisional_opaque", expand=False)
            if anchor:
                register_search_terms(anchor, (clean_span,))

    if not ordered:
        add("gameplay.core", "fallback")

    # Wire explicit requires edges from atomic capability ontology
    from .canonical_capability_ontology import atomic_capability_definitions
    atomics = atomic_capability_definitions()
    for node in ordered:
        cap_def = atomics.get(node)
        if cap_def and cap_def.default_dependencies:
            for dep in cap_def.default_dependencies:
                if dep in seen:
                    edge = (node, dep)
                    if edge not in edges:
                        edges.append(edge)

    for node in ordered:
        if node not in search_terms or not search_terms[node]:
            register_search_terms(node, (node.replace(".", " "),))
    return CapabilityGraph(
        nodes=tuple(ordered),
        edges=tuple(edges),
        sources=tuple((node, sources[node]) for node in ordered),
        search_terms=tuple(
            (node, tuple(search_terms.get(node, (node.replace(".", " "),))))
            for node in ordered
        ),
    )


def _capability_id(raw: Any) -> str:
    text = str(raw or "").strip().casefold()
    if not text:
        return ""
    text = text.removeprefix("capability:")
    semantic_prefix = next(
        (prefix for prefix in ("unresolved:", "provisional:") if text.startswith(prefix)),
        "",
    )
    if semantic_prefix:
        text = text[len(semantic_prefix) :]
    if text in _CAPABILITY_HINTS:
        return semantic_prefix + text
    clean = re.sub(r"[^a-z0-9_.-]+", "_", text.casefold()).strip("_.-")
    clean = re.sub(r"_+", "_", clean)
    if not clean:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        clean = f"opaque_{digest}"
    if clean in {"minecraft", "mod", "module", "system", "feature"}:
        return ""
    return semantic_prefix + clean


def _is_explicit_capability_identifier(value: str) -> bool:
    return bool(
        re.fullmatch(r"[a-z][a-z0-9]*(?:[._:/-][a-z0-9_]+)+", value)
    )


def _expand_capability(value: str) -> tuple[str, ...]:
    # Structured identifiers are authored requirement identities.  They are not
    # ontology hints merely because they use underscores instead of dots.
    if _is_explicit_capability_identifier(value) and not value.startswith("unresolved:"):
        return (value,)
    res = resolve_capabilities_from_phrase(value)
    return res if res else (value,)


@dataclass(frozen=True)
class ReuseDecision:
    capability: str
    mode: str
    confidence: float
    fresh_implementation_cost: float
    fresh_verification_cost: float
    adaptation_cost: float = 0.0
    integration_cost: float = 0.0
    dependency_cost: float = 0.0
    reuse_verification_cost: float = 0.0
    uncertainty_penalty: float = 0.0
    source_id: str = ""
    donor: Mapping[str, Any] | None = None
    donor_slice: Any | None = None
    artifact_bundle: ReusableArtifactBundle | None = None
    rationale: str = ""
    proof_level: str = "DISCOVERED"
    proof_receipt: Any | None = None

    @property
    def fresh_total(self) -> float:
        return self.fresh_implementation_cost + self.fresh_verification_cost

    @property
    def expected_cost(self) -> float:
        if self.mode == "fresh":
            return self.fresh_total
        return max(
            0.0,
            self.adaptation_cost
            + self.integration_cost
            + self.dependency_cost
            + self.reuse_verification_cost
            + self.uncertainty_penalty,
        )

    @property
    def actual_reuse_gain(self) -> float:
        return max(0.0, self.fresh_total - self.expected_cost)

    @property
    def verified_reuse(self) -> bool:
        if self.mode == "fresh":
            return False
        from .proof_level import ProofLevel
        lvl = ProofLevel.from_value(self.proof_level)
        return lvl.is_verified()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "capability": self.capability,
            "mode": self.mode,
            "confidence": round(self.confidence, 6),
            "fresh_implementation_cost": round(self.fresh_implementation_cost, 4),
            "fresh_verification_cost": round(self.fresh_verification_cost, 4),
            "adaptation_cost": round(self.adaptation_cost, 4),
            "integration_cost": round(self.integration_cost, 4),
            "dependency_cost": round(self.dependency_cost, 4),
            "reuse_verification_cost": round(self.reuse_verification_cost, 4),
            "uncertainty_penalty": round(self.uncertainty_penalty, 4),
            "fresh_total_cost": round(self.fresh_total, 4),
            "expected_cost": round(self.expected_cost, 4),
            "actual_reuse_gain": round(self.actual_reuse_gain, 4),
            "verified_reuse": self.verified_reuse,
            "source_id": self.source_id,
            "rationale": self.rationale,
            "proof_level": self.proof_level,
        }
        if self.donor is not None:
            value["donor"] = dict(self.donor)
        if self.proof_receipt is not None:
            value["proof_receipt"] = self.proof_receipt.to_dict() if hasattr(self.proof_receipt, "to_dict") else self.proof_receipt
        if self.artifact_bundle is not None:
            value["artifact_bundle"] = self.artifact_bundle.to_dict()
        return value


@dataclass(frozen=True)
class CompositionSelection:
    bundles: tuple[ReusableArtifactBundle, ...] = ()
    joint_build_receipt: Mapping[str, Any] = field(default_factory=dict)
    conflicts_resolved: tuple[str, ...] = ()
    total_covered_requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/composition-selection-v1",
            "bundles": [b.to_dict() if hasattr(b, "to_dict") else b for b in self.bundles],
            "joint_build_receipt": dict(self.joint_build_receipt),
            "conflicts_resolved": list(self.conflicts_resolved),
            "total_covered_requirements": list(self.total_covered_requirements),
        }


@dataclass(frozen=True)
class TargetImplementationPlan:
    adapter: PlatformAdapter
    capabilities: tuple[ReuseDecision, ...]
    platform_evidence: _platform.TargetEvidence | None
    cross_component_integration_cost: float
    platform_verification_cost: float
    maintenance_risk: float
    total_expected_cost: float
    weighted_verified_reuse: float
    fresh_work: float
    adaptation_work: float
    verification_work: float
    uncertainty: float
    reusable_registry_candidates: int
    capability_graph: Mapping[str, Any] | None = None
    selected_composition: CompositionSelection | None = None
    residual_contracts: tuple[ResidualGenerationContract, ...] = ()

    @property
    def unresolved_capabilities(self) -> int:
        return sum(not item.verified_reuse for item in self.capabilities)

    @property
    def rank_key(self) -> tuple[float | int, ...]:
        evidence = self.platform_evidence
        quality = evidence.evidence_quality if evidence is not None else 0.0
        research = evidence.research_quality if evidence is not None else 0.0
        freshness = evidence.freshness if evidence is not None else 0.0
        # Lower total cost dominates. Reuse value and evidence quality break ties;
        # freshness is intentionally last.
        return (
            -round(self.total_expected_cost, 8),
            round(self.weighted_verified_reuse, 8),
            -self.unresolved_capabilities,
            quality,
            research,
            -round(self.maintenance_risk, 8),
            freshness,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/target-implementation-plan-v1",
            "target": {
                **self.adapter.public_dict(),
            },
            "capabilities": [item.to_dict() for item in self.capabilities],
            "reuse_ledger": [
                {
                    "capability": item.capability,
                    "status": (
                        "FRESH_REQUIRED"
                        if item.mode == "fresh"
                        else "VERIFIED_REUSE"
                        if item.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "HOST_VERIFIED"}
                        else "PARTIAL_REUSE"
                        if item.proof_level in {"PARTIAL_REUSE", "SUBGRAPH_COMPILE_VERIFIED"}
                        else "MATERIALIZED"
                        if item.proof_level == "MATERIALIZED"
                        else "READY_FOR_PROOF"
                        if item.proof_level == "CLOSURE_COMPLETE"
                        else "PINNED_CANDIDATE"
                        if item.proof_level == "PINNED"
                        else "CANDIDATE"
                        if item.proof_level == "DISCOVERED"
                        else "PARTIAL_REUSE"
                        if item.mode == "adapt"
                        else "FRESH_REQUIRED"
                    ),
                    "mode": item.mode,
                    "proof_level": item.proof_level,
                    "source_id": item.source_id,
                    "fresh_generation_scope": (
                        "forbidden"
                        if item.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "HOST_VERIFIED", "INTEGRATION_VERIFIED", "RUNTIME_BOOT_VERIFIED"} and item.mode != "adapt"
                        else "residual_only"
                        if item.proof_level in {"PARTIAL_REUSE", "SUBGRAPH_COMPILE_VERIFIED"} or (item.mode == "adapt" and item.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "PARTIAL_REUSE", "HOST_VERIFIED"})
                        else "full"
                    ),
                }
                for item in self.capabilities
            ],
            "weighted_verified_reuse": round(self.weighted_verified_reuse, 4),
            "fresh_work": round(self.fresh_work, 4),
            "adaptation_work": round(self.adaptation_work, 4),
            "verification_work": round(self.verification_work, 4),
            "cross_component_integration_cost": round(self.cross_component_integration_cost, 4),
            "platform_verification_cost": round(self.platform_verification_cost, 4),
            "maintenance_risk": round(self.maintenance_risk, 4),
            "uncertainty": round(self.uncertainty, 4),
            "total_expected_cost": round(self.total_expected_cost, 4),
            "unresolved_capabilities": self.unresolved_capabilities,
            "reusable_registry_candidates": self.reusable_registry_candidates,
            "capability_graph": dict(self.capability_graph) if isinstance(self.capability_graph, Mapping) else {"nodes": [item.capability for item in self.capabilities], "edges": [], "sources": []},
            "selected_composition": (
                self.selected_composition.to_dict()
                if self.selected_composition is not None
                else None
            ),
            "residual_contracts": [
                contract.to_dict() for contract in self.residual_contracts
            ],
            "selection_basis": (
                "minimum_expected_implementation_and_verification_work_after_verified_reuse"
            ),
        }


@dataclass(frozen=True)
class ReuseAwareOptimization:
    selected: PlatformAdapter
    selected_plan: TargetImplementationPlan
    base_optimization: _platform.PlatformOptimization
    candidate_plans: tuple[TargetImplementationPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/joint-platform-reuse-optimizer-v1",
            "selected": self.selected_plan.to_dict(),
            "candidates": [item.to_dict() for item in self.candidate_plans],
            "base_evidence": self.base_optimization.to_dict(),
            "selection_order": [
                "executable_provider_gate",
                "minimum_total_expected_cost",
                "verified_reuse_value",
                "minimum_fresh_delta",
                "evidence_quality",
                "maintenance_risk",
                "freshness_last",
            ],
            "research_translation": [
                "feature_location",
                "structural_code_search",
                "software_transplantation",
                "bounded_dependency_closure",
                "provenance_preserving_copy_reuse",
            ],
        }


def _load_verified_components_or_empty() -> tuple[VerifiedComponent, ...]:
    """Treat the remote reuse registry as optional evidence, never a plan blocker."""

    try:
        return load_verified_components()
    except Exception as exc:  # noqa: BLE001 - unavailable registry means fresh-only reuse
        _emit_discovery_log(
            f"verified reuse registry unavailable: {type(exc).__name__}: {exc}; "
            "continuing with fresh-only reuse where needed",
            exc_info=True,
        )
        return ()


def _resolve_target_adapters(
    target_keys: Sequence[tuple[str, str]],
) -> tuple[tuple[PlatformAdapter, ...], tuple[str, ...]]:
    """Resolve independent provider receipts concurrently, preserving key order."""

    resolved: dict[int, PlatformAdapter] = {}
    errors: dict[int, str] = {}

    def resolve(index: int, loader: str, version: str) -> tuple[int, PlatformAdapter]:
        return index, adapter_for_target(version, loader)

    workers = min(_workers(), len(target_keys))
    if workers <= 1:
        for index, (loader, version) in enumerate(target_keys):
            try:
                _, adapter = resolve(index, loader, version)
                resolved[index] = adapter
            except Exception as exc:  # noqa: BLE001 - candidate failures are independent
                message = (
                    f"target resolution skipped loader={loader} version={version}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors[index] = message
                # adapter_for_target already emitted the full root traceback.  Keep
                # this layer as one candidate summary instead of printing it twice.
                _emit_discovery_log(message)
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm-platform-resolve",
        ) as pool:
            futures = {
                pool.submit(resolve, index, loader, version): (index, loader, version)
                for index, (loader, version) in enumerate(target_keys)
            }
            for future in as_completed(futures):
                index, loader, version = futures[future]
                try:
                    resolved_index, adapter = future.result()
                    resolved[resolved_index] = adapter
                except Exception as exc:  # noqa: BLE001 - candidate failures are independent
                    message = (
                        f"target resolution skipped loader={loader} version={version}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    errors[index] = message
                    _emit_discovery_log(message)

    return (
        tuple(resolved[index] for index in sorted(resolved)),
        tuple(errors[index] for index in sorted(errors)),
    )


def optimize_platform_and_reuse(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    target_research_fn: _platform.TargetResearchFn | None = None,
    discovery_client: EcosystemDiscoveryClient | None = None,
    semantic_router: Any = None,
) -> ReuseAwareOptimization:
    """Evaluate the bounded live-target window and select the lowest-cost plan."""

    if (
        isinstance(design, Mapping)
        and isinstance(design.get("_evidence_request_catalog"), Mapping)
        and not isinstance(design.get("_pre_retrieval_plan"), Mapping)
    ):
        raise TypeError(
            "Platform/reuse discovery cannot start before the pre-retrieval semantic "
            "plan is frozen."
        )
    graph = decompose_capability_graph(
        prompt,
        design=design,
        module_kinds=module_kinds,
        semantic_router=semantic_router,
    )
    if graph.source_plan_sha256:
        print(
            "planning: verified semantic plan -> platform/reuse discovery "
            f"plan_sha256={graph.source_plan_sha256} capabilities={len(graph.nodes)}",
            flush=True,
        )
    queries = graph.nodes
    platform_diagnostics: list[str] = []
    target_keys = discover_target_keys(
        loader=loader_constraint,
        minecraft_version=version_constraint,
        limit_per_loader=_target_candidate_limit(),
        diagnostics=platform_diagnostics,
    )
    adapters, resolution_errors = _resolve_target_adapters(target_keys)
    if not adapters:
        detail = "; ".join((*platform_diagnostics, *resolution_errors))
        raise ValueError(
            "No executable platform provider can satisfy the requested target constraints. "
            f"Diagnostics: {detail or 'provider discovery returned no executable target'}"
        )

    discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
    if discovery_mode not in {"auto", "on", "off"}:
        raise ValueError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
    client = discovery_client
    evidence_discovery_enabled = discovery_mode != "off"
    if client is None and evidence_discovery_enabled:
        try:
            client = EcosystemDiscoveryClient()
        except Exception as exc:  # noqa: BLE001 - optional ecosystem client is recoverable
            _emit_discovery_log(
                f"ecosystem discovery client unavailable: {type(exc).__name__}: {exc}; "
                "using provider receipt and fresh-only planning",
                exc_info=True,
            )
            evidence_discovery_enabled = False
    if discovery_mode == "off" and len(adapters) != 1:
        raise ValueError(
            "Reuse-aware automatic version selection requires ecosystem discovery when multiple "
            "executable targets remain."
        )

    # Source search is capability-level and target-neutral. Do it once, then evaluate
    # every executable target against the same pinned donor candidates. This avoids
    # the old capability x version public-search cross product.
    donor_discovery = _parallel_donor_repository_discovery
    grounded_donors_available = bool(
        getattr(donor_discovery, "__mmm_grounded_donors__", False)
    )
    if evidence_discovery_enabled or grounded_donors_available:
        try:
            repository_candidates = donor_discovery(
                queries,
                client if evidence_discovery_enabled else None,
                capability_graph=graph.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - donor search is optional evidence
            message = (
                f"donor repository discovery failed: {type(exc).__name__}: {exc}"
            )
            _emit_discovery_log(
                f"discovery {message}; using fresh-only planning", exc_info=True
            )
            repository_candidates = {capability: () for capability in queries}
    else:
        repository_candidates = {capability: () for capability in queries}

    registry = _load_verified_components_or_empty()
    same_project = _declared_same_project_capabilities(design)
    plan_results: list[TargetImplementationPlan] = []

    def build(adapter: PlatformAdapter) -> TargetImplementationPlan:
        return _plan_target(
            adapter,
            capabilities=queries,
            design=design,
            platform_evidence=None,
            registry=registry,
            same_project=same_project,
            discovery_client=client,
            allow_network=evidence_discovery_enabled,
            capability_graph=graph.to_dict(),
            repository_candidates=repository_candidates,
        )

    def build_with_fallback(adapter: PlatformAdapter) -> TargetImplementationPlan:
        try:
            return build(adapter)
        except Exception as exc:  # noqa: BLE001 - donor analysis has a fresh-only fallback
            message = (
                f"target implementation analysis failed loader={adapter.loader} "
                f"version={adapter.minecraft_version}: {type(exc).__name__}: {exc}"
            )
            _emit_discovery_log(f"discovery {message}; using fresh-only planning", exc_info=True)
            return _fresh_only_plan(
                adapter,
                queries,
                None,
                len(registry),
                capability_graph=graph.to_dict(),
                discovery_errors=(message,),
            )

    workers = min(_workers(), len(adapters))
    if workers <= 1:
        plan_results = [build_with_fallback(adapter) for adapter in adapters]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-reuse-target") as pool:
            futures = {pool.submit(build, adapter): adapter for adapter in adapters}
            for future in as_completed(futures):
                try:
                    plan_results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - one target has a fresh-only fallback
                    # An individual target can lose its donor evidence without making
                    # all other executable targets unusable.
                    adapter = futures[future]
                    message = (
                        f"target implementation analysis failed loader={adapter.loader} "
                        f"version={adapter.minecraft_version}: {type(exc).__name__}: {exc}"
                    )
                    _emit_discovery_log(f"discovery {message}; using fresh-only planning", exc_info=True)
                    plan_results.append(
                        _fresh_only_plan(
                            adapter,
                            queries,
                            None,
                            len(registry),
                            capability_graph=graph.to_dict(),
                            discovery_errors=(message,),
                        )
                    )
    initial_ranked = tuple(
        sorted(
            plan_results,
            key=lambda item: (
                item.rank_key,
                item.adapter.loader,
                item.adapter.minecraft_version,
                item.adapter.adapter_id,
            ),
            reverse=True,
        )
    )
    if not initial_ranked:
        raise ValueError("No executable target produced an implementation plan.")

    # Every feasible candidate that may participate in selection receives the same
    # target-evidence pass.  A cheap pre-evidence score is not authority to discard
    # candidates, and an unverified candidate is never allowed to win.
    evidence_targets = _near_cost_targets(initial_ranked)
    evidence_by_id: dict[str, _platform.TargetEvidence] = {}
    if evidence_discovery_enabled and evidence_targets:
        try:
            matrix, matrix_errors = _platform._parallel_support_matrix(
                evidence_targets, queries, client
            )
        except Exception as exc:  # noqa: BLE001 - optional matrix failure is recoverable
            message = f"support matrix failed: {type(exc).__name__}: {exc}"
            _emit_discovery_log(f"discovery {message}; using empty matrix", exc_info=True)
            matrix = {adapter.adapter_id: {} for adapter in evidence_targets}
            matrix_errors = (message,)
        try:
            deep = _platform._parallel_deep(
                evidence_targets,
                queries=queries,
                matrix=matrix,
                client=client,
                target_research_fn=None,
                inherited_errors=matrix_errors,
                shallow_candidate_count=sum(len(v) for v in repository_candidates.values()),
            )
        except Exception as exc:  # noqa: BLE001 - optional evidence failure is recoverable
            message = f"deep target evidence failed: {type(exc).__name__}: {exc}"
            _emit_discovery_log(f"discovery {message}; using fresh-only evidence", exc_info=True)
            deep = ()
        if not deep:
            message = "deep target evidence returned no candidates"
            _emit_discovery_log(f"discovery {message}; using fresh-only evidence")
            deep = tuple(
                _fresh_evidence(
                    adapter,
                    queries,
                    discovery_errors=(*matrix_errors, message),
                )
                for adapter in evidence_targets
            )
        evidence_by_id = {item.adapter.adapter_id: item for item in deep}

    if not evidence_discovery_enabled:
        selectable = initial_ranked
    else:
        selectable = tuple(
            item
            for item in initial_ranked
            if item.adapter.adapter_id in evidence_by_id
        )
        if not selectable:
            raise ValueError("No executable platform target survived evidence verification.")
    adjusted = [
        _apply_platform_evidence(item, evidence_by_id.get(item.adapter.adapter_id))
        for item in selectable
    ]
    ranked = tuple(
        sorted(
            adjusted,
            key=lambda item: (
                item.rank_key,
                item.adapter.loader,
                item.adapter.minecraft_version,
                item.adapter.adapter_id,
            ),
            reverse=True,
        )
    )
    # Official/agentic target research is a winner-admission gate.  Verify ranked
    # candidates one at a time: a failed/unavailable/unresolved receipt removes
    # that candidate, then the next candidate is tried.  This avoids both an
    # unverified winner and an all-target RAG fan-out.
    if target_research_fn is not None:
        rejected_ids: set[str] = set()
        selected_plan: TargetImplementationPlan | None = None
        for candidate in ranked:
            try:
                selected_research = target_research_fn(candidate.adapter)
            except Exception as exc:  # noqa: BLE001 - try the next research candidate
                _emit_discovery_log(
                    f"target research failed loader={candidate.adapter.loader} "
                    f"version={candidate.adapter.minecraft_version}: "
                    f"{type(exc).__name__}: {exc}; trying next candidate",
                    exc_info=True,
                )
                selected_research = None
            if not _valid_target_research_receipt(selected_research, candidate.adapter):
                _emit_discovery_log(
                    f"target research rejected loader={candidate.adapter.loader} "
                    f"version={candidate.adapter.minecraft_version}; trying next candidate"
                )
                rejected_ids.add(candidate.adapter.adapter_id)
                continue
            evidence = candidate.platform_evidence
            if evidence is None:
                rejected_ids.add(candidate.adapter.adapter_id)
                continue
            quality, _ = _platform._research_quality(selected_research)
            evidence = _replace_evidence_research(evidence, selected_research, quality)
            selected_plan = _replace_plan_evidence(candidate, evidence)
            break
        if selected_plan is None:
            raise ValueError(
                "No executable platform target produced valid target research evidence."
            )
        ranked = (
            selected_plan,
            *(
                item
                for item in ranked
                if item.adapter.adapter_id not in rejected_ids
                and item.adapter.adapter_id != selected_plan.adapter.adapter_id
            ),
        )
    else:
        selected_plan = ranked[0]

    evidence_items = tuple(
        item.platform_evidence
        for item in ranked
        if item.platform_evidence is not None
    )
    selected_evidence = selected_plan.platform_evidence
    if selected_evidence is None:
        selected_evidence = _fresh_evidence(selected_plan.adapter, queries)
    base = _platform.PlatformOptimization(
        selected=selected_plan.adapter,
        evidence=selected_evidence,
        candidates=evidence_items or (selected_evidence,),
        capability_queries=tuple(queries),
        discovery_mode="all-executable-targets_joint-reuse-cost",
    )
    return ReuseAwareOptimization(
        selected=selected_plan.adapter,
        selected_plan=selected_plan,
        base_optimization=base,
        candidate_plans=ranked,
    )


def plan_fixed_target(
    adapter: PlatformAdapter,
    *,
    capabilities: Sequence[str],
    design: Mapping[str, Any] | None,
    platform_evidence: _platform.TargetEvidence | None = None,
    discovery_client: EcosystemDiscoveryClient | None = None,
    allow_network: bool = True,
    capability_graph: Mapping[str, Any] | None = None,
    repository_candidates: Mapping[str, Sequence[str]] | None = None,
) -> TargetImplementationPlan:
    client = discovery_client
    if client is None and allow_network:
        try:
            client = EcosystemDiscoveryClient()
        except Exception as exc:  # noqa: BLE001 - optional ecosystem client is recoverable
            _emit_discovery_log(
                f"ecosystem discovery client unavailable: {type(exc).__name__}: {exc}; "
                "using fresh-only planning",
                exc_info=True,
            )
            allow_network = False
    registry = _load_verified_components_or_empty()
    repositories = (
        dict(repository_candidates)
        if isinstance(repository_candidates, Mapping)
        else (
            _parallel_donor_repository_discovery(
                capabilities, client, capability_graph=capability_graph
            )
            if allow_network
            else {capability: () for capability in capabilities}
        )
    )
    return _plan_target(
        adapter,
        capabilities=capabilities,
        design=design,
        platform_evidence=platform_evidence,
        registry=registry,
        same_project=_declared_same_project_capabilities(design),
        discovery_client=client,
        allow_network=allow_network,
        capability_graph=capability_graph or {"schema_version": "mmm/capability-graph-v1", "nodes": list(capabilities), "edges": [], "sources": []},
        repository_candidates=repositories,
    )


def _plan_target(
    adapter: PlatformAdapter,
    *,
    capabilities: Sequence[str],
    design: Mapping[str, Any] | None,
    platform_evidence: _platform.TargetEvidence | None,
    registry: Sequence[VerifiedComponent],
    same_project: set[str],
    discovery_client: EcosystemDiscoveryClient | None,
    allow_network: bool,
    capability_graph: Mapping[str, Any] | None = None,
    repository_candidates: Mapping[str, Sequence[str]] | None = None,
) -> TargetImplementationPlan:
    candidates_by_cap: dict[str, Sequence[DonorSlice]] = {}
    if allow_network:
        for capability in capabilities:
            key = capability.casefold()
            if (
                key not in same_project
                and find_verified_component(registry, capability=capability, minecraft_version=adapter.minecraft_version, loader=adapter.loader) is None
                and capability not in adapter.deterministic_module_kinds
            ):
                cands = _discover_donor_candidates(
                    capability,
                    adapter=adapter,
                    discovery_client=discovery_client,
                    repositories=(repository_candidates or {}).get(capability, ()),
                )
                if cands:
                    candidates_by_cap[capability] = cands

    selected_composition_donors: dict[str, DonorSlice] = {}
    joint_composition_receipts: dict[str, Any] = {}
    if len(candidates_by_cap) > 1:
        from .composition_solver import (
            search_ranked_donor_composition_beams,
            verify_joint_composition_sandbox,
        )
        design_map = design if isinstance(design, Mapping) else {}
        target_ctx = {
            "target_package": str(design_map.get("package") or "ai.minecraft.generated.mod").strip(),
            "target_modid": str(design_map.get("mod_id") or "generated_mod").strip(),
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "java_version": adapter.java_version,
        }
        ranked_beams = search_ranked_donor_composition_beams(
            candidates_by_cap,
            target_loader=adapter.loader,
            target_minecraft=adapter.minecraft_version,
        )
        for comp_res in ranked_beams:
            if comp_res and comp_res.is_valid and comp_res.selected_donors:
                joint_passed, joint_build_receipt = verify_joint_composition_sandbox(
                    comp_res.selected_donors,
                    target_context=target_ctx,
                )
                if joint_passed:
                    for d in comp_res.selected_donors:
                        selected_composition_donors[d.capability] = d
                        joint_composition_receipts[d.capability] = joint_build_receipt
                    break

    decisions: list[ReuseDecision] = []
    for capability in capabilities:
        fresh_impl, fresh_verify = _fresh_cost(capability)
        key = capability.casefold()
        if key in same_project:
            decisions.append(
                ReuseDecision(
                    capability=capability,
                    mode="same_project",
                    confidence=0.99,
                    fresh_implementation_cost=fresh_impl,
                    fresh_verification_cost=fresh_verify,
                    integration_cost=0.08 * fresh_impl,
                    reuse_verification_cost=0.18 * fresh_verify,
                    source_id="current_project",
                    rationale="Existing project capability is explicitly evidenced and retained in-place.",
                    proof_level="HOST_VERIFIED",
                )
            )
            continue
        verified = find_verified_component(
            registry,
            capability=capability,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
        )
        if verified is not None:
            decisions.append(
                ReuseDecision(
                    capability=capability,
                    mode="mmm_verified",
                    confidence=0.97,
                    fresh_implementation_cost=fresh_impl,
                    fresh_verification_cost=fresh_verify,
                    adaptation_cost=0.05 * fresh_impl,
                    integration_cost=0.08 * fresh_impl,
                    reuse_verification_cost=0.12 * fresh_verify,
                    source_id=verified.component_id,
                    donor={"registry_component": verified.to_dict()},
                    rationale="Previously verified MMM component matches the exact target and capability.",
                    proof_level="COMPILE_VERIFIED",
                )
            )
            continue
        if capability in adapter.deterministic_module_kinds:
            decisions.append(
                ReuseDecision(
                    capability=capability,
                    mode="library",
                    confidence=0.95,
                    fresh_implementation_cost=fresh_impl,
                    fresh_verification_cost=fresh_verify,
                    integration_cost=0.10 * fresh_impl,
                    reuse_verification_cost=0.15 * fresh_verify,
                    source_id=f"host-api:{adapter.source_api_family}",
                    rationale="Executable provider exposes a host-verified deterministic API/generator path.",
                    proof_level="HOST_VERIFIED",
                )
            )
            continue

        candidates = (selected_composition_donors[capability],) if capability in selected_composition_donors else candidates_by_cap.get(capability, ())
        if not candidates and allow_network:
            candidates = _discover_donor_candidates(
                capability,
                adapter=adapter,
                discovery_client=discovery_client,
                repositories=(repository_candidates or {}).get(capability, ()),
            )
        if candidates:
            from .reuse_proof_executor import execute_candidate_fallback_loop

            design_map = design if isinstance(design, Mapping) else {}
            target_ws = str(design_map.get("_target_workspace") or os.environ.get("MMM_TARGET_WORKSPACE") or "").strip()
            target_ctx = {
                "target_package": str(design_map.get("package") or "ai.minecraft.generated.mod").strip(),
                "target_modid": str(design_map.get("mod_id") or "generated_mod").strip(),
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "java_version": adapter.java_version,
            }
            best_donor, receipts = execute_candidate_fallback_loop(
                candidates=candidates,
                capability=capability,
                target_workspace=target_ws,
                target_context=target_ctx,
                discovery_client=discovery_client,
            )
            if best_donor is not None:
                donor = best_donor
                donor_identity = f"{donor.repository}@{donor.commit_sha}"
                winning_receipt = next(
                    (r for r in receipts if r.candidate_id == donor_identity),
                    None,
                )
                proof_lvl = winning_receipt.proof_level if winning_receipt else "UNVERIFIED"

                closure_scale = max(1.0, len(donor.files) / 3.0)
                if proof_lvl in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "HOST_VERIFIED"}:
                    decisions.append(
                        ReuseDecision(
                            capability=capability,
                            mode="source_transplant",
                            confidence=donor.confidence,
                            fresh_implementation_cost=fresh_impl,
                            fresh_verification_cost=fresh_verify,
                            adaptation_cost=min(0.35 * fresh_impl, 0.10 * fresh_impl * closure_scale),
                            integration_cost=min(0.25 * fresh_impl, 0.07 * fresh_impl * closure_scale),
                            dependency_cost=0.04 * fresh_impl * max(0, len(donor.required_dependencies)) + 0.02 * fresh_impl * max(0, len(donor.files) - 1),
                            reuse_verification_cost=0.30 * fresh_verify,
                            uncertainty_penalty=(1.0 - donor.confidence) * fresh_impl,
                            source_id=f"{donor.repository}@{donor.commit_sha}",
                            donor=donor.to_dict(),
                            donor_slice=donor,
                            rationale="Pinned permissive donor slice passed compilation verification in isolated target sandbox.",
                            proof_level=proof_lvl,
                            proof_receipt=winning_receipt,
                        )
                    )
                    continue
                elif proof_lvl in {"PARTIAL_REUSE", "SUBGRAPH_COMPILE_VERIFIED", "MATERIALIZED", "CLOSURE_COMPLETE"}:
                    decisions.append(
                        ReuseDecision(
                            capability=capability,
                            mode="adapt",
                            confidence=max(0.5, donor.confidence - 0.15),
                            fresh_implementation_cost=fresh_impl,
                            fresh_verification_cost=fresh_verify,
                            adaptation_cost=0.45 * fresh_impl,
                            integration_cost=0.18 * fresh_impl,
                            dependency_cost=0.06 * fresh_impl * max(0, len(donor.required_dependencies)) + 0.03 * fresh_impl * max(0, len(donor.files) - 1),
                            reuse_verification_cost=0.45 * fresh_verify,
                            uncertainty_penalty=0.18 * fresh_impl,
                            source_id=f"{donor.repository}@{donor.commit_sha}",
                            donor=donor.to_dict(),
                            donor_slice=donor,
                            rationale="Pinned donor slice is structurally useful but requires adaptation or residual fresh generation.",
                            proof_level=proof_lvl,
                            proof_receipt=winning_receipt,
                        )
                    )
                    continue
            # If all candidates failed compilation (best_donor is None), fall through to fresh implementation!

        decisions.append(
            ReuseDecision(
                capability=capability,
                mode="fresh",
                confidence=1.0,
                fresh_implementation_cost=fresh_impl,
                fresh_verification_cost=fresh_verify,
                rationale="No verified reusable implementation survived provenance, target, and source-slice gates.",
            )
        )

    # The planner owns the conversion from historical planner shapes to the
    # production assembly contract.  DonorSlice remains provenance metadata;
    # FinalProjectAssembler consumes proof-bound bundles exclusively.
    bundles: list[ReusableArtifactBundle] = []
    residual_contracts: list[ResidualGenerationContract] = []
    joint_build_receipt: dict[str, Any] = {}
    bound_decisions: list[ReuseDecision] = []
    for decision in decisions:
        bundle = decision.artifact_bundle
        receipt = decision.proof_receipt

        if decision.mode == "same_project":
            bundle = _same_project_bundle(design, decision.capability)
            receipt = bundle.proof_receipt if bundle is not None else None
        elif decision.mode == "mmm_verified":
            component = find_verified_component(
                registry,
                capability=decision.capability,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
            )
            bundle = (
                _verified_component_bundle(component, decision.capability)
                if component is not None
                else None
            )
            receipt = bundle.proof_receipt if bundle is not None else None
        elif decision.donor_slice is not None and receipt is not None:
            level = ProofLevel.from_value(getattr(receipt, "proof_level", ""))
            if level.allows_reuse():
                contract = getattr(receipt, "contract", None)
                protected = (
                    contract.protected_artifacts
                    if isinstance(contract, ResidualGenerationContract)
                    else {}
                )
                bundle = ReusableArtifactBundle.from_donor_slice(
                    decision.donor_slice,
                    proof_receipt=receipt,
                    requirement_ids=(decision.capability,),
                    protected_artifacts=protected,
                )

        proof_level = decision.proof_level
        if receipt is not None:
            receipt_level = (
                receipt.get("proof_level")
                if isinstance(receipt, Mapping)
                else getattr(receipt, "proof_level", None)
            )
            if receipt_level:
                proof_level = str(receipt_level)

        if bundle is not None and bundle_proof_allows_reuse(bundle, receipt):
            bundles.append(bundle)
            contract = (
                getattr(receipt, "contract", None)
                if receipt is not None
                else None
            )
            if isinstance(contract, ResidualGenerationContract):
                residual_contracts.append(contract)
            if decision.capability in joint_composition_receipts:
                joint_build_receipt[decision.capability] = joint_composition_receipts[
                    decision.capability
                ]
            bound_decisions.append(
                replace(
                    decision,
                    artifact_bundle=bundle,
                    proof_receipt=receipt,
                    proof_level=proof_level,
                )
            )
            continue

        if decision.mode in {"same_project", "mmm_verified", "source_transplant", "adapt"}:
            fresh_impl, fresh_verify = _fresh_cost(decision.capability)
            bound_decisions.append(
                ReuseDecision(
                    capability=decision.capability,
                    mode="fresh",
                    confidence=1.0,
                    fresh_implementation_cost=fresh_impl,
                    fresh_verification_cost=fresh_verify,
                    rationale=(
                        "No materializable artifact bundle survived exact proof receipt "
                        "and byte-hash binding."
                    ),
                    proof_level="FRESH_REQUIRED",
                )
            )
            continue

        bound_decisions.append(replace(decision, artifact_bundle=None, proof_receipt=receipt))

    decisions = bound_decisions
    unique_contracts_list: list[ResidualGenerationContract] = []
    seen_contract_capabilities: set[str] = set()
    for contract in residual_contracts:
        if contract.capability in seen_contract_capabilities:
            continue
        seen_contract_capabilities.add(contract.capability)
        unique_contracts_list.append(contract)
    unique_contracts = tuple(unique_contracts_list)
    selected_composition = None
    if bundles:
        selected_composition = CompositionSelection(
            bundles=tuple(bundles),
            joint_build_receipt={
                "schema_version": "mmm/joint-reuse-proof-v1",
                "per_capability": joint_build_receipt,
            },
            total_covered_requirements=tuple(
                dict.fromkeys(
                    requirement_id
                    for bundle in bundles
                    for requirement_id in bundle.requirement_ids
                )
            ),
        )

    fresh_work = sum(item.fresh_implementation_cost for item in decisions if item.mode == "fresh")
    adaptation_work = sum(item.adaptation_cost + item.integration_cost for item in decisions if item.mode != "fresh")
    verification = sum(
        item.fresh_verification_cost if item.mode == "fresh" else item.reuse_verification_cost
        for item in decisions
    )
    uncertainty = sum(item.uncertainty_penalty for item in decisions)
    donor_decisions = [item for item in decisions if item.mode in {"source_transplant", "adapt"}]
    donor_sources = {item.source_id for item in donor_decisions if item.source_id}
    donor_count = len(donor_sources)
    cohesion_reuse = max(0, len(donor_decisions) - donor_count)
    dependency_edges = platform_evidence.dependency_edges if platform_evidence is not None else 0
    cross_component = max(
        0.0,
        0.15 * max(0, len(decisions) - 1)
        + 0.08 * donor_count
        - 0.05 * cohesion_reuse,
    )
    platform_verify = 1.0 + 0.05 * dependency_edges
    maintenance_risk = (
        (platform_evidence.integration_risk if platform_evidence is not None else 0.0)
        + 0.25 * donor_count
        + 0.50 * sum(item.mode == "fresh" for item in decisions)
    )
    total = (
        sum(item.expected_cost for item in decisions)
        + cross_component
        + platform_verify
        + maintenance_risk
    )
    verified_value = sum(
        item.actual_reuse_gain * item.confidence
        for item in decisions
        if item.verified_reuse
    )
    return TargetImplementationPlan(
        adapter=adapter,
        capabilities=tuple(decisions),
        platform_evidence=platform_evidence,
        cross_component_integration_cost=round(cross_component, 4),
        platform_verification_cost=round(platform_verify, 4),
        maintenance_risk=round(maintenance_risk, 4),
        total_expected_cost=round(total, 4),
        weighted_verified_reuse=round(verified_value, 4),
        fresh_work=round(fresh_work, 4),
        adaptation_work=round(adaptation_work, 4),
        verification_work=round(verification, 4),
        uncertainty=round(uncertainty, 4),
        reusable_registry_candidates=len(registry),
        capability_graph=dict(capability_graph) if isinstance(capability_graph, Mapping) else None,
        selected_composition=selected_composition,
        residual_contracts=unique_contracts,
    )


def _parallel_donor_repository_discovery(
    capabilities: Sequence[str],
    client: EcosystemDiscoveryClient,
    *,
    capability_graph: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Discover source donors broadly, then deep-inspect only representative repos."""

    return discover_repositories_for_graph(
        capabilities,
        client,
        capability_graph=capability_graph,
    )


def _near_cost_targets(plans: Sequence[TargetImplementationPlan]) -> tuple[PlatformAdapter, ...]:
    """Return every feasible candidate that can participate in target selection.

    Kept as a private compatibility seam for tests/extensions that patched the old
    shortlist helper.  There is deliberately no fixed evidence width or cost-based
    pre-verification exclusion.
    """

    return tuple(item.adapter for item in plans)


def _apply_platform_evidence(
    plan: TargetImplementationPlan,
    evidence: _platform.TargetEvidence | None,
) -> TargetImplementationPlan:
    if evidence is None:
        return plan
    from dataclasses import replace

    platform_verify = 1.0 + 0.05 * evidence.dependency_edges
    donor_count = sum(item.mode in {"source_transplant", "adapt"} for item in plan.capabilities)
    maintenance = evidence.integration_risk + 0.25 * donor_count + 0.50 * plan.unresolved_capabilities
    decision_cost = sum(item.expected_cost for item in plan.capabilities)
    total = decision_cost + plan.cross_component_integration_cost + platform_verify + maintenance
    return replace(
        plan,
        platform_evidence=evidence,
        platform_verification_cost=round(platform_verify, 4),
        maintenance_risk=round(maintenance, 4),
        total_expected_cost=round(total, 4),
    )


def _discover_best_donor(
    capability: str,
    *,
    adapter: PlatformAdapter,
    discovery_client: EcosystemDiscoveryClient,
    repositories: Sequence[str],
) -> DonorSlice | None:
    """Inspect shortlisted repositories concurrently and return the strongest slice."""

def _discover_donor_candidates(
    capability: str,
    adapter: PlatformAdapter,
    discovery_client: EcosystemDiscoveryClient,
    repositories: Sequence[str],
) -> tuple[DonorSlice, ...]:
    """Inspect shortlisted repositories concurrently and return ranked candidate slices."""

    ordered = tuple(dict.fromkeys(repository for repository in repositories if repository))
    if not ordered:
        return ()

    def inspect(repository: str) -> DonorSlice | None:
        try:
            return inspect_repository_slice(
                repository=repository,
                capability=capability,
                adapter=adapter,
                discovery_client=discovery_client,
            )
        except Exception as exc:  # noqa: BLE001 - one donor is independently recoverable
            _emit_discovery_log(
                f"donor inspection failed repository={repository} "
                f"capability={capability}: {type(exc).__name__}: {exc}; continuing",
                exc_info=True,
            )
            return None

    donors: list[DonorSlice] = []
    workers = min(_workers(), len(ordered))
    if workers <= 1:
        donors = [donor for donor in (inspect(repository) for repository in ordered) if donor is not None]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-donor-inspect") as pool:
            futures = {
                pool.submit(inspect, repository): repository
                for repository in ordered
            }
            for future in as_completed(futures):
                try:
                    donor = future.result()
                except Exception as exc:  # noqa: BLE001 - one donor is independently recoverable
                    repository = futures[future]
                    _emit_discovery_log(
                        f"donor inspection failed repository={repository} "
                        f"capability={capability}: {type(exc).__name__}: {exc}; continuing",
                        exc_info=True,
                    )
                    continue
                if donor is not None:
                    donors.append(donor)
    if not donors:
        return ()

    def executable_gain(donor: DonorSlice) -> float:
        # gain = fresh_cost - expected(adaptation + dependency_risk + truncation_penalty)
        fresh_w, _ = _fresh_cost(capability)
        adaptation_penalty = 0.05 * getattr(donor, "adaptation_cost", 0.0)
        dep_penalty = 0.25 * len(donor.required_dependencies)
        truncation_penalty = 2.0 if not getattr(donor, "closure_complete", True) else 0.0
        return (fresh_w * donor.confidence) - (adaptation_penalty + dep_penalty + truncation_penalty)

    return tuple(
        sorted(
            donors,
            key=lambda donor: (
                executable_gain(donor),
                donor.exact_target,
                donor.confidence,
                -getattr(donor, "adaptation_cost", 0.0),
                -len(donor.required_dependencies),
                donor.repository,
            ),
            reverse=True,
        )
    )


def _discover_best_donor(
    capability: str,
    adapter: PlatformAdapter,
    discovery_client: EcosystemDiscoveryClient,
    repositories: Sequence[str],
) -> DonorSlice | None:
    """Return the highest ranked candidate donor slice."""
    candidates = _discover_donor_candidates(
        capability=capability,
        adapter=adapter,
        discovery_client=discovery_client,
        repositories=repositories,
    )
    return candidates[0] if candidates else None


def _fresh_cost(capability: str) -> tuple[float, float]:
    tokens = {
        token.casefold()
        for token in _TOKEN.findall(capability.replace(".", " ").replace("-", " "))
    }
    structural = 1.0 + 0.20 * min(12, len(tokens))
    hard = sum(
        term in capability.casefold()
        for term in (
            "network", "transaction", "persistence", "sync", "permission", "world",
            "entity", "render", "migration", "serialization", "security",
        )
    )
    implementation = 10.0 * structural + 3.0 * hard
    verification = 4.0 * structural + 2.0 * hard
    return round(implementation, 4), round(verification, 4)


def _validated_existing_inventory(
    design: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(design, Mapping):
        return None
    raw_inventory = design.get("_existing_project_inventory")
    if not isinstance(raw_inventory, Mapping):
        raw_inventory = design.get("_existing_snapshot")
    if not isinstance(raw_inventory, Mapping):
        return None
    try:
        from .project_inventory import validate_project_inventory_payload

        return validate_project_inventory_payload(raw_inventory)
    except (ImportError, ValueError, TypeError, RecursionError):
        return None


def _same_project_bundle(
    design: Mapping[str, Any] | None,
    capability: str,
) -> ReusableArtifactBundle | None:
    inventory = _validated_existing_inventory(design)
    if not isinstance(inventory, Mapping):
        return None
    catalog = inventory.get("component_catalog")
    components = catalog.get("components") if isinstance(catalog, Mapping) else None
    if not isinstance(components, list):
        return None

    key = capability.casefold()
    selected: list[Mapping[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            continue
        provided = {
            str(value).strip().casefold().removeprefix("capability:")
            for value in component.get("provides", ())
        }
        if key in provided:
            selected.append(component)
    if not selected:
        return None

    file_hashes = {
        str(component.get("locator") or "").split("#", 1)[0].replace("\\", "/"): str(
            component.get("content_sha256") or ""
        )
        for component in selected
        if str(component.get("locator") or "").strip()
        and str(component.get("content_sha256") or "").strip()
    }
    if not file_hashes:
        return None
    source_ref = str(inventory.get("project_snapshot_sha256") or "current_workspace")
    bundle_id = f"same_project:{capability}"
    proof_receipt = {
        "schema_version": "mmm/same-project-proof-receipt-v1",
        "proof_level": "HOST_VERIFIED",
        "capability": capability,
        "bundle_id": bundle_id,
        "source_ref": source_ref,
        "inventory_sha256": str(inventory.get("inventory_sha256") or ""),
        "component_ids": [str(item.get("component_id") or "") for item in selected],
        "file_hashes": dict(sorted(file_hashes.items())),
    }
    symbols = tuple(
        dict.fromkeys(
            str(item.get("name") or "").strip()
            for item in selected
            if str(item.get("name") or "").strip()
        )
    )
    return ReusableArtifactBundle.from_same_project(
        capability,
        file_hashes=file_hashes,
        symbols=symbols,
        proof_receipt=proof_receipt,
        source_ref=source_ref,
        provenance={
            "source": "current_project",
            "inventory_sha256": str(inventory.get("inventory_sha256") or ""),
            "components": [dict(item) for item in selected],
        },
    )


def _registry_artifact_hashes(artifact: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    target_path = str(artifact.get("target_path") or "").replace("\\", "/").strip("/")
    target_sha = str(artifact.get("sha256") or "").strip()
    if target_path and target_sha:
        hashes[target_path] = target_sha
    raw_files = artifact.get("files")
    if isinstance(raw_files, Mapping):
        for path, value in raw_files.items():
            digest = value.get("sha256") if isinstance(value, Mapping) else ""
            normalized = str(path).replace("\\", "/").strip("/")
            if normalized and str(digest or "").strip():
                hashes[normalized] = str(digest)
    elif isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)):
        for item in raw_files:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path") or "").replace("\\", "/").strip("/")
            digest = str(item.get("sha256") or "").strip()
            if path and digest:
                hashes[path] = digest
    return hashes


def _registry_artifact_files(artifact: Mapping[str, Any]) -> dict[str, str | bytes]:
    """Extract only immutable registry file payloads; hashes alone are not bytes."""

    raw_files = artifact.get("files")
    files: dict[str, str | bytes] = {}
    if isinstance(raw_files, Mapping):
        iterable = raw_files.items()
        for raw_path, raw_value in iterable:
            path = str(raw_path).replace("\\", "/").strip("/")
            value = raw_value
            if isinstance(raw_value, Mapping):
                value = raw_value.get("content", raw_value.get("text"))
            if path and isinstance(value, (str, bytes)):
                files[path] = value
    elif isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)):
        for raw_value in raw_files:
            if not isinstance(raw_value, Mapping):
                continue
            path = str(raw_value.get("path") or "").replace("\\", "/").strip("/")
            value = raw_value.get("content", raw_value.get("text"))
            if path and isinstance(value, (str, bytes)):
                files[path] = value
    return files


def _verified_component_bundle(
    component: VerifiedComponent,
    capability: str,
) -> ReusableArtifactBundle | None:
    file_hashes = _registry_artifact_hashes(component.artifact)
    files = _registry_artifact_files(component.artifact)
    # A registry receipt without immutable source/resource bytes is useful
    # evidence, but cannot be staged as production reuse.
    if not file_hashes or set(file_hashes) - set(files):
        return None
    bundle = ReusableArtifactBundle.from_verified_component(
        component.component_id,
        capability,
        files=files,
        file_hashes=file_hashes,
        dependencies=component.required_dependencies,
        symbols=component.public_symbols,
        provenance={
            "component_id": component.component_id,
            "source_origin": component.source_origin,
            "source_commit": component.source_commit,
            "license_id": component.license_id,
            "artifact": dict(component.artifact),
        },
    )
    proof_receipt = {
        "schema_version": "mmm/registry-component-proof-receipt-v1",
        "proof_level": "COMPILE_VERIFIED",
        "capability": capability,
        "bundle_id": bundle.bundle_id,
        "source_ref": bundle.source_ref,
        "file_hashes": dict(bundle.file_hashes),
        "component_id": component.component_id,
        "source_commit": component.source_commit,
        "test_receipts": list(component.test_receipts),
        "target": {
            "minecraft_version": component.minecraft_version,
            "loader": component.loader,
        },
    }
    return replace(bundle, proof_receipt=proof_receipt)


def _declared_same_project_capabilities(design: Mapping[str, Any] | None) -> set[str]:
    """Return only capabilities backed by a validated existing-project inventory.

    Model-authored design prose is never reuse evidence.  The inventory scanner owns
    the locators and byte hashes and emits deterministic ``capability:`` aliases for
    exact symbol/resource identities.
    """

    inventory = _validated_existing_inventory(design)
    if not isinstance(inventory, Mapping):
        return set()
    catalog = inventory.get("component_catalog")
    components = catalog.get("components") if isinstance(catalog, Mapping) else None
    if not isinstance(components, list):
        return set()
    capabilities: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            continue
        for value in component.get("provides", ()):
            text = str(value).strip().casefold()
            if not text.startswith("capability:"):
                continue
            capability = text.removeprefix("capability:").strip()
            if capability:
                capabilities.add(capability)
    return capabilities


def _fresh_only_plan(
    adapter: PlatformAdapter,
    capabilities: Sequence[str],
    evidence: _platform.TargetEvidence | None,
    registry_count: int,
    *,
    capability_graph: Mapping[str, Any] | None = None,
    discovery_errors: Sequence[str] = (),
) -> TargetImplementationPlan:
    rationale = "Target donor analysis failed; fresh generation remains available."
    if discovery_errors:
        rationale += " See platform discovery diagnostics for the failed optional stages."
    decisions = tuple(
        ReuseDecision(
            capability=cap,
            mode="fresh",
            confidence=1.0,
            fresh_implementation_cost=_fresh_cost(cap)[0],
            fresh_verification_cost=_fresh_cost(cap)[1],
            rationale=rationale,
        )
        for cap in capabilities
    )
    total = sum(item.expected_cost for item in decisions) + 1.0 + 0.5 * len(decisions)
    return TargetImplementationPlan(
        adapter=adapter,
        capabilities=decisions,
        platform_evidence=evidence,
        cross_component_integration_cost=0.15 * max(0, len(decisions) - 1),
        platform_verification_cost=1.0,
        maintenance_risk=0.5 * len(decisions),
        total_expected_cost=round(total, 4),
        weighted_verified_reuse=0.0,
        fresh_work=round(sum(item.fresh_implementation_cost for item in decisions), 4),
        adaptation_work=0.0,
        verification_work=round(sum(item.fresh_verification_cost for item in decisions), 4),
        uncertainty=0.0,
        reusable_registry_candidates=registry_count,
        capability_graph=dict(capability_graph) if isinstance(capability_graph, Mapping) else None,
    )


def _fresh_evidence(
    adapter: PlatformAdapter,
    queries: Sequence[str],
    *,
    discovery_errors: Sequence[str] = (),
) -> _platform.TargetEvidence:
    return _platform.TargetEvidence(
        adapter=adapter,
        requested_capabilities=tuple(queries),
        covered_capabilities=(),
        exact_projects=(),
        exact_versions=0,
        verified_hash_files=0,
        dependency_edges=0,
        maintenance_signals=0,
        adoption=0,
        freshness=0.0,
        evidence_quality=0.0,
        integration_risk=float(len(queries)),
        residual_cost=len(queries),
        dependency_complexity=len(discovery_errors),
        discovery_errors=tuple(sorted(set(discovery_errors))),
    )


def _valid_target_research_receipt(
    payload: Any,
    adapter: PlatformAdapter,
) -> bool:
    """Validate the host research graph that admits a ranked target as winner."""

    if not isinstance(payload, Mapping):
        return False
    if payload.get("schema_version") != "mmm/central-evidence-graph-v1":
        return False
    if str(payload.get("status", "")).strip().casefold() in {
        "failed",
        "failure",
        "unavailable",
        "error",
    }:
        return False
    errors = payload.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) and errors:
        return False
    unresolved = payload.get("unresolved_official_domains")
    if not isinstance(unresolved, list) or unresolved:
        return False
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        return False
    if any(
        not isinstance(domain, Mapping) or not str(domain.get("domain_id", "")).strip()
        for domain in domains
    ):
        return False
    target = payload.get("target")
    if not isinstance(target, Mapping):
        return False
    if (
        str(target.get("minecraft_version", "")) != adapter.minecraft_version
        or str(target.get("loader", "")).strip().casefold() != adapter.loader
        or str(target.get("mappings", "")) != adapter.yarn_mappings
    ):
        return False
    if payload.get("authorization") != "none" or payload.get("retrieval_is_authority") is not False:
        return False
    claimed = str(payload.get("evidence_sha256", ""))
    if not claimed.startswith("sha256:") or len(claimed) != 71:
        return False
    from .spec import canonical_json

    unsigned = dict(payload)
    unsigned.pop("evidence_sha256", None)
    expected = "sha256:" + hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    return claimed == expected


def _replace_evidence_research(
    evidence: _platform.TargetEvidence,
    payload: Mapping[str, Any],
    quality: float,
) -> _platform.TargetEvidence:
    from dataclasses import replace

    return replace(
        evidence,
        research_quality=round(float(quality), 6),
        deep_research=dict(payload),
    )


def _replace_plan_evidence(
    plan: TargetImplementationPlan,
    evidence: _platform.TargetEvidence,
) -> TargetImplementationPlan:
    from dataclasses import replace

    return replace(plan, platform_evidence=evidence)


def _workers() -> int:
    raw = os.environ.get("MMM_REUSE_PARALLELISM", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 6
    return max(1, min(value, 12))


__all__ = [
    "REUSE_MODES",
    "CapabilityGraph",
    "ReuseAwareOptimization",
    "ReuseDecision",
    "TargetImplementationPlan",
    "compile_pre_retrieval_plan",
    "decompose_capability_graph",
    "optimize_platform_and_reuse",
    "plan_fixed_target",
    "validate_pre_retrieval_plan",
]
