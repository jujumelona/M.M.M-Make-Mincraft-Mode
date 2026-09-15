from __future__ import annotations

"""Static capability/reuse planning built on immutable platform evidence receipts.

The legacy planner mixed semantic decomposition, provider discovery, private optimizer
calls, bounded candidate windows, fresh-only fallbacks, and runtime monkeypatch targets.
This module keeps only deterministic planning contracts. Platform discovery and ranking
are delegated to :mod:`platform_evidence_pipeline`, which owns exhaustive evidence
collection and returns immutable ``PlatformAdapter`` receipts.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import platform_evidence_pipeline as _platform
from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import PlatformAdapter
from .proof_level import ProofLevel
from .residual_generation_contract import ResidualGenerationContract
from .reuse_artifacts import ReusableArtifactBundle

REUSE_MODES = (
    "same_project",
    "mmm_verified",
    "library",
    "source_transplant",
    "adapt",
    "fresh",
)
_PRE_RETRIEVAL_PLAN_SCHEMA = "mmm/pre-retrieval-semantic-plan-v1"
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class CapabilityGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]
    search_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    source_plan_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "mmm/capability-graph-v1",
            "nodes": list(self.nodes),
            "edges": [{"from": left, "to": right} for left, right in self.edges],
            "sources": [
                {"capability": capability, "source": source}
                for capability, source in self.sources
            ],
            "search_terms": [
                {"capability": capability, "terms": list(terms)}
                for capability, terms in self.search_terms
            ],
        }
        if self.source_plan_sha256:
            payload["source_plan_sha256"] = self.source_plan_sha256
        return payload


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
        return self.mode != "fresh" and ProofLevel.from_value(self.proof_level).is_verified()

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
            value["proof_receipt"] = (
                self.proof_receipt.to_dict()
                if hasattr(self.proof_receipt, "to_dict")
                else self.proof_receipt
            )
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
            "bundles": [
                item.to_dict() if hasattr(item, "to_dict") else item for item in self.bundles
            ],
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
            "target": self.adapter.public_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
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
            "capability_graph": (
                dict(self.capability_graph)
                if isinstance(self.capability_graph, Mapping)
                else {"nodes": [item.capability for item in self.capabilities], "edges": [], "sources": []}
            ),
            "selected_composition": (
                self.selected_composition.to_dict()
                if self.selected_composition is not None
                else None
            ),
            "residual_contracts": [item.to_dict() for item in self.residual_contracts],
            "selection_basis": "receipt_native_verified_evidence_then_residual_work",
        }


@dataclass(frozen=True)
class ReuseAwareOptimization:
    selected: PlatformAdapter
    selected_plan: TargetImplementationPlan
    base_optimization: _platform.PlatformOptimization
    candidate_plans: tuple[TargetImplementationPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/joint-platform-reuse-optimizer-v2",
            "selected": self.selected_plan.to_dict(),
            "candidates": [item.to_dict() for item in self.candidate_plans],
            "base_evidence": self.base_optimization.to_dict(),
            "selection_order": [
                "executable_provider_gate",
                "verified_evidence",
                "minimum_residual_work",
                "evidence_quality",
                "maintenance_risk",
            ],
        }


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


def _capability_id(raw: Any) -> str:
    text = str(raw or "").strip().casefold().removeprefix("capability:")
    if not text:
        return ""
    clean = re.sub(r"[^a-z0-9_.:/-]+", "_", text).strip("_.:/-")
    clean = re.sub(r"_+", "_", clean)
    return clean


def _requirement_capabilities(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    raw = requirement.get("provides")
    values: Iterable[Any]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = raw
    else:
        values = (requirement.get("capability"),)
    result = tuple(
        dict.fromkeys(
            capability for item in values if (capability := _capability_id(item))
        )
    )
    return result


def _planned_work_id(requirement_id: str) -> str:
    digest = hashlib.sha256(requirement_id.encode("utf-8")).hexdigest()[:12]
    return f"intent_{digest}"


def _catalog_requirements(design: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    catalog = design.get("_evidence_request_catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError("Authoritative request catalog must be frozen before retrieval planning.")
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Pre-retrieval semantic planning requires request requirements.")
    if not all(isinstance(item, Mapping) for item in requirements):
        raise TypeError("Request requirements must be objects.")
    return tuple(requirements)  # type: ignore[return-value]


def decompose_capability_graph(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    semantic_router: Any = None,
) -> CapabilityGraph:
    del semantic_router
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    sources: list[tuple[str, str]] = []
    terms: list[tuple[str, tuple[str, ...]]] = []

    if isinstance(design, Mapping):
        frozen = design.get("_pre_retrieval_plan")
        if isinstance(frozen, Mapping):
            validate_pre_retrieval_plan(frozen, prompt=prompt, design=design)
            raw = frozen.get("capability_graph")
            assert isinstance(raw, Mapping)
            return CapabilityGraph(
                nodes=tuple(str(item) for item in raw.get("nodes", ())),
                edges=tuple(
                    (str(item.get("from")), str(item.get("to")))
                    for item in raw.get("edges", ())
                    if isinstance(item, Mapping)
                ),
                sources=tuple(
                    (str(item.get("capability")), str(item.get("source")))
                    for item in raw.get("sources", ())
                    if isinstance(item, Mapping)
                ),
                search_terms=tuple(
                    (
                        str(item.get("capability")),
                        tuple(str(term) for term in item.get("terms", ()) if str(term).strip()),
                    )
                    for item in raw.get("search_terms", ())
                    if isinstance(item, Mapping)
                ),
                source_plan_sha256=str(frozen.get("plan_sha256") or ""),
            )

        requirements = _catalog_requirements(design)
        by_requirement: dict[str, tuple[str, ...]] = {}
        dependency_map: dict[str, tuple[str, ...]] = {}
        for raw in requirements:
            requirement_id = str(raw.get("requirement_id") or raw.get("id") or "").strip()
            if not requirement_id:
                raise ValueError("Every request requirement must have a stable identifier.")
            capabilities = _requirement_capabilities(raw)
            if not capabilities:
                raise ValueError(f"Request requirement {requirement_id} has no capability.")
            by_requirement[requirement_id] = capabilities
            deps = raw.get("depends_on", ())
            dependency_map[requirement_id] = (
                tuple(str(item).strip() for item in deps if str(item).strip())
                if isinstance(deps, Sequence) and not isinstance(deps, (str, bytes, bytearray))
                else ()
            )
            statement = str(
                raw.get("semantic_statement")
                or raw.get("statement")
                or raw.get("capability")
                or requirement_id
            ).strip()
            for capability in capabilities:
                if capability not in nodes:
                    nodes.append(capability)
                    sources.append((capability, f"request_catalog.{requirement_id}"))
                    terms.append(
                        (
                            capability,
                            tuple(dict.fromkeys((statement, capability.replace(".", " ")))),
                        )
                    )
        known = set(by_requirement)
        for child_id, dependencies in dependency_map.items():
            unknown = sorted(set(dependencies) - known)
            if unknown:
                raise ValueError(
                    f"Request requirement {child_id} has unknown dependencies: {unknown}."
                )
            for parent_id in dependencies:
                for child_capability in by_requirement[child_id]:
                    for parent_capability in by_requirement[parent_id]:
                        edge = (child_capability, parent_capability)
                        if child_capability != parent_capability and edge not in edges:
                            edges.append(edge)
    else:
        for raw in module_kinds:
            capability = _capability_id(raw)
            if capability and capability not in nodes:
                nodes.append(capability)
                sources.append((capability, "module_kind"))
                terms.append((capability, (capability.replace(".", " "),)))

    if not nodes:
        raise ValueError(
            "Capability decomposition requires an approved request catalog or explicit module kinds."
        )
    return CapabilityGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        sources=tuple(sources),
        search_terms=tuple(terms),
    )


def compile_pre_retrieval_plan(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
    catalog = design.get("_evidence_request_catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError("Authoritative request catalog must be frozen before retrieval planning.")
    requirements = _catalog_requirements(design)
    graph = decompose_capability_graph(prompt, design=design)

    requirement_ids = [
        str(item.get("requirement_id") or item.get("id") or "").strip()
        for item in requirements
    ]
    if any(not item for item in requirement_ids) or len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("Pre-retrieval requirement identifiers must be unique and non-empty.")
    work_by_requirement = {
        requirement_id: _planned_work_id(requirement_id) for requirement_id in requirement_ids
    }
    planned_work: list[dict[str, Any]] = []
    for raw in requirements:
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
            raise ValueError(f"Pre-retrieval requirement {requirement_id} has no capability.")
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
    if plan.get("prompt_sha256") != _sha256(prompt) or plan.get("prompt_char_length") != len(prompt):
        raise ValueError("Pre-retrieval semantic plan is stale for the request.")
    catalog = design.get("_evidence_request_catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError("Pre-retrieval plan has no authoritative request catalog.")
    if plan.get("request_catalog_sha256") != catalog.get("catalog_sha256"):
        raise ValueError("Pre-retrieval semantic plan is not bound to the request catalog.")

    requirements = _catalog_requirements(design)
    work = plan.get("planned_work")
    if not isinstance(work, list):
        raise TypeError("Pre-retrieval semantic plan has no planned work catalog.")
    requirement_ids = {
        str(item.get("requirement_id") or item.get("id") or "").strip()
        for item in requirements
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
        if (
            not isinstance(edge, Mapping)
            or edge.get("from") not in nodes
            or edge.get("to") not in nodes
        ):
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


def _decision_from_evidence(capability: str, evidence: _platform.TargetEvidence | None) -> ReuseDecision:
    composition = dict(evidence.composition_modes) if evidence is not None else {}
    mode = composition.get(capability, "fresh")
    if mode != "reuse":
        return ReuseDecision(
            capability=capability,
            mode="fresh",
            confidence=1.0,
            fresh_implementation_cost=1.0,
            fresh_verification_cost=1.0,
            rationale="No verified reusable artifact receipt covers this capability.",
            proof_level="DISCOVERED",
        )
    project = evidence.exact_projects[0] if evidence and evidence.exact_projects else ""
    return ReuseDecision(
        capability=capability,
        mode="library",
        confidence=max(0.0, min(1.0, evidence.evidence_quality if evidence else 0.0)),
        fresh_implementation_cost=1.0,
        fresh_verification_cost=1.0,
        reuse_verification_cost=1.0,
        source_id=project,
        rationale="Receipt-native platform evidence found a reusable project candidate; proof is still required before writes are protected.",
        proof_level="DISCOVERED",
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
    del design, discovery_client, allow_network, repository_candidates
    adapter.validate()
    normalized = tuple(dict.fromkeys(_capability_id(item) for item in capabilities if _capability_id(item)))
    if not normalized:
        raise ValueError("A fixed-target implementation plan requires at least one capability.")
    decisions = tuple(_decision_from_evidence(item, platform_evidence) for item in normalized)
    fresh_work = sum(item.fresh_implementation_cost for item in decisions if item.mode == "fresh")
    adaptation_work = sum(item.adaptation_cost for item in decisions)
    verification_work = sum(
        item.fresh_verification_cost if item.mode == "fresh" else item.reuse_verification_cost
        for item in decisions
    )
    uncertainty = sum(item.uncertainty_penalty for item in decisions)
    integration = max(0.0, float(len(decisions) - 1) * 0.1)
    platform_verification = 1.0
    maintenance = platform_evidence.integration_risk if platform_evidence is not None else 1.0
    total = (
        sum(item.expected_cost for item in decisions)
        + integration
        + platform_verification
        + maintenance
    )
    verified_reuse = sum(item.actual_reuse_gain for item in decisions if item.verified_reuse)
    residual_contracts = tuple(
        ResidualGenerationContract(capability=item.capability)
        for item in decisions
        if not item.verified_reuse
    )
    return TargetImplementationPlan(
        adapter=adapter,
        capabilities=decisions,
        platform_evidence=platform_evidence,
        cross_component_integration_cost=integration,
        platform_verification_cost=platform_verification,
        maintenance_risk=maintenance,
        total_expected_cost=total,
        weighted_verified_reuse=verified_reuse,
        fresh_work=fresh_work,
        adaptation_work=adaptation_work,
        verification_work=verification_work,
        uncertainty=uncertainty,
        reusable_registry_candidates=(
            len(platform_evidence.exact_projects) if platform_evidence is not None else 0
        ),
        capability_graph=capability_graph,
        residual_contracts=residual_contracts,
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
    graph = decompose_capability_graph(
        prompt,
        design=design,
        module_kinds=module_kinds,
        semantic_router=semantic_router,
    )
    base = _platform.optimize_platform_evidence(
        prompt,
        design=design,
        module_kinds=graph.nodes,
        loader_constraint=loader_constraint,
        version_constraint=version_constraint,
        discovery_client=discovery_client,
        target_research_fn=target_research_fn,
    )
    evidence_by_id = {item.adapter.adapter_id: item for item in base.candidates}
    plans = tuple(
        plan_fixed_target(
            evidence.adapter,
            capabilities=graph.nodes,
            design=design,
            platform_evidence=evidence,
            discovery_client=discovery_client,
            capability_graph=graph.to_dict(),
        )
        for evidence in base.candidates
    )
    if not plans:
        raise ValueError("Platform evidence returned no executable implementation plan.")
    ranked = tuple(
        sorted(
            plans,
            key=lambda item: (
                item.rank_key,
                item.adapter.adapter_id,
            ),
            reverse=True,
        )
    )
    selected = next(
        (item for item in ranked if item.adapter.adapter_id == base.selected.adapter_id),
        ranked[0],
    )
    if selected.adapter.adapter_id not in evidence_by_id:
        raise ValueError("Selected platform receipt is not present in the evidence ledger.")
    return ReuseAwareOptimization(
        selected=selected.adapter,
        selected_plan=selected,
        base_optimization=base,
        candidate_plans=ranked,
    )


__all__ = [
    "REUSE_MODES",
    "CapabilityGraph",
    "CompositionSelection",
    "ReuseAwareOptimization",
    "ReuseDecision",
    "TargetImplementationPlan",
    "compile_pre_retrieval_plan",
    "decompose_capability_graph",
    "optimize_platform_and_reuse",
    "plan_fixed_target",
    "validate_pre_retrieval_plan",
]
