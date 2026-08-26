from __future__ import annotations

"""Capability-level joint platform/reuse optimisation.

This module upgrades platform selection from "a compatible project exists" to an
implementation plan.  Every executable target is scored from the work that remains
after verified same-project, MMM-registry, host/API, pinned source-transplant, adapt,
and fresh options are considered.  Counts are diagnostic only; selection minimises
expected implementation + verification work and preserves donor provenance.
"""

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .component_registry import (
    VerifiedComponent,
    find_verified_component,
    load_verified_components,
)
from .ecosystem_discovery import EcosystemDiscoveryClient
from .reuse_discovery import discover_repositories_for_graph
from .platform_catalog import PlatformAdapter, adapter_for_target, discover_target_keys
from . import platform_optimizer as _platform
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
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3131-\u318e\uac00-\ud7a3]+")
_CAPABILITY_KEYS = frozenset({
    "capabilities", "systems", "features", "requirements", "behaviors", "services",
    "subsystems", "modules", "actions", "operations",
})
from .canonical_capability_ontology import (
    canonical_domain_map as _canonical_domain_map,
    resolve_capabilities_from_phrase,
    search_queries_for_capability,
    romanize_korean_universal,
)

_CAPABILITY_HINTS = _canonical_domain_map()
_PROMPT_CAPABILITY_WORDS = frozenset(_CAPABILITY_HINTS)


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


@dataclass(frozen=True)
class CapabilityGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]
    search_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/capability-graph-v1",
            "nodes": list(self.nodes),
            "edges": [{"from": left, "to": right} for left, right in self.edges],
            "sources": [{"capability": cap, "source": source} for cap, source in self.sources],
            "search_terms": [
                {"capability": cap, "terms": list(terms)}
                for cap, terms in self.search_terms
            ],
        }


def decompose_capability_graph(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
) -> CapabilityGraph:
    """Locate reusable feature boundaries without using whole-mod topical similarity.

    Dotted/structured capability IDs from semantic design are preserved. Composite
    system labels are expanded into reusable behavioral boundaries. Prompt fallback
    intentionally recognizes only capability-bearing words; arbitrary theme/title
    tokens never become donor-search queries.
    """

    ordered: list[str] = []
    edges: list[tuple[str, str]] = []
    sources: dict[str, str] = {}
    search_terms: dict[str, list[str]] = {}
    seen: set[str] = set()
    limit = _capability_graph_limit()

    def at_limit() -> bool:
        return limit > 0 and len(ordered) >= limit

    def add(raw: Any, source: str, parent: str = "") -> str:
        value = _capability_id(raw)
        if not value:
            return ""
        expanded = _expand_capability(value)
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

    def register_search_terms(capability: str, values: Iterable[Any]) -> None:
        if not capability:
            return
        bucket = search_terms.setdefault(capability, [])
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
    if isinstance(design, Mapping):
        catalog = design.get("_evidence_request_catalog")
        requirements = catalog.get("requirements") if isinstance(catalog, Mapping) else None
        if isinstance(requirements, Sequence) and not isinstance(requirements, (str, bytes)):
            for requirement in requirements:
                if not isinstance(requirement, Mapping):
                    continue
                capability = requirement.get("capability")
                if not str(capability or "").strip():
                    provides = requirement.get("provides")
                    if isinstance(provides, Sequence) and not isinstance(provides, (str, bytes)):
                        capability = next((item for item in provides if str(item or "").strip()), "")
                source = f"evidence_request_catalog.{requirement.get('requirement_id') or 'requirement'}"
                anchor = add(capability, source)
                if anchor:
                    register_search_terms(
                        anchor,
                        (capability, requirement.get("statement")),
                    )
                    catalog_used = True
        if not catalog_used:
            for key, value in design.items():
                if str(key).casefold() in _CAPABILITY_KEYS:
                    walk(value, f"design.{key}")
    if not catalog_used:
        for kind in module_kinds:
            add(kind, "module_kind")

    # Always resolve structured capabilities from prompt and enrich with semantic inference
    from .canonical_capability_ontology import resolve_capabilities_from_phrase_structured
    from .capability_semantic_inference import enrich_resolution_with_semantic_inference

    prompt_res = resolve_capabilities_from_phrase_structured(str(prompt or ""))
    enriched_res = enrich_resolution_with_semantic_inference(prompt_res)
    for node in enriched_res.nodes:
        anchor = add(node.capability_id, f"prompt_resolution.{node.origin}")
        if anchor:
            register_search_terms(anchor, (node.source_span or anchor,))
    for u, v in enriched_res.edges:
        if u in seen and v in seen and (u, v) not in edges:
            edges.append((u, v))

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
    if text in _CAPABILITY_HINTS:
        return text
    romanized = romanize_korean_universal(text)
    clean = re.sub(r"[^a-z0-9_.-]+", "_", romanized.casefold()).strip("_.-")
    clean = re.sub(r"_+", "_", clean)
    if not clean or clean in {"minecraft", "mod", "module", "system", "feature"}:
        return ""
    return clean[:128]


def _expand_capability(value: str) -> tuple[str, ...]:
    return resolve_capabilities_from_phrase(value)


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
    rationale: str = ""
    proof_level: str = "DISCOVERED"

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
        return self.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED"} or self.mode in {"same_project", "mmm_verified"}

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
            "source_id": self.source_id,
            "rationale": self.rationale,
            "proof_level": self.proof_level,
        }
        if self.donor is not None:
            value["donor"] = dict(self.donor)
        return value


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

    @property
    def unresolved_capabilities(self) -> int:
        return sum(item.mode == "fresh" for item in self.capabilities)

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
                        if item.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED"} or item.mode in {"same_project", "mmm_verified"}
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
                        else "VERIFIED_REUSE"
                    ),
                    "mode": item.mode,
                    "proof_level": item.proof_level,
                    "source_id": item.source_id,
                    "fresh_generation_scope": (
                        "full" if item.mode == "fresh" else "residual_only" if item.mode == "adapt" else "forbidden"
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


def optimize_platform_and_reuse(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    target_research_fn: _platform.TargetResearchFn | None = None,
    discovery_client: EcosystemDiscoveryClient | None = None,
) -> ReuseAwareOptimization:
    """Evaluate every executable target and select the lowest expected-cost plan."""

    graph = decompose_capability_graph(prompt, design=design, module_kinds=module_kinds)
    queries = graph.nodes
    target_keys = discover_target_keys(
        loader=loader_constraint,
        minecraft_version=version_constraint,
        limit_per_loader=32,
    )
    adapters: list[PlatformAdapter] = []
    for loader, version in target_keys:
        try:
            adapters.append(adapter_for_target(version, loader))
        except ValueError:
            continue
    if not adapters:
        raise ValueError("No executable platform provider can satisfy the requested target constraints.")

    discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
    if discovery_mode not in {"auto", "on", "off"}:
        raise ValueError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
    client = discovery_client or EcosystemDiscoveryClient()
    if discovery_mode == "off" and len(adapters) != 1:
        raise ValueError(
            "Reuse-aware automatic version selection requires ecosystem discovery when multiple "
            "executable targets remain."
        )

    # Source search is capability-level and target-neutral. Do it once, then evaluate
    # every executable target against the same pinned donor candidates. This avoids
    # the old capability x version public-search cross product.
    repository_candidates = (
        _parallel_donor_repository_discovery(queries, client, capability_graph=graph.to_dict())
        if discovery_mode != "off"
        else {capability: () for capability in queries}
    )

    registry = load_verified_components()
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
            allow_network=discovery_mode != "off",
            capability_graph=graph.to_dict(),
            repository_candidates=repository_candidates,
        )

    workers = min(_workers(), len(adapters))
    if workers <= 1:
        plan_results = [build(adapter) for adapter in adapters]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-reuse-target") as pool:
            futures = {pool.submit(build, adapter): adapter for adapter in adapters}
            for future in as_completed(futures):
                try:
                    plan_results.append(future.result())
                except Exception:
                    # An individual target can lose its donor evidence without making
                    # all other executable targets unusable.
                    adapter = futures[future]
                    plan_results.append(
                        _fresh_only_plan(
                            adapter, queries, None, len(registry), capability_graph=graph.to_dict()
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
    if discovery_mode != "off" and evidence_targets:
        matrix, matrix_errors = _platform._parallel_support_matrix(evidence_targets, queries, client)
        deep = _platform._parallel_deep(
            evidence_targets,
            queries=queries,
            matrix=matrix,
            client=client,
            target_research_fn=None,
            inherited_errors=matrix_errors,
            shallow_candidate_count=sum(len(v) for v in repository_candidates.values()),
        )
        evidence_by_id = {item.adapter.adapter_id: item for item in deep}

    if discovery_mode == "off":
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
            except Exception:
                selected_research = None
            if not _valid_target_research_receipt(selected_research, candidate.adapter):
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
    client = discovery_client or EcosystemDiscoveryClient()
    registry = load_verified_components()
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
    discovery_client: EcosystemDiscoveryClient,
    allow_network: bool,
    capability_graph: Mapping[str, Any] | None = None,
    repository_candidates: Mapping[str, Sequence[str]] | None = None,
) -> TargetImplementationPlan:
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
                )
            )
            continue

        candidates = ()
        if allow_network:
            candidates = _discover_donor_candidates(
                capability,
                adapter=adapter,
                discovery_client=discovery_client,
                repositories=(repository_candidates or {}).get(capability, ()),
            )
        if candidates:
            from .reuse_proof_executor import execute_candidate_fallback_loop

            target_ctx = {
                "target_package": "ai.minecraft.generated.mod",
                "target_modid": "generated_mod",
                "minecraft_version": "1.21.1",
                "loader": "fabric",
            }
            best_donor, receipts = execute_candidate_fallback_loop(
                candidates=candidates,
                capability=capability,
                target_workspace="",
                target_context=target_ctx,
            )
            donor = best_donor or candidates[0]
            winning_receipt = next((r for r in receipts if r.candidate_id.startswith(donor.repository)), receipts[-1] if receipts else None)
            proof_lvl = winning_receipt.proof_level if winning_receipt else ("COMPILE_VERIFIED" if donor.exact_target else "MATERIALIZED")

            closure_scale = max(1.0, len(donor.files) / 3.0)
            if donor.exact_target or proof_lvl in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED"}:
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
                        rationale="Pinned permissive donor slice matches target metadata and has a bounded source closure.",
                        proof_level=proof_lvl,
                    )
                )
            else:
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
                        rationale="Pinned donor slice is structurally useful but target metadata requires adaptation.",
                        proof_level=proof_lvl,
                    )
                )
            continue
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
        return inspect_repository_slice(
            repository=repository,
            capability=capability,
            adapter=adapter,
            discovery_client=discovery_client,
        )

    donors: list[DonorSlice] = []
    workers = min(_workers(), len(ordered))
    if workers <= 1:
        donors = [donor for donor in (inspect(repository) for repository in ordered) if donor is not None]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-donor-inspect") as pool:
            futures = [pool.submit(inspect, repository) for repository in ordered]
            for future in as_completed(futures):
                try:
                    donor = future.result()
                except Exception:
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


def _declared_same_project_capabilities(design: Mapping[str, Any] | None) -> set[str]:
    """Return only capabilities backed by a validated existing-project inventory.

    Model-authored design prose is never reuse evidence.  The inventory scanner owns
    the locators and byte hashes and emits deterministic ``capability:`` aliases for
    exact symbol/resource identities.
    """

    if not isinstance(design, Mapping):
        return set()
    raw_inventory = design.get("_existing_project_inventory")
    if not isinstance(raw_inventory, Mapping):
        raw_inventory = design.get("_existing_snapshot")
    if not isinstance(raw_inventory, Mapping):
        return set()
    try:
        from .project_inventory import validate_project_inventory_payload

        inventory = validate_project_inventory_payload(raw_inventory)
    except (ImportError, ValueError, TypeError, RecursionError):
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
) -> TargetImplementationPlan:
    decisions = tuple(
        ReuseDecision(
            capability=cap,
            mode="fresh",
            confidence=1.0,
            fresh_implementation_cost=_fresh_cost(cap)[0],
            fresh_verification_cost=_fresh_cost(cap)[1],
            rationale="Target donor analysis failed; fresh generation remains available.",
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


def _fresh_evidence(adapter: PlatformAdapter, queries: Sequence[str]) -> _platform.TargetEvidence:
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
        dependency_complexity=0,
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
    "decompose_capability_graph",
    "optimize_platform_and_reuse",
    "plan_fixed_target",
]
