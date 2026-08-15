from __future__ import annotations

"""Capability-level joint platform/reuse optimisation.

This module upgrades platform selection from "a compatible project exists" to an
implementation plan.  Every executable target is scored from the work that remains
after verified same-project, MMM-registry, host/API, pinned source-transplant, adapt,
and fresh options are considered.  Counts are diagnostic only; selection minimises
expected implementation + verification work and preserves donor provenance.
"""

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
from .platform_catalog import PlatformAdapter, adapter_for_target, discover_target_keys
from . import platform_optimizer as _platform
from .source_transplant import (
    DonorSlice,
    inspect_repository_slice,
    repository_from_candidate,
)

REUSE_MODES = (
    "same_project",
    "mmm_verified",
    "library",
    "source_transplant",
    "adapt",
    "fresh",
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_CAPABILITY_KEYS = frozenset({
    "capabilities", "systems", "features", "requirements", "behaviors", "services",
    "subsystems", "modules", "actions", "operations",
})
_CAPABILITY_HINTS = {
    "trade": ("trade.offer_model", "trade.transaction", "trade.validation", "inventory.transfer", "network.trade_sync", "persistence.trade_state"),
    "shop": ("trade.shop_registry", "trade.player_shop", "ui.shop_menu", "permission.shop_owner"),
    "economy": ("economy.currency", "economy.balance_store", "economy.transaction"),
    "currency": ("economy.currency", "economy.balance_store"),
    "inventory": ("inventory.transfer", "inventory.validation"),
    "network": ("network.action_sync", "network.server_validation"),
    "sync": ("network.action_sync", "network.server_validation"),
    "persistence": ("persistence.state_store", "persistence.serialization"),
    "storage": ("persistence.state_store", "persistence.serialization"),
    "permission": ("permission.access_control",),
    "gui": ("ui.menu", "ui.action_validation"),
    "ui": ("ui.menu", "ui.action_validation"),
    "quest": ("quest.state", "quest.progression", "quest.reward"),
    "combat": ("combat.damage", "combat.validation", "network.combat_sync"),
    "entity": ("entity.lifecycle", "entity.state_sync"),
    "recipe": ("crafting.recipe",),
    "crafting": ("crafting.recipe", "crafting.validation"),
    "command": ("command.registration", "command.permission"),
    "worldgen": ("worldgen.placement", "worldgen.configuration"),
    "config": ("config.schema", "config.persistence"),
    "audit": ("audit.event_log",),
}
_PROMPT_CAPABILITY_WORDS = frozenset(_CAPABILITY_HINTS)
_GRAPH_LIMIT = 32


@dataclass(frozen=True)
class CapabilityGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/capability-graph-v1",
            "nodes": list(self.nodes),
            "edges": [{"from": left, "to": right} for left, right in self.edges],
            "sources": [{"capability": cap, "source": source} for cap, source in self.sources],
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
    seen: set[str] = set()

    def add(raw: Any, source: str, parent: str = "") -> str:
        value = _capability_id(raw)
        if not value:
            return ""
        expanded = _expand_capability(value)
        anchor = expanded[0] if expanded else value
        for node in expanded or (value,):
            if node not in seen and len(ordered) < _GRAPH_LIMIT:
                seen.add(node)
                ordered.append(node)
                sources[node] = source
            if parent and parent != node and parent in seen and node in seen:
                edge = (parent, node)
                if edge not in edges:
                    edges.append(edge)
        return anchor

    def walk(value: Any, source: str, parent: str = "", depth: int = 0) -> None:
        if depth > 6 or len(ordered) >= _GRAPH_LIMIT:
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

    if isinstance(design, Mapping):
        for key, value in design.items():
            if str(key).casefold() in _CAPABILITY_KEYS:
                walk(value, f"design.{key}")
    for kind in module_kinds:
        add(kind, "module_kind")

    if not ordered:
        words = {token.casefold() for token in _TOKEN.findall(str(prompt))}
        for word in sorted(words & _PROMPT_CAPABILITY_WORDS):
            add(word, "prompt_capability_word")
    if not ordered:
        add("gameplay.core", "fallback")
    return CapabilityGraph(
        nodes=tuple(ordered),
        edges=tuple(edges),
        sources=tuple((node, sources[node]) for node in ordered),
    )


def _capability_id(raw: Any) -> str:
    text = str(raw or "").strip().casefold()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_.-")
    text = re.sub(r"_+", "_", text)
    if not text or text in {"minecraft", "mod", "module", "system", "feature"}:
        return ""
    return text[:128]


def _expand_capability(value: str) -> tuple[str, ...]:
    if "." in value:
        return (value,)
    tokens = tuple(token.casefold() for token in _TOKEN.findall(value.replace("-", "_")))
    expanded: list[str] = []
    for token in tokens:
        for capability in _CAPABILITY_HINTS.get(token, ()):
            if capability not in expanded:
                expanded.append(capability)
    if expanded:
        return tuple(expanded)
    return (value,)


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
        return self.mode in {"same_project", "mmm_verified", "library", "source_transplant", "adapt"}

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
        _parallel_donor_repository_discovery(queries, client)
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

    # Ecosystem project metadata is secondary evidence, not a reuse admission gate.
    # Deep-check only near-optimal reuse-cost targets so version count does not create
    # an O(targets x capabilities) network bottleneck.
    near = _near_cost_targets(initial_ranked)
    evidence_by_id: dict[str, _platform.TargetEvidence] = {}
    if discovery_mode != "off" and near:
        matrix, matrix_errors = _platform._parallel_support_matrix(near, queries, client)
        deep = _platform._parallel_deep(
            near,
            queries=queries,
            matrix=matrix,
            client=client,
            target_research_fn=None,
            inherited_errors=matrix_errors,
            shallow_candidate_count=sum(len(v) for v in repository_candidates.values()),
        )
        evidence_by_id = {item.adapter.adapter_id: item for item in deep}

    adjusted = [
        _apply_platform_evidence(item, evidence_by_id.get(item.adapter.adapter_id))
        for item in initial_ranked
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
    selected_plan = ranked[0]

    # Official/agentic target research is a selected-target verification step, not
    # a reason to multiply expensive RAG calls across all versions.
    if target_research_fn is not None:
        try:
            selected_research = target_research_fn(selected_plan.adapter)
        except Exception:
            selected_research = None
        if isinstance(selected_research, Mapping):
            evidence = selected_plan.platform_evidence
            if evidence is not None:
                quality, _ = _platform._research_quality(selected_research)
                evidence = _replace_evidence_research(evidence, selected_research, quality)
                selected_plan = _replace_plan_evidence(selected_plan, evidence)
                ranked = tuple(
                    selected_plan if item.adapter.adapter_id == selected_plan.adapter.adapter_id else item
                    for item in ranked
                )

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
            _parallel_donor_repository_discovery(capabilities, client)
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

        donor = None
        if allow_network:
            donor = _discover_best_donor(
                capability,
                adapter=adapter,
                discovery_client=discovery_client,
                repositories=(repository_candidates or {}).get(capability, ()),
            )
        if donor is not None:
            closure_scale = max(1.0, len(donor.files) / 3.0)
            if donor.exact_target:
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
    donor_count = sum(item.mode in {"source_transplant", "adapt"} for item in decisions)
    dependency_edges = platform_evidence.dependency_edges if platform_evidence is not None else 0
    cross_component = 0.15 * max(0, len(decisions) - 1) + 0.08 * donor_count
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
) -> dict[str, tuple[str, ...]]:
    """Search OSS once per capability, independent of Minecraft target version."""

    result: dict[str, tuple[str, ...]] = {capability: () for capability in capabilities}

    def run(capability: str) -> tuple[str, tuple[str, ...]]:
        semantic = " ".join(_TOKEN.findall(capability.replace(".", " ")))
        queries = [f"{semantic} minecraft mod source"]
        if "trade" in capability.casefold():
            queries.insert(0, f"{semantic} transaction service persistence minecraft")
        repositories: list[str] = []
        for query in queries:
            try:
                page = client.search("github", query, limit=8, target_profile="minecraft_mod")
            except Exception:
                continue
            raw = page.get("candidates") if isinstance(page, Mapping) else None
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                continue
            for candidate in raw:
                if not isinstance(candidate, Mapping):
                    continue
                repository = repository_from_candidate(candidate)
                if repository and repository not in repositories:
                    repositories.append(repository)
                if len(repositories) >= 8:
                    break
        return capability, tuple(repositories)

    workers = min(_workers(), max(1, len(capabilities)))
    if workers <= 1:
        for capability in capabilities:
            key, values = run(capability)
            result[key] = values
        return result
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-donor-search") as pool:
        futures = {pool.submit(run, capability): capability for capability in capabilities}
        for future in as_completed(futures):
            capability = futures[future]
            try:
                key, values = future.result()
                result[key] = values
            except Exception:
                result[capability] = ()
    return result


def _near_cost_targets(plans: Sequence[TargetImplementationPlan]) -> tuple[PlatformAdapter, ...]:
    if not plans:
        return ()
    best = min(item.total_expected_cost for item in plans)
    limit = max(best * 1.15, best + 8.0)
    selected = [item.adapter for item in plans if item.total_expected_cost <= limit]
    return tuple(selected[:12])


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
    # Repository discovery happened once per capability before target evaluation.
    # Here we only test the same structural donor candidates against this target.
    best: DonorSlice | None = None
    for repository in repositories[:6]:
        donor = inspect_repository_slice(
            repository=repository,
            capability=capability,
            adapter=adapter,
            discovery_client=discovery_client,
        )
        if donor is None:
            continue
        if best is None or (
            donor.exact_target,
            donor.confidence,
            -len(donor.files),
            donor.repository,
        ) > (
            best.exact_target,
            best.confidence,
            -len(best.files),
            best.repository,
        ):
            best = donor
    return best


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
    if not isinstance(design, Mapping):
        return set()
    values: list[str] = []
    for key in ("existing_capabilities", "same_project_capabilities", "preserved_capabilities"):
        raw = design.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(str(item).strip().casefold() for item in raw if str(item).strip())
    return set(values)


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
