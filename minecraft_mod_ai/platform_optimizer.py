from __future__ import annotations

"""Evidence-first, provider-backed Minecraft target optimisation.

Platform coordinates are host-owned. The optimiser decomposes semantic capabilities,
performs platform-neutral ecosystem discovery, builds an exact-metadata support matrix
across executable provider targets, deep-checks only the best hypotheses, fuses optional
target-scoped official research, resolves dependency graphs on demand, and ranks
verified reuse before popularity or freshness.
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import PlatformAdapter, adapter_for_target, discover_target_keys


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.-]{2,}")
_DEPENDENCY_NODE_BUDGET = 64


@dataclass(frozen=True)
class EcosystemProject:
    """Compatibility view retained for deterministic synthetic fixture injection."""

    project_id: str
    title: str
    loaders: frozenset[str]
    versions: frozenset[str]
    downloads: int
    modified: str


@dataclass(frozen=True)
class TargetEvidence:
    adapter: PlatformAdapter
    requested_capabilities: tuple[str, ...]
    covered_capabilities: tuple[str, ...]
    exact_projects: tuple[str, ...]
    exact_versions: int
    verified_hash_files: int
    dependency_edges: int
    maintenance_signals: int
    adoption: int
    freshness: float
    evidence_quality: float
    integration_risk: float
    residual_cost: int
    dependency_complexity: int
    discovery_errors: tuple[str, ...] = ()
    composition_modes: tuple[tuple[str, str], ...] = ()
    research_quality: float = 0.0
    deep_research: Mapping[str, Any] | None = None
    shallow_candidate_count: int = 0
    dependency_projects: tuple[str, ...] = ()
    dependency_closure_complete: bool = True

    @property
    def mandatory_coverage(self) -> int:
        return int(
            not self.requested_capabilities
            or len(self.covered_capabilities) == len(self.requested_capabilities)
        )

    @property
    def reuse_coverage(self) -> int:
        if self.composition_modes:
            return sum(mode == "reuse" for _capability, mode in self.composition_modes)
        return min(len(self.exact_projects), len(self.covered_capabilities))

    @property
    def rank_key(self) -> tuple[float | int, ...]:
        # Hard implementation coverage and verified reuse dominate quality/risk.
        # Adoption and freshness are intentionally last.
        return (
            self.mandatory_coverage,
            self.reuse_coverage,
            self.evidence_quality,
            self.research_quality,
            -self.integration_risk,
            -self.residual_cost,
            -self.dependency_complexity,
            self.maintenance_signals,
            self.adoption,
            self.freshness,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": {
                "minecraft_version": self.adapter.minecraft_version,
                "loader": self.adapter.loader,
                "adapter_id": self.adapter.adapter_id,
            },
            "requested_capabilities": list(self.requested_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "composition_modes": [
                {"capability": capability, "mode": mode}
                for capability, mode in self.composition_modes
            ],
            "exact_projects": list(self.exact_projects),
            "exact_versions": self.exact_versions,
            "verified_hash_files": self.verified_hash_files,
            "dependency_edges": self.dependency_edges,
            "dependency_projects": list(self.dependency_projects),
            "dependency_closure_complete": self.dependency_closure_complete,
            "maintenance_signals": self.maintenance_signals,
            "adoption": self.adoption,
            "freshness": self.freshness,
            "evidence_quality": self.evidence_quality,
            "research_quality": self.research_quality,
            "integration_risk": self.integration_risk,
            "residual_cost": self.residual_cost,
            "dependency_complexity": self.dependency_complexity,
            "mandatory_coverage": self.mandatory_coverage,
            "reuse_coverage": self.reuse_coverage,
            "rank_key": list(self.rank_key),
            "shallow_candidate_count": self.shallow_candidate_count,
            "discovery_errors": list(self.discovery_errors),
        }
        if isinstance(self.deep_research, Mapping):
            payload["deep_research"] = dict(self.deep_research)
        return payload


@dataclass(frozen=True)
class PlatformOptimization:
    selected: PlatformAdapter
    evidence: TargetEvidence
    candidates: tuple[TargetEvidence, ...]
    capability_queries: tuple[str, ...]
    discovery_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/platform-optimizer-v2",
            "discovery_mode": self.discovery_mode,
            "capability_queries": list(self.capability_queries),
            "selected": self.evidence.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "executed_methods": [
                "capability_decomposition",
                "platform_neutral_shallow_discovery",
                "executable_provider_support_matrix",
                "exact_target_deep_discovery",
                "on_demand_dependency_graph_resolution",
                "dependency_and_digest_inspection",
                "target_scoped_agentic_rag_when_supplied",
                "quality_aware_composition_ranking",
            ],
            "selection_order": [
                "executable_provider_gate",
                "mandatory_capability_coverage",
                "verified_reuse_coverage",
                "evidence_quality",
                "official_research_quality",
                "integration_risk",
                "residual_implementation_cost",
                "dependency_closure_complexity",
                "maintenance",
                "adoption",
                "freshness_last",
            ],
        }


SearchFn = Callable[[str], Sequence[EcosystemProject]]
VersionFn = Callable[[str], Sequence[Mapping[str, Any]]]
TargetResearchFn = Callable[[PlatformAdapter], Mapping[str, Any]]


def capability_queries(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
) -> tuple[str, ...]:
    """CAPIR/PERC-style request -> bounded semantic capability queries."""

    labels: list[str] = []
    labels.extend(_clean(value) for value in module_kinds if _clean(value))
    if isinstance(design, Mapping):
        for key in ("capabilities", "systems", "features"):
            raw = design.get(key)
            if not isinstance(raw, (list, tuple)):
                continue
            for item in raw:
                if isinstance(item, str):
                    value = _clean(item)
                elif isinstance(item, Mapping):
                    value = _clean(
                        str(item.get("name") or item.get("id") or item.get("kind") or "")
                    )
                else:
                    value = ""
                if value:
                    labels.append(value)
        modules = design.get("modules")
        if isinstance(modules, list):
            for item in modules:
                if not isinstance(item, Mapping):
                    continue
                value = _clean(
                    str(item.get("plugin_id") or item.get("kind") or item.get("name") or "")
                )
                if value:
                    labels.append(value)
    if not labels:
        stop = {
            "minecraft", "mod", "mods", "make", "create", "fabric", "forge",
            "neoforge", "quilt", "version", "java", "with", "that", "this",
            "the", "and", "for", "please",
        }
        labels.extend(
            token.casefold()
            for token in _TOKEN_RE.findall(str(prompt))
            if token.casefold() not in stop
        )
    result: list[str] = []
    seen: set[str] = set()
    for value in labels:
        key = value.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) == 12:
            break
    return tuple(result or ("minecraft gameplay",))


def optimize_platform(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    top_k: int = 4,
    discovery_client: EcosystemDiscoveryClient | None = None,
    target_research_fn: TargetResearchFn | None = None,
    search_fn: SearchFn | None = None,
    version_fn: VersionFn | None = None,
) -> PlatformOptimization:
    """Select an executable target from verified evidence, never newest-first policy.

    Synthetic ``search_fn``/``version_fn`` hooks exist only for deterministic unit
    fixtures. Production discovery goes exclusively through ``EcosystemDiscoveryClient``
    so network, cursor, license and exact-target policy have one owner.
    """

    queries = capability_queries(prompt, design=design, module_kinds=module_kinds)
    target_keys = discover_target_keys(
        loader=loader_constraint,
        minecraft_version=version_constraint,
        limit_per_loader=12,
    )
    if not target_keys:
        target = "/".join(
            value for value in (version_constraint, loader_constraint) if value
        ) or "automatic"
        raise ValueError(f"No executable platform provider can satisfy target {target!r}.")

    adapters: list[PlatformAdapter] = []
    for loader, version in target_keys:
        try:
            adapters.append(adapter_for_target(version, loader))
        except ValueError:
            continue
    if not adapters:
        raise ValueError("Executable platform targets were discovered but none resolved.")

    if search_fn is not None or version_fn is not None:
        return _optimize_fixture_path(
            queries,
            adapters=adapters,
            search_fn=search_fn,
            version_fn=version_fn,
        )

    discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
    if discovery_mode not in {"auto", "on", "off"}:
        raise ValueError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
    if discovery_mode == "off":
        # Do not silently perform public-network discovery when the operator or test
        # harness disabled it. A single executable provider receipt is sufficient to
        # preserve exact target ownership; multiple unscored candidates are ambiguous
        # and therefore fail closed instead of falling back to newest/order bias.
        if len(adapters) != 1:
            raise ValueError(
                "Ecosystem discovery is disabled and multiple executable platform "
                "targets remain. Supply an explicit Minecraft target or enable discovery."
            )
        offline = _optimize_fixture_path(
            queries,
            adapters=adapters,
            search_fn=lambda _query: (),
            version_fn=lambda _project: (),
        )
        return PlatformOptimization(
            selected=offline.selected,
            evidence=offline.evidence,
            candidates=offline.candidates,
            capability_queries=offline.capability_queries,
            discovery_mode="provider-receipt-only_discovery-disabled",
        )

    client = discovery_client or EcosystemDiscoveryClient()
    neutral, neutral_errors = _parallel_neutral_shallow(queries, client)
    shallow_count = sum(len(value) for value in neutral.values())

    matrix, matrix_errors = _parallel_support_matrix(adapters, queries, client)
    hypotheses = sorted(
        adapters,
        key=lambda adapter: (
            -_support_score(adapter, queries, matrix.get(adapter.adapter_id, {})),
            adapter.loader,
            adapter.minecraft_version,
            adapter.adapter_id,
        ),
    )
    width = max(1, min(int(top_k), 8, len(hypotheses)))
    shortlist = tuple(hypotheses[:width])

    deep = _parallel_deep(
        shortlist,
        queries=queries,
        matrix=matrix,
        client=client,
        target_research_fn=target_research_fn,
        inherited_errors=(*neutral_errors, *matrix_errors),
        shallow_candidate_count=shallow_count,
    )
    ranked = tuple(
        sorted(
            deep,
            key=lambda item: (
                item.rank_key,
                item.adapter.loader,
                item.adapter.minecraft_version,
                item.adapter.adapter_id,
            ),
            reverse=True,
        )
    )
    if not ranked:
        raise ValueError("No executable platform target survived evidence verification.")
    return PlatformOptimization(
        selected=ranked[0].adapter,
        evidence=ranked[0],
        candidates=ranked,
        capability_queries=queries,
        discovery_mode=(
            "neutral-shallow_then-executable-support-matrix_then-"
            "exact-deep_with-agentic-rag"
        ),
    )


def _parallel_neutral_shallow(
    queries: Sequence[str],
    client: EcosystemDiscoveryClient,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    errors: list[str] = []

    def run(query: str) -> tuple[str, tuple[str, ...]]:
        page = client.search("modrinth", query, limit=20, target_profile="minecraft_mod")
        return query, _candidate_ids(page)

    with ThreadPoolExecutor(
        max_workers=min(_workers(), max(1, len(queries))),
        thread_name_prefix="mmm-platform-neutral",
    ) as pool:
        futures = {pool.submit(run, query): query for query in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                key, ids = future.result()
                found[key] = ids
            except Exception as exc:
                found[query] = ()
                errors.append(f"neutral:{query}: {type(exc).__name__}: {exc}")
    return found, tuple(sorted(errors))


def _parallel_support_matrix(
    adapters: Sequence[PlatformAdapter],
    queries: Sequence[str],
    client: EcosystemDiscoveryClient,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], tuple[str, ...]]:
    matrix: dict[str, dict[str, tuple[str, ...]]] = {
        adapter.adapter_id: {} for adapter in adapters
    }
    errors: list[str] = []

    def run(adapter: PlatformAdapter, query: str) -> tuple[str, str, tuple[str, ...]]:
        page = client.search(
            "modrinth",
            query,
            limit=8,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            target_profile="minecraft_mod",
        )
        return adapter.adapter_id, query, _candidate_ids(page)

    jobs = [(adapter, query) for adapter in adapters for query in queries]
    with ThreadPoolExecutor(
        max_workers=min(_workers(), max(1, len(jobs))),
        thread_name_prefix="mmm-platform-matrix",
    ) as pool:
        futures = {
            pool.submit(run, adapter, query): (adapter, query)
            for adapter, query in jobs
        }
        for future in as_completed(futures):
            adapter, query = futures[future]
            try:
                adapter_id, key, ids = future.result()
                matrix[adapter_id][key] = ids
            except Exception as exc:
                matrix[adapter.adapter_id][query] = ()
                errors.append(
                    f"matrix:{adapter.minecraft_version}/{adapter.loader}:{query}: "
                    f"{type(exc).__name__}: {exc}"
                )
    return matrix, tuple(sorted(errors))


def _support_score(
    adapter: PlatformAdapter,
    queries: Sequence[str],
    exact_by_query: Mapping[str, Sequence[str]],
) -> int:
    return sum(
        bool(exact_by_query.get(query)) or query in adapter.deterministic_module_kinds
        for query in queries
    )


def _parallel_deep(
    adapters: Sequence[PlatformAdapter],
    *,
    queries: Sequence[str],
    matrix: Mapping[str, Mapping[str, tuple[str, ...]]],
    client: EcosystemDiscoveryClient,
    target_research_fn: TargetResearchFn | None,
    inherited_errors: Sequence[str],
    shallow_candidate_count: int,
) -> tuple[TargetEvidence, ...]:
    result: list[TargetEvidence] = []
    with ThreadPoolExecutor(
        max_workers=min(_workers(), max(1, len(adapters))),
        thread_name_prefix="mmm-platform-target",
    ) as pool:
        futures = {
            pool.submit(
                _deep_evidence,
                adapter,
                queries=queries,
                exact_by_query=matrix.get(adapter.adapter_id, {}),
                client=client,
                target_research_fn=target_research_fn,
                inherited_errors=inherited_errors,
                shallow_candidate_count=shallow_candidate_count,
            ): adapter
            for adapter in adapters
        }
        for future in as_completed(futures):
            try:
                result.append(future.result())
            except Exception:
                continue
    return tuple(result)


def _deep_evidence(
    adapter: PlatformAdapter,
    *,
    queries: Sequence[str],
    exact_by_query: Mapping[str, Sequence[str]],
    client: EcosystemDiscoveryClient,
    target_research_fn: TargetResearchFn | None,
    inherited_errors: Sequence[str],
    shallow_candidate_count: int,
) -> TargetEvidence:
    errors = list(inherited_errors)
    project_to_queries: dict[str, set[str]] = {}
    for query, candidate_ids in exact_by_query.items():
        for candidate_id in candidate_ids:
            if not candidate_id.startswith("modrinth:"):
                continue
            project_id = candidate_id.split(":", 1)[1]
            project_to_queries.setdefault(project_id, set()).add(query)

    root_ids = tuple(sorted(project_to_queries))[:16]
    research_payload: Mapping[str, Any] | None = None
    root_inspections: dict[str, Mapping[str, Any]] = {}
    jobs = len(root_ids) + int(target_research_fn is not None)
    if jobs:
        with ThreadPoolExecutor(
            max_workers=min(_workers(), max(1, jobs)),
            thread_name_prefix="mmm-platform-deep",
        ) as pool:
            inspection_futures = {
                pool.submit(
                    client.inspect_modrinth_project,
                    project_id,
                    minecraft_version=adapter.minecraft_version,
                    loader=adapter.loader,
                ): project_id
                for project_id in root_ids
            }
            research_future = pool.submit(target_research_fn, adapter) if target_research_fn else None
            for future in as_completed(inspection_futures):
                project_id = inspection_futures[future]
                try:
                    value = future.result()
                    if isinstance(value, Mapping):
                        root_inspections[project_id] = dict(value)
                except Exception as exc:
                    errors.append(f"inspect:{project_id}: {type(exc).__name__}: {exc}")
            if research_future is not None:
                try:
                    value = research_future.result()
                    if isinstance(value, Mapping):
                        research_payload = dict(value)
                except Exception as exc:
                    errors.append(
                        f"rag:{adapter.minecraft_version}/{adapter.loader}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    verified_roots: list[str] = []
    verified_queries: set[str] = set()
    exact_versions = 0
    hash_files = 0
    dependency_edges = 0
    maintenance = 0
    freshness = 0.0
    license_penalty = 0
    required_dependencies: set[str] = set()
    unresolved_dependencies = 0
    incompatible_edges = 0

    for project_id, inspection in root_inspections.items():
        metrics = _inspection_metrics(inspection)
        if not metrics["eligible"]:
            continue
        if not metrics["permissive"]:
            license_penalty += 1
            continue
        verified_roots.append(project_id)
        verified_queries.update(project_to_queries.get(project_id, ()))
        exact_versions += metrics["exact_versions"]
        hash_files += metrics["hash_files"]
        dependency_edges += metrics["dependency_edges"]
        freshness = max(freshness, metrics["freshness"])
        maintenance += 1
        required_dependencies.update(metrics["required_projects"])
        unresolved_dependencies += metrics["unresolved_required"]
        incompatible_edges += metrics["incompatible_edges"]

    closure = _resolve_dependency_closure(
        client,
        adapter,
        required_dependencies - set(verified_roots),
        errors=errors,
    )
    exact_versions += closure["exact_versions"]
    hash_files += closure["hash_files"]
    dependency_edges += closure["dependency_edges"]
    maintenance += closure["maintenance"]
    freshness = max(freshness, closure["freshness"])
    license_penalty += closure["license_penalty"]
    unresolved_dependencies += closure["unresolved"]
    incompatible_edges += closure["incompatible_edges"]

    covered: set[str] = set()
    modes: list[tuple[str, str]] = []
    for query in queries:
        if query in verified_queries:
            mode = "reuse"
            covered.add(query)
        elif query in adapter.deterministic_module_kinds:
            mode = "direct"
            covered.add(query)
        elif exact_by_query.get(query):
            mode = "compat"
            covered.add(query)
        elif verified_roots:
            mode = "addon"
        else:
            mode = "direct"
        modes.append((query, mode))

    unresolved_capabilities = max(0, len(queries) - len(covered))
    residual = sum(
        capability not in covered and mode in {"direct", "addon"}
        for capability, mode in modes
    )
    research_quality, research_unresolved = _research_quality(research_payload)
    dependency_complexity = (
        dependency_edges
        + license_penalty
        + research_unresolved
        + unresolved_dependencies
        + incompatible_edges
    )
    reuse_ratio = len(verified_queries) / max(1, len(queries))
    digest_ratio = hash_files / max(1, exact_versions)
    closure_bonus = 1.0 if closure["complete"] else 0.0
    evidence_quality = min(
        1.0,
        0.15
        + 0.35 * reuse_ratio
        + 0.20 * min(1.0, digest_ratio)
        + 0.20 * research_quality
        + 0.10 * closure_bonus,
    )
    integration_risk = (
        float(unresolved_capabilities)
        + 0.20 * sum(mode == "compat" for _capability, mode in modes)
        + 0.35 * sum(mode == "addon" for _capability, mode in modes)
        + 0.50 * unresolved_dependencies
        + 0.35 * incompatible_edges
        + math.log2(1 + dependency_complexity) * 0.15
    )

    return TargetEvidence(
        adapter=adapter,
        requested_capabilities=tuple(queries),
        covered_capabilities=tuple(sorted(covered)),
        exact_projects=tuple(sorted(verified_roots)),
        exact_versions=exact_versions,
        verified_hash_files=hash_files,
        dependency_edges=dependency_edges,
        maintenance_signals=maintenance,
        adoption=0,
        freshness=freshness,
        evidence_quality=round(evidence_quality, 6),
        research_quality=round(research_quality, 6),
        integration_risk=round(integration_risk, 6),
        residual_cost=residual,
        dependency_complexity=dependency_complexity,
        discovery_errors=tuple(sorted(set(errors))),
        composition_modes=tuple(modes),
        deep_research=research_payload,
        shallow_candidate_count=shallow_candidate_count,
        dependency_projects=tuple(sorted(closure["projects"])),
        dependency_closure_complete=bool(closure["complete"]),
    )


def _resolve_dependency_closure(
    client: EcosystemDiscoveryClient,
    adapter: PlatformAdapter,
    seed_projects: Iterable[str],
    *,
    errors: list[str],
) -> dict[str, Any]:
    """Resolve only the dependency graph reachable from candidate roots.

    This is deliberately on-demand rather than a static ecosystem graph. A safety
    budget prevents a malicious/degenerate metadata graph from expanding without
    bound; hitting it is recorded as incomplete evidence and penalised by ranking.
    """

    pending = sorted({str(value).strip() for value in seed_projects if str(value).strip()})
    seen: set[str] = set()
    exact_versions = 0
    hash_files = 0
    dependency_edges = 0
    maintenance = 0
    freshness = 0.0
    license_penalty = 0
    unresolved = 0
    incompatible_edges = 0

    while pending and len(seen) < _DEPENDENCY_NODE_BUDGET:
        batch = tuple(project for project in pending if project not in seen)
        pending = []
        if not batch:
            break
        remaining = _DEPENDENCY_NODE_BUDGET - len(seen)
        batch = batch[:remaining]
        with ThreadPoolExecutor(
            max_workers=min(_workers(), max(1, len(batch))),
            thread_name_prefix="mmm-platform-deps",
        ) as pool:
            futures = {
                pool.submit(
                    client.inspect_modrinth_project,
                    project_id,
                    minecraft_version=adapter.minecraft_version,
                    loader=adapter.loader,
                ): project_id
                for project_id in batch
            }
            for future in as_completed(futures):
                project_id = futures[future]
                seen.add(project_id)
                try:
                    inspection = future.result()
                except Exception as exc:
                    unresolved += 1
                    errors.append(f"dependency:{project_id}: {type(exc).__name__}: {exc}")
                    continue
                if not isinstance(inspection, Mapping):
                    unresolved += 1
                    continue
                metrics = _inspection_metrics(inspection)
                if not metrics["eligible"]:
                    unresolved += 1
                    continue
                if not metrics["permissive"]:
                    license_penalty += 1
                exact_versions += metrics["exact_versions"]
                hash_files += metrics["hash_files"]
                dependency_edges += metrics["dependency_edges"]
                maintenance += 1
                freshness = max(freshness, metrics["freshness"])
                unresolved += metrics["unresolved_required"]
                incompatible_edges += metrics["incompatible_edges"]
                for dependency in metrics["required_projects"]:
                    if dependency not in seen:
                        pending.append(dependency)
        pending = sorted(set(pending))

    complete = not pending
    if not complete:
        errors.append(
            "dependency-closure: reachable graph exceeded the safety node budget; "
            "candidate remains incomplete and is penalised"
        )
        unresolved += len(pending)
    return {
        "projects": seen,
        "exact_versions": exact_versions,
        "hash_files": hash_files,
        "dependency_edges": dependency_edges,
        "maintenance": maintenance,
        "freshness": freshness,
        "license_penalty": license_penalty,
        "unresolved": unresolved,
        "incompatible_edges": incompatible_edges,
        "complete": complete,
    }


def _inspection_metrics(inspection: Mapping[str, Any]) -> dict[str, Any]:
    policy = str(inspection.get("license_policy", ""))
    versions = inspection.get("versions")
    eligible_versions = [
        value
        for value in versions
        if isinstance(value, Mapping) and value.get("eligible_for_selection")
    ] if isinstance(versions, list) else []
    exact_versions = len(eligible_versions)
    hash_files = 0
    dependency_edges = 0
    required_projects: set[str] = set()
    unresolved_required = 0
    incompatible_edges = 0
    freshness = 0.0
    for version in eligible_versions:
        freshness = max(freshness, _timestamp(str(version.get("date_published", ""))))
        files = version.get("files")
        if isinstance(files, list):
            hash_files += sum(
                bool(str(item.get("sha512", "")))
                for item in files
                if isinstance(item, Mapping)
            )
        dependencies = version.get("dependencies")
        if not isinstance(dependencies, list):
            continue
        dependency_edges += len(dependencies)
        for dependency in dependencies:
            if not isinstance(dependency, Mapping):
                continue
            dependency_type = str(dependency.get("dependency_type", "")).strip().casefold()
            project_id = str(dependency.get("project_id", "") or "").strip()
            if dependency_type == "required":
                if project_id:
                    required_projects.add(project_id)
                else:
                    unresolved_required += 1
            elif dependency_type == "incompatible":
                incompatible_edges += 1
    return {
        "eligible": bool(eligible_versions),
        "permissive": policy.startswith("permissive_candidate"),
        "exact_versions": exact_versions,
        "hash_files": hash_files,
        "dependency_edges": dependency_edges,
        "required_projects": required_projects,
        "unresolved_required": unresolved_required,
        "incompatible_edges": incompatible_edges,
        "freshness": freshness,
    }


def _research_quality(payload: Mapping[str, Any] | None) -> tuple[float, int]:
    if not isinstance(payload, Mapping):
        return 0.0, 0
    domains = payload.get("domains")
    coverages: list[float] = []
    if isinstance(domains, list):
        for domain in domains:
            if not isinstance(domain, Mapping):
                continue
            fusion = domain.get("fusion")
            critic = fusion.get("critic") if isinstance(fusion, Mapping) else None
            if isinstance(critic, Mapping):
                try:
                    coverages.append(float(critic.get("mean_coverage", 0.0)))
                except (TypeError, ValueError):
                    pass
    unresolved_raw = payload.get("unresolved_official_domains")
    unresolved = len(unresolved_raw) if isinstance(unresolved_raw, list) else 0
    if coverages:
        score = sum(max(0.0, min(1.0, value)) for value in coverages) / len(coverages)
    else:
        score = 0.0 if unresolved else 0.5
    return max(0.0, min(1.0, score)), unresolved


def _candidate_ids(page: Mapping[str, Any]) -> tuple[str, ...]:
    raw = page.get("candidates")
    if not isinstance(raw, list):
        return ()
    return tuple(
        str(item.get("candidate_id", ""))
        for item in raw
        if isinstance(item, Mapping) and str(item.get("candidate_id", ""))
    )


def _optimize_fixture_path(
    queries: Sequence[str],
    *,
    adapters: Sequence[PlatformAdapter],
    search_fn: SearchFn | None,
    version_fn: VersionFn | None,
) -> PlatformOptimization:
    search = search_fn or (lambda _query: ())
    versions = version_fn or (lambda _project: ())
    projects_by_query: dict[str, Sequence[EcosystemProject]] = {
        query: tuple(search(query)) for query in queries
    }
    evidence: list[TargetEvidence] = []
    for adapter in adapters:
        covered: list[str] = []
        project_ids: set[str] = set()
        for query in queries:
            exact = [
                project
                for project in projects_by_query.get(query, ())
                if adapter.loader in project.loaders
                and adapter.minecraft_version in project.versions
            ]
            if exact or query in adapter.deterministic_module_kinds:
                covered.append(query)
            project_ids.update(project.project_id for project in exact)
        exact_versions = 0
        hashes = 0
        dependencies = 0
        for project_id in sorted(project_ids):
            for raw in versions(project_id):
                loaders = {str(value).casefold() for value in raw.get("loaders", ())}
                game_versions = {str(value) for value in raw.get("game_versions", ())}
                if adapter.loader not in loaders or adapter.minecraft_version not in game_versions:
                    continue
                exact_versions += 1
                raw_dependencies = raw.get("dependencies")
                if isinstance(raw_dependencies, list):
                    dependencies += len(raw_dependencies)
                raw_files = raw.get("files")
                if isinstance(raw_files, list):
                    hashes += sum(
                        bool(
                            isinstance(item, Mapping)
                            and isinstance(item.get("hashes"), Mapping)
                            and item["hashes"].get("sha512")
                        )
                        for item in raw_files
                    )
        residual = max(0, len(queries) - len(covered))
        evidence.append(
            TargetEvidence(
                adapter=adapter,
                requested_capabilities=tuple(queries),
                covered_capabilities=tuple(covered),
                exact_projects=tuple(sorted(project_ids)),
                exact_versions=exact_versions,
                verified_hash_files=hashes,
                dependency_edges=dependencies,
                maintenance_signals=len(project_ids),
                adoption=sum(
                    project.downloads
                    for values in projects_by_query.values()
                    for project in values
                    if project.project_id in project_ids
                ),
                freshness=max(
                    (
                        _timestamp(project.modified)
                        for values in projects_by_query.values()
                        for project in values
                        if project.project_id in project_ids
                    ),
                    default=0.0,
                ),
                evidence_quality=1.0 if covered else 0.0,
                integration_risk=float(residual),
                residual_cost=residual,
                dependency_complexity=dependencies,
            )
        )
    ranked = tuple(sorted(evidence, key=lambda item: item.rank_key, reverse=True))
    return PlatformOptimization(
        selected=ranked[0].adapter,
        evidence=ranked[0],
        candidates=ranked,
        capability_queries=tuple(queries),
        discovery_mode="fixture-injected",
    )


def _workers() -> int:
    raw = os.environ.get("MMM_RESEARCH_WORKERS", "").strip()
    try:
        value = int(raw) if raw else 8
    except ValueError:
        value = 8
    return max(1, min(16, value))


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _timestamp(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


__all__ = [
    "EcosystemProject",
    "PlatformOptimization",
    "TargetEvidence",
    "TargetResearchFn",
    "capability_queries",
    "optimize_platform",
]
