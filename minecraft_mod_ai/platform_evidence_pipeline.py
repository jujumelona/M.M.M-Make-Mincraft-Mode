from __future__ import annotations

"""Receipt-native, fail-closed platform evidence and optimisation.

The planner first discovers reuse hypotheses without resolving platform providers. It
then builds a Pareto frontier over stable Minecraft versions and resolves only frontier
members. Once a PlatformAdapter is admitted, every evidence operation consumes that
immutable receipt directly; no downstream layer converts it back to strings and asks a
provider to resolve it again.

Provider page size is a transport concern. Search pages are followed to exhaustion and
required dependency graphs are traversed to closure. There are no semantic top-k, root
count, or dependency-node truncation constants in this module.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .ecosystem_discovery import (
    EcosystemDiscoveryClient,
    EcosystemDiscoveryUnavailable,
    _MAX_PAGE_ITEMS,
    _code_license_policy,
    _normalize_modrinth_version,
)
from .platform_catalog import PlatformAdapter, executable_loaders, provider_for_loader
from .platform_live_discovery import discover_game_versions
from .spec import SpecValidationError

TargetResearchFn = Callable[[PlatformAdapter], Mapping[str, Any]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.-]{2,}|[가-힣]{2,}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_GENERIC_QUERY_TOKENS = frozenset(
    {
        "minecraft",
        "mod",
        "mods",
        "system",
        "semantic",
        "implementation",
        "implement",
        "task",
        "feature",
        "mechanic",
        "module",
        "generated",
        "generator",
        "interaction",
        "logic",
        "code",
    }
)


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
        # Every residual capability is implemented as custom source. The target is only
        # admitted if its provider receipt can execute that source-generation path.
        return 1

    @property
    def reuse_coverage(self) -> int:
        return sum(mode == "reuse" for _capability, mode in self.composition_modes)

    @property
    def rank_key(self) -> tuple[float | int, ...]:
        # Transparent lexicographic policy: maximize verified reuse first, then evidence
        # completeness/quality, then minimize residual work and dependency complexity.
        return (
            self.reuse_coverage,
            self.evidence_quality,
            self.research_quality,
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
            "schema_version": "mmm/platform-optimizer-v3",
            "discovery_mode": self.discovery_mode,
            "capability_queries": list(self.capability_queries),
            "selected": self.evidence.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "executed_methods": [
                "lossless_capability_decomposition",
                "exhaustive_platform_neutral_modrinth_metadata",
                "reuse_pareto_frontier",
                "single_provider_resolution_per_frontier_target",
                "receipt_native_exact_project_inspection",
                "required_dependency_closure",
                "target_scoped_official_research",
                "lexicographic_verified_reuse_ranking",
            ],
            "selection_order": [
                "executable_provider_gate",
                "verified_reuse_coverage",
                "evidence_quality",
                "official_research_quality",
                "residual_implementation_cost",
                "dependency_closure_complexity",
                "maintenance",
                "adoption",
                "freshness_last",
            ],
        }


@dataclass(frozen=True)
class _ShallowProject:
    project_id: str
    versions: frozenset[str]
    loaders: frozenset[str]
    downloads: int
    modified: str


@dataclass(frozen=True)
class _VerifiedProject:
    project_id: str
    exact_versions: int
    hash_files: int
    dependency_edges: int
    required_projects: tuple[str, ...]
    freshness: float


def capability_queries(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return every distinct semantic capability represented by the request/design."""

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
            "minecraft",
            "mod",
            "mods",
            "make",
            "create",
            "fabric",
            "forge",
            "neoforge",
            "quilt",
            "version",
            "java",
            "with",
            "that",
            "this",
            "the",
            "and",
            "for",
            "please",
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
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result or ("minecraft gameplay",))


def optimize_platform_evidence(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    discovery_client: EcosystemDiscoveryClient | None = None,
    target_research_fn: TargetResearchFn | None = None,
) -> PlatformOptimization:
    queries = capability_queries(prompt, design=design, module_kinds=module_kinds)
    client = discovery_client or EcosystemDiscoveryClient()
    discovery_mode = __import__("os").environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
    if discovery_mode not in {"auto", "on", "off"}:
        raise SpecValidationError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")

    loaders = (
        (provider_for_loader(loader_constraint).loader,)
        if loader_constraint
        else executable_loaders()
    )
    if not loaders:
        raise SpecValidationError("No executable platform provider is registered.")

    requested_version = str(version_constraint or "").strip()
    if requested_version:
        frontier = tuple((loader, requested_version) for loader in loaders)
        shallow_by_query: dict[str, tuple[_ShallowProject, ...]] = {
            query: () for query in queries
        }
        shallow_count = 0
    elif discovery_mode == "off":
        stable_versions = _stable_game_versions()
        if not stable_versions:
            raise SpecValidationError("Official game-version discovery returned no stable release.")
        frontier = tuple((loader, stable_versions[0]) for loader in loaders)
        shallow_by_query = {query: () for query in queries}
        shallow_count = 0
    else:
        shallow_by_query = {
            query: _search_modrinth_exhaustive(client, query) for query in queries
        }
        shallow_count = sum(len(values) for values in shallow_by_query.values())
        frontier = _reuse_frontier(
            loaders,
            queries,
            shallow_by_query,
            stable_versions=_stable_game_versions(),
        )

    adapters, resolution_errors = _resolve_frontier(frontier)
    if not adapters:
        detail = "; ".join(resolution_errors) or "no frontier target produced a provider receipt"
        raise SpecValidationError(
            "No executable platform target survived provider resolution. Diagnostics: " + detail
        )

    evidence: list[TargetEvidence] = []
    failures: list[str] = []
    for adapter in adapters:
        try:
            evidence.append(
                _build_target_evidence(
                    adapter,
                    queries=queries,
                    shallow_by_query=shallow_by_query,
                    client=client,
                    target_research_fn=target_research_fn,
                    shallow_candidate_count=shallow_count,
                    discovery_disabled=(discovery_mode == "off"),
                )
            )
        except Exception as exc:
            failures.append(
                f"{adapter.minecraft_version}/{adapter.loader}: {type(exc).__name__}: {exc}"
            )

    if not evidence:
        detail = "; ".join(failures) or "no target produced complete evidence"
        raise SpecValidationError(
            "Platform evidence failed closed for every resolved target. Diagnostics: " + detail
        )

    ranked = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.rank_key,
                item.adapter.loader,
                item.adapter.minecraft_version,
                item.adapter.adapter_id,
            ),
            reverse=True,
        )
    )
    return PlatformOptimization(
        selected=ranked[0].adapter,
        evidence=ranked[0],
        candidates=ranked,
        capability_queries=queries,
        discovery_mode=(
            "provider-receipt-only" if discovery_mode == "off" else
            "exhaustive-neutral-reuse_frontier_single-resolution_receipt-native-deep"
        ),
    )


def _stable_game_versions() -> tuple[str, ...]:
    rows = discover_game_versions()
    versions = [
        str(row.get("version") or "").strip()
        for row in rows
        if isinstance(row, Mapping) and row.get("stable")
    ]
    return tuple(dict.fromkeys(value for value in versions if value))


def _search_modrinth_exhaustive(
    client: EcosystemDiscoveryClient,
    query: str,
) -> tuple[_ShallowProject, ...]:
    variants = _search_variants(query) or (query,)
    projects: dict[str, _ShallowProject] = {}
    for variant in variants:
        offset = 0
        while True:
            raw = client._get_json(
                "https://api.modrinth.com/v2/search",
                params={
                    "query": variant,
                    "facets": '[["project_type:mod"],["open_source:true"]]',
                    "index": "relevance",
                    "offset": str(offset),
                    "limit": str(_MAX_PAGE_ITEMS),
                },
            )
            if not isinstance(raw, Mapping) or not isinstance(raw.get("hits"), list):
                raise EcosystemDiscoveryUnavailable(
                    "Modrinth returned an invalid target-neutral search response."
                )
            hits = raw["hits"]
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                project_id = str(hit.get("project_id") or "").strip()
                if not project_id:
                    continue
                versions = frozenset(
                    str(value).strip()
                    for value in hit.get("versions", ())
                    if str(value).strip()
                )
                loaders = frozenset(
                    str(value).strip().casefold()
                    for value in hit.get("categories", ())
                    if str(value).strip()
                )
                projects.setdefault(
                    project_id,
                    _ShallowProject(
                        project_id=project_id,
                        versions=versions,
                        loaders=loaders,
                        downloads=_nonnegative_int(hit.get("downloads")),
                        modified=str(hit.get("date_modified") or ""),
                    ),
                )
            total = _nonnegative_int(raw.get("total_hits"))
            offset += len(hits)
            if not hits or offset >= total:
                break
    return tuple(projects.values())


def _reuse_frontier(
    loaders: Sequence[str],
    queries: Sequence[str],
    shallow_by_query: Mapping[str, Sequence[_ShallowProject]],
    *,
    stable_versions: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    if not stable_versions:
        raise SpecValidationError("Official stable Minecraft version catalogue is empty.")
    frontier: list[tuple[str, str]] = []
    for loader in loaders:
        best = -1
        for version in stable_versions:
            score = sum(
                any(
                    version in project.versions and loader in project.loaders
                    for project in shallow_by_query.get(query, ())
                )
                for query in queries
            )
            if score > best:
                frontier.append((loader, version))
                best = score
            if best == len(queries):
                break
    return tuple(frontier)


def _resolve_frontier(
    frontier: Sequence[tuple[str, str]],
) -> tuple[tuple[PlatformAdapter, ...], tuple[str, ...]]:
    adapters: list[PlatformAdapter] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for loader, version in frontier:
        key = (loader, version)
        if key in seen:
            continue
        seen.add(key)
        provider = provider_for_loader(loader)
        try:
            adapter = provider.resolve(version)
            adapter.validate()
        except Exception as exc:
            errors.append(f"{loader}/{version}: {type(exc).__name__}: {exc}")
            continue
        adapters.append(adapter)
    return tuple(adapters), tuple(errors)


def _build_target_evidence(
    adapter: PlatformAdapter,
    *,
    queries: Sequence[str],
    shallow_by_query: Mapping[str, Sequence[_ShallowProject]],
    client: EcosystemDiscoveryClient,
    target_research_fn: TargetResearchFn | None,
    shallow_candidate_count: int,
    discovery_disabled: bool,
) -> TargetEvidence:
    adapter.validate()
    research_payload: Mapping[str, Any] | None = None
    if target_research_fn is not None:
        research_payload = target_research_fn(adapter)
        if not isinstance(research_payload, Mapping):
            raise SpecValidationError("Target research returned a non-object receipt.")
        if research_payload.get("status") == "unavailable":
            raise SpecValidationError("Target research reported unavailable evidence.")

    verified_by_query: dict[str, _VerifiedProject] = {}
    dependency_projects: set[str] = set()
    dependency_edges = 0
    exact_versions = 0
    hash_files = 0
    maintenance = 0
    adoption = 0
    freshness = 0.0

    if not discovery_disabled:
        inspection_cache: dict[str, _VerifiedProject | None] = {}
        for query in queries:
            candidates = [
                project
                for project in shallow_by_query.get(query, ())
                if adapter.minecraft_version in project.versions
                and adapter.loader in project.loaders
            ]
            for candidate in candidates:
                if candidate.project_id not in inspection_cache:
                    inspection_cache[candidate.project_id] = _inspect_project_receipt_native(
                        client,
                        candidate.project_id,
                        adapter,
                    )
                verified = inspection_cache[candidate.project_id]
                if verified is None:
                    continue
                closure = _required_dependency_closure(
                    client,
                    adapter,
                    verified.required_projects,
                    inspection_cache=inspection_cache,
                )
                dependency_projects.update(closure)
                verified_by_query[query] = verified
                exact_versions += verified.exact_versions
                hash_files += verified.hash_files
                dependency_edges += verified.dependency_edges
                maintenance += 1
                adoption = max(adoption, candidate.downloads)
                freshness = max(freshness, verified.freshness, _timestamp(candidate.modified))
                break

    modes: list[tuple[str, str]] = []
    covered: list[str] = []
    for query in queries:
        if query in verified_by_query:
            mode = "reuse"
            covered.append(query)
        elif query in adapter.deterministic_module_kinds:
            mode = "direct"
            covered.append(query)
        else:
            mode = "custom"
        modes.append((query, mode))

    residual = sum(mode == "custom" for _query, mode in modes)
    research_quality = _research_quality(research_payload)
    evidence_quality = (
        len(verified_by_query) / len(queries) if queries else 1.0
    )
    return TargetEvidence(
        adapter=adapter,
        requested_capabilities=tuple(queries),
        covered_capabilities=tuple(covered),
        exact_projects=tuple(sorted({value.project_id for value in verified_by_query.values()})),
        exact_versions=exact_versions,
        verified_hash_files=hash_files,
        dependency_edges=dependency_edges,
        maintenance_signals=maintenance,
        adoption=adoption,
        freshness=freshness,
        evidence_quality=round(evidence_quality, 6),
        research_quality=round(research_quality, 6),
        integration_risk=float(dependency_edges),
        residual_cost=residual,
        dependency_complexity=dependency_edges + len(dependency_projects),
        discovery_errors=(),
        composition_modes=tuple(modes),
        deep_research=dict(research_payload) if isinstance(research_payload, Mapping) else None,
        shallow_candidate_count=shallow_candidate_count,
        dependency_projects=tuple(sorted(dependency_projects)),
        dependency_closure_complete=True,
    )


def _inspect_project_receipt_native(
    client: EcosystemDiscoveryClient,
    project_id: str,
    adapter: PlatformAdapter,
) -> _VerifiedProject | None:
    project_url = f"https://api.modrinth.com/v2/project/{project_id}"
    project = client._get_json(project_url)
    versions = client._get_json(
        project_url + "/version",
        params={
            "loaders": __import__("json").dumps([adapter.loader]),
            "game_versions": __import__("json").dumps([adapter.minecraft_version]),
            "include_changelog": "false",
        },
    )
    if not isinstance(project, Mapping) or not isinstance(versions, list):
        raise EcosystemDiscoveryUnavailable(
            f"Modrinth returned invalid project inspection for {project_id}."
        )
    license_value = project.get("license")
    license_id = (
        str(license_value.get("id") or "").strip()
        if isinstance(license_value, Mapping)
        else str(license_value or "").strip()
    )
    if not _code_license_policy(license_id).startswith("permissive_candidate"):
        return None

    normalized = [
        _normalize_modrinth_version(
            dict(version),
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
        )
        for version in versions
        if isinstance(version, Mapping)
    ]
    eligible = [value for value in normalized if value.get("eligible_for_selection")]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda value: (
            str(value.get("date_published") or ""),
            str(value.get("version_id") or ""),
        ),
    )
    files = selected.get("files") if isinstance(selected.get("files"), list) else []
    dependencies = (
        selected.get("dependencies")
        if isinstance(selected.get("dependencies"), list)
        else []
    )
    required: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            continue
        if str(dependency.get("dependency_type") or "").casefold() != "required":
            continue
        dependency_project = str(dependency.get("project_id") or "").strip()
        if not dependency_project:
            dependency_version = str(dependency.get("version_id") or "").strip()
            if dependency_version:
                dependency_project = _project_for_version(client, dependency_version)
        if not dependency_project:
            raise SpecValidationError(
                f"Required dependency of {project_id} has no resolvable project identity."
            )
        required.append(dependency_project)
    return _VerifiedProject(
        project_id=project_id,
        exact_versions=1,
        hash_files=sum(bool(str(file.get("sha512") or "")) for file in files if isinstance(file, Mapping)),
        dependency_edges=len(dependencies),
        required_projects=tuple(dict.fromkeys(required)),
        freshness=_timestamp(str(selected.get("date_published") or "")),
    )


def _project_for_version(client: EcosystemDiscoveryClient, version_id: str) -> str:
    raw = client._get_json(f"https://api.modrinth.com/v2/version/{version_id}")
    if not isinstance(raw, Mapping):
        raise EcosystemDiscoveryUnavailable(
            f"Modrinth returned invalid dependency version metadata for {version_id}."
        )
    return str(raw.get("project_id") or "").strip()


def _required_dependency_closure(
    client: EcosystemDiscoveryClient,
    adapter: PlatformAdapter,
    seeds: Sequence[str],
    *,
    inspection_cache: dict[str, _VerifiedProject | None],
) -> set[str]:
    pending = list(dict.fromkeys(str(value).strip() for value in seeds if str(value).strip()))
    seen: set[str] = set()
    while pending:
        project_id = pending.pop(0)
        if project_id in seen:
            continue
        seen.add(project_id)
        if project_id not in inspection_cache:
            inspection_cache[project_id] = _inspect_project_receipt_native(
                client,
                project_id,
                adapter,
            )
        verified = inspection_cache[project_id]
        if verified is None:
            raise SpecValidationError(
                f"Required dependency {project_id} failed exact-target/license/digest gates."
            )
        for dependency in verified.required_projects:
            if dependency not in seen:
                pending.append(dependency)
    return seen


def _research_quality(payload: Mapping[str, Any] | None) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        unresolved = payload.get("unresolved_official_domains")
        return 0.0 if isinstance(unresolved, list) and unresolved else 1.0
    resolved = 0
    for domain in domains:
        if not isinstance(domain, Mapping):
            continue
        fusion = domain.get("fusion")
        critic = fusion.get("critic") if isinstance(fusion, Mapping) else None
        if isinstance(critic, Mapping) and float(critic.get("mean_coverage", 0.0) or 0.0) > 0:
            resolved += 1
    return resolved / len(domains)


def _search_variants(query: str) -> tuple[str, ...]:
    original = " ".join(str(query or "").split()).strip()
    if not original:
        return ()
    expanded = _CAMEL_BOUNDARY.sub(" ", original)
    expanded = re.sub(r"[_/:\-]+", " ", expanded)
    useful = [
        token
        for token in _TOKEN_RE.findall(expanded)
        if token.casefold() not in _GENERIC_QUERY_TOKENS
    ]
    values = [original, " ".join(useful).strip()]
    return tuple(dict.fromkeys(value for value in values if value))


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip()


def _timestamp(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


def _nonnegative_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


__all__ = [
    "PlatformOptimization",
    "TargetEvidence",
    "TargetResearchFn",
    "capability_queries",
    "optimize_platform_evidence",
]
