from __future__ import annotations

"""Evidence-first, loader-neutral Minecraft target optimiser.

The host owns compatibility and final target choice. Discovery first searches
capabilities without version/loader facets, then deep-checks only the best executable
target hypotheses. Correctness/reuse outrank popularity and freshness.
"""

import json
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .platform_catalog import PlatformAdapter, adapter_for_target, discover_target_keys


_MODRINTH_API = "https://api.modrinth.com/v2"
_LOADER_NAMES = frozenset({"fabric", "neoforge", "forge", "quilt"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.-]{2,}")


@dataclass(frozen=True)
class EcosystemProject:
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

    @property
    def mandatory_coverage(self) -> int:
        return int(
            not self.requested_capabilities
            or len(self.covered_capabilities) == len(self.requested_capabilities)
        )

    @property
    def reuse_coverage(self) -> int:
        return len(self.covered_capabilities)

    @property
    def rank_key(self) -> tuple[float | int, ...]:
        return (
            self.mandatory_coverage,
            self.reuse_coverage,
            self.evidence_quality,
            -self.integration_risk,
            -self.residual_cost,
            -self.dependency_complexity,
            self.maintenance_signals,
            self.adoption,
            self.freshness,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": {
                "minecraft_version": self.adapter.minecraft_version,
                "loader": self.adapter.loader,
                "adapter_id": self.adapter.adapter_id,
            },
            "requested_capabilities": list(self.requested_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "exact_projects": list(self.exact_projects),
            "exact_versions": self.exact_versions,
            "verified_hash_files": self.verified_hash_files,
            "dependency_edges": self.dependency_edges,
            "maintenance_signals": self.maintenance_signals,
            "adoption": self.adoption,
            "freshness": self.freshness,
            "evidence_quality": self.evidence_quality,
            "integration_risk": self.integration_risk,
            "residual_cost": self.residual_cost,
            "dependency_complexity": self.dependency_complexity,
            "mandatory_coverage": self.mandatory_coverage,
            "reuse_coverage": self.reuse_coverage,
            "rank_key": list(self.rank_key),
            "discovery_errors": list(self.discovery_errors),
        }


@dataclass(frozen=True)
class PlatformOptimization:
    selected: PlatformAdapter
    evidence: TargetEvidence
    candidates: tuple[TargetEvidence, ...]
    capability_queries: tuple[str, ...]
    discovery_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/platform-optimizer-v1",
            "discovery_mode": self.discovery_mode,
            "capability_queries": list(self.capability_queries),
            "selected": self.evidence.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
        }


SearchFn = Callable[[str], Sequence[EcosystemProject]]
VersionFn = Callable[[str], Sequence[Mapping[str, Any]]]


def capability_queries(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
) -> tuple[str, ...]:
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
                    value = _clean(str(item.get("name") or item.get("id") or item.get("kind") or ""))
                else:
                    value = ""
                if value:
                    labels.append(value)
    if not labels:
        stop = {
            "minecraft", "mod", "mods", "make", "create", "fabric", "forge",
            "neoforge", "version", "java", "with", "that", "this", "the", "and",
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
    search_fn: SearchFn | None = None,
    version_fn: VersionFn | None = None,
) -> PlatformOptimization:
    queries = capability_queries(prompt, design=design, module_kinds=module_kinds)
    target_keys = discover_target_keys(
        loader=loader_constraint,
        minecraft_version=version_constraint,
        limit_per_loader=12,
    )
    if not target_keys:
        target = "/".join(value for value in (version_constraint, loader_constraint) if value) or "automatic"
        raise ValueError(f"No executable platform provider can satisfy target {target!r}.")

    projects_by_query, shallow_errors = _parallel_search(
        queries,
        search_fn or _search_modrinth_unscoped,
    )
    hypotheses: list[tuple[PlatformAdapter, int]] = []
    for loader, version in target_keys:
        try:
            adapter = adapter_for_target(version, loader)
        except ValueError:
            continue
        coverage = sum(
            _query_supported(query, adapter, projects_by_query.get(query, ()))
            for query in queries
        )
        hypotheses.append((adapter, coverage))
    if not hypotheses:
        raise ValueError("Executable platform targets were discovered but none resolved.")

    hypotheses.sort(
        key=lambda item: (-item[1], item[0].loader, item[0].minecraft_version)
    )
    shortlist = tuple(
        adapter for adapter, _ in hypotheses[: max(1, min(int(top_k), 8))]
    )
    deep = _parallel_deep(
        shortlist,
        queries=queries,
        projects_by_query=projects_by_query,
        version_fn=version_fn or _fetch_modrinth_versions,
        inherited_errors=shallow_errors,
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
        discovery_mode="neutral-shallow_then_exact-target-deep",
    )


def _parallel_search(
    queries: Sequence[str],
    search_fn: SearchFn,
) -> tuple[dict[str, Sequence[EcosystemProject]], tuple[str, ...]]:
    found: dict[str, Sequence[EcosystemProject]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(
        max_workers=min(8, max(1, len(queries))),
        thread_name_prefix="mmm-platform-shallow",
    ) as pool:
        futures = {pool.submit(search_fn, query): query for query in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                found[query] = tuple(future.result())
            except Exception as exc:
                found[query] = ()
                errors.append(f"{query}: {type(exc).__name__}: {exc}")
    return found, tuple(sorted(errors))


def _parallel_deep(
    adapters: Sequence[PlatformAdapter],
    *,
    queries: Sequence[str],
    projects_by_query: Mapping[str, Sequence[EcosystemProject]],
    version_fn: VersionFn,
    inherited_errors: Sequence[str],
) -> tuple[TargetEvidence, ...]:
    with ThreadPoolExecutor(
        max_workers=min(6, max(1, len(adapters))),
        thread_name_prefix="mmm-platform-deep",
    ) as pool:
        futures = {
            pool.submit(
                _deep_evidence,
                adapter,
                queries=queries,
                projects_by_query=projects_by_query,
                version_fn=version_fn,
                inherited_errors=inherited_errors,
            ): adapter
            for adapter in adapters
        }
        result: list[TargetEvidence] = []
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
    projects_by_query: Mapping[str, Sequence[EcosystemProject]],
    version_fn: VersionFn,
    inherited_errors: Sequence[str],
) -> TargetEvidence:
    candidate_projects: dict[str, EcosystemProject] = {}
    covered: set[str] = set()
    for query in queries:
        exact = [
            project
            for project in projects_by_query.get(query, ())
            if adapter.loader in project.loaders and adapter.minecraft_version in project.versions
        ]
        if exact or query in adapter.deterministic_module_kinds:
            covered.add(query)
        for project in exact[:4]:
            candidate_projects[project.project_id] = project

    version_payloads: dict[str, Sequence[Mapping[str, Any]]] = {}
    errors = list(inherited_errors)
    project_ids = tuple(sorted(candidate_projects))[:16]
    if project_ids:
        with ThreadPoolExecutor(
            max_workers=min(8, len(project_ids)),
            thread_name_prefix="mmm-platform-project",
        ) as pool:
            futures = {pool.submit(version_fn, project_id): project_id for project_id in project_ids}
            for future in as_completed(futures):
                project_id = futures[future]
                try:
                    version_payloads[project_id] = tuple(future.result())
                except Exception as exc:
                    errors.append(f"{project_id}: {type(exc).__name__}: {exc}")

    exact_versions = 0
    hash_files = 0
    dependency_edges = 0
    verified_projects: list[str] = []
    for project_id, payloads in version_payloads.items():
        project_verified = False
        for raw in payloads:
            loaders = {str(value).casefold() for value in raw.get("loaders", ())}
            game_versions = {str(value) for value in raw.get("game_versions", ())}
            if adapter.loader not in loaders or adapter.minecraft_version not in game_versions:
                continue
            exact_versions += 1
            dependencies = raw.get("dependencies")
            if isinstance(dependencies, list):
                dependency_edges += len(dependencies)
            files = raw.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, Mapping):
                        hashes = item.get("hashes")
                        if isinstance(hashes, Mapping) and hashes.get("sha512"):
                            hash_files += 1
            project_verified = True
        if project_verified:
            verified_projects.append(project_id)

    residual = max(0, len(queries) - len(covered))
    project_count = len(candidate_projects)
    exact_ratio = exact_versions / max(1, project_count)
    hash_ratio = hash_files / max(1, exact_versions)
    quality = min(1.0, 0.45 + 0.30 * min(1.0, exact_ratio) + 0.25 * min(1.0, hash_ratio))
    if not candidate_projects:
        quality = 0.35 if covered else 0.0
    adoption = sum(project.downloads for project in candidate_projects.values())
    freshness = max((_timestamp(project.modified) for project in candidate_projects.values()), default=0.0)
    maintenance = sum(bool(project.modified) for project in candidate_projects.values())
    complexity = dependency_edges + max(0, project_count - len(verified_projects))
    risk = float(residual) + math.log2(1 + complexity) * 0.15
    return TargetEvidence(
        adapter=adapter,
        requested_capabilities=tuple(queries),
        covered_capabilities=tuple(sorted(covered)),
        exact_projects=tuple(sorted(verified_projects)),
        exact_versions=exact_versions,
        verified_hash_files=hash_files,
        dependency_edges=dependency_edges,
        maintenance_signals=maintenance,
        adoption=adoption,
        freshness=freshness,
        evidence_quality=round(quality, 6),
        integration_risk=round(risk, 6),
        residual_cost=residual,
        dependency_complexity=complexity,
        discovery_errors=tuple(sorted(set(errors))),
    )


def _query_supported(
    query: str,
    adapter: PlatformAdapter,
    projects: Sequence[EcosystemProject],
) -> int:
    if query in adapter.deterministic_module_kinds:
        return 1
    return int(
        any(
            adapter.loader in project.loaders
            and adapter.minecraft_version in project.versions
            for project in projects
        )
    )


def _search_modrinth_unscoped(query: str) -> tuple[EcosystemProject, ...]:
    payload = _json_get(
        "/search?" + urllib.parse.urlencode({"query": query, "limit": 20, "index": "relevance"})
    )
    hits = payload.get("hits") if isinstance(payload, Mapping) else None
    if not isinstance(hits, list):
        return ()
    result: list[EcosystemProject] = []
    for raw in hits:
        if not isinstance(raw, Mapping):
            continue
        project_id = str(raw.get("project_id") or "").strip()
        loaders = frozenset(
            str(value).casefold()
            for value in raw.get("categories", ())
            if str(value).casefold() in _LOADER_NAMES
        )
        versions = frozenset(str(value) for value in raw.get("versions", ()))
        if project_id and loaders and versions:
            result.append(
                EcosystemProject(
                    project_id=project_id,
                    title=str(raw.get("title") or project_id),
                    loaders=loaders,
                    versions=versions,
                    downloads=int(raw.get("downloads") or 0),
                    modified=str(raw.get("date_modified") or ""),
                )
            )
    return tuple(result)


def _fetch_modrinth_versions(project_id: str) -> tuple[Mapping[str, Any], ...]:
    payload = _json_get(f"/project/{urllib.parse.quote(project_id, safe='')}/version")
    if not isinstance(payload, list):
        return ()
    return tuple(item for item in payload if isinstance(item, Mapping))


def _json_get(path: str) -> Any:
    request = urllib.request.Request(
        _MODRINTH_API + path,
        headers={
            "Accept": "application/json",
            "User-Agent": "M.M.M-Make-Mincraft-Mode/platform-optimizer",
        },
    )
    with urllib.request.urlopen(request, timeout=8.0) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().split())[:120]


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
