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

Platform-evidence tracing is intentionally always on. Every material decision records
the target, project, parent/dependency path, gate, pass/fail result and a bounded,
non-secret input summary so a later failure can be traced to its first rejected gate.
"""

import itertools
import json
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
_TRACE_SEQUENCE = itertools.count(1)
_TRACE_PREFIX = "PLATFORM EVIDENCE TRACE: "
_TRACE_STRING_LIMIT = 512
_TRACE_COLLECTION_LIMIT = 64


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


@dataclass(frozen=True)
class _ProjectInspection:
    project_id: str
    role: str
    verified: _VerifiedProject | None
    failed_gate: str = ""
    failure_reason: str = ""
    license_id: str = ""
    license_policy: str = ""


def _bounded_trace_value(value: Any, *, depth: int = 0) -> Any:
    """Return a deterministic, bounded and non-secret trace representation."""
    if depth >= 4:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _TRACE_STRING_LIMIT else value[:_TRACE_STRING_LIMIT] + "…"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _TRACE_COLLECTION_LIMIT:
                result["<truncated>"] = len(value) - _TRACE_COLLECTION_LIMIT
                break
            normalized_key = str(key)
            if any(secret in normalized_key.casefold() for secret in ("token", "secret", "password", "authorization", "cookie")):
                result[normalized_key] = "<redacted>"
            else:
                result[normalized_key] = _bounded_trace_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        bounded = [
            _bounded_trace_value(item, depth=depth + 1)
            for item in items[:_TRACE_COLLECTION_LIMIT]
        ]
        if len(items) > _TRACE_COLLECTION_LIMIT:
            bounded.append(f"<truncated:{len(items) - _TRACE_COLLECTION_LIMIT}>")
        return bounded
    return _bounded_trace_value(str(value), depth=depth + 1)


def _emit_platform_trace(
    event: str,
    *,
    adapter: PlatformAdapter | None = None,
    project_id: str = "",
    parent_project_id: str = "",
    dependency_path: Sequence[str] = (),
    role: str = "",
    gate: str = "",
    passed: bool | None = None,
    reason: str = "",
    details: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "mmm/platform-evidence-trace-v1",
        "trace_seq": next(_TRACE_SEQUENCE),
        "event": event,
    }
    if adapter is not None:
        payload["target"] = {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "adapter_id": adapter.adapter_id,
        }
    if project_id:
        payload["project_id"] = project_id
    if parent_project_id:
        payload["parent_project_id"] = parent_project_id
    if dependency_path:
        payload["dependency_path"] = list(dependency_path)
    if role:
        payload["role"] = role
    if gate:
        payload["gate"] = gate
    if passed is not None:
        payload["passed"] = passed
    if reason:
        payload["reason"] = reason
    if details:
        payload["details"] = _bounded_trace_value(details)
    print(
        _TRACE_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current)[:_TRACE_STRING_LIMIT],
            }
        )
        next_exc = current.__cause__ if current.__cause__ is not None else current.__context__
        current = next_exc
    return chain


def _dependency_license_policy(license_id: str) -> tuple[bool, str]:
    """Policy for resolving an external dependency without copying its source/assets.

    A declared non-permissive license is not treated as permission to vendor, shade,
    copy or redistribute that project. It is, however, distinct from source reuse:
    dependency metadata can be resolved and tracked so license obligations remain
    explicit. Missing/ARR/NOASSERTION remains fail-closed.
    """
    normalized = str(license_id or "").strip()
    if not normalized or normalized.upper() in {"ARR", "NOASSERTION"}:
        return False, "dependency_license_missing_or_unreviewable"
    source_policy = _code_license_policy(normalized)
    if source_policy.startswith("permissive_candidate"):
        return True, "declared_permissive_dependency_license"
    return (
        True,
        "declared_dependency_license_reference_only; "
        "source_copy_asset_reuse_vendoring_shading_or_redistribution_requires_separate_review",
    )


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
    _emit_platform_trace(
        "optimizer_start",
        gate="configuration",
        passed=discovery_mode in {"auto", "on", "off"},
        reason="platform evidence optimizer invoked",
        details={
            "discovery_mode": discovery_mode,
            "loader_constraint": loader_constraint or "",
            "version_constraint": version_constraint or "",
            "capability_count": len(queries),
            "capabilities": queries,
        },
    )
    if discovery_mode not in {"auto", "on", "off"}:
        raise SpecValidationError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")

    loaders = (
        (provider_for_loader(loader_constraint).loader,)
        if loader_constraint
        else executable_loaders()
    )
    _emit_platform_trace(
        "loader_gate",
        gate="executable_provider",
        passed=bool(loaders),
        reason="resolved executable loader set" if loaders else "no executable platform provider is registered",
        details={"loaders": loaders},
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
        _emit_platform_trace(
            "stable_version_gate",
            gate="official_game_versions",
            passed=bool(stable_versions),
            reason="stable versions discovered" if stable_versions else "no stable release returned",
            details={"stable_versions": stable_versions},
        )
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

    _emit_platform_trace(
        "frontier_built",
        gate="reuse_frontier",
        passed=bool(frontier),
        reason="frontier candidates built" if frontier else "reuse frontier is empty",
        details={"frontier": frontier, "shallow_candidate_count": shallow_count},
    )
    adapters, resolution_errors = _resolve_frontier(frontier)
    _emit_platform_trace(
        "frontier_resolved",
        gate="provider_resolution",
        passed=bool(adapters),
        reason="provider receipts resolved" if adapters else "all provider resolutions failed",
        details={
            "resolved_targets": [
                {
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "adapter_id": adapter.adapter_id,
                }
                for adapter in adapters
            ],
            "resolution_errors": resolution_errors,
        },
    )
    if not adapters:
        detail = "; ".join(resolution_errors) or "no frontier target produced a provider receipt"
        raise SpecValidationError(
            "No executable platform target survived provider resolution. Diagnostics: " + detail
        )

    evidence: list[TargetEvidence] = []
    failures: list[str] = []
    for adapter in adapters:
        _emit_platform_trace(
            "target_evidence_start",
            adapter=adapter,
            gate="target_evidence",
            passed=None,
            reason="building target evidence",
        )
        try:
            target_evidence = _build_target_evidence(
                adapter,
                queries=queries,
                shallow_by_query=shallow_by_query,
                client=client,
                target_research_fn=target_research_fn,
                shallow_candidate_count=shallow_count,
                discovery_disabled=(discovery_mode == "off"),
            )
            evidence.append(target_evidence)
            _emit_platform_trace(
                "target_evidence_success",
                adapter=adapter,
                gate="target_evidence",
                passed=True,
                reason="target evidence complete",
                details={
                    "reuse_coverage": target_evidence.reuse_coverage,
                    "residual_cost": target_evidence.residual_cost,
                    "dependency_projects": target_evidence.dependency_projects,
                },
            )
        except Exception as exc:
            failure = f"{adapter.minecraft_version}/{adapter.loader}: {type(exc).__name__}: {exc}"
            failures.append(failure)
            _emit_platform_trace(
                "target_evidence_failure",
                adapter=adapter,
                gate="target_evidence",
                passed=False,
                reason=str(exc),
                details={"exception_chain": _exception_chain(exc)},
            )

    if not evidence:
        detail = "; ".join(failures) or "no target produced complete evidence"
        _emit_platform_trace(
            "optimizer_failure",
            gate="target_evidence",
            passed=False,
            reason="every resolved target failed",
            details={"failures": failures},
        )
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
    _emit_platform_trace(
        "optimizer_selected",
        adapter=ranked[0].adapter,
        gate="ranking",
        passed=True,
        reason="highest verified lexicographic rank selected",
        details={
            "rank_key": ranked[0].rank_key,
            "candidate_count": len(ranked),
            "reuse_coverage": ranked[0].reuse_coverage,
        },
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
        page_index = 0
        while True:
            page_index += 1
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
            valid_page = isinstance(raw, Mapping) and isinstance(raw.get("hits"), list)
            _emit_platform_trace(
                "discovery_page",
                gate="modrinth_search_response",
                passed=valid_page,
                reason="valid Modrinth search page" if valid_page else "invalid Modrinth search response",
                details={
                    "query": query,
                    "variant": variant,
                    "page_index": page_index,
                    "offset": offset,
                    "response_type": type(raw).__name__,
                    "has_hits_list": isinstance(raw, Mapping) and isinstance(raw.get("hits"), list),
                },
            )
            if not valid_page:
                raise EcosystemDiscoveryUnavailable(
                    "Modrinth returned an invalid target-neutral search response."
                )
            hits = raw["hits"]
            accepted_on_page = 0
            for hit in hits:
                if not isinstance(hit, Mapping):
                    _emit_platform_trace(
                        "discovery_hit_rejected",
                        gate="shallow_project_metadata",
                        passed=False,
                        reason="search hit is not an object",
                        details={"query": query, "variant": variant},
                    )
                    continue
                project_id = str(hit.get("project_id") or "").strip()
                if not project_id:
                    _emit_platform_trace(
                        "discovery_hit_rejected",
                        gate="shallow_project_identity",
                        passed=False,
                        reason="search hit has no project_id",
                        details={"query": query, "variant": variant},
                    )
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
                if project_id not in projects:
                    projects[project_id] = _ShallowProject(
                        project_id=project_id,
                        versions=versions,
                        loaders=loaders,
                        downloads=_nonnegative_int(hit.get("downloads")),
                        modified=str(hit.get("date_modified") or ""),
                    )
                    accepted_on_page += 1
            total = _nonnegative_int(raw.get("total_hits"))
            _emit_platform_trace(
                "discovery_page_complete",
                gate="shallow_project_metadata",
                passed=True,
                reason="search page normalized",
                details={
                    "query": query,
                    "variant": variant,
                    "page_index": page_index,
                    "hits": len(hits),
                    "new_projects": accepted_on_page,
                    "provider_total": total,
                },
            )
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
            improved = score > best
            _emit_platform_trace(
                "frontier_score",
                gate="reuse_frontier_score",
                passed=True,
                reason="evaluated target reuse coverage",
                details={
                    "loader": loader,
                    "minecraft_version": version,
                    "score": score,
                    "previous_best": best,
                    "frontier_added": improved,
                },
            )
            if improved:
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
            _emit_platform_trace(
                "provider_resolution_skipped",
                gate="provider_resolution",
                passed=True,
                reason="duplicate frontier target",
                details={"loader": loader, "minecraft_version": version},
            )
            continue
        seen.add(key)
        try:
            provider = provider_for_loader(loader)
            adapter = provider.resolve(version)
            adapter.validate()
        except Exception as exc:
            error = f"{loader}/{version}: {type(exc).__name__}: {exc}"
            errors.append(error)
            _emit_platform_trace(
                "provider_resolution_failure",
                gate="provider_resolution",
                passed=False,
                reason=str(exc),
                details={
                    "loader": loader,
                    "minecraft_version": version,
                    "exception_chain": _exception_chain(exc),
                },
            )
            continue
        adapters.append(adapter)
        _emit_platform_trace(
            "provider_resolution_success",
            adapter=adapter,
            gate="provider_resolution",
            passed=True,
            reason="provider receipt resolved and validated",
        )
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
    _emit_platform_trace(
        "adapter_gate",
        adapter=adapter,
        gate="adapter_validation",
        passed=True,
        reason="platform adapter validated",
    )
    research_payload: Mapping[str, Any] | None = None
    if target_research_fn is not None:
        try:
            research_payload = target_research_fn(adapter)
        except Exception as exc:
            _emit_platform_trace(
                "target_research_failure",
                adapter=adapter,
                gate="target_research",
                passed=False,
                reason=str(exc),
                details={"exception_chain": _exception_chain(exc)},
            )
            raise
        if not isinstance(research_payload, Mapping):
            _emit_platform_trace(
                "target_research_failure",
                adapter=adapter,
                gate="target_research_shape",
                passed=False,
                reason="target research returned a non-object receipt",
                details={"response_type": type(research_payload).__name__},
            )
            raise SpecValidationError("Target research returned a non-object receipt.")
        if research_payload.get("status") == "unavailable":
            _emit_platform_trace(
                "target_research_failure",
                adapter=adapter,
                gate="target_research_availability",
                passed=False,
                reason="target research reported unavailable evidence",
                details={"status": research_payload.get("status")},
            )
            raise SpecValidationError("Target research reported unavailable evidence.")
        _emit_platform_trace(
            "target_research_success",
            adapter=adapter,
            gate="target_research",
            passed=True,
            reason="target research receipt accepted",
            details={
                "status": research_payload.get("status", ""),
                "research_quality": _research_quality(research_payload),
            },
        )

    verified_by_query: dict[str, _VerifiedProject] = {}
    dependency_projects: set[str] = set()
    dependency_edges = 0
    exact_versions = 0
    hash_files = 0
    maintenance = 0
    adoption = 0
    freshness = 0.0
    discovery_errors: list[str] = []

    if not discovery_disabled:
        inspection_cache: dict[str, _VerifiedProject | None] = {}
        for query in queries:
            candidates = [
                project
                for project in shallow_by_query.get(query, ())
                if adapter.minecraft_version in project.versions
                and adapter.loader in project.loaders
            ]
            _emit_platform_trace(
                "capability_candidate_set",
                adapter=adapter,
                gate="shallow_exact_target_filter",
                passed=True,
                reason="candidate set filtered for exact target",
                details={
                    "query": query,
                    "candidate_count": len(candidates),
                    "candidate_ids": [candidate.project_id for candidate in candidates],
                },
            )
            for candidate in candidates:
                inspection = _inspect_project_receipt_native_detailed(
                    client,
                    candidate.project_id,
                    adapter,
                    role="source_reuse",
                )
                inspection_cache[candidate.project_id] = inspection.verified
                verified = inspection.verified
                if verified is None:
                    reason = (
                        f"{candidate.project_id}: gate={inspection.failed_gate}; "
                        f"reason={inspection.failure_reason}"
                    )
                    discovery_errors.append(reason)
                    _emit_platform_trace(
                        "reuse_candidate_rejected",
                        adapter=adapter,
                        project_id=candidate.project_id,
                        role="source_reuse",
                        gate=inspection.failed_gate or "project_inspection",
                        passed=False,
                        reason=inspection.failure_reason or "project inspection rejected",
                        details={
                            "query": query,
                            "license_id": inspection.license_id,
                            "license_policy": inspection.license_policy,
                        },
                    )
                    continue
                try:
                    closure = _required_dependency_closure(
                        client,
                        adapter,
                        verified.required_projects,
                        inspection_cache=inspection_cache,
                        root_project_id=verified.project_id,
                    )
                except SpecValidationError as exc:
                    discovery_errors.append(
                        f"{candidate.project_id}: dependency_closure: {exc}"
                    )
                    _emit_platform_trace(
                        "reuse_candidate_rejected",
                        adapter=adapter,
                        project_id=candidate.project_id,
                        role="source_reuse",
                        gate="dependency_closure",
                        passed=False,
                        reason=str(exc),
                        details={"query": query, "exception_chain": _exception_chain(exc)},
                    )
                    # Reuse evidence is optional. A bad third-party candidate must not
                    # invalidate a platform that can still implement this capability
                    # as custom source.
                    continue
                dependency_projects.update(closure)
                verified_by_query[query] = verified
                exact_versions += verified.exact_versions
                hash_files += verified.hash_files
                dependency_edges += verified.dependency_edges
                maintenance += 1
                adoption = max(adoption, candidate.downloads)
                freshness = max(freshness, verified.freshness, _timestamp(candidate.modified))
                _emit_platform_trace(
                    "reuse_candidate_accepted",
                    adapter=adapter,
                    project_id=candidate.project_id,
                    role="source_reuse",
                    gate="dependency_closure",
                    passed=True,
                    reason="source reuse candidate and required dependency closure verified",
                    details={"query": query, "dependency_projects": sorted(closure)},
                )
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
        _emit_platform_trace(
            "capability_composition",
            adapter=adapter,
            gate="composition_mode",
            passed=True,
            reason=f"capability assigned {mode} mode",
            details={"query": query, "mode": mode},
        )

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
        discovery_errors=tuple(discovery_errors),
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
    """Compatibility wrapper retaining the previous private-call contract."""
    return _inspect_project_receipt_native_detailed(
        client,
        project_id,
        adapter,
        role="source_reuse",
    ).verified


def _inspect_project_receipt_native_detailed(
    client: EcosystemDiscoveryClient,
    project_id: str,
    adapter: PlatformAdapter,
    *,
    role: str,
    parent_project_id: str = "",
    dependency_path: Sequence[str] = (),
) -> _ProjectInspection:
    if role not in {"source_reuse", "required_dependency"}:
        raise ValueError(f"Unsupported project inspection role: {role}")

    project_url = f"https://api.modrinth.com/v2/project/{project_id}"
    _emit_platform_trace(
        "project_inspection_start",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="provider_metadata",
        passed=None,
        reason="requesting project and exact-target version metadata",
    )
    try:
        project = client._get_json(project_url)
        versions = client._get_json(
            project_url + "/version",
            params={
                "loaders": json.dumps([adapter.loader]),
                "game_versions": json.dumps([adapter.minecraft_version]),
                "include_changelog": "false",
            },
        )
    except Exception as exc:
        _emit_platform_trace(
            "project_inspection_failure",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate="provider_metadata",
            passed=False,
            reason=str(exc),
            details={"exception_chain": _exception_chain(exc)},
        )
        raise

    provider_shape_ok = isinstance(project, Mapping) and isinstance(versions, list)
    _emit_platform_trace(
        "project_metadata_gate",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="provider_metadata_shape",
        passed=provider_shape_ok,
        reason="project and version responses have expected shapes" if provider_shape_ok else "invalid project/version response shape",
        details={
            "project_response_type": type(project).__name__,
            "versions_response_type": type(versions).__name__,
            "version_record_count": len(versions) if isinstance(versions, list) else -1,
        },
    )
    if not provider_shape_ok:
        raise EcosystemDiscoveryUnavailable(
            f"Modrinth returned invalid project inspection for {project_id}."
        )

    license_value = project.get("license")
    license_id = (
        str(license_value.get("id") or "").strip()
        if isinstance(license_value, Mapping)
        else str(license_value or "").strip()
    )
    source_license_policy = _code_license_policy(license_id)
    if role == "source_reuse":
        license_passed = source_license_policy.startswith("permissive_candidate")
        license_policy = source_license_policy
        license_reason = (
            "source reuse license is permissive"
            if license_passed
            else "source reuse requires a permissive license; dependency-reference permission is a separate policy"
        )
    else:
        license_passed, dependency_policy = _dependency_license_policy(license_id)
        license_policy = dependency_policy
        license_reason = (
            "declared license accepted for dependency-reference inspection"
            if license_passed
            else "required dependency has no reviewable declared license"
        )
    _emit_platform_trace(
        "project_license_gate",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="source_reuse_license" if role == "source_reuse" else "dependency_license",
        passed=license_passed,
        reason=license_reason,
        details={
            "license_id": license_id,
            "license_policy": license_policy,
            "source_reuse_policy": source_license_policy,
        },
    )

    normalized = [
        _normalize_modrinth_version(
            dict(version),
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
        )
        for version in versions
        if isinstance(version, Mapping)
    ]
    malformed_version_records = len(versions) - len(normalized)
    if malformed_version_records:
        _emit_platform_trace(
            "version_record_gate",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate="version_record_shape",
            passed=False,
            reason="one or more version records were not objects",
            details={"malformed_version_records": malformed_version_records},
        )

    for value in normalized:
        files = value.get("files") if isinstance(value.get("files"), list) else []
        primary_files = [
            file for file in files
            if isinstance(file, Mapping) and file.get("primary") is True
        ]
        _emit_platform_trace(
            "version_candidate_gate",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate="exact_target_and_artifact",
            passed=bool(value.get("eligible_for_selection")),
            reason=(
                "version satisfies exact target and artifact gates"
                if value.get("eligible_for_selection")
                else "version rejected by one or more normalized gates"
            ),
            details={
                "version_id": value.get("version_id", ""),
                "version_number": value.get("version_number", ""),
                "date_published": value.get("date_published", ""),
                "game_versions": value.get("game_versions", []),
                "loaders": value.get("loaders", []),
                "rejection_reasons": value.get("unresolved_or_rejected_gates", []),
                "file_count": len(files),
                "primary_file_count": len(primary_files),
                "primary_files": [
                    {
                        "filename": file.get("filename", ""),
                        "size": file.get("size", 0),
                        "safe_filename": bool(file.get("safe_filename")),
                        "safe_origin": bool(file.get("safe_origin")),
                        "sha512_valid": bool(file.get("strong_digest_valid")),
                    }
                    for file in primary_files
                    if isinstance(file, Mapping)
                ],
            },
        )

    eligible = [value for value in normalized if value.get("eligible_for_selection")]
    exact_target_candidates = [
        value
        for value in normalized
        if "off_target_minecraft_version_or_loader"
        not in set(value.get("unresolved_or_rejected_gates", ()))
    ]
    _emit_platform_trace(
        "exact_target_gate",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="exact_target",
        passed=bool(exact_target_candidates),
        reason=(
            "at least one returned version matches the selected Minecraft version and loader"
            if exact_target_candidates
            else "no returned version matches the selected Minecraft version and loader"
        ),
        details={
            "returned_version_count": len(normalized),
            "exact_target_candidate_count": len(exact_target_candidates),
        },
    )

    artifact_candidates = [
        value
        for value in exact_target_candidates
        if not any(
            reason in {
                "exactly_one_primary_file_is_required",
                "primary_file_requires_safe_origin_size_and_sha512",
            }
            for reason in value.get("unresolved_or_rejected_gates", ())
        )
    ]
    _emit_platform_trace(
        "artifact_digest_gate",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="artifact_digest",
        passed=bool(artifact_candidates),
        reason=(
            "at least one exact-target version has an acceptable primary artifact and SHA-512"
            if artifact_candidates
            else "no exact-target version has exactly one safe primary artifact with positive size and valid SHA-512"
        ),
        details={
            "exact_target_candidate_count": len(exact_target_candidates),
            "artifact_candidate_count": len(artifact_candidates),
        },
    )

    if not eligible:
        all_reasons = sorted(
            {
                str(reason)
                for value in normalized
                for reason in value.get("unresolved_or_rejected_gates", ())
                if str(reason)
            }
        )
        if not exact_target_candidates:
            failed_gate = "exact_target"
            failure_reason = "no exact-target version was returned"
        elif not artifact_candidates:
            failed_gate = "artifact_digest"
            failure_reason = "exact-target versions failed primary artifact/origin/size/SHA-512 requirements"
        else:
            failed_gate = "version_metadata"
            failure_reason = "exact-target artifact exists but another version metadata gate failed"
        _emit_platform_trace(
            "project_inspection_rejected",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate=failed_gate,
            passed=False,
            reason=failure_reason,
            details={"rejection_reasons": all_reasons, "license_id": license_id},
        )
        return _ProjectInspection(
            project_id=project_id,
            role=role,
            verified=None,
            failed_gate=failed_gate,
            failure_reason=f"{failure_reason}; reasons={','.join(all_reasons) or 'none'}",
            license_id=license_id,
            license_policy=license_policy,
        )

    if not license_passed:
        failed_gate = "source_reuse_license" if role == "source_reuse" else "dependency_license"
        _emit_platform_trace(
            "project_inspection_rejected",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate=failed_gate,
            passed=False,
            reason=license_reason,
            details={"license_id": license_id, "license_policy": license_policy},
        )
        return _ProjectInspection(
            project_id=project_id,
            role=role,
            verified=None,
            failed_gate=failed_gate,
            failure_reason=license_reason,
            license_id=license_id,
            license_policy=license_policy,
        )

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
    _emit_platform_trace(
        "version_selected",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="version_selection",
        passed=True,
        reason="newest eligible exact-target version selected",
        details={
            "version_id": selected.get("version_id", ""),
            "version_number": selected.get("version_number", ""),
            "date_published": selected.get("date_published", ""),
            "file_count": len(files),
            "dependency_record_count": len(dependencies),
        },
    )

    required: list[str] = []
    for dependency_index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            _emit_platform_trace(
                "dependency_record_gate",
                adapter=adapter,
                project_id=project_id,
                parent_project_id=parent_project_id,
                dependency_path=dependency_path,
                role=role,
                gate="dependency_record_shape",
                passed=False,
                reason="dependency record is not an object",
                details={"dependency_index": dependency_index},
            )
            continue
        dependency_type = str(dependency.get("dependency_type") or "").casefold()
        dependency_project = str(dependency.get("project_id") or "").strip()
        dependency_version = str(dependency.get("version_id") or "").strip()
        _emit_platform_trace(
            "dependency_record_gate",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=dependency_path,
            role=role,
            gate="dependency_record",
            passed=True,
            reason="dependency record normalized",
            details={
                "dependency_index": dependency_index,
                "dependency_type": dependency_type,
                "project_id": dependency_project,
                "version_id": dependency_version,
            },
        )
        if dependency_type != "required":
            continue
        if not dependency_project and dependency_version:
            try:
                dependency_project = _project_for_version(client, dependency_version)
            except Exception as exc:
                failure_reason = (
                    f"required dependency version {dependency_version} could not be resolved: "
                    f"{type(exc).__name__}: {exc}"
                )
                _emit_platform_trace(
                    "dependency_identity_gate",
                    adapter=adapter,
                    project_id=project_id,
                    parent_project_id=parent_project_id,
                    dependency_path=dependency_path,
                    role=role,
                    gate="dependency_identity",
                    passed=False,
                    reason=failure_reason,
                    details={"exception_chain": _exception_chain(exc)},
                )
                return _ProjectInspection(
                    project_id=project_id,
                    role=role,
                    verified=None,
                    failed_gate="dependency_identity",
                    failure_reason=failure_reason,
                    license_id=license_id,
                    license_policy=license_policy,
                )
        if not dependency_project:
            failure_reason = f"required dependency index {dependency_index} has no resolvable project identity"
            _emit_platform_trace(
                "dependency_identity_gate",
                adapter=adapter,
                project_id=project_id,
                parent_project_id=parent_project_id,
                dependency_path=dependency_path,
                role=role,
                gate="dependency_identity",
                passed=False,
                reason=failure_reason,
            )
            return _ProjectInspection(
                project_id=project_id,
                role=role,
                verified=None,
                failed_gate="dependency_identity",
                failure_reason=failure_reason,
                license_id=license_id,
                license_policy=license_policy,
            )
        required.append(dependency_project)

    verified = _VerifiedProject(
        project_id=project_id,
        exact_versions=1,
        hash_files=sum(
            bool(str(file.get("sha512") or ""))
            for file in files
            if isinstance(file, Mapping)
        ),
        dependency_edges=len(dependencies),
        required_projects=tuple(dict.fromkeys(required)),
        freshness=_timestamp(str(selected.get("date_published") or "")),
    )
    _emit_platform_trace(
        "project_inspection_success",
        adapter=adapter,
        project_id=project_id,
        parent_project_id=parent_project_id,
        dependency_path=dependency_path,
        role=role,
        gate="project_inspection",
        passed=True,
        reason="all applicable project gates passed",
        details={
            "license_id": license_id,
            "license_policy": license_policy,
            "required_projects": verified.required_projects,
            "verified_hash_files": verified.hash_files,
        },
    )
    return _ProjectInspection(
        project_id=project_id,
        role=role,
        verified=verified,
        license_id=license_id,
        license_policy=license_policy,
    )


def _project_for_version(client: EcosystemDiscoveryClient, version_id: str) -> str:
    raw = client._get_json(f"https://api.modrinth.com/v2/version/{version_id}")
    valid = isinstance(raw, Mapping)
    project_id = str(raw.get("project_id") or "").strip() if valid else ""
    _emit_platform_trace(
        "dependency_version_resolution",
        gate="dependency_identity",
        passed=bool(valid and project_id),
        reason=(
            "dependency version resolved to project identity"
            if valid and project_id
            else "dependency version metadata did not resolve to a project identity"
        ),
        details={
            "version_id": version_id,
            "response_type": type(raw).__name__,
            "project_id": project_id,
        },
    )
    if not valid:
        raise EcosystemDiscoveryUnavailable(
            f"Modrinth returned invalid dependency version metadata for {version_id}."
        )
    return project_id


def _required_dependency_closure(
    client: EcosystemDiscoveryClient,
    adapter: PlatformAdapter,
    seeds: Sequence[str],
    *,
    inspection_cache: dict[str, _VerifiedProject | None],
    root_project_id: str = "",
) -> set[str]:
    pending: list[tuple[str, tuple[str, ...]]] = []
    for value in seeds:
        project_id = str(value).strip()
        if not project_id:
            continue
        initial_path = tuple(
            part for part in (root_project_id, project_id) if part
        )
        if not any(existing_id == project_id for existing_id, _path in pending):
            pending.append((project_id, initial_path))

    seen: set[str] = set()
    while pending:
        project_id, path = pending.pop(0)
        if project_id in seen:
            _emit_platform_trace(
                "dependency_closure_skip",
                adapter=adapter,
                project_id=project_id,
                dependency_path=path,
                role="required_dependency",
                gate="dependency_cycle_or_duplicate",
                passed=True,
                reason="dependency already inspected in this closure",
            )
            continue
        seen.add(project_id)
        parent_project_id = path[-2] if len(path) >= 2 else ""

        # A cached successful source-reuse inspection is stricter than dependency
        # inspection and is safe to reuse. A cached None is NOT reused: it may only
        # mean the project failed the stricter source-reuse license policy.
        verified = inspection_cache.get(project_id)
        if verified is not None:
            _emit_platform_trace(
                "dependency_inspection_cache_hit",
                adapter=adapter,
                project_id=project_id,
                parent_project_id=parent_project_id,
                dependency_path=path,
                role="required_dependency",
                gate="inspection_cache",
                passed=True,
                reason="cached verified project reused",
            )
        else:
            inspection = _inspect_project_receipt_native_detailed(
                client,
                project_id,
                adapter,
                role="required_dependency",
                parent_project_id=parent_project_id,
                dependency_path=path,
            )
            verified = inspection.verified
            inspection_cache[project_id] = verified
            if verified is None:
                path_text = " -> ".join(path) or project_id
                message = (
                    f"Required dependency path {path_text} failed "
                    f"gate={inspection.failed_gate or 'project_inspection'}; "
                    f"reason={inspection.failure_reason or 'unspecified'}; "
                    f"license_id={inspection.license_id or '<missing>'}; "
                    f"license_policy={inspection.license_policy or '<missing>'}"
                )
                _emit_platform_trace(
                    "dependency_closure_failure",
                    adapter=adapter,
                    project_id=project_id,
                    parent_project_id=parent_project_id,
                    dependency_path=path,
                    role="required_dependency",
                    gate=inspection.failed_gate or "project_inspection",
                    passed=False,
                    reason=message,
                )
                raise SpecValidationError(message)

        _emit_platform_trace(
            "dependency_closure_node",
            adapter=adapter,
            project_id=project_id,
            parent_project_id=parent_project_id,
            dependency_path=path,
            role="required_dependency",
            gate="dependency_closure",
            passed=True,
            reason="required dependency node verified",
            details={"required_projects": verified.required_projects},
        )
        for dependency in verified.required_projects:
            if dependency not in seen:
                pending.append((dependency, path + (dependency,)))
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
