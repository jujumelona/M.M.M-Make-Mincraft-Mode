from __future__ import annotations

"""Fail-closed platform selection for the canonical planning pipeline.

Candidate incompatibility is allowed only while discovering candidates. Once a
PlatformAdapter has been admitted, it is never re-resolved. Evidence acquisition for a
shortlisted target must complete; transport/research failures are not converted into a
fresh-only target that can win ranking.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import (
    PlatformAdapter,
    _candidate_versions,
    _resolve_candidate,
    executable_loaders,
    provider_for_loader,
)
from . import platform_optimizer as legacy_optimizer
from .platform_optimizer import PlatformOptimization, TargetEvidence, TargetResearchFn
from .platform_resolver import PlatformSelection
from .spec import SpecValidationError


def _candidate_page_size(top_k: int) -> int:
    """Return an operator-visible resource page size, not a semantic feature cap."""
    raw = os.environ.get("MMM_PLATFORM_CANDIDATE_LIMIT", "").strip()
    if not raw:
        return max(1, int(top_k))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SpecValidationError(
            "MMM_PLATFORM_CANDIDATE_LIMIT must be a positive integer."
        ) from exc
    if value < 1:
        raise SpecValidationError(
            "MMM_PLATFORM_CANDIDATE_LIMIT must be a positive integer."
        )
    return value


def _resolved_candidates(
    *,
    loader_constraint: str | None,
    version_constraint: str | None,
    page_size: int,
) -> tuple[PlatformAdapter, ...]:
    """Resolve each candidate exactly once and return validated adapter receipts."""
    loaders = (
        (provider_for_loader(loader_constraint).loader,)
        if loader_constraint
        else executable_loaders()
    )
    requested_version = str(version_constraint or "").strip()
    adapters: list[PlatformAdapter] = []
    diagnostics: list[str] = []

    for loader in loaders:
        provider = provider_for_loader(loader)
        if requested_version:
            adapter, error = _resolve_candidate(provider, requested_version)
            if adapter is None:
                raise SpecValidationError(
                    "Explicit platform target is not executable: "
                    f"{loader}/{requested_version}: {error or 'unknown provider failure'}"
                )
            adapters.append(adapter)
            continue

        try:
            versions = _candidate_versions(provider, limit=page_size)
        except Exception as exc:
            diagnostics.append(
                f"{loader}: version discovery {type(exc).__name__}: {exc}"
            )
            continue

        for version in versions:
            adapter, error = _resolve_candidate(provider, version)
            if adapter is not None:
                adapters.append(adapter)
            else:
                diagnostics.append(
                    f"{loader}/{version}: {error or 'incompatible candidate'}"
                )

    if not adapters:
        detail = "; ".join(diagnostics) or "no provider returned a validated adapter receipt"
        raise SpecValidationError(
            "No executable platform candidate was resolved. Diagnostics: " + detail
        )
    return tuple(adapters)


def _fatal_target_evidence_errors(
    evidence: TargetEvidence,
    *,
    inherited_errors: Sequence[str],
) -> tuple[str, ...]:
    inherited = set(inherited_errors)
    local = [error for error in evidence.discovery_errors if error not in inherited]
    fatal_prefixes = ("rag:", "inspect:", "dependency:", "dependency-closure:")
    return tuple(error for error in local if error.startswith(fatal_prefixes))


def _parallel_deep_fail_closed(
    adapters: Sequence[PlatformAdapter],
    *,
    queries: Sequence[str],
    matrix: Mapping[str, Mapping[str, tuple[str, ...]]],
    client: EcosystemDiscoveryClient,
    target_research_fn: TargetResearchFn | None,
    inherited_errors: Sequence[str],
    shallow_candidate_count: int,
) -> tuple[TargetEvidence, ...]:
    """Run deep evidence without legacy fresh-only recovery."""
    accepted: list[TargetEvidence] = []
    failures: list[str] = []
    workers = min(legacy_optimizer._workers(), max(1, len(adapters)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-platform-strict") as pool:
        futures = {
            pool.submit(
                legacy_optimizer._deep_evidence,
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
            adapter = futures[future]
            try:
                evidence = future.result()
            except Exception as exc:
                failures.append(
                    f"{adapter.minecraft_version}/{adapter.loader}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            fatal = _fatal_target_evidence_errors(
                evidence,
                inherited_errors=inherited_errors,
            )
            if fatal:
                failures.append(
                    f"{adapter.minecraft_version}/{adapter.loader}: " + "; ".join(fatal)
                )
                continue
            accepted.append(evidence)

    if not accepted:
        detail = "; ".join(failures) or "all shortlisted targets lacked complete evidence"
        raise SpecValidationError(
            "Platform target evidence did not complete for any shortlisted target. "
            "Refusing fresh-only fallback. Diagnostics: " + detail
        )
    return tuple(accepted)


def optimize_platform_fail_closed(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    top_k: int = 4,
    discovery_client: EcosystemDiscoveryClient | None = None,
    target_research_fn: TargetResearchFn | None = None,
    search_fn: legacy_optimizer.SearchFn | None = None,
    version_fn: legacy_optimizer.VersionFn | None = None,
) -> PlatformOptimization:
    """Select only from single-resolution provider receipts and complete evidence."""
    if int(top_k) < 1:
        raise SpecValidationError("top_k must be a positive resource budget.")
    queries = legacy_optimizer.capability_queries(
        prompt,
        design=design,
        module_kinds=module_kinds,
    )
    adapters = _resolved_candidates(
        loader_constraint=loader_constraint,
        version_constraint=version_constraint,
        page_size=_candidate_page_size(int(top_k)),
    )

    if search_fn is not None or version_fn is not None:
        return legacy_optimizer._optimize_fixture_path(
            queries,
            adapters=list(adapters),
            search_fn=search_fn,
            version_fn=version_fn,
        )

    discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
    if discovery_mode not in {"auto", "on", "off"}:
        raise SpecValidationError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
    if discovery_mode == "off":
        if len(adapters) != 1:
            raise SpecValidationError(
                "Ecosystem discovery is disabled and multiple executable platform "
                "targets remain. Supply a single target or enable discovery."
            )
        offline = legacy_optimizer._optimize_fixture_path(
            queries,
            adapters=list(adapters),
            search_fn=lambda _query: (),
            version_fn=lambda _project: (),
        )
        return PlatformOptimization(
            selected=offline.selected,
            evidence=offline.evidence,
            candidates=offline.candidates,
            capability_queries=offline.capability_queries,
            discovery_mode="single-resolution-provider-receipt-only",
        )

    client = discovery_client or EcosystemDiscoveryClient()
    neutral, neutral_errors = legacy_optimizer._parallel_neutral_shallow(queries, client)
    shallow_count = sum(len(value) for value in neutral.values())
    matrix, matrix_errors = legacy_optimizer._parallel_support_matrix(adapters, queries, client)
    inherited_errors = (*neutral_errors, *matrix_errors)

    hypotheses = sorted(
        adapters,
        key=lambda adapter: (
            -legacy_optimizer._support_score(
                adapter,
                queries,
                matrix.get(adapter.adapter_id, {}),
            ),
            adapter.loader,
            adapter.minecraft_version,
            adapter.adapter_id,
        ),
    )
    shortlist = tuple(hypotheses[: min(int(top_k), len(hypotheses))])
    deep = _parallel_deep_fail_closed(
        shortlist,
        queries=queries,
        matrix=matrix,
        client=client,
        target_research_fn=target_research_fn,
        inherited_errors=inherited_errors,
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
    return PlatformOptimization(
        selected=ranked[0].adapter,
        evidence=ranked[0],
        candidates=ranked,
        capability_queries=queries,
        discovery_mode=(
            "single-resolution_receipts_then-support-matrix_then-fail-closed-deep-evidence"
        ),
    )


def resolve_platform_fail_closed(
    prompt: str,
    *,
    design: dict[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    existing_version: str | None = None,
    existing_loader: str | None = None,
    target_research_fn: TargetResearchFn | None = None,
) -> PlatformSelection:
    """Resolve a planning target without the legacy rediscovery/skip path."""
    from . import platform_resolver as resolver

    text = str(prompt or "")
    explicit_version = resolver._explicit_minecraft_version(text)
    explicit_loader = resolver._explicit_loader(text)
    migration_requested = bool(existing_version and resolver._MIGRATION_RE.search(text))
    kinds = tuple(str(value).strip() for value in module_kinds if str(value).strip())

    if explicit_loader:
        try:
            provider_for_loader(explicit_loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc

    if existing_version and not migration_requested:
        adapter = resolver._existing_adapter(existing_version, existing_loader)
        resolver._require_supported_kinds(adapter, kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source="existing_project_target",
            reason=(
                f"Existing project target {adapter.minecraft_version}/{adapter.loader} "
                "is preserved because no migration was requested."
            ),
            explicit_version=False,
            explicit_loader=False,
            preserved_existing_target=True,
        )

    optimization = optimize_platform_fail_closed(
        text,
        design=design,
        module_kinds=kinds,
        loader_constraint=explicit_loader,
        version_constraint=None,
        target_research_fn=target_research_fn,
    )
    return resolver._optimized_selection(
        optimization,
        source=(
            "fail_closed_reuse_optimizer_with_version_hint"
            if explicit_version
            else "fail_closed_reuse_optimizer"
        ),
        explicit_version=bool(explicit_version),
        explicit_loader=bool(explicit_loader),
        migration_requested=migration_requested,
    )


__all__ = [
    "optimize_platform_fail_closed",
    "resolve_platform_fail_closed",
]
