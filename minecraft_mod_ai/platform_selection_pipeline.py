from __future__ import annotations

"""Canonical fail-closed platform selection.

This module owns target selection as a deterministic boundary around the receipt-native
evidence pipeline. It contains no legacy optimizer, no post-admission re-resolution, no
fresh-only recovery, and no semantic top-k shortlist.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from .platform_catalog import provider_for_loader
from .platform_evidence_pipeline import (
    PlatformOptimization,
    TargetResearchFn,
    optimize_platform_evidence,
)
from .platform_resolver import PlatformSelection
from .spec import SpecValidationError


def optimize_platform_fail_closed(
    prompt: str,
    *,
    design: Mapping[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
    discovery_client: Any | None = None,
    target_research_fn: TargetResearchFn | None = None,
) -> PlatformOptimization:
    """Compatibility name for the canonical evidence optimiser.

    The old implementation accepted top-k/search fixture knobs that changed semantic
    coverage. Those knobs are intentionally gone; provider paging and dependency closure
    belong to the evidence pipeline and must either complete or fail.
    """

    return optimize_platform_evidence(
        prompt,
        design=design,
        module_kinds=module_kinds,
        loader_constraint=loader_constraint,
        version_constraint=version_constraint,
        discovery_client=discovery_client,
        target_research_fn=target_research_fn,
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
    """Resolve one immutable provider receipt and never re-resolve it downstream."""

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

    optimization = optimize_platform_evidence(
        text,
        design=design,
        module_kinds=kinds,
        loader_constraint=explicit_loader,
        version_constraint=explicit_version,
        target_research_fn=target_research_fn,
    )
    return resolver._optimized_selection(
        optimization,
        source=(
            "canonical_exact_target_evidence"
            if explicit_version
            else "canonical_reuse_frontier_evidence"
        ),
        explicit_version=bool(explicit_version),
        explicit_loader=bool(explicit_loader),
        migration_requested=migration_requested,
    )


__all__ = [
    "optimize_platform_fail_closed",
    "resolve_platform_fail_closed",
]
