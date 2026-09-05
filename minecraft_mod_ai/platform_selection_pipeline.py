from __future__ import annotations

"""Canonical fail-closed platform selection.

This module owns target selection as a deterministic boundary around the receipt-native
evidence pipeline. It contains no legacy optimizer, no post-admission re-resolution, no
fresh-only recovery, and no semantic top-k shortlist.
"""

import os
import re
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

_HOST_VERSION_CONSTRAINT_RE = re.compile(
    r"\[HOST_TARGET_CONSTRAINT\s+Minecraft\s+([^\]\s]+)\]",
    re.IGNORECASE,
)
_HOST_LOADER_CONSTRAINT_RE = re.compile(
    r"\[HOST_LOADER_CONSTRAINT\s+([^\]\s]+)\]",
    re.IGNORECASE,
)


def _host_target_constraints(text: str) -> tuple[str | None, str | None]:
    """Read host-owned target markers without granting them semantic authority."""
    version_matches = _HOST_VERSION_CONSTRAINT_RE.findall(str(text or ""))
    loader_matches = _HOST_LOADER_CONSTRAINT_RE.findall(str(text or ""))
    version = str(version_matches[-1]).strip() if version_matches else None
    loader = str(loader_matches[-1]).strip().casefold() if loader_matches else None
    return version or None, loader or None


def _host_retarget_requested(
    *,
    existing_version: str | None,
    existing_loader: str | None,
    host_version: str | None,
    host_loader: str | None,
) -> bool:
    """Return whether an explicit host target differs from the imported project target."""
    if not str(existing_version or "").strip():
        return False
    existing_version_value = str(existing_version or "").strip()
    existing_loader_value = str(existing_loader or "").strip().casefold()
    return bool(
        (host_version and host_version != existing_version_value)
        or (host_loader and host_loader != existing_loader_value)
    )


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


def _provider_only_adapter(
    loaders: tuple[str, ...],
    *,
    version_constraint: str | None,
):
    """Resolve one executable receipt without consulting ecosystem/live catalogues.

    ``MMM_ECOSYSTEM_DISCOVERY=off`` is an explicit isolation boundary. In that mode the
    platform provider is the sole authority for candidate versions; using Mojang's live
    stable catalogue here both violated that boundary and made deterministic tests drift
    to whatever release happened to be newest.
    """

    requested = str(version_constraint or "").strip()
    failures: list[str] = []
    for loader in loaders:
        provider = provider_for_loader(loader)
        if requested:
            versions = (requested,)
        else:
            try:
                versions = tuple(
                    dict.fromkeys(
                        value
                        for item in provider.discover_versions(32)
                        if (value := str(item).strip())
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{loader}: discovery {type(exc).__name__}: {exc}")
                continue
        for version in versions:
            try:
                adapter = provider.resolve(version)
                adapter.validate()
                return adapter
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{loader}/{version}: {type(exc).__name__}: {exc}")
    detail = "; ".join(failures) or "providers returned no executable candidates"
    raise SpecValidationError(
        "No executable platform target survived provider-only resolution. Diagnostics: "
        + detail
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
    host_version, host_loader = _host_target_constraints(text)
    parsed_version = resolver._explicit_minecraft_version(text)
    parsed_loader = None if host_loader else resolver._explicit_loader(text)
    explicit_version = host_version or parsed_version
    explicit_loader = host_loader or parsed_loader
    host_retarget = _host_retarget_requested(
        existing_version=existing_version,
        existing_loader=existing_loader,
        host_version=host_version,
        host_loader=host_loader,
    )
    migration_requested = bool(
        existing_version and (resolver._MIGRATION_RE.search(text) or host_retarget)
    )
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

    discovery_mode = str(os.getenv("MMM_ECOSYSTEM_DISCOVERY", "auto")).strip().casefold()
    if discovery_mode not in {"auto", "on", "off"}:
        raise SpecValidationError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")

    if discovery_mode == "off":
        loaders = (
            (provider_for_loader(explicit_loader).loader,)
            if explicit_loader
            else resolver.executable_loaders()
        )
        adapter = _provider_only_adapter(
            tuple(loaders),
            version_constraint=explicit_version,
        )
        resolver._require_supported_kinds(adapter, kinds, explicit=bool(explicit_version))
        return PlatformSelection(
            adapter=adapter,
            source="provider_receipt_only",
            reason=(
                f"Ecosystem discovery is disabled; executable provider receipt "
                f"{adapter.adapter_id} selected {adapter.minecraft_version}/{adapter.loader}."
            ),
            explicit_version=bool(explicit_version),
            explicit_loader=bool(explicit_loader),
            migration_requested=migration_requested,
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
