from __future__ import annotations

"""Platform selection DTOs, parsing, and proposal binding.

Actual optimisation lives in platform_evidence_pipeline through the canonical
platform_selection_pipeline. This module performs no search, retry, or ranking itself.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .platform_catalog import (
    PlatformAdapter,
    adapter_for_target,
    adapters_for_version,
    discover_target_keys,
    executable_loaders,
    provider_for_loader,
)
from .platform_evidence_pipeline import PlatformOptimization, TargetResearchFn
from .spec import PlatformLock, Proposal, SpecValidationError, platform_receipt_sha256

_VERSION_RE = re.compile(r"(?<!\d)(1\.\d{1,2}(?:\.\d{1,2})?|\d{2,4}\.\d+(?:\.\d+)?)(?!\d)")
_ASCII_WORD = r"A-Za-z0-9_"
_FABRIC_RE = re.compile(rf"(?<![{_ASCII_WORD}])fabric(?![{_ASCII_WORD}])|패브릭", re.IGNORECASE)
_NEOFORGE_RE = re.compile(rf"(?<![{_ASCII_WORD}])neoforge(?![{_ASCII_WORD}])|네오포지", re.IGNORECASE)
_FORGE_RE = re.compile(rf"(?<![{_ASCII_WORD}])forge(?![{_ASCII_WORD}])|(?<!네오)포지", re.IGNORECASE)
_MIGRATION_RE = re.compile(
    r"마이그레이션|버전\s*(?:변경|업|올려|내려)|업데이트\s*해|포팅|이식|"
    r"migrat|port\s+(?:to|from)|upgrade\s+to|downgrade\s+to",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlatformSelection:
    adapter: PlatformAdapter
    source: str
    reason: str
    explicit_version: bool
    explicit_loader: bool
    preserved_existing_target: bool = False
    migration_requested: bool = False
    optimization: PlatformOptimization | None = None

    @property
    def lock(self) -> PlatformLock:
        return lock_from_adapter(self.adapter)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "mmm/platform-selection-v5",
            "adapter_id": self.adapter.adapter_id,
            "source": self.source,
            "reason": self.reason,
            "explicit_version": self.explicit_version,
            "explicit_loader": self.explicit_loader,
            "preserved_existing_target": self.preserved_existing_target,
            "migration_requested": self.migration_requested,
            "target": {
                "adapter_id": self.adapter.adapter_id,
                "edition": self.adapter.edition,
                "loader": self.adapter.loader,
                "minecraft_version": self.adapter.minecraft_version,
                "java_version": self.adapter.java_version,
                "yarn_mappings": self.adapter.yarn_mappings,
                "mappings_kind": self.adapter.mappings_kind,
                "mappings_version": self.adapter.mappings_version,
                "mappings": self.adapter.yarn_mappings,
                "fabric_loader": self.adapter.fabric_loader,
                "fabric_api": self.adapter.fabric_api,
                "fabric_loom": self.adapter.fabric_loom,
                "gradle": self.adapter.gradle,
                "gradle_sha256": self.adapter.gradle_sha256,
                "data_pack_version": self.adapter.data_pack_version,
                "resource_pack_version": self.adapter.resource_pack_version,
                "resource_pack_format": self.adapter.resource_pack_format,
                "release_metadata_url": self.adapter.release_metadata_url,
                "source_api_family": self.adapter.source_api_family,
            },
        }
        if self.optimization is not None:
            payload["optimizer"] = self.optimization.to_dict()
        return payload


def lock_from_adapter(adapter: PlatformAdapter) -> PlatformLock:
    """Copy the already-validated provider receipt without re-resolving it.

    PlatformLock is the immutable execution boundary.  Dropping the extended receipt
    fields here used to turn a fully verified adapter back into a legacy/partial lock,
    which then failed native-name and receipt validation downstream.
    """

    adapter.validate()
    lock = PlatformLock(
        adapter_id=adapter.adapter_id,
        edition=adapter.edition,
        loader=adapter.loader,
        minecraft_version=adapter.minecraft_version,
        java_version=adapter.java_version,
        yarn_mappings=adapter.yarn_mappings,
        mappings_kind=adapter.mappings_kind,
        mappings_version=adapter.mappings_version,
        fabric_loader=adapter.fabric_loader,
        fabric_api=adapter.fabric_api,
        fabric_loom=adapter.fabric_loom,
        gradle=adapter.gradle,
        gradle_sha256=adapter.gradle_sha256,
        gradle_distribution_url=(
            f"https://services.gradle.org/distributions/gradle-{adapter.gradle}-bin.zip"
        ),
        data_pack_version=adapter.data_pack_version,
        resource_pack_version=adapter.resource_pack_version,
        resource_pack_format=adapter.resource_pack_format,
        release_metadata_url=adapter.release_metadata_url,
        source_api_family=adapter.source_api_family,
        deterministic_module_kinds=tuple(sorted(adapter.deterministic_module_kinds)),
    )
    lock = replace(lock, receipt_sha256=platform_receipt_sha256(lock))
    lock.validate()
    return lock


def resolve_platform(
    prompt: str,
    *,
    design: dict[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    existing_version: str | None = None,
    existing_loader: str | None = None,
    router: Any | None = None,
    target_research_fn: TargetResearchFn | None = None,
) -> PlatformSelection:
    """Compatibility entrypoint routed only to the canonical fail-closed selector."""

    del router
    from .platform_selection_pipeline import resolve_platform_fail_closed

    return resolve_platform_fail_closed(
        prompt,
        design=design,
        module_kinds=module_kinds,
        existing_version=existing_version,
        existing_loader=existing_loader,
        target_research_fn=target_research_fn,
    )


def _optimized_selection(
    optimization: PlatformOptimization,
    *,
    source: str,
    explicit_version: bool,
    explicit_loader: bool,
    migration_requested: bool,
) -> PlatformSelection:
    adapter = optimization.selected
    evidence = optimization.evidence
    return PlatformSelection(
        adapter=adapter,
        source=source,
        reason=(
            f"Verified evidence selected {adapter.minecraft_version}/{adapter.loader}: "
            f"reuse {evidence.reuse_coverage}/{len(evidence.requested_capabilities)}, "
            f"residual {evidence.residual_cost}, dependency closure "
            f"{'complete' if evidence.dependency_closure_complete else 'incomplete'}."
        ),
        explicit_version=explicit_version,
        explicit_loader=explicit_loader,
        migration_requested=migration_requested,
        optimization=optimization,
    )


def _exact_adapter(version: str, loader: str) -> PlatformAdapter:
    try:
        return adapter_for_target(version, loader)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc


def _existing_adapter(version: str, loader: str | None) -> PlatformAdapter:
    if loader and str(loader).strip():
        return _exact_adapter(str(version), str(loader))
    candidates = adapters_for_version(str(version))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SpecValidationError(
            f"기존 프로젝트 Minecraft {version} target을 실행할 provider가 없습니다."
        )
    raise SpecValidationError(
        "기존 프로젝트 loader를 식별할 수 없고 같은 Minecraft 버전에 여러 실행 "
        "provider가 존재합니다. 기존 target을 추측하지 않습니다."
    )


def retarget_proposal(proposal: Proposal, selection: PlatformSelection) -> Proposal:
    from .knowledge import evidence_for_target, evidence_snapshot_hash

    spec = replace(proposal.spec, platform=selection.lock)
    evidence_terms: list[str] = [proposal.requested_prompt, "project build metadata dependency"]
    if proposal.spec.boss is not None:
        evidence_terms.append("boss entity gametest runtime server")
    if proposal.spec.contents:
        evidence_terms.append("item block recipe data generation resource")
    evidence = evidence_for_target(
        " ".join(evidence_terms),
        minecraft_version=selection.adapter.minecraft_version,
    )
    assumptions = tuple(
        value
        for value in proposal.assumptions
        if not ("Minecraft Java Edition" in value or "Minecraft " in value and "Fabric" in value)
    ) + (
        (
            f"Target: Minecraft Java {selection.adapter.minecraft_version}, "
            f"{selection.adapter.loader}, Java {selection.adapter.java_version}. "
            f"{selection.reason}"
        ),
    )
    return replace(
        proposal,
        spec=spec,
        assumptions=assumptions,
        evidence_sources=evidence,
        evidence_snapshot_hash=evidence_snapshot_hash(evidence),
        approval_hash="",
    ).with_hash()


def supported_target_summary() -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    for loader, version in discover_target_keys(limit_per_loader=32):
        result.append(
            {
                "minecraft_version": version,
                "loader": loader,
                "provider": provider_for_loader(loader).provider_id,
            }
        )
    return tuple(result)


def _explicit_minecraft_version(prompt: str) -> str | None:
    matches = _VERSION_RE.findall(prompt)
    if not matches:
        return None
    return list(dict.fromkeys(matches))[-1]


def _explicit_loader(prompt: str) -> str | None:
    found: list[str] = []
    if _FABRIC_RE.search(prompt):
        found.append("fabric")
    if _NEOFORGE_RE.search(prompt):
        found.append("neoforge")
    if _FORGE_RE.search(prompt):
        found.append("forge")
    unique = list(dict.fromkeys(found))
    if len(unique) > 1:
        raise SpecValidationError(f"하나의 프로젝트에 여러 로더가 동시에 명시되었습니다: {unique}")
    return unique[0] if unique else None


def _require_supported_kinds(
    adapter: PlatformAdapter,
    module_kinds: Iterable[str],
    *,
    explicit: bool,
) -> None:
    if adapter.source_api_family == "fabric_live_ai":
        return
    kinds = {str(value).strip() for value in module_kinds if str(value).strip()}
    unsupported = sorted(kinds - adapter.deterministic_module_kinds)
    if not unsupported:
        return
    prefix = "명시한 target" if explicit else "선택된 target"
    raise SpecValidationError(
        f"{prefix} {adapter.minecraft_version}/{adapter.loader}의 deterministic legacy "
        f"generator가 지원하지 않는 종류가 있습니다: {unsupported}."
    )


def executable_target_names() -> tuple[str, ...]:
    return tuple(
        f"{version}/{loader}"
        for loader in executable_loaders()
        for candidate_loader, version in discover_target_keys(loader=loader, limit_per_loader=32)
        if candidate_loader == loader
    )


__all__ = [
    "PlatformSelection",
    "executable_target_names",
    "lock_from_adapter",
    "resolve_platform",
    "retarget_proposal",
    "supported_target_summary",
]
