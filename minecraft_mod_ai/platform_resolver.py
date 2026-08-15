from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .platform_catalog import (
    PlatformAdapter,
    adapter_for_target,
    adapters_for_version,
    discover_target_keys,
    executable_loaders,
    provider_for_loader,
)
from .platform_optimizer import PlatformOptimization, optimize_platform
from .spec import PlatformLock, Proposal, SpecValidationError


_VERSION_RE = re.compile(r"(?<!\d)(1\.\d{1,2}(?:\.\d{1,2})?|\d{2,4}\.\d+(?:\.\d+)?)(?!\d)")
_ASCII_WORD = r"A-Za-z0-9_"
_FABRIC_RE = re.compile(
    rf"(?<![{_ASCII_WORD}])fabric(?![{_ASCII_WORD}])|패브릭",
    re.IGNORECASE,
)
_NEOFORGE_RE = re.compile(
    rf"(?<![{_ASCII_WORD}])neoforge(?![{_ASCII_WORD}])|네오포지",
    re.IGNORECASE,
)
_FORGE_RE = re.compile(
    rf"(?<![{_ASCII_WORD}])forge(?![{_ASCII_WORD}])|(?<!네오)포지",
    re.IGNORECASE,
)
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
        mappings_kind = "mojang" if self.adapter.yarn_mappings == "mojang" else "yarn"
        payload: dict[str, Any] = {
            "schema_version": "mmm/platform-selection-v4",
            "adapter_id": self.adapter.adapter_id,
            "source": self.source,
            "reason": self.reason,
            "explicit_version": self.explicit_version,
            "explicit_loader": self.explicit_loader,
            "preserved_existing_target": self.preserved_existing_target,
            "migration_requested": self.migration_requested,
            "target": {
                "edition": self.adapter.edition,
                "loader": self.adapter.loader,
                "minecraft_version": self.adapter.minecraft_version,
                "java_version": self.adapter.java_version,
                "mappings_kind": mappings_kind,
                "mappings": self.adapter.yarn_mappings,
                "fabric_loader": self.adapter.fabric_loader,
                "fabric_api": self.adapter.fabric_api,
                "fabric_loom": self.adapter.fabric_loom,
                "gradle": self.adapter.gradle,
                "source_api_family": self.adapter.source_api_family,
            },
        }
        if self.optimization is not None:
            payload["optimizer"] = self.optimization.to_dict()
        return payload


def lock_from_adapter(adapter: PlatformAdapter) -> PlatformLock:
    return PlatformLock(
        edition=adapter.edition,
        loader=adapter.loader,
        minecraft_version=adapter.minecraft_version,
        java_version=adapter.java_version,
        yarn_mappings=adapter.yarn_mappings,
        fabric_loader=adapter.fabric_loader,
        fabric_api=adapter.fabric_api,
        fabric_loom=adapter.fabric_loom,
        gradle=adapter.gradle,
    )


def resolve_platform(
    prompt: str,
    *,
    design: dict[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    existing_version: str | None = None,
    existing_loader: str | None = None,
    router: Any | None = None,
) -> PlatformSelection:
    """Resolve one executable target using hard constraints then host evidence ranking.

    The router is accepted for API compatibility but does not choose coordinates.  The
    small model may already have contributed semantic capability labels in ``design``;
    exact compatibility, provider gates and final target ranking are host-owned.
    """
    del router
    text = str(prompt or "")
    explicit_version = _explicit_minecraft_version(text)
    explicit_loader = _explicit_loader(text)
    migration_requested = bool(existing_version and _MIGRATION_RE.search(text))
    kinds = tuple(str(value).strip() for value in module_kinds if str(value).strip())

    if explicit_loader:
        try:
            provider_for_loader(explicit_loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc

    if explicit_version and explicit_loader:
        adapter = _exact_adapter(explicit_version, explicit_loader)
        _require_supported_kinds(adapter, kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source="user_explicit_migration_target" if migration_requested else "user_explicit_target",
            reason=_explicit_reason(adapter, migration_requested),
            explicit_version=True,
            explicit_loader=True,
            migration_requested=migration_requested,
        )

    if explicit_version and not explicit_loader:
        exact = adapters_for_version(explicit_version)
        if not exact:
            raise SpecValidationError(
                f"Minecraft {explicit_version}을 실행할 수 있는 provider가 없습니다."
            )
        if len(exact) == 1:
            adapter = exact[0]
            _require_supported_kinds(adapter, kinds, explicit=True)
            return PlatformSelection(
                adapter=adapter,
                source="user_explicit_version_unique_provider",
                reason=(
                    f"사용자가 Minecraft {adapter.minecraft_version}을 명시했고, "
                    f"실행 가능한 provider가 {adapter.loader} 하나뿐입니다."
                ),
                explicit_version=True,
                explicit_loader=False,
                migration_requested=migration_requested,
            )
        optimization = _optimize(
            text,
            design=design,
            module_kinds=kinds,
            version_constraint=explicit_version,
        )
        return _optimized_selection(
            optimization,
            source="host_optimizer_explicit_version",
            explicit_version=True,
            explicit_loader=False,
            migration_requested=migration_requested,
        )

    if existing_version and not migration_requested:
        adapter = _existing_adapter(existing_version, existing_loader)
        _require_supported_kinds(adapter, kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source="existing_project_target",
            reason=(
                f"Revise 입력 프로젝트의 Minecraft {adapter.minecraft_version} "
                f"{adapter.loader} target을 그대로 유지합니다."
            ),
            explicit_version=False,
            explicit_loader=False,
            preserved_existing_target=True,
        )

    optimization = _optimize(
        text,
        design=design,
        module_kinds=kinds,
        loader_constraint=explicit_loader,
    )
    return _optimized_selection(
        optimization,
        source="host_evidence_optimizer",
        explicit_version=False,
        explicit_loader=bool(explicit_loader),
        migration_requested=migration_requested,
    )


def _optimize(
    prompt: str,
    *,
    design: dict[str, Any] | None,
    module_kinds: Iterable[str],
    loader_constraint: str | None = None,
    version_constraint: str | None = None,
) -> PlatformOptimization:
    try:
        return optimize_platform(
            prompt,
            design=design,
            module_kinds=module_kinds,
            loader_constraint=loader_constraint,
            version_constraint=version_constraint,
        )
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc


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
            f"실행 provider gate를 통과한 후보를 비교해 {adapter.minecraft_version}/"
            f"{adapter.loader}을 선택했습니다: 필수 capability "
            f"{len(evidence.covered_capabilities)}/{len(evidence.requested_capabilities)}, "
            f"검증 project {len(evidence.exact_projects)}, residual {evidence.residual_cost}. "
            "최신성은 마지막 tie-breaker로만 사용됩니다."
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


def _explicit_reason(adapter: PlatformAdapter, migration_requested: bool) -> str:
    if migration_requested:
        return (
            f"기존 프로젝트를 사용자가 명시한 Minecraft {adapter.minecraft_version} "
            f"{adapter.loader} target으로 migration합니다."
        )
    return f"사용자가 Minecraft {adapter.minecraft_version} {adapter.loader}을 명시했습니다."


def retarget_proposal(proposal: Proposal, selection: PlatformSelection) -> Proposal:
    from .knowledge import evidence_for_target, evidence_snapshot_hash

    spec = replace(proposal.spec, platform=selection.lock)
    evidence = evidence_for_target(
        proposal.requested_prompt,
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
