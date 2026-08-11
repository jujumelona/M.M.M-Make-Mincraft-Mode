from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .platform_catalog import (
    PlatformAdapter,
    adapter_for_target,
    newest_adapter,
    supported_minecraft_versions,
)
from .spec import PlatformLock, Proposal, SpecValidationError


_VERSION_RE = re.compile(r"(?<!\d)(1\.\d{1,2}(?:\.\d{1,2})?)(?!\d)")
_FABRIC_RE = re.compile(r"\bfabric\b|패브릭", re.IGNORECASE)
_NEOFORGE_RE = re.compile(r"\bneoforge\b|네오포지", re.IGNORECASE)
_FORGE_RE = re.compile(r"(?<!neo)\bforge\b|(?<!네오)포지", re.IGNORECASE)
_MIGRATION_RE = re.compile(
    r"마이그레이션|버전\s*(?:변경|업|올려|내려)|업데이트\s*해|포팅|이식|"
    r"migrat|port\s+(?:to|from)|upgrade\s+to|downgrade\s+to",
    re.IGNORECASE,
)

# These deterministic generators still emit the mature 1.20.1 source family. A
# request containing one of them defaults to 1.20.1 until that source family has a
# reviewed 1.21.1 adapter. Explicit 1.21.1 never silently falls back.
_LEGACY_SOURCE_TERMS = (
    "보스", "boss", "몹", "mob", "entity", "npc", "엔피시", "작물", "crop",
    "농사", "farming", "무기", "weapon", "도구", "tool", "방어구", "armor",
    "음식", "food", "기계", "machine", "상태효과", "effect", "인챈트",
    "enchant", "차원", "dimension", "바이옴", "biome", "월드젠", "worldgen",
    "구조물", "structure", "gui", "network", "네트워크", "quest", "퀘스트",
    "skill", "스킬", "party", "파티", "guild", "길드",
)


@dataclass(frozen=True)
class PlatformSelection:
    adapter: PlatformAdapter
    source: str
    reason: str
    explicit_version: bool
    explicit_loader: bool
    preserved_existing_target: bool = False

    @property
    def lock(self) -> PlatformLock:
        return lock_from_adapter(self.adapter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/platform-selection-v1",
            "adapter_id": self.adapter.adapter_id,
            "source": self.source,
            "reason": self.reason,
            "explicit_version": self.explicit_version,
            "explicit_loader": self.explicit_loader,
            "preserved_existing_target": self.preserved_existing_target,
            "target": {
                "edition": self.adapter.edition,
                "loader": self.adapter.loader,
                "minecraft_version": self.adapter.minecraft_version,
                "java_version": self.adapter.java_version,
                "mappings": self.adapter.yarn_mappings,
                "fabric_loader": self.adapter.fabric_loader,
                "fabric_api": self.adapter.fabric_api,
                "fabric_loom": self.adapter.fabric_loom,
                "gradle": self.adapter.gradle,
                "source_api_family": self.adapter.source_api_family,
            },
        }


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
) -> PlatformSelection:
    text = str(prompt or "")
    explicit_version = _explicit_minecraft_version(text)
    explicit_loader = _explicit_loader(text)

    if explicit_loader and explicit_loader != "fabric":
        raise SpecValidationError(
            f"요청한 로더 {explicit_loader!r}는 아직 실행 가능한 소스 어댑터가 없습니다. "
            "지원되는 로더를 가장한 코드를 생성하지 않습니다."
        )

    if explicit_version:
        try:
            adapter = adapter_for_target(explicit_version, explicit_loader or "fabric")
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
        _require_supported_kinds(adapter, module_kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source="user_explicit_target",
            reason=(
                f"사용자가 Minecraft {adapter.minecraft_version} {adapter.loader}을 명시했습니다."
            ),
            explicit_version=True,
            explicit_loader=bool(explicit_loader),
            preserved_existing_target=False,
        )

    if existing_version and not _MIGRATION_RE.search(text):
        loader = (existing_loader or "fabric").strip().lower()
        try:
            adapter = adapter_for_target(existing_version, loader)
        except ValueError as exc:
            raise SpecValidationError(
                "기존 프로젝트의 target을 보존할 실행 어댑터가 없습니다: " + str(exc)
            ) from exc
        _require_supported_kinds(adapter, module_kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source="existing_project_target",
            reason=(
                f"Revise 입력 프로젝트의 Minecraft {adapter.minecraft_version} "
                f"{adapter.loader} target을 유지합니다."
            ),
            explicit_version=False,
            explicit_loader=False,
            preserved_existing_target=True,
        )

    requested_kinds = {str(value).strip() for value in module_kinds if str(value).strip()}
    advanced = _requires_mature_source_family(text, design) or bool(
        requested_kinds - newest_adapter().deterministic_module_kinds
    )
    if advanced:
        adapter = adapter_for_target("1.20.1", "fabric")
        _require_supported_kinds(adapter, requested_kinds, explicit=False)
        reason = (
            "요청 기능에 현재 1.20.1 소스 어댑터에서만 검증된 API 계열이 포함되어 "
            "가장 최신 버전이라는 이유만으로 1.21.1을 강제하지 않습니다."
        )
    else:
        adapter = newest_adapter(loader="fabric")
        _require_supported_kinds(adapter, requested_kinds, explicit=False)
        reason = (
            "특정 버전 제약이나 레거시 전용 API 요구가 없어, 현재 검토된 어댑터 중 "
            f"가장 최신인 Minecraft {adapter.minecraft_version}을 선택했습니다."
        )
    return PlatformSelection(
        adapter=adapter,
        source="host_capability_resolution",
        reason=reason,
        explicit_version=False,
        explicit_loader=bool(explicit_loader),
        preserved_existing_target=False,
    )


def retarget_proposal(proposal: Proposal, selection: PlatformSelection) -> Proposal:
    """Bind a proposal and its authoritative evidence to one selected target."""

    from .knowledge import evidence_for_target, evidence_snapshot_hash

    spec = replace(proposal.spec, platform=selection.lock)
    evidence = evidence_for_target(
        proposal.requested_prompt,
        minecraft_version=selection.adapter.minecraft_version,
    )
    assumptions = tuple(
        value
        for value in proposal.assumptions
        if not (
            "Minecraft Java Edition" in value
            or "Minecraft " in value and "Fabric" in value
        )
    ) + (
        (
            f"Target: Minecraft Java {selection.adapter.minecraft_version}, "
            f"Fabric, Java {selection.adapter.java_version}. {selection.reason}"
        ),
    )
    updated = replace(
        proposal,
        spec=spec,
        assumptions=assumptions,
        evidence_sources=evidence,
        evidence_snapshot_hash=evidence_snapshot_hash(evidence),
        approval_hash="",
    )
    return updated.with_hash()


def supported_target_summary() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "minecraft_version": version,
            "loader": "fabric",
            "adapter_id": adapter_for_target(version, "fabric").adapter_id,
        }
        for version in supported_minecraft_versions(loader="fabric")
    )


def _explicit_minecraft_version(prompt: str) -> str | None:
    matches = _VERSION_RE.findall(prompt)
    if not matches:
        return None
    unique = list(dict.fromkeys(matches))
    # A migration request can mention source and destination versions. Prefer the
    # final version because natural requests overwhelmingly phrase it as A -> B.
    return unique[-1]


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
        raise SpecValidationError(
            f"하나의 프로젝트에 여러 로더가 동시에 명시되었습니다: {unique}"
        )
    return unique[0] if unique else None


def _requires_mature_source_family(
    prompt: str,
    design: dict[str, Any] | None,
) -> bool:
    lowered = prompt.casefold()
    if any(term.casefold() in lowered for term in _LEGACY_SOURCE_TERMS):
        return True
    if not isinstance(design, dict):
        return False
    serialized = repr(
        {
            "modules": design.get("modules", []),
            "combat": design.get("combat", {}),
            "mod_context": design.get("mod_context", {}),
        }
    ).casefold()
    return any(term.casefold() in serialized for term in _LEGACY_SOURCE_TERMS)


def _require_supported_kinds(
    adapter: PlatformAdapter,
    module_kinds: Iterable[str],
    *,
    explicit: bool,
) -> None:
    kinds = {str(value).strip() for value in module_kinds if str(value).strip()}
    unsupported = sorted(kinds - adapter.deterministic_module_kinds)
    if not unsupported:
        return
    prefix = "명시한 target" if explicit else "선택된 target"
    raise SpecValidationError(
        f"{prefix} {adapter.minecraft_version}/{adapter.loader}의 현재 코드 어댑터가 "
        f"아직 지원하지 않는 생성 종류가 있습니다: {unsupported}. "
        "다른 버전으로 가장하거나 깨지는 소스를 생성하지 않습니다."
    )
