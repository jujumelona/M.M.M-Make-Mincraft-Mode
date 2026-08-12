from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .platform_catalog import (
    PlatformAdapter,
    adapter_for_target,
    supported_minecraft_versions,
)
from .spec import PlatformLock, Proposal, SpecValidationError


_VERSION_RE = re.compile(r"(?<!\d)(1\.\d{1,2}(?:\.\d{1,2})?|\d{2,4}\.\d+(?:\.\d+)?)(?!\d)")
_FABRIC_RE = re.compile(r"\bfabric\b|패브릭", re.IGNORECASE)
_NEOFORGE_RE = re.compile(r"\bneoforge\b|네오포지", re.IGNORECASE)
_FORGE_RE = re.compile(r"(?<!neo)\bforge\b|(?<!네오)포지", re.IGNORECASE)
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

    @property
    def lock(self) -> PlatformLock:
        return lock_from_adapter(self.adapter)

    def to_dict(self) -> dict[str, Any]:
        mappings_kind = (
            "mojang" if self.adapter.yarn_mappings == "mojang" else "yarn"
        )
        return {
            "schema_version": "mmm/platform-selection-v3",
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
    """Resolve an immutable target without a Minecraft-version source allowlist.

    Explicit user targets and existing-project targets are hard constraints. A Revise
    target mismatch is permitted only when the request explicitly asks for a version
    migration/port. For a new unpinned project, the host first resolves complete
    official toolchain profiles and the central planner AI chooses only among them.
    """

    text = str(prompt or "")
    explicit_version = _explicit_minecraft_version(text)
    explicit_loader = _explicit_loader(text)
    migration_requested = bool(existing_version and _MIGRATION_RE.search(text))

    if explicit_loader and explicit_loader != "fabric":
        raise SpecValidationError(
            f"요청한 로더 {explicit_loader!r}는 아직 실행 가능한 provider가 없습니다. "
            "지원되는 척 다른 로더 코드를 생성하지 않습니다."
        )

    if explicit_version:
        try:
            adapter = adapter_for_target(explicit_version, explicit_loader or "fabric")
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
        _require_supported_kinds(adapter, module_kinds, explicit=True)
        return PlatformSelection(
            adapter=adapter,
            source=(
                "user_explicit_migration_target"
                if migration_requested
                else "user_explicit_target"
            ),
            reason=(
                f"기존 프로젝트를 사용자가 명시한 Minecraft {adapter.minecraft_version} "
                f"{adapter.loader} target으로 migration합니다."
                if migration_requested
                else f"사용자가 Minecraft {adapter.minecraft_version} {adapter.loader}을 명시했습니다."
            ),
            explicit_version=True,
            explicit_loader=bool(explicit_loader),
            migration_requested=migration_requested,
        )

    if existing_version and not migration_requested:
        loader = (existing_loader or "fabric").strip().lower()
        try:
            adapter = adapter_for_target(existing_version, loader)
        except ValueError as exc:
            raise SpecValidationError(
                "기존 프로젝트 target을 보존할 provider가 없습니다: " + str(exc)
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
            migration_requested=False,
        )

    ready = _ready_live_profiles(limit=8)
    if not ready:
        raise SpecValidationError(
            "Fabric의 공식 stable 버전은 발견됐지만 Loader/API/Loom/Java/Gradle까지 "
            "완전히 resolve된 실행 가능한 target이 없습니다."
        )
    by_version = {adapter.minecraft_version: adapter for adapter in ready}
    candidates = tuple(by_version)
    selected_version, ai_reason = _choose_with_central_ai(
        router,
        prompt=text,
        design=design,
        candidates=ready,
    )
    if selected_version not in by_version:
        selected_version = candidates[0]
        ai_reason = (
            "중앙 AI 응답이 host가 완전히 resolve한 후보 밖이어서 가장 최신의 "
            "실행 가능한 공식 stable profile로 fail-closed했습니다."
        )
    adapter = by_version[selected_version]
    _require_supported_kinds(adapter, module_kinds, explicit=False)
    return PlatformSelection(
        adapter=adapter,
        source=(
            "central_ai_migration_over_live_discovery"
            if migration_requested
            else "central_ai_over_live_discovery"
        ),
        reason=ai_reason,
        explicit_version=False,
        explicit_loader=bool(explicit_loader),
        migration_requested=migration_requested,
    )


def _ready_live_profiles(*, limit: int) -> tuple[PlatformAdapter, ...]:
    result: list[PlatformAdapter] = []
    seen: set[str] = set()
    for version in supported_minecraft_versions(loader="fabric"):
        if version in seen:
            continue
        seen.add(version)
        try:
            adapter = adapter_for_target(version, "fabric")
        except ValueError:
            continue
        result.append(adapter)
        if len(result) >= max(1, int(limit)):
            break
    return tuple(result)


def _choose_with_central_ai(
    router: Any | None,
    *,
    prompt: str,
    design: dict[str, Any] | None,
    candidates: tuple[PlatformAdapter, ...],
) -> tuple[str, str]:
    newest = candidates[0].minecraft_version
    if router is None:
        return newest, (
            "중앙 AI router가 없는 API 경로이므로 공식 메타데이터가 완전히 resolve된 "
            f"최신 stable profile Minecraft {newest}을 선택했습니다."
        )

    design_view = design if isinstance(design, dict) else {}
    encoded_design = json.dumps(design_view, ensure_ascii=False, default=str)
    if len(encoded_design) > 12000:
        encoded_design = encoded_design[:12000]
    profiles = [
        {
            "minecraft_version": item.minecraft_version,
            "loader": item.loader,
            "java": item.java_version,
            "mappings": (
                "mojang" if item.yarn_mappings == "mojang" else item.yarn_mappings
            ),
            "fabric_loader": item.fabric_loader,
            "fabric_api": item.fabric_api,
            "loom": item.fabric_loom,
            "gradle": item.gradle,
            "generation_mode": (
                "central_ai_compile_repair"
                if item.source_api_family == "fabric_live_ai"
                else "legacy_optimized_adapter"
            ),
        }
        for item in candidates
    ]
    candidate_versions = tuple(item.minecraft_version for item in candidates)
    request = {
        "task": "choose_minecraft_fabric_target",
        "user_request": prompt,
        "game_design": encoded_design,
        "ready_profiles": profiles,
        "candidate_order": "newest_stable_first_after_full_official_resolution",
        "rules": [
            "Choose exactly one ready_profiles.minecraft_version value.",
            "Prefer the newest ready profile unless the requested mod has a concrete compatibility reason to use an older profile.",
            "Evaluate the supplied Java, mappings, Loader, Fabric API, Loom and Gradle profile; never invent or alter coordinates.",
            "Existing-project preservation and explicit user versions are handled before this decision.",
        ],
        "output": {
            "minecraft_version": "exact ready profile version",
            "reason": "short Korean or English compatibility reason",
        },
    }
    try:
        raw = router.generate_text(
            "planner",
            [
                {
                    "role": "system",
                    "content": (
                        "You are the central platform-selection controller. Select one "
                        "fully host-resolved Fabric target profile. Never invent a version or "
                        "toolchain coordinate. Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            response_format="json",
        )
        payload = json.loads(str(raw).strip())
        selected = str(payload.get("minecraft_version", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        if selected in candidate_versions and reason:
            return selected, reason
    except Exception:
        pass
    return newest, (
        "중앙 AI의 플랫폼 선택 응답을 host가 검증하지 못해, 완전히 resolve된 "
        f"최신 stable profile Minecraft {newest}을 사용했습니다."
    )


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
        if not (
            "Minecraft Java Edition" in value
            or "Minecraft " in value and "Fabric" in value
        )
    ) + (
        (
            f"Target: Minecraft Java {selection.adapter.minecraft_version}, Fabric, "
            f"Java {selection.adapter.java_version}. {selection.reason}"
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
            "provider": "official_live_discovery",
        }
        for version in supported_minecraft_versions(loader="fabric")
    )


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
        raise SpecValidationError(
            f"하나의 프로젝트에 여러 로더가 동시에 명시되었습니다: {unique}"
        )
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
