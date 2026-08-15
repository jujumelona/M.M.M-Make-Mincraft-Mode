from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Protocol

from .capabilities import capability_manifest_hash
from .intent import (
    CountIntent,
    is_requested,
    latest_intent_event,
    requested_count,
)
from .knowledge import evidence_for_mvp, evidence_snapshot_hash
from .spec import (
    BossSpec,
    ContentKind,
    ContentSpec,
    DeferredRequest,
    ModSpec,
    PlatformLock,
    Proposal,
    ProposalStatus,
    SpecValidationError,
)


class Planner(Protocol):
    def plan(self, prompt: str) -> Proposal: ...


THEMES: tuple[tuple[tuple[str, ...], str, str, str, str], ...] = (
    (("메이플", "maple"), "maple", "Maple", "단풍", "#e67e22"),
    (("달빛", "moon", "lunar"), "moon", "Moonlight", "달빛", "#a6adc8"),
    (("얼음", "서리", "ice", "frost"), "frost", "Frost", "서리", "#89dceb"),
    (("불", "화염", "fire", "flame"), "ember", "Ember", "불꽃", "#fab387"),
    (("번개", "전기", "lightning", "storm"), "storm", "Storm", "폭풍", "#f9e2af"),
    (("그림자", "어둠", "shadow", "dark"), "shadow", "Shadow", "그림자", "#585b70"),
    (("숲", "자연", "forest", "nature"), "grove", "Grove", "숲", "#a6e3a1"),
    (("바다", "물", "ocean", "water"), "tide", "Tide", "파도", "#74c7ec"),
)

ADVANCED_CAPABILITIES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("스킬", "skill", "마법", "magic"), "skill_system", "future"),
    (("몹", "mob", "엔티티", "entity"), "custom_entity", "future"),
    (("퀘스트", "quest"), "quest_system", "future"),
    (("직업", "class", "클래스"), "class_system", "future"),
    (("npc", "엔피시"), "npc_system", "future"),
    (("gui", "화면", "메뉴"), "custom_gui", "phase-3"),
    (("음성", "voice", "tts", "asr"), "voice", "phase-5"),
    (("사운드", "sound", "음악", "music"), "sound_system", "future"),
    (("animation", "애니메이션"), "custom_animation", "phase-5"),
    (("차원", "dimension"), "custom_dimension", "phase-5"),
)

ITEM_TERMS = ("아이템", "item", "결정", "crystal", "도구", "tool")
BLOCK_TERMS = ("블록", "block", "광석", "ore", "벽돌", "brick")


def _contains(
    prompt: str,
    words: tuple[str, ...],
    *,
    cascade_removals: tuple[str, ...] = (),
) -> bool:
    return is_requested(
        prompt,
        words,
        cascade_removals=cascade_removals,
    )


def _theme(prompt: str) -> tuple[str, str, str, str]:
    selected: tuple[int, tuple[str, str, str, str]] | None = None
    for words, stem, english, korean, color in THEMES:
        event = latest_intent_event(prompt, words)
        if event is None or not event.requested:
            continue
        candidate = (event.start, (stem, english, korean, color))
        if selected is None or candidate[0] > selected[0]:
            selected = candidate
    return selected[1] if selected is not None else (
        "crafted",
        "Crafted",
        "제작",
        "#cba6f7",
    )


def _count(
    prompt: str,
    terms: tuple[str, ...],
    default: int,
) -> CountIntent:
    return requested_count(
        prompt,
        terms=terms,
        default=default,
    )


class HeuristicPlanner:
    """Offline semantic planner; platform selection is deliberately deferred."""

    def plan(self, prompt: str) -> Proposal:
        prompt = prompt.strip()
        if not prompt:
            raise SpecValidationError("프롬프트를 입력해 주세요.")

        stem, english_theme, korean_theme, color = _theme(prompt)
        asks_item = _contains(prompt, ITEM_TERMS)
        asks_block = _contains(prompt, BLOCK_TERMS)
        item_count_intent = (
            _count(prompt, ITEM_TERMS, 1)
            if asks_item
            else CountIntent(count=0, explicit=False)
        )
        block_count_intent = (
            _count(prompt, BLOCK_TERMS, 1)
            if asks_block
            else CountIntent(count=0, explicit=False)
        )
        item_count = item_count_intent.count
        block_count = block_count_intent.count
        contents: list[ContentSpec] = []

        item_suffixes = ("crystal", "shard", "relic", "gem", "dust", "core", "token", "charm")
        item_english = ("Crystal", "Shard", "Relic", "Gem", "Dust", "Core", "Token", "Charm")
        item_korean = ("결정", "조각", "유물", "보석", "가루", "핵", "토큰", "부적")
        for index in range(item_count):
            suffix = (
                item_suffixes[index]
                if index < len(item_suffixes)
                else f"item_{index + 1:06d}"
            )
            english_label = (
                item_english[index]
                if index < len(item_english)
                else f"Item {index + 1}"
            )
            korean_label = (
                item_korean[index]
                if index < len(item_korean)
                else f"아이템 {index + 1}"
            )
            contents.append(
                ContentSpec(
                    content_id=f"{stem}_{suffix}",
                    kind=ContentKind.ITEM,
                    display_name_en=f"{english_theme} {english_label}",
                    display_name_ko=f"{korean_theme} {korean_label}",
                    color=color,
                )
            )

        block_base = "ore" if _contains(prompt, ("광석", "ore")) else "crystal_block"
        block_suffixes = (block_base, "bricks", "tiles", "pillar", "lamp", "stone", "glass", "slab")
        block_english = ("Ore" if block_base == "ore" else "Crystal Block", "Bricks", "Tiles", "Pillar", "Lamp", "Stone", "Glass", "Slab")
        block_korean = ("광석" if block_base == "ore" else "결정 블록", "벽돌", "타일", "기둥", "등불", "돌", "유리", "반 블록")
        for index in range(block_count):
            suffix = (
                block_suffixes[index]
                if index < len(block_suffixes)
                else f"block_{index + 1:06d}"
            )
            english_label = (
                block_english[index]
                if index < len(block_english)
                else f"Block {index + 1}"
            )
            korean_label = (
                block_korean[index]
                if index < len(block_korean)
                else f"블록 {index + 1}"
            )
            contents.append(
                ContentSpec(
                    content_id=f"{stem}_{suffix}",
                    kind=ContentKind.BLOCK,
                    display_name_en=f"{english_theme} {english_label}",
                    display_name_ko=f"{korean_theme} {korean_label}",
                    color=color,
                )
            )

        asks_boss = _contains(prompt, ("보스", "boss"))
        asks_3d = _contains(prompt, ("3d", "모델링", "model", "모델"))
        unsupported_boss_shape = asks_boss and _contains(
            prompt,
            (
                "드래곤",
                "용형",
                "짐승형",
                "골렘",
                "dragon",
                "beast",
                "quadruped",
                "golem",
            ),
        )
        unsupported_boss_combat = asks_boss and _contains(
            prompt,
            (
                "원거리",
                "투사체",
                "마법",
                "소환",
                "비행",
                "페이즈",
                "ranged",
                "projectile",
                "magic",
                "summon",
                "flying",
                "phase",
            ),
        )
        boss = (
            BossSpec(
                entity_id=f"{stem}_warden",
                display_name_en=f"{english_theme} Warden",
                display_name_ko=f"{korean_theme} 수호자",
                primary_color=color,
                secondary_color="#cba6f7",
            )
            if asks_boss and not unsupported_boss_shape
            else None
        )
        deferred: list[DeferredRequest] = []
        for keywords, capability, phase in ADVANCED_CAPABILITIES:
            if _contains(prompt, keywords):
                deferred.append(
                    DeferredRequest(
                        capability=capability,
                        reason=(
                            "이 기능은 아직 실행 가능한 생성기에 연결되지 않았습니다. "
                            "요청을 다른 기능으로 바꾸지 않고 그대로 보류합니다."
                        ),
                        suggested_phase=phase,
                    )
                )
        if item_count_intent.overflow is not None:
            deferred.append(
                DeferredRequest(
                    capability="item_count_limit",
                    reason=(
                        f"아이템 {item_count_intent.overflow}개 요청은 한 번에 지원하는 "
                        "최대 8개를 넘습니다. 수를 줄이거나 제작 단계를 나눠야 합니다."
                    ),
                    suggested_phase="planning",
                )
            )
        if block_count_intent.overflow is not None:
            deferred.append(
                DeferredRequest(
                    capability="block_count_limit",
                    reason=(
                        f"블록 {block_count_intent.overflow}개 요청은 한 번에 지원하는 "
                        "최대 8개를 넘습니다. 수를 줄이거나 제작 단계를 나눠야 합니다."
                    ),
                    suggested_phase="planning",
                )
            )
        if unsupported_boss_shape:
            deferred.append(
                DeferredRequest(
                    capability="unsupported_boss_shape",
                    reason=(
                        "요청한 보스 형태를 현재 인간형 모델로 바꾸지 않습니다. "
                        "형태 전용 모델·충돌체·애니메이션 구현이 필요합니다."
                    ),
                    suggested_phase="future",
                )
            )
        if unsupported_boss_combat:
            deferred.append(
                DeferredRequest(
                    capability="unsupported_boss_combat",
                    reason=(
                        "요청한 전투 방식을 기본 근접 전투로 바꾸지 않습니다. "
                        "전용 행동·동기화·GameTest 구현이 필요합니다."
                    ),
                    suggested_phase="future",
                )
            )

        if asks_3d and boss is None:
            deferred.append(
                DeferredRequest(
                    capability="general_3d_assets",
                    reason="어떤 대상을 3D로 만들지 대화로 정해야 합니다.",
                    suggested_phase="future",
                )
            )
        if not contents and boss is None and not deferred:
            deferred.append(
                DeferredRequest(
                    capability="creative_brief",
                    reason="사용자가 원하는 모드 기능을 대화로 정해야 합니다.",
                    suggested_phase="planning",
                )
            )

        mod_id = f"{stem}_works"
        spec = ModSpec(
            mod_id=mod_id,
            mod_name=f"{english_theme} Works",
            package_name=f"ai.minecraft.generated.{mod_id}",
            version="1.0.0",
            summary=(
                f"{prompt} 요청에서 명시된"
                + (" 아이템·블록" if contents else "")
                + (" 보스" if boss else "")
                + ("·3D" if boss and asks_3d else "")
                + " 구성"
            ),
            contents=tuple(contents),
            boss=boss,
            platform=PlatformLock(),
        )
        evidence_query = (
            f"{prompt} project build metadata dependency "
            + ("boss entity gametest " if boss else "item block recipe data generation ")
        )
        evidence_sources = evidence_for_mvp(evidence_query)
        proposal = Proposal(
            schema_version="minecraft-mod-ai/proposal-v1",
            proposal_version=1,
            status=ProposalStatus.AWAITING_APPROVAL,
            requested_prompt=prompt,
            spec=spec,
            assumptions=(
                "플랫폼 target은 사용자 제약, 기존 프로젝트 target, 실행 가능한 provider 근거를 바탕으로 중앙 optimizer가 선택합니다.",
                "텍스처는 라이선스 문제가 없는 결정론적 PNG를 생성합니다.",
                "사용자의 실제 Minecraft 월드에는 쓰지 않습니다.",
            ),
            exclusions=(
                "자동 게시, 실제 사용자 월드 수정, 외부 서버 명령",
                "자유형 애니메이션·임의 차원·GUI·음성 생성",
                "사용자 승인 없이 게임 내 명령을 실행하거나 보스를 소환하는 동작",
            ),
            deferred_requests=tuple(deferred),
            acceptance_tests=(
                "승인 해시가 일치해야만 프로젝트 파일을 생성한다.",
                "모든 JSON/PNG/리소스 참조와 ID가 결정론적 검증을 통과한다.",
                "provider가 선택한 toolchain build가 exit code 0으로 끝난 경우에만 JAR를 제공한다.",
                "JAR가 ZIP 형식이며 선택된 loader metadata, 클래스, 생성 리소스를 포함한다.",
                "영어와 한국어 번역 키가 모든 생성 콘텐츠를 포함한다.",
                "보스 요청 시 서버 권위 엔티티·보스바·loot·spawn egg와 실행 검증을 포함한다.",
                "지원되는 보스 3D 요청 시 bbmodel·texture·runtime renderer를 포함한다.",
            ),
            evidence_sources=evidence_sources,
            evidence_snapshot_hash=evidence_snapshot_hash(evidence_sources),
            capability_manifest_hash=capability_manifest_hash(),
            imported_source_snapshot_hash="",
            risk_approvals=(
                "빌드는 승인 후 선택된 provider가 검증한 공식 플랫폼·게임 저장소에 네트워크로 접근할 수 있습니다.",
            ),
        ).with_hash()
        # Platform selection deliberately happens after semantic planning. Exact
        # proposal validation therefore belongs to the resolver/approval boundary.
        return proposal


class LocalTransformersPlanner:
    """Compatibility wrapper around the role-based model registry.

    Direct model identifiers and fallback planners are intentionally rejected so
    every local backend uses ``config/model_registry.yaml`` and failures remain
    visible to the caller.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        fallback: Planner | None = None,
        max_new_tokens: int | None = None,
        profile: str = "t4_local",
    ) -> None:
        if model_id is not None:
            raise SpecValidationError(
                "Direct model_id overrides are disabled. Configure the model in "
                "config/model_registry.yaml and select a profile."
            )
        if fallback is not None:
            raise SpecValidationError(
                "Silent or automatic planner fallback is disabled."
            )
        if max_new_tokens is not None:
            raise SpecValidationError(
                "Per-call max_new_tokens overrides are disabled. Configure the role "
                "limit in config/model_registry.yaml."
            )
        self.profile = profile
        self.last_backend = f"role-router:{profile}"

    def plan(self, prompt: str) -> Proposal:
        from .routed_planner import RoutedPlanner

        return RoutedPlanner(profile=self.profile).plan(prompt)


class OpenAICompatiblePlanner:
    """Planner for an explicitly configured HTTPS chat-completions API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 90,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        api_key = api_key.strip()
        if not base_url.startswith("https://"):
            raise ValueError("외부 AI API 주소는 https://로 시작해야 합니다.")
        if not model:
            raise ValueError("외부 AI API 모델 이름을 입력해 주세요.")
        if not api_key:
            raise ValueError("외부 AI API 키가 없습니다.")
        if not 5 <= timeout_seconds <= 300:
            raise ValueError("API timeout_seconds must be between 5 and 300.")
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def plan(self, prompt: str) -> Proposal:
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _planner_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=request_body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mmm-make-mincraft-mode/0.1",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                payload_bytes = response.read(2 * 1024 * 1024 + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("외부 AI API 호출에 실패했습니다.") from exc
        if len(payload_bytes) > 2 * 1024 * 1024:
            raise RuntimeError("외부 AI API 응답이 허용 크기를 초과했습니다.")
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("외부 AI API 응답 형식이 올바르지 않습니다.") from exc
        if not isinstance(content, str):
            raise RuntimeError("외부 AI API가 텍스트 계획을 반환하지 않았습니다.")
        model_data = _extract_json_object(content)
        return _proposal_from_model_data(prompt, model_data)


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecValidationError("The local model did not return a JSON object.")


def _planner_system_prompt() -> str:
    return """
You are a Minecraft Java mod requirements planner. Output exactly one JSON object
and no markdown. Do not choose or assume a Minecraft version, loader, mappings,
Java version, dependency coordinate, or build-tool version. Platform selection is
owned by the host optimizer after semantic planning. You may plan only simple items
and blocks. Never emit Java, paths, shell commands, entities, GUI, worldgen,
networking, or publication actions.

JSON contract:
{
  "mod_id": "lowercase_snake_case",
  "mod_name": "English name",
  "package_name": "ai.minecraft.generated.lowercase_name",
  "summary": "short Korean or English summary",
  "contents": [
    {
      "content_id": "lowercase_snake_case",
      "kind": "item or block",
      "display_name_en": "English",
      "display_name_ko": "Korean",
      "color": "#RRGGBB",
      "recipe": true
    }
  ],
  "deferred_capabilities": ["unsupported capability names"]
}
Create 0-8 content entries. Include only items or blocks the user explicitly
requested. For vague or unsupported requests, return an empty contents list and
preserve the request in deferred_capabilities. Never invent a boss, item, block,
map type, gameplay system, platform, or version.
""".strip()


def _proposal_from_model_data(prompt: str, data: dict[str, Any]) -> Proposal:
    exact_keys = {
        "mod_id",
        "mod_name",
        "package_name",
        "summary",
        "contents",
        "deferred_capabilities",
    }
    if set(data) != exact_keys:
        raise SpecValidationError(
            f"Local model returned unexpected fields: {sorted(set(data) ^ exact_keys)}"
        )
    model_contents = tuple(
        ContentSpec(
            content_id=item["content_id"],
            kind=ContentKind(item["kind"]),
            display_name_en=item["display_name_en"],
            display_name_ko=item["display_name_ko"],
            color=item["color"],
            recipe=item["recipe"],
        )
        for item in data["contents"]
    )
    base = HeuristicPlanner().plan(prompt)
    heuristic_spec = base.spec
    remaining_by_kind = {
        kind: sum(content.kind is kind for content in heuristic_spec.contents)
        for kind in ContentKind
    }
    accepted_contents: list[ContentSpec] = []
    for content in model_contents:
        if remaining_by_kind.get(content.kind, 0) <= 0:
            continue
        accepted_contents.append(content)
        remaining_by_kind[content.kind] -= 1
    contents = tuple(accepted_contents)
    spec = ModSpec(
        mod_id=data["mod_id"],
        mod_name=data["mod_name"],
        package_name=data["package_name"],
        version="1.0.0",
        summary=data["summary"],
        contents=contents,
        boss=heuristic_spec.boss,
        platform=PlatformLock(),
    )
    deferred = base.deferred_requests
    proposal = replace(base, spec=spec, deferred_requests=deferred, approval_hash="")
    return proposal.with_hash()
