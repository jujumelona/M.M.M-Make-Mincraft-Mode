from __future__ import annotations

import re
from typing import Any, Iterable

from .complete_spec import ProductionModule


_VISIBLE_SECTION_ITEMS = 24
_VISIBLE_TEXT_CHARS = 2000


_KIND_LABELS_KO = {
    "item": "아이템",
    "block": "블록",
    "tool": "도구",
    "weapon": "무기와 전투 장비",
    "armor": "방어구",
    "food": "음식과 요리",
    "crop": "작물과 재배",
    "fluid": "유체",
    "machine": "기계와 자동화",
    "recipe": "제작법",
    "effect": "상태 효과",
    "enchantment": "마법 부여",
    "entity": "생물과 캐릭터",
    "boss": "보스 전투",
    "npc": "NPC와 상호작용",
    "quest": "퀘스트",
    "class": "직업",
    "skill": "스킬",
    "economy": "경제",
    "shop": "상점",
    "gui": "게임 화면과 메뉴",
    "networking": "멀티플레이 동기화",
    "party": "파티",
    "guild": "길드",
    "command": "게임 명령",
    "structure": "건축물과 장소",
    "biome": "생태 지역",
    "dimension": "차원",
    "world_event": "월드 이벤트",
    "advancement": "도전 과제",
    "loot": "보상과 전리품",
    "audio": "음악과 효과음",
    "integration": "다른 모드와의 연동",
    "custom_java": "요청에 맞춘 전용 게임 로직",
}
_KIND_LABELS_EN = {
    "item": "items",
    "block": "blocks",
    "tool": "tools",
    "weapon": "weapons and combat equipment",
    "armor": "armor",
    "food": "food and cooking",
    "crop": "crops and farming",
    "fluid": "fluids",
    "machine": "machines and automation",
    "recipe": "recipes",
    "effect": "status effects",
    "enchantment": "enchantments",
    "entity": "creatures and characters",
    "boss": "boss encounters",
    "npc": "NPCs and interactions",
    "quest": "quests",
    "class": "classes",
    "skill": "skills",
    "economy": "economy",
    "shop": "shops",
    "gui": "game screens and menus",
    "networking": "multiplayer synchronization",
    "party": "parties",
    "guild": "guilds",
    "command": "game commands",
    "structure": "structures and places",
    "biome": "biomes",
    "dimension": "dimensions",
    "world_event": "world events",
    "advancement": "advancements",
    "loot": "rewards and loot",
    "audio": "music and sound effects",
    "integration": "integration with other mods",
    "custom_java": "custom gameplay logic for the request",
}


def render_complete_plan(
    *,
    requested_prompt: str,
    game_design: dict[str, Any],
    modules: Iterable[ProductionModule],
    acceptance_tests: Iterable[str],
) -> str:
    """Render the user-facing game plan without internal protocol machinery."""

    korean = bool(re.search(r"[가-힣]", requested_prompt))
    title = _bounded_text(
        str(game_design.get("title", "")).strip(),
        korean=korean,
    ) or (
        "새 Minecraft 모드" if korean else "New Minecraft Mod"
    )
    pitch = _bounded_text(
        str(game_design.get("pitch", "")).strip(),
        korean=korean,
    )
    core_loop = _bounded_list(
        _strings(game_design.get("core_loop")),
        korean=korean,
    )
    progression = _bounded_list(
        _strings(game_design.get("progression")),
        korean=korean,
    )
    tests = _bounded_list(_strings(acceptance_tests), korean=korean)
    module_values = tuple(modules)
    integration_lines = _bounded_list(
        _mod_context_lines(game_design.get("mod_context"), korean=korean),
        korean=korean,
    )
    system_lines = _bounded_list(
        list(
            dict.fromkeys(
                [
                    *_production_outline_lines(
                        game_design.get("production_outline")
                    ),
                    *_system_lines(
                        game_design.get("modules"),
                        (module.kind for module in module_values),
                        korean=korean,
                    ),
                ]
            )
        ),
        korean=korean,
    )
    research_lines = _bounded_list(
        _research_lines(game_design.get("_research_brief")),
        korean=korean,
    )
    technology_lines = _bounded_list(
        _technology_lines(
            game_design.get("_technology_radar"),
            korean=korean,
        ),
        korean=korean,
    )
    quality_lines = _quality_lines(
        game_design.get("_production_contract"),
        korean=korean,
    )

    if korean:
        lines = [f"“{title}”는 이런 게임으로 만들겠습니다."]
        if pitch:
            lines.extend(("", f"한 줄 기획: {pitch}"))
        lines.extend(("", "플레이 흐름"))
        lines.extend(_numbered(core_loop, empty="요청에 맞춰 핵심 플레이 흐름을 먼저 확정합니다."))
        lines.extend(("", "성장과 진행"))
        lines.extend(_numbered(progression, empty="별도 성장 시스템은 요청하지 않은 상태입니다."))
        if integration_lines:
            lines.extend(("", "마인크래프트 연동 범위"))
            lines.extend(f"- {value}" for value in integration_lines)
        lines.extend(("", "제작 범위"))
        lines.extend(f"- {value}" for value in system_lines)
        if research_lines:
            lines.extend(("", "조사와 근거 확인"))
            lines.extend(f"- {value}" for value in research_lines)
        if technology_lines:
            lines.extend(("", "AI·음성 기술 설계"))
            lines.extend(f"- {value}" for value in technology_lines)
        if quality_lines:
            lines.extend(("", "완성 기준"))
            lines.extend(f"- {value}" for value in quality_lines)
        lines.extend(("", "완성 확인"))
        lines.extend(f"- {value}" for value in (tests or ["요청한 기능을 게임 안에서 직접 확인합니다."]))
        lines.extend(
            (
                "",
                "이 방향으로 만들까요? 규모, 분위기, 시스템, 장소를 바꾸고 싶으면 "
                "원하는 대로 말해 주세요.",
            )
        )
    else:
        lines = [f'I will build “{title}” as the following game.']
        if pitch:
            lines.extend(("", f"Pitch: {pitch}"))
        lines.extend(("", "Player loop"))
        lines.extend(_numbered(core_loop, empty="We will define the core loop from the request."))
        lines.extend(("", "Progression"))
        lines.extend(_numbered(progression, empty="No separate progression system was requested."))
        if integration_lines:
            lines.extend(("", "Minecraft integration scope"))
            lines.extend(f"- {value}" for value in integration_lines)
        lines.extend(("", "Production scope"))
        lines.extend(f"- {value}" for value in system_lines)
        if research_lines:
            lines.extend(("", "Research and evidence"))
            lines.extend(f"- {value}" for value in research_lines)
        if technology_lines:
            lines.extend(("", "AI and speech architecture"))
            lines.extend(f"- {value}" for value in technology_lines)
        if quality_lines:
            lines.extend(("", "Completion standard"))
            lines.extend(f"- {value}" for value in quality_lines)
        lines.extend(("", "Acceptance"))
        lines.extend(f"- {value}" for value in (tests or ["Verify every requested feature in game."]))
        lines.extend(
            (
                "",
                "Should I build this direction? Tell me any change in scale, tone, systems, "
                "or places.",
            )
        )
    return "\n".join(lines).strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Iterable) or isinstance(value, (dict, bytes)):
        return []
    return [
        text
        for item in value
        if (text := str(item).strip())
    ]


def _bounded_text(value: str, *, korean: bool) -> str:
    if len(value) <= _VISIBLE_TEXT_CHARS:
        return value
    remaining = len(value) - _VISIBLE_TEXT_CHARS
    suffix = (
        f" … (나머지 {remaining}자는 저장된 계획에 보존됨)"
        if korean
        else f" … ({remaining} more characters remain in the stored plan)"
    )
    return value[:_VISIBLE_TEXT_CHARS].rstrip() + suffix


def _bounded_list(values: list[str], *, korean: bool) -> list[str]:
    bounded = [
        _bounded_text(value, korean=korean)
        for value in values[:_VISIBLE_SECTION_ITEMS]
    ]
    remaining = len(values) - len(bounded)
    if remaining > 0:
        bounded.append(
            (
                f"그 밖의 {remaining}개 항목은 저장된 계획에 그대로 보존됨"
                if korean
                else f"{remaining} more entries remain in the stored plan"
            )
        )
    return bounded


def _numbered(values: list[str], *, empty: str) -> list[str]:
    selected = values or [empty]
    return [f"{index}. {value}" for index, value in enumerate(selected, start=1)]


def _mod_context_lines(value: Any, *, korean: bool) -> list[str]:
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    for key in ("vanilla_integration", "compatibility_targets"):
        values = value.get(key)
        if isinstance(values, list):
            result.extend(
                str(item).strip() for item in values if str(item).strip()
            )
    return list(dict.fromkeys(result))


def _system_lines(
    design_modules: Any,
    kinds: Iterable[str],
    *,
    korean: bool,
) -> list[str]:
    result: list[str] = []
    if isinstance(design_modules, list):
        for item in design_modules:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            reason = str(item.get("reason", "")).strip()
            if reason:
                result.append(reason)
            elif name:
                result.append(name)
    labels = _KIND_LABELS_KO if korean else _KIND_LABELS_EN
    result.extend(
        labels[kind]
        for kind in sorted(set(kinds))
        if kind in labels
    )
    result = list(dict.fromkeys(result))
    if result:
        return result
    return ["요청 범위를 대화로 확정" if korean else "Confirm scope in conversation"]


def _production_outline_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope", "")).strip()
        if scope:
            result.append(scope)
    return result


def _research_lines(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    domains = value.get("domains")
    if not isinstance(domains, list):
        return []
    result: list[str] = []
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        objective = str(domain.get("objective") or "").strip()
        requirements = domain.get("requirements")
        requirement = ""
        if isinstance(requirements, list):
            requirement = "; ".join(
                str(item).strip()
                for item in requirements
                if str(item).strip()
            )
        if objective and requirement:
            result.append(f"{objective}: {requirement}")
        elif objective or requirement:
            result.append(objective or requirement)
    return list(dict.fromkeys(result))


def _technology_lines(value: Any, *, korean: bool) -> list[str]:
    if not isinstance(value, dict):
        return []
    requirements = value.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return []
    labels_ko = {
        "ai_inference": "AI 추론",
        "agent_tool_use": "에이전트 도구 사용",
        "speech_recognition": "음성 인식",
        "voice_activity_detection": "발화 감지",
        "speech_synthesis": "음성 합성",
        "voice_adaptation": "동의된 음성 적응",
        "voice_conversion": "동의된 음성 변환",
        "voice_transport": "음성 전송",
        "language_intersection": "전체 음성 경로의 지원 언어",
        "translation": "번역",
    }
    labels_en = {
        "ai_inference": "AI inference",
        "agent_tool_use": "agent tool use",
        "speech_recognition": "speech recognition",
        "voice_activity_detection": "voice activity detection",
        "speech_synthesis": "speech synthesis",
        "voice_adaptation": "consented voice adaptation",
        "voice_conversion": "consented voice conversion",
        "voice_transport": "voice transport",
        "language_intersection": "full-pipeline language support",
        "translation": "translation",
    }
    topology_ko = {
        "in_process_java": "Java 내부",
        "local_sidecar": "로컬 보조 프로세스",
        "remote_api": "동의된 원격 API",
        "offline_build_tool": "빌드 시 오프라인 도구",
    }
    topology_en = {
        "in_process_java": "in-process Java",
        "local_sidecar": "local sidecar",
        "remote_api": "consented remote API",
        "offline_build_tool": "offline build-time tool",
    }
    labels = labels_ko if korean else labels_en
    topology_labels = topology_ko if korean else topology_en
    result: list[str] = []
    for item in requirements:
        if not isinstance(item, dict):
            continue
        capability = str(item.get("capability_kind") or "").strip()
        label = labels.get(capability, capability.replace("_", " "))
        topologies = [
            topology_labels.get(str(name), str(name))
            for name in item.get("allowed_topologies", [])
            if str(name).strip()
        ]
        fallback = str(item.get("deterministic_fallback") or "").strip()
        if korean:
            line = f"{label}: 호환성·라이선스·성능 확인 후 " + ", ".join(topologies)
            if fallback:
                line += f" 중에서 선택하고, 실패하면 {fallback}"
        else:
            line = f"{label}: choose among " + ", ".join(topologies)
            line += " only after compatibility, license, and performance checks"
            if fallback:
                line += f"; fallback: {fallback}"
        result.append(line)
    return list(dict.fromkeys(result))


def _quality_lines(value: Any, *, korean: bool) -> list[str]:
    if not isinstance(value, dict):
        return []
    stats = value.get("catalog_stats")
    dimensions = value.get("quality_dimension_catalog")
    if not isinstance(stats, dict) or not isinstance(dimensions, list):
        return []
    requirement_count = stats.get("requirements")
    check_count = stats.get("acceptance_tests")
    if type(requirement_count) is not int or type(check_count) is not int:
        return []
    titles = [
        str(item.get("title") or "").strip()
        for item in dimensions
        if isinstance(item, dict) and str(item.get("title") or "").strip()
    ]
    if korean:
        return [
            f"요청에서 추적한 요구사항 {requirement_count}개와 관찰 가능한 확인 항목 {check_count}개를 끝까지 연결합니다.",
            "관련 품질 영역마다 현재 결과에 묶인 독립 검증 증거가 있어야 완성으로 판정합니다.",
        ]
    return [
        f"Trace all {requirement_count} request-derived requirements through {check_count} observable checks.",
        "Completion requires fresh, artifact-bound independent evidence for every relevant quality dimension"
        + (f": {', '.join(titles)}." if titles else "."),
    ]
