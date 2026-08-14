from __future__ import annotations

import html
import logging
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .importer import ExistingProjectReport, inspect_existing_project_archive
from .intent import latest_intent_event
from .pipeline import MinecraftModPipeline
from .planner import (
    HeuristicPlanner,
    LocalTransformersPlanner,
    OpenAICompatiblePlanner,
)
from .spec import Proposal


LOGGER = logging.getLogger(__name__)

_START_MESSAGE = (
    "만들고 싶은 모드를 자유롭게 말해 주세요. 간단한 모드는 짧게, 대규모 모드는 "
    "게임플레이·모드 시스템·리소스·제작 단계까지 계획합니다. 말하지 않은 콘텐츠는 "
    "임의로 추가하지 않고, 부족한 내용은 질문하겠습니다."
)

_FOCUS_LABELS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("스킬", "skill", "마법", "magic"), "스킬과 동작 표현"),
    (("퀘스트", "quest"), "퀘스트"),
    (("직업", "class", "클래스"), "직업"),
    (("npc", "엔피시"), "NPC"),
    (("아이템", "item", "도구", "weapon", "무기"), "아이템"),
    (("블록", "block", "광석", "ore"), "블록"),
    (("3d", "모델", "model", "모델링"), "3D 모델"),
    (("보스", "boss"), "보스"),
    (("몹", "mob", "엔티티", "entity"), "몹과 엔티티"),
    (("gui", "메뉴", "화면"), "화면과 메뉴"),
    (("음성", "voice", "tts", "asr"), "음성"),
    (("사운드", "sound", "음악", "music"), "사운드와 음악"),
    (("애니메이션", "animation"), "애니메이션"),
    (("차원", "dimension"), "차원"),
)

_CAPABILITY_LABELS = {
    "skill_system": "스킬 시스템과 표현",
    "general_3d_assets": "사용자 지정 3D 대상",
    "custom_entity": "몹과 엔티티",
    "unsupported_boss_shape": "요청한 보스 형태",
    "unsupported_boss_combat": "요청한 보스 전투 방식",
    "item_count_limit": "아이템 수",
    "block_count_limit": "블록 수",
    "quest_system": "퀘스트",
    "class_system": "직업",
    "npc_system": "NPC",
    "custom_gui": "사용자 화면과 메뉴",
    "custom_animation": "사용자 지정 애니메이션",
    "custom_dimension": "사용자 지정 차원",
    "voice": "음성",
    "sound_system": "사운드와 음악",
    "creative_brief": "모드 구성",
}

_CAPABILITY_FOCUS_LABELS = {
    "skill_system": "스킬과 동작 표현",
    "general_3d_assets": "3D 모델",
    "custom_entity": "몹과 엔티티",
    "unsupported_boss_shape": "보스",
    "unsupported_boss_combat": "보스",
    "item_count_limit": "아이템",
    "block_count_limit": "블록",
    "quest_system": "퀘스트",
    "class_system": "직업",
    "npc_system": "NPC",
    "custom_gui": "화면과 메뉴",
    "custom_animation": "3D 모델",
    "voice": "음성",
    "sound_system": "사운드와 음악",
    "custom_dimension": "차원",
}

_APPROVAL_MESSAGES = {
    "네",
    "응",
    "좋아",
    "진행",
    "진행해",
    "만들어",
    "만들어줘",
    "이대로",
    "이대로만들어",
    "이대로만들어줘",
    "이대로진행",
    "확정",
    "yes",
    "ok",
    "okay",
    "go",
}


def _initial_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": _START_MESSAGE}]


def _history_with(
    history: list[dict[str, str]] | None,
    role: str,
    content: str,
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for message in history or []:
        if isinstance(message, dict):
            copied.append(
                {
                    "role": str(message.get("role", "assistant")),
                    "content": str(message.get("content", "")),
                }
            )
        else:
            copied.append(
                {
                    "role": str(getattr(message, "role", "assistant")),
                    "content": str(getattr(message, "content", "")),
                }
            )
    copied.append({"role": role, "content": content})
    return copied


def _existing_input_markdown(report: ExistingProjectReport | None) -> str:
    if report is None:
        return "새 모드를 만드는 대화입니다."
    source_note = "소스 포함" if report.has_sources else "소스 없음"
    return (
        f"기존 모드 수정: **{report.archive_name}** · "
        f"{report.mod_id or '모드 이름 확인 불가'} · {source_note}"
    )


def _merge_brief(current: str, message: str) -> str:
    message = message.strip()
    if not current.strip():
        return message
    return f"{current.rstrip()}\n{message}"


def _explicit_focus(brief: str) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for words, label in _FOCUS_LABELS:
        event = latest_intent_event(
            brief,
            words,
            cascade_removals=(),
        )
        if event is not None and event.requested:
            matches.append((event.start, label))
    ordered: list[str] = []
    for _, label in sorted(matches):
        if label not in ordered:
            ordered.append(label)
    return tuple(ordered)


def _allows_defaults(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        phrase in lowered
        for phrase in ("알아서", "추천해", "네가 정해", "ai가 정해", "auto decide")
    )


def _has_skill_behavior(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        word in lowered
        for word in (
            "투사체",
            "광역",
            "버프",
            "회복",
            "이동기",
            "소환",
            "근접",
            "원거리",
            "projectile",
            "area",
            "buff",
            "heal",
            "dash",
            "summon",
            "melee",
            "ranged",
        )
    )


def _skill_behavior_summary(brief: str) -> str:
    lowered = brief.lower()
    behaviors = (
        (("근접", "melee"), "근접"),
        (("원거리", "ranged"), "원거리"),
        (("투사체", "projectile"), "투사체"),
        (("광역", "area"), "광역"),
        (("버프", "buff"), "버프"),
        (("회복", "heal"), "회복"),
        (("이동기", "dash"), "이동"),
        (("소환", "summon"), "소환"),
    )
    selected = [
        label
        for words, label in behaviors
        if any(word in lowered for word in words)
    ]
    return " · ".join(selected)


def _project_scope(brief: str, proposal: Proposal) -> tuple[str, bool]:
    lowered = brief.lower()
    scope_patterns = (
        (
            (
                r"(?:대규모|초대형|개큰)(?:\s+\S+){0,3}\s+(?:모드|프로젝트|규모|스케일)",
                r"(?:모드|프로젝트|규모|스케일)(?:\s+\S+){0,3}\s+(?:대규모|초대형|개큰)",
                r"큰\s*스케일",
                r"(?<![a-z0-9_])(?:huge|massive|large-scale)"
                r"(?:\s+[a-z0-9_-]+){0,3}\s+"
                r"(?:mod|project|scope|overhaul|conversion)(?![a-z0-9_])",
                r"(?<![a-z0-9_])(?:mod|project|scope|overhaul)"
                r"(?:\s+[a-z0-9_-]+){0,3}\s+"
                r"(?:huge|massive|large-scale)(?![a-z0-9_])",
                r"(?<![a-z0-9_])total\s+conversion(?![a-z0-9_])",
            ),
            "대규모 프로젝트",
        ),
        (
            (
                r"중규모(?:\s+\S+){0,3}\s+(?:모드|프로젝트|규모|스케일)",
                r"(?:모드|프로젝트|규모|스케일)(?:\s+\S+){0,3}\s+중규모",
                r"(?<![a-z0-9_])(?:medium-scale|medium-sized)"
                r"(?:\s+[a-z0-9_-]+){0,3}\s+(?:mod|project|scope)(?![a-z0-9_])",
            ),
            "중규모 프로젝트",
        ),
        (
            (
                r"(?:간단한?|소규모|미니)(?:\s+\S+){0,3}\s+(?:모드|프로젝트)",
                r"(?:모드|프로젝트)(?:\s+\S+){0,3}\s+(?:간단|소규모|미니)",
                r"(?<![a-z0-9_])(?:simple|small)"
                r"(?:\s+[a-z0-9_-]+){0,3}\s+(?:mod|project|scope)(?![a-z0-9_])",
            ),
            "소규모 프로젝트",
        ),
    )
    candidates = [
        (match.start(), label)
        for patterns, label in scope_patterns
        for pattern in patterns
        for match in re.finditer(pattern, lowered, re.IGNORECASE)
    ]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1], True

    focus = set(_explicit_focus(brief))
    if (
        focus
        and focus <= {"아이템", "블록"}
        and len(proposal.spec.contents) <= 3
        and not proposal.deferred_requests
    ):
        return "현재 요청 기준 소규모 단일 기능", True
    return "아직 정하지 않음", False


def _has_core_loop(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        word in lowered
        for word in (
            "핵심 루프",
            "플레이 루프",
            "시작해서",
            "성장",
            "진행",
            "목표",
            "반복",
            "탐험하고",
            "준비하고",
            "core loop",
            "progression",
            "starts by",
            "goal is",
        )
    )


def _has_first_playable_scope(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        phrase in lowered
        for phrase in (
            "1차 범위",
            "첫 플레이",
            "첫 버전",
            "버티컬 슬라이스",
            "mvp",
            "first playable",
            "vertical slice",
            "first version",
        )
    )


def _has_project_name(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        phrase in lowered
        for phrase in (
            "이름은",
            "이름:",
            "모드명",
            "프로젝트명",
            "called ",
            "named ",
            "title:",
        )
    )


def _has_content_quantity(brief: str, *, kind: str) -> bool:
    lowered = brief.lower()
    nouns = ("아이템", "item") if kind == "item" else ("블록", "block")
    number = (
        r"(?:\d+|한|하나|두|둘|세|셋|네|넷|여러|"
        r"one|two|three|four|five|six|seven|eight)"
    )
    return any(
        re.search(
            rf"{number}(?:\s+[\w가-힣-]+){{0,4}}\s+{noun}s?\b",
            lowered,
        )
        or re.search(rf"{number}\s*(?:개(?:의)?\s*)?{noun}", lowered)
        or re.search(rf"{noun}\s*{number}\s*개?", lowered)
        for noun in nouns
    )


def _has_boss_shape(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        word in lowered
        for word in (
            "인간형",
            "휴머노이드",
            "언데드",
            "골렘",
            "드래곤",
            "용형",
            "짐승형",
            "humanoid",
            "biped",
            "undead",
            "golem",
            "dragon",
            "beast",
        )
    )


def _has_boss_combat(brief: str) -> bool:
    lowered = brief.lower()
    return any(
        word in lowered
        for word in (
            "근접",
            "원거리",
            "투사체",
            "마법",
            "소환",
            "돌진",
            "광역",
            "단계",
            "패턴",
            "melee",
            "ranged",
            "projectile",
            "magic",
            "summon",
            "charge",
            "phase",
            "pattern",
        )
    )


def _clarification_questions(brief: str, proposal: Proposal) -> tuple[str, ...]:
    if _allows_defaults(brief):
        return ()
    focus = _explicit_focus(brief)
    questions: list[str] = []
    if not focus:
        questions.append("모드에 반드시 들어가야 할 기능을 자유롭게 말해 주세요.")
    scope_label, scope_is_set = _project_scope(brief, proposal)
    complex_request = (
        scope_label == "대규모 프로젝트"
        or len(focus) >= 3
        or any(
            label in focus
            for label in (
                "스킬과 동작 표현",
                "퀘스트",
                "직업",
                "NPC",
            )
        )
    )
    if complex_request and not scope_is_set:
        questions.append(
            "프로젝트 규모가 간단한 모드인지 중·대규모 프로젝트인지와 "
            "첫 플레이 가능 범위에 어디까지 넣을지 말해 주세요."
        )
    elif (
        scope_label == "대규모 프로젝트"
        and not _has_first_playable_scope(brief)
    ):
        questions.append(
            "대규모 전체 계획 중 첫 플레이 가능 범위에 넣을 구역과 시스템을 "
            "말해 주세요."
        )
    if complex_request and not _has_core_loop(brief):
        questions.append(
            "플레이어가 시작해서 성장하고 목표를 달성하는 핵심 플레이 흐름을 "
            "말해 주세요."
        )
    if "스킬과 동작 표현" in focus and not _has_skill_behavior(brief):
        questions.append("스킬이 실제 게임에서 어떻게 동작해야 하는지 말해 주세요.")
    if "3D 모델" in focus and not any(
        label in focus
        for label in ("아이템", "블록", "몹과 엔티티", "보스", "NPC")
    ):
        questions.append("어떤 대상을 3D로 만들지 말해 주세요.")
    if "아이템" in focus and not _has_content_quantity(brief, kind="item"):
        questions.append("아이템의 수와 각각의 플레이 역할을 말해 주세요.")
    if "블록" in focus and not _has_content_quantity(brief, kind="block"):
        questions.append("블록의 수와 각각의 용도를 말해 주세요.")
    if "보스" in focus and not _has_boss_shape(brief):
        questions.append("요청한 보스의 형태를 말해 주세요.")
    if "보스" in focus and not _has_boss_combat(brief):
        questions.append("요청한 보스의 전투 방식과 주요 패턴을 말해 주세요.")
    deferred_capabilities = {
        request.capability for request in proposal.deferred_requests
    }
    if "item_count_limit" in deferred_capabilities:
        questions.append("한 제작 단계의 아이템 수를 1~8개로 정해 주세요.")
    if "block_count_limit" in deferred_capabilities:
        questions.append("한 제작 단계의 블록 수를 1~8개로 정해 주세요.")
    return tuple(questions)


def _buildable(proposal: Proposal, questions: tuple[str, ...]) -> bool:
    has_generated_content = bool(
        proposal.spec.contents
        or proposal.spec.boss is not None
    )
    return has_generated_content and not proposal.deferred_requests and not questions


def _render_plan(proposal: Proposal, questions: tuple[str, ...]) -> str:
    brief_lines = [
        html.escape(line.strip())
        for line in proposal.requested_prompt.splitlines()
        if line.strip()
    ]
    quoted_brief = "\n".join(f"> {line}" for line in brief_lines)
    focus = _explicit_focus(proposal.requested_prompt)
    scope_label, _ = _project_scope(proposal.requested_prompt, proposal)
    lines = [
        "# 게임 개발 계획서",
        "",
        "## 프로젝트 정의",
        "",
        "- 프로젝트 이름: "
        + (
            "요청에서 지정됨"
            if _has_project_name(proposal.requested_prompt)
            else "아직 정하지 않음"
        ),
        f"- 제작 규모: {scope_label}",
        "- 실행 대상: Minecraft Java 1.20.1 · Fabric · Java 17",
        "- 원문 요구:",
        quoted_brief,
        "",
        "요청하지 않은 콘텐츠는 계획에 추가하지 않았습니다.",
    ]

    if (
        scope_label.startswith("현재 요청 기준 소규모")
        or scope_label == "소규모 프로젝트"
    ):
        first_playable = "현재 요청 전체"
    elif _has_first_playable_scope(proposal.requested_prompt):
        first_playable = "요청에서 확인됨"
    else:
        first_playable = "아직 정하지 않음"
    lines.extend(
        (
            "",
            "## 핵심 게임플레이 루프",
            "",
            (
                "- 시작·성장·목표 흐름: 요청에서 확인됨"
                if _has_core_loop(proposal.requested_prompt)
                else "- 시작·성장·목표 흐름: 아직 정하지 않음"
            ),
            f"- 첫 플레이 가능 범위: {first_playable}",
        )
    )

    lines.extend(("", "## 시스템/콘텐츠", ""))
    if focus:
        lines.extend(
            (
                "| 작업 영역 | 기획 상태 | 현재 구현 연결 |",
                "|---|---|---|",
            )
        )
        deferred_focus = {
            _CAPABILITY_FOCUS_LABELS.get(request.capability, request.capability)
            for request in proposal.deferred_requests
        }
        item_count = sum(
            content.kind.value == "item" for content in proposal.spec.contents
        )
        block_count = sum(
            content.kind.value == "block" for content in proposal.spec.contents
        )
        for label in focus:
            if label == "아이템":
                detail = f"{item_count}개" if item_count else "수량·역할 미정"
                implementation = "생성 가능" if item_count else "세부 정보 필요"
            elif label == "블록":
                detail = f"{block_count}개" if block_count else "수량·용도 미정"
                implementation = "생성 가능" if block_count else "세부 정보 필요"
            elif label == "보스":
                detail = (
                    "형태·전투 정보 확인"
                    if _has_boss_shape(proposal.requested_prompt)
                    and _has_boss_combat(proposal.requested_prompt)
                    else "형태 또는 전투 방식 미정"
                )
                implementation = (
                    "지원 형태 검토"
                    if proposal.spec.boss is not None
                    else "전용 구현 연결 필요"
                )
            elif label == "스킬과 동작 표현":
                detail = _skill_behavior_summary(proposal.requested_prompt) or "동작 미정"
                implementation = "전용 구현 연결 필요"
            else:
                detail = "요청에 포함"
                implementation = (
                    "전용 구현 연결 필요"
                    if label in deferred_focus
                    else "세부 정보 검토"
                )
            lines.append(f"| {label} | {detail} | {implementation} |")
    else:
        lines.append("- 반드시 들어갈 시스템이나 콘텐츠가 아직 정해지지 않았습니다.")

    if "3D 모델" in focus or any(
        label in focus for label in ("화면과 메뉴",)
    ):
        lines.extend(
            (
                "",
                "## 아트/3D/사용자 경험",
                "",
                "- 3D 제작 대상: "
                + (
                    "요청에서 확인됨"
                    if "3D 모델" in focus
                    and any(
                        label in focus
                        for label in ("아이템", "블록", "몹과 엔티티", "보스", "NPC")
                    )
                    else "아직 정하지 않음"
                ),
                "- 모델·텍스처·애니메이션은 각각 게임 내 동작 검증 항목과 연결합니다.",
            )
        )

    lines.extend(("", "## 제작 마일스톤", ""))
    if scope_label == "대규모 프로젝트":
        lines.extend(
            (
                "1. 프리프로덕션: 핵심 루프, 전체 범위, 기술 제약 확정",
                "2. 버티컬 슬라이스: 대표 구역과 대표 시스템을 끝까지 플레이 가능하게 제작",
                "3. 시스템·월드 생산: 승인된 구역과 기능을 작업 단위별로 구현",
                "4. 콘텐츠·3D 통합: 모델, 텍스처, 동작, 데이터 연결",
                "5. 멀티플레이·성능·회귀 검증",
                "6. 릴리스 후보 빌드와 설치 검증",
            )
        )
    elif scope_label.startswith("현재 요청 기준 소규모") or scope_label == "소규모 프로젝트":
        lines.extend(
            (
                "1. 요구 확정",
                "2. 콘텐츠 구현과 게임 내 동작 연결",
                "3. 빌드·GameTest·릴리스 검사",
            )
        )
    else:
        lines.extend(
            (
                "1. 전체 규모와 첫 플레이 가능 범위 확정",
                "2. 대표 기능 버티컬 슬라이스 제작",
                "3. 검증 결과를 보고 다음 제작 범위 확정",
            )
        )

    lines.extend(
        (
            "",
            "## 검증/릴리스",
            "",
            "- 요청한 기능과 생성 결과가 서로 일치해야 합니다.",
            "- 소스 검증, Fabric 빌드, GameTest가 통과해야 합니다.",
            "- 실패한 빌드는 설치용 JAR로 제공하지 않습니다.",
        )
    )
    if scope_label == "대규모 프로젝트":
        lines.append("- 버티컬 슬라이스 합격 후 다음 제작 범위를 엽니다.")

    if questions:
        lines.extend(("", "## 기획 확정을 위해 더 필요한 답"))
        lines.extend(f"- {question}" for question in questions)
        lines.extend(("", "아래 입력창에서 답하거나 다른 수정 내용을 말해 주세요."))
    elif _buildable(proposal, questions):
        lines.extend(
            (
                "",
                "## 제작 확인",
                "",
                "**이 계획으로 첫 구현을 만들까요?**",
                "바꿀 내용은 계속 말해 주세요. 괜찮으면 `이대로 만들기`를 누르거나 "
                "`진행해`라고 입력하세요.",
            )
        )
    else:
        lines.extend(
            (
                "",
                "아직 바로 만들 수 있는 계획이 아닙니다. 필요한 내용을 계속 말해 주세요.",
            )
        )
    return "\n".join(lines)


def _is_approval_message(message: str) -> bool:
    normalized = re.sub(r"[\s.!?,~]+", "", message.strip().lower())
    return normalized in _APPROVAL_MESSAGES


def _new_execution_root(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return output_root / "runs" / f"{timestamp}-{secrets.token_hex(4)}"


def _result_message(result: Any) -> str:
    if result.release_ready:
        return (
            "## 만들기 완료\n\n"
            "빌드와 검사가 끝났습니다. 아래에서 release ZIP을 내려받으세요."
        )
    if result.status == "SOURCE_READY":
        return (
            "## 소스 생성 완료\n\n"
            "소스만 생성했습니다. 설치용 JAR가 필요한 경우 빌드 옵션으로 다시 실행하세요."
        )
    return (
        "## 만들기 중 문제가 발생했습니다\n\n"
        "설치용 JAR는 만들지 않았습니다. 계획을 수정하거나 다시 시도해 주세요."
    )


def create_demo(
    *,
    output_root: Path,
    local_model: bool = False,
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    existing_input: str | Path | None = None,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("UI extras are missing. Install with: pip install -e '.[ui]'") from exc

    output_root = output_root.resolve()
    existing_path = Path(existing_input).resolve() if existing_input is not None else None
    existing_report = (
        inspect_existing_project_archive(existing_path)
        if existing_path is not None
        else None
    )
    if api_base_url or api_model or api_key:
        if not (api_base_url and api_model and api_key):
            raise ValueError(
                "외부 AI API를 사용하려면 주소, 모델 이름, API 키가 모두 필요합니다."
            )
        planner = OpenAICompatiblePlanner(
            base_url=api_base_url,
            model=api_model,
            api_key=api_key,
        )
    elif local_model:
        planner = LocalTransformersPlanner()
    else:
        planner = HeuristicPlanner()
    pipeline = MinecraftModPipeline(planner=planner)

    def execute_proposal(
        proposal: Proposal | None,
        history: list[dict[str, str]] | None,
        source_only: bool,
    ) -> tuple[list[dict[str, str]], Proposal | None, Any, str, str | None]:
        if not isinstance(proposal, Proposal):
            response = "먼저 대화로 계획을 만들어 주세요."
            return (
                _history_with(history, "assistant", response),
                None,
                gr.update(interactive=False),
                response,
                None,
            )
        questions = _clarification_questions(proposal.requested_prompt, proposal)
        if not _buildable(proposal, questions):
            response = (
                "아직 정하지 않았거나 구현과 연결되지 않은 내용이 있습니다. "
                "대화에서 필요한 내용을 더 정해 주세요."
            )
            return (
                _history_with(history, "assistant", response),
                proposal,
                gr.update(interactive=False),
                response,
                None,
            )
        try:
            result = pipeline.execute(
                proposal,
                approval_hash=proposal.approval_hash,
                output_root=_new_execution_root(output_root),
                build=not source_only,
                run_gametest=not source_only,
                existing_input=existing_path,
            )
            response = _result_message(result)
            return (
                _history_with(history, "assistant", response),
                proposal,
                gr.update(interactive=False),
                response,
                result.release_zip,
            )
        except Exception:
            LOGGER.exception("Mod generation failed")
            response = (
                "만들기에 실패했습니다. 기존 작업과 겹치지 않는 새 작업 공간으로 "
                "다시 시도할 수 있습니다. 계획을 수정하거나 다시 눌러 주세요."
            )
            return (
                _history_with(history, "assistant", response),
                proposal,
                gr.update(interactive=True),
                response,
                None,
            )

    def send_message(
        message: str,
        history: list[dict[str, str]] | None,
        brief: str,
        proposal: Proposal | None,
        source_only: bool,
    ) -> tuple[
        str,
        list[dict[str, str]],
        str,
        Proposal | None,
        Any,
        str,
        str | None,
    ]:
        message = message.strip()
        if not message:
            return (
                "",
                history or _initial_history(),
                brief,
                proposal,
                gr.update(),
                "",
                None,
            )
        updated_history = _history_with(history, "user", message)
        if _is_approval_message(message) and isinstance(proposal, Proposal):
            (
                executed_history,
                executed_proposal,
                button_update,
                result_text,
                release_path,
            ) = execute_proposal(proposal, updated_history, source_only)
            return (
                "",
                executed_history,
                brief,
                executed_proposal,
                button_update,
                result_text,
                release_path,
            )

        updated_brief = _merge_brief(brief, message)
        try:
            updated_proposal = pipeline.plan(
                updated_brief,
                existing_input=existing_path,
            )
            questions = _clarification_questions(updated_brief, updated_proposal)
            response = _render_plan(updated_proposal, questions)
            updated_history = _history_with(updated_history, "assistant", response)
            return (
                "",
                updated_history,
                updated_brief,
                updated_proposal,
                gr.update(interactive=_buildable(updated_proposal, questions)),
                "",
                None,
            )
        except Exception:
            LOGGER.exception("Planning failed")
            response = (
                "요청을 계획으로 정리하지 못했습니다. 내용을 조금 다르게 말해 주세요. "
                "기존 계획은 실행하지 않았습니다."
            )
            updated_history = _history_with(updated_history, "assistant", response)
            return (
                "",
                updated_history,
                brief,
                proposal,
                gr.update(interactive=False),
                response,
                None,
            )

    def approve_current(
        proposal: Proposal | None,
        history: list[dict[str, str]] | None,
        source_only: bool,
    ) -> tuple[list[dict[str, str]], Proposal | None, Any, str, str | None]:
        return execute_proposal(proposal, history, source_only)

    def reset_conversation() -> tuple[
        str,
        list[dict[str, str]],
        str,
        None,
        Any,
        str,
        None,
    ]:
        return (
            "",
            _initial_history(),
            "",
            None,
            gr.update(interactive=False),
            "",
            None,
        )

    with gr.Blocks(title="M.M.M Make Mincraft Mode") as demo:
        gr.Markdown(
            """
# M.M.M Make Mincraft Mode

원하는 내용을 말하면 AI가 자연어 계획으로 정리합니다.
규모에 맞는 게임 개발 계획을 검토하고 계속 수정한 뒤,
마음에 들면 **이대로 만들기**를 누르세요.
"""
        )
        gr.Markdown(_existing_input_markdown(existing_report))

        chatbot_kwargs: dict[str, Any] = {
            "value": _initial_history(),
            "label": "모드 제작 대화",
            "height": 520,
            "layout": "bubble",
            "allow_tags": False,
        }
        if "type" in __import__("inspect").signature(gr.Chatbot).parameters:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(**chatbot_kwargs)
        message = gr.Textbox(
            label="AI에게 말하기",
            placeholder="만들고 싶은 모드나 수정할 내용을 자유롭게 적으세요.",
            lines=3,
        )
        with gr.Row():
            send_button = gr.Button("보내기", variant="primary")
            reset_button = gr.Button("새 대화")
        with gr.Accordion("고급 옵션", open=False):
            source_only = gr.Checkbox(
                label="소스만 생성하고 빌드는 나중에 하기",
                value=False,
            )
        build_button = gr.Button(
            "이대로 만들기",
            variant="primary",
            interactive=False,
        )
        result_status = gr.Markdown()
        release_file = gr.File(label="완성된 release ZIP")

        brief_state = gr.State("")
        proposal_state = gr.State(None)

        event_inputs = [
            message,
            chatbot,
            brief_state,
            proposal_state,
            source_only,
        ]
        event_outputs = [
            message,
            chatbot,
            brief_state,
            proposal_state,
            build_button,
            result_status,
            release_file,
        ]
        def _bind_event(event_target, fn, **kwargs):
            import inspect
            sig = inspect.signature(event_target)
            if "api_visibility" in sig.parameters:
                kwargs["api_visibility"] = "private"
            return event_target(fn, **kwargs)

        _bind_event(send_button.click, send_message, inputs=event_inputs, outputs=event_outputs)
        _bind_event(message.submit, send_message, inputs=event_inputs, outputs=event_outputs)
        _bind_event(
            build_button.click,
            approve_current,
            inputs=[proposal_state, chatbot, source_only],
            outputs=[
                chatbot,
                proposal_state,
                build_button,
                result_status,
                release_file,
            ],
        )
        _bind_event(reset_button.click, reset_conversation, inputs=[], outputs=event_outputs)
    return demo


def launch(
    *,
    output_root: str | Path = "mmm-output",
    local_model: bool = False,
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
    existing_input: str | Path | None = None,
    auth: tuple[str, str] | None = None,
) -> Any:
    demo = create_demo(
        output_root=Path(output_root),
        local_model=local_model,
        api_base_url=api_base_url,
        api_model=api_model,
        api_key=api_key,
        existing_input=existing_input,
    )
    return demo.launch(share=share, server_name=server_name, auth=auth)
