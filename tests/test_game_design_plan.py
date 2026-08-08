from __future__ import annotations

import re
from pathlib import Path

from minecraft_mod_ai import HeuristicPlanner, ModAISession


LARGE_PLAN_SECTIONS = (
    "프로젝트 정의",
    "핵심 게임플레이 루프",
    "시스템/콘텐츠",
    "제작 마일스톤",
    "검증/릴리스",
)


def _session(tmp_path: Path) -> ModAISession:
    return ModAISession(output_root=tmp_path / "output", planner=HeuristicPlanner())


def _markdown_section(text: str, title: str) -> str:
    match = re.search(rf"(?ms)^#+\s*{re.escape(title)}\s*$\n(.*?)(?=^#+\s|\Z)", text)
    assert match is not None, f"missing plan section: {title!r}\n{text}"
    return match.group(1)


def _assert_reader_facing_plan(text: str) -> None:
    lowered = text.lower()
    for forbidden in ("schema_version", "approval_hash", "approval hash", "승인 해시", "sha256:", "json"):
        assert forbidden not in lowered
    assert re.search(r"\brag\b", lowered) is None
    assert re.search(r"\bmcp\b", lowered) is None


def test_one_item_request_gets_a_concise_small_scope_game_plan(tmp_path: Path) -> None:
    reply = _session(tmp_path).plan("Create exactly one frost item and no other content.")
    assert reply.ready_to_build
    assert len(reply.proposal.spec.contents) == 1
    assert reply.proposal.spec.contents[0].kind.value == "item"
    assert reply.proposal.spec.boss is None

    plan = reply.message
    assert "프로젝트 정의" in plan
    assert "핵심 게임플레이 루프" in plan
    assert "시스템/콘텐츠" in plan
    assert "아이템" in _markdown_section(plan, "시스템/콘텐츠")
    assert "소규모" in plan
    assert "대규모" not in plan
    assert "월드/레벨 설계" not in plan
    assert len(plan) < 2_400
    _assert_reader_facing_plan(plan)


def test_large_mod_request_gets_feature_and_release_sections_without_map_design(
    tmp_path: Path,
) -> None:
    reply = _session(tmp_path).plan(
        "대규모 RPG 모드를 계획해줘. 근접 공격·원거리 투사체·회복 스킬, 퀘스트, 직업, "
        "NPC, 3D 장비, 사운드와 메뉴를 넣어줘."
    )
    plan = reply.message
    for title in LARGE_PLAN_SECTIONS:
        assert title in plan
    assert "월드/레벨 설계" not in plan
    systems_plan = _markdown_section(plan, "시스템/콘텐츠")
    assert "스킬" in systems_plan
    assert reply.proposal.spec.boss is None
    _assert_reader_facing_plan(plan)


def test_ambiguous_multi_system_request_asks_for_scale_and_first_playable_scope(
    tmp_path: Path,
) -> None:
    reply = _session(tmp_path).plan("스킬, 퀘스트와 직업이 있는 RPG 모드를 만들어줘.")
    assert not reply.ready_to_build
    questions = "\n".join((*reply.questions, reply.message))
    assert "프로젝트 규모" in questions
    assert "첫 플레이 가능 범위" in questions
    _assert_reader_facing_plan(reply.message)


def test_map_reference_does_not_add_an_arena_or_world_section(tmp_path: Path) -> None:
    reply = _session(tmp_path).plan("Create one frost item and a 41x41 arena. Do not add a boss.")
    assert reply.proposal.spec.boss is None
    assert len(reply.proposal.spec.contents) == 1
    assert "월드/레벨 설계" not in reply.message
    _assert_reader_facing_plan(reply.message)
