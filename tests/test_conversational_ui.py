from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.validator import ProjectValidator
from minecraft_mod_ai.webui import (
    _clarification_questions,
    _new_execution_root,
    _render_plan,
    create_demo,
)


def test_vague_mod_request_does_not_invent_content_or_boss() -> None:
    proposal = HeuristicPlanner().plan(
        "메이플스토리 느낌의 모드를 만들어줘. 스킬 표현과 맵 퀄리티가 중요해."
    )

    assert proposal.spec.contents == ()
    assert proposal.spec.boss is None
    assert proposal.spec.arena is None
    assert {request.capability for request in proposal.deferred_requests} == {
        "skill_system",
        "custom_map",
    }

    questions = _clarification_questions(proposal.requested_prompt, proposal)
    rendered = _render_plan(proposal, questions)
    assert "스킬과 동작 표현" in rendered
    assert "맵과 월드" in rendered
    assert "보스" not in rendered
    assert "schema_version" not in rendered
    assert "approval_hash" not in rendered
    assert "sha256:" not in rendered
    assert "승인 해시" not in rendered


def test_named_world_parts_are_preserved_without_becoming_an_arena() -> None:
    proposal = HeuristicPlanner().plan(
        "마을, 필드, 던전이 모두 이어지는 모드가 필요해"
    )

    assert proposal.spec.boss is None
    assert proposal.spec.arena is None
    assert proposal.spec.contents == ()
    assert {request.capability for request in proposal.deferred_requests} == {
        "village_map",
        "field_map",
        "dungeon_map",
    }
    rendered = _render_plan(
        proposal,
        _clarification_questions(proposal.requested_prompt, proposal),
    )
    for label in ("마을", "필드", "던전"):
        assert label in rendered


def test_mob_request_is_not_silently_changed_into_a_boss() -> None:
    proposal = HeuristicPlanner().plan("숲을 돌아다니는 사슴 몹과 3D 모델을 만들어줘")

    assert proposal.spec.boss is None
    assert proposal.spec.arena is None
    assert {
        request.capability for request in proposal.deferred_requests
    } >= {"custom_entity", "general_3d_assets"}


def test_dragon_boss_is_not_silently_changed_into_a_humanoid() -> None:
    proposal = HeuristicPlanner().plan(
        "비행하며 원거리 화염을 쏘는 드래곤 보스를 만들어줘"
    )

    assert proposal.spec.boss is None
    assert {
        request.capability for request in proposal.deferred_requests
    } >= {"unsupported_boss_shape", "unsupported_boss_combat"}


def test_latest_revision_can_remove_boss_without_removing_requested_arena() -> None:
    brief = (
        "서리 보스와 41x41 아레나, 아이템 하나를 만들어줘\n"
        "보스 빼고 아레나는 유지해"
    )
    proposal = HeuristicPlanner().plan(brief)
    rendered = _render_plan(proposal, _clarification_questions(brief, proposal))

    assert proposal.spec.boss is None
    assert proposal.spec.arena is not None
    assert proposal.spec.arena.radius == 20
    assert "보스:" not in rendered
    assert "아레나" in rendered
    assert "41×41" in rendered


def test_latest_revision_wins_for_theme_counts_and_arena_scale() -> None:
    proposal = HeuristicPlanner().plan(
        "Create 2 maple items and a 41x41 arena.\n"
        "Change that to 3 frost items and make the arena small."
    )

    assert proposal.spec.mod_id == "frost_works"
    assert len(proposal.spec.contents) == 3
    assert proposal.spec.arena is not None
    assert proposal.spec.arena.radius == 8


def test_arena_scale_is_not_taken_from_a_different_map_zone() -> None:
    proposal = HeuristicPlanner().plan(
        "65x65 아레나와 33x33 던전을 계획해줘"
    )

    assert proposal.spec.arena is not None
    assert proposal.spec.arena.radius == 32


def test_english_do_not_add_boss_is_respected() -> None:
    proposal = HeuristicPlanner().plan(
        "Create one maple item and a 41x41 arena. Do not add a boss."
    )

    assert proposal.spec.boss is None
    assert proposal.spec.arena is not None


def test_explicit_arena_can_be_generated_without_a_boss(tmp_path: Path) -> None:
    proposal = HeuristicPlanner().plan(
        "41x41 대형 아레나와 단풍 아이템 하나를 만들어줘"
    )
    assert proposal.spec.boss is None
    assert proposal.spec.arena is not None
    assert proposal.spec.arena.radius == 20

    generated = FabricProjectGenerator().generate(proposal.spec, tmp_path / "project")
    report = ProjectValidator().validate(generated.root, proposal.spec)
    assert report.passed, report.to_dict()

    function_path = (
        generated.root
        / "src/main/resources/data"
        / proposal.spec.mod_id
        / "functions"
        / f"build_{proposal.spec.arena.arena_id}.mcfunction"
    )
    assert "summon " not in function_path.read_text(encoding="utf-8")
    world_ir = json.loads(
        (
            generated.root
            / ".minecraft_ai/world"
            / f"{proposal.spec.arena.arena_id}.world_design.json"
        ).read_text(encoding="utf-8")
    )
    assert world_ir["navigation"]["required_paths"] == [["entry", "map_center"]]
    assert world_ir["spawn"] is None


def test_ui_uses_server_state_without_raw_json_or_hash_controls(tmp_path: Path) -> None:
    demo = create_demo(output_root=tmp_path / "output")
    components = demo.config["components"]

    assert all(component["type"] != "json" for component in components)
    assert any(component["type"] == "state" for component in components)
    rendered_config = json.dumps(demo.config, ensure_ascii=False, default=str)
    assert "승인 대상 해시" not in rendered_config
    assert "승인 해시 재입력" not in rendered_config
    assert "검토할 제안서" not in rendered_config


def test_each_ui_execution_gets_a_new_output_root(tmp_path: Path) -> None:
    proposal = MinecraftModPipeline().plan("단풍 아이템 하나를 만들어줘")
    first_root = _new_execution_root(tmp_path)
    second_root = _new_execution_root(tmp_path)
    assert first_root != second_root

    pipeline = MinecraftModPipeline()
    first = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=first_root,
        build=False,
    )
    second = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=second_root,
        build=False,
    )
    assert first.status == "SOURCE_READY"
    assert second.status == "SOURCE_READY"
