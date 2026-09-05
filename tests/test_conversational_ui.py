from __future__ import annotations

import json
from dataclasses import replace

from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.webui import (
    _clarification_questions,
    _explicit_focus,
    _new_execution_root,
    _render_plan,
    create_demo,
)


def test_vague_mod_request_does_not_invent_content_or_map_scope() -> None:
    proposal = HeuristicPlanner().plan(
        "메이플스토리 느낌의 모드를 만들어줘. 스킬 표현과 맵 퀄리티가 중요해."
    )

    assert proposal.spec.contents == ()
    assert proposal.spec.boss is None
    assert {request.capability for request in proposal.deferred_requests} == {
        "skill_system"
    }

    questions = _clarification_questions(proposal.requested_prompt, proposal)
    rendered = _render_plan(proposal, questions)
    assert "스킬과 동작 표현" in rendered
    assert "월드/레벨 설계" not in rendered
    assert "schema_version" not in rendered
    assert "approval_hash" not in rendered
    assert "sha256:" not in rendered
    assert "승인 해시" not in rendered


def test_map_words_are_ignored_until_a_mod_feature_is_requested() -> None:
    proposal = HeuristicPlanner().plan(
        "마을, 필드, 던전, 아레나가 이어지는 모드가 필요해"
    )

    assert proposal.spec.boss is None
    assert proposal.spec.contents == ()
    assert {request.capability for request in proposal.deferred_requests} == {
        "creative_brief"
    }
    assert _explicit_focus(proposal.requested_prompt) == ()


def test_mob_request_is_not_silently_changed_into_a_boss() -> None:
    proposal = HeuristicPlanner().plan("숲을 돌아다니는 사슴 몹과 3D 모델을 만들어줘")

    assert proposal.spec.boss is None
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


def test_map_scale_text_does_not_change_requested_item_scope() -> None:
    proposal = HeuristicPlanner().plan(
        "Create 2 maple items near a 41x41 arena. Change that to 3 frost items."
    )

    assert proposal.spec.mod_id == "frost_works"
    assert len(proposal.spec.contents) == 3
    assert _explicit_focus(proposal.requested_prompt) == ("아이템",)


def test_ui_uses_server_state_without_raw_json_or_hash_controls(tmp_path) -> None:
    demo = create_demo(output_root=tmp_path / "output")
    components = demo.config["components"]

    assert all(component["type"] != "json" for component in components)
    assert any(component["type"] == "state" for component in components)
    rendered_config = json.dumps(demo.config, ensure_ascii=False, default=str)
    assert "승인 대상 해시" not in rendered_config
    assert "승인 해시 재입력" not in rendered_config
    assert "검토할 제안서" not in rendered_config


def test_each_ui_execution_gets_a_new_output_root(tmp_path, synthetic_platform_lock) -> None:
    # This test owns output-root isolation, not target discovery. Proposal approval now
    # validates the immutable platform lock before generation starts, so bind the
    # reviewed synthetic deterministic target explicitly instead of relying on a later
    # generator fixture to repair an unresolved proposal.
    proposal = HeuristicPlanner().plan("단풍 아이템 하나를 만들어줘")
    proposal = replace(
        proposal,
        spec=replace(proposal.spec, platform=synthetic_platform_lock),
    )
    first_root = _new_execution_root(tmp_path)
    second_root = _new_execution_root(tmp_path)
    assert first_root != second_root

    pipeline = MinecraftModPipeline(planner=HeuristicPlanner())
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
