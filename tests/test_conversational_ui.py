from __future__ import annotations

import json
from dataclasses import replace

from minecraft_mod_ai.conversational_ui import _new_execution_root, create_demo
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner


def test_ui_starts_in_conversational_mode(tmp_path) -> None:
    demo = create_demo(output_root=tmp_path / "output")
    components = demo.config["components"]
    assert any(component["type"] == "chatbot" for component in components)


def test_ui_hides_advanced_controls_until_requested(tmp_path) -> None:
    demo = create_demo(output_root=tmp_path / "output")
    components = demo.config["components"]
    rendered_config = json.dumps(demo.config, ensure_ascii=False, default=str)
    assert "고급 설정" in rendered_config
    assert "승인 대상 해시" not in rendered_config
    assert "승인 해시 재입력" not in rendered_config
    assert "검토할 제안서" not in rendered_config


def test_ui_does_not_expose_raw_proposal_json(tmp_path) -> None:
    demo = create_demo(output_root=tmp_path / "output")
    rendered_config = json.dumps(demo.config, ensure_ascii=False, default=str)
    assert "검토할 제안서" not in rendered_config
    assert "raw json" not in rendered_config.casefold()


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
    # reviewed synthetic deterministic target explicitly and recompute the immutable
    # proposal receipt instead of retaining a hash for the pre-binding payload.
    proposal = HeuristicPlanner().plan("단풍 아이템 하나를 만들어줘")
    proposal = replace(
        proposal,
        spec=replace(proposal.spec, platform=synthetic_platform_lock),
        approval_hash="",
    ).with_hash()
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
