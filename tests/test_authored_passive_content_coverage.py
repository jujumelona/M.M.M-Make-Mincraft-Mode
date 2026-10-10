"""Regression checks: passive content must not be mistaken for a typed no-op feature."""
from pathlib import Path

from minecraft_mod_ai.final_artifact import (
    _authored_feature_semantic_findings,
    _is_passive_content_only_authored_plan,
)


def _passive_plan():
    return {
        "typed_plan_ir": {
            "initialize": [],
            "functions": [],
            "event_bindings": [],
            "platform_modules": [],
        },
        "structured_sections": {
            "resources_and_ui": {"specification": {"registries": []}},
        },
        "content_design": {
            "modules": [
                {"module_id": "crystal_fragment", "kind": "item", "config": {"display_name": "Crystal Fragment", "stack_limit": 64}},
                {"module_id": "crystal_block", "kind": "block", "config": {"display_name": "Crystal Block", "hardness": 2.0, "resistance": 3.0}},
                {"module_id": "crystal_block_recipe", "kind": "recipe", "config": {"recipe_kind": "shaped", "result": "crystal_block"}},
            ],
        },
    }


def test_passive_content_only_typed_plan_does_not_need_a_fake_executable_adapter():
    assert _is_passive_content_only_authored_plan(_passive_plan())


def test_passive_exception_does_not_cover_active_typed_or_unknown_content():
    plan = _passive_plan()
    plan["typed_plan_ir"]["event_bindings"] = [{"event": "on_use"}]
    assert not _is_passive_content_only_authored_plan(plan)

    plan = _passive_plan()
    plan["content_design"]["modules"][0]["config"]["action"] = "grant_currency"
    assert not _is_passive_content_only_authored_plan(plan)

    plan = _passive_plan()
    plan["structured_sections"]["algorithm"] = {"atomic_mutations": [{"id": "purchase"}]}
    assert not _is_passive_content_only_authored_plan(plan)

    plan = _passive_plan()
    plan["content_design"]["modules"] = []
    assert not _is_passive_content_only_authored_plan(plan)


def test_content_only_noop_is_classified_without_suppressing_missing_source(tmp_path: Path):
    relative = Path("src/main/java/example/AuthoredProgram.java")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example; public final class AuthoredProgram {"
        "public static void initialize() {}"
        "}",
        encoding="utf-8",
    )
    units = [{"module_id": "authored_typed_plan", "path": str(relative), "symbol": "AuthoredProgram"}]

    findings = _authored_feature_semantic_findings(tmp_path, units)
    assert any("no executable behavior" in finding for finding in findings)
    assert _authored_feature_semantic_findings(
        tmp_path, units, passive_content_only=True
    ) == []

    source.unlink()
    findings = _authored_feature_semantic_findings(
        tmp_path, units, passive_content_only=True
    )
    assert any("missing or unsafe" in finding for finding in findings)
