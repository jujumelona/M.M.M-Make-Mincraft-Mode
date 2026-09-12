from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.model_response_templates import response_schema
from minecraft_mod_ai.task_template_catalog import load_template


ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "minecraft_mod_ai" / "complete_orchestrator_services.py"


def test_visual_review_policy_is_template_owned_and_schema_reused() -> None:
    template = load_template("validation/visual_review")
    assert template["standalone"] is True
    assert set(template["input"]) == {
        "game_design",
        "acceptance_tests",
        "runtime_screenshots",
    }
    assert template["response_contract"] == "visual_review"
    schema = response_schema(template["response_contract"])
    assert schema["type"] == "object"

    source = SERVICES.read_text(encoding="utf-8")
    assert 'load_template("validation/visual_review")' in source
    assert "response_schema(response_contract)" in source
    assert "response_template_prompt" not in source
    assert "Return JSON {status: PASS|FAIL" not in source
    assert "Reject missing textures" not in source


def test_visual_review_template_keeps_semantics_small_and_visible_only() -> None:
    template = load_template("validation/visual_review")
    rules = tuple(str(rule) for rule in template["rules"])
    assert len(rules) == 4
    assert any("exactly once" in rule for rule in rules)
    assert any("Do not mark non-visual behavior" in rule for rule in rules)
    assert any("do not invent hidden state" in rule for rule in rules)
