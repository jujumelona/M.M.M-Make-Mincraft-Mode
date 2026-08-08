import json

import pytest

from minecraft_mod_ai.game_design import GameDesignPlanner
from minecraft_mod_ai.spec import SpecValidationError


class FakeRouter:
    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        return json.dumps(
            {
                "game_design": {
                    "title": "Moon Forge",
                    "pitch": "Craft moon relics and complete lunar quests.",
                    "core_loop": ["mine", "craft", "complete quests"],
                    "progression": ["ore", "relic", "quest reward"],
                    "combat": {"player_verbs": [], "enemy_roles": []},
                    "mod_context": {
                        "vanilla_integration": ["crafting"],
                        "compatibility_targets": [],
                    },
                    "modules": [
                        {"plugin_id": "fabric-core", "status": "implemented", "reason": "available"},
                        {"plugin_id": "quest-system", "status": "blocked", "reason": "not implemented"},
                    ],
                    "assets": [{"id": "moon_crystal", "kind": "item", "brief": "blue crystal"}],
                    "acceptance_tests": ["item is registered"],
                },
                "build_slice": {
                    "mod_id": "moon_forge",
                    "mod_name": "Moon Forge",
                    "package_name": "ai.minecraft.generated.moon_forge",
                    "summary": "Moon content",
                    "contents": [
                        {
                            "content_id": "moon_crystal",
                            "kind": "item",
                            "display_name_en": "Moon Crystal",
                            "display_name_ko": "달 결정",
                            "color": "#89dceb",
                            "recipe": True,
                        }
                    ],
                    "deferred_capabilities": ["quest_system"],
                },
            },
            ensure_ascii=False,
        )


def test_multimodal_design_keeps_blocked_modules_visible() -> None:
    design, proposal = GameDesignPlanner(FakeRouter()).plan(
        "달 결정 아이템과 퀘스트를 만들어줘"
    )
    assert any(module["status"] == "blocked" for module in design["modules"])
    assert proposal.spec.mod_id == "moon_forge"
    assert proposal.approval_hash == proposal.calculate_hash()


class _StaticTextRouter:
    """Return a captured local-model style response without changing it."""

    def __init__(self, text: str) -> None:
        self.text = text

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        return self.text


def _valid_planner_payload() -> dict:
    return json.loads(FakeRouter().generate_text("planner", []))


def test_multimodal_design_skips_a_json_reasoning_draft_before_the_contract() -> None:
    """Local reasoning models can emit a valid JSON scratch object before the answer."""

    payload = _valid_planner_payload()
    text = (
        "Planning draft: "
        + json.dumps({"analysis": "First choose the player loop."})
        + "\nFinal JSON: "
        + json.dumps(payload)
    )

    design, proposal = GameDesignPlanner(_StaticTextRouter(text)).plan(
        "Create a moon crystal item."
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge"


def test_multimodal_design_skips_an_incomplete_contract_draft() -> None:
    """A reasoning draft can name contract keys without being the final response."""

    payload = _valid_planner_payload()
    draft = {
        "game_design": {"title": "Unfinished draft"},
        "build_slice": {"mod_id": "unfinished"},
    }
    text = json.dumps(draft) + "\n" + json.dumps(payload)

    design, proposal = GameDesignPlanner(_StaticTextRouter(text)).plan(
        "Create a moon crystal item."
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge"


def test_multimodal_design_ignores_incidental_top_level_metadata() -> None:
    """A valid contract must not fail merely because a model adds a harmless note."""

    payload = _valid_planner_payload()
    payload["generation_metadata"] = {
        "reasoning": "Kept the first bootstrap slice intentionally small."
    }
    payload["build_slice"]["planner_note"] = "This is explanatory metadata."

    design, proposal = GameDesignPlanner(
        _StaticTextRouter(json.dumps(payload))
    ).plan("Create a moon crystal item.")

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_name == "Moon Forge"


def test_multimodal_design_unwraps_a_nested_response_data_envelope() -> None:
    """OpenAI-compatible gateways may wrap a valid JSON plan in response/data."""

    payload = _valid_planner_payload()
    payload["planner_trace"] = {"model": "local-planner"}
    wrapped = {
        "request_id": "planner-42",
        "response": {"data": payload},
    }

    design, proposal = GameDesignPlanner(
        _StaticTextRouter(json.dumps(wrapped))
    ).plan("Create a moon crystal item.")

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge"


def test_multimodal_design_ignores_nested_model_metadata() -> None:
    """Only the explicit art direction survives alongside canonical fields."""

    payload = _valid_planner_payload()
    payload["game_design"]["art_direction"] = {
        "visual_tone": "cool moonlit blue",
        "texture_guidance": ["clear relic silhouettes"],
        "model_animation_guidance": [],
    }
    payload["game_design"]["model_trace"] = {
        "palette": "cool moonlit blue",
        "silhouette": "small relics with clear readable shapes",
    }
    payload["build_slice"]["generation_note"] = "Bootstrap only; expand later."

    design, proposal = GameDesignPlanner(
        _StaticTextRouter(json.dumps(payload))
    ).plan("Create a moon crystal item.")

    assert design["art_direction"]["visual_tone"] == "cool moonlit blue"
    assert "model_trace" not in design
    assert proposal.spec.mod_id == "moon_forge"


def test_multimodal_design_rejects_missing_canonical_bootstrap_fields() -> None:
    """Extra metadata may be ignored, but the executable bootstrap stays required."""

    payload = _valid_planner_payload()
    del payload["build_slice"]["summary"]
    payload["build_slice"]["generation_note"] = "This does not replace summary."

    with pytest.raises(SpecValidationError, match="build_slice is missing summary"):
        GameDesignPlanner(_StaticTextRouter(json.dumps(payload))).plan(
            "Create a moon crystal item."
        )
