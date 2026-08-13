import json

import pytest

from minecraft_mod_ai.game_design import (
    GameDesignPlanner,
    _extract_valid_game_design,
    _planner_plugin_manifest,
    _repair_system_prompt,
    _system_prompt,
    _validate_design,
)
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
    assert proposal.spec.mod_id == "moon_forge_mod"
    assert proposal.approval_hash == proposal.calculate_hash()


class _StaticTextRouter:
    """Return a captured local-model style response without changing it."""

    def __init__(self, text: str) -> None:
        self.text = text

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        return self.text


class _SequenceTextRouter:
    """Return one captured response per bounded planner attempt."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[tuple[list[dict], dict]] = []

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        self.calls.append((messages, kwargs))
        return self.texts.pop(0)


def _valid_planner_payload() -> dict:
    return json.loads(FakeRouter().generate_text("planner", []))


def test_initial_stage_requests_only_a_bounded_design_spine() -> None:
    payload = _valid_planner_payload()
    del payload["build_slice"]
    router = _SequenceTextRouter(json.dumps(payload))
    prompt = "Create a moon crystal item and a quest system."

    design, proposal = GameDesignPlanner(router).plan(
        prompt,
        media_paths=("reference.png",),
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge_mod"
    assert len(router.calls) == 1
    messages, kwargs = router.calls[0]
    assert messages[-1] == {"role": "user", "content": prompt}
    assert '"build_slice"' not in messages[0]["content"]
    assert '"research_brief"' not in messages[0]["content"]
    assert '"game_design"' in messages[0]["content"]
    assert kwargs["media_paths"] == ("reference.png",)


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
    assert proposal.spec.mod_id == "moon_forge_mod"


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
    assert proposal.spec.mod_id == "moon_forge_mod"


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
    assert proposal.spec.mod_id == "moon_forge_mod"


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
    assert proposal.spec.mod_id == "moon_forge_mod"


def test_multimodal_design_ignores_model_bootstrap_without_second_model_call() -> None:
    """A complete design never depends on the legacy model-authored bootstrap."""

    payload = _valid_planner_payload()
    del payload["build_slice"]["summary"]
    payload["build_slice"]["generation_note"] = "This does not replace summary."

    router = _SequenceTextRouter(json.dumps(payload))
    design, proposal = GameDesignPlanner(router).plan(
        "Create a moon crystal item."
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.summary
    assert proposal.spec.mod_id == "moon_forge_mod"
    assert len(router.calls) == 1


def test_multimodal_design_retries_without_reinjecting_large_malformed_output() -> None:
    """Retry keeps the request intact but excludes the model's oversized output."""

    payload = _valid_planner_payload()
    malformed = "MALFORMED_OUTPUT_SENTINEL" * 4_000
    router = _SequenceTextRouter(
        malformed,
        json.dumps(payload),
    )
    prompt = ("Create a moon crystal item. " + (
        "Keep this exact user requirement in planner context. " * 50
    )).strip()

    design, proposal = GameDesignPlanner(router).plan(
        prompt,
        media_paths=("reference.png",),
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge_mod"
    assert len(router.calls) == 2
    assert router.calls[0][0][-1]["content"] == prompt
    repair_messages, repair_kwargs = router.calls[1]
    assert [message["role"] for message in repair_messages] == ["system", "user"]
    assert repair_messages[-1]["content"] == prompt
    assert all(
        "MALFORMED_OUTPUT_SENTINEL" not in message["content"]
        for message in repair_messages
    )
    assert sum(len(message["content"]) for message in repair_messages) < len(malformed)
    assert repair_kwargs["media_paths"] == ("reference.png",)
    assert "Current executable plugin manifest" not in repair_messages[0]["content"]


def test_multimodal_design_recovers_a_complete_design_from_truncated_envelope() -> None:
    """A complete design uses the request-derived bootstrap without another model call."""

    design_payload = _valid_planner_payload()["game_design"]
    truncated = (
        '{"game_design":'
        + json.dumps(design_payload)
        + ',"build_slice":'
    )
    router = _SequenceTextRouter(truncated)

    design, proposal = GameDesignPlanner(router).plan(
        "Create a moon crystal item."
    )

    assert design["title"] == "Moon Forge"
    assert proposal.spec.mod_id == "moon_forge_mod"
    assert proposal.spec.mod_name == "Moon Forge"
    assert proposal.approval_hash == proposal.calculate_hash()
    assert len(router.calls) == 1


def test_multimodal_design_recovery_uses_stable_non_latin_identity() -> None:
    design_payload = _valid_planner_payload()["game_design"]
    design_payload["title"] = "달빛 유물"
    truncated = (
        '{"game_design":'
        + json.dumps(design_payload, ensure_ascii=False)
        + ',"build_slice":'
    )

    first = GameDesignPlanner(_SequenceTextRouter(truncated)).plan(
        "달빛 유물을 추가해 줘."
    )[1]
    second = GameDesignPlanner(_SequenceTextRouter(truncated)).plan(
        "달빛 유물을 추가해 줘."
    )[1]

    assert first.spec.mod_id.startswith("mmm_")
    assert first.spec.mod_id.endswith("_mod")
    assert first.spec.mod_id == second.spec.mod_id
    assert first.spec.mod_id != "crafted_works"
    assert first.spec.mod_name == "달빛 유물"


def test_multimodal_design_does_not_bootstrap_without_essential_design() -> None:
    """Format recovery must not replace a missing game design with a template."""

    incomplete_design = _valid_planner_payload()["game_design"]
    del incomplete_design["acceptance_tests"]
    truncated = (
        '{"game_design":'
        + json.dumps(incomplete_design)
        + ',"build_slice":'
    )
    router = _SequenceTextRouter(truncated, truncated)

    with pytest.raises(SpecValidationError, match="complete game_design"):
        GameDesignPlanner(router).plan("Create a moon crystal item.")

    assert len(router.calls) == 2


def test_game_design_system_prompt_keeps_only_the_compact_design_contract() -> None:
    manifest = _planner_plugin_manifest()
    prompt = _system_prompt()

    assert set(manifest) == {
        "product_scope",
        "standalone_map_generation",
        "plugins",
    }
    assert all(
        set(plugin) == {"plugin_id", "status"}
        for plugin in manifest["plugins"]
    )
    assert len(prompt) < 5_000
    assert '"game_design"' in prompt
    assert '"build_slice"' not in prompt
    assert '"research_brief"' not in prompt
    assert "later paginated production" in prompt
    assert '"required_mcp"' not in prompt





def test_multimodal_design_repairs_compact_module_metadata_without_third_model_call() -> None:
    """A local planner may keep the module choice while abbreviating its metadata."""

    payload = _valid_planner_payload()
    payload.pop("build_slice", None)
    payload["game_design"]["modules"] = [
        {"plugin_id": "custom_weather"},
        {
            "module_id": "seasonal_cooking",
            "description": "계절별 요리 진행 시스템",
        },
    ]
    router = _SequenceTextRouter(
        "not a JSON object",
        json.dumps(payload, ensure_ascii=False),
    )

    design, proposal = GameDesignPlanner(router).plan(
        "계절별 날씨와 요리 진행 시스템을 만들어줘."
    )

    assert len(router.calls) == 2
    assert design["modules"] == [
        {
            "plugin_id": "custom_weather",
            "status": "custom",
            "reason": "custom_weather",
        },
        {
            "module_id": "seasonal_cooking",
            "description": "계절별 요리 진행 시스템",
            "plugin_id": "seasonal_cooking",
            "status": "custom",
            "reason": "계절별 요리 진행 시스템",
        },
    ]
    assert proposal.spec.mod_id


def test_module_shape_recovery_does_not_accept_non_object_module_entries() -> None:
    payload = _valid_planner_payload()
    payload["game_design"]["modules"] = ["quest_system"]

    with pytest.raises(
        SpecValidationError,
        match="game_design.modules entries must contain",
    ):
        _extract_valid_game_design(json.dumps(payload))


def test_repair_prompt_restates_nested_module_and_asset_entry_contracts() -> None:
    prompt = _repair_system_prompt()
    assert '"plugin_id":"from catalog or custom"' in prompt
    assert '"status":"implemented|custom"' in prompt
    assert '"reason":"why requested"' in prompt
    assert '"id":"snake_case"' in prompt
    assert "Every modules entry must contain" in prompt


def test_game_design_prompts_define_strict_nested_collection_types() -> None:
    for prompt in (_system_prompt(), _repair_system_prompt()):
        assert (
            "combat and mod_context must be JSON objects whose values, when present, "
            "are arrays of\nnon-empty strings"
        ) in prompt

    invalid = _valid_planner_payload()["game_design"]
    invalid["mod_context"] = {
        "loader": "Fabric",
        "minecraft_version": "1.20.1",
    }
    with pytest.raises(
        SpecValidationError,
        match="game_design.mod_context values must be lists",
    ):
        _validate_design(invalid)
