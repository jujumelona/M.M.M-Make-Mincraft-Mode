from __future__ import annotations

from typing import Any

import pytest

from minecraft_mod_ai.atomic_design_pipeline import (
    DESIGN_SLOTS,
    compile_atomic_design,
)
from minecraft_mod_ai.atomic_slot_executor import SlotDefinition
from minecraft_mod_ai.implementation_fact import FactType
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.task_template_catalog import load_template


def test_design_templates_exist_and_satisfy_strict_atomicity():
    assert len(DESIGN_SLOTS) == 32
    for slot_id in DESIGN_SLOTS:
        tmpl = load_template(slot_id)
        assert tmpl["id"] == slot_id
        schema = tmpl.get("record_schema") or tmpl.get("output_schema")
        assert schema is not None
        # Must pass strict atomicity bounds: 1 field, depth 1, maxLength <= 256
        assert_atomic_model_schema(schema, surface=f"design template {slot_id}")
        slot_def = SlotDefinition(slot_id=slot_id, schema=schema)
        slot_def.validate_schema()


def test_compile_atomic_design_korean_space_mod():
    design = compile_atomic_design("우주모드 만들어")
    assert design["title"]
    assert len(design["core_loop"]) == 1
    assert len(design["progression"]) == 3
    assert "First Goal:" in design["progression"][0]
    assert "Condition:" in design["progression"][1]
    assert "Reward:" in design["progression"][2]

    # Verify slots dictionary: only active resolved slots exist
    slots = design["_design_slots"]
    assert "core_loop" in slots
    assert "first_goal" in slots
    assert "progression_condition" in slots
    assert "reward" in slots
    assert "visual_identity" in slots
    assert "theme" in slots
    # Unactivated slots are omitted, not fake-filled
    assert "audio_identity" not in slots
    assert "npc_role" not in slots
    assert "machine_role" not in slots
    for k, v in slots.items():
        assert isinstance(v, str)
        assert 0 < len(v) <= 256

    # Verify atomic implementation facts
    facts = design["_implementation_facts"]
    assert len(facts) >= 1
    fact_types = [f.fact_type for f in facts]
    assert FactType.ITEM_EXISTS in fact_types or FactType.BLOCK_EXISTS in fact_types or FactType.ENTITY_EXISTS in fact_types

    # Verify dynamic modules and asset requests
    modules = design["modules"]
    assets = design["assets"]
    assert len(modules) >= 1
    assert len(assets) >= 1
    for m in modules:
        m.validate()
    for a in assets:
        a.validate()
        assert a.width in (16, 64, 256) and a.height in (16, 64, 256)
        assert "Pixel Art" in a.prompt


def test_compile_atomic_design_empty_prompt_fails_closed():
    with pytest.raises(ValueError, match="ATOMIC_DESIGN"):
        compile_atomic_design("")


class _MockSlotRouter:
    def __init__(self, mapping: dict[str, Any]):
        self.mapping = mapping
        self.calls: list[str] = []

    def generate_tool_decision(self, _role, _messages, *, tool_name, parameters, description=""):
        self.calls.append(tool_name)
        props = parameters.get("properties", {})
        for slot_id, p_schema in props.items():
            val = self.mapping.get(slot_id)
            if val is not None:
                return {slot_id: val}
            if isinstance(p_schema, dict) and p_schema.get("type") == "array":
                return {slot_id: ["item"]}
            return {slot_id: f"Valid mock {slot_id}"}
        return {}


def test_compile_atomic_design_with_router():
    router = _MockSlotRouter({
        "core_loop": "Build deep space telescopes and discover exoplanets.",
        "first_goal": "Assemble a telescope lens from polished obsidian.",
        "progression_condition": "Calibrate the lens against the North Star.",
        "reward": "Starmap item showing asteroid cluster coordinates.",
    })

    design = compile_atomic_design("space stargazing mod", router=router)
    assert design["core_loop"] == ["Build deep space telescopes and discover exoplanets."]
    assert "First Goal: Assemble a telescope lens from polished obsidian." in design["progression"][0]
    assert "Condition: Calibrate the lens against the North Star." in design["progression"][1]
    assert "Reward: Starmap item showing asteroid cluster coordinates." in design["progression"][2]


def test_compile_atomic_design_with_research_context():
    research = {
        "summary": "Moon geology and lunar basalt materials",
        "known": ["Lunar basalt is dense", "Moon vacuum requires sealed helmets"],
        "references": ["Galacticraft", "Ad Astra"],
    }
    design = compile_atomic_design("moon base mod", research=research)
    assert design["title"]
    assert 5 <= len(design["_design_slots"]) <= 16
    assert "npc_role" not in design["_design_slots"]
    assert len(design["modules"]) >= 1
    assert len(design["assets"]) >= 1


def test_dynamic_slot_execution_count_is_bounded():
    router = _MockSlotRouter({
        "domains": ["item"],
    })
    design = compile_atomic_design("simple ruby item mod", router=router)
    assert 8 <= len(router.calls) < 20
    assert len(design["_design_slots"]) == len(router.calls) - 1
    assert "npc_role" not in design["_design_slots"]


def test_targeted_rename_activates_minimal_slots():
    design = compile_atomic_design("아이템 이름만 바꿔줘")
    assert "theme" in design["_design_slots"]
    assert "core_loop" not in design["_design_slots"]
    assert "progression_condition" not in design["_design_slots"]
    assert "npc_role" not in design["_design_slots"]


def test_targeted_texture_activates_minimal_slots():
    design = compile_atomic_design("아이템 텍스처만 바꿔줘")
    assert "visual_identity" in design["_design_slots"]
    assert "texture_requirement" in design["_design_slots"]
    assert "core_loop" not in design["_design_slots"]


def test_resolve_content_domains_fails_closed_when_router_fails():
    from minecraft_mod_ai.atomic_slot_executor import SlotFillError
    from minecraft_mod_ai.atomic_design_pipeline import resolve_content_domains

    class FailingRouter:
        def generate_tool_decision(self, *args, **kwargs):
            raise RuntimeError("Model generation failed")

    with pytest.raises(SlotFillError):
        resolve_content_domains(FailingRouter(), prompt="우주모드")


def test_custom_stack_limit_emitted_and_default_omitted():
    # Default stack limit (64) is omitted from facts
    design_default = compile_atomic_design("simple ruby item")
    stack_facts_default = [f for f in design_default["_implementation_facts"] if f.fact_type == FactType.ITEM_STACK_LIMIT]
    assert len(stack_facts_default) == 0

    # Custom stack limit (16) is emitted
    design_custom = compile_atomic_design("루나이트 원석은 16개까지 겹쳐져")
    stack_facts_custom = [f for f in design_custom["_implementation_facts"] if f.fact_type == FactType.ITEM_STACK_LIMIT]
    assert len(stack_facts_custom) == 1
    assert stack_facts_custom[0].value == 16


def test_entity_and_gui_domains_do_not_force_item():
    router_entity = _MockSlotRouter({"domains": ["entity"]})
    design_entity = compile_atomic_design("alien boss", router=router_entity)
    module_kinds = [m.kind for m in design_entity["modules"]]
    assert "entity" in module_kinds
    assert "item" not in module_kinds

