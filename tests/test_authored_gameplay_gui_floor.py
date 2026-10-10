"""Regression: a successful GUI compile is not a playable authored mod."""

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_production import _assert_executable_gameplay_floor
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai import authored_structured_design


def _plan(*, mutations: bool) -> SimpleNamespace:
    # The test is for the production binding contract, not worksheet syntax.
    return SimpleNamespace(
        structured_sections={"_test_mutations": mutations},
        typed_plan_ir={
            "event_bindings": [{"event": "player_join", "function": "join"}],
            "functions": [{
                "id": "join",
                "body": [{"op": "state_set", "key": {"op": "literal", "value": "started"}}],
            }],
        },
    )


def _concerns(sections: dict, section: str) -> dict:
    if section == "algorithm":
        return {
            "steps": [{"operation": "fabricate_ship_component"}],
            "atomic_mutations": [{"mutations": "deduct_resources"}]
            if sections["_test_mutations"] else [],
        }
    return {}


def test_gui_only_space_gameplay_rejected_before_build(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    modules = (
        ProductionModule("ship_construction", "gui", {"name": "Ship Construction"}),
        ProductionModule("planet_travel", "gui", {"name": "Planet Travel"}),
    )
    with pytest.raises(ValueError, match="GAMEPLAY_IMPLEMENTATION_ABSENT.*non-bootstrap"):
        _assert_executable_gameplay_floor(plan, modules, {"capabilities": []})


def test_actionable_non_gui_module_is_not_treated_as_gui_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    modules = (
        ProductionModule("ship_construction", "gui", {"name": "Ship Construction"}),
        ProductionModule("ore_item", "item", {"action": "consume_ore_to_fabricate"}),
    )
    _assert_executable_gameplay_floor(plan, modules, {"capabilities": []})


def test_passive_gui_can_still_be_authored_without_gameplay_mutations(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=False)
    _assert_executable_gameplay_floor(
        plan, (ProductionModule("about_screen", "gui", {}),),
        {"capabilities": []},
    )


def test_player_join_flag_does_not_count_as_a_gameplay_loop(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    with pytest.raises(ValueError, match="GAMEPLAY_IMPLEMENTATION_ABSENT.*non-bootstrap"):
        _assert_executable_gameplay_floor(plan, (), {"capabilities": []})



def test_gui_does_not_block_proven_runtime_gameplay_handler(monkeypatch) -> None:
    # A GUI can be a presentation layer for a real server command.  Do not
    # reject it when the typed host owns a reachable authoritative writer.
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    plan.typed_plan_ir["event_bindings"] = [
        {"event": "command", "function": "join"},
    ]
    _assert_executable_gameplay_floor(
        plan,
        (ProductionModule("ship_screen", "gui", {}),),
        {"capabilities": []},
    )



def test_gameplay_regeneration_schema_excludes_gui() -> None:
    from minecraft_mod_ai.design_generation_schema import (
        context_bound_record_schema,
    )
    from minecraft_mod_ai.content_design_contract import PRIMARY_CONTENT_KINDS
    from minecraft_mod_ai.task_template_catalog import load_template

    context = {
        "requirement": "fabricate a physical ship hull component",
        "allowed_content_kinds": [
            kind for kind in PRIMARY_CONTENT_KINDS if kind != "gui"
        ],
    }
    template = load_template("design/content_entity")
    schema = context_bound_record_schema(
        "design/content_entity", template["record_schema"], context,
    )
    assert "gui" not in schema["properties"]["kind"]["enum"]
    assert "item" in schema["properties"]["kind"]["enum"]
    assert "block" in schema["properties"]["kind"]["enum"]



def test_atomic_gameplay_writers_become_content_generation_inputs(monkeypatch) -> None:
    from minecraft_mod_ai import authored_content_contract

    def concerns(_sections: dict, section: str) -> dict:
        if section == "algorithm":
            return {
                "steps": [{"operation": "fabricate_ship"}],
                "atomic_mutations": [
                    {"mutations": "deduct_ore_and_create_hull"},
                ],
            }
        return {}

    monkeypatch.setattr(
        authored_content_contract, "active_concern_records", concerns,
    )
    requirements = authored_content_contract._gameplay_content_requirements(
        {}, requested_prompt="build playable spacecraft",
    )
    sources = {
        (r["design_context"]["source_section"], r["design_context"]["source_concern"])
        for r in requirements
    }
    assert ("algorithm", "steps") in sources
    assert ("algorithm", "atomic_mutations") in sources


@pytest.mark.parametrize(
    ("concern", "expected_allowed"),
    [
        ("steps", None),
        ("atomic_mutations", {"custom_item_behavior", "custom_block_behavior"}),
    ],
)
def test_gameplay_gui_only_discovery_retries_with_executable_owner(
    monkeypatch, concern, expected_allowed,
) -> None:
    from minecraft_mod_ai import content_design_graph

    class InspectedRetry(Exception):
        pass

    attempts = []

    def fake_records(_router, identifier, *, context, **_kwargs):
        assert identifier == "design/content_entity"
        attempts.append(dict(context))
        if "rejected_gui_only_candidates" in context:
            allowed = set(context["allowed_content_kinds"])
            if expected_allowed is None:
                assert "gui" not in allowed
                assert "item" in allowed
            else:
                assert allowed == expected_allowed
            raise InspectedRetry()
        return {"records": [{"kind": "gui", "entity_id": "display_only", "role": "screen"}]}

    monkeypatch.setattr(content_design_graph, "run_record_template", fake_records)
    with pytest.raises(InspectedRetry):
        content_design_graph.compile_content_graph(
            "implement a playable state transition",
            object(),
            request_catalog={
                "requirements": [{
                    "requirement_id": "gameplay_mutation_1",
                    "statement": "Implement executable gameplay, not a GUI-only stub",
                    "design_context": {
                        "source_section": "algorithm",
                        "source_concern": concern,
                    },
                }],
                "host_constraints": {},
            },
        )
    assert len(attempts) == 2
