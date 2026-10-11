"""Regression tests for authored content discovery and executable module routing."""

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai import authored_structured_design


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
        ("atomic_mutations", None),
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


def test_unrelated_server_tick_does_not_make_join_initialization_gameplay(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    plan.typed_plan_ir["event_bindings"].append({
        "event": "server_tick", "function": "tick",
    })
    plan.typed_plan_ir["functions"].append({
        "id": "tick",
        "body": [{
            "op": "expr",
            "value": {"op": "capability", "id": "player.send_message", "args": []},
        }],
    })
    with pytest.raises(ValueError, match="GAMEPLAY_IMPLEMENTATION_ABSENT.*non-bootstrap"):
        _assert_executable_gameplay_floor(plan, (), {"capabilities": []})


def test_dispatch_event_guard_does_not_misattribute_join_mutation() -> None:
    from minecraft_mod_ai.typed_plan_ir import typed_plan_mutating_events

    def literal(t, value):
        return {"op": "literal", "type": t, "value": value}

    def wrapper(event):
        return {
            "id": "on_" + event,
            "body": [{
                "op": "expr",
                "value": {
                    "op": "call", "function": "logic_dispatch",
                    "args": [literal("string", event)],
                },
            }],
        }

    plan = {
        "event_bindings": [
            {"event": "player_join", "function": "on_player_join"},
            {"event": "server_tick", "function": "on_server_tick"},
        ],
        "functions": [
            wrapper("player_join"), wrapper("server_tick"),
            {
                "id": "logic_dispatch",
                "parameters": [{"name": "event", "type": "string"}],
                "body": [{
                    "op": "if",
                    "condition": {
                        "op": "binary",
                        "operator": "==",
                        "left": {"op": "ref", "name": "event"},
                        "right": literal("string", "player_join"),
                    },
                    "then": [{"op": "state_set", "key": literal("string", "initialized")}],
                    "else": [],
                }],
            },
        ],
    }
    mutations = typed_plan_mutating_events(plan)
    assert mutations["player_join"] == ("state_set",)
    assert mutations["server_tick"] == ()


def test_atomic_mutation_passive_item_requires_actionable_retry(monkeypatch) -> None:
    from minecraft_mod_ai import content_design_graph

    class InspectedMutationRetry(Exception):
        pass

    calls = []

    def fake_records(_router, identifier, *, context, **_kwargs):
        assert identifier == "design/content_entity"
        calls.append(dict(context))
        if "rejected_passive_candidates" in context:
            assert set(context["allowed_content_kinds"]) == {
                "custom_item_behavior", "custom_block_behavior",
            }
            assert context["rejected_passive_candidates"][0]["kind"] == "item"
            raise InspectedMutationRetry()
        return {"records": [{
            "kind": "item", "entity_id": "ship_upgrade_material", "role": "stock",
        }]}

    monkeypatch.setattr(content_design_graph, "run_record_template", fake_records)
    with pytest.raises(InspectedMutationRetry):
        content_design_graph.compile_content_graph(
            "buy an upgrade by spending credits",
            object(),
            request_catalog={
                "requirements": [{
                    "requirement_id": "gameplay_upgrade",
                    "statement": "Deduct credits and grant the upgrade",
                    "design_context": {
                        "source_section": "algorithm",
                        "source_concern": "atomic_mutations",
                    },
                }],
                "host_constraints": {},
            },
        )
    assert len(calls) == 2


def test_multiple_authored_actions_reject_command_counter_stub(monkeypatch) -> None:
    """A runnable command writing one flag cannot implement a farm/harvest/cook loop."""
    def concerns(sections: dict, section: str) -> dict:
        if section == "resources_and_ui":
            return {"interactions": [
                {"server_action": "plant_crop", "trigger": "player_plant_crop"},
                {"server_action": "harvest_crop", "trigger": "player_harvest_crop"},
                {"server_action": "cook_dish", "trigger": "player_cook_dish"},
            ]}
        return _concerns(sections, section)

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    plan = _plan(mutations=True)
    plan.typed_plan_ir["event_bindings"] = [{"event": "command", "function": "join"}]
    plan.typed_plan_ir["functions"][0]["body"] = [{
        "op": "state_set",
        "key": {"op": "literal", "type": "string", "value": "generic_counter"},
        "value": {"op": "literal", "type": "int", "value": 1},
    }]
    with pytest.raises(ValueError, match="GAMEPLAY_ACTION_COVERAGE_MISSING.*plant_crop"):
        _assert_executable_gameplay_floor(
            plan,
            (ProductionModule("farm_screen", "gui", {}),),
            {"capabilities": []},
        )


def test_individually_guarded_gameplay_actions_pass_coverage(monkeypatch) -> None:
    """One command can legitimately dispatch multiple *proven* mutating actions."""
    actions = ("plant_crop", "harvest_crop", "cook_dish")

    def concerns(sections: dict, section: str) -> dict:
        if section == "resources_and_ui":
            return {"interactions": [
                {"server_action": action, "trigger": "player_" + action}
                for action in actions
            ]}
        return _concerns(sections, section)

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    plan = _plan(mutations=True)
    plan.typed_plan_ir["event_bindings"] = [{"event": "command", "function": "join"}]
    plan.typed_plan_ir["functions"][0]["body"] = [
        {
            "op": "if",
            "condition": {
                "op": "binary",
                "operator": "==",
                "left": {"op": "ref", "name": "action"},
                "right": {"op": "literal", "type": "string", "value": action},
            },
            "then": [{
                "op": "state_set",
                "key": {"op": "literal", "type": "string", "value": action},
                "value": {"op": "literal", "type": "int", "value": 1},
            }],
            "else": [],
        }
        for action in actions
    ]
    _assert_executable_gameplay_floor(
        plan,
        (ProductionModule("farm_screen", "gui", {}),),
        {"capabilities": []},
    )


def test_empty_atomic_mutation_discovery_gets_actionable_retry(monkeypatch) -> None:
    """An empty first model response must not silently discard a transaction."""
    from minecraft_mod_ai import content_design_graph

    class RetryObserved(Exception):
        pass

    calls = []

    def fake_records(_router, identifier, *, context, **_kwargs):
        assert identifier == "design/content_entity"
        calls.append(dict(context))
        if "rejected_passive_candidates" in context:
            assert context["rejected_passive_candidates"] == []
            assert set(context["allowed_content_kinds"]) == {
                "custom_item_behavior", "custom_block_behavior",
            }
            raise RetryObserved()
        return {"records": []}

    monkeypatch.setattr(content_design_graph, "run_record_template", fake_records)
    with pytest.raises(RetryObserved):
        content_design_graph.compile_content_graph(
            "implement planting and harvesting",
            object(),
            request_catalog={
                "requirements": [{
                    "requirement_id": "gameplay_mutation_empty",
                    "statement": "Plant a crop when the player chooses a crop",
                    "design_context": {
                        "source_section": "algorithm",
                        "source_concern": "atomic_mutations",
                    },
                }],
                "host_constraints": {},
            },
        )
    assert len(calls) == 2
