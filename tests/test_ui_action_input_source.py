"""Input ABI follows executable source, rather than a second prose guess."""
from copy import deepcopy
import json

import pytest

from minecraft_mod_ai.typed_action_authoring import (
    author_action_functions, author_action_metadata, promote_action_sections,
)
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from test_typed_ui_action_authoring import sections


class NoInference:
    def generate_structured_data(self, *args, **kwargs):
        raise AssertionError("Canonical input definitions must not be re-inferred")


def prose_design():
    return sections("Record the selected amount on the server")


def test_production_form_and_body_share_the_canonical_input_name():
    design = prose_design()
    before = deepcopy(design)
    promoted = promote_action_sections(NoInference(), design)
    metadata = promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]
    assert metadata["intent"] == "Record the selected amount on the server"
    assert metadata["inputs"] == [{
        "name": "amount", "type": "int", "label": "amount",
        "minimum": -2147483648, "maximum": 2147483647,
    }]
    functions, bindings = author_action_functions(
        NoInference(), "Record my amount", promoted,
        typed_host_capability_contracts(), first_index=0, prefer_grounded=True,
    )
    validated = validate_typed_plan_ir({
        "schema_version": "mmm/typed-plan-ir-v1", "source_sha256": "b" * 64,
        "functions": functions, "event_bindings": bindings, "initialize": [],
    }, capabilities=typed_host_capability_contracts())
    assert validated["functions"][0]["parameters"][1] == {"name": "amount", "type": "int"}
    assert validated["functions"][0]["body"][0]["value"] == {"op": "ref", "name": "amount"}
    assert design == before


def test_server_state_reads_and_constant_transactions_create_no_client_fields():
    design = prose_design()
    update = design["state_model"]["specification"]["updates"][0]
    update["mutation"][0]["operator"] = "+="
    update["mutation"][0]["value"] = {"kind": "number", "value": "1"}
    design["state_model"]["specification"]["transitions"] = [{
        "trigger": update["trigger"], "from_state": "ready", "to_state": "ready",
        "guard": {"kind": "compare", "op": ">=",
                  "left": {"kind": "state_ref", "name": "amount"},
                  "right": {"kind": "number", "value": "0"}},
        "mutation": deepcopy(update["mutation"]),
    }]
    design["state_model"]["specification"]["updates"] = []
    promoted = promote_action_sections(NoInference(), design)
    assert promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]["inputs"] == []


@pytest.mark.parametrize("scalar,value,expected", [
    ("boolean", {"kind": "context_ref", "name": "enabled"},
     {"name": "enabled", "type": "boolean", "label": "enabled"}),
    ("long", {"kind": "context_ref", "name": "quantity"},
     {"name": "quantity", "type": "long", "label": "quantity",
      "minimum": -9223372036854775808, "maximum": 9223372036854775807}),
    ("string", {"kind": "context_ref", "name": "selection"},
     {"name": "selection", "type": "string", "label": "selection",
      "max_length": 160, "choices": []}),
])
def test_input_types_come_from_the_mutation_target(scalar, value, expected):
    design = prose_design()
    variable = design["state_model"]["specification"]["variables"][0]
    variable.update(type=scalar, default="false" if scalar == "boolean" else "0")
    design["state_model"]["specification"]["updates"][0]["mutation"][0]["value"] = value
    promoted = promote_action_sections(NoInference(), design)
    assert promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]["inputs"] == [expected]


def test_a_guard_can_supply_a_boolean_input_without_exposing_server_state():
    design = prose_design()
    update = design["state_model"]["specification"]["updates"][0]
    update["mutation"][0]["value"] = {"kind": "number", "value": "1"}
    design["state_model"]["specification"]["transitions"] = [{
        "trigger": update["trigger"], "from_state": "ready", "to_state": "done",
        "guard": {"kind": "context_ref", "name": "confirmed"},
        "mutation": deepcopy(update["mutation"]),
    }]
    design["state_model"]["specification"]["updates"] = []
    promoted = promote_action_sections(NoInference(), design)
    assert promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]["inputs"] == [
        {"name": "confirmed", "type": "boolean", "label": "confirmed"},
    ]


def test_conflicting_context_types_are_not_repaired_by_renaming_inputs():
    design = prose_design()
    state = design["state_model"]["specification"]
    state["variables"].append({
        "name": "description", "type": "string", "default": "", "owner": "player",
        "domain": "text", "unit": "text",
    })
    state["updates"][0]["mutation"].append({
        "target": "description", "operator": "=",
        "value": {"kind": "context_ref", "name": "amount"},
    })
    with pytest.raises(ValueError, match="ACTION_SOURCE_INPUT_TYPE_CONFLICT"):
        promote_action_sections(NoInference(), design)


def test_explicit_metadata_keeps_its_declared_gameplay_bounds():
    design = sections()
    assert promote_action_sections(NoInference(), design) == design


def test_legacy_decisions_include_task_request_and_source_rules():
    design = prose_design()
    design["state_model"]["specification"]["updates"][0]["trigger"] = "unrelated event"
    record = design["resources_and_ui"]["specification"]["interactions"][0]
    seen = []
    answers = [{"label": "Set quantity"}, {"count": 1},
               {"type": "int", "label": "Quantity"}, {"minimum": 1, "maximum": 10}]

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            payload = json.loads(messages[-1]["content"])
            seen.append((payload, messages[0]["content"]))
            return answers.pop(0)

    metadata = author_action_metadata(Router(), record, source_sections=design,
                                      requested_prompt="Let me set the quantity")
    assert metadata["inputs"][0]["label"] == "Quantity"
    assert [p["decision"] for p, _ in seen] == ["label", "input_count", "input_type", "input_bounds"]
    assert all(p["requested_prompt"] == "Let me set the quantity" for p, _ in seen)
    assert seen[2][0]["state_context"]["updates"][0]["trigger"] == "unrelated event"
    assert "player-supplied value" in seen[2][1]
    assert seen[0][1] != seen[2][1]


def test_ui_design_receives_canonical_state_prerequisites():
    from minecraft_mod_ai.authored_structured_design import _authored_chunk_messages

    design = prose_design()
    messages = _authored_chunk_messages(
        "Record my amount", section="resources_and_ui", chunk_index=1, chunk_count=1,
        concerns=("interactions",), completed=design, record_counts={"interactions": 1},
    )
    user = messages[-1]["content"]
    assert '"context_ref"' in user
    assert '"name":"amount"' in user or '"name": "amount"' in user


def test_long_input_guard_preserves_type_with_literal_on_the_left():
    design = prose_design()
    state = design["state_model"]["specification"]
    state["variables"][0]["type"] = "long"
    update = state["updates"].pop()
    state["transitions"] = [{
        "trigger": update["trigger"], "from_state": "ready", "to_state": "done",
        "guard": {"kind": "compare", "op": "<=",
                  "left": {"kind": "number", "value": "0"},
                  "right": {"kind": "context_ref", "name": "amount"}},
        "mutation": update["mutation"],
    }]
    promoted = promote_action_sections(NoInference(), design)
    action = promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]
    assert action["inputs"][0]["type"] == "long"
    functions, _ = author_action_functions(
        NoInference(), "Set amount", promoted, typed_host_capability_contracts(),
        first_index=0, prefer_grounded=True,
    )
    assert functions[0]["body"][0]["condition"]["left"] == {
        "op": "literal", "type": "long", "value": 0,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_type_constraints_are_collected_before_literal_defaults(reverse):
    design = prose_design()
    state = design["state_model"]["specification"]
    state["variables"][0]["type"] = "long"
    update = state["updates"].pop()
    state["transitions"] = [{
        "trigger": update["trigger"], "from_state": "ready", "to_state": "first",
        "guard": {"kind": "compare", "op": ">",
                  "left": {"kind": "context_ref", "name": "amount"},
                  "right": {"kind": "number", "value": "0"}},
        "mutation": [{"target": "amount", "operator": "=",
                      "value": {"kind": "number", "value": "1"}}],
    }, {
        "trigger": update["trigger"], "from_state": "first", "to_state": "done",
        "guard": {"kind": "literal", "value": True}, "mutation": update["mutation"],
    }]
    if reverse:
        state["transitions"].reverse()
    promoted = promote_action_sections(NoInference(), design)
    action = promoted["resources_and_ui"]["specification"]["interactions"][0]["server_action"]
    assert action["inputs"] == [{
        "name": "amount", "type": "long", "label": "amount",
        "minimum": -9223372036854775808, "maximum": 9223372036854775807,
    }]


def test_saved_prose_migration_uses_source_inputs_and_retains_unrelated_functions():
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.typed_action_authoring import migrate_ui_action_plan

    unrelated = {"id": "unrelated", "parameters": [], "return_type": "void",
                 "body": [], "covers": []}
    old_ir = {"schema_version": "mmm/typed-plan-ir-v1", "source_sha256": "b" * 64,
              "initialize": [], "functions": [unrelated], "event_bindings": []}
    saved = AuthoredPlan("Let me set an amount", "Set the supplied amount",
                         structured_sections=prose_design(), typed_plan_ir=old_ir)
    before = deepcopy(saved.to_dict())
    migrated = migrate_ui_action_plan(NoInference(), saved)
    assert migrated.typed_plan_ir["functions"][0] == unrelated
    assert migrated.typed_plan_ir["event_bindings"][0]["config"]["action"]["inputs"][0]["name"] == "amount"
    assert saved.to_dict() == before
