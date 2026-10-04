from __future__ import annotations

import json
import re

import pytest
from worksheet_fixtures import row

from minecraft_mod_ai.authored_structured_design import (
    _generate_authored_chunk,
    _PlannerPageRequest,
    author_structured_sections,
    render_structured_sections,
)
from minecraft_mod_ai.planner_state_authoring import lower_state_expr_to_dsl
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS
from minecraft_mod_ai.structured_output import StructuredOutputValidationError
from minecraft_mod_ai.structured_state_runtime import (
    StateSymbolTable,
    compile_mutation_ir,
    validate_state_concern,
)
from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
)


class StateChoices:
    """Model boundary fixture: semantic choices, never a hand-written DSL script."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(next(self.responses))


def generate(field, concern, responses, names=("starship_hull_integrity",)):
    chunks = pack_section_concerns("state_model")
    page = next(p for p in chunks if p.field_projection.get(concern) == (field,))
    router = StateChoices(responses)
    result = _generate_authored_chunk(
        router,
        "Create a starship with hull integrity and fuel; repair restores hull to 100.",
        page=_PlannerPageRequest("state_model", chunks.index(page) + 1, len(chunks), page, {}, ()),
        state_symbols=StateSymbolTable(names),
        section_context={concern: [{"trigger": "repair"}]},
        record_counts={concern: 1},
    )
    return result[concern][0][field], router


@pytest.mark.parametrize("concern,field", [
    ("transitions", "mutation"), ("initialization", "initial_state"),
    ("updates", "mutation"), ("cleanup", "action"),
])
def test_state_assignment_is_assembled_from_complete_semantic_choices(concern, field):
    value, router = generate(field, concern, [
        {field: [{"target": "starship_hull_integrity", "operator": "=", "value": {"kind": "literal", "value": "100"}}]}
    ])
    assert value == "starship_hull_integrity = 100"
    validate_state_concern(concern, [{field: value}], symbols={"starship_hull_integrity"})
    assert compile_mutation_ir(value, declared={"starship_hull_integrity"}) == (
        'setState("starship_hull_integrity", Double.valueOf("100"));'
    )
    assert len(router.calls) == 1


def test_long_expression_keeps_both_operands_and_compiles():
    value, router = generate("mutation", "updates", [
        {"mutation": [{"target": "starship_hull_integrity", "operator": "=", "value": {
            "kind": "arithmetic", "op": "+",
            "left": {"kind": "state_ref", "name": "starship_hull_integrity"},
            "right": {"kind": "state_ref", "name": "starship_repair_amount"},
        }}]}
    ], names=("starship_hull_integrity", "starship_repair_amount"))
    assert value == "starship_hull_integrity = (starship_hull_integrity + starship_repair_amount)"
    assert len(value) > 64
    java = compile_mutation_ir(value, declared={"starship_hull_integrity", "starship_repair_amount"})
    assert '$mmmRead("starship_repair_amount", context)' in java
    assert len(router.calls) == 1


@pytest.mark.parametrize("concern,field", [("transitions", "guard"), ("invariants", "condition")])
def test_state_predicate_uses_complete_operands(concern, field):
    value, router = generate(field, concern, [
        {field: {"kind": "compare", "op": ">", "left": {"kind": "state_ref", "name": "starship_hull_integrity"}, "right": {"kind": "literal", "value": "0"}}}
    ])
    assert value == "(starship_hull_integrity > 0)"
    validate_state_concern(concern, [{field: value}], symbols={"starship_hull_integrity"})
    assert len(router.calls) == 1


def test_empty_state_action_survives_canonical_merge():
    canonical = row("state_model")["specification"]
    canonical["cleanup"][0]["action"] = ""
    chunks = []
    for page in pack_section_concerns("state_model"):
        concern = page[0]
        chunks.append({concern: [
            {field: record[field] for field in page.field_projection[concern]}
            for record in canonical[concern]
        ]})
    merged = merge_worksheet_section_chunks("state_model", chunks, set())
    assert merged["specification"]["cleanup"][0]["action"] == ""


def test_no_mutation_is_an_explicit_zero_count_decision():
    value, router = generate("action", "cleanup", [{"action": []}])
    assert value == ""
    assert len(router.calls) == 1


@pytest.mark.parametrize("value", ["", "100; ghost = 1", "starship_hull_integrity="])
def test_incomplete_or_injected_operand_is_rejected_at_its_component(value):
    with pytest.raises((StructuredOutputValidationError, ValueError)):
        generate("mutation", "updates", [
            {"mutation": [{"target": "starship_hull_integrity", "operator": "=", "value": {"kind": "number", "value": value}}]}
        ])


def test_quoted_semicolon_is_data_not_an_extra_assignment():
    value, router = generate("mutation", "updates", [
        {"mutation": [{"target": "status", "operator": "=", "value": {"kind": "literal", "value": 'docked; "safe"'}}]}
    ], names=("status",))
    assert value == 'status = "docked; \\"safe\\""'
    java = compile_mutation_ir(value, declared={"status"})
    assert java.count("setState(") == 1
    assert len(router.calls) == 1


def test_expression_growth_stops_at_finite_host_frontier():
    deep_not = {"kind": "literal", "value": "1"}
    for _ in range(32):
        deep_not = {"kind": "not", "term": deep_not}
    with pytest.raises(ValueError, match="PLANNER_STATE_EXPRESSION_BOUND"):
        lower_state_expr_to_dsl(deep_not)


@pytest.mark.parametrize("name", ["abs", "count", "size", "len"])
def test_host_fixes_unary_function_arity(name):
    value, router = generate("guard", "transitions", [
        {"guard": {"kind": "call", "name": name, "args": [{"kind": "state_ref", "name": "starship_hull_integrity"}, {"kind": "literal", "value": "10"}]}}
    ])
    assert value == f"{name}(starship_hull_integrity)"
    assert len(router.calls) == 1


def test_variadic_function_preserves_all_authored_arguments():
    value, router = generate("guard", "transitions", [
        {"guard": {"kind": "call", "name": "min", "args": [{"kind": "literal", "value": "10"}, {"kind": "state_ref", "name": "starship_hull_integrity"}]}}
    ])
    assert value == "min(10, starship_hull_integrity)"
    assert len(router.calls) == 1


def test_complete_structured_authoring_finishes_after_state_assembly():
    class DesignRouter:
        def generate_text(self, role, messages, **kwargs):
            text = "\n".join(message["content"] for message in messages)
            section = re.search(r"^Section: (\w+)", text, re.MULTILINE)[1]
            properties = kwargs["response_schema"]["properties"]
            if "Author state mutation for" in text or any(f in properties for f in ("mutation", "initial_state", "action")):
                field = next(f for f in ("mutation", "initial_state", "action") if f in properties)
                return json.dumps({field: [{"target": "stateValue", "operator": "=", "value": {"kind": "literal", "value": "100"}}]})
            if "Author state expression for" in text or any(f in properties for f in ("guard", "condition")):
                field = next(f for f in ("guard", "condition") if f in properties)
                return json.dumps({field: {"kind": "literal", "value": "100"}})
            if "record_count" in properties:
                return json.dumps({"record_count": 1})
            concern = next(iter(properties))
            fields = properties[concern]["items"]["properties"]
            source = row(section)["specification"][concern][0]
            if section == "authority_and_network" and concern == "synchronization":
                source["recipients"] = ["tracking_players"]
            if section == "persistence" and concern == "missing_defaults":
                source["default"] = []
            values = {field: source[field][:64] if isinstance(source[field], str)
                      else source[field] for field in fields}
            return json.dumps({concern: [values]})

    sections = author_structured_sections(DesignRouter(), "Create a starship with repairable hull.")
    assert set(sections) == set(WORKSHEET_SECTIONS)
    assert sections["state_model"]["specification"]["transitions"][0]["mutation"] == "stateValue = 100"
    assert "mutation=stateValue = 100" in render_structured_sections(sections)
    assert sections["state_model"]["specification"]["invariants"][0]["condition"] == "100"


def test_state_executable_page_fails_closed_if_not_singleton() -> None:
    from minecraft_mod_ai.authored_structured_design import _generate_authored_page_value

    page = _PlannerPageRequest("state_model", 1, 1, ("invariants",), {}, ())
    schema = {
        "properties": {
            "invariants": {
                "items": {
                    "required": ["condition", "enforcement"],
                    "properties": {
                        "condition": {"type": "string"},
                        "enforcement": {"type": "string"},
                    },
                },
                "minItems": 1,
            },
        },
    }

    class DummyRouter:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("must fail before calling model")

    with pytest.raises(ValueError, match="STATE_EXECUTABLE_PAGE_MUST_BE_SINGLETON"):
        _generate_authored_page_value(DummyRouter(), (), page, schema, None)
