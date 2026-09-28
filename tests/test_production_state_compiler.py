from __future__ import annotations

import json

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.production_state_compiler import (
    _parse_semantic_page,
    bind_production_state_contract,
    compile_production_state_section,
)
from minecraft_mod_ai.structured_state_runtime import render_state_model_concern


class PlanRouter:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return self.response


class ProductionStateRouter:
    def __init__(self):
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        concern = payload["concern"]
        self.calls.append((role, concern, kwargs))
        records = {
            "variables": [
                {
                    "name": "credits",
                    "owner": "player",
                    "type": "integer",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [
                {
                    "from_state": "dock",
                    "trigger": "launch",
                    "guard": "shipStatus == ShipStatus.COMPLETE AND credits >= cost",
                    "mutation": "credits -= cost",
                    "to_state": "space",
                }
            ],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
        }[concern]
        return json.dumps({"records": records, "complete": True})


def _plan_text() -> str:
    return (
        "# StarForge\n"
        "## behavior_contract\n"
        "- actors: player ship\n"
        "## state_model\n"
        "- variables: credits and ship state\n"
        "- transitions: launch only when Ship Status is Complete and enough credits exist\n"
        "## algorithm\n"
        "- steps: launch flow\n"
        "## integration\n"
        "- lifecycle: initialize\n"
        "## verification\n"
        "- tests: compile and launch\n"
    )


def test_plan_remains_free_markdown_generation_without_structured_compiler():
    router = PlanRouter(_plan_text())
    planner = CompleteGameDesignPlanner(router)

    plan = planner.plan("make a space mod")

    assert plan.text == _plan_text()
    assert plan.structured_sections == {}
    assert len(router.calls) == 1
    role, _messages, kwargs = router.calls[0]
    assert role == "planner"
    assert kwargs["response_format"] == "text"
    assert kwargs["response_schema"] is None
    assert kwargs["enable_tools"] is False


def test_malformed_json_like_state_output_is_parsed_without_json_validation():
    raw = (
        '{"type": "object", "properties": {"records": ['
        '{"name": "credits";"owner": "player";"type": "double";'
        '"unit": "currency";"default": "0.0";"domain": "financial"}, '
        '{"name": "ship_state";"owner": "player";"type": "string";'
        '"unit": "status";"default": "\\\"Docked\\\"";"domain": "navigation"}'
        '], "complete": true}'
    )

    records, complete = _parse_semantic_page(
        raw,
        fields=("name", "owner", "type", "unit", "default", "domain"),
    )

    assert complete is True
    assert records == [
        {
            "name": "credits",
            "owner": "player",
            "type": "double",
            "unit": "currency",
            "default": "0.0",
            "domain": "financial",
        },
        {
            "name": "ship_state",
            "owner": "player",
            "type": "string",
            "unit": "status",
            "default": '"Docked"',
            "domain": "navigation",
        },
    ]


def test_state_concern_extraction_never_paginates_on_missing_completion_signal():
    class Router:
        def __init__(self):
            self.calls = []

        def generate_text(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            self.calls.append(payload)
            concern = payload["concern"]
            if concern == "variables":
                return (
                    "RECORD\n"
                    "name=credits\nowner=player\ntype=integer\nunit=credits\n"
                    "default=0\ndomain=integer >= 0\nEND"
                )
            if concern == "invariants":
                # Deliberately omit STATUS/DONE. Older code kept asking for another page.
                return (
                    "RECORD\n"
                    "condition=credits >= 0\n"
                    "enforcement=reject negative balances\nEND"
                )
            return "STATUS=EMPTY"

    router = Router()
    section = compile_production_state_section(
        router,
        AuthoredPlan("make a space mod", _plan_text()),
    )

    assert len(router.calls) == 7
    assert [call["concern"] for call in router.calls].count("invariants") == 1
    assert all("page" not in call for call in router.calls)
    assert all("already_accepted_records" not in call for call in router.calls)
    assert section["specification"]["invariants"] == [
        {
            "condition": "credits >= 0",
            "enforcement": "reject negative balances",
        }
    ]


def test_production_state_lowering_normalizes_small_model_dsl_and_java_symbols():
    router = ProductionStateRouter()
    plan = AuthoredPlan("make a space mod", _plan_text())

    section = compile_production_state_section(router, plan)

    transitions = section["specification"]["transitions"]
    assert transitions[0]["guard"] == 'shipStatus == "COMPLETE" && credits >= cost'
    assert transitions[0]["mutation"] == "credits -= cost"
    assert all(
        kwargs["response_format"] == "text"
        and kwargs["response_schema"] is None
        and kwargs["enable_tools"] is False
        for _role, _concern, kwargs in router.calls
    )
    assert [concern for _role, concern, _kwargs in router.calls] == [
        "variables",
        "transitions",
        "invariants",
        "initialization",
        "updates",
        "cleanup",
        "concurrency",
    ]

    obligations = []
    for concern in ("variables", "transitions"):
        obligations.append(
            json.dumps(
                {
                    "instruction": json.dumps(
                        {"section": "state_model", "concern": concern}
                    ),
                    "structured_records": section["specification"][concern],
                }
            )
        )
    task = {"implementation_obligations": obligations}
    java = render_state_model_concern(task, "transitions", include_runtime=True)

    assert java is not None
    assert "ShipStatus.COMPLETE" not in java
    assert '$mmmRead("shipStatus", context)' in java
    assert '$mmmRead("credits", context)' in java
    assert '$mmmRead("cost", context)' in java


def test_production_binding_preserves_approved_plan_text():
    router = ProductionStateRouter()
    original = AuthoredPlan("make a space mod", _plan_text())

    bound = bind_production_state_contract(router, original)

    assert bound.text == original.text
    assert bound.requested_prompt == original.requested_prompt
    assert "state_model" in bound.structured_sections
    assert original.structured_sections == {}
