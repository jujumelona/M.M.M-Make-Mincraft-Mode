import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import planning_state_pipeline
from minecraft_mod_ai.authored_structured_design import _recover_planner_string_overflow
from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.planning_pipeline import (
    PlanningGenerationInterrupted,
    PlanningPipeline,
)
from minecraft_mod_ai.structured_output import StructuredOutputValidationError
from minecraft_mod_ai.task_template_catalog import load_record_template

SCHEMA = {
    "type": "object",
    "properties": {"statement": {"type": "string", "maxLength": 256}},
    "required": ["statement"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("concern", DETAIL_RECORDS["behavior_contract"])
def test_behavior_records_author_missing_details_instead_of_requiring_proof(concern):
    template = load_record_template("feature/behavior_contract/" + concern)
    rules = " ".join(template["rules"])
    assert "fail rather" not in rules
    assert "design choices need no external evidence or approval" in rules


def _router(monkeypatch):
    router = ModelRouter(profile="fast_test")
    requests = []

    def generate(request):
        requests.append(request)
        return json.dumps(
            {"statement": "Players assemble modular spacecraft and recruit crews."}
        )

    config = SimpleNamespace(adapter="llama_cpp")
    adapter = SimpleNamespace(generate=generate)
    monkeypatch.setattr(router.registry, "role", lambda profile, role: config)
    monkeypatch.setattr(router, "_generation_adapter", lambda role: (config, adapter))
    monkeypatch.setattr(router, "_generation_scope", lambda config: nullcontext())
    return router, requests


@pytest.mark.parametrize("surface", ["fixed_template", "decision", "text"])
def test_all_planning_surfaces_accept_content_without_a_function_call(
    monkeypatch, surface
):
    router, requests = _router(monkeypatch)
    messages = [{"role": "user", "content": "Design a space mod."}]
    if surface == "fixed_template":
        result = generate_fixed_template_value(
            router, "planner", messages, response_schema=SCHEMA, enable_tools=False
        )
    elif surface == "decision":
        result = router.generate_tool_decision(
            "planner",
            messages,
            tool_name="submit_researched_requirements",
            parameters=SCHEMA,
        )
    else:
        result = json.loads(
            router.generate_text(
                "planner",
                messages,
                response_format="json",
                response_schema=SCHEMA,
                enable_tools=False,
            )
        )
    assert result["statement"].startswith("Players assemble")
    assert len(requests) == 1
    assert requests[0].tools == ()
    assert requests[0].response_schema == SCHEMA
    assert "interchange shape" in str(requests[0].messages)
    assert "Call the required function" not in str(requests[0].messages)


def test_interrupted_writer_reports_original_cause_and_keeps_draft(monkeypatch):
    state = {
        "plan_ready": False,
        "decisions": [{"statement": "modular spacecraft"}],
        "generation_interruption": {
            "reason": "connection reset during actors",
            "cause_type": "ConnectionError",
        },
    }
    monkeypatch.setattr(
        planning_state_pipeline, "prepare_planning_state", lambda *a, **k: state
    )
    pipeline = PlanningPipeline(object())
    monkeypatch.setattr(
        pipeline,
        "_semantic_design",
        lambda *a, **k: pytest.fail("unfinished data was lowered"),
    )
    with pytest.raises(
        PlanningGenerationInterrupted, match="connection reset during actors"
    ) as caught:
        pipeline.prepare("space mod")
    assert caught.value.planning_state == state
    assert "HANDOFF_READY" not in str(caught.value)


def test_planner_overlong_string_is_clamped_without_model_retry():
    schema = {
        "type": "object",
        "properties": {
            "formulae": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clamping": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                        }
                    },
                    "required": ["clamping"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 3,
            }
        },
        "required": ["formulae"],
        "additionalProperties": False,
    }
    raw = json.dumps(
        {"formulae": [{"clamping": "x" * 320}]},
        ensure_ascii=False,
    )
    failure = StructuredOutputValidationError(
        output=raw,
        errors=('$["formulae"][0]["clamping"]: value is too long',),
    )

    repaired = _recover_planner_string_overflow(failure, schema)

    assert len(repaired["formulae"][0]["clamping"]) == 256
    assert repaired["formulae"][0]["clamping"] == "x" * 256


def test_planner_schema_recovery_does_not_hide_non_length_errors():
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "note": {"type": "string", "maxLength": 4},
        },
        "required": ["count", "note"],
        "additionalProperties": False,
    }
    raw = json.dumps({"count": "wrong", "note": "too long"})
    failure = StructuredOutputValidationError(
        output=raw,
        errors=(
            '$["count"]: value is not of type integer',
            '$["note"]: value is too long',
        ),
    )

    with pytest.raises(StructuredOutputValidationError) as raised:
        _recover_planner_string_overflow(failure, schema)

    assert raised.value is failure
