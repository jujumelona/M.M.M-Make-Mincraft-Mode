from __future__ import annotations

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai import planning_state_implementation as planning
from minecraft_mod_ai.model_output_atomicity_contract import is_atomic_model_schema
from minecraft_mod_ai.planning_detail_template import (
    WORKSHEET_SECTIONS, validate_worksheet_section, worksheet_section_prompt,
    worksheet_section_schema,
)
from minecraft_mod_ai.planning_handoff_contract import project_detailed_plan_for_request_catalog
from minecraft_mod_ai.worksheet_atomic_chunker import pack_section_concerns, worksheet_chunk_schema
from worksheet_fixtures import row


@pytest.mark.parametrize("section", WORKSHEET_SECTIONS)
def test_fixed_template_matches_prompt_decode_and_host_validation(section):
    schema = worksheet_section_schema(section)
    payload = row(section)
    prompt_schema = worksheet_section_prompt(section).split("Exact response schema: ")[1].splitlines()[0]
    assert json.loads(prompt_schema) == schema
    Draft202012Validator(schema).validate(payload)
    assert validate_worksheet_section(payload, set(), section) == payload
    for invalid in ("a long legacy prose specification", {"arbitrary": "object"}):
        bad = {**payload, "specification": invalid}
        assert list(Draft202012Validator(schema).iter_errors(bad))
        with pytest.raises(ValueError, match="fixed specification template"):
            validate_worksheet_section(bad, set(), section)


def test_nested_fields_cannot_change_shape_or_silently_disappear():
    payload = row("behavior_contract")
    for mutation in (lambda p: p["specification"]["actors"][0].pop("authority"),
                     lambda p: p["specification"]["actors"][0].update(extra="invented"),
                     lambda p: p["specification"]["actors"][0].update(name={"text": "Player"})):
        bad = deepcopy(payload)
        mutation(bad)
        with pytest.raises(ValueError, match="fixed specification template"):
            validate_worksheet_section(bad, set(), "behavior_contract")


def test_inapplicable_concern_needs_explicit_reason():
    payload = row("persistence")
    payload["specification"]["migration"] = []
    with pytest.raises(ValueError, match="inapplicable reason"):
        validate_worksheet_section(payload, set(), "persistence")
    payload["specification"]["inapplicable_concerns"] = [
        {"concern": "migration", "reason": "First storage version; no prior data schema exists."}
    ]
    assert validate_worksheet_section(payload, set(), "persistence") == payload


def test_behavior_contract_oversized_records_are_field_paged_atomically():
    chunks = pack_section_concerns("behavior_contract")
    input_pages = [chunk for chunk in chunks if "inputs" in chunk]
    assert len(input_pages) >= 2
    projected_fields = []
    for index, chunk in enumerate(chunks):
        schema = worksheet_chunk_schema(
            "behavior_contract", chunk, include_evidence=index == 0
        )
        assert is_atomic_model_schema(schema)
        if "inputs" in chunk:
            projected_fields.extend(chunk.field_projection["inputs"])
    assert projected_fields == ["name", "type", "unit", "range", "default", "source"]


def test_ten_section_dag_preserves_objects_through_handoff():
    calls = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            section = messages[-1]["content"].split("Section: ", 1)[1].splitlines()[0]
            schema = kwargs["response_schema"]
            full = row(section)
            payload: dict = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop == "constraint_evidence_refs":
                    payload[prop] = full.get(prop, [])
                elif prop == "inapplicable_concerns":
                    payload[prop] = [
                        item for item in full["specification"].get("inapplicable_concerns", [])
                        if item["concern"] in schema.get("properties", {})
                    ]
                elif prop in full["specification"]:
                    allowed_fields = set(
                        prop_schema.get("items", {}).get("properties", {})
                    )
                    payload[prop] = [
                        {key: value for key, value in item.items() if key in allowed_fields}
                        for item in full["specification"][prop]
                    ]
            Draft202012Validator(schema).validate(payload)
            assert kwargs["response_format"] == "json" and kwargs["enable_tools"] is False
            calls.append(section)
            return json.dumps(payload)

    requirement = {"requirement_id": "req_001", "statement": "Gather a resource."}
    state = {
        "research_queue": [{"research_id": "r", "requirement_ref": "req_001", "status": "complete"}],
        "evidence": [{"research_ref": "r", "sufficient": True, "evidence_refs": ["ev:1"]}],
    }
    plan = planning._compile_requirement_plans_dag(
        Router(), state, [requirement], {"req_001": WORKSHEET_SECTIONS}, workers=2,
    )[0]
    projected = project_detailed_plan_for_request_catalog(plan, {"ev:1"})
    assert len(calls) >= 10 and set(calls) == set(WORKSHEET_SECTIONS)
    assert projected["engineering_worksheet"] == {s: row(s) for s in WORKSHEET_SECTIONS}
    assert '"success_cases"' in plan["verification_obligations"][0]["check"]


def test_prerequisite_json_preserves_long_tail_and_newlines():
    payload = row("behavior_contract")
    payload["specification"]["actors"][0]["role"] = "x" * 14000 + "\nTAIL_RULE"
    context = planning._section_dependency_context("state_model", WORKSHEET_SECTIONS, {"behavior_contract": payload})
    assert json.loads(context) == {"behavior_contract": payload}
