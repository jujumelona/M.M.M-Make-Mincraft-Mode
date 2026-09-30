from __future__ import annotations

import json
import threading
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


def test_authored_omissions_do_not_need_a_justification_to_pass():
    payload = row("persistence")
    payload["specification"]["migration"] = []
    assert validate_worksheet_section(payload, set(), "persistence") == payload
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

        def generate_tool_decision(
            self, role, messages, *, tool_name, parameters, description=""
        ):
            return json.loads(
                self.generate_text(
                    role,
                    messages,
                    response_schema=parameters,
                    response_format="json",
                    enable_tools=False,
                )
            )

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


def test_schema_rejection_decomposes_to_field_work_instead_of_retrying_same_chunk():
    rejected_shapes = []
    accepted_shapes = []

    class Router:
        @staticmethod
        def _value(schema):
            enum = schema.get("enum")
            if isinstance(enum, list) and enum:
                return enum[0]
            raw_type = schema.get("type")
            if isinstance(raw_type, list):
                raw_type = next((item for item in raw_type if item != "null"), "string")
            if raw_type == "array":
                return [Router._value(schema.get("items") or {})]
            if raw_type == "object":
                return {
                    key: Router._value(child)
                    for key, child in (schema.get("properties") or {}).items()
                }
            if raw_type == "integer":
                return 1
            if raw_type == "number":
                return 1.0
            if raw_type == "boolean":
                return True
            return "authored"

        def generate_text(self, role, messages, **kwargs):
            del role, messages
            schema = kwargs["response_schema"]
            properties = schema["properties"]
            concerns = [
                key for key in properties
                if key not in {"inapplicable_concerns", "constraint_evidence_refs"}
            ]
            field_counts = [
                len(properties[concern]["items"]["properties"])
                for concern in concerns
            ]
            shape = (tuple(concerns), tuple(field_counts))
            if len(concerns) > 1 or any(count > 1 for count in field_counts):
                rejected_shapes.append(shape)
                raise ValueError("fixture rejects non-isolated worksheet work")

            accepted_shapes.append(shape)
            concern = concerns[0]
            item_properties = properties[concern]["items"]["properties"]
            payload = {
                concern: [{
                    field: self._value(field_schema)
                    for field, field_schema in item_properties.items()
                }]
            }
            if "constraint_evidence_refs" in properties:
                payload["constraint_evidence_refs"] = []
            return json.dumps(payload)

    result = planning._compile_worksheet_section(
        Router(),
        requirement={"requirement_id": "r", "statement": "Gather a resource."},
        selected_sections=("behavior_contract",),
        section="behavior_contract",
        evidence=[],
        allowed=set(),
        completed={},
    )

    assert rejected_shapes
    assert accepted_shapes
    assert all(len(concerns) == 1 and counts == (1,) for concerns, counts in accepted_shapes)
    assert result["specification"]["actors"]
    assert result["specification"]["inputs"]


def test_section_chunk_dag_overlaps_independent_concerns_but_orders_field_pages(monkeypatch):
    barrier = threading.Barrier(2, timeout=2)
    first_page_done = threading.Event()
    schema_counts: list[tuple[tuple[str, ...], dict[str, int]]] = []

    chunks = [
        ("alpha",),
        ("beta",),
        ("alpha",),
    ]

    monkeypatch.setattr(planning, "pack_section_concerns", lambda _section: chunks)
    monkeypatch.setattr(planning, "router_native_model_parallelism", lambda _router: 2)

    def fake_schema(_section, concerns, *, include_evidence, record_counts):
        del include_evidence
        schema_counts.append((tuple(concerns), dict(record_counts)))
        return {"type": "object", "properties": {}, "additionalProperties": False}

    monkeypatch.setattr(planning, "worksheet_chunk_schema", fake_schema)
    monkeypatch.setattr(planning, "_chunk_messages", lambda *args, **kwargs: ())

    def fake_generate(
        _router,
        _messages,
        *,
        section,
        index,
        concerns,
        chunk_schema,
        recovery_context,
    ):
        del section, chunk_schema, recovery_context
        if index in {1, 2}:
            barrier.wait()
        if index == 1:
            first_page_done.set()
            return {"alpha": [{"field": "first"}]}
        if index == 2:
            return {"beta": [{"field": "independent"}]}
        assert index == 3
        assert first_page_done.is_set()
        assert concerns == ("alpha",)
        return {"alpha": [{"field": "continued"}]}

    monkeypatch.setattr(planning, "_generate_chunk", fake_generate)
    monkeypatch.setattr(
        planning,
        "merge_worksheet_section_chunks",
        lambda _section, values, _allowed: {"values": values},
    )

    result = planning._compile_worksheet_section(
        object(),
        requirement={"requirement_id": "r", "statement": "x"},
        selected_sections=("behavior_contract",),
        section="behavior_contract",
        evidence=[],
        allowed=set(),
        completed={},
    )

    assert len(result["values"]) == 3
    assert any(
        concerns == ("alpha",) and counts.get("alpha") == 1
        for concerns, counts in schema_counts
    )
