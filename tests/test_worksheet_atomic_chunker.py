from __future__ import annotations

import json

import pytest
from worksheet_fixtures import row

from minecraft_mod_ai.execution_contract_policy import (
    PLANNER_RECORD_FIELD_MAX_CHARS,
    PLANNER_RECORD_PAGE_MAX_FIELDS,
)
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.planning_detail_template import (
    WORKSHEET_SECTIONS,
    validate_worksheet_section,
)
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)
from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
    worksheet_chunk_prompt,
    worksheet_chunk_schema,
    worksheet_concern_cardinality_prompt,
    worksheet_concern_cardinality_schema,
)


def _page_for_field(section: str, concern: str, field: str):
    for chunk in pack_section_concerns(section):
        projection = getattr(chunk, "field_projection", {})
        if concern in chunk and field in projection.get(concern, ()):
            return chunk
    raise AssertionError(f"missing planner page for {section}.{concern}.{field}")


def _fixed_count(chunk, count: int = 1) -> dict[str, int]:
    return {str(chunk[0]): count}


def test_field_page_schema_requires_host_fixed_cardinality():
    page = pack_section_concerns("behavior_contract")[0]

    with pytest.raises(
        TypeError,
        match="record_counts",
    ):
        worksheet_chunk_schema("behavior_contract", page)


@pytest.mark.parametrize("section", WORKSHEET_SECTIONS)
def test_all_packed_chunks_satisfy_atomicity_contract(section: str):
    chunks = pack_section_concerns(section)
    assert chunks

    projected_fields: dict[str, set[str]] = {}
    for index, concern_group in enumerate(chunks):
        schema = worksheet_chunk_schema(
            section,
            concern_group,
            record_counts=_fixed_count(concern_group),
        )
        # Every model-facing field page must satisfy the strict atomicity boundary.
        assert_atomic_model_schema(schema, surface=f"{section} chunk {index}")
        prompt = worksheet_chunk_prompt(
            section,
            index + 1,
            len(chunks),
            concern_group,
            record_counts=_fixed_count(concern_group),
        )
        assert f"Section: {section}" in prompt
        projection = getattr(concern_group, "field_projection", {})
        for concern in concern_group:
            assert concern in prompt
            projected_fields.setdefault(concern, set()).update(projection.get(concern, ()))

    assert projected_fields == {
        concern: set(columns.split())
        for concern, columns in DETAIL_RECORDS[section].items()
    }


@pytest.mark.parametrize("section", WORKSHEET_SECTIONS)
def test_deterministic_merge_reconstructs_canonical_section(section: str):
    canonical = row(section)
    chunks_def = pack_section_concerns(section)
    chunk_payloads = []

    for index, concern_group in enumerate(chunks_def):
        payload: dict = {
            "inapplicable_concerns": [
                item
                for item in canonical["specification"].get("inapplicable_concerns", [])
                if item["concern"] in concern_group
            ]
        }
        for concern in concern_group:
            payload[concern] = canonical["specification"][concern]
        if index == 0:
            payload["constraint_evidence_refs"] = canonical["constraint_evidence_refs"]
        chunk_payloads.append(payload)

    allowed_refs = set(canonical["constraint_evidence_refs"])
    merged = merge_worksheet_section_chunks(section, chunk_payloads, allowed_refs)
    assert merged == canonical
    assert validate_worksheet_section(merged, allowed_refs, section) == canonical


def test_planner_pages_are_prebounded_and_continuations_fix_cardinality():
    chunks = pack_section_concerns("behavior_contract")
    assert chunks

    for chunk in chunks:
        assert len(chunk) == 1
        concern = str(chunk[0])
        projection = getattr(chunk, "field_projection", {})
        fields = projection[concern]
        assert 1 <= len(fields) <= PLANNER_RECORD_PAGE_MAX_FIELDS

        schema = worksheet_chunk_schema(
            "behavior_contract",
            chunk,
            record_counts=_fixed_count(chunk),
        )
        assert set(schema["properties"]) == {concern}
        assert schema["required"] == [concern]
        assert "anyOf" not in schema
        assert "inapplicable_concerns" not in schema["properties"]
        assert "constraint_evidence_refs" not in schema["properties"]
        item_properties = schema["properties"][concern]["items"]["properties"]
        assert set(item_properties) == set(fields)
        for field_schema in item_properties.values():
            raw_type = field_schema.get("type")
            types = set(raw_type) if isinstance(raw_type, list) else {raw_type}
            if "string" in types:
                assert field_schema["maxLength"] <= PLANNER_RECORD_FIELD_MAX_CHARS

    actors_pages = [chunk for chunk in chunks if "actors" in chunk]
    assert actors_pages
    continued = worksheet_chunk_schema(
        "behavior_contract",
        actors_pages[-1],
        record_counts={"actors": 3},
    )
    actors_array = continued["properties"]["actors"]
    assert actors_array["minItems"] == 3
    assert actors_array["maxItems"] == 3

    prompt = worksheet_chunk_prompt(
        "behavior_contract",
        1,
        len(chunks),
        actors_pages[-1],
        record_counts={"actors": 3},
    )
    assert "fixed record cardinality" in prompt
    assert "actors=3 row(s)" in prompt


def test_cardinality_decision_is_tiny_and_content_free():
    schema = worksheet_concern_cardinality_schema(
        "behavior_contract",
        "actors",
    )
    assert schema == {
        "type": "object",
        "properties": {
            "record_count": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            }
        },
        "required": ["record_count"],
        "additionalProperties": False,
    }
    prompt = worksheet_concern_cardinality_prompt(
        "behavior_contract",
        "actors",
    )
    assert "Do not author record content" in prompt
    assert "0 through 4" in prompt

def test_state_model_symbols_do_not_change_planning_chunk_field_types():
    target = _page_for_field("state_model", "transitions", "guard")

    baseline = worksheet_chunk_schema(
        "state_model",
        target,
        record_counts=_fixed_count(target),
    )
    with_symbols = worksheet_chunk_schema(
        "state_model",
        target,
        state_symbols={"ship_blueprint", "player_currency"},
        record_counts=_fixed_count(target),
    )

    assert with_symbols == baseline
    guard = with_symbols["properties"]["transitions"]["items"]["properties"]["guard"]
    assert guard["type"] == "string"


def test_root_integration_prerequisite_accepts_null_and_canonicalizes():
    chunks = pack_section_concerns("integration")
    target = _page_for_field("integration", "initialization_order", "prerequisite")
    schema = worksheet_chunk_schema(
        "integration",
        target,
        record_counts=_fixed_count(target),
    )
    prerequisite = schema["properties"]["initialization_order"]["items"]["properties"]["prerequisite"]
    assert prerequisite["type"] == ["string", "null"]

    canonical = row("integration")
    payloads = []
    for index, chunk in enumerate(chunks):
        payload = {"inapplicable_concerns": []}
        for concern in chunk:
            payload[concern] = canonical["specification"][concern]
        if "initialization_order" in chunk:
            payload["initialization_order"] = [
                {"component": "EconomySystem", "prerequisite": None, "order": "1"}
            ]
        if index == 0:
            payload["constraint_evidence_refs"] = canonical["constraint_evidence_refs"]
        payloads.append(payload)

    merged = merge_worksheet_section_chunks("integration", payloads, set())
    first = merged["specification"]["initialization_order"][0]
    assert first["prerequisite"] == "no prerequisite"


def test_synchronization_recipients_preserve_bounded_array_type():
    target = _page_for_field("authority_and_network", "synchronization", "recipients")
    schema = worksheet_chunk_schema(
        "authority_and_network",
        target,
        record_counts=_fixed_count(target),
    )
    recipients = schema["properties"]["synchronization"]["items"]["properties"]["recipients"]
    assert recipients["type"] == "array"
    assert recipients["maxItems"] == 4
    assert recipients["items"]["type"] == "string"
    assert_atomic_model_schema(schema, surface="authority_and_network synchronization")


def test_synchronization_recipients_survive_merge_as_list():
    canonical = row("authority_and_network")
    chunks = pack_section_concerns("authority_and_network")
    payloads = []
    expected = ["server_authoritative_state", "client_ui_panel_ship_design"]

    for index, chunk in enumerate(chunks):
        payload = {"inapplicable_concerns": []}
        for concern in chunk:
            payload[concern] = canonical["specification"][concern]
        if "synchronization" in chunk:
            payload["synchronization"] = [
                {
                    "state": "ship_component_slot_inventory_state_delta",
                    "recipients": expected,
                    "trigger": "player_action_module_purchase_confirmed",
                }
            ]
        if index == 0:
            payload["constraint_evidence_refs"] = canonical["constraint_evidence_refs"]
        payloads.append(payload)

    merged = merge_worksheet_section_chunks(
        "authority_and_network",
        payloads,
        set(canonical["constraint_evidence_refs"]),
    )
    assert merged["specification"]["synchronization"][0]["recipients"] == expected


def test_persistence_missing_default_accepts_and_preserves_empty_list():
    chunks = pack_section_concerns("persistence")
    target = _page_for_field("persistence", "missing_defaults", "default")
    schema = worksheet_chunk_schema(
        "persistence",
        target,
        record_counts=_fixed_count(target),
    )
    validate_structured_output(
        json.dumps({
            "missing_defaults": [
                {"default": []}
            ]
        }),
        response_format="json",
        response_schema=schema,
    )
    assert_atomic_model_schema(schema, surface="persistence missing_defaults")

    canonical = row("persistence")
    payloads = []
    for index, chunk in enumerate(chunks):
        payload = {"inapplicable_concerns": []}
        for concern in chunk:
            payload[concern] = canonical["specification"][concern]
        if "missing_defaults" in chunk:
            payload["missing_defaults"] = [
                {"field": "unlockable_blueprint_ids", "default": []}
            ]
        if index == 0:
            payload["constraint_evidence_refs"] = canonical["constraint_evidence_refs"]
        payloads.append(payload)

    merged = merge_worksheet_section_chunks(
        "persistence",
        payloads,
        set(canonical["constraint_evidence_refs"]),
    )
    assert merged["specification"]["missing_defaults"][0]["default"] == []


def test_merge_rejects_missing_chunk_page():
    expected_chunks = pack_section_concerns("behavior_contract")
    assert len(expected_chunks) > 1
    supplied_chunks = [{} for _ in expected_chunks[:-1]]

    with pytest.raises(
        ValueError,
        match=rf"expected {len(expected_chunks)} chunks, got {len(supplied_chunks)}",
    ):
        merge_worksheet_section_chunks("behavior_contract", supplied_chunks, set())


def test_chunk_schema_rejects_undeclared_fields_before_merge():
    concern_group = pack_section_concerns("behavior_contract")[0]
    schema = worksheet_chunk_schema(
        "behavior_contract",
        concern_group,
        record_counts=_fixed_count(concern_group),
    )
    concern = str(concern_group[0])
    fields = concern_group.field_projection[concern]
    canonical_item = row("behavior_contract")["specification"][concern][0]
    output = json.dumps({
        concern: [
            {field: canonical_item[field] for field in fields}
        ],
        "extra_hallucinated_field": "bad",
    })

    with pytest.raises(
        StructuredOutputValidationError,
        match="Additional properties are not allowed",
    ):
        validate_structured_output(
            output,
            response_format="json",
            response_schema=schema,
        )


def test_merge_auto_reconciles_empty_concerns_without_inapplicable_reasons():
    canonical = row("behavior_contract")
    chunks = pack_section_concerns("behavior_contract")
    chunk_payloads = []
    for index, concern_group in enumerate(chunks):
        payload: dict = {"inapplicable_concerns": []}
        for concern in concern_group:
            if concern in ("preconditions", "boundaries"):
                # Small model left these empty and forgot to put them in inapplicable_concerns.
                payload[concern] = []
            elif concern == "rejection_postconditions":
                # Small model put a dummy placeholder record.
                payload[concern] = [{"condition": "", "preserved_state": "", "observation": ""}]
            else:
                payload[concern] = canonical["specification"][concern]
        if index == 0:
            # Model hallucinated an evidence ref.
            payload["constraint_evidence_refs"] = ["allowed_ref_1", "hallucinated_ref"]
        chunk_payloads.append(payload)

    allowed_refs = {"allowed_ref_1"}
    merged = merge_worksheet_section_chunks("behavior_contract", chunk_payloads, allowed_refs)

    # Inapplicable concerns are automatically reconciled for empty concerns.
    reconciled_concerns = {
        item["concern"] for item in merged["specification"]["inapplicable_concerns"]
    }
    assert "preconditions" in reconciled_concerns
    assert "boundaries" in reconciled_concerns
    assert "rejection_postconditions" in reconciled_concerns

    # Hallucinated evidence refs are filtered out to keep refs strictly host-bounded.
    assert merged["constraint_evidence_refs"] == ["allowed_ref_1"]

    # Merged section passes canonical validation without ValueError.
    assert validate_worksheet_section(merged, allowed_refs, "behavior_contract") == merged
