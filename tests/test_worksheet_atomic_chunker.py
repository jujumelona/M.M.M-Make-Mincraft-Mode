from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet_section
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)
from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
    worksheet_chunk_prompt,
    worksheet_chunk_schema,
)
from worksheet_fixtures import row


@pytest.mark.parametrize("section", WORKSHEET_SECTIONS)
def test_all_packed_chunks_satisfy_atomicity_contract(section: str):
    chunks = pack_section_concerns(section)
    assert chunks

    projected_fields: dict[str, set[str]] = {}
    for index, concern_group in enumerate(chunks):
        is_first = index == 0
        schema = worksheet_chunk_schema(section, concern_group, include_evidence=is_first)
        # Every model-facing field page must satisfy the strict atomicity boundary.
        assert_atomic_model_schema(schema, surface=f"{section} chunk {index}")
        prompt = worksheet_chunk_prompt(section, index + 1, len(chunks), concern_group, include_evidence=is_first)
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
        include_evidence=True,
    )
    output = json.dumps({"extra_hallucinated_field": "bad"})

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
