from __future__ import annotations

import pytest

from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS, validate_worksheet_section
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
    assert len(chunks) >= 2
    for index, concern_group in enumerate(chunks):
        is_first = index == 0
        schema = worksheet_chunk_schema(section, concern_group, include_evidence=is_first)
        # Must pass atomicity assertion without raising ModelConfigurationError
        assert_atomic_model_schema(schema, surface=f"{section} chunk {index}")
        prompt = worksheet_chunk_prompt(section, index + 1, len(chunks), concern_group, include_evidence=is_first)
        assert f"Section: {section}" in prompt
        for c in concern_group:
            assert c in prompt


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
        for c in concern_group:
            payload[c] = canonical["specification"][c]
        if index == 0:
            payload["constraint_evidence_refs"] = canonical["constraint_evidence_refs"]
        chunk_payloads.append(payload)

    allowed_refs = set(canonical["constraint_evidence_refs"])
    merged = merge_worksheet_section_chunks(section, chunk_payloads, allowed_refs)
    assert merged == canonical
    assert validate_worksheet_section(merged, allowed_refs, section) == canonical


def test_merge_rejects_missing_concerns():
    chunks = [{"actors": [{"name": "P", "role": "User", "authority": "client"}], "inapplicable_concerns": []}]
    with pytest.raises(ValueError, match="missing concerns in merged"):
        merge_worksheet_section_chunks("behavior_contract", chunks, set())


def test_merge_rejects_undeclared_fields():
    canonical = row("behavior_contract")
    chunks = pack_section_concerns("behavior_contract")
    chunk_payloads = []
    for index, concern_group in enumerate(chunks):
        payload: dict = {"inapplicable_concerns": []}
        for c in concern_group:
            payload[c] = canonical["specification"][c]
        if index == 0:
            payload["extra_hallucinated_field"] = "bad"
        chunk_payloads.append(payload)

    with pytest.raises(ValueError, match="undeclared field"):
        merge_worksheet_section_chunks("behavior_contract", chunk_payloads, set())


def test_merge_auto_reconciles_empty_concerns_without_inapplicable_reasons():
    canonical = row("behavior_contract")
    chunks = pack_section_concerns("behavior_contract")
    chunk_payloads = []
    for index, concern_group in enumerate(chunks):
        payload: dict = {"inapplicable_concerns": []}
        for c in concern_group:
            if c in ("preconditions", "boundaries"):
                # Small model left these empty and forgot to put them in inapplicable_concerns
                payload[c] = []
            elif c == "rejection_postconditions":
                # Small model put a dummy placeholder record
                payload[c] = [{"condition": "", "preserved_state": "", "observation": ""}]
            else:
                payload[c] = canonical["specification"][c]
        if index == 0:
            # Model hallucinated an evidence ref
            payload["constraint_evidence_refs"] = ["allowed_ref_1", "hallucinated_ref"]
        chunk_payloads.append(payload)

    allowed_refs = {"allowed_ref_1"}
    merged = merge_worksheet_section_chunks("behavior_contract", chunk_payloads, allowed_refs)

    # Inapplicable concerns are automatically reconciled for empty concerns
    reconciled_concerns = {item["concern"] for item in merged["specification"]["inapplicable_concerns"]}
    assert "preconditions" in reconciled_concerns
    assert "boundaries" in reconciled_concerns
    assert "rejection_postconditions" in reconciled_concerns

    # Hallucinated evidence ref is filtered out to keep constraint_evidence_refs strictly bounded
    assert merged["constraint_evidence_refs"] == ["allowed_ref_1"]

    # Merged section passes canonical validation without ValueError
    assert validate_worksheet_section(merged, allowed_refs, "behavior_contract") == merged
