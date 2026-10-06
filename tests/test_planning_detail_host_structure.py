from __future__ import annotations

from minecraft_mod_ai.planning_detail_template import (
    validate_worksheet_section,
    worksheet_section_schema,
)

from tests.worksheet_fixtures import specification


def test_worksheet_evidence_schema_does_not_delegate_uniqueness_to_model() -> None:
    schema = worksheet_section_schema("resources_and_ui")
    evidence_schema = schema["properties"]["constraint_evidence_refs"]

    assert evidence_schema["type"] == "array"
    assert "uniqueItems" not in evidence_schema


def test_worksheet_evidence_duplicates_are_normalized_by_host() -> None:
    value = {
        "specification": specification("resources_and_ui"),
        "constraint_evidence_refs": ["evidence_a", "evidence_a"],
    }

    normalized = validate_worksheet_section(
        value,
        {"evidence_a"},
        "resources_and_ui",
    )

    assert normalized["constraint_evidence_refs"] == ["evidence_a"]
