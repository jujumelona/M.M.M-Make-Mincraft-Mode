from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import authored_structured_design as design
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
)


class _MinimumCountRouter:
    def __init__(self):
        self.schemas = []

    def generate_text(self, role, messages, **kwargs):
        schema = kwargs["response_schema"]
        self.schemas.append(schema)
        return json.dumps({
            concern: constraint["enum"][0]
            for concern, constraint in schema["properties"].items()
        })


@pytest.mark.parametrize("section, required", [
    ("behavior_contract", {"actors", "entry_conditions", "outputs", "success_postconditions"}),
    ("integration", {"entry_points", "responsibilities"}),
    ("verification", {"success_cases"}),
])
def test_core_gameplay_count_selection_cannot_erase_feature(section, required):
    router = _MinimumCountRouter()
    result = design._generate_section_record_counts(
        router, "Add a usable player teleport action", section=section, completed={},
    )
    for concern in DETAIL_RECORDS[section]:
        assert result[concern] == (1 if concern in required else 0)
        allowed = router.schemas[0]["properties"][concern]["enum"]
        assert ("0" not in allowed) == (concern in required)


def test_unneeded_state_model_can_select_zero_without_adding_fake_state():
    router = _MinimumCountRouter()
    result = design._generate_section_record_counts(
        router, "Add a simple static decorative block", section="state_model",
        completed={},
    )
    assert all(count == 0 for count in result.values())


def _empty_chunks(section):
    return [{str(page[0]): []} for page in pack_section_concerns(section)]


def test_optional_empty_worksheet_does_not_crash_merger():
    section = "state_model"
    merged = merge_worksheet_section_chunks(section, _empty_chunks(section), set())
    assert all(merged["specification"][key] == [] for key in DETAIL_RECORDS[section])
    assert merged["constraint_evidence_refs"] == []


@pytest.mark.parametrize("section", ("behavior_contract", "integration", "verification"))
def test_empty_core_worksheet_is_still_rejected(section):
    with pytest.raises(ValueError, match="contains no gameplay records"):
        merge_worksheet_section_chunks(section, _empty_chunks(section), set())
