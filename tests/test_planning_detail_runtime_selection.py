from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_template import (
    CORE_WORKSHEET_SECTIONS,
    WORKSHEET_SECTIONS,
)
from minecraft_mod_ai.planning_state_implementation import (
    _host_section_selection,
    _parameters_for_sections,
)


def _requirements() -> list[dict[str, str]]:
    return [
        {"requirement_id": "REQ-1"},
        {"requirement_id": "REQ-2"},
    ]


def test_dynamic_tool_schema_matches_exact_host_selection() -> None:
    parameters = _parameters_for_sections(CORE_WORKSHEET_SECTIONS)
    worksheet = parameters["properties"]["engineering_worksheet"]

    assert tuple(worksheet["required"]) == CORE_WORKSHEET_SECTIONS
    assert tuple(worksheet["properties"]) == CORE_WORKSHEET_SECTIONS
    assert parameters["additionalProperties"] is False


def test_dynamic_tool_schema_defaults_to_full_fail_safe_contract() -> None:
    parameters = _parameters_for_sections(None)
    worksheet = parameters["properties"]["engineering_worksheet"]

    assert tuple(worksheet["required"]) == WORKSHEET_SECTIONS
    assert tuple(worksheet["properties"]) == WORKSHEET_SECTIONS


def test_host_selection_defaults_every_requirement_to_full_contract() -> None:
    selected = _host_section_selection(_requirements(), None)

    assert selected == {
        "REQ-1": WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }


def test_host_selection_can_narrow_one_requirement_without_weakening_others() -> None:
    selected = _host_section_selection(
        _requirements(),
        {"REQ-1": tuple(reversed(CORE_WORKSHEET_SECTIONS))},
    )

    assert selected["REQ-1"] == CORE_WORKSHEET_SECTIONS
    assert selected["REQ-2"] == WORKSHEET_SECTIONS


def test_host_selection_rejects_unknown_requirement() -> None:
    with pytest.raises(ValueError, match="unknown requirement"):
        _host_section_selection(
            _requirements(),
            {"REQ-NOT-REAL": CORE_WORKSHEET_SECTIONS},
        )


def test_host_selection_cannot_omit_core_section() -> None:
    selection = tuple(
        key for key in WORKSHEET_SECTIONS if key != "verification"
    )

    with pytest.raises(ValueError, match="core section.*cannot be omitted"):
        _host_section_selection(_requirements(), {"REQ-1": selection})


def test_host_selection_rejects_model_shaped_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="host selection must be a requirement mapping"):
        _host_section_selection(_requirements(), [CORE_WORKSHEET_SECTIONS])
