from __future__ import annotations

from worksheet_fixtures import specification

import pytest

from minecraft_mod_ai.complete_planner import _design_writing_template
from minecraft_mod_ai.planning_detail_template import (
    CONDITIONAL_WORKSHEET_SECTIONS,
    CORE_WORKSHEET_SECTIONS,
    WORKSHEET_SCHEMA,
    WORKSHEET_SECTIONS,
    normalize_required_sections,
    validate_worksheet,
    worksheet_prompt,
    worksheet_schema,
)


def _row(label: str) -> dict[str, object]:
    return {
        "specification": specification(label),
        "constraint_evidence_refs": ["EVD-1"],
    }


def _core_value() -> dict[str, dict[str, object]]:
    return {key: _row(key) for key in CORE_WORKSHEET_SECTIONS}


def test_default_selection_is_sparse_baseline_worksheet() -> None:
    assert normalize_required_sections() == CORE_WORKSHEET_SECTIONS
    assert tuple(WORKSHEET_SCHEMA["required"]) == CORE_WORKSHEET_SECTIONS
    assert tuple(WORKSHEET_SCHEMA["properties"]) == CORE_WORKSHEET_SECTIONS
    prompt = worksheet_prompt()
    assert f"exactly the {len(CORE_WORKSHEET_SECTIONS)} host-required sections" in prompt
    for key in CORE_WORKSHEET_SECTIONS:
        assert f"- {key}:" in prompt
    for key in CONDITIONAL_WORKSHEET_SECTIONS:
        assert f"- {key}:" not in prompt


def test_only_conditional_sections_can_be_omitted() -> None:
    requested = tuple(reversed(CORE_WORKSHEET_SECTIONS))

    selected = normalize_required_sections(requested)
    schema = worksheet_schema(requested)
    prompt = worksheet_prompt(requested)

    assert selected == tuple(key for key in WORKSHEET_SECTIONS if key in CORE_WORKSHEET_SECTIONS)
    assert tuple(schema["required"]) == selected
    assert tuple(schema["properties"]) == selected
    assert f"exactly the {len(CORE_WORKSHEET_SECTIONS)} host-required sections" in prompt
    for key in CORE_WORKSHEET_SECTIONS:
        assert f"- {key}:" in prompt
    for key in CONDITIONAL_WORKSHEET_SECTIONS:
        assert f"- {key}:" not in prompt


def test_subset_validator_accepts_all_core_sections_without_conditionals() -> None:
    selected = CORE_WORKSHEET_SECTIONS
    value = _core_value()

    validated = validate_worksheet(value, {"EVD-1"}, selected)

    assert tuple(validated) == selected


def test_subset_validator_rejects_missing_required_core_section() -> None:
    selected = CORE_WORKSHEET_SECTIONS
    value = _core_value()
    value.pop("verification")

    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(value, {"EVD-1"}, selected)


def test_subset_validator_rejects_unrequested_conditional_section() -> None:
    value = _core_value()
    value["persistence"] = _row("persistence")

    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(value, {"EVD-1"}, CORE_WORKSHEET_SECTIONS)


def test_baseline_section_cannot_be_omitted_by_any_explicit_selection() -> None:
    for omitted in CORE_WORKSHEET_SECTIONS:
        selection = tuple(key for key in WORKSHEET_SECTIONS if key != omitted)
        with pytest.raises(ValueError, match="baseline section.*cannot be omitted"):
            normalize_required_sections(selection)


@pytest.mark.parametrize(
    "selection, message",
    [
        ((), "cannot be empty"),
        (WORKSHEET_SECTIONS + ("behavior_contract",), "duplicate"),
        (WORKSHEET_SECTIONS + ("made_up_section",), "unknown section"),
    ],
)
def test_invalid_explicit_host_selection_is_rejected(
    selection: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_required_sections(selection)


def test_string_is_not_treated_as_iterable_section_selection() -> None:
    with pytest.raises(ValueError, match="iterable of section names"):
        normalize_required_sections("verification")


def test_default_validator_accepts_sparse_baseline_worksheet() -> None:
    validated = validate_worksheet(_core_value(), {"EVD-1"})

    assert tuple(validated) == CORE_WORKSHEET_SECTIONS


def test_explicit_full_selection_still_rejects_partial_worksheet() -> None:
    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(_core_value(), {"EVD-1"}, WORKSHEET_SECTIONS)

def test_design_writing_template_does_not_split_string_descriptors_into_characters() -> None:
    rendered = _design_writing_template(
        {
            "state_model": {
                "variables": "name, owner, type, unit, default, domain",
                "transitions": ("trigger", "guard", "next_state"),
            }
        }
    )

    assert "variables: name, owner, type, unit, default, domain" in rendered
    assert "variables: n, a, m, e" not in rendered
    assert "transitions: trigger, guard, next_state" in rendered
