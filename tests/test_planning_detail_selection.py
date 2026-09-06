from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_template import (
    WORKSHEET_SCHEMA,
    WORKSHEET_SECTIONS,
    normalize_required_sections,
    validate_worksheet,
    worksheet_prompt,
    worksheet_schema,
)


def _row(label: str) -> dict[str, object]:
    return {
        "specification": f"Concrete {label} contract with observable behavior and bounded outcomes.",
        "evidence_refs": ["EVD-1"],
    }


def test_default_selection_remains_fail_safe_full_worksheet() -> None:
    assert normalize_required_sections() == WORKSHEET_SECTIONS
    assert tuple(WORKSHEET_SCHEMA["required"]) == WORKSHEET_SECTIONS
    assert set(WORKSHEET_SCHEMA["properties"]) == set(WORKSHEET_SECTIONS)
    assert "Fill all ten sections" in worksheet_prompt()


def test_explicit_selection_is_canonical_and_schema_contains_only_selected() -> None:
    requested = ("verification", "behavior_contract", "integration")

    selected = normalize_required_sections(requested)
    schema = worksheet_schema(requested)
    prompt = worksheet_prompt(requested)

    assert selected == ("behavior_contract", "integration", "verification")
    assert tuple(schema["required"]) == selected
    assert tuple(schema["properties"]) == selected
    assert "exactly the 3 host-required sections" in prompt
    assert "- behavior_contract:" in prompt
    assert "- integration:" in prompt
    assert "- verification:" in prompt
    assert "- persistence:" not in prompt
    assert "- authority_and_network:" not in prompt


def test_subset_validator_accepts_only_explicit_host_required_sections() -> None:
    selected = ("behavior_contract", "verification")
    value = {
        "behavior_contract": _row("behavior"),
        "verification": _row("verification"),
    }

    validated = validate_worksheet(value, {"EVD-1"}, selected)

    assert tuple(validated) == selected


def test_subset_validator_rejects_missing_required_section() -> None:
    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(
            {"behavior_contract": _row("behavior")},
            {"EVD-1"},
            ("behavior_contract", "verification"),
        )


def test_subset_validator_rejects_unrequested_section() -> None:
    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(
            {
                "behavior_contract": _row("behavior"),
                "verification": _row("verification"),
                "persistence": _row("persistence"),
            },
            {"EVD-1"},
            ("behavior_contract", "verification"),
        )


@pytest.mark.parametrize(
    "selection, message",
    [
        ((), "cannot be empty"),
        (("behavior_contract", "behavior_contract"), "duplicate"),
        (("behavior_contract", "made_up_section"), "unknown section"),
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


def test_default_validator_still_rejects_partial_worksheet() -> None:
    with pytest.raises(ValueError, match="host-required engineering sections"):
        validate_worksheet(
            {
                "behavior_contract": _row("behavior"),
                "verification": _row("verification"),
            },
            {"EVD-1"},
        )
