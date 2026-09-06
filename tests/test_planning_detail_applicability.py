from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_detail_applicability import (
    APPLICABILITY_FIELD,
    normalize_detail_section_applicability,
    required_detail_sections_for_requirement,
    required_sections_by_requirement,
)
from minecraft_mod_ai.planning_detail_template import (
    CORE_WORKSHEET_SECTIONS,
    WORKSHEET_SECTIONS,
)


def _requirement(
    requirement_id: str = "REQ-1",
    *,
    applicability: dict[str, str] | None = None,
    statement: str = "Do one bounded thing.",
) -> dict[str, object]:
    row: dict[str, object] = {
        "decision_type": "requirement",
        "requirement_id": requirement_id,
        "statement": statement,
    }
    if applicability is not None:
        row[APPLICABILITY_FIELD] = applicability
    return row


def test_missing_applicability_keeps_full_fail_safe_contract() -> None:
    assert required_detail_sections_for_requirement(_requirement()) == WORKSHEET_SECTIONS


def test_unknown_applicability_keeps_conditional_sections() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "unknown",
                "persistence": "unknown",
                "resources_and_ui": "unknown",
            }
        )
    )

    assert selected == WORKSHEET_SECTIONS


def test_only_explicit_not_applicable_omits_conditional_sections() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "not_applicable",
                "persistence": "not_applicable",
                "resources_and_ui": "not_applicable",
            }
        )
    )

    assert selected == CORE_WORKSHEET_SECTIONS


def test_required_and_unknown_conditionals_are_retained() -> None:
    selected = required_detail_sections_for_requirement(
        _requirement(
            applicability={
                "authority_and_network": "required",
                "persistence": "not_applicable",
                "resources_and_ui": "unknown",
            }
        )
    )

    assert "authority_and_network" in selected
    assert "resources_and_ui" in selected
    assert "persistence" not in selected
    assert all(section in selected for section in CORE_WORKSHEET_SECTIONS)


def test_prompt_wording_never_omits_a_section() -> None:
    requirement = _requirement(
        statement=(
            "This is local-only, has no networking, never persists, and needs no UI or resources."
        )
    )

    assert required_detail_sections_for_requirement(requirement) == WORKSHEET_SECTIONS


def test_state_projection_uses_each_requirement_owned_applicability() -> None:
    state = {
        "decisions": [
            _requirement(
                "REQ-1",
                applicability={
                    "authority_and_network": "not_applicable",
                    "persistence": "not_applicable",
                    "resources_and_ui": "not_applicable",
                },
            ),
            _requirement("REQ-2"),
        ]
    }

    assert required_sections_by_requirement(state) == {
        "REQ-1": CORE_WORKSHEET_SECTIONS,
        "REQ-2": WORKSHEET_SECTIONS,
    }


def test_missing_status_keys_fail_closed_to_unknown() -> None:
    normalized = normalize_detail_section_applicability(
        {"persistence": "not_applicable"}
    )

    assert normalized == {
        "authority_and_network": "unknown",
        "persistence": "not_applicable",
        "resources_and_ui": "unknown",
    }


@pytest.mark.parametrize(
    "value, message",
    [
        (["persistence"], "section mapping"),
        ({"made_up": "not_applicable"}, "unknown conditional section"),
        ({"persistence": "optional"}, "invalid status"),
    ],
)
def test_invalid_host_applicability_is_rejected(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_detail_section_applicability(value)
