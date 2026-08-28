from __future__ import annotations

import pytest

import minecraft_mod_ai.evidence_first_planning as evidence
from minecraft_mod_ai import production_contract as production
from minecraft_mod_ai.production_boundary_contract import (
    _approved_acceptance,
    _filter_evidence_input_acceptance,
)


def test_evidence_planner_uses_production_public_acceptance_guard() -> None:
    assert getattr(
        evidence._is_public_acceptance,
        "_mmm_production_public_acceptance_guard",
        False,
    )

    clean = (
        "Test the requested behavior for player events in a representative "
        "Minecraft scenario and record the observable result."
    )
    dirty = (
        "Test task_support_player_events in a representative Minecraft scenario "
        "and record the observable result."
    )

    assert evidence._is_public_acceptance(clean) is True
    assert evidence._is_public_acceptance(dirty) is False
    with pytest.raises(
        production.ProductionContractError,
        match="public acceptance contains internal task or integrity language",
    ):
        production._validate_public_acceptance(dirty)


@pytest.mark.parametrize(
    "statement",
    [
        "Verify task_sha256 matches the generated behavior in a representative scenario.",
        "Verify required_gates before recording the requested observable behavior.",
        "Verify declared_provides before recording the requested observable behavior.",
        "Verify owned_anchor remains valid before recording the requested observable behavior.",
        "Verify done_predicate before recording the requested observable behavior.",
    ],
)
def test_planner_and_production_reject_the_same_internal_acceptance(
    statement: str,
) -> None:
    assert evidence._is_public_acceptance(statement) is False
    with pytest.raises(production.ProductionContractError):
        production._validate_public_acceptance(statement)


def test_evidence_mode_filters_only_non_authoritative_internal_input_acceptance() -> None:
    clean = "Verify the requested block can be placed and its observable result is recorded."
    dirty = "Verify task_place_block can be placed and its observable result is recorded."
    acceptance = (dirty, clean)

    assert _filter_evidence_input_acceptance(acceptance, {}) == (clean,)
    assert _filter_evidence_input_acceptance(acceptance, None) == acceptance


def test_canonical_requirement_acceptance_remains_fail_closed() -> None:
    dirty_requirement = {
        "requirement_id": "requirement_player_events",
        "acceptance": [
            "Verify task_player_events produces the requested observable result."
        ],
    }

    with pytest.raises(
        production.ProductionContractError,
        match="approved public acceptance contains internal task/integrity language",
    ):
        _approved_acceptance(dirty_requirement)

def test_evidence_plan_public_acceptance_is_sanitized_before_production_compile():
    from minecraft_mod_ai.production_boundary_contract import (
        _sanitize_evidence_plan_public_acceptance,
    )

    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_visible",
                    "source_span": {"text": "Show the requested visible behavior"},
                    "acceptance": [
                        "The requested block is visible in game; task_sha256 must match"
                    ],
                }
            ]
        },
        "acceptance_release_bindings": [
            {
                "requirement_ref": "req_visible",
                "acceptance": ["required_gates must pass"],
            }
        ],
    }

    sanitized = _sanitize_evidence_plan_public_acceptance(plan)
    expected = ["The requested block is visible in game"]

    assert sanitized["request_catalog"]["requirements"][0]["acceptance"] == expected
    assert sanitized["acceptance_release_bindings"][0]["acceptance"] == expected


def test_public_acceptance_sanitizer_falls_back_without_internal_integrity_language():
    from minecraft_mod_ai.production_boundary_contract import (
        _sanitize_evidence_plan_public_acceptance,
    )
    from minecraft_mod_ai.production_contract import _validate_public_acceptance

    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_fallback",
                    "source_span": {"text": "Show a visible block in game"},
                    "acceptance": ["task_sha256 must match required_gates"],
                }
            ]
        },
        "acceptance_release_bindings": [
            {
                "requirement_ref": "req_fallback",
                "acceptance": ["done_predicate must pass"],
            }
        ],
    }

    sanitized = _sanitize_evidence_plan_public_acceptance(plan)
    values = sanitized["request_catalog"]["requirements"][0]["acceptance"]
    assert values
    for value in values:
        _validate_public_acceptance(value)
    assert sanitized["acceptance_release_bindings"][0]["acceptance"] == values

