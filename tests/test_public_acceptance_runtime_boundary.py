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
