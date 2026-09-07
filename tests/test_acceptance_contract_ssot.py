from __future__ import annotations

import pytest

from minecraft_mod_ai import acceptance_contracts, production_contract
from minecraft_mod_ai import evidence_first_planning as evidence
from minecraft_mod_ai.production_boundary_contract import _approved_acceptance


def test_runtime_public_acceptance_policy_has_one_owner() -> None:
    assert evidence._is_public_acceptance is acceptance_contracts.is_public_acceptance
    assert (
        getattr(
            production_contract._validate_public_acceptance,
            "_mmm_acceptance_contract_owner",
            "",
        )
        == acceptance_contracts.CANONICAL_ACCEPTANCE_OWNER
    )


@pytest.mark.parametrize("count", [2, 3, 4, 5, 16])
def test_multi_check_requirement_preserves_every_public_contract(count: int) -> None:
    checks = [f"Observable acceptance check {index}." for index in range(count)]
    requirement = {
        "requirement_id": "req_multi",
        "acceptance": checks,
    }

    assert _approved_acceptance(requirement) == "; ".join(checks)


def test_internal_acceptance_is_rejected_by_every_runtime_consumer() -> None:
    dirty = "Verify task_player_events and all declared provides before success."

    assert acceptance_contracts.is_public_acceptance(dirty) is False
    assert evidence._is_public_acceptance(dirty) is False
    with pytest.raises(production_contract.ProductionContractError):
        production_contract._validate_public_acceptance(dirty)
