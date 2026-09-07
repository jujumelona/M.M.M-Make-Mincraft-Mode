from __future__ import annotations

from pathlib import Path

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
    assert (
        getattr(production_contract._validate_public_acceptance, "func", None)
        is acceptance_contracts.validate_runtime_public_acceptance
    )


def test_acceptance_policy_has_no_secondary_source_definition() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    owner = package_root / "acceptance_contracts.py"
    forbidden = (
        "PUBLIC_ACCEPTANCE_INTERNAL_MARKERS =",
        "def validate_public_acceptance(",
        "def validate_runtime_public_acceptance(",
        "def is_public_acceptance(",
        "def _is_public_acceptance(",
        "exposes multiple public acceptance contracts",
    )

    for path in package_root.rglob("*.py"):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, (
                f"acceptance policy must be owned only by {owner.name}; "
                f"secondary definition {marker!r} found in {path.relative_to(package_root)}"
            )


def test_acceptance_ssot_drift_is_fail_closed_not_runtime_repaired() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    boundary = (package_root / "production_boundary_contract.py").read_text(
        encoding="utf-8"
    )
    evidence_source = (package_root / "evidence_first_planning.py").read_text(
        encoding="utf-8"
    )
    production_source = (package_root / "production_contract.py").read_text(
        encoding="utf-8"
    )

    assert "_evidence._is_public_acceptance =" not in boundary
    assert "_production._validate_public_acceptance =" not in boundary
    assert "_assert_canonical_acceptance_bindings()" in boundary
    assert (
        "from .acceptance_contracts import is_public_acceptance as _is_public_acceptance"
        in evidence_source
    )
    assert "validate_runtime_public_acceptance" in production_source


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
