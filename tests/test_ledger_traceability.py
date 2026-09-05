from __future__ import annotations

from minecraft_mod_ai.ledger_traceability import (
    ACCEPTANCE_MANIFEST,
    FAMILY_OWNERS,
    REGRESSION_MANIFEST,
    audit_ledger_text,
    validate_manifest_snapshot,
)


def _ledger_fixture(*, include_owner_row: bool = True) -> str:
    owner_row = (
        "| `REQ-OWNER` | Ledger self-audit / architecture boundary governance | all owners |\n"
        if include_owner_row
        else ""
    )
    return f"""
**REQ-GOV-001 — FIXED**
**REQ-OWNER-001 — FIXED**
**REQ-TRACE-001 — FIXED**
**REG-001**
`REG-SEM-001`
**ACC-001 — Planner/domain/linker acceptance**
**ACC-064 — epistemic self-audit**

## 36.25 Requirement-family ownership matrix
| Requirement family | Primary architectural owner | Mandatory collaborators |
|---|---|---|
| `REQ-GOV` | Orchestrator / ledger governance | all layers |
| `REQ-TRACE` | Regression/Acceptance/Decision trace registry | benchmark |
{owner_row}
## 36.26 Regression, acceptance, and decision traceability
"""


def test_manifest_snapshot_covers_all_current_ledger_routes() -> None:
    validate_manifest_snapshot()
    assert len(FAMILY_OWNERS) == 62
    assert len(REGRESSION_MANIFEST) == 39
    assert set(ACCEPTANCE_MANIFEST) == {
        f"ACC-{value:03d}" for value in range(1, 65)
    }
    assert "REG-037" in REGRESSION_MANIFEST
    assert "REG-SEM-001" in REGRESSION_MANIFEST
    assert "REG-DESIGN-001" in REGRESSION_MANIFEST
    assert "REQ-OWNER" in FAMILY_OWNERS


def test_audit_accepts_routed_fixture() -> None:
    report = audit_ledger_text(_ledger_fixture())
    assert report.ok, report.issues
    assert report.requirement_count == 3
    assert report.family_count == 3
    assert report.regression_count == 2
    assert report.acceptance_count == 2


def test_audit_rejects_requirement_family_missing_from_ledger_owner_matrix() -> None:
    report = audit_ledger_text(_ledger_fixture(include_owner_row=False))
    assert not report.ok
    assert any(
        issue == "requirement families missing ownership-matrix row: REQ-OWNER"
        for issue in report.issues
    )


def test_audit_rejects_unknown_regression_and_acceptance_ids() -> None:
    text = _ledger_fixture() + "\n`REG-999`\n`ACC-999`\n"
    report = audit_ledger_text(text)
    assert not report.ok
    assert any("REG-999" in issue for issue in report.issues)
    assert any("ACC-999" in issue for issue in report.issues)


def test_audit_rejects_duplicate_requirement_ids() -> None:
    text = _ledger_fixture() + "\n**REQ-GOV-001 — FIXED** duplicate\n"
    report = audit_ledger_text(text)
    assert not report.ok
    assert any("duplicate requirement IDs: REQ-GOV-001" == issue for issue in report.issues)
