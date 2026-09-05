from __future__ import annotations

import pytest

from minecraft_mod_ai.ledger_traceability import (
    ACCEPTANCE_MANIFEST,
    FAMILY_OWNERS,
    REGRESSION_MANIFEST,
    DecisionReceipt,
    audit_ledger_text,
    validate_decision_receipt,
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


def test_trace_routes_expose_required_evidence_links_without_claiming_execution() -> None:
    regression = REGRESSION_MANIFEST["REG-001"]
    assert regression.triggering_fixture_or_log == "ledger:REG-001"
    assert regression.violated_requirements == ("REQ-LOG-004",)
    assert regression.expected_first_cause_class == "original_primary_failure"
    assert regression.required_acceptance == ("ACC-029",)
    assert regression.execution_status == "planned"

    acceptance = ACCEPTANCE_MANIFEST["ACC-045"]
    assert acceptance.owning_requirements == ("REQ-MADP-004",)
    assert acceptance.verifier_case == "architecture_regression_045"
    assert acceptance.receipt_type == "regression_suite_receipt"
    assert "large-context" in acceptance.pass_predicate
    assert acceptance.applicability == "always"


def test_decision_receipt_requires_repeated_exact_identity_evidence() -> None:
    receipt = DecisionReceipt(
        requirement_id="REQ-MODEL-049",
        alternatives=("native_tool", "two_pass"),
        model_identity="checkpoint:abc+quant:q4+template:def",
        target_identity="minecraft:26.2+fabric:verified",
        runtime_identity="llama.cpp:build-123+kv:q8",
        corpus_version="mmm-heldout-v1",
        repeated_trial_results=("trial-1:pass", "trial-2:pass"),
        failure_taxonomy=("tool_parse_failure=0",),
        selected_option="native_tool",
        rejection_reasons=("two_pass slower with no correctness gain",),
        expiration_or_retest_triggers=("model/template/runtime identity changes",),
    )
    digest = validate_decision_receipt(receipt)
    assert digest.startswith("sha256:")
    assert digest == validate_decision_receipt(receipt)

    invalid = DecisionReceipt(
        requirement_id=receipt.requirement_id,
        alternatives=receipt.alternatives,
        model_identity=receipt.model_identity,
        target_identity=receipt.target_identity,
        runtime_identity=receipt.runtime_identity,
        corpus_version=receipt.corpus_version,
        repeated_trial_results=("trial-1:pass",),
        failure_taxonomy=receipt.failure_taxonomy,
        selected_option=receipt.selected_option,
        rejection_reasons=receipt.rejection_reasons,
        expiration_or_retest_triggers=receipt.expiration_or_retest_triggers,
    )
    with pytest.raises(ValueError, match="DECISION_RECEIPT_REPEATED_TRIALS"):
        validate_decision_receipt(invalid)


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
