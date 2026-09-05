from __future__ import annotations

import pytest

from minecraft_mod_ai.verifier_receipt_truth_contract import (
    VerifierReceiptTruthError,
    _decorate_receipt,
)


def _row(stage: str, node_id: str = "node", input_hash: str = "sha256:input"):
    return {"stage": stage, "node_id": node_id, "input_hash": input_hash}


def test_generation_success_is_explicitly_phase_only_not_verified():
    receipt = _decorate_receipt(
        _row("generate:custom", "generate-custom-00000001"),
        {"schema_version": "mmm/generation-work-node-v1", "status": "SUCCEEDED"},
    )
    evidence = receipt["_mmm_completion_evidence"]
    assert evidence["completion_scope"] == "phase_only"
    assert evidence["verifier"] is None
    assert evidence["input_hash"] == "sha256:input"


def test_source_validation_pass_requires_actual_checks_and_snapshot_manifest():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("validate:source", "validate-source"),
            {"status": "PASS", "checks_run": 0},
        )

    receipt = _decorate_receipt(
        _row("validate:source", "validate-source"),
        {"status": "PASS", "checks_run": 12, "project_manifest": "sha256:tree"},
    )
    evidence = receipt["_mmm_completion_evidence"]
    assert evidence["completion_scope"] == "verified_stage"
    assert evidence["verifier"] == "source_validator"
    assert evidence["checks_run"] == 12


def test_jar_validation_pass_requires_independent_jar_receipt_identity():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("validate:jar", "validate-jar"),
            {"status": "PASS", "checks_run": 3},
        )

    receipt = _decorate_receipt(
        _row("validate:jar", "validate-jar"),
        {"status": "PASS", "checks_run": 3, "jar_sha256": "sha256:abc"},
    )
    assert receipt["_mmm_completion_evidence"]["artifact_sha256"] == "sha256:abc"


def test_build_pass_cannot_be_invented_without_command_and_artifact_receipts():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("build", "build-project"),
            {"status": "PASS", "build": {"status": "PASS"}},
        )

    receipt = _decorate_receipt(
        _row("build", "build-project"),
        {
            "status": "PASS",
            "build": {
                "status": "PASS",
                "commands": [
                    {"name": "clean_build", "exit_code": 0, "timed_out": False}
                ],
                "artifact_receipt": {"sha256": "sha256:jar"},
            },
            "final_build_receipt": {
                "status": "PASS",
                "production_jar": "PASS",
                "artifact_sha256": "sha256:jar",
                "toolchain_attested": True,
            },
        },
    )
    evidence = receipt["_mmm_completion_evidence"]
    assert evidence["verifier"] == "gradle_and_final_artifact"
    assert evidence["artifact_sha256"] == "sha256:jar"


def test_stale_completion_evidence_is_rejected_for_changed_input_hash():
    original = _decorate_receipt(
        _row("validate:source", "validate-source", "sha256:one"),
        {"status": "PASS", "checks_run": 1, "project_manifest": "sha256:tree"},
    )
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
        _decorate_receipt(
            _row("validate:source", "validate-source", "sha256:two"),
            original,
        )


def test_quality_pass_requires_verification_receipt_hash():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("validate:quality", "validate-quality-correctness"),
            {"status": "PASS", "dimension_id": "correctness", "receipt_id": "r1"},
        )

    receipt = _decorate_receipt(
        _row("validate:quality", "validate-quality-correctness"),
        {
            "status": "PASS",
            "dimension_id": "correctness",
            "receipt_id": "r1",
            "receipt_sha256": "sha256:quality",
        },
    )
    assert receipt["_mmm_completion_evidence"]["receipt_sha256"] == "sha256:quality"
