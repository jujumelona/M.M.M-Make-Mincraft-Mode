from __future__ import annotations

import pytest

from minecraft_mod_ai.verifier_receipt_truth_contract import (
    VerifierReceiptTruthError,
    _decorate_receipt,
)


def _row(
    stage: str,
    node_id: str = "node",
    input_hash: str = "sha256:input",
    *,
    payload: dict | None = None,
):
    return {
        "stage": stage,
        "node_id": node_id,
        "input_hash": input_hash,
        "payload": {} if payload is None else payload,
    }


def _source_receipt():
    return {"status": "PASS", "checks_run": 12, "project_manifest": "sha256:tree"}


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
        _source_receipt(),
    )
    evidence = receipt["_mmm_completion_evidence"]
    assert evidence["completion_scope"] == "verified_stage"
    assert evidence["verifier"] == "source_validator"
    assert evidence["checks_run"] == 12


def test_verified_receipt_binds_input_verifier_version_and_config_hashes():
    receipt = _decorate_receipt(
        _row(
            "validate:source",
            "validate-source",
            payload={"target": "fabric", "validator_config": {"strict": True}},
        ),
        _source_receipt(),
    )
    evidence = receipt["_mmm_completion_evidence"]
    assert evidence["schema_version"] == "mmm/work-completion-evidence-v2"
    assert evidence["verifier_input_hash"] == "sha256:input"
    assert evidence["verifier_version_hash"].startswith("sha256:")
    assert evidence["verifier_config_hash"].startswith("sha256:")


def test_unchanged_verified_receipt_is_reusable_idempotently():
    row = _row(
        "validate:source",
        "validate-source",
        payload={"validator_config": {"strict": True}},
    )
    receipt = _decorate_receipt(row, _source_receipt())
    assert _decorate_receipt(row, receipt) == receipt


def test_same_input_hash_with_changed_verifier_config_rejects_stale_receipt():
    original = _decorate_receipt(
        _row(
            "validate:source",
            "validate-source",
            payload={"validator_config": {"strict": True}},
        ),
        _source_receipt(),
    )
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
        _decorate_receipt(
            _row(
                "validate:source",
                "validate-source",
                payload={"validator_config": {"strict": False}},
            ),
            original,
        )


def test_changed_verifier_version_hash_rejects_stale_receipt():
    row = _row("validate:source", "validate-source")
    original = _decorate_receipt(row, _source_receipt())
    tampered = dict(original)
    evidence = dict(tampered["_mmm_completion_evidence"])
    evidence["verifier_version_hash"] = "sha256:old-verifier"
    tampered["_mmm_completion_evidence"] = evidence
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
        _decorate_receipt(row, tampered)


def test_legacy_verified_receipt_without_reuse_fingerprints_fails_closed():
    row = _row("validate:source", "validate-source")
    legacy = _source_receipt()
    legacy["_mmm_completion_evidence"] = {
        "schema_version": "mmm/work-completion-evidence-v1",
        "node_id": "validate-source",
        "stage": "validate:source",
        "input_hash": "sha256:input",
        "completion_scope": "verified_stage",
        "verifier": "source_validator",
    }
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
        _decorate_receipt(row, legacy)


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
