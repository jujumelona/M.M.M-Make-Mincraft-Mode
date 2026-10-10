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
    original = {"schema_version": "mmm/generation-work-node-v1", "status": "SUCCEEDED"}
    receipt = _decorate_receipt(
        _row("generate:custom", "generate-custom-00000001"),
        original,
    )
    assert receipt == original
    assert "_mmm_completion_evidence" not in receipt


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


def test_incremental_build_receipt_is_truthful_when_argv_runs_gradle_build():
    receipt = _decorate_receipt(
        _row("build", "build-project"),
        {
            "status": "PASS",
            "build": {
                "status": "PASS",
                "commands": [
                    {
                        "name": "incremental_build",
                        "command": [
                            "/root/.cache/mmm/gradle/bin/gradle",
                            "--daemon",
                            "--parallel",
                            "--max-workers=4",
                            "build",
                            "--build-cache",
                            "--stacktrace",
                        ],
                        "exit_code": 0,
                        "timed_out": False,
                    }
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


def test_incremental_non_build_task_cannot_satisfy_truth_contract():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("build", "build-project"),
            {
                "status": "PASS",
                "build": {
                    "status": "PASS",
                    "commands": [
                        {
                            "name": "incremental_build",
                            "command": ["gradle", "compileJava", "--stacktrace"],
                            "exit_code": 0,
                            "timed_out": False,
                        }
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
                    {"name": "clean_build", "command": ["gradle", "clean", "build"], "exit_code": 0, "timed_out": False}
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


def test_build_artifact_package_rejects_nonpassing_receipt():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("package", "package-build-artifact"),
            {"status": "FAIL", "release_ready": False},
        )


def test_unknown_package_node_fails_closed():
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_UNSUPPORTED_PACKAGE_NODE"):
        _decorate_receipt(_row("package", "unexpected-package-node"), {"status": "PASS"})


@pytest.mark.parametrize(
    "receipt",
    (
        {"status": "PASS"},
        {"status": "PASS", "build_bundle_zip": "/tmp/output.zip"},
        {
            "status": "PASS",
            "build_bundle_zip": "/tmp/output.zip",
            "release_ready": True,
            "unresolved_gates": ["quality:runtime"],
        },
        {
            "status": "PASS",
            "build_bundle_zip": "/tmp/output.zip",
            "release_ready": False,
            "unresolved_gates": [""],
        },
    ),
)
def test_package_attestation_rejects_missing_or_inconsistent_bundle_evidence(receipt):
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(_row("package", "package-build-artifact"), receipt)


def test_unready_build_bundle_is_durable_but_never_a_verified_release():
    receipt = _decorate_receipt(
        _row("package", "package-build-artifact"),
        {
            "status": "PASS",
            "build_bundle_zip": "/tmp/build-artifact.zip",
            "release_ready": False,
            "unresolved_gates": ["quality:runtime"],
        },
    )
    proof = receipt["_mmm_completion_evidence"]
    assert proof["completion_scope"] == "packaging_stage"
    assert proof["build_bundle_zip"] == "/tmp/build-artifact.zip"
    assert proof["release_ready"] is False
    assert proof["unresolved_gates"] == ["quality:runtime"]


def test_ready_build_bundle_requires_no_unresolved_gates():
    receipt = _decorate_receipt(
        _row("package", "package-build-artifact"),
        {
            "status": "PASS",
            "build_bundle_zip": "/tmp/build-artifact.zip",
            "release_ready": True,
            "unresolved_gates": [],
        },
    )
    assert receipt["_mmm_completion_evidence"]["release_ready"] is True


def test_cached_package_evidence_rejects_swapped_bundle_path_or_gate_state():
    row = _row("package", "package-build-artifact")
    original = _decorate_receipt(
        row,
        {
            "status": "PASS",
            "build_bundle_zip": "/tmp/original.zip",
            "release_ready": False,
            "unresolved_gates": ["quality:runtime"],
        },
    )
    assert _decorate_receipt(row, original) == original
    for update in (
        {"build_bundle_zip": "/tmp/other.zip"},
        {"release_ready": True, "unresolved_gates": []},
        {"unresolved_gates": ["quality:security"]},
    ):
        altered = {**original, **update}
        with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
            _decorate_receipt(row, altered)


def test_cached_verifier_receipt_rejects_modified_source_check_count():
    row = _row("validate:source", "validate-source")
    original = _decorate_receipt(row, _source_receipt())
    altered = {**original, "checks_run": 1}
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_STALE"):
        _decorate_receipt(row, altered)



@pytest.mark.parametrize("name", ("build", "clean_build", "incremental_build"))
def test_gradle_build_name_without_executed_task_argv_cannot_certify_release(name):
    from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
    report = {
        "status": "PASS",
        "commands": [{"name": name, "exit_code": 0, "timed_out": False}],
    }
    assert not CompleteProductionOrchestrator._full_gradle_build_receipt_passed(report)
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        _decorate_receipt(
            _row("build", "build-project"),
            {
                "status": "PASS",
                "build": report,
                "final_build_receipt": {
                    "status": "PASS", "production_jar": "PASS",
                    "artifact_sha256": "sha256:jar", "toolchain_attested": True,
                },
            },
        )
