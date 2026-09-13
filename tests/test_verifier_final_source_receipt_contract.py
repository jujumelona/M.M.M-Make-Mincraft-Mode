from __future__ import annotations

import pytest

from minecraft_mod_ai.verifier_receipt_truth_contract import (
    VerifierReceiptTruthError,
    _decorate_receipt,
)


def _row() -> dict[str, object]:
    return {
        "node_id": "validate-source-final",
        "stage": "validate:source-final",
        "input_hash": "sha256:final-source-input",
        "payload": {"kind": "final-source-validation"},
    }


def test_final_source_validation_uses_source_verifier_truth_contract() -> None:
    decorated = _decorate_receipt(
        _row(),
        {
            "status": "PASS",
            "checks_run": 7,
            "project_manifest": "sha256:final-source-manifest",
        },
    )

    evidence = decorated["_mmm_completion_evidence"]
    assert evidence["stage"] == "validate:source-final"
    assert evidence["completion_scope"] == "verified_stage"
    assert evidence["verifier"] == "source_validator"
    assert evidence["checks_run"] == 7
    assert evidence["project_manifest"] == "sha256:final-source-manifest"
    assert str(evidence["verifier_version_hash"]).startswith("sha256:")
    assert str(evidence["verifier_config_hash"]).startswith("sha256:")


def test_final_source_validation_still_fails_closed_without_evidence() -> None:
    with pytest.raises(
        VerifierReceiptTruthError,
        match="source validation requires positive checks_run and project_manifest",
    ):
        _decorate_receipt(
            _row(),
            {
                "status": "PASS",
                "checks_run": 0,
                "project_manifest": "",
            },
        )
