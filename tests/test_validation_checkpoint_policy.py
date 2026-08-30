from __future__ import annotations

import pytest

from minecraft_mod_ai import validation_checkpoint_policy


def test_validation_resume_reuses_only_stable_exact_results() -> None:
    scoped = validation_checkpoint_policy.validation_checkpoint_input(
        "validate-source",
        {"graph_hash": "g", "project_manifest": "m"},
    )
    assert scoped["graph_hash"] == "g"
    assert scoped["project_manifest"] == "m"
    assert str(scoped["_mmm_validation_implementation"]).startswith("sha256:")
    assert validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-source",
        {"status": "PASS", "checks_run": 3, "findings": []},
    )
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-source",
        {"status": "PASS"},
    )
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-source",
        {
            "status": "PASS",
            "checks_run": 3,
            "findings": [{"severity": "error", "code": "BAD"}],
        },
    )
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-source",
        {"status": "FAIL", "checks_run": 3, "findings": []},
    )


def _complete_jdt_receipt() -> dict[str, object]:
    return {
        "schema_version": "mmm/java-diagnostics-v2",
        "files_opened": 2,
        "page_count": 1,
        "pages": [
            {
                "page_index": 0,
                "file_count": 2,
                "diagnostic_uri_count": 2,
                "error_count": 1,
                "warning_count": 0,
            }
        ],
        "error_count": 1,
        "warning_count": 0,
        "diagnostics": {
            "file:///A.java": [],
            "file:///B.java": [{"severity": 1, "message": "cannot find symbol"}],
        },
    }


def test_jdt_resume_never_reuses_unavailable_or_partial_result() -> None:
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        {"status": "UNAVAILABLE", "error": "jdtls missing"},
    )

    complete = _complete_jdt_receipt()
    assert validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        complete,
    )

    partial = _complete_jdt_receipt()
    partial["diagnostics"] = {"file:///A.java": []}
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        partial,
    )

    inconsistent_counts = _complete_jdt_receipt()
    inconsistent_counts["error_count"] = 0
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        inconsistent_counts,
    )


def test_jdt_resume_cross_checks_orchestrator_transformed_receipt() -> None:
    transformed = _complete_jdt_receipt()
    diagnostics_by_uri = transformed["diagnostics"]
    assert isinstance(diagnostics_by_uri, dict)
    expected_error = diagnostics_by_uri["file:///B.java"][0]
    transformed["diagnostics_by_uri"] = diagnostics_by_uri
    transformed["diagnostics"] = [expected_error]
    assert validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        transformed,
    )

    missing_legacy_error = _complete_jdt_receipt()
    missing_legacy_error["diagnostics_by_uri"] = missing_legacy_error["diagnostics"]
    missing_legacy_error["diagnostics"] = []
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        missing_legacy_error,
    )


def test_jdt_fingerprint_covers_every_runtime_owner() -> None:
    names = {
        module.__name__
        for module in validation_checkpoint_policy._validation_modules("validate-jdt")
    }
    assert {
        "minecraft_mod_ai.java_lsp",
        "minecraft_mod_ai.java_lsp_process_safety_contract",
        "minecraft_mod_ai.validation_diagnostic_contract",
        "minecraft_mod_ai.validation_execution_contract",
        "minecraft_mod_ai.research_validation_fingerprint_performance",
    } <= names


def test_validation_checkpoint_scope_changes_with_mmm_runtime_policy(monkeypatch) -> None:
    monkeypatch.setenv("MMM_VALIDATION_CACHE_TEST_SCOPE", "one")
    first = validation_checkpoint_policy.validation_implementation_fingerprint(
        "validate-jdt"
    )
    monkeypatch.setenv("MMM_VALIDATION_CACHE_TEST_SCOPE", "two")
    second = validation_checkpoint_policy.validation_implementation_fingerprint(
        "validate-jdt"
    )
    assert first != second


def test_validation_checkpoint_policy_rejects_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unsupported validation checkpoint"):
        validation_checkpoint_policy.validation_checkpoint_input(
            "validate-unknown",
            {"graph_hash": "g"},
        )
