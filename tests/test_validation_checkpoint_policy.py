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
        {"status": "PASS"},
    )
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-source",
        {"status": "FAIL"},
    )


def test_jdt_resume_never_reuses_unavailable_result() -> None:
    assert not validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        {"status": "UNAVAILABLE", "error": "jdtls missing"},
    )
    assert validation_checkpoint_policy.cached_validation_is_reusable(
        "validate-jdt",
        {
            "schema_version": "mmm/java-diagnostics-v2",
            "diagnostics": {},
        },
    )


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
