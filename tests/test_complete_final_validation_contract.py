from __future__ import annotations

from minecraft_mod_ai.complete_orchestrator import (
    _blocking_jdt_errors,
    _final_validation_failure,
)


def _jdt_error_receipt() -> dict:
    return {
        "status": "PASS",
        "diagnostics": {
            "file:///src/main/java/demo/Feature.java": [
                {
                    "severity": 1,
                    "source": "jdtls",
                    "code": "compiler.err.cant.resolve.location",
                    "message": "The method missing() is undefined for the type Feature",
                }
            ]
        },
    }


def test_final_source_validation_failure_cannot_be_marked_pass() -> None:
    assert _final_validation_failure(
        source_report={"status": "FAIL"},
        jdt_receipt=None,
        run_jdt=False,
    ) == "Final project failed deterministic source validation after build/repair."


def test_final_jdt_source_error_blocks_success_even_after_gradle_build() -> None:
    receipt = _jdt_error_receipt()

    assert _blocking_jdt_errors(receipt)
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=receipt,
        run_jdt=True,
    ) == "Final JDT validation still reports source errors after build/repair."


def test_jdt_infrastructure_unavailable_is_not_misclassified_as_source_error() -> None:
    receipt = {
        "status": "UNAVAILABLE",
        "error": "RuntimeError: JDT workspace unavailable",
        "diagnostics": {},
    }

    assert _blocking_jdt_errors(receipt) == []
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=receipt,
        run_jdt=True,
    ) is None


def test_jdt_receipt_is_ignored_when_jdt_was_not_requested() -> None:
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=_jdt_error_receipt(),
        run_jdt=False,
    ) is None
