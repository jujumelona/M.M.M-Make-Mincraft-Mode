from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _blocking_jdt_errors,
    _final_validation_failure,
    _gametest_attestation_status,
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


def test_requested_gametest_requires_structured_matching_evidence(tmp_path) -> None:
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "commands": [
            {"name": "build", "exit_code": 0, "timed_out": False},
            {"name": "gametest", "exit_code": 0, "timed_out": False},
        ],
        "gametest_report": str(report),
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "PASS"


def test_requested_gametest_without_report_is_not_attested() -> None:
    build = {
        "status": "PASS",
        "commands": [{"name": "gametest", "exit_code": 0, "timed_out": False}],
        "gametest_report": None,
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "NO_EVIDENCE"
    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=False,
    ) == "NOT_REQUIRED"


def test_execute_wires_required_gate_failures_into_both_release_paths() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert source.count("self._required_gate_failures(") == 2
    assert "build_report=None" in source
    assert "build_report=build" in source
    assert source.count("unresolved.extend(") >= 2


def test_optional_runtime_checks_do_not_create_unresolved_release_gates() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('runtime:not-requested')"
    ) in source
    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('mineflayer:not-requested')"
    ) in source
    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('visual-review:not-requested')"
    ) in source
