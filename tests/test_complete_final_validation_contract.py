from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _blocking_jdt_errors,
    _final_validation_failure,
    _gametest_attestation_status,
    _runtime_verification_passed,
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


def test_cached_build_requires_real_build_command_and_requested_gametest(tmp_path) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "jar_path": str(jar),
        "commands": [
            {"name": "build", "exit_code": 0, "timed_out": False},
            {"name": "gametest", "exit_code": 0, "timed_out": False},
        ],
        "gametest_report": str(report),
    }

    assert CompleteProductionOrchestrator._cached_build_exists(build)
    assert CompleteProductionOrchestrator._cached_build_exists(
        build,
        require_gametest=True,
        spec=SimpleNamespace(mod_id="demo"),
    )

    no_build_receipt = {**build, "commands": build["commands"][1:]}
    assert not CompleteProductionOrchestrator._cached_build_exists(no_build_receipt)

    missing_gametest = {**build, "gametest_report": None}
    assert not CompleteProductionOrchestrator._cached_build_exists(
        missing_gametest,
        require_gametest=True,
        spec=SimpleNamespace(mod_id="demo"),
    )


def test_required_runtime_needs_live_client_playtest_and_visual_evidence() -> None:
    runtime = {
        "status": "PASS",
        "server": {"server_running": True},
        "client": {"client_running": True},
    }
    playtest = {
        "status": "PASS",
        "interaction_count": 1,
        "assertion_count": 1,
    }
    visual = {"status": "PASS"}

    assert _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt={**runtime, "client": None},
        playtest_receipt=playtest,
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt={**playtest, "assertion_count": 0},
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt={"status": "FAIL"},
    )


def test_optional_runtime_is_not_required_for_release_readiness() -> None:
    assert _runtime_verification_passed(
        required=False,
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )
