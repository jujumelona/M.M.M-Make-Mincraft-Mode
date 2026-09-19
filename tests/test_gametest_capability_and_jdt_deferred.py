from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _unchanged_postbuild_validation,
)
from minecraft_mod_ai.repairability import source_repair_block_reason
from minecraft_mod_ai.runner import CommandResult, GradleRunner
from minecraft_mod_ai.validation_diagnostic_contract import diagnostic_errors


def _command(name: str, log_path: Path) -> CommandResult:
    return CommandResult(
        name=name,
        command=("gradle", name),
        exit_code=0,
        duration_seconds=1.0,
        log_path=str(log_path),
        timed_out=False,
    )


def test_integrated_build_gametest_is_detected_from_real_gradle_log(tmp_path: Path) -> None:
    log = tmp_path / "gradle-build.log"
    log.write_text(
        "> Task :configureLaunch\n"
        "> Task :runGameTest\n"
        "BUILD SUCCESSFUL\n",
        encoding="utf-8",
    )

    assert GradleRunner._executed_gametest_task(_command("build", log)) == "runGameTest"


def test_gametest_task_fallback_uses_gradle_task_listing_not_build_script_hint(
    tmp_path: Path,
) -> None:
    listing = tmp_path / "tasks.log"
    listing.write_text(
        "runGameTest - Runs server game tests\n"
        "runGameTestServer - Legacy compatibility task\n",
        encoding="utf-8",
    )

    assert GradleRunner._task_from_listing(listing) == "runGameTest"


def test_integrated_build_is_valid_gametest_execution_evidence(tmp_path: Path) -> None:
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="MmmDebugFixtureModGameTests.generatedRegistriesAreLive"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    build_log = tmp_path / "gradle-build.log"
    build_log.write_text("> Task :runGameTest\nBUILD SUCCESSFUL\n", encoding="utf-8")
    build = {
        "status": "PASS",
        "gametest_mode": "integrated_build",
        "gametest_task": "runGameTest",
        "gametest_report": str(report),
        "commands": [
            {
                "name": "build",
                "exit_code": 0,
                "timed_out": False,
                "log_path": str(build_log),
            }
        ],
    }

    assert CompleteProductionOrchestrator._gametest_receipt_passed(
        build,
        SimpleNamespace(mod_id="mmm_debug_fixture"),
    )


def test_deferred_jdt_is_not_misclassified_as_unavailable() -> None:
    receipt = {
        "schema_version": "mmm/java-diagnostics-v3",
        "status": "DEFERRED_TO_POST_BUILD",
        "available": False,
        "complete": False,
        "diagnostics": {},
    }

    assert diagnostic_errors(receipt) == []


def test_deferred_jdt_is_refreshed_after_unchanged_successful_build() -> None:
    deferred = {
        "schema_version": "mmm/java-diagnostics-v3",
        "status": "DEFERRED_TO_POST_BUILD",
        "available": False,
        "complete": False,
        "diagnostics": {},
    }
    final = {
        "schema_version": "mmm/java-diagnostics-v3",
        "status": "PASS",
        "available": True,
        "complete": True,
        "diagnostics": {},
        "error_count": 0,
        "files_opened": 1,
    }
    calls = []

    source, jdt, refreshed = _unchanged_postbuild_validation(
        {"status": "PASS"},
        deferred,
        lambda: calls.append(True) or final,
    )

    assert source == {"status": "PASS"}
    assert jdt == final
    assert refreshed is True
    assert calls == [True]


def test_gametest_capability_failure_cannot_enter_source_repair() -> None:
    reason = source_repair_block_reason(
        build={
            "status": "FAIL",
            "failure_class": "verifier",
            "error_code": "GAMETEST_CAPABILITY_MISSING",
            "repairable": False,
            "commands": [],
        }
    )

    assert reason is not None
