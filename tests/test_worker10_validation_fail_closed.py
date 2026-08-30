from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.repair_engine import RepairEngine
from minecraft_mod_ai.validation_diagnostic_contract import diagnostic_errors


def test_jdt_unavailable_receipt_is_a_blocking_validation_error() -> None:
    errors = diagnostic_errors(
        {
            "status": "UNAVAILABLE",
            "error": "TimeoutError: JDT LS did not become ready",
            "diagnostics": {},
        }
    )
    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert errors[0]["severity"] == 1
    assert "UNAVAILABLE" in errors[0]["message"]


def test_malformed_jdt_receipt_fails_closed() -> None:
    errors = diagnostic_errors({"schema_version": "mmm/java-diagnostics-v2"})
    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "malformed" in errors[0]["message"]


def test_jdt_mapping_keeps_warnings_non_blocking() -> None:
    errors = diagnostic_errors(
        {
            "diagnostics": {
                "file:///Main.java": [
                    {"severity": 1, "message": "compile error"},
                    {"severity": 2, "message": "warning"},
                ]
            }
        }
    )
    assert [item["message"] for item in errors] == ["compile error"]


def test_progressive_repair_does_not_start_gradle_when_jdt_is_unavailable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    class Diagnostics:
        def diagnostics(self, *_args, **_kwargs):
            return {
                "status": "UNAVAILABLE",
                "error": "RuntimeError: jdtls missing",
                "diagnostics": {},
            }

    class Runner:
        def __init__(self, _cache):
            raise AssertionError("Gradle must not start without usable JDT evidence")

    repair = RepairEngine(
        router=SimpleNamespace(),
        gradle_cache=tmp_path / "cache",
        diagnostics_factory=Diagnostics,
        runner_factory=Runner,
    )
    evidence = repair._evidence(root, run_gametest=True)
    assert evidence["passed"] is False
    assert evidence["build"]["status"] == "SKIPPED"
    assert evidence["diagnostics"]["status"] == "UNAVAILABLE"
