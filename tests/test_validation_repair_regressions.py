from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_mod_ai.repair_engine as repair_module
from minecraft_mod_ai.repair_engine import RepairEngine
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.validation_checkpoint_policy import (
    cached_validation_is_reusable,
    validation_checkpoint_input,
    validation_implementation_fingerprint,
)


def _failed_evidence(log_path: Path) -> dict[str, object]:
    return {
        "passed": False,
        "diagnostics": {
            "schema_version": "mmm/java-diagnostics-v2",
            "status": "PASS",
            "diagnostics": {},
            "diagnostics_by_uri": {},
            "pages": [],
            "files_opened": 0,
            "page_count": 0,
            "error_count": 0,
            "warning_count": 0,
        },
        "build": {
            "status": "FAIL",
            "error": "Gradle build failed.",
            "commands": [
                {
                    "name": "build",
                    "exit_code": 1,
                    "timed_out": False,
                    "log_path": str(log_path),
                }
            ],
        },
    }


def test_final_validation_checkpoint_aliases_share_validator_family() -> None:
    assert validation_implementation_fingerprint("validate-source-final") == validation_implementation_fingerprint("validate-source")
    assert validation_implementation_fingerprint("validate-jdt-final") == validation_implementation_fingerprint("validate-jdt")
    scoped = validation_checkpoint_input("validate-source-final", {"project_manifest": "abc"})
    assert scoped["project_manifest"] == "abc"
    assert scoped["_mmm_validation_implementation"].startswith("sha256:")
    source_receipt = {"status": "PASS", "checks_run": 1, "findings": []}
    assert cached_validation_is_reusable("validate-source-final", source_receipt)
    jdt_receipt = {
        "schema_version": "mmm/java-diagnostics-v2",
        "diagnostics": {},
        "diagnostics_by_uri": {},
        "pages": [],
        "files_opened": 0,
        "page_count": 0,
        "error_count": 0,
        "warning_count": 0,
    }
    assert cached_validation_is_reusable("validate-jdt-final", jdt_receipt)


def test_unknown_validation_checkpoint_remains_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported validation checkpoint"):
        validation_implementation_fingerprint("validate-unknown")
    assert not cached_validation_is_reusable("validate-unknown", {"status": "PASS", "checks_run": 1, "findings": []})


def test_repair_context_reads_bounded_failed_gradle_log(tmp_path: Path) -> None:
    log_path = tmp_path / "gradle-build.log"
    marker = "error: cannot find symbol ExampleRegistry"
    log_path.write_text("x" * 5000 + "\n" + marker + "\n", encoding="utf-8")

    engine = object.__new__(RepairEngine)
    engine.policy = ScalePolicy.from_environment()
    engine.router = None
    context = engine._context(tmp_path, _failed_evidence(log_path))
    assert context["build_logs"]
    output = context["build_logs"][0]["output"]
    assert marker in output
    assert len(output) <= repair_module._REPAIR_LOG_SNIPPET_CHARS


def test_repair_signature_tracks_actual_failed_gradle_log(tmp_path: Path) -> None:
    log_path = tmp_path / "gradle-build.log"
    log_path.write_text("first concrete compiler failure", encoding="utf-8")
    evidence = _failed_evidence(log_path)
    first = RepairEngine._signature(evidence)
    assert first == RepairEngine._signature(evidence)
    log_path.write_text("second different resource failure", encoding="utf-8")
    assert RepairEngine._signature(evidence) != first


def test_repair_log_reader_fails_safe_for_missing_file(tmp_path: Path) -> None:
    evidence = _failed_evidence(tmp_path / "missing.log")
    assert RepairEngine._signature(evidence)
    assert repair_module._failed_build_log_diagnostics(evidence) == [
        {"name": "build", "exit_code": 1, "timed_out": False}
    ]
