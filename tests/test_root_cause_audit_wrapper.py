from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.diagnostics import FailureCategory
from tools import root_cause_audit_wrapper as audit_wrapper
from tools.root_cause_audit_wrapper import (
    failure_groups_from_report,
    normalize_report_semantics,
    render_report_summary,
    validate_report_consistency,
)


def _complete_report(*, status: str = "PASS", detail: str = "ok") -> dict:
    checks = [{"name": "probe", "category": "test", "status": status, "detail": detail}]
    return {
        "schema_version": "mmm/full-project-audit-v3",
        "overall_status": "failed" if status == "FAIL" else "warning" if status == "WARN" else "passed",
        "summary": {
            "total": 1,
            "passed": int(status == "PASS"),
            "warned": int(status == "WARN"),
            "failed": int(status == "FAIL"),
            "skipped": int(status == "SKIP"),
        },
        "checks": checks,
        "failed_checks": ["probe"] if status == "FAIL" else [],
        "warning_checks": ["probe"] if status == "WARN" else [],
        "skipped_checks": ["probe"] if status == "SKIP" else [],
    }


def test_duplicate_failed_check_is_collapsed_to_one_root_with_attempt_count() -> None:
    report = {
        "summary": {"passed": 1, "warned": 0, "failed": 2, "skipped": 0},
        "checks": [
            {"name": "provider", "category": "network", "status": "FAIL", "detail": "timeout"},
            {"name": "provider", "category": "network", "status": "FAIL", "detail": "timeout"},
        ],
    }
    groups = failure_groups_from_report(report)
    assert len(groups) == 1
    assert groups[0].attempts == 2
    rendered = render_report_summary(report)
    assert rendered.count("ROOT FAILURE") == 1
    assert "ATTEMPTS\n2" in rendered
    assert "CHECKS pass=1 warn=0 fail=2 skip=0" in rendered


def test_internal_audit_check_is_not_mislabeled_validation() -> None:
    report = {
        "summary": {"passed": 0, "warned": 0, "failed": 1, "skipped": 0},
        "checks": [
            {
                "name": "probe",
                "category": "audit-internal",
                "status": "FAIL",
                "detail": "TypeError: broken audit code",
            }
        ],
    }
    groups = failure_groups_from_report(report)
    assert len(groups) == 1
    assert groups[0].event.category is FailureCategory.INTERNAL
    assert groups[0].event.cause_type == "AuditInternalFailure"


def test_warning_is_not_mislabeled_as_pass() -> None:
    report = {
        "summary": {"passed": 3, "warned": 1, "failed": 0, "skipped": 2},
        "checks": [],
    }
    assert render_report_summary(report).startswith("FINAL STATUS\nWARN\n")


def test_clean_report_is_explicit_pass() -> None:
    report = {
        "summary": {"passed": 4, "warned": 0, "failed": 0, "skipped": 1},
        "checks": [],
    }
    assert render_report_summary(report).startswith("FINAL STATUS\nPASS\n")


def test_normalized_artifact_only_marks_pass_as_passed() -> None:
    report = {
        "checks": [
            {"status": "PASS", "passed": True},
            {"status": "WARN", "passed": True},
            {"status": "SKIP", "passed": True},
            {"status": "FAIL", "passed": False},
        ]
    }
    normalized = normalize_report_semantics(report)
    assert [check["passed"] for check in normalized["checks"]] == [
        True,
        False,
        False,
        False,
    ]
    assert [check["non_blocking"] for check in normalized["checks"]] == [
        False,
        True,
        True,
        False,
    ]
    assert [check["blocking_failure"] for check in normalized["checks"]] == [
        False,
        False,
        False,
        True,
    ]


def test_unknown_audit_status_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported status"):
        normalize_report_semantics({"checks": [{"status": "MAYBE"}]})


def test_report_consistency_rejects_summary_mismatch() -> None:
    report = normalize_report_semantics(_complete_report())
    report["summary"]["passed"] = 0
    with pytest.raises(ValueError, match="summary/check mismatch"):
        validate_report_consistency(report)


def test_report_consistency_rejects_status_index_mismatch() -> None:
    report = normalize_report_semantics(_complete_report(status="FAIL", detail="broken"))
    report["failed_checks"] = []
    with pytest.raises(ValueError, match="failed_checks"):
        validate_report_consistency(report)


def test_main_never_reuses_stale_report_or_runner_log(tmp_path, monkeypatch, capsys) -> None:
    audit_dir = tmp_path / "audit"
    report_path = audit_dir / "FULL_PROJECT_AUDIT.json"
    runner_log = audit_dir / "FULL_PROJECT_AUDIT.runner.log"
    audit_dir.mkdir()
    report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")
    runner_log.write_text("stale runner evidence\n", encoding="utf-8")

    monkeypatch.setattr(audit_wrapper, "ROOT", tmp_path)
    monkeypatch.setattr(audit_wrapper, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit_wrapper, "REPORT_PATH", report_path)
    monkeypatch.setattr(audit_wrapper, "RUNNER_LOG_PATH", runner_log)

    def fake_run(*args, **kwargs):
        assert not report_path.exists(), "stale report must be removed before the subprocess starts"
        assert runner_log.read_text(encoding="utf-8") == ""
        assert kwargs["stdout"] is not subprocess.PIPE
        kwargs["stdout"].write("current run crashed before writing a report\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 7
    output = capsys.readouterr().out
    assert "MissingAuditReport" in output
    assert "FINAL STATUS\nPASS" not in output
    assert runner_log.read_text(encoding="utf-8").startswith("current run crashed")


def test_main_fails_closed_when_process_and_report_disagree(tmp_path, monkeypatch, capsys) -> None:
    audit_dir = tmp_path / "audit"
    report_path = audit_dir / "FULL_PROJECT_AUDIT.json"
    runner_log = audit_dir / "FULL_PROJECT_AUDIT.runner.log"
    audit_dir.mkdir()

    monkeypatch.setattr(audit_wrapper, "ROOT", tmp_path)
    monkeypatch.setattr(audit_wrapper, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit_wrapper, "REPORT_PATH", report_path)
    monkeypatch.setattr(audit_wrapper, "RUNNER_LOG_PATH", runner_log)

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write("runner returned nonzero\n")
        report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 1
    output = capsys.readouterr().out
    assert "AuditExitMismatch" in output
    assert "report requires exit=0" in output
