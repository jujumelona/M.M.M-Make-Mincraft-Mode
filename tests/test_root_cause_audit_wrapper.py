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


def _redirect_wrapper_paths(tmp_path, monkeypatch) -> tuple:
    audit_dir = tmp_path / "audit"
    report_path = audit_dir / "FULL_PROJECT_AUDIT.json"
    runner_log = audit_dir / "FULL_PROJECT_AUDIT.runner.log"
    audit_dir.mkdir()
    monkeypatch.setattr(audit_wrapper, "ROOT", tmp_path)
    monkeypatch.setattr(audit_wrapper, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit_wrapper, "REPORT_PATH", report_path)
    monkeypatch.setattr(audit_wrapper, "RUNNER_LOG_PATH", runner_log)
    return audit_dir, report_path, runner_log


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


def test_report_failure_detail_is_redacted_before_console_render(monkeypatch) -> None:
    exact_secret = "poisoned-report-secret-value"
    monkeypatch.setenv("MMM_TEST_SECRET", exact_secret)
    report = {
        "summary": {"passed": 0, "warned": 0, "failed": 1, "skipped": 0},
        "checks": [
            {
                "name": "provider",
                "category": "network",
                "status": "FAIL",
                "detail": f"token=label-secret {exact_secret}",
            }
        ],
    }
    rendered = render_report_summary(report)
    assert "label-secret" not in rendered
    assert exact_secret not in rendered
    assert "token=<redacted>" in rendered


def test_internal_exception_message_is_redacted_before_console_render(monkeypatch) -> None:
    exact_secret = "internal-exception-secret-value"
    monkeypatch.setenv("MMM_TEST_TOKEN", exact_secret)
    rendered = audit_wrapper._render_internal_report_error(
        RuntimeError(f"password=label-secret {exact_secret}"),
        "parse audit report",
    )
    assert "label-secret" not in rendered
    assert exact_secret not in rendered
    assert "password=<redacted>" in rendered
    assert "Traceback" not in rendered


def test_main_never_reuses_stale_report_or_runner_log(tmp_path, monkeypatch, capsys) -> None:
    audit_dir, report_path, runner_log = _redirect_wrapper_paths(tmp_path, monkeypatch)
    report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")
    runner_log.write_text("stale runner evidence\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        assert not report_path.exists(), "stale report must be removed before the subprocess starts"
        assert runner_log.read_text(encoding="utf-8") == ""
        assert kwargs["stdout"] is not subprocess.PIPE
        assert kwargs["timeout"] == audit_wrapper._DEFAULT_AUDIT_TIMEOUT_SECONDS
        kwargs["stdout"].write("current run crashed before writing a report\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 7
    output = capsys.readouterr().out
    assert "MissingAuditReport" in output
    assert "FINAL STATUS\nPASS" not in output
    assert runner_log.read_text(encoding="utf-8").startswith("current run crashed")
    assert not any(path.name.startswith(".audit-runner-output-") for path in audit_dir.iterdir())


def test_main_fails_closed_when_process_and_report_disagree(tmp_path, monkeypatch, capsys) -> None:
    _, report_path, _ = _redirect_wrapper_paths(tmp_path, monkeypatch)

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write("runner returned nonzero\n")
        report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 1
    output = capsys.readouterr().out
    assert "AuditExitMismatch" in output
    assert "report requires exit=0" in output


def test_main_redacts_runner_output_before_persisting_artifact(
    tmp_path, monkeypatch, capsys
) -> None:
    _, report_path, runner_log = _redirect_wrapper_paths(tmp_path, monkeypatch)
    exact_secret = "runner-exact-secret-value"
    monkeypatch.setenv("MMM_TEST_TOKEN", exact_secret)

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write(
            f"unlabelled={exact_secret}\n"
            "token=labelled secret with spaces\n"
            '{"password": "quoted secret value", "safe": 1}\n'
        )
        report_path.write_text(json.dumps(_complete_report()), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 0
    assert "FINAL STATUS\nPASS" in capsys.readouterr().out

    persisted = runner_log.read_text(encoding="utf-8")
    assert exact_secret not in persisted
    assert "labelled secret with spaces" not in persisted
    assert "quoted secret value" not in persisted
    assert "unlabelled=<redacted>" in persisted
    assert "token=<redacted>" in persisted
    assert '"password": "<redacted>"' in persisted
    assert '"safe": 1' in persisted


def test_audit_timeout_is_bounded_transient_and_preserves_redacted_output(
    tmp_path, monkeypatch, capsys
) -> None:
    audit_dir, report_path, runner_log = _redirect_wrapper_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(audit_wrapper, "_DEFAULT_AUDIT_TIMEOUT_SECONDS", 17)

    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == 17
        kwargs["stdout"].write("token=timeout-secret\nbefore timeout\n")
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fake_run)
    assert audit_wrapper.main() == 1
    output = capsys.readouterr().out
    assert "AuditTimeout" in output
    assert "[TRANSIENT]" in output
    assert "timeout=17s" in output
    assert "Traceback" not in output
    assert not report_path.exists()

    persisted = runner_log.read_text(encoding="utf-8")
    assert "timeout-secret" not in persisted
    assert "token=<redacted>" in persisted
    assert "before timeout" in persisted
    assert not any(path.name.startswith(".audit-runner-output-") for path in audit_dir.iterdir())


def test_nonpositive_audit_timeout_fails_before_subprocess(tmp_path, monkeypatch, capsys) -> None:
    _redirect_wrapper_paths(tmp_path, monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess must not be called for an invalid timeout")

    monkeypatch.setattr(audit_wrapper.subprocess, "run", fail_if_called)
    assert audit_wrapper._run_audit(timeout_seconds=0) is None
    output = capsys.readouterr().out
    assert "InvalidAuditTimeout" in output
    assert "[INPUT]" in output
