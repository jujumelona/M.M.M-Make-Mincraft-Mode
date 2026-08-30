from __future__ import annotations

import sys
from pathlib import Path

from tools import full_project_audit as audit


def _redirect_artifacts(tmp_path: Path, monkeypatch) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "LOG_PATH", audit_dir / "FULL_PROJECT_AUDIT.log")
    monkeypatch.setattr(audit, "REPORT_PATH", audit_dir / "FULL_PROJECT_AUDIT.json")
    audit.LOG_PATH.write_text("", encoding="utf-8")


def test_check_payload_has_one_unambiguous_status_semantics() -> None:
    expected = {
        audit.PASS: (True, False, False),
        audit.WARN: (False, False, True),
        audit.SKIP: (False, False, True),
        audit.FAIL: (False, True, False),
    }
    for status, semantics in expected.items():
        payload = audit.Check("probe", status, "detail").payload()
        assert (
            payload["passed"],
            payload["blocking_failure"],
            payload["non_blocking"],
        ) == semantics


def test_logged_process_streams_full_output_but_keeps_bounded_tail(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    line_count = 5000
    result = audit._run_logged_process(
        [
            sys.executable,
            "-c",
            f"for i in range({line_count}): print('row-%05d-' % i + 'x' * 200)",
        ],
        timeout=30,
        cwd=tmp_path,
        label="large-output",
    )
    assert result.returncode == 0
    assert result.error is None
    assert result.timed_out is False
    assert len(result.output_tail) <= audit._OUTPUT_TAIL_LIMIT
    assert "row-04999" in result.output_tail

    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "row-00000" in log
    assert "row-04999" in log
    assert log.count("row-") == line_count


def test_streamed_process_log_redacts_environment_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    monkeypatch.setenv("MMM_TEST_API_KEY", "do-not-leak-this-value")
    result = audit._run_logged_process(
        [sys.executable, "-c", "print('do-not-leak-this-value')"],
        timeout=30,
        cwd=tmp_path,
        label="secret-output",
    )
    assert result.returncode == 0
    assert "do-not-leak-this-value" not in result.output_tail
    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "do-not-leak-this-value" not in log
    assert "<redacted>" in log


def test_timeout_preserves_output_tail_without_memory_capture(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    result = audit._run_logged_process(
        [
            sys.executable,
            "-c",
            "import time; print('before-timeout', flush=True); time.sleep(2)",
        ],
        timeout=0.05,
        cwd=tmp_path,
        label="timeout-output",
    )
    assert result.timed_out is True
    assert result.returncode is None
    assert "before-timeout" in result.output_tail
    assert "process_timeout=" in audit.LOG_PATH.read_text(encoding="utf-8")
