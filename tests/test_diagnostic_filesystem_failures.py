from __future__ import annotations

from pathlib import Path

from tools import pytest_diagnostics
from tools import root_cause_audit_wrapper as audit_wrapper


def test_audit_directory_creation_failure_is_structured(monkeypatch, capsys) -> None:
    class FailingAuditDirectory:
        def mkdir(self, *args, **kwargs) -> None:
            raise PermissionError("audit directory denied")

    monkeypatch.setattr(audit_wrapper, "AUDIT_DIR", FailingAuditDirectory())
    assert audit_wrapper._prepare_audit_directory() is False
    output = capsys.readouterr().out
    assert "ROOT FAILURE" in output
    assert "prepare audit directory" in output
    assert "PermissionError" in output
    assert "Traceback" not in output


def test_pytest_output_directory_failure_is_structured(monkeypatch, capsys) -> None:
    def fail_mkdir(self: Path, *args, **kwargs) -> None:
        raise PermissionError("pytest output denied")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    assert (
        pytest_diagnostics._prepare_output_directories(
            Path("audit/pytest.log"), Path("audit/pytest.xml")
        )
        is False
    )
    output = capsys.readouterr().out
    assert "ROOT FAILURE" in output
    assert "prepare diagnostic output directories" in output
    assert "PermissionError" in output
    assert "Traceback" not in output
