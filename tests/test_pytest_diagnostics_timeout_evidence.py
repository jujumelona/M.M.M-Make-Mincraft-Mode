from __future__ import annotations

import subprocess
import sys

from tools import pytest_diagnostics


def test_capture_pytest_timeout_preserves_process_evidence_and_redacts_secrets(
    tmp_path, monkeypatch
) -> None:
    secret = "pytest-diagnostics-secret-sentinel"
    monkeypatch.setenv("DIAG_TOKEN", secret)
    log_path = tmp_path / "timeout.log"

    command = [
        sys.executable,
        "-c",
        (
            "import os, subprocess, sys, time; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "print('child_pid=' + str(child.pid), flush=True); "
            "print(os.environ['DIAG_TOKEN'], flush=True); "
            "time.sleep(30)"
        ),
    ]

    returncode, failure = pytest_diagnostics._capture_pytest(
        command,
        log_path,
        timeout_seconds=1,
    )

    assert returncode is None
    assert isinstance(failure, subprocess.TimeoutExpired)
    text = log_path.read_text(encoding="utf-8")
    assert "PYTEST TIMEOUT PROCESS SNAPSHOT" in text
    assert "pytest_pid=" in text
    assert "child_pid=" in text
    assert secret not in text
    assert "<redacted>" in text


def test_parse_args_rejects_nonpositive_faulthandler_timeout(tmp_path, capsys) -> None:
    exit_code = pytest_diagnostics.main(
        [
            "--log",
            str(tmp_path / "pytest.log"),
            "--junit",
            str(tmp_path / "pytest.xml"),
            "--faulthandler-timeout-seconds",
            "0",
            "tests/test_pytest_diagnostics_timeout_evidence.py",
        ]
    )

    assert exit_code == 2
    assert "InvalidFaulthandlerTimeout" in capsys.readouterr().out
