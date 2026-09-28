from __future__ import annotations

import subprocess

from tools import colab_runtime_setup as setup


def test_project_install_retries_after_transient_pip_failure(monkeypatch, tmp_path):
    receipt = tmp_path / "project-install-receipt.json"
    monkeypatch.setattr(setup, "_project_install_receipt_path", lambda: receipt)
    monkeypatch.setattr(setup, "_project_install_fingerprint", lambda _target: "fingerprint")
    calls: list[list[str]] = []

    def run_logged(command, *, cwd=None):
        del cwd
        calls.append(list(command))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "_run_logged", run_logged)

    setup._install_project(local_profile=True)

    assert len(calls) == 2
    first, second = calls
    assert first[first.index("--timeout") + 1] == "60"
    assert first[first.index("--retries") + 1] == "8"
    assert second[second.index("--timeout") + 1] == "180"
    assert second[second.index("--retries") + 1] == "12"
    assert "--disable-pip-version-check" in first
    assert "--disable-pip-version-check" in second
    assert receipt.is_file()


def test_project_install_does_not_write_receipt_after_all_attempts_fail(
    monkeypatch, tmp_path
):
    receipt = tmp_path / "project-install-receipt.json"
    monkeypatch.setattr(setup, "_project_install_receipt_path", lambda: receipt)
    monkeypatch.setattr(setup, "_project_install_fingerprint", lambda _target: "fingerprint")

    def run_logged(command, *, cwd=None):
        del cwd
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "_run_logged", run_logged)

    try:
        setup._install_project(local_profile=True)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("pip failure must propagate after bounded retries")

    assert not receipt.exists()
