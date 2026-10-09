from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools import colab_runtime_setup as setup


def _isolated_install(monkeypatch, tmp_path: Path) -> Path:
    receipt = tmp_path / "project-install-receipt.json"
    monkeypatch.setattr(setup, "_project_install_receipt_path", lambda: receipt)
    monkeypatch.setattr(setup, "_project_install_fingerprint", lambda _target: "fingerprint")
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/timeout" if name == "timeout" else None)
    return receipt


def test_project_install_deadline_is_a_whole_process_timeout(monkeypatch, tmp_path):
    receipt = _isolated_install(monkeypatch, tmp_path)
    commands = []

    def fail_with_deadline(command, *, cwd=None):
        del cwd
        commands.append(list(command))
        raise subprocess.CalledProcessError(124, command)

    monkeypatch.setattr(setup, "_run_logged", fail_with_deadline)
    with pytest.raises(RuntimeError, match="COLAB_PIP_INSTALL_DEADLINE_EXCEEDED"):
        setup._install_project(local_profile=True)

    assert len(commands) == 1
    command = commands[0]
    assert command[:4] == [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s",
        f"{setup._PROJECT_PIP_INSTALL_DEADLINE_SECONDS}s",
    ]
    assert command[4:7] == [setup.sys.executable, "-m", "pip"]
    assert command[command.index("--timeout") + 1] == "25"
    assert command[command.index("--retries") + 1] == "2"
    assert "--constraint" in command
    assert command[command.index("--constraint") + 1].endswith("colab_pip_constraints.txt")
    assert "-e" in command
    assert setup.LOCAL_PROJECT_INSTALL_TARGET == command[-1]
    assert not receipt.exists()


def test_project_install_resolver_conflict_fails_without_full_install_retry(
    monkeypatch, tmp_path
):
    receipt = _isolated_install(monkeypatch, tmp_path)
    calls = []

    def fail_with_conflict(command, *, cwd=None):
        del cwd
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "_run_logged", fail_with_conflict)
    with pytest.raises(RuntimeError, match="COLAB_PIP_INSTALL_FAILED"):
        setup._install_project(local_profile=True)
    assert len(calls) == 1
    assert not receipt.exists()


def test_project_install_fails_closed_if_deadline_unavailable(monkeypatch, tmp_path):
    receipt = _isolated_install(monkeypatch, tmp_path)
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        setup, "_run_logged",
        lambda command, **kwargs: pytest.fail("unbounded pip must never start"),
    )
    with pytest.raises(RuntimeError, match="COLAB_PIP_DEADLINE_UNAVAILABLE"):
        setup._install_project(local_profile=True)
    assert not receipt.exists()


def test_colab_image_stack_pins_backtracked_package_families():
    constraints = (
        Path(setup.__file__).with_name("colab_pip_constraints.txt")
        .read_text(encoding="utf-8")
    )
    values = {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for line in constraints.splitlines()
        if line and not line.startswith("#") and "==" in line
    }
    for package in (
        "numpy", "numba", "llvmlite", "rembg", "accelerate",
        "bitsandbytes", "diffusers", "gradio", "gradio-client",
        "huggingface-hub", "peft", "sentence-transformers", "transformers",
    ):
        assert package in values
    assert values["rembg"] == "2.0.77"
    assert setup._PROJECT_PIP_INSTALL_DEADLINE_SECONDS == 240
