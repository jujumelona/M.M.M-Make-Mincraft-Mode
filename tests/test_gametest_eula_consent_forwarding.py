"""Regression coverage for Colab -> orchestrator -> Gradle GameTest consent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.runner import GradleRunner, _PreparedBuild


def _host_build_script(tmp_path: Path) -> Path:
    root = tmp_path / "mod"
    root.mkdir()
    (root / "build.gradle").write_text(
        "// M.M.M host-owned server GameTest contract\n"
        "eula = (System.getenv(\"MMM_ACCEPT_MINECRAFT_EULA\") ?: \"false\").equalsIgnoreCase(\"true\")\n",
        encoding="utf-8",
    )
    return root


def _prepared(root: Path, env: dict[str, str]) -> _PreparedBuild:
    return _PreparedBuild(
        project_root=root,
        gradle_version="8.6",
        gradle_sha256="0" * 64,
        gradle=root / "fake-gradle",
        logs=root,
        environment=env,
    )


def test_approved_execution_consent_reaches_actual_gradle_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("MMM_ACCEPT_MINECRAFT_EULA", raising=False)
    root = _host_build_script(tmp_path)
    runner = GradleRunner(tmp_path / "cache", eula_accepted=True)
    environment = runner._gametest_environment({"CI": "true"})
    assert environment["MMM_ACCEPT_MINECRAFT_EULA"] == "true"
    assert GradleRunner._host_gametest_eula_preflight_error(
        _prepared(root, environment), "runGameTest"
    ) is None
    assert "MMM_ACCEPT_MINECRAFT_EULA" not in __import__("os").environ


def test_unapproved_execution_never_invents_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("MMM_ACCEPT_MINECRAFT_EULA", raising=False)
    root = _host_build_script(tmp_path)
    runner = GradleRunner(tmp_path / "cache")
    environment = runner._gametest_environment({"CI": "true"})
    assert "MMM_ACCEPT_MINECRAFT_EULA" not in environment
    assert "requires Minecraft EULA acceptance" in (
        GradleRunner._host_gametest_eula_preflight_error(
            _prepared(root, environment), "runGameTest"
        ) or ""
    )


def test_existing_explicit_environment_consent_is_preserved(tmp_path):
    runner = GradleRunner(tmp_path / "cache")
    inherited = {"MMM_ACCEPT_MINECRAFT_EULA": "true", "CI": "true"}
    assert runner._gametest_environment(inherited) == inherited


def test_stale_colab_consent_fails_before_proposal_parsing(tmp_path, monkeypatch):
    from minecraft_mod_ai.complete_orchestrator import (
        CompleteExecutionOptions, CompleteProductionError, CompleteProductionOrchestrator,
    )

    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "/content/setup-receipt.json")
    monkeypatch.delenv("MMM_ACCEPT_MINECRAFT_EULA", raising=False)
    runner = CompleteProductionOrchestrator(workspace_root=tmp_path / "output")
    with pytest.raises(CompleteProductionError, match="GRADLE_GAMETEST_EULA_REQUIRED \\(before generation\\)"):
        runner.execute({}, approval_hash="", run_name="consent-regression", options=CompleteExecutionOptions())


def test_committed_colab_notebook_has_consistent_eula_settings():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8")
    )
    settings = "".join(notebook["cells"][1]["source"])
    build = "".join(notebook["cells"][6]["source"])
    assert "ACCEPT_EULA = True" in settings
    assert 'os.environ["MMM_ACCEPT_MINECRAFT_EULA"] = "true" if ACCEPT_EULA else "false"' in settings
    assert "GAMETEST_EULA_PREFLIGHT" in build
    assert "eula_accepted=ACCEPT_EULA" in build
    assert "options = _replace_dataclass(options, eula_accepted=True)" in build
