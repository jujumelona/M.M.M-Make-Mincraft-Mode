from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.runner_target_java_contract import target_java_environment


class _WorkspaceError(RuntimeError):
    pass


class _Adapter:
    java_version = "25"
    gradle = "9.1.0"
    minecraft_version = "26.2"


def _module(*, java_home: Path | None):
    def resolve(required_java: int):
        assert required_java == 25
        if java_home is None:
            raise _WorkspaceError("JDK 25 missing")
        return java_home

    return SimpleNamespace(
        _resolve_project_java_home=resolve,
        JDTWorkspaceBootstrapError=_WorkspaceError,
        BuildRunnerError=RuntimeError,
    )


def test_target_java_environment_is_project_scoped(tmp_path: Path) -> None:
    java_home = tmp_path / "jdk-25"
    java_home.mkdir()
    module = _module(java_home=java_home)
    host_environment = {
        "JAVA_HOME": "/host/jdk17",
        "PATH": "/host/bin",
        "MMM_SENTINEL": "keep",
    }

    prepared, error = target_java_environment(
        runner_module=module,
        adapter=_Adapter(),
        environment=host_environment,
    )

    assert error is None
    assert prepared["JAVA_HOME"] == str(java_home.resolve())
    assert prepared["PATH"].split(os.pathsep, 1)[0] == str(java_home.resolve() / "bin")
    assert prepared["MMM_SENTINEL"] == "keep"
    assert host_environment == {
        "JAVA_HOME": "/host/jdk17",
        "PATH": "/host/bin",
        "MMM_SENTINEL": "keep",
    }


def test_missing_target_java_reports_unavailable_without_environment_mutation(tmp_path: Path) -> None:
    module = _module(java_home=None)
    host_environment = {"JAVA_HOME": "/host/jdk17", "PATH": "/host/bin"}

    prepared, error = target_java_environment(
        runner_module=module,
        adapter=_Adapter(),
        environment=host_environment,
    )

    assert prepared == host_environment
    assert prepared is not host_environment
    assert error is not None
    assert "Java 25 toolchain unavailable for target 26.2" in error
    assert host_environment == {"JAVA_HOME": "/host/jdk17", "PATH": "/host/bin"}


def test_invalid_target_java_is_a_contract_error(tmp_path: Path) -> None:
    module = _module(java_home=tmp_path / "jdk-25")
    adapter = SimpleNamespace(java_version="invalid", minecraft_version="26.2")

    with pytest.raises(RuntimeError, match="invalid Java version"):
        target_java_environment(
            runner_module=module,
            adapter=adapter,
            environment={"PATH": "/host/bin"},
        )
