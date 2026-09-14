from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.runner_target_java_contract import install


class _WorkspaceError(RuntimeError):
    pass


class _Report:
    def __init__(
        self,
        *,
        status: str,
        gradle_version: str,
        commands=(),
        jar_path=None,
        gametest_report=None,
        error=None,
    ) -> None:
        self.status = status
        self.gradle_version = gradle_version
        self.commands = commands
        self.jar_path = jar_path
        self.gametest_report = gametest_report
        self.error = error


class _Adapter:
    java_version = "25"
    gradle = "9.1.0"
    minecraft_version = "26.2"


def _module(*, java_home: Path | None, captured: list[dict[str, str]]):
    class Runner:
        def _run(self, *, name, executable, arguments, cwd, env, log_path):
            captured.append(dict(env))
            return "ran"

        def _build_locked(self, project_root: Path, *, run_gametest: bool):
            assert run_gametest is False
            return self._run(
                name="build",
                executable=project_root / "gradle",
                arguments=("build",),
                cwd=project_root,
                env={"JAVA_HOME": "/host/jdk17", "PATH": "/host/bin"},
                log_path=project_root / "build.log",
            )

    def resolve(required_java: int):
        assert required_java == 25
        if java_home is None:
            raise _WorkspaceError("JDK 25 missing")
        return java_home

    return SimpleNamespace(
        GradleRunner=Runner,
        adapter_from_project=lambda _root: _Adapter(),
        _resolve_project_java_home=resolve,
        JDTWorkspaceBootstrapError=_WorkspaceError,
        BuildRunnerError=RuntimeError,
        BuildReport=_Report,
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "build.gradle").write_text("\n", encoding="utf-8")
    return project


def test_target_java_is_injected_without_mutating_host_environment(tmp_path: Path) -> None:
    captured: list[dict[str, str]] = []
    java_home = tmp_path / "jdk-25"
    java_home.mkdir()
    module = _module(java_home=java_home, captured=captured)
    install(module)

    result = module.GradleRunner()._build_locked(
        _project(tmp_path),
        run_gametest=False,
    )

    assert result == "ran"
    assert captured[0]["JAVA_HOME"] == str(java_home.resolve())
    assert captured[0]["PATH"].split(__import__("os").pathsep, 1)[0] == str(
        java_home.resolve() / "bin"
    )


def test_missing_target_java_returns_unavailable_before_gradle(tmp_path: Path) -> None:
    captured: list[dict[str, str]] = []
    module = _module(java_home=None, captured=captured)
    install(module)

    result = module.GradleRunner()._build_locked(
        _project(tmp_path),
        run_gametest=False,
    )

    assert result.status == "UNAVAILABLE"
    assert result.gradle_version == "9.1.0"
    assert "Java 25 toolchain unavailable for target 26.2" in result.error
    assert captured == []
