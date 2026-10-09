from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.runner import CommandResult, GradleRunner, _PreparedBuild


class _RecordingRunner(GradleRunner):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @staticmethod
    def _wrapper_is_current(project_root, gradle_version, gradle_sha256):
        del project_root, gradle_version, gradle_sha256
        return True

    def _run(self, *, name, executable, arguments, cwd, env, log_path):
        del executable, cwd, env
        self.calls.append((name, tuple(arguments)))
        return CommandResult(
            name=name,
            command=tuple(arguments),
            exit_code=0,
            duration_seconds=0.01,
            log_path=str(log_path),
            timed_out=False,
        )

    @staticmethod
    def _find_release_jar(project_root):
        return str(project_root / "build/libs/test.jar")

    @staticmethod
    def _gametest_report(project_root):
        return str(project_root / "build/gametest-report.xml")


def _prepared(root: Path) -> _PreparedBuild:
    logs = root / ".minecraft_ai/logs"
    logs.mkdir(parents=True)
    return _PreparedBuild(
        project_root=root,
        gradle_version="9.5.1",
        gradle_sha256="a" * 64,
        gradle=Path("/gradle"),
        logs=logs,
        environment={},
    )


def test_official_scaffold_runs_modern_gametest_once(tmp_path: Path) -> None:
    project = tmp_path / "modern"
    project.mkdir()
    (project / "build.gradle").write_text(
        """
fabricApi {
    configureTests {
        createSourceSet = false
        enableGameTests = true
    }
}
loom {
    runs {
        gameTest {
            vmArg "-Dfabric-api.gametest.report-file=x"
        }
    }
}
""",
        encoding="utf-8",
    )
    runner = _RecordingRunner(tmp_path / "cache")

    report = runner._execute_prepared_build(_prepared(project), run_gametest=True)

    assert report.status == "PASS"
    assert report.gametest_mode == "explicit_task"
    assert report.gametest_task == "runGameTest"
    assert [name for name, _args in runner.calls] == ["build", "gametest"]
    build_args = runner.calls[0][1]
    gametest_args = runner.calls[1][1]
    assert build_args.count("runGameTest") == 1
    assert ("-x", "runGameTest") == (
        build_args[build_args.index("-x")],
        build_args[build_args.index("-x") + 1],
    )
    assert gametest_args == ("--no-daemon", "runGameTest", "--stacktrace")


def test_legacy_scaffold_keeps_run_gametest_server_task(tmp_path: Path) -> None:
    project = tmp_path / "legacy"
    project.mkdir()
    (project / "build.gradle").write_text(
        """
loom {
    runs {
        gameTestServer {
            server()
        }
    }
}
""",
        encoding="utf-8",
    )
    runner = _RecordingRunner(tmp_path / "cache")

    report = runner._execute_prepared_build(_prepared(project), run_gametest=True)

    assert report.status == "PASS"
    assert runner.calls[0][0] == "build"
    assert "-x" not in runner.calls[0][1]
    assert runner.calls[1][1] == (
        "--no-daemon",
        "runGameTestServer",
        "--stacktrace",
    )

def test_legacy_integrated_gametest_is_not_launched_twice(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "legacy-integrated"
    project.mkdir()
    (project / "build.gradle").write_text(
        """
loom {
    runs {
        gameTestServer {
            server()
        }
    }
}
""",
        encoding="utf-8",
    )
    runner = _RecordingRunner(tmp_path / "cache")
    monkeypatch.setattr(
        runner,
        "_executed_gametest_task",
        lambda _command: "runGameTestServer",
    )

    report = runner._execute_prepared_build(_prepared(project), run_gametest=True)

    assert report.status == "PASS"
    assert report.gametest_mode == "integrated_build"
    assert report.gametest_task == "runGameTestServer"
    assert [name for name, _args in runner.calls] == ["build"]



def test_gametest_timeout_is_non_repairable_verifier_failure(tmp_path: Path) -> None:
    project = tmp_path / "timeout"
    project.mkdir()
    (project / "build.gradle").write_text(
        "fabricApi { configureTests { enableGameTests = true } }\n"
        "loom { runs { gameTest { vmArg '-Dfabric-api.gametest.report-file=x' } } }\n",
        encoding="utf-8",
    )

    class TimeoutRunner(_RecordingRunner):
        def _run(self, *, name, executable, arguments, cwd, env, log_path):
            del executable, cwd, env
            self.calls.append((name, tuple(arguments)))
            return CommandResult(
                name=name, command=tuple(arguments),
                exit_code=124 if name == "gametest" else 0,
                duration_seconds=1205.0, log_path=str(log_path),
                timed_out=name == "gametest",
            )

    runner = TimeoutRunner(tmp_path / "cache")
    result = runner._execute_prepared_build(_prepared(project), run_gametest=True)
    assert result.status == "FAIL"
    assert result.error_code == "GRADLE_GAMETEST_TIMEOUT"
    assert result.failure_class == "infrastructure_timeout"
    assert result.repairable is False
    assert [name for name, _ in runner.calls] == ["build", "gametest"]



def _host_owned_eula_project(tmp_path: Path, *, eula_setting: str) -> Path:
    project = tmp_path / "host-eula"
    project.mkdir()
    (project / "build.gradle").write_text(
        "// M.M.M host-owned server GameTest contract\n"
        "fabricApi { configureTests { enableGameTests = true\n"
        f" {eula_setting}\n"
        "} }\n",
        encoding="utf-8",
    )
    return project


def test_host_owned_gametest_without_eula_opt_in_fails_before_gradle(tmp_path: Path) -> None:
    project = _host_owned_eula_project(
        tmp_path,
        eula_setting='eula = (System.getenv("MMM_ACCEPT_MINECRAFT_EULA") ?: "false").equalsIgnoreCase("true")',
    )
    runner = _RecordingRunner(tmp_path / "cache")
    result = runner._execute_prepared_build(_prepared(project), run_gametest=True)
    assert result.status == "FAIL"
    assert result.error_code == "GRADLE_GAMETEST_EULA_REQUIRED"
    assert result.repairable is False
    assert "Minecraft EULA" in result.error
    assert runner.calls == []


def test_host_owned_gametest_runs_after_explicit_env_opt_in(tmp_path: Path) -> None:
    project = _host_owned_eula_project(
        tmp_path,
        eula_setting='eula = (System.getenv("MMM_ACCEPT_MINECRAFT_EULA") ?: "false").equalsIgnoreCase("true")',
    )
    runner = _RecordingRunner(tmp_path / "cache")
    prepared = _prepared(project)
    prepared.environment["MMM_ACCEPT_MINECRAFT_EULA"] = "true"
    result = runner._execute_prepared_build(prepared, run_gametest=True)
    assert result.status == "PASS"
    assert [name for name, _ in runner.calls] == ["build", "gametest"]


def test_host_owned_legacy_eula_config_needs_migration(tmp_path: Path) -> None:
    project = _host_owned_eula_project(
        tmp_path,
        eula_setting="enableClientGameTests = false",
    )
    runner = _RecordingRunner(tmp_path / "cache")
    prepared = _prepared(project)
    prepared.environment["MMM_ACCEPT_MINECRAFT_EULA"] = "true"
    result = runner._execute_prepared_build(prepared, run_gametest=True)
    assert result.status == "FAIL"
    assert result.error_code == "GRADLE_GAMETEST_EULA_REQUIRED"
    assert "predates" in result.error
    assert runner.calls == []


def test_host_owned_literal_eula_acceptance_is_respected(tmp_path: Path) -> None:
    project = _host_owned_eula_project(tmp_path, eula_setting="eula = true")
    runner = _RecordingRunner(tmp_path / "cache")
    result = runner._execute_prepared_build(_prepared(project), run_gametest=True)
    assert result.status == "PASS"
    assert [name for name, _ in runner.calls] == ["build", "gametest"]
