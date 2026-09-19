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
