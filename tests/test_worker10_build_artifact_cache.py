from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

from minecraft_mod_ai import runner as runner_module
from minecraft_mod_ai import runner_parallel_validation_contract as parallel
from minecraft_mod_ai import validation_execution_contract as validation
from minecraft_mod_ai.runner import BuildReport, CommandResult, GradleRunner
from minecraft_mod_ai.runner_parallel_validation_contract import (
    _passing_gametest_xml,
    _safe_regular_file,
)


def _project(root: Path) -> Path:
    (root / "src/main/java/demo").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (root / "settings.gradle").write_text("rootProject.name='demo'\n", encoding="utf-8")
    (root / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx1g\n", encoding="utf-8")
    (root / "src/main/java/demo/Main.java").write_text(
        "package demo; public final class Main {}\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        '{"schemaVersion":1,"id":"demo_mod","version":"1.0.0"}\n',
        encoding="utf-8",
    )
    return root


def _fake_runner(tmp_path: Path, root: Path) -> tuple[GradleRunner, dict[str, int]]:
    validation._SUCCESSFUL_BUILDS.clear()
    runner = GradleRunner(tmp_path / "cache")
    calls = {"count": 0}

    def fake_build_locked(self, project_root: Path, *, run_gametest: bool):
        calls["count"] += 1
        jar = project_root / "build/libs/demo.jar"
        jar.parent.mkdir(parents=True, exist_ok=True)
        jar.write_bytes(f"jar-{calls['count']}".encode("ascii"))
        report: Path | None = None
        if run_gametest:
            report = project_root / "build/gametest-report.xml"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                f'<testsuite tests="1" failures="0" errors="0" skipped="0" name="run-{calls["count"]}"><testcase name="ok"/></testsuite>\n',
                encoding="utf-8",
            )
        return BuildReport(
            status="PASS",
            gradle_version="test",
            commands=(),
            jar_path=str(jar),
            gametest_report=str(report) if report is not None else None,
            error=None,
        )

    runner._build_locked = MethodType(fake_build_locked, runner)
    return runner, calls


def test_success_cache_reuses_only_unchanged_jar(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    runner, calls = _fake_runner(tmp_path, root)

    first = runner.build(root, run_gametest=False)
    second = runner.build(root, run_gametest=False)
    assert first.status == second.status == "PASS"
    assert calls["count"] == 1

    jar = Path(first.jar_path or "")
    jar.write_bytes(b"tampered")
    third = runner.build(root, run_gametest=False)
    assert third.status == "PASS"
    assert calls["count"] == 2

    Path(third.jar_path or "").unlink()
    fourth = runner.build(root, run_gametest=False)
    assert fourth.status == "PASS"
    assert calls["count"] == 3


def test_success_cache_binds_gametest_evidence(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    runner, calls = _fake_runner(tmp_path, root)

    first = runner.build(root, run_gametest=True)
    second = runner.build(root, run_gametest=True)
    assert first.status == second.status == "PASS"
    assert calls["count"] == 1

    report = Path(first.gametest_report or "")
    report.write_text("<tampered/>\n", encoding="utf-8")
    third = runner.build(root, run_gametest=True)
    assert third.status == "PASS"
    assert calls["count"] == 2


def test_success_cache_is_scoped_to_runtime_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _project(tmp_path / "project")
    runner, calls = _fake_runner(tmp_path, root)

    monkeypatch.setenv("JAVA_OPTS", "-Xmx1g")
    runner.build(root, run_gametest=False)
    runner.build(root, run_gametest=False)
    assert calls["count"] == 1

    monkeypatch.setenv("JAVA_OPTS", "-Xmx2g")
    runner.build(root, run_gametest=False)
    assert calls["count"] == 2


def test_gametest_resource_gate_fails_closed_without_evidence(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    missing = root / ".minecraft_ai/logs/gradle-gametest.log"
    findings = validation.gametest_resource_errors(root, missing)
    assert findings
    assert "unavailable" in findings[0].lower()

    (root / "src/main/resources/fabric.mod.json").unlink()
    log = tmp_path / "gametest.log"
    log.write_text("clean\n", encoding="utf-8")
    findings = validation.gametest_resource_errors(root, log)
    assert findings
    assert "fabric.mod.json" in findings[0]


def test_gametest_xml_requires_real_passing_tests(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    report = root / "build/gametest-report.xml"
    report.parent.mkdir(parents=True)

    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite>\n',
        encoding="utf-8",
    )
    assert _passing_gametest_xml(root, report) is True

    for payload in (
        '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
        '<testsuite tests="1" failures="1" errors="0" skipped="0"><testcase name="bad"><failure/></testcase></testsuite>',
        '<testsuite tests="1" failures="0" errors="0" skipped="1"><testcase name="skip"><skipped/></testcase></testsuite>',
        '<broken>',
    ):
        report.write_text(payload, encoding="utf-8")
        assert _passing_gametest_xml(root, report) is False


def test_build_locked_preserves_clean_build_evidence_and_rejects_bad_xml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _project(tmp_path / "project")
    runner = GradleRunner(tmp_path / "cache")
    executable = tmp_path / "cache/gradle-test/bin/gradle"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    jar = root / "build/libs/demo.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"jar")
    calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        runner_module,
        "adapter_from_project",
        lambda _root: SimpleNamespace(gradle="test", gradle_sha256="a" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_ensure_gradle",
        MethodType(lambda self, _version, _sha: executable, runner),
    )
    monkeypatch.setattr(
        runner,
        "_find_release_jar",
        MethodType(lambda self, _root: str(jar), runner),
    )
    monkeypatch.setattr(
        parallel,
        "_sync_wrapper",
        lambda *args, **kwargs: CommandResult(
            name="wrapper",
            command=("wrapper",),
            exit_code=0,
            duration_seconds=0.0,
            log_path=str(root / ".minecraft_ai/logs/gradle-wrapper.log"),
            timed_out=False,
        ),
    )

    def fake_run(self, *, name, executable, arguments, cwd, env, log_path):
        calls.append((name, tuple(arguments)))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("clean\n", encoding="utf-8")
        if name == "gametest":
            report = root / "build/gametest-report.xml"
            report.write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="1"><testcase name="skip"><skipped/></testcase></testsuite>\n',
                encoding="utf-8",
            )
        return CommandResult(
            name=name,
            command=(str(executable), *arguments),
            exit_code=0,
            duration_seconds=0.0,
            log_path=str(log_path),
            timed_out=False,
        )

    monkeypatch.setattr(runner, "_run", MethodType(fake_run, runner))
    result = runner._build_locked(root, run_gametest=True)

    assert calls[0][0] == "clean_build"
    assert calls[0][1][:3] == ("--no-daemon", "clean", "build")
    assert "--build-cache" in calls[0][1]
    assert result.status == "FAIL"
    assert "GameTest report" in str(result.error)


def test_build_rejects_symlink_project_root(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    link = tmp_path / "project-link"
    link.symlink_to(root, target_is_directory=True)
    runner, _calls = _fake_runner(tmp_path, root)
    try:
        runner.build(link, run_gametest=False)
    except runner_module.BuildRunnerError as exc:
        assert "symbolic link" in str(exc)
    else:
        raise AssertionError("symlink project root must not be validated")


def test_safe_output_file_rejects_symlink_and_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    inside = root / "inside.jar"
    inside.write_bytes(b"ok")
    assert _safe_regular_file(root, inside) == inside.resolve()

    outside = tmp_path / "outside.jar"
    outside.write_bytes(b"outside")
    assert _safe_regular_file(root, outside) is None

    link = root / "link.jar"
    link.symlink_to(inside)
    assert _safe_regular_file(root, link) is None
