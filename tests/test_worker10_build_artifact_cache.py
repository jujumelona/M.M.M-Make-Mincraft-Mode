from __future__ import annotations

from pathlib import Path
from types import MethodType

from minecraft_mod_ai import validation_execution_contract as validation
from minecraft_mod_ai.runner import BuildReport, GradleRunner
from minecraft_mod_ai.runner_parallel_validation_contract import _safe_regular_file


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
                f'<testsuite tests="1" failures="0" errors="0" skipped="0" name="run-{calls["count"]}"/>\n',
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
