from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import minecraft_mod_ai.runner as runner_module
from minecraft_mod_ai.runner import CommandResult, GradleRunner


def test_verified_gradle_installation_is_reused_without_archive_io(
    monkeypatch,
    tmp_path: Path,
) -> None:
    version = "8.11"
    digest = "a" * 64
    distribution = tmp_path / f"gradle-{version}"
    executable = distribution / "bin" / ("gradle.bat" if os.name == "nt" else "gradle")
    executable.parent.mkdir(parents=True)
    executable.write_text("stub", encoding="utf-8")
    (distribution / ".minecraft-ai-gradle-sha256").write_text(
        digest + "\n",
        encoding="ascii",
    )

    def unexpected_archive_hash(_path: Path) -> str:
        raise AssertionError("verified Gradle reuse must not read/hash the archive")

    monkeypatch.setattr(runner_module, "_sha256", unexpected_archive_hash)

    actual = GradleRunner(tmp_path)._ensure_gradle(version, digest)

    assert actual == executable


def test_current_wrapper_is_not_regenerated(tmp_path: Path) -> None:
    version = "8.11"
    digest = "b" * 64
    wrapper_dir = tmp_path / "gradle" / "wrapper"
    wrapper_dir.mkdir(parents=True)
    (tmp_path / ("gradlew.bat" if os.name == "nt" else "gradlew")).write_text(
        "stub",
        encoding="utf-8",
    )
    (wrapper_dir / "gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/"
        f"gradle-{version}-bin.zip\n"
        f"distributionSha256Sum={digest}\n",
        encoding="utf-8",
    )

    assert GradleRunner._wrapper_is_current(tmp_path, version, digest) is True
    assert GradleRunner._wrapper_is_current(tmp_path, version, "c" * 64) is False


def test_build_hot_path_is_incremental_and_skips_current_wrapper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    cache = tmp_path / "cache"
    gradle = cache / "gradle-8.11" / "bin" / ("gradle.bat" if os.name == "nt" else "gradle")

    monkeypatch.setattr(
        runner_module,
        "adapter_from_project",
        lambda _root: SimpleNamespace(gradle="8.11", gradle_sha256="d" * 64),
    )
    runner = GradleRunner(cache)
    monkeypatch.setattr(runner, "_ensure_gradle", lambda _version, _sha: gradle)
    monkeypatch.setattr(runner, "_wrapper_is_current", lambda *_args: True)
    release_jar = project / "build/libs/mod.jar"
    release_jar.parent.mkdir(parents=True, exist_ok=True)
    release_jar.write_bytes(b"jar")
    monkeypatch.setattr(runner, "_find_release_jar", lambda _root: str(release_jar))

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(*, name, executable, arguments, cwd, env, log_path):
        calls.append((name, tuple(arguments)))
        return CommandResult(
            name=name,
            command=(str(executable), *arguments),
            exit_code=0,
            duration_seconds=0.01,
            log_path=str(log_path),
        )

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.delenv("MMM_GRADLE_FORCE_CLEAN", raising=False)

    report = runner._build_locked(project, run_gametest=False)

    assert report.status == "PASS"
    assert calls[-1][0] == "incremental_build"
    assert "build" in calls[-1][1]
    assert "clean" not in calls[-1][1]
    assert "--build-cache" in calls[-1][1]
    assert "clean" not in calls[0][1]

def test_compile_java_runs_only_compile_task(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    cache = tmp_path / "cache"
    gradle = cache / "gradle-8.11" / "bin" / ("gradle.bat" if os.name == "nt" else "gradle")
    logs = project / ".minecraft_ai" / "logs"
    logs.mkdir(parents=True)

    prepared = runner_module._PreparedBuild(
        project_root=project,
        gradle_version="8.11",
        gradle_sha256="e" * 64,
        gradle=gradle,
        logs=logs,
        environment={},
    )
    runner = GradleRunner(cache)
    monkeypatch.setattr(runner, "_prepare_build_context", lambda _root: prepared)

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(*, name, executable, arguments, cwd, env, log_path):
        calls.append((name, tuple(arguments)))
        return CommandResult(
            name=name,
            command=(str(executable), *arguments),
            exit_code=0,
            duration_seconds=0.01,
            log_path=str(log_path),
        )

    monkeypatch.setattr(runner, "_run", fake_run)

    report = runner.compile_java(project)

    assert report.status == "PASS"
    assert report.jar_path is None
    assert report.gametest_report is None
    assert calls == [
        ("compile_java", ("--no-daemon", "compileJava", "--stacktrace"))
    ]

