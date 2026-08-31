from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.runner_parallel_validation_contract import install


@dataclass(frozen=True)
class _CommandResult:
    name: str
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    log_path: str
    timed_out: bool = False


@dataclass(frozen=True)
class _BuildReport:
    status: str
    gradle_version: str
    commands: tuple[object, ...]
    jar_path: str | None
    gametest_report: str | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class _FakeGradleRunner:
    active_builds = 0
    max_active_builds = 0
    build_calls = 0
    counter_lock = threading.Lock()

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.command_timeout_seconds = 10
        self.ensure_calls: list[tuple[str, str]] = []

    def _ensure_gradle(self, gradle_version: str, gradle_sha256: str) -> Path:
        self.ensure_calls.append((gradle_version, gradle_sha256))
        executable = self.cache_dir / f"gradle-{gradle_version}" / "bin" / (
            "gradle.bat" if os.name == "nt" else "gradle"
        )
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("fake", encoding="utf-8")
        return executable

    def build(self, project_root: Path, *, run_gametest: bool = True):
        raise AssertionError("legacy whole-build lock path must be bypassed")

    def _run(self, *, name, executable, arguments, cwd, env, log_path):
        del executable, env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(name, encoding="utf-8")
        if name == "wrapper":
            (cwd / "gradle/wrapper").mkdir(parents=True, exist_ok=True)
            (cwd / "gradlew").write_text("wrapper", encoding="utf-8")
            (cwd / "gradlew.bat").write_text("wrapper", encoding="utf-8")
            (cwd / "gradle/wrapper/gradle-wrapper.jar").write_bytes(b"jar")
            version = arguments[arguments.index("--gradle-version") + 1]
            sha256 = arguments[
                arguments.index("--gradle-distribution-sha256-sum") + 1
            ]
            (cwd / "gradle/wrapper/gradle-wrapper.properties").write_text(
                f"distributionUrl=https://example/gradle-{version}-bin.zip\n"
                f"distributionSha256Sum={sha256}\n",
                encoding="utf-8",
            )
        elif name == "clean_build":
            with self.counter_lock:
                type(self).active_builds += 1
                type(self).build_calls += 1
                type(self).max_active_builds = max(
                    type(self).max_active_builds,
                    type(self).active_builds,
                )
            try:
                time.sleep(0.1)
                libs = cwd / "build/libs"
                libs.mkdir(parents=True, exist_ok=True)
                (libs / "mod.jar").write_bytes(b"jar")
            finally:
                with self.counter_lock:
                    type(self).active_builds -= 1
        return _CommandResult(
            name=name,
            command=(name,),
            exit_code=0,
            duration_seconds=0.01,
            log_path=str(log_path),
        )

    @staticmethod
    def _find_release_jar(project_root: Path) -> str | None:
        jar = project_root / "build/libs/mod.jar"
        return str(jar) if jar.is_file() else None

    @staticmethod
    def _gametest_report(project_root: Path) -> str | None:
        del project_root
        return None


_CACHE_LOCK = threading.RLock()
_SUCCESSFUL_BUILDS = {}
_RECENT_BUILDS = {}


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("build.gradle", "gradle.properties"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _bounded_put(mapping, key, value):
    mapping[key] = value


_cache_lock = threading.RLock()


@contextmanager
def _exclusive_cache_lock(cache_dir: Path, *, timeout_seconds: int):
    del cache_dir, timeout_seconds
    with _cache_lock:
        yield


def _runner_module():
    def adapter_from_project(root: Path):
        config = json.loads((root / "target.json").read_text(encoding="utf-8"))
        return SimpleNamespace(
            gradle=config["gradle"],
            gradle_sha256=config["sha256"],
        )

    return SimpleNamespace(
        GradleRunner=_FakeGradleRunner,
        BuildRunnerError=RuntimeError,
        BuildReport=_BuildReport,
        CommandResult=_CommandResult,
        adapter_from_project=adapter_from_project,
        _exclusive_cache_lock=_exclusive_cache_lock,
    )


def _validation_module():
    return SimpleNamespace(
        _CACHE_LOCK=_CACHE_LOCK,
        _SUCCESSFUL_BUILDS=_SUCCESSFUL_BUILDS,
        _RECENT_BUILDS=_RECENT_BUILDS,
        _bounded_put=_bounded_put,
        project_build_fingerprint=_fingerprint,
        gametest_resource_errors=lambda *_args, **_kwargs: (),
    )


def _project(root: Path, name: str, version: str, sha256: str) -> Path:
    project = root / name
    project.mkdir(parents=True)
    (project / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (project / "target.json").write_text(
        json.dumps({"gradle": version, "sha256": sha256}),
        encoding="utf-8",
    )
    return project


def _reset() -> None:
    _SUCCESSFUL_BUILDS.clear()
    _RECENT_BUILDS.clear()
    _FakeGradleRunner.active_builds = 0
    _FakeGradleRunner.max_active_builds = 0
    _FakeGradleRunner.build_calls = 0


def test_target_distribution_api_uses_explicit_version_and_sha(tmp_path: Path) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    runner = _FakeGradleRunner(tmp_path / "cache")

    version = "8.10.2"
    sha256 = "a" * 64
    first = runner._ensure_gradle(version, sha256)
    second = runner._ensure_gradle(version, sha256)

    assert first == second
    assert runner.ensure_calls == [(version, sha256)]


def test_different_projects_validate_in_parallel(tmp_path: Path) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    cache = tmp_path / "cache"
    runner_a = _FakeGradleRunner(cache)
    runner_b = _FakeGradleRunner(cache)
    project_a = _project(tmp_path, "a", "8.10.2", "a" * 64)
    project_b = _project(tmp_path, "b", "8.11.1", "b" * 64)
    barrier = threading.Barrier(3)
    errors = []

    def worker(runner, project):
        barrier.wait()
        try:
            report = runner.build(project, run_gametest=False)
            assert report.passed
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(runner_a, project_a)),
        threading.Thread(target=worker, args=(runner_b, project_b)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert not errors
    assert _FakeGradleRunner.max_active_builds == 2
    assert _FakeGradleRunner.build_calls == 2


def test_same_project_is_single_writer_and_exact_cache_reused(tmp_path: Path) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    cache = tmp_path / "cache"
    runner_a = _FakeGradleRunner(cache)
    runner_b = _FakeGradleRunner(cache)
    project = _project(tmp_path, "same", "8.10.2", "c" * 64)
    barrier = threading.Barrier(3)
    reports = []

    def worker(runner):
        barrier.wait()
        reports.append(runner.build(project, run_gametest=False))

    threads = [
        threading.Thread(target=worker, args=(runner_a,)),
        threading.Thread(target=worker, args=(runner_b,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert len(reports) == 2
    assert all(report.passed for report in reports)
    assert _FakeGradleRunner.max_active_builds == 1
    assert _FakeGradleRunner.build_calls == 1
