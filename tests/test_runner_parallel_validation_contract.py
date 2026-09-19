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

from minecraft_mod_ai.runner_parallel_validation_contract import (
    _executed_gametest_task,
    _structured_gametest_report,
    _task_from_listing,
    install,
)


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
    gametest_mode: str | None = None
    gametest_task: str | None = None
    failure_class: str | None = None
    error_code: str | None = None
    repairable: bool | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class _FakeGradleRunner:
    active_builds = 0
    max_active_builds = 0
    build_calls = 0
    run_calls: list[str] = []
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
        type(self).run_calls.append(name)
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
        elif name in {"clean_build", "incremental_build"}:
            if name == "incremental_build" and (cwd / "simulate-integrated-gametest").is_file():
                log_path.write_text(
                    "> Task :configureLaunch\n> Task :runGameTest\nBUILD SUCCESSFUL\n",
                    encoding="utf-8",
                )
                report = cwd / "build/gametest-report.xml"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    '<testsuite tests="1" failures="0" errors="0" skipped="0">'
                    '<testcase name="MmmDebugFixtureModGameTests.generatedRegistriesAreLive"/>'
                    "</testsuite>",
                    encoding="utf-8",
                )
            elif (
                name == "incremental_build"
                and (cwd / "simulate-integrated-gametest-log-only").is_file()
            ):
                log_path.write_text(
                    "> Task :configureLaunch\n"
                    "> Task :runGameTest\n"
                    "2 tests are now running at position 0, 0, 0!\n"
                    "========= 2 GAME TESTS COMPLETE IN 7.570 s ======================\n"
                    "All 2 required tests passed :)\n"
                    "> Task :build\n"
                    "BUILD SUCCESSFUL\n",
                    encoding="utf-8",
                )
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
        elif name == "gametest_capabilities":
            log_path.write_text(
                "runGameTest - Runs server game tests\n",
                encoding="utf-8",
            )
        elif name == "gametest":
            report = cwd / "build/gametest-report.xml"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="0">'
                '<testcase name="MmmDebugFixtureModGameTests.generatedRegistriesAreLive"/>'
                "</testsuite>",
                encoding="utf-8",
            )
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



def _install_host_gametest_fixture(
    project: Path,
    *,
    include_bootstrap: bool = True,
) -> None:
    source = (
        project
        / "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureModGameTests.java"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "package dev.mmm.debugfixture;\n\n"
        "import net.fabricmc.fabric.api.gametest.v1.GameTest;\n"
        "import net.fabricmc.loader.api.FabricLoader;\n"
        "import net.minecraft.gametest.framework.GameTestHelper;\n\n"
        "public final class MmmDebugFixtureModGameTests {\n"
        "    @GameTest\n"
        "    public void generatedRegistriesAreLive(GameTestHelper context) {\n"
        "        if (!FabricLoader.getInstance().isModLoaded(\"mmm_debug_fixture\")) {\n"
        "            throw new AssertionError(\"generated mod was not loaded by Fabric\");\n"
        "        }\n"
        "        context.succeed();\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (project / "build.gradle").write_text(
        """plugins {}

fabricApi {
    configureTests {
        createSourceSet = false
        enableGameTests = true
        enableClientGameTests = false
    }
}

loom {
    runs {
        gameTest {
            property "fabric-api.gametest.report-file", file('build/gametest-report.xml').absolutePath
        }
    }
}
""",
        encoding="utf-8",
    )
    metadata = project / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(
            {
                "id": "mmm_debug_fixture",
                "entrypoints": {
                    "fabric-gametest": [
                        "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    lock_payload = {
        "adapter_id": "test",
        "loader": "fabric",
        "minecraft_version": "26.2",
        "java_version": "25",
        "fabric_loader": "0.19.5",
        "fabric_api": "0.140.0+26.2",
        "fabric_loom": "1.17-SNAPSHOT",
        "gradle": "8.10.2",
        "gradle_sha256": "f" * 64,
    }
    if include_bootstrap:
        lock_payload["bootstrap"] = {
            "gametest_contract": {
                "task": "runGameTest",
                "report": "build/gametest-report.xml",
                "entrypoint": (
                    "dev.mmm.debugfixture.MmmDebugFixtureModGameTests"
                ),
                "source": (
                    "src/main/java/dev/mmm/debugfixture/"
                    "MmmDebugFixtureModGameTests.java"
                ),
            }
        }
    lock = project / ".minecraft_ai/platform-lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(lock_payload), encoding="utf-8")


def _reset() -> None:
    _SUCCESSFUL_BUILDS.clear()
    _RECENT_BUILDS.clear()
    _FakeGradleRunner.active_builds = 0
    _FakeGradleRunner.max_active_builds = 0
    _FakeGradleRunner.build_calls = 0
    _FakeGradleRunner.run_calls = []


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


def test_integrated_build_gametest_is_reused_without_legacy_task_call(
    tmp_path: Path,
) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    runner = _FakeGradleRunner(tmp_path / "cache")
    project = _project(tmp_path, "integrated", "8.10.2", "d" * 64)
    (project / "simulate-integrated-gametest").write_text("1", encoding="utf-8")

    report = runner.build(project, run_gametest=True)

    assert report.passed
    assert report.gametest_mode == "integrated_build"
    assert report.gametest_task == "runGameTest"
    assert "gametest" not in _FakeGradleRunner.run_calls
    assert "gametest_capabilities" not in _FakeGradleRunner.run_calls


def test_gametest_helpers_prefer_real_run_game_test_evidence(tmp_path: Path) -> None:
    build_log = tmp_path / "build.log"
    build_log.write_text(
        "> Task :configureLaunch\n> Task :runGameTest\nBUILD SUCCESSFUL\n",
        encoding="utf-8",
    )
    tasks_log = tmp_path / "tasks.log"
    tasks_log.write_text(
        "runGameTestServer - Legacy run configuration\n"
        "runGameTest - Runs server game tests\n",
        encoding="utf-8",
    )

    assert _executed_gametest_task(build_log) == "runGameTest"
    assert _task_from_listing(tasks_log) == "runGameTest"


def test_fallback_discovers_real_gradle_task_instead_of_hardcoding_server(
    tmp_path: Path,
) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    runner = _FakeGradleRunner(tmp_path / "cache")
    project = _project(tmp_path, "fallback", "8.10.2", "e" * 64)

    report = runner.build(project, run_gametest=True)

    assert report.passed
    assert report.gametest_mode == "explicit_task"
    assert report.gametest_task == "runGameTest"
    assert "gametest_capabilities" in _FakeGradleRunner.run_calls
    assert "gametest" in _FakeGradleRunner.run_calls


def test_log_only_integrated_gametest_reconstructs_host_bound_xml(
    tmp_path: Path,
) -> None:
    _reset()
    runner_module = _runner_module()
    install(runner_module=runner_module, validation_module=_validation_module())
    runner = _FakeGradleRunner(tmp_path / "cache")
    project = _project(tmp_path, "log-only", "8.10.2", "f" * 64)
    _install_host_gametest_fixture(project)
    (project / "simulate-integrated-gametest-log-only").write_text(
        "1",
        encoding="utf-8",
    )

    report = runner.build(project, run_gametest=True)

    assert report.passed
    assert report.gametest_mode == "integrated_build"
    assert report.gametest_task == "runGameTest"
    assert report.gametest_report is not None
    assert Path(report.gametest_report).name == "mmm-gametest-attestation.xml"
    assert "gametest" not in _FakeGradleRunner.run_calls
    assert "gametest_capabilities" not in _FakeGradleRunner.run_calls


def test_log_summary_cannot_create_evidence_without_host_contract(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, "unbound", "8.10.2", "1" * 64)
    log = project / ".minecraft_ai/logs/gradle-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "> Task :runGameTest\n"
        "========= 2 GAME TESTS COMPLETE IN 1.0 s ======================\n"
        "All 2 required tests passed :)\n"
        "BUILD SUCCESSFUL\n",
        encoding="utf-8",
    )

    assert _structured_gametest_report(
        project,
        log,
        project / "build/gametest-report.xml",
    ) is None


def test_log_only_gametest_reconstructs_contract_when_bootstrap_receipt_is_missing(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, "receipt-missing", "8.10.2", "f" * 64)
    _install_host_gametest_fixture(project, include_bootstrap=False)
    log = project / ".minecraft_ai/logs/gradle-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "> Task :runGameTest\n"
        "[12:45:11] [Server thread/INFO] (Minecraft) "
        "========= 2 GAME TESTS COMPLETE IN 4.935 s ======================\n"
        "[12:45:11] [Server thread/INFO] (Minecraft) "
        "All 2 required tests passed :)\n"
        "> Task :build\n"
        "BUILD SUCCESSFUL\n",
        encoding="utf-8",
    )

    report = _structured_gametest_report(
        project,
        log,
        project / "build/gametest-report.xml",
    )

    assert report is not None
    assert report.name == "mmm-gametest-attestation.xml"


def test_log_only_gametest_rejects_tampered_host_source_without_bootstrap(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, "tampered-host-source", "8.10.2", "f" * 64)
    _install_host_gametest_fixture(project, include_bootstrap=False)
    source = (
        project
        / "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureModGameTests.java"
    )
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "context.succeed();",
            "throw new AssertionError(\"tampered\");",
        ),
        encoding="utf-8",
    )
    log = project / ".minecraft_ai/logs/gradle-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "> Task :runGameTest\n"
        "========= 2 GAME TESTS COMPLETE IN 1.0 s ======================\n"
        "All 2 required tests passed :)\n"
        "BUILD SUCCESSFUL\n",
        encoding="utf-8",
    )

    assert _structured_gametest_report(
        project,
        log,
        project / "build/gametest-report.xml",
    ) is None
