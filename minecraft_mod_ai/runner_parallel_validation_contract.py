from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any

_PROJECT_BUILD_LOCKS_GUARD = threading.Lock()
_PROJECT_BUILD_LOCKS: dict[str, threading.RLock] = {}
_BUILD_POLICY_ENV = (
    "JAVA_HOME",
    "JDK_HOME",
    "PATH",
    "GRADLE_OPTS",
    "JAVA_OPTS",
)


def _path_lock(path: Path) -> threading.RLock:
    """Return the exact single-writer lock for one canonical project root."""
    key = str(path.expanduser().resolve())
    with _PROJECT_BUILD_LOCKS_GUARD:
        lock = _PROJECT_BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_BUILD_LOCKS[key] = lock
        return lock


def _safe_regular_file(root: Path, path_value: str | Path | None) -> Path | None:
    """Resolve a regular file while rejecting root escape and every symlink hop."""

    if path_value is None:
        return None
    root = Path(root).expanduser().resolve()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return None

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _file_digest(root: Path, path_value: str | Path | None) -> str | None:
    path = _safe_regular_file(root, path_value)
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _build_cache_profile(self: Any) -> tuple[Any, ...]:
    return (
        str(Path(self.cache_dir).expanduser().resolve()),
        int(getattr(self, "download_timeout_seconds", 0)),
        int(getattr(self, "command_timeout_seconds", 0)),
        tuple((name, os.environ.get(name, "")) for name in _BUILD_POLICY_ENV),
    )


def _cache_entry(root: Path, report: Any, *, run_gametest: bool) -> dict[str, Any] | None:
    if not getattr(report, "passed", False):
        return None
    jar_digest = _file_digest(root, getattr(report, "jar_path", None))
    if jar_digest is None:
        return None
    gametest_digest: str | None = None
    if run_gametest:
        gametest_digest = _file_digest(root, getattr(report, "gametest_report", None))
        if gametest_digest is None:
            return None
    return {
        "report": copy.deepcopy(report),
        "jar_sha256": jar_digest,
        "gametest_sha256": gametest_digest,
    }


def _cached_report(
    root: Path,
    cached: Any,
    *,
    run_gametest: bool,
) -> Any | None:
    if not isinstance(cached, dict):
        return None
    report = cached.get("report")
    if not getattr(report, "passed", False):
        return None
    jar_digest = cached.get("jar_sha256")
    if not isinstance(jar_digest, str) or _file_digest(
        root, getattr(report, "jar_path", None)
    ) != jar_digest:
        return None
    if run_gametest:
        gametest_digest = cached.get("gametest_sha256")
        if not isinstance(gametest_digest, str) or _file_digest(
            root, getattr(report, "gametest_report", None)
        ) != gametest_digest:
            return None
    return copy.deepcopy(report)


def _uncertifiable_report(runner_module: Any, report: Any, message: str) -> Any:
    return runner_module.BuildReport(
        status="FAIL",
        gradle_version=report.gradle_version,
        commands=report.commands,
        jar_path=report.jar_path,
        gametest_report=report.gametest_report,
        error=message,
    )


def _marker_path(cache_dir: Path, version: str, sha256: str) -> Path:
    token = hashlib.sha256(f"{version}\0{sha256}".encode()).hexdigest()[:16]
    return cache_dir / f".mmm-gradle-{version}-{token}-verified.json"


def _distribution_executable(cache_dir: Path, version: str) -> Path:
    return cache_dir / f"gradle-{version}" / "bin" / (
        "gradle.bat" if os.name == "nt" else "gradle"
    )


def _valid_distribution_marker(path: Path, version: str, sha256: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == "mmm/verified-gradle-cache-v2"
        and payload.get("version") == version
        and payload.get("distribution_sha256") == sha256
    )


def _write_distribution_marker(path: Path, version: str, sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "mmm/verified-gradle-cache-v2",
                "version": version,
                "distribution_sha256": sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _wrapper_template_dir(cache_dir: Path, version: str, sha256: str) -> Path:
    token = hashlib.sha256(f"{version}\0{sha256}".encode()).hexdigest()[:12]
    return cache_dir / f"wrapper-template-{version}-{token}"


def _wrapper_artifacts(root: Path) -> tuple[Path, ...]:
    return (
        root / "gradlew",
        root / "gradlew.bat",
        root / "gradle/wrapper/gradle-wrapper.jar",
        root / "gradle/wrapper/gradle-wrapper.properties",
    )


def _valid_wrapper_template(root: Path, version: str, sha256: str) -> bool:
    if root.is_symlink() or not root.is_dir():
        return False
    artifacts = _wrapper_artifacts(root)
    if not all(path.is_file() and not path.is_symlink() for path in artifacts):
        return False
    try:
        properties = artifacts[-1].read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        f"gradle-{version}-bin.zip" in properties
        and f"distributionSha256Sum={sha256}" in properties
    )


def _copy_wrapper_template(source: Path, destination: Path) -> None:
    for source_path in _wrapper_artifacts(source):
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        if target.name == "gradlew" and os.name != "nt":
            target.chmod(0o755)


def _sync_wrapper(
    self: Any,
    *,
    runner_module: Any,
    gradle: Path,
    project_root: Path,
    environment: dict[str, str],
    logs: Path,
    version: str,
    sha256: str,
) -> Any:
    started = time.monotonic()
    template = _wrapper_template_dir(self.cache_dir, version, sha256)
    log_path = logs / "gradle-wrapper.log"

    if not _valid_wrapper_template(template, version, sha256):
        with runner_module._exclusive_cache_lock(
            self.cache_dir,
            timeout_seconds=max(300, self.command_timeout_seconds * 3),
        ):
            if not _valid_wrapper_template(template, version, sha256):
                if template.is_symlink():
                    template.unlink()
                elif template.exists():
                    shutil.rmtree(template)
                template.mkdir(parents=True, exist_ok=False)
                (template / "settings.gradle").write_text(
                    "rootProject.name = 'mmm-wrapper-template'\n",
                    encoding="utf-8",
                )
                (template / "build.gradle").write_text("\n", encoding="utf-8")
                result = self._run(
                    name="wrapper",
                    executable=gradle,
                    arguments=(
                        "--no-daemon",
                        "wrapper",
                        "--gradle-version",
                        version,
                        "--gradle-distribution-sha256-sum",
                        sha256,
                        "--stacktrace",
                    ),
                    cwd=template,
                    env=environment,
                    log_path=log_path,
                )
                if result.exit_code != 0 or not _valid_wrapper_template(
                    template, version, sha256
                ):
                    return result

    _copy_wrapper_template(template, project_root)
    duration = round(time.monotonic() - started, 3)
    log_path.write_text(
        "Verified target-specific Gradle wrapper template copied into project.\n",
        encoding="utf-8",
    )
    return runner_module.CommandResult(
        name="wrapper",
        command=("mmm:verified-wrapper-cache", version),
        exit_code=0,
        duration_seconds=duration,
        log_path=str(log_path),
        timed_out=False,
    )


def install(*, runner_module: Any, validation_module: Any) -> None:
    cls = runner_module.GradleRunner

    current_ensure = cls._ensure_gradle
    if not getattr(current_ensure, "_mmm_target_parallel_distribution", False):
        base_ensure = current_ensure

        @wraps(base_ensure)
        def target_cached_ensure(
            self: Any,
            gradle_version: str,
            gradle_sha256: str,
        ) -> Path:
            version = str(gradle_version).strip()
            sha256 = str(gradle_sha256).strip().lower()
            if not version or len(sha256) != 64:
                raise runner_module.BuildRunnerError(
                    "Gradle version/SHA-256 cache key is invalid."
                )
            executable = _distribution_executable(self.cache_dir, version)
            marker = _marker_path(self.cache_dir, version, sha256)
            safe_executable = _safe_regular_file(self.cache_dir, executable)
            if safe_executable is not None and _valid_distribution_marker(
                marker, version, sha256
            ):
                return safe_executable

            with runner_module._exclusive_cache_lock(
                self.cache_dir,
                timeout_seconds=max(300, self.command_timeout_seconds * 3),
            ):
                safe_executable = _safe_regular_file(self.cache_dir, executable)
                if safe_executable is not None and _valid_distribution_marker(
                    marker, version, sha256
                ):
                    return safe_executable
                executable = base_ensure(self, version, sha256)
                safe_executable = _safe_regular_file(self.cache_dir, executable)
                if safe_executable is None:
                    raise runner_module.BuildRunnerError(
                        "Gradle executable is missing or unsafe after extraction."
                    )
                _write_distribution_marker(marker, version, sha256)
                return safe_executable

        target_cached_ensure._mmm_target_parallel_distribution = True
        cls._ensure_gradle = target_cached_ensure

    def target_build_locked(
        self: Any,
        project_root: Path,
        *,
        run_gametest: bool,
    ) -> Any:
        root = Path(project_root).expanduser().resolve()
        if not (root / "build.gradle").is_file():
            raise runner_module.BuildRunnerError(
                f"Not a generated Gradle project: {root}"
            )
        try:
            adapter = runner_module.adapter_from_project(root)
        except ValueError as exc:
            raise runner_module.BuildRunnerError(
                f"Project platform lock is missing, mixed, or unsupported: {exc}"
            ) from exc
        version = str(adapter.gradle)
        sha256 = str(adapter.gradle_sha256)
        state = root / ".minecraft_ai"
        logs = state / "logs"
        if state.is_symlink() or logs.is_symlink():
            raise runner_module.BuildRunnerError(
                "Validation log state must not traverse symbolic links."
            )
        logs.mkdir(parents=True, exist_ok=True)

        gradle = self._ensure_gradle(version, sha256)
        environment = os.environ.copy()
        environment["GRADLE_USER_HOME"] = str(self.cache_dir / "gradle-user-home")
        environment["CI"] = "true"
        commands: list[Any] = []
        gametest_report: str | None = None

        wrapper_result = _sync_wrapper(
            self,
            runner_module=runner_module,
            gradle=gradle,
            project_root=root,
            environment=environment,
            logs=logs,
            version=version,
            sha256=sha256,
        )
        commands.append(wrapper_result)
        if wrapper_result.exit_code != 0:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=version,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=None,
                error="Gradle wrapper generation failed.",
            )

        build_result = self._run(
            name="build",
            executable=gradle,
            arguments=("--no-daemon", "build", "--build-cache", "--stacktrace"),
            cwd=root,
            env=environment,
            log_path=logs / "gradle-build.log",
        )
        commands.append(build_result)
        if build_result.exit_code != 0:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=version,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=None,
                error="Gradle build failed.",
            )

        if run_gametest:
            gametest_result = self._run(
                name="gametest",
                executable=gradle,
                arguments=(
                    "--no-daemon",
                    "runGameTestServer",
                    "--build-cache",
                    "--stacktrace",
                ),
                cwd=root,
                env=environment,
                log_path=logs / "gradle-gametest.log",
            )
            commands.append(gametest_result)
            if gametest_result.exit_code != 0:
                return runner_module.BuildReport(
                    status="FAIL",
                    gradle_version=version,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(root),
                    gametest_report=None,
                    error="Headless Fabric GameTest failed.",
                )
            resource_errors = validation_module.gametest_resource_errors(
                root,
                gametest_result.log_path,
            )
            if resource_errors:
                return runner_module.BuildReport(
                    status="FAIL",
                    gradle_version=version,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(root),
                    gametest_report=None,
                    error=(
                        "Headless Fabric GameTest resource evidence failed: "
                        + " | ".join(resource_errors[:8])
                    ),
                )
            report_candidate = root / "build" / "gametest-report.xml"
            safe_report = _safe_regular_file(root, report_candidate)
            if safe_report is None:
                return runner_module.BuildReport(
                    status="FAIL",
                    gradle_version=version,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(root),
                    gametest_report=None,
                    error=(
                        "GameTest reported success but no safe gametest-report.xml "
                        "evidence was produced."
                    ),
                )
            gametest_report = str(safe_report)

        jar_path = self._find_release_jar(root)
        if jar_path is None:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=version,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=gametest_report,
                error="Gradle reported success but no remapped release JAR was found.",
            )
        return runner_module.BuildReport(
            status="PASS",
            gradle_version=version,
            commands=tuple(commands),
            jar_path=jar_path,
            gametest_report=gametest_report,
            error=None,
        )

    target_build_locked._mmm_target_parallel_build_locked = True
    cls._build_locked = target_build_locked

    current_build = cls.build
    if getattr(current_build, "_mmm_project_parallel_validation", False):
        return

    @wraps(current_build)
    def parallel_cached_build(
        self: Any,
        project_root: Path,
        *,
        run_gametest: bool = True,
    ) -> Any:
        root = Path(project_root).expanduser().resolve()
        with _path_lock(root):
            fingerprint = validation_module.project_build_fingerprint(root)
            profile = _build_cache_profile(self)
            key = (str(root), fingerprint, bool(run_gametest), profile)
            with validation_module._CACHE_LOCK:
                cached = validation_module._SUCCESSFUL_BUILDS.get(key)
                reusable = _cached_report(root, cached, run_gametest=run_gametest)
                if reusable is not None:
                    return reusable
                if cached is not None:
                    validation_module._SUCCESSFUL_BUILDS.pop(key, None)

            report = self._build_locked(root, run_gametest=run_gametest)
            final_fingerprint = validation_module.project_build_fingerprint(root)
            if final_fingerprint != fingerprint:
                return _uncertifiable_report(
                    runner_module,
                    report,
                    "Project inputs changed during validation; result is not certifiable.",
                )

            entry = _cache_entry(root, report, run_gametest=run_gametest)
            if report.passed and entry is None:
                return _uncertifiable_report(
                    runner_module,
                    report,
                    "Validation outputs changed, disappeared, or became unsafe before certification.",
                )
            if entry is not None:
                final_key = (str(root), final_fingerprint, bool(run_gametest), profile)
                with validation_module._CACHE_LOCK:
                    validation_module._bounded_put(
                        validation_module._SUCCESSFUL_BUILDS,
                        final_key,
                        entry,
                    )
            return report

    parallel_cached_build._mmm_project_parallel_validation = True
    parallel_cached_build._mmm_exact_input_cache = True
    parallel_cached_build._mmm_output_bound_validation_cache = True
    cls.build = parallel_cached_build


__all__ = ["install"]
