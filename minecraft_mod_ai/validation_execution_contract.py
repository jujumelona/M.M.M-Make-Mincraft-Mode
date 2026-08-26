from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any

_CACHE_LOCK = threading.RLock()
_SUCCESSFUL_BUILDS: dict[tuple[str, str, bool], Any] = {}
_RECENT_BUILDS: dict[tuple[str, str, bool], Any] = {}
_JDT_RESULTS: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
_CACHE_LIMIT = 24

_SKIP_TOP_LEVEL = {
    ".git",
    ".gradle",
    "build",
    "logs",
    "node_modules",
    "run",
}
_SKIP_PREFIXES = (
    ".minecraft_ai/logs/",
    ".minecraft_ai/runtime/",
    ".minecraft_ai/validation-cache/",
)
_SKIP_WRAPPER_PATHS = {
    "gradlew",
    "gradlew.bat",
    "gradle/wrapper/gradle-wrapper.jar",
    "gradle/wrapper/gradle-wrapper.properties",
}


def _bounded_put(mapping: dict[Any, Any], key: Any, value: Any) -> None:
    mapping[key] = value
    while len(mapping) > _CACHE_LIMIT:
        mapping.pop(next(iter(mapping)))


def _is_build_input(relative: str) -> bool:
    path = Path(relative)
    if path.parts and path.parts[0] in _SKIP_TOP_LEVEL:
        return False
    if relative in _SKIP_WRAPPER_PATHS:
        return False
    return not any(relative.startswith(prefix) for prefix in _SKIP_PREFIXES)


def project_build_fingerprint(project_root: str | Path) -> str:
    """Hash exact build-relevant project bytes, excluding outputs and evidence logs."""

    root = Path(project_root).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if not _is_build_input(relative):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _java_fingerprint(
    project_root: str | Path,
    relative_files: Iterable[str] | None,
) -> tuple[str, tuple[str, ...]]:
    root = Path(project_root).expanduser().resolve()
    if relative_files is None:
        paths = sorted(
            path
            for path in root.rglob("*.java")
            if path.is_file() and not path.is_symlink()
        )
        relative = tuple(path.relative_to(root).as_posix() for path in paths)
    else:
        relative = tuple(sorted(set(str(value).replace("\\", "/") for value in relative_files)))
        paths = [root / value for value in relative]

    digest = hashlib.sha256()
    for config_name in ("build.gradle", "settings.gradle", "gradle.properties"):
        config = root / config_name
        if config.is_file() and not config.is_symlink():
            digest.update(config_name.encode("utf-8"))
            digest.update(config.read_bytes())
    for rel, path in zip(relative, paths):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative


def _gradle_distribution_marker(runner_module: Any, cache_dir: Path) -> Path:
    return cache_dir / f".mmm-gradle-{runner_module.GRADLE_VERSION}-verified.json"


def _install_fast_gradle_distribution(runner_module: Any) -> None:
    cls = runner_module.GradleRunner
    original = cls._ensure_gradle
    if getattr(original, "_mmm_verified_distribution_cache", False):
        return

    @wraps(original)
    def cached_ensure_gradle(self: Any) -> Path:
        distribution_dir = self.cache_dir / f"gradle-{runner_module.GRADLE_VERSION}"
        executable = distribution_dir / "bin" / (
            "gradle.bat" if os.name == "nt" else "gradle"
        )
        marker = _gradle_distribution_marker(runner_module, self.cache_dir)
        if executable.is_file() and marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if (
                payload.get("version") == runner_module.GRADLE_VERSION
                and payload.get("distribution_sha256") == runner_module.GRADLE_SHA256
            ):
                return executable

        executable = original(self)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": "mmm/verified-gradle-cache-v1",
                    "version": runner_module.GRADLE_VERSION,
                    "distribution_sha256": runner_module.GRADLE_SHA256,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return executable

    cached_ensure_gradle._mmm_verified_distribution_cache = True
    cls._ensure_gradle = cached_ensure_gradle


def _wrapper_template_dir(cache_dir: Path, runner_module: Any) -> Path:
    return cache_dir / f"wrapper-template-{runner_module.GRADLE_VERSION}"


def _wrapper_artifacts(root: Path) -> tuple[Path, ...]:
    return (
        root / "gradlew",
        root / "gradlew.bat",
        root / "gradle/wrapper/gradle-wrapper.jar",
        root / "gradle/wrapper/gradle-wrapper.properties",
    )


def _valid_wrapper_template(root: Path, runner_module: Any) -> bool:
    artifacts = _wrapper_artifacts(root)
    if not all(path.is_file() and not path.is_symlink() for path in artifacts):
        return False
    try:
        properties = artifacts[-1].read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        f"gradle-{runner_module.GRADLE_VERSION}-bin.zip" in properties
        and f"distributionSha256Sum={runner_module.GRADLE_SHA256}" in properties
    )


def _copy_wrapper_template(source: Path, destination: Path) -> None:
    for source_path in _wrapper_artifacts(source):
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        if target.name == "gradlew" and os.name != "nt":
            target.chmod(0o755)


def _sync_verified_wrapper(
    self: Any,
    *,
    runner_module: Any,
    gradle: Path,
    project_root: Path,
    environment: dict[str, str],
    logs: Path,
) -> Any:
    started = time.monotonic()
    template = _wrapper_template_dir(self.cache_dir, runner_module)
    template.mkdir(parents=True, exist_ok=True)
    log_path = logs / "gradle-wrapper.log"

    if not _valid_wrapper_template(template, runner_module):
        # Generate the wrapper in a tiny Gradle project. Running `wrapper` inside the
        # generated Fabric/Loom project needlessly configures and remaps Minecraft.
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
                runner_module.GRADLE_VERSION,
                "--gradle-distribution-sha256-sum",
                runner_module.GRADLE_SHA256,
                "--stacktrace",
            ),
            cwd=template,
            env=environment,
            log_path=log_path,
        )
        if result.exit_code != 0 or not _valid_wrapper_template(template, runner_module):
            return result
        _copy_wrapper_template(template, project_root)
        return result

    _copy_wrapper_template(template, project_root)
    duration = round(time.monotonic() - started, 3)
    log_path.write_text(
        "Verified cached Gradle wrapper template copied into project.\n",
        encoding="utf-8",
    )
    return runner_module.CommandResult(
        name="wrapper",
        command=("mmm:verified-wrapper-cache", runner_module.GRADLE_VERSION),
        exit_code=0,
        duration_seconds=duration,
        log_path=str(log_path),
        timed_out=False,
    )


def gametest_resource_errors(project_root: str | Path, log_path: str | Path) -> tuple[str, ...]:
    """Return high-signal generated-namespace resource loading errors."""

    root = Path(project_root).expanduser().resolve()
    fabric = root / "src/main/resources/fabric.mod.json"
    try:
        mod_id = str(json.loads(fabric.read_text(encoding="utf-8"))["id"])
    except Exception:
        return ()
    path = Path(log_path)
    if not path.is_file() or path.is_symlink():
        return ()

    namespace = f"{mod_id}:"
    markers = (
        "couldn't parse element",
        "parsing error loading",
        "failed to parse",
        "couldn't load",
        "error loading",
        "unknown item",
        "unknown block",
        "unknown registry",
    )
    findings: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        lowered = raw.lower()
        if namespace not in raw or not any(marker in lowered for marker in markers):
            continue
        compact = raw.strip()
        if compact and compact not in findings:
            findings.append(compact[:2000])
    return tuple(findings)


def _install_incremental_build(runner_module: Any) -> None:
    cls = runner_module.GradleRunner
    original_build = cls.build
    if getattr(original_build, "_mmm_incremental_validation", False):
        return

    def optimized_build_locked(
        self: Any,
        project_root: Path,
        *,
        run_gametest: bool,
    ) -> Any:
        project_root = project_root.resolve()
        if not (project_root / "build.gradle").is_file():
            raise runner_module.BuildRunnerError(
                f"Not a generated Gradle project: {project_root}"
            )
        logs = project_root / ".minecraft_ai" / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        gradle = self._ensure_gradle()
        commands: list[Any] = []
        environment = os.environ.copy()
        environment["GRADLE_USER_HOME"] = str(self.cache_dir / "gradle-user-home")
        environment["CI"] = "true"

        wrapper_result = _sync_verified_wrapper(
            self,
            runner_module=runner_module,
            gradle=gradle,
            project_root=project_root,
            environment=environment,
            logs=logs,
        )
        commands.append(wrapper_result)
        if wrapper_result.exit_code != 0:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=runner_module.GRADLE_VERSION,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=None,
                error="Gradle wrapper generation failed.",
            )

        build_result = self._run(
            name="build",
            executable=gradle,
            arguments=(
                "--no-daemon",
                "build",
                "--build-cache",
                "--stacktrace",
            ),
            cwd=project_root,
            env=environment,
            log_path=logs / "gradle-build.log",
        )
        commands.append(build_result)
        if build_result.exit_code != 0:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=runner_module.GRADLE_VERSION,
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
                cwd=project_root,
                env=environment,
                log_path=logs / "gradle-gametest.log",
            )
            commands.append(gametest_result)
            if gametest_result.exit_code != 0:
                return runner_module.BuildReport(
                    status="FAIL",
                    gradle_version=runner_module.GRADLE_VERSION,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(project_root),
                    gametest_report=self._gametest_report(project_root),
                    error="Headless Fabric GameTest failed.",
                )
            resource_errors = gametest_resource_errors(
                project_root,
                gametest_result.log_path,
            )
            if resource_errors:
                return runner_module.BuildReport(
                    status="FAIL",
                    gradle_version=runner_module.GRADLE_VERSION,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(project_root),
                    gametest_report=self._gametest_report(project_root),
                    error=(
                        "Headless Fabric GameTest loaded generated resources with "
                        "errors: " + " | ".join(resource_errors[:8])
                    ),
                )

        jar_path = self._find_release_jar(project_root)
        if jar_path is None:
            return runner_module.BuildReport(
                status="FAIL",
                gradle_version=runner_module.GRADLE_VERSION,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=self._gametest_report(project_root),
                error="Gradle reported success but no remapped release JAR was found.",
            )
        return runner_module.BuildReport(
            status="PASS",
            gradle_version=runner_module.GRADLE_VERSION,
            commands=tuple(commands),
            jar_path=jar_path,
            gametest_report=self._gametest_report(project_root),
            error=None,
        )

    cls._build_locked = optimized_build_locked

    @wraps(original_build)
    def cached_build(self: Any, project_root: Path, *, run_gametest: bool = True) -> Any:
        root = Path(project_root).expanduser().resolve()
        fingerprint = project_build_fingerprint(root)
        key = (str(root), fingerprint, bool(run_gametest))
        with _CACHE_LOCK:
            cached = _SUCCESSFUL_BUILDS.get(key)
            if cached is not None:
                return copy.deepcopy(cached)

        report = original_build(self, root, run_gametest=run_gametest)
        with _CACHE_LOCK:
            _bounded_put(_RECENT_BUILDS, key, copy.deepcopy(report))
            if report.passed:
                _bounded_put(_SUCCESSFUL_BUILDS, key, copy.deepcopy(report))
        return report

    cached_build._mmm_incremental_validation = True
    cached_build._mmm_exact_input_cache = True
    cls.build = cached_build


def _consume_recent_build(
    project_root: Path,
    *,
    run_gametest: bool,
) -> Any | None:
    fingerprint = project_build_fingerprint(project_root)
    key = (str(project_root.resolve()), fingerprint, bool(run_gametest))
    with _CACHE_LOCK:
        value = _RECENT_BUILDS.pop(key, None)
    return copy.deepcopy(value) if value is not None else None


def _install_jdt_cache(java_lsp_module: Any) -> None:
    cls = java_lsp_module.JavaLanguageService
    original = cls.diagnostics
    if getattr(original, "_mmm_exact_java_cache", False):
        return

    @wraps(original)
    def cached_diagnostics(
        self: Any,
        project_root: str | Path,
        *,
        relative_files: Iterable[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        fingerprint, normalized = _java_fingerprint(root, relative_files)
        key = (str(root), fingerprint, normalized)
        with _CACHE_LOCK:
            cached = _JDT_RESULTS.get(key)
            if cached is not None:
                return copy.deepcopy(cached)
        result = original(
            self,
            root,
            relative_files=(normalized if relative_files is not None else None),
            timeout_seconds=timeout_seconds,
        )
        with _CACHE_LOCK:
            _bounded_put(_JDT_RESULTS, key, copy.deepcopy(result))
        return result

    cached_diagnostics._mmm_exact_java_cache = True
    cls.diagnostics = cached_diagnostics


def _diagnostic_errors(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in diagnostics.get("diagnostics", [])
        if isinstance(item, dict) and int(item.get("severity", 1)) <= 2
    ]


def _install_progressive_repair(repair_module: Any) -> None:
    cls = repair_module.RepairEngine
    original_request_patch = cls._request_patch
    if not getattr(original_request_patch, "_mmm_tracks_repair_scope", False):

        @wraps(original_request_patch)
        def scoped_request_patch(self: Any, evidence: dict[str, Any], context: dict[str, Any]):
            operations = original_request_patch(self, evidence, context)
            self._mmm_last_java_paths = tuple(
                sorted(
                    str(item.get("path", "")).replace("\\", "/")
                    for item in operations
                    if str(item.get("path", "")).lower().endswith(".java")
                )
            )
            return operations

        scoped_request_patch._mmm_tracks_repair_scope = True
        cls._request_patch = scoped_request_patch

    original_evidence = cls._evidence
    if getattr(original_evidence, "_mmm_progressive_evidence", False):
        return

    def progressive_evidence(self: Any, root: Path, *, run_gametest: bool) -> dict[str, Any]:
        relative_files = getattr(self, "_mmm_last_java_paths", ()) or None
        try:
            diagnostics = self.diagnostics_factory().diagnostics(
                root,
                relative_files=relative_files,
                timeout_seconds=90,
            )
        except TypeError:
            # Test doubles and legacy diagnostics factories may not expose the
            # relative_files keyword. Preserve their contract.
            diagnostics = self.diagnostics_factory().diagnostics(
                root,
                timeout_seconds=90,
            )
        except Exception as exc:
            diagnostics = {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "diagnostics": [],
            }

        errors = _diagnostic_errors(diagnostics)
        if errors:
            return {
                "passed": False,
                "diagnostics": diagnostics,
                "build": {
                    "status": "SKIPPED",
                    "error": "Gradle/GameTest deferred until JDT diagnostics are clean.",
                    "commands": [],
                },
            }

        cached = _consume_recent_build(root, run_gametest=run_gametest)
        if cached is not None:
            build = cached.to_dict()
        else:
            try:
                build = self.runner_factory(self.gradle_cache).build(
                    root,
                    run_gametest=run_gametest,
                ).to_dict()
            except Exception as exc:
                build = {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "commands": [],
                }
        return {
            "passed": build.get("status") == "PASS" and not errors,
            "diagnostics": diagnostics,
            "build": build,
        }

    progressive_evidence._mmm_progressive_evidence = True
    cls._evidence = progressive_evidence


def install(
    runner_module: Any,
    java_lsp_module: Any,
    repair_module: Any,
) -> None:
    """Install a fail-fast, exact-input-aware verification execution contract."""

    _install_fast_gradle_distribution(runner_module)
    _install_incremental_build(runner_module)
    _install_jdt_cache(java_lsp_module)
    _install_progressive_repair(repair_module)
