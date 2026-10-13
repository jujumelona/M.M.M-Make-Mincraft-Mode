"""Resolve the *actual* Fabric compile classpath before model generation.

No guessed APIs, SDK version downgrades, or jar guesses from repository names.
Gradle classpath inspection must succeed before expensive coder inference.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from threading import RLock
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .runner import DEFAULT_GRADLE_COMMAND_TIMEOUT_SECONDS

_BEGIN = "MMM_COMPILE_CLASSPATH_BEGIN"
_END = "MMM_COMPILE_CLASSPATH_END"
_TASK = "mmmInspectCompileClasspath"
_REPORT_PROPERTY = "mmm.compile.classpath.report"
_INIT = r'''
import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.Classpath
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.TaskAction

abstract class MmmInspectCompileClasspath extends DefaultTask {
    @InputFiles
    @Classpath
    abstract ConfigurableFileCollection getInspectedClasspath()

    @OutputFile
    abstract RegularFileProperty getClasspathReport()

    @TaskAction
    void reportClasspath() {
        // Gradle resolves declared task inputs under the owning project's
        // lock. The action consumes files without accessing Project/SourceSet.
        def files = inspectedClasspath.files
        def payload = groovy.json.JsonOutput.toJson(files.collect { it.absolutePath })
        classpathReport.get().asFile.setText(payload, "UTF-8")
        println("MMM_COMPILE_CLASSPATH_BEGIN")
        println(payload)
        println("MMM_COMPILE_CLASSPATH_END")
    }
}

gradle.beforeProject { project ->
    if (project.path != ":") {
        return
    }
    project.pluginManager.withPlugin("java") {
        def sets = project.extensions.getByName("sourceSets")
        project.tasks.register("mmmInspectCompileClasspath", MmmInspectCompileClasspath) {
            // Keep the FileCollection lazy; resolving it in a global lifecycle
            // callback bypasses Gradle's project lock, even with cache disabled.
            inspectedClasspath.from(sets.main.compileClasspath)
            def reportPath = System.getProperty("mmm.compile.classpath.report")
            if (!reportPath) throw new GradleException("MMM compile classpath report path required")
            classpathReport.set(new File(reportPath))
        }
    }
}
'''


_INSPECTION_LOCK = RLock()
_INPUTS = ("settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts",
           "gradle.properties", "gradle/libs.versions.toml",
           "gradle/gradle-wrapper.properties", "gradle/gradle-daemon-jvm.properties")


def _config_signature(root: Path) -> tuple[tuple[str, str], ...]:
    # Do not reuse an old classpath when this Fabric project's declared
    # dependencies or Gradle configuration have changed.
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest())
        for name in _INPUTS if (root / name).is_file()
    )


def resolve_fabric_compile_classpath(project_root: str, *,
                                    timeout_seconds: int = DEFAULT_GRADLE_COMMAND_TIMEOUT_SECONDS,
                                    java_version: int | None = None) -> tuple[str, ...]:
    from .runner import production_gradle_cache_dir

    root = Path(project_root).resolve()
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("CONSTRUCTION_GRADLE_TIMEOUT_REQUIRED")
    java_home = ""
    required_java = None
    if java_version is not None:
        from .java_lsp import _resolve_project_java_home

        required_java = int(java_version)
        if required_java <= 0:
            raise ValueError("CONSTRUCTION_JAVA_VERSION_REQUIRED")
        java_home = str(_resolve_project_java_home(required_java, require_compiler=True).resolve())
    with _INSPECTION_LOCK:
        return _resolve_inspected_classpath(
            str(root), timeout_seconds, _config_signature(root), java_home,
            str(production_gradle_cache_dir()), required_java,
        )


@lru_cache(maxsize=4)
def _resolve_inspected_classpath(project_root: str, timeout_seconds: int,
                                config_signature: tuple[tuple[str, str], ...],
                                java_home: str = "", cache_dir: str = "",
                                java_version: int | None = None) -> tuple[str, ...]:
    from .root_cause_trace import durable_trace_path
    from .runner import GradleRunner, production_gradle_cache_dir

    root = Path(project_root).resolve()
    wrapper = root / "gradlew"
    if not wrapper.is_file():
        raise RuntimeError("CONSTRUCTION_GRADLE_WRAPPER_MISSING: " + str(wrapper))
    cache = Path(cache_dir) if cache_dir else production_gradle_cache_dir()
    runner = GradleRunner(cache, command_timeout_seconds=timeout_seconds)
    log_path = durable_trace_path().parent / "gradle_commands" / f"classpath-{uuid4().hex}.log"
    raw, report_error = None, None
    with TemporaryDirectory(prefix="mmm-classpath-") as directory:
        init_script = Path(directory) / "mmm-classpath.gradle"
        report_path = Path(directory) / "classpath.json"
        init_script.write_text(_INIT, encoding="utf-8")
        # Reuse the SHA-verified distribution, production user home, bootstrap
        # lock, target launcher, and process-tree cleanup used by actual builds.
        # Keep project-owned daemon properties/criteria authoritative.
        result = runner.run_project_command(
            root, name="inspect_compile_classpath",
            arguments=("--no-daemon", "--no-configuration-cache", "--console=plain",
                       "--init-script", str(init_script),
                       f"-D{_REPORT_PROPERTY}={report_path}", "--info", "--stacktrace", ":" + _TASK),
            log_path=log_path, java_version=java_version,
        )
        if not result.timed_out and result.exit_code == 0:
            try:
                # Logs are redacted for humans. The private result channel must
                # preserve exact paths even when a name resembles a credential.
                raw = report_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                report_error = exc
    output = Path(result.log_path).read_text(encoding="utf-8", errors="replace")
    evidence = f"log={result.log_path}\n" + output[-4000:]
    if result.timed_out:
        raise RuntimeError(
            f"CONSTRUCTION_GRADLE_CLASSPATH_TIMEOUT: timeout_seconds={timeout_seconds} " + evidence
        )
    if result.exit_code:
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_FAILED: " + evidence)
    if report_error is not None:
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_UNREPORTED: " + evidence) from report_error
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_UNREPORTED: " + evidence) from exc
    if not isinstance(rows, list) or not rows or any(not isinstance(row, str) for row in rows):
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_EMPTY")
    # Nonexistent output directories do not expose callable symbols.
    paths = tuple(dict.fromkeys(str(Path(row).resolve()) for row in rows if Path(row).exists()))
    if not paths or not any(Path(row).suffix == ".jar" for row in paths):
        raise RuntimeError("CONSTRUCTION_GRADLE_DEPENDENCIES_NOT_RESOLVED")
    return paths
