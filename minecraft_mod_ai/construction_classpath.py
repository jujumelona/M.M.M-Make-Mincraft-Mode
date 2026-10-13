"""Resolve the *actual* Fabric compile classpath before model generation.

No guessed APIs, SDK version downgrades, or jar guesses from repository names.
The first Gradle configuration must succeed before expensive coder inference.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from threading import RLock
from pathlib import Path
from tempfile import TemporaryDirectory

_BEGIN = "MMM_COMPILE_CLASSPATH_BEGIN"
_END = "MMM_COMPILE_CLASSPATH_END"
_INIT = r'''
// Resolve the inspected classpath during configuration, NOT inside a task
// execution action. Gradle 9.x configuration cache disallows reading Project
// from doLast, and sourceSets.main is project state.
gradle.projectsEvaluated {
    def root = gradle.rootProject
    def sets = root.extensions.findByName("sourceSets")
    if (sets == null || sets.findByName("main") == null) {
        throw new GradleException("MMM_SOURCESETS_UNAVAILABLE")
    }
    def files = sets.main.compileClasspath.files
    println("MMM_COMPILE_CLASSPATH_BEGIN")
    println(groovy.json.JsonOutput.toJson(files.collect { it.absolutePath }))
    println("MMM_COMPILE_CLASSPATH_END")
}
'''


_INSPECTION_LOCK = RLock()
_INPUTS = ("settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts",
           "gradle.properties", "gradle/libs.versions.toml")


def _config_signature(root: Path) -> tuple[tuple[str, str], ...]:
    # Do not reuse an old classpath when this Fabric project's declared
    # dependencies or Gradle configuration have changed.
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest())
        for name in _INPUTS if (root / name).is_file()
    )


def resolve_fabric_compile_classpath(project_root: str, *, timeout_seconds: int = 180) -> tuple[str, ...]:
    root = Path(project_root).resolve()
    with _INSPECTION_LOCK:
        return _resolve_inspected_classpath(str(root), timeout_seconds, _config_signature(root))


@lru_cache(maxsize=4)
def _resolve_inspected_classpath(project_root: str, timeout_seconds: int,
                                config_signature: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    root = Path(project_root).resolve()
    wrapper = root / "gradlew"
    if not wrapper.is_file():
        raise RuntimeError("CONSTRUCTION_GRADLE_WRAPPER_MISSING: " + str(wrapper))
    with TemporaryDirectory(prefix="mmm-classpath-") as directory:
        init_script = Path(directory) / "mmm-classpath.gradle"
        init_script.write_text(_INIT, encoding="utf-8")
        try:
            result = subprocess.run(
                ["bash", str(wrapper), "--no-daemon", "--no-configuration-cache",
                 "--console=plain", "--init-script", str(init_script), "-q", "help"],
                cwd=root, capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_TIMEOUT") from exc
    if result.returncode:
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_FAILED: " + (result.stderr or result.stdout)[-2000:])
    try:
        raw = result.stdout.split(_BEGIN, 1)[1].split(_END, 1)[0].strip()
        rows = json.loads(raw)
    except (IndexError, ValueError, TypeError) as exc:
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_UNREPORTED") from exc
    if not isinstance(rows, list) or not rows or any(not isinstance(row, str) for row in rows):
        raise RuntimeError("CONSTRUCTION_GRADLE_CLASSPATH_EMPTY")
    # Nonexistent output directories do not expose callable symbols.
    paths = tuple(dict.fromkeys(str(Path(row).resolve()) for row in rows if Path(row).exists()))
    if not paths or not any(Path(row).suffix == ".jar" for row in paths):
        raise RuntimeError("CONSTRUCTION_GRADLE_DEPENDENCIES_NOT_RESOLVED")
    return paths
