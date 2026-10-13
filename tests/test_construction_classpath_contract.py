"""Python-side I/O checks for Fabric compile classpath inspection.

The probe is read-only: it may resolve actual declared classpath dependencies,
but must not rely on a Gradle execution-time Project object or generated APIs.
Gradle is mocked here, so these checks do not establish Gradle compatibility
or exercise its configuration-resolution locking rules.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pytest

from minecraft_mod_ai import construction_classpath as probe
from minecraft_mod_ai import root_cause_trace
from minecraft_mod_ai import runner as command_authority


@pytest.fixture(autouse=True)
def _isolated_probe_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(root_cause_trace, "durable_trace_path", lambda: tmp_path / "traces/root_cause.jsonl")
    monkeypatch.setattr(command_authority, "production_gradle_cache_dir", lambda: tmp_path / "production-gradle")


def _project(tmp_path):
    root = tmp_path / "fabric"
    root.mkdir()
    (root / "gradlew").write_text("#!/bin/bash\n", encoding="utf-8")
    (root / "build.gradle").write_text("plugins { id 'java' }", encoding="utf-8")
    jar = root / "dependencies" / "fabric-real.jar"
    jar.parent.mkdir()
    jar.write_bytes(b"test-placeholder-for-existing-jar")
    return root, jar


def _response(jar, *, arguments, log_path, exit_code=0, output=None):
    if exit_code == 0:
        prefix = "-D" + probe._REPORT_PROPERTY + "="
        report_path = Path(next(value[len(prefix):] for value in arguments if value.startswith(prefix)))
        report_path.write_text(json.dumps([str(jar)]), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        output if output is not None else "noise\n" + probe._BEGIN + "\n" + json.dumps([str(jar)])
        + "\n" + probe._END + "\n", encoding="utf-8",
    )
    return command_authority.CommandResult(
        name="inspect_compile_classpath", command=arguments, exit_code=exit_code,
        duration_seconds=0, log_path=str(log_path),
    )


def test_inspection_invokes_classpath_task_and_disables_cache(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []

    def fake_run(self, project_root, *, name, arguments, log_path, java_version):
        calls.append(arguments)
        init = Path(arguments[arguments.index("--init-script") + 1])
        assert "--no-configuration-cache" in arguments
        assert arguments[-1] == ":mmmInspectCompileClasspath"
        assert "--info" in arguments
        assert name == "inspect_compile_classpath"
        assert self.command_timeout_seconds == command_authority.DEFAULT_GRADLE_COMMAND_TIMEOUT_SECONDS
        assert init.is_file()
        assert root == project_root
        return _response(jar, arguments=arguments, log_path=log_path)

    monkeypatch.setattr(command_authority.GradleRunner, "run_project_command", fake_run)
    assert probe.resolve_fabric_compile_classpath(str(root)) == (str(jar.resolve()),)
    assert len(calls) == 1


def test_inspection_invocation_single_flight_across_parallel_nodes(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []
    lock = Lock()

    def fake_run(self, project_root, *, name, arguments, log_path, java_version):
        with lock:
            calls.append(arguments)
        return _response(jar, arguments=arguments, log_path=log_path)

    monkeypatch.setattr(command_authority.GradleRunner, "run_project_command", fake_run)
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(probe.resolve_fabric_compile_classpath, str(root))
                   for _ in range(4)]
        assert all(f.result() == (str(jar.resolve()),) for f in futures)
    assert len(calls) == 1


def test_build_script_change_invalidates_inspected_classpath(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []
    def fake_run(self, project_root, *, name, arguments, log_path, java_version):
        calls.append(True)
        return _response(jar, arguments=arguments, log_path=log_path)
    monkeypatch.setattr(command_authority.GradleRunner, "run_project_command", fake_run)
    probe.resolve_fabric_compile_classpath(str(root))
    (root / "build.gradle").write_text("plugins { id 'java-library' }", encoding="utf-8")
    probe.resolve_fabric_compile_classpath(str(root))
    assert len(calls) == 2


def test_gradle_failure_remains_fail_closed(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)

    def fake_run(self, project_root, *, name, arguments, log_path, java_version):
        return _response(jar, arguments=arguments, log_path=log_path,
                         exit_code=1, output="CONFIGURATION_ERROR")

    monkeypatch.setattr(command_authority.GradleRunner, "run_project_command", fake_run)
    with pytest.raises(RuntimeError, match="CONSTRUCTION_GRADLE_CLASSPATH_FAILED"):
        probe.resolve_fabric_compile_classpath(str(root))


def test_jdk_runtime_image_fingerprint_invalidates_compiler_symbol_cache(monkeypatch):
    from minecraft_mod_ai import construction_runtime

    calls = []

    def fake_build(paths, *, java_version, namespace, classpath_sides):
        calls.append((paths, java_version, namespace, classpath_sides))
        return {"call": len(calls)}

    monkeypatch.setattr(construction_runtime, "build_target_symbols", fake_build)
    construction_runtime._symbol_inventory.cache_clear()
    try:
        base = ("jdk/lib/modules", "target.jar")
        first = construction_runtime._symbol_inventory(base, 25, "target", (100, 10))
        same = construction_runtime._symbol_inventory(base, 25, "target", (100, 10))
        updated = construction_runtime._symbol_inventory(base, 25, "target", (200, 11))
        assert first == same
        assert updated != first
        assert len(calls) == 2
    finally:
        construction_runtime._symbol_inventory.cache_clear()
