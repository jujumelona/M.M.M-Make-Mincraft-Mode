"""Python-side I/O checks for Fabric compile classpath inspection.

The probe is read-only: it may resolve actual declared classpath dependencies,
but must not rely on a Gradle execution-time Project object or generated APIs.
Gradle is mocked here, so these checks do not establish Gradle compatibility
or exercise its configuration-resolution locking rules.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pytest

from minecraft_mod_ai import construction_classpath as probe


def _project(tmp_path):
    root = tmp_path / "fabric"
    root.mkdir()
    (root / "gradlew").write_text("#!/bin/bash\n", encoding="utf-8")
    (root / "build.gradle").write_text("plugins { id 'java' }", encoding="utf-8")
    jar = root / "dependencies" / "fabric-real.jar"
    jar.parent.mkdir()
    jar.write_bytes(b"test-placeholder-for-existing-jar")
    return root, jar


def _response(jar):
    return subprocess.CompletedProcess(
        args=[], returncode=0, stderr="",
        stdout="noise\n" + probe._BEGIN + "\n" + json.dumps([str(jar)])
        + "\n" + probe._END + "\n",
    )


def test_inspection_invokes_classpath_task_and_disables_cache(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []

    def fake_run(argv, *, cwd, capture_output, text, timeout, check):
        calls.append(tuple(argv))
        init = Path(argv[argv.index("--init-script") + 1])
        assert "--no-configuration-cache" in argv
        assert argv[-2:] == ["-q", ":mmmInspectCompileClasspath"]
        assert init.is_file()
        assert str(root) == str(cwd)
        return _response(jar)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    assert probe.resolve_fabric_compile_classpath(str(root)) == (str(jar.resolve()),)
    assert len(calls) == 1


def test_inspection_invocation_single_flight_across_parallel_nodes(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []
    lock = Lock()

    def fake_run(argv, **kwargs):
        with lock:
            calls.append(argv)
        return _response(jar)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(probe.resolve_fabric_compile_classpath, str(root))
                   for _ in range(4)]
        assert all(f.result() == (str(jar.resolve()),) for f in futures)
    assert len(calls) == 1


def test_build_script_change_invalidates_inspected_classpath(tmp_path, monkeypatch):
    root, jar = _project(tmp_path)
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(True)
        return _response(jar)
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    probe.resolve_fabric_compile_classpath(str(root))
    (root / "build.gradle").write_text("plugins { id 'java-library' }", encoding="utf-8")
    probe.resolve_fabric_compile_classpath(str(root))
    assert len(calls) == 2


def test_gradle_failure_remains_fail_closed(tmp_path, monkeypatch):
    root, _ = _project(tmp_path)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 1, "", "CONFIGURATION_ERROR"))
    with pytest.raises(RuntimeError, match="CONSTRUCTION_GRADLE_CLASSPATH_FAILED"):
        probe.resolve_fabric_compile_classpath(str(root))
