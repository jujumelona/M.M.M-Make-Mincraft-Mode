"""Run actual Gradle gates and retain content-addressed records and raw evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from xml.etree import ElementTree

from .implementation_identity import compute_content_hash, compute_json_schema_hash


GRADLE_CLASSPATH_INIT = r"""
allprojects {
    tasks.withType(org.gradle.api.tasks.compile.JavaCompile).configureEach {
        if (name == 'compileJava') {
            doFirst {
                def paths = classpath.files.collect { it.canonicalPath }.sort()
                rootProject.file('.mmm-compile-classpath.json').text = groovy.json.JsonOutput.toJson(paths)
            }
        }
    }
}
"""


def _report_counts(raw):
    """Count actual testcases, including Minecraft's nested suites without totals."""
    xml = ElementTree.fromstring(raw)
    if xml.tag not in {"testsuite", "testsuites"}:
        raise ValueError("UNKNOWN_TEST_REPORT_FORMAT")
    cases = list(xml.iter("testcase"))
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for singular, plural in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            counts[plural] += int(case.find(singular) is not None)
    # Suite-level failures must not disappear when a producer omits a testcase.
    for key in ("failures", "errors"):
        counts[key] = max(counts[key], sum(int(suite.get(key, "0")) for suite in xml.iter("testsuite")))
    return counts


def _blob(store, data: bytes) -> str:
    digest = compute_content_hash(data)
    path = store.storage_path / "blobs" / digest.split(":")[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("EVIDENCE_BLOB_CORRUPT")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return digest


def _read_blob(store, digest):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("INVALID_EVIDENCE_HASH")
    data = (store.storage_path / "blobs" / digest.split(":")[1]).read_bytes()
    if compute_content_hash(data) != digest:
        raise ValueError("EVIDENCE_BLOB_CORRUPT")
    return data


def run_gradle_evidence(project_root, *, store, expected: dict, gametest_task: str,
                        report_glob: str, timeout=600, classpath=(), candidate=None) -> str:
    """Build an isolated project copy; never accept old build/test reports."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:]*", gametest_task):
        raise ValueError("INVALID_GAMETEST_TASK")
    if Path(report_glob).is_absolute() or ".." in Path(report_glob).parts:
        raise ValueError("INVALID_GAMETEST_REPORT_PATH")
    if not expected or not all(expected.get(k) for k in (
        "leaf_id", "implementation_id", "implementation_sha256", "validator_sha256",
        "input_schema_sha256", "output_schema_sha256", "target_sha256", "authority_sha256")):
        raise ValueError("EVIDENCE_BINDING_INCOMPLETE")
    project = Path(project_root).resolve()
    if candidate is not None:
        target = (project / candidate["target_path"]).resolve()
        if not target.is_relative_to(project) or not target.is_file():
            raise ValueError("CANDIDATE_TARGET_MISSING")
        if compute_content_hash(target.read_bytes()) != candidate["materialized_sha256"]:
            raise ValueError("CANDIDATE_SOURCE_CHANGED")
    files = {}
    excluded = {".git", ".gradle", "build", ".mmm", ".minecraft_ai", ".venv", "node_modules"}
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if any(p in excluded for p in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("EVIDENCE_PROJECT_SYMLINK")
        if path.is_file():
            files[relative.as_posix()] = _blob(store, path.read_bytes())
    classpath_hashes = {str(Path(p).resolve()): _blob(store, Path(p).read_bytes()) for p in classpath}
    gates = {}
    with tempfile.TemporaryDirectory(prefix="mmm-evidence-") as temp:
        root = Path(temp)
        for name, digest in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_read_blob(store, digest))
        wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        if not wrapper.is_file() or not (root / "gradle/wrapper/gradle-wrapper.jar").is_file():
            raise ValueError("GRADLE_WRAPPER_REQUIRED")
        if os.name != "nt":
            wrapper.chmod(wrapper.stat().st_mode | 0o100)
        init_script = root / "integrity-classpath.init.gradle"
        init_script.write_text(GRADLE_CLASSPATH_INIT, encoding="utf-8")
        base = [str(wrapper), "--init-script", str(init_script), "--no-daemon", "--no-build-cache", "--rerun-tasks", "--console=plain"]
        for gate, tasks in (("compile", ["clean", "classes"]), ("gametest", [gametest_task])):
            started = time.monotonic()
            try:
                result = subprocess.run(base + tasks, cwd=root, capture_output=True, timeout=timeout)
                stdout, stderr, code = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr, code = exc.stdout or b"", exc.stderr or b"", -1
            gates[gate] = {"status": "PASS" if code == 0 else "FAIL", "exit_code": code,
                           "command": base[1:] + tasks, "duration_ms": int((time.monotonic()-started)*1000),
                           "stdout": _blob(store, stdout), "stderr": _blob(store, stderr)}
            if code:
                break
        resolved_classpath = {}
        classpath_report = root / ".mmm-compile-classpath.json"
        if gates["compile"]["status"] == "PASS":
            if not classpath_report.is_file():
                raise ValueError("GRADLE_RESOLVED_CLASSPATH_MISSING")
            for raw_path in json.loads(classpath_report.read_text(encoding="utf-8")):
                path = Path(raw_path)
                if not path.is_file():
                    raise ValueError("GRADLE_CLASSPATH_FILE_REQUIRED")
                resolved_classpath[str(path.resolve())] = _blob(store, path.read_bytes())
            if classpath_hashes and set(classpath_hashes.values()) != set(resolved_classpath.values()):
                raise ValueError("GRADLE_ACTUAL_CLASSPATH_MISMATCH")
        artifacts = {p.relative_to(root).as_posix(): _blob(store, p.read_bytes())
                     for p in root.rglob("*.class") if "build" in p.relative_to(root).parts}
        if gates["compile"]["status"] == "PASS" and not artifacts:
            gates["compile"]["status"] = "FAIL"
            gates["compile"]["reason"] = "NO_COMPILED_CLASSES"
        reports = {}
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        for path in root.glob(report_glob):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            reports[path.relative_to(root).as_posix()] = _blob(store, raw)
            found = _report_counts(raw)
            for key in counts:
                counts[key] += found[key]
        if "gametest" in gates:
            gates["gametest"]["counts"] = counts
            if counts["tests"] - counts["skipped"] <= 0 or counts["failures"] or counts["errors"]:
                gates["gametest"]["status"] = "FAIL"
        record = {"format": "mmm.execution-evidence.v1", "expected": expected,
                  "project_files": files, "classpath": classpath_hashes or resolved_classpath,
                  "resolved_classpath": resolved_classpath, "classpath_collector": _blob(store, GRADLE_CLASSPATH_INIT.encode()), "gates": gates,
                  "artifacts": artifacts, "reports": reports, "candidate": candidate}
        data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return _blob(store, data)


def verify_execution_evidence(store, evidence_id, *, expected, required_gates=("compile", "gametest")):
    record = json.loads(_read_blob(store, evidence_id))
    if record.get("format") != "mmm.execution-evidence.v1" or record.get("expected") != expected:
        raise ValueError("EVIDENCE_BINDING_MISMATCH")
    for category in ("project_files", "classpath", "artifacts", "reports"):
        for digest in record[category].values():
            _read_blob(store, digest)
    for name in required_gates:
        gate = record["gates"].get(name)
        if not gate or gate["status"] != "PASS" or gate["exit_code"] != 0:
            raise ValueError(f"EVIDENCE_GATE_NOT_PASSED: {name}")
        _read_blob(store, gate["stdout"])
        _read_blob(store, gate["stderr"])
        if name == "compile" and not record["artifacts"]:
            raise ValueError("EVIDENCE_CLASSES_MISSING")
        if name == "gametest":
            counts = gate["counts"]
            if counts["tests"] - counts["skipped"] <= 0 or counts["errors"] or counts["failures"] or not record["reports"]:
                raise ValueError("EVIDENCE_TESTS_MISSING_OR_FAILED")
            actual = dict.fromkeys(counts, 0)
            for digest in record["reports"].values():
                found = _report_counts(_read_blob(store, digest))
                for key in actual:
                    actual[key] += found[key]
            if counts != actual:
                raise ValueError("EVIDENCE_TEST_COUNTS_MISMATCH")
    if "compile" in required_gates:
        if _read_blob(store, record["classpath_collector"]) != GRADLE_CLASSPATH_INIT.encode():
            raise ValueError("EVIDENCE_CLASSPATH_COLLECTOR_CHANGED")
        if set(record["classpath"].values()) != set(record["resolved_classpath"].values()):
            raise ValueError("EVIDENCE_RESOLVED_CLASSPATH_MISMATCH")
        for digest in record["resolved_classpath"].values():
            _read_blob(store, digest)
        from .jar_api_extractor import parse_class
        for digest in record["artifacts"].values():
            parse_class(_read_blob(store, digest))
    return record


def binding_expectations(leaf, implementation, target):
    return {"leaf_id": leaf, **{key: implementation[key] for key in (
        "implementation_id", "implementation_sha256", "validator_sha256",
        "input_schema_sha256", "output_schema_sha256", "authority_sha256")},
        "target_sha256": compute_json_schema_hash(target)}


def contract_hash(specification):
    """Context ID contains catalog evidence IDs; exclude it to avoid a hash cycle."""
    return compute_json_schema_hash({k: v for k, v in specification.items() if k != "context_id"})
