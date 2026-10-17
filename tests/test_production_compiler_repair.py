"""Full-build model repair is restricted to hash-proven direct Java and real javac."""
from pathlib import Path
import hashlib
import json

from minecraft_mod_ai import production_compiler_repair as pc

TARGET = "src/main/java/demo/Alien.java"
SOURCE = "package demo; public class Alien { public void init(){ BAD(); } }"


def _fixture(tmp_path):
    dest = tmp_path / TARGET
    dest.parent.mkdir(parents=True)
    dest.write_text(SOURCE, encoding="utf-8")
    authored = tmp_path / ".mmm" / "construction_authorship"
    authored.mkdir(parents=True)
    (authored / "receipt.json").write_text(json.dumps({
        "schema_version": "mmm/construction-authorship-v1",
        "status": "DIRECT_SOURCE_RENDERED_RUNTIME_UNVERIFIED",
        "target_path": TARGET,
        "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
    }), encoding="utf-8")
    return dest


def test_real_owned_javac_diagnostic_can_invoke_one_model_repair(tmp_path, monkeypatch):
    _fixture(tmp_path)
    calls = []
    def repair(router, root, *, target_path):
        calls.append(target_path)
        original_sha = hashlib.sha256(
            (Path(root)/target_path).read_bytes()).hexdigest()
        repaired_source = SOURCE.replace("BAD()", "System.out.println(1)")
        (Path(root)/target_path).write_text(repaired_source, encoding="utf-8")
        return {"status": "COMPILED", "compile_passed": True,
                "original_sha256": original_sha,
                "compiled_sha256": hashlib.sha256(
                    repaired_source.encode()).hexdigest()}
    monkeypatch.setattr(
        "minecraft_mod_ai.focused_method_compile_repair.repair_staged_java_with_model", repair)
    report = pc.repair_failed_build_with_model(object(), tmp_path, {
        "status": "FAIL", "diagnostics": [{
            "path": TARGET, "line": 1, "source": "javac", "severity": 1,
            "message": "cannot find symbol BAD",
        }],
    })
    assert report["status"] == "COMPILER_REPAIRED"
    assert report["build_recheck_required"] is True
    assert calls == [TARGET]


def test_gametest_failure_does_not_invoke_model(tmp_path):
    _fixture(tmp_path)
    report = pc.repair_failed_build_with_model(object(), tmp_path, {
        "status": "FAIL",
        "diagnostics": [{"path": TARGET, "line": 1, "source": "runtime",
                         "severity": 1, "message": "assertion failed"}],
    })
    assert report["status"] == "NO_OWNED_JAVAC_ERROR"
    assert not report["build_recheck_required"]


def test_source_hash_mismatch_revokes_authority(tmp_path):
    target = _fixture(tmp_path)
    target.write_text(SOURCE + "// altered", encoding="utf-8")
    report = pc.repair_failed_build_with_model(object(), tmp_path, {
        "status": "FAIL", "diagnostics": [{"path": TARGET, "source": "javac",
                                        "line": 1, "severity": 1}],
    })
    assert report["status"] == "NO_OWNED_JAVAC_ERROR"


def test_no_repair_when_multiple_direct_sources_fail(tmp_path):
    _fixture(tmp_path)
    other = tmp_path / "src/main/java/demo/Other.java"
    other.write_text(SOURCE.replace("Alien", "Other"), encoding="utf-8")
    (tmp_path/".mmm"/"construction_authorship"/"other.json").write_text(json.dumps({
        "schema_version": "mmm/construction-authorship-v1",
        "status": "DIRECT_SOURCE_RENDERED_RUNTIME_UNVERIFIED",
        "target_path": "src/main/java/demo/Other.java",
        "source_sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
    }))
    report = pc.repair_failed_build_with_model(object(), tmp_path, {
        "status": "FAIL", "diagnostics": [
            {"path": path, "source": "javac", "severity": 1}
            for path in (TARGET, "src/main/java/demo/Other.java")
        ],
    })
    assert report["status"] == "MULTIPLE_OWNED_FILES"
