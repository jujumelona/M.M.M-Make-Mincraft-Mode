"""Focused host-compiled feature repair never loops or changes other files."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.feature_compile_repair import compile_and_repair_feature

TARGET = "src/main/java/demo/Alien.java"
BAD = "package demo; public class Alien { public void attack(){ MISSING(); } }"
GOOD = "package demo; public class Alien { public void attack(){ System.out.println(1); } }"


def _write(root, text=BAD):
    path = root / TARGET
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_compiler_driven_repair_stops_at_first_real_pass(tmp_path):
    file = _write(tmp_path)
    edits = []
    runs = []

    def compile_target(_root, *, target_path):
        runs.append(target_path)
        if "MISSING()" in file.read_text(encoding="utf-8"):
            return {"status": "FAIL", "diagnostics": [{
                "path": TARGET, "line": 1,
                "code": "compiler.err.cant.resolve",
                "message": "cannot find symbol MISSING",
            }]}
        return {"status": "PASS", "diagnostics": []}

    def edit(source, diagnostics, attempt):
        edits.append((source, diagnostics, attempt))
        return GOOD

    result = compile_and_repair_feature(tmp_path, target_path=TARGET, edit=edit,
                                         compile_target=compile_target)
    assert result["compile_passed"] is True
    assert result["runtime_verified"] is False
    assert result["repair_attempts"] == 1
    assert len(runs) == 2 and len(edits) == 1
    assert file.read_text(encoding="utf-8") == GOOD


def test_same_failed_diagnostic_never_triggers_endless_inference(tmp_path):
    file = _write(tmp_path)
    calls = []
    def compiler(_root, *, target_path):
        return {"status": "FAIL", "diagnostics": [{
            "path": TARGET, "line": 1, "code": "unresolved",
            "message": "symbol not found",
        }]}
    def edit(source, diagnostics, attempt):
        calls.append(attempt)
        return source.replace("MISSING()", "STILL_BROKEN()")

    outcome = compile_and_repair_feature(tmp_path, target_path=TARGET,
                                          edit=edit, compile_target=compiler)
    assert outcome["status"] == "REPAIR_EXHAUSTED"
    assert calls == [1]
    assert file.read_text(encoding="utf-8") == BAD


def test_compiler_unavailable_must_not_make_model_repair(tmp_path):
    path = _write(tmp_path)
    outcome = compile_and_repair_feature(
        tmp_path, target_path=TARGET,
        edit=lambda *_: pytest.fail("no repair if compiler cannot run"),
        compile_target=lambda *_a, **_k: {"status": "UNAVAILABLE", "diagnostics": []},
    )
    assert outcome["status"] == "COMPILER_UNAVAILABLE"
    assert not outcome["compile_passed"]
    assert path.read_text(encoding="utf-8") == BAD


def test_task_owned_path_validation(tmp_path):
    _write(tmp_path)
    with pytest.raises(ValueError, match="TARGET_INVALID|OUTSIDE_PROJECT"):
        compile_and_repair_feature(
            tmp_path, target_path="../outside.java", edit=lambda *_: GOOD,
            compile_target=lambda *_a, **_k: {"status": "PASS"})
