from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

from minecraft_mod_ai import repair_engine
from minecraft_mod_ai.repair_engine import RepairEngine


def _bind_test_platform(root: Path) -> None:
    (root / "gradle.properties").write_text(
        "minecraft_version=1.21.1\nloader=fabric\n",
        encoding="utf-8",
    )


def test_unbounded_repair_uses_semantic_convergence_not_two_attempt_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/demo/Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; final class Example { int value = 0; }\n", encoding="utf-8")
    _bind_test_platform(root)

    class Index:
        def __init__(self, _root, policy=None):
            self.files = ()

        def update_files(self, _paths):
            return None

        def write_manifest(self):
            return None

        def manifest_receipt(self):
            return {"sha256": "sha256:test"}

    class Patcher:
        def __init__(self, _root):
            pass

        def apply(self, operations):
            return {
                "schema_version": "mmm/source-patch-receipt-v1",
                "status": "APPLIED",
                "operations": [dict(item) for item in operations],
            }

    monkeypatch.setattr(repair_engine, "ProjectIndex", Index)
    monkeypatch.setattr(repair_engine, "TransactionalSourcePatcher", Patcher)

    engine = RepairEngine(
        router=SimpleNamespace(),
        gradle_cache=tmp_path / "cache",
    )
    evidence_calls = {"count": 0}

    def evidence(self, _root, *, run_gametest):
        evidence_calls["count"] += 1
        count = evidence_calls["count"]
        if count == 4:
            return {
                "passed": True,
                "diagnostics": {"diagnostics": {}},
                "build": {"status": "PASS", "commands": []},
            }
        return {
            "passed": False,
            "diagnostics": {"diagnostics": {}},
            "build": {
                "status": "FAIL",
                "error": f"compile-state-{count}",
                "commands": [],
            },
        }

    def context(self, _root, _evidence):
        return {}

    patch_calls = {"count": 0}

    def request_patch(self, _evidence, _context):
        patch_calls["count"] += 1
        return [
            {
                "operation": "replace",
                "path": "src/main/java/demo/Example.java",
                "content": (
                    "package demo; final class Example "
                    f"{{ int value = {patch_calls['count']}; }}\n"
                ),
            }
        ]

    engine._evidence = MethodType(evidence, engine)
    engine._context = MethodType(context, engine)
    engine._request_patch = MethodType(request_patch, engine)

    result = engine.repair(root, run_gametest=False, max_attempts=None)

    assert result["status"] == "PASS"
    assert result["attempts"] == 3
    assert patch_calls["count"] == 3


def test_repeated_verifier_signature_still_terminates_unbounded_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _bind_test_platform(root)

    class Index:
        def __init__(self, _root, policy=None):
            pass

    monkeypatch.setattr(repair_engine, "ProjectIndex", Index)
    engine = RepairEngine(router=SimpleNamespace(), gradle_cache=tmp_path / "cache")

    def evidence(self, _root, *, run_gametest):
        return {
            "passed": False,
            "diagnostics": {"diagnostics": {}},
            "build": {
                "status": "FAIL",
                "error": "same compile state",
                "commands": [],
            },
        }

    engine._evidence = MethodType(evidence, engine)
    engine._context = MethodType(lambda self, root, evidence: {}, engine)
    engine._request_patch = MethodType(lambda self, evidence, context: [], engine)

    result = engine.repair(root, run_gametest=False, max_attempts=None)

    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "repeated_signature"
    assert result["attempts"] == 0
