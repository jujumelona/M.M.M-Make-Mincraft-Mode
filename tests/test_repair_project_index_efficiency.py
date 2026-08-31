from __future__ import annotations

import importlib
from pathlib import Path

from minecraft_mod_ai.project_index import ProjectIndex as RealProjectIndex
from minecraft_mod_ai.repair_engine import RepairEngine
from minecraft_mod_ai.source_patch import sha256_file

repair_module = importlib.import_module("minecraft_mod_ai.repair_engine")


def _base_repair_function():
    current = RepairEngine.repair
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def _failed(message: str, path: str) -> dict:
    return {
        "passed": False,
        "diagnostics": {
            "status": "AVAILABLE",
            "diagnostics": {
                path: [
                    {
                        "path": path,
                        "message": message,
                        "severity": 1,
                    }
                ]
            },
        },
        "build": {
            "status": "FAIL",
            "error": message,
            "commands": [],
        },
    }


def _passed() -> dict:
    return {
        "passed": True,
        "diagnostics": {"status": "AVAILABLE", "diagnostics": {}},
        "build": {"status": "PASS", "commands": []},
    }


def test_repair_scans_project_once_and_incrementally_updates_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "src/main/java/example"
    source.mkdir(parents=True)
    first = source / "First.java"
    second = source / "Second.java"
    first.write_text("package example; final class First {}\n", encoding="utf-8")
    second.write_text("package example; final class Second {}\n", encoding="utf-8")

    constructions = 0

    class CountingProjectIndex(RealProjectIndex):
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(repair_module, "ProjectIndex", CountingProjectIndex)

    engine = RepairEngine(router=object(), gradle_cache=tmp_path / "gradle-cache")
    evidence = iter(
        [
            _failed("first compile failure", "src/main/java/example/First.java"),
            _failed("second compile failure", "src/main/java/example/Second.java"),
            _passed(),
        ]
    )
    engine._evidence = lambda _root, *, run_gametest: next(evidence)

    patches = iter(
        [
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/example/First.java",
                    "expected_sha256": sha256_file(first),
                    "content": "package example; final class First { int fixed = 1; }\n",
                }
            ],
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/example/Second.java",
                    "expected_sha256": sha256_file(second),
                    "content": "package example; final class Second { int fixed = 2; }\n",
                }
            ],
        ]
    )
    engine._request_patch = lambda _evidence, _context: next(patches)

    contexts: list[dict] = []
    installed_context = engine._context

    def capture_context(root: Path, current_evidence: dict) -> dict:
        value = installed_context(root, current_evidence)
        contexts.append(value)
        return value

    engine._context = capture_context

    result = _base_repair_function()(
        engine,
        tmp_path,
        run_gametest=False,
        max_attempts=2,
    )

    assert result["status"] == "PASS"
    assert result["attempts"] == 2
    assert constructions == 1
    assert len(contexts) == 2
    assert contexts[0]["manifest"]["sha256"] != contexts[1]["manifest"]["sha256"]
    assert "fixed = 1" in first.read_text(encoding="utf-8")
    assert "fixed = 2" in second.read_text(encoding="utf-8")
    assert getattr(RepairEngine._context, "_mmm_reuses_repair_project_index", False)


def test_failed_repair_reuses_last_evidence_without_extra_validation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/main/java/example"
    source.mkdir(parents=True)
    target = source / "Only.java"
    target.write_text("package example; final class Only {}\n", encoding="utf-8")

    engine = RepairEngine(router=object(), gradle_cache=tmp_path / "gradle-cache")
    calls = 0

    def evidence(_root: Path, *, run_gametest: bool) -> dict:
        nonlocal calls
        calls += 1
        return _failed(
            f"compile failure {calls}",
            "src/main/java/example/Only.java",
        )

    engine._evidence = evidence
    engine._request_patch = lambda _evidence, _context: [
        {
            "operation": "replace",
            "path": "src/main/java/example/Only.java",
            "expected_sha256": sha256_file(target),
            "content": "package example; final class Only { int changed = 1; }\n",
        }
    ]

    result = _base_repair_function()(
        engine,
        tmp_path,
        run_gametest=False,
        max_attempts=1,
    )

    assert result["status"] == "FAIL"
    assert calls == 2
    assert result["evidence"]["build"]["error"] == "compile failure 2"


def _run_distinct_empty_repairs(tmp_path: Path, *, max_attempts=None):
    engine = RepairEngine(router=object(), gradle_cache=tmp_path / "gradle-cache")
    counts = {"evidence": 0, "patch": 0}

    def evidence(_root: Path, *, run_gametest: bool) -> dict:
        counts["evidence"] += 1
        return _failed(
            f"distinct compile failure {counts['evidence']}",
            "src/main/java/example/Only.java",
        )

    def request_patch(_evidence: dict, _context: dict):
        counts["patch"] += 1
        return []

    engine._evidence = evidence
    engine._context = lambda _root, _evidence: {}
    engine._request_patch = request_patch
    kwargs = {} if max_attempts is None else {"max_attempts": max_attempts}
    result = _base_repair_function()(
        engine,
        tmp_path,
        run_gametest=False,
        **kwargs,
    )
    return result, counts


def test_default_repair_hard_cap_bounds_distinct_failure_states(tmp_path: Path) -> None:
    result, counts = _run_distinct_empty_repairs(tmp_path)

    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "hard_max_attempts"
    assert counts == {"evidence": 3, "patch": 2}


def test_explicit_large_repair_budget_cannot_raise_host_hard_cap(tmp_path: Path) -> None:
    result, counts = _run_distinct_empty_repairs(tmp_path, max_attempts=8)

    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "hard_max_attempts"
    assert counts == {"evidence": 3, "patch": 2}
