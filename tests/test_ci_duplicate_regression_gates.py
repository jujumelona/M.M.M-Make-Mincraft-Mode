from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ci_duplicate_pairs import introduced_duplicate_pairs


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality_snapshot(groups: list[list[str]]) -> dict[str, object]:
    return {
        "functions": {},
        "files": {},
        "import_cycles": [],
        "duplicate_function_groups": groups,
        "serial_expensive_loops": [],
        "parse_errors": [],
    }


def test_duplicate_group_shrink_does_not_create_regression() -> None:
    assert introduced_duplicate_pairs(
        [["a.py:f", "b.py:f"]],
        [["a.py:f", "b.py:f", "c.py:f"]],
    ) == []


def test_duplicate_group_extension_reports_only_new_pairs() -> None:
    assert introduced_duplicate_pairs(
        [["a.py:f", "b.py:f", "c.py:f"]],
        [["a.py:f", "b.py:f"]],
    ) == [
        ("a.py:f", "c.py:f"),
        ("b.py:f", "c.py:f"),
    ]


def test_architecture_gate_ignores_group_shrink() -> None:
    audit = _load_script("audit_code_quality_regression")
    violations = audit.compare_snapshots(
        _quality_snapshot([["a.py:f", "b.py:f", "c.py:f"]]),
        _quality_snapshot([["a.py:f", "b.py:f"]]),
    )
    assert not [
        row for row in violations if row.get("category") == "new_duplicate_function_body"
    ]


def test_architecture_gate_reports_only_new_duplicate_relations() -> None:
    audit = _load_script("audit_code_quality_regression")
    violations = audit.compare_snapshots(
        _quality_snapshot([["a.py:f", "b.py:f"]]),
        _quality_snapshot([["a.py:f", "b.py:f", "c.py:f"]]),
    )
    duplicate_violations = [
        row for row in violations if row.get("category") == "new_duplicate_function_body"
    ]
    assert duplicate_violations == [
        {"category": "new_duplicate_function_body", "subject": ["a.py:f", "c.py:f"]},
        {"category": "new_duplicate_function_body", "subject": ["b.py:f", "c.py:f"]},
    ]
