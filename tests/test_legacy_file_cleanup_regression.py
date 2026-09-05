from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_obsolete_and_duplicate_legacy_files_are_absent() -> None:
    """Ensure retired/dead files and duplicate broken scripts stay deleted."""
    retired_paths = (
        ROOT / "tests" / "test_colab_gpu_handoff_contract.py",
        ROOT / "tests" / "test_complete_technology_integration.py",
        ROOT / "tests" / "test_mcp_technology_tools.py",
        ROOT / "tools" / "verify_reference_build.py",
    )
    for path in retired_paths:
        assert not path.exists(), f"Retired legacy file should not exist: {path.name}"


def test_every_test_file_contains_executable_tests() -> None:
    """Ensure no hollowed-out test stubs with 0 tests exist in tests directory."""
    tests_dir = ROOT / "tests"
    empty_test_files: list[str] = []

    for test_path in sorted(tests_dir.glob("test_*.py")):
        content = test_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(test_path))
        test_funcs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        test_classes = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
        ]
        if not test_funcs and not test_classes:
            empty_test_files.append(test_path.name)

    assert (
        empty_test_files == []
    ), f"Found hollowed-out test files with 0 tests: {empty_test_files}"


def test_skillbank_lock_releases_cleanly(tmp_path: Path) -> None:
    """Verify _skillbank_lock properly cleans up lock state without syntax warning or leaked locks."""
    from minecraft_mod_ai.external_procedural_skill_contract import (
        _PATH_LOCKS,
        _PATH_LOCKS_GUARD,
        _skillbank_lock,
    )

    dummy_path = tmp_path / "skillbank_test"
    key = str(dummy_path.expanduser().resolve())

    with _skillbank_lock(dummy_path):
        with _PATH_LOCKS_GUARD:
            assert key in _PATH_LOCKS
            assert _PATH_LOCKS[key][1] == 1

    with _PATH_LOCKS_GUARD:
        assert key not in _PATH_LOCKS
