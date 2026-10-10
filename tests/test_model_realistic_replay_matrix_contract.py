"""Keep the model-realistic replay matrix measured in executable pytest cases.

Parametrized scenarios are independent failure/admission behaviors even when they
share one source file. File-count thresholds incorrectly demand unrelated modules
and conceal coverage gaps when a module's test cases are removed.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tools.model_realistic_replay_matrix import ROOT, discover


def _executable_replay_cases(path: str) -> int:
    tree = ast.parse((Path(ROOT) / path).read_text(encoding="utf-8"), filename=path)
    count = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        cases = 1
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
            ):
                continue
            assert len(decorator.args) >= 2, (
                f"Uninspectable replay parametrization in {path}::{node.name}"
            )
            values = ast.literal_eval(decorator.args[1])
            assert isinstance(values, (tuple, list)) and values, (
                f"Unbounded or empty replay scenarios in {path}::{node.name}"
            )
            cases *= len(values)
        count += cases
    return count


def test_every_realistic_replay_behavior_has_an_executable_matrix_case():
    files = discover()
    assert "tests/model_realistic_replay/test_mimo_native_tool_transport.py" in files
    assert len(files) == len(set(files))
    assert all(path.startswith("tests/model_realistic_replay/test_") for path in files)
    assert sum(_executable_replay_cases(path) for path in files) >= 17


def test_replay_tests_are_not_accidentally_invisible_to_recursive_pytest():
    files = discover()
    assert all(path.endswith(".py") for path in files)
