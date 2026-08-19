from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"


def _module(name: str) -> ast.Module:
    return ast.parse((PACKAGE / name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _direct_call_names(function: ast.FunctionDef) -> list[str]:
    calls: list[str] = []
    for statement in function.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        callee = statement.value.func
        if isinstance(callee, ast.Name):
            calls.append(callee.id)
    return calls


def test_bootstrap_owns_execution_extensions_before_adaptive_hardening() -> None:
    tree = _module("runtime_bootstrap.py")
    function = _function(tree, "_install_post_bootstrap_contracts")
    calls = _direct_call_names(function)

    assert calls.count("install_small_model_execution_extensions") == 1
    assert calls.index("install_small_model_execution_extensions") < calls.index(
        "harden_adaptive_compute"
    )

    imports = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "small_model_execution_extensions_contract",
        "install",
        "install_small_model_execution_extensions",
    ) in imports


def test_adaptive_hardener_does_not_compose_execution_extensions() -> None:
    tree = _module("small_model_adaptive_compute.py")
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "small_model_execution_extensions_contract" not in imported_modules

    function = _function(tree, "harden")
    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "install_execution_extensions" not in called_names
