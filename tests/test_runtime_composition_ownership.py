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
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        calls.append((node.lineno, node.col_offset, node.func.id))
    calls.sort()
    return [name for _, _, name in calls]


def test_pre_design_pipeline_owns_procedural_composition_explicitly() -> None:
    pipeline = _module("pre_design_research_pipeline.py")
    collector = _function(pipeline, "collect_design_research")
    calls = _direct_call_names(collector)

    assert calls.count("attach_procedural_skillbank") == 1
    assert calls.count("compose_research_skillbank") == 1
    assert calls.index("attach_procedural_skillbank") < calls.index(
        "compose_research_skillbank"
    )

    bootstrap = _module("runtime_bootstrap.py")
    post = _function(bootstrap, "_install_post_bootstrap_contracts")
    post_calls = _direct_call_names(post)
    assert "install_small_model_execution_extensions" not in post_calls
    assert "install_external_procedural_skill" not in post_calls


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


def test_finalization_rechecks_wrapper_integrity_after_late_installers() -> None:
    tree = _module("runtime_finalization.py")
    function = _function(tree, "finalize_runtime")
    calls = _direct_call_names(function)

    assert calls.count("verify_installed_wrappers") == 1
    assert calls.index("install_llama_server_response_resilience") < calls.index(
        "verify_installed_wrappers"
    )
    assert calls.index("verify_installed_wrappers") < calls.index(
        "run_runtime_live_path_preflight"
    )

    imports = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "runtime_wrapper_integrity",
        "verify_installed_wrappers",
        None,
    ) in imports
