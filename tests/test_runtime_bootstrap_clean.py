from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
_BOOTSTRAP = PACKAGE / "runtime_bootstrap.py"


def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _composition_imports(path: Path) -> dict[str, str]:
    """Return local names that import installer functions from policy modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        if not ("contract" in module or module.endswith("_tuning")):
            continue
        for imported in node.names:
            if imported.name != "install":
                continue
            aliases[imported.asname or imported.name] = module
    return aliases


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_package_init_has_one_bootstrap_and_no_contract_patch_chain() -> None:
    source = _text("__init__.py")
    assert source.count("initialize_runtime()") == 1
    assert "_install_" not in source
    assert "_contract import" not in source
    assert "integrated_contract_bootstrap" not in source
    assert "platform_mcp_compatibility_contract" not in source


def test_runtime_bootstrap_is_flat_not_nested() -> None:
    source = _text("runtime_bootstrap.py")
    assert "integrated_contract_bootstrap" not in source
    assert "final_architecture_contract" not in source
    assert "platform_mcp_compatibility_contract" not in source
    assert source.count("install_scheduler_parallel_safety(") == 1
    assert source.count("install_llama_parallel_runtime(") == 1
    assert source.count("install_llama_efficiency(") == 1
    assert source.count("install_llama_runtime_tuning(") == 1
    assert source.count("install_llama_cache_reuse(") == 1
    assert source.count("install_llama_stream_efficiency(") == 1
    assert source.count("install_proposal_deserialization(") == 1


def test_contract_composition_exists_only_in_runtime_bootstrap() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path == _BOOTSTRAP:
            continue
        imported_installers = _composition_imports(path)
        if not imported_installers:
            continue
        called = _called_names(path)
        for local_name, module in imported_installers.items():
            if local_name in called:
                offenders.append(f"{path.name}: {module}.install via {local_name}()")
    assert offenders == [], "nested contract composition:\n" + "\n".join(offenders)


def test_specialized_installers_are_single_responsibility() -> None:
    tuning = _text("performance_final_tuning.py")
    assert "llama_server_efficiency_contract" not in tuning
    assert "project_manifest_hash_efficiency_contract" not in tuning
    assert "final_architecture_contract" not in tuning

    llama_efficiency = _text("llama_server_efficiency_contract.py")
    assert "importlib" not in llama_efficiency
    assert "llama_server_runtime_tuning" not in llama_efficiency
    assert "llama_cache_reuse_efficiency_contract" not in llama_efficiency
    assert "llama_stream_efficiency_contract" not in llama_efficiency
    assert "llama_parallel_runtime_contract" not in llama_efficiency

    cache_reuse = _text("llama_cache_reuse_efficiency_contract.py")
    assert "max_performance_module" not in cache_reuse
    assert "runtime_tuning_module" in cache_reuse

    assert not (PACKAGE / "integrated_contract_bootstrap.py").exists()
    assert not (PACKAGE / "platform_mcp_compatibility_contract.py").exists()
    assert not (PACKAGE / "llama_server_max_performance.py").exists()
