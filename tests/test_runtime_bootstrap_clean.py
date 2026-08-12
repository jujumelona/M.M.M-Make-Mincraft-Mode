from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
_BOOTSTRAP = PACKAGE / "runtime_bootstrap.py"


def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _is_policy_module(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return "contract" in leaf or leaf.endswith("_tuning")


def _policy_imports(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return direct installer aliases and imported policy-module aliases."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    installers: dict[str, str] = {}
    modules: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and _is_policy_module(node.module):
                for imported in node.names:
                    if imported.name == "install":
                        installers[imported.asname or imported.name] = node.module
            if node.level and node.module is None:
                for imported in node.names:
                    if _is_policy_module(imported.name):
                        modules[imported.asname or imported.name] = imported.name
            continue

        if isinstance(node, ast.Import):
            for imported in node.names:
                if not _is_policy_module(imported.name):
                    continue
                local_name = imported.asname or imported.name.split(".")[-1]
                modules[local_name] = imported.name

    return installers, modules


def _composition_calls(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    direct: set[str] = set()
    module_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            direct.add(node.func.id)
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "install"
            and isinstance(node.func.value, ast.Name)
        ):
            module_calls.add(node.func.value.id)
    return direct, module_calls


def test_package_init_has_one_bootstrap_and_no_contract_patch_chain() -> None:
    path = PACKAGE / "__init__.py"
    source = path.read_text(encoding="utf-8")
    assert source.count("initialize_runtime()") == 1
    assert "_install_" not in source

    # Public API modules may legitimately end in ``_contract`` (for example
    # production_contract). Reject only imports that are actually invoked as policy
    # installers from package __init__, rather than matching a filename substring.
    installers, modules = _policy_imports(path)
    direct_calls, module_calls = _composition_calls(path)
    assert not {
        local_name: module
        for local_name, module in installers.items()
        if local_name in direct_calls
    }
    assert not {
        local_name: module
        for local_name, module in modules.items()
        if local_name in module_calls
    }
    assert "integrated_contract_bootstrap" not in source
    assert "platform_mcp_compatibility_contract" not in source


def test_runtime_bootstrap_is_flat_not_nested() -> None:
    source = _text("runtime_bootstrap.py")
    assert "integrated_contract_bootstrap" not in source
    assert "final_architecture_contract" not in source
    assert "platform_mcp_compatibility_contract" not in source

    # Flatness is not enough: every policy that used to be hidden behind a child
    # installer must still exist exactly once at the explicit runtime root.
    required_once = (
        "install_runner_lock(",
        "install_scheduler_parallel_safety(",
        "install_llama_parallel_runtime(",
        "install_llama_efficiency(",
        "install_llama_runtime_tuning(",
        "install_llama_cache_reuse(",
        "install_llama_stream_efficiency(",
        "install_project_index_execution_reuse(",
        "install_proposal_deserialization(",
        "install_external_mcp_target_validation(",
        "install_platform_live_rag(",
        "install_platform_technology(",
        "install_platform_ecosystem(",
        "install_platform_prompts(",
        "install_planner_json_runtime(",
        "install_planner_strict_json(",
        "install_planner_outline_prompt(",
        "install_incremental_repair(",
        "install_checkpoint_journal(",
        "install_agentic_search_efficiency(",
        "install_asset_resume_efficiency(",
        "install_audio_resume_efficiency(",
        "install_scheduler_poll_efficiency(",
        "install_production_stream_efficiency(",
        "install_production_stream_resume(",
        "install_execution_efficiency(",
        "install_incremental_resume(",
        "install_planner_pagination_safety(",
        "install_planner_production_page(",
        "install_runtime_helpers(",
        "install_runtime_helper_json_deadline(",
        "install_mcp_runtime(",
        "install_mcp_federation(",
        "install_mcp_repair_batch(",
        "install_mcp_repair_diagnostic_shape(",
        "install_skill_policy(",
    )
    for call in required_once:
        assert source.count(call) == 1, call

    assert source.index("install_planner_pagination_safety(") < source.index(
        "install_planner_production_page("
    )
    assert source.index("install_external_mcp_target_validation(") < source.index(
        "install_external_mcp_bridge_safety("
    )


def test_contract_composition_exists_only_in_runtime_bootstrap() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path == _BOOTSTRAP:
            continue
        installers, modules = _policy_imports(path)
        direct_calls, module_calls = _composition_calls(path)
        offenders.extend(
            f"{path.name}: {module}.install via {local_name}()"
            for local_name, module in installers.items()
            if local_name in direct_calls
        )
        offenders.extend(
            f"{path.name}: {module}.install via {local_name}.install()"
            for local_name, module in modules.items()
            if local_name in module_calls
        )
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

    strict_json = _text("planner_strict_json_contract.py")
    assert "planner_outline_prompt_contract" not in strict_json
    assert "planner_incremental_repair_contract" not in strict_json
    assert "planner_incremental_resume_contract" not in strict_json

    outline_prompt = _text("planner_outline_prompt_contract.py")
    assert "planner_production_page_contract" not in outline_prompt

    incremental_resume = _text("planner_incremental_resume_contract.py")
    assert "install_" not in incremental_resume
    assert "production_stream_efficiency_contract" not in incremental_resume
    assert "scheduler_poll_efficiency_contract" not in incremental_resume

    cache_reuse = _text("llama_cache_reuse_efficiency_contract.py")
    assert "max_performance_module" not in cache_reuse
    assert "runtime_tuning_module" in cache_reuse

    assert not (PACKAGE / "integrated_contract_bootstrap.py").exists()
    assert not (PACKAGE / "platform_mcp_compatibility_contract.py").exists()
    assert not (PACKAGE / "final_architecture_contract.py").exists()
    assert not (PACKAGE / "llama_server_max_performance.py").exists()
