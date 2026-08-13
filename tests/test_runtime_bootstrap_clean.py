from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
_BOOTSTRAP = PACKAGE / "runtime_bootstrap.py"
_LLAMA_PIPELINE = PACKAGE / "llama_tuning_pipeline.py"
_APPROVED_COMPOSERS = {_BOOTSTRAP, _LLAMA_PIPELINE}


def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _is_policy_module(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return "contract" in leaf or leaf.endswith("_tuning")


def _policy_imports(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return direct installer aliases and imported contract/tuning-module aliases."""
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


def test_runtime_bootstrap_is_flat_with_one_owned_llama_pipeline() -> None:
    source = _text("runtime_bootstrap.py")
    assert "integrated_contract_bootstrap" not in source
    assert "final_architecture_contract" not in source
    assert "platform_mcp_compatibility_contract" not in source
    assert "agent_tool_calling_contract" not in source
    assert "platform_policy_runtime_contract" not in source

    required_once = (
        "install_runner_lock(",
        "install_gpu_handoff(",
        "install_scheduler_parallel_safety(",
        "install_llama_parallel_runtime(",
        "install_native_llama_tuning_pipeline(",
        "install_llama_stream_efficiency(",
        "install_project_index_execution_reuse(",
        "install_proposal_deserialization(",
        "install_platform_live_rag(",
        "install_platform_technology(",
        "install_platform_ecosystem(",
        "install_platform_prompts(",
        "install_mod_scope(",
        "install_parallel_platform_rag(",
        "install_colab_auto_platform(",
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
        "install_execution_efficiency(",
        "install_incremental_resume(",
        "install_planner_pagination_safety(",
        "install_planner_production_page(",
        "install_platform_mcp(",
        "install_platform_release(",
    )
    for call in required_once:
        assert source.count(call) == 1, call

    for obsolete_direct_call in (
        "install_llama_efficiency(",
        "install_llama_runtime_tuning(",
        "install_llama_cache_reuse(",
        "install_llama_decode_speed(",
    ):
        assert obsolete_direct_call not in source

    assert source.index("install_planner_pagination_safety(") < source.index(
        "install_planner_production_page("
    )


def test_llama_pipeline_is_the_only_approved_child_composer() -> None:
    assert _LLAMA_PIPELINE.is_file()
    installers, modules = _policy_imports(_LLAMA_PIPELINE)
    direct_calls, module_calls = _composition_calls(_LLAMA_PIPELINE)
    # This generic detector intentionally indexes modules named *contract or *_tuning.
    # llama_server_hardware_policy is checked explicitly below because its filename is
    # outside that naming convention.
    actual = {
        module
        for local_name, module in installers.items()
        if local_name in direct_calls
    } | {
        module
        for local_name, module in modules.items()
        if local_name in module_calls
    }
    assert actual == {
        "llama_server_efficiency_contract",
        "llama_server_runtime_tuning",
        "llama_cache_reuse_efficiency_contract",
        "llama_decode_speed_contract",
        "qwen35_mtp_hotpath_contract",
        "planner_single_stream_search_contract",
    }

    source = _LLAMA_PIPELINE.read_text(encoding="utf-8")
    assert (
        "from .llama_server_hardware_policy import install as install_hardware"
        in source
    )
    assert "install_hardware(self.autotune)" in source
    assert "qwen35_t4_single_stream_tuning" not in source
    assert "install_qwen35_t4_single_stream" not in source
    order = (
        "TuningStage(\"hardware\"",
        "TuningStage(\n                \"efficiency\"",
        "TuningStage(\"runtime\"",
        "TuningStage(\n                \"cache-reuse\"",
        "TuningStage(\"decode-speed\"",
    )
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert (
        source.index("install_decode_speed(")
        < source.index("install_qwen35_hotpath(")
        < source.index("install_single_stream_agentic_policy(")
    )


def test_contract_composition_is_limited_to_explicit_owners() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path in _APPROVED_COMPOSERS:
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
    assert "production_stream_efficiency_contract" not in incremental_resume
    assert "scheduler_poll_efficiency_contract" not in incremental_resume

    cache_reuse = _text("llama_cache_reuse_efficiency_contract.py")
    assert "max_performance_module" not in cache_reuse
    assert "runtime_tuning_module" in cache_reuse

    assert not (PACKAGE / "integrated_contract_bootstrap.py").exists()
    assert not (PACKAGE / "platform_mcp_compatibility_contract.py").exists()
    assert not (PACKAGE / "final_architecture_contract.py").exists()
    assert not (PACKAGE / "llama_server_max_performance.py").exists()
    assert not (PACKAGE / "production_stream_resume_contract.py").exists()
