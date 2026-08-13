from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "minecraft_mod_ai"
_BOOTSTRAP = _PACKAGE / "runtime_bootstrap.py"
_LLAMA_PIPELINE = _PACKAGE / "llama_tuning_pipeline.py"


def _policy_imports(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    installers: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module.rsplit(".", 1)[-1]
        if not (module.endswith("_contract") or module.endswith("_tuning")):
            continue
        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name in {"install", "bind_structured_decode_policy"}:
                installers[local] = module
            else:
                modules[local] = module
    return installers, modules


def _composition_calls(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    direct: set[str] = set()
    module_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            direct.add(func.id)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "install"
            and isinstance(func.value, ast.Name)
        ):
            module_calls.add(func.value.id)
    return direct, module_calls


def _installed_policy_modules(path: Path) -> set[str]:
    installers, modules = _policy_imports(path)
    direct_calls, module_calls = _composition_calls(path)
    return {
        module
        for local_name, module in installers.items()
        if local_name in direct_calls
    } | {
        module
        for local_name, module in modules.items()
        if local_name in module_calls
    }


def test_runtime_bootstrap_is_single_top_level_policy_owner() -> None:
    assert _BOOTSTRAP.is_file()
    source = _BOOTSTRAP.read_text(encoding="utf-8")

    required_once = (
        "install_native_llama_tuning_pipeline(",
        "install_llama_parallel_runtime(",
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
    order = (
        "TuningStage(\"hardware\"",
        "TuningStage(\n                \"efficiency\"",
        "TuningStage(\"runtime\"",
        "TuningStage(\n                \"cache-reuse\"",
        "TuningStage(\n                \"decode-speed\"",
        "TuningStage(\n                \"qwen35-mtp-hotpath\"",
        "TuningStage(\n                \"single-stream-plan-search\"",
    )
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_contract_composition_is_limited_to_explicit_owners() -> None:
    offenders: list[str] = []
    allowed = {_BOOTSTRAP.resolve(), _LLAMA_PIPELINE.resolve()}
    for path in sorted(_PACKAGE.glob("*.py")):
        if path.resolve() in allowed:
            continue
        installed = _installed_policy_modules(path)
        if installed:
            offenders.append(f"{path.name}: {sorted(installed)}")
    assert not offenders, "unexpected nested runtime composition: " + "; ".join(offenders)


def test_runtime_bootstrap_does_not_reenter_package_initialization() -> None:
    tree = ast.parse(_BOOTSTRAP.read_text(encoding="utf-8"), filename=str(_BOOTSTRAP))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "initialize_runtime" not in calls - {"initialize_runtime"}


def test_runtime_bootstrap_has_no_dynamic_importlib_composition() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf-8")
    assert "importlib.import_module" not in source
    assert "__import__(" not in source
