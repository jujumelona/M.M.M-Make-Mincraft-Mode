from __future__ import annotations
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'minecraft_mod_ai'
_BOOTSTRAP = PACKAGE / 'runtime_bootstrap.py'
_LLAMA_PIPELINE = PACKAGE / 'llama_tuning_pipeline.py'
_APPROVED_COMPOSERS = {_BOOTSTRAP, _LLAMA_PIPELINE}

def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding='utf-8')

def _is_policy_module(name: str) -> bool:
    leaf = name.rsplit('.', 1)[-1]
    return 'contract' in leaf or leaf.endswith('_tuning')

def _policy_imports(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return direct installer aliases and imported contract/tuning-module aliases."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    installers: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and _is_policy_module(node.module):
                for imported in node.names:
                    if imported.name == 'install':
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
                local_name = imported.asname or imported.name.split('.')[-1]
                modules[local_name] = imported.name
    return (installers, modules)

def _composition_calls(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    direct: set[str] = set()
    module_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            direct.add(node.func.id)
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'install' and isinstance(node.func.value, ast.Name):
            module_calls.add(node.func.value.id)
    return (direct, module_calls)

def test_package_init_has_one_bootstrap_and_no_contract_patch_chain() -> None:
    path = PACKAGE / '__init__.py'
    source = path.read_text(encoding='utf-8')
    assert source.count('initialize_runtime()') == 1
    assert '_install_' not in source
    installers, modules = _policy_imports(path)
    direct_calls, module_calls = _composition_calls(path)
    assert not {local_name: module for local_name, module in installers.items() if local_name in direct_calls}
    assert not {local_name: module for local_name, module in modules.items() if local_name in module_calls}
    assert 'integrated_contract_bootstrap' not in source
    assert 'platform_mcp_compatibility_contract' not in source

def test_llama_pipeline_is_the_only_approved_child_composer() -> None:
    assert _LLAMA_PIPELINE.is_file()
    installers, modules = _policy_imports(_LLAMA_PIPELINE)
    direct_calls, module_calls = _composition_calls(_LLAMA_PIPELINE)
    actual = {module for local_name, module in installers.items() if local_name in direct_calls} | {module for local_name, module in modules.items() if local_name in module_calls}
    assert actual == {'llama_server_efficiency_contract', 'llama_server_runtime_tuning', 'llama_cache_reuse_efficiency_contract', 'llama_decode_speed_contract', 'llama_multimodal_contract', 'qwen35_mtp_hotpath_contract', 'qwen35_runtime_efficiency_contract', 'qwen_runtime_transport_contract', 'planner_single_stream_search_contract', 'runtime_stability_contract'}
    source = _LLAMA_PIPELINE.read_text(encoding='utf-8')
    assert 'from .llama_server_hardware_policy import install as install_hardware' in source
    assert 'install_hardware(self.autotune)' in source
    assert 'qwen35_t4_single_stream_tuning' not in source
    assert 'install_qwen35_t4_single_stream' not in source
    order = ('TuningStage("hardware"', 'TuningStage(\n                "efficiency"', 'TuningStage("runtime"', 'TuningStage(\n                "cache-reuse"', 'TuningStage("decode-speed"')
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)
    transport_order = ('TuningStage("kernel-autotune"', 'TuningStage("qwen-transport"', 'TuningStage(\n                "multimodal"')
    transport_positions = [source.index(marker) for marker in transport_order]
    assert transport_positions == sorted(transport_positions)
    decode_order = ('install_decode_speed(', 'install_qwen35_hotpath(', 'install_qwen35_runtime_efficiency(', 'install_single_stream_agentic_policy(')
    decode_positions = [source.index(marker) for marker in decode_order]
    assert decode_positions == sorted(decode_positions)

def test_contract_composition_is_limited_to_explicit_owners() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob('*.py')):
        if path in _APPROVED_COMPOSERS:
            continue
        installers, modules = _policy_imports(path)
        direct_calls, module_calls = _composition_calls(path)
        offenders.extend((f'{path.name}: {module}.install via {local_name}()' for local_name, module in installers.items() if local_name in direct_calls))
        offenders.extend((f'{path.name}: {module}.install via {local_name}.install()' for local_name, module in modules.items() if local_name in module_calls))
    assert offenders == [], 'nested contract composition:\n' + '\n'.join(offenders)
