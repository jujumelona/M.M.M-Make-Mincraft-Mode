from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'minecraft_mod_ai'
_BOOTSTRAP = PACKAGE / 'runtime_bootstrap.py'
_LLAMA_PIPELINE = PACKAGE / 'llama_tuning_pipeline.py'
_FINALIZATION = PACKAGE / 'runtime_finalization.py'
_APPROVED_COMPOSERS = {_BOOTSTRAP, _LLAMA_PIPELINE, _FINALIZATION}

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
        # A tuning installer may be passed as the callable owned by TuningStage
        # instead of being invoked through a lambda. That is still composition and
        # must be visible to this static ownership audit.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == 'TuningStage'
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            direct.add(node.args[1].id)
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
    assert 'finalize_runtime' not in source
    assert 'validate_catalog' not in source
    assert 'runtime_finalization' not in source
    assert '_install_' not in source
    installers, modules = _policy_imports(path)
    direct_calls, module_calls = _composition_calls(path)
    assert not {local_name: module for local_name, module in installers.items() if local_name in direct_calls}
    assert not {local_name: module for local_name, module in modules.items() if local_name in module_calls}
    assert 'integrated_contract_bootstrap' not in source
    assert 'platform_mcp_compatibility_contract' not in source
    for legacy in (
        'java_toolchain_separation_installation',
        'source_set_boundary_installation',
        'versioned_reference_context_installation',
        'prepared_project_resume_integrity',
        'execution_feedback_semantic_convergence_installation',
    ):
        assert legacy not in source


def test_runtime_bootstrap_owns_full_initialization_order() -> None:
    source = _BOOTSTRAP.read_text(encoding='utf-8')
    start = source.index('def initialize_runtime() -> None:')
    end = source.index('def runtime_initialized() -> bool:')
    initialize = source[start:end]
    positions = [
        initialize.index('_install_runtime_contracts()'),
        initialize.index('_validate_runtime_template_authority()'),
        initialize.index('finalize_runtime()'),
        initialize.index('_INITIALIZED = True'),
    ]
    assert positions == sorted(positions)


def test_package_has_no_legacy_installation_modules() -> None:
    assert list(PACKAGE.glob("*_installation.py")) == []


def test_removed_runtime_patch_modules_do_not_return() -> None:
    for name in (
        "mutation_authority_final_guard.py",
        "planir_mutation_authority_contract.py",
        "repair_mutation_recovery_contract.py",
        "execution_feedback_semantic_convergence_installation.py",
        "coder_mutation_authority_contract.py",
        "execution_feedback_owner_precision_contract.py",
        "generation_verifier_runtime_contract.py",
        "model_tool_alias_permission_policy.py",
    ):
        assert not (PACKAGE / name).exists()


def test_managed_llama_reuse_is_owned_by_model_runtime_stage() -> None:
    init_source = _text('__init__.py')
    assert 'managed_llama_reuse_contract' not in init_source

    source = _BOOTSTRAP.read_text(encoding='utf-8')
    model_start = source.index('def _install_model_runtime_contracts() -> None:')
    validation_start = source.index('def _install_validation_contracts() -> None:')
    model_runtime = source[model_start:validation_start]
    assert 'from .managed_llama_reuse_contract import install as install_managed_llama_reuse' in model_runtime
    assert 'install_managed_llama_reuse()' in model_runtime

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
    transport_order = ('TuningStage("kernel-autotune"', 'TuningStage("qwen-transport"', 'TuningStage("multimodal"')
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


def test_execution_feedback_is_not_late_runtime_patched() -> None:
    feedback = _text("execution_feedback_replan_contract.py")
    finalization = _FINALIZATION.read_text(encoding="utf-8")

    assert "def install(" not in feedback
    assert "_install_ledger_feedback" not in feedback
    assert "_install_observation_owner" not in feedback
    assert "_install_run_context" not in feedback
    assert "sys.modules" not in feedback
    assert "execution_feedback_replan_contract.install" not in finalization


def test_repair_grounding_is_not_late_runtime_patched() -> None:
    adaptive = _text("adaptive_retrieval_contract.py")
    finalization = _FINALIZATION.read_text(encoding="utf-8")

    assert "_install_repository_grounding" not in adaptive
    assert "_install_repository_grounding" not in finalization
    assert "install_repository_grounding" not in finalization


def test_model_output_atomicity_has_one_install_owner() -> None:
    bootstrap = _BOOTSTRAP.read_text(encoding="utf-8")
    finalization = _FINALIZATION.read_text(encoding="utf-8")

    assert bootstrap.count("install_model_output_atomicity()") == 1
    assert "install as install_model_output_atomicity" not in finalization
    assert "install_model_output_atomicity(" not in finalization
    assert "assert_model_output_atomicity" in finalization
