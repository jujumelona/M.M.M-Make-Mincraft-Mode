from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{path}: expected {count} exact matches, found {found}")
    target.write_text(text.replace(old, new, count), encoding="utf-8")


replace(
    "minecraft_mod_ai/validation_checkpoint_policy.py",
    '''        from . import (\n            java_lsp,\n            java_lsp_process_safety_contract,\n            orchestrator_jdt_gate_contract,\n        )\n\n        common.extend(\n            (\n                java_lsp,\n                java_lsp_process_safety_contract,\n                orchestrator_jdt_gate_contract,\n            )\n        )\n''',
    '''        from . import (\n            java_lsp,\n            java_lsp_process_safety_contract,\n            validation_diagnostic_contract,\n        )\n\n        common.extend(\n            (\n                java_lsp,\n                java_lsp_process_safety_contract,\n                validation_diagnostic_contract,\n            )\n        )\n''',
)

# Runtime contract composition belongs to runtime_bootstrap, not to leaf contracts.
replace(
    "minecraft_mod_ai/llama_prefill_telemetry_contract.py",
    '''def install(hardware_policy_module: Any) -> None:\n    # These worker-5 runtime policies are installed after the native tuning/stream\n    # pipeline exists, so no shared runtime-bootstrap owner is needed here.\n    from . import (\n        forced_tool_execution_contract,\n        llama_completion_liveness_contract,\n        llama_decode_speed_contract,\n        llama_stream_efficiency_contract,\n        model_context_budget,\n        model_router,\n    )\n    from .llama_completion_liveness_contract import install as install_completion_liveness\n    from .llama_context_safety_contract import install as install_context_safety\n    from .llama_forced_tool_capability_contract import (\n        install as install_forced_tool_capability,\n    )\n    from .llama_kv_correctness_contract import install as install_kv_correctness\n    from .llama_sse_error_contract import install as install_sse_errors\n    from .llama_tool_round_safety_contract import install as install_tool_round_safety\n    from .model_adapters import llama_cpp_adapter\n\n    install_completion_liveness(llama_stream_efficiency_contract, llama_cpp_adapter)\n    install_sse_errors(\n        llama_completion_liveness_contract,\n        llama_stream_efficiency_contract,\n    )\n    install_kv_correctness(llama_decode_speed_contract)\n    install_context_safety(model_context_budget)\n    install_tool_round_safety(model_router)\n    # forced_tool_execution.install runs immediately after this prefill hook. Patching\n    # its module-level probe owners now makes the later adapter wrapper capture the\n    # recoverable capability policy without adding a second bootstrap stage.\n    install_forced_tool_capability(forced_tool_execution_contract)\n\n''',
    '''def install(hardware_policy_module: Any) -> None:\n    """Install prompt-prefill telemetry only; bootstrap owns policy composition."""\n\n''',
)

replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    from . import (\n        complete_orchestrator_services,\n        llama_server_autotune,\n        llama_server_hardware_policy,\n        llama_server_runtime_tuning,\n        model_registry,\n        model_router,\n    )\n''',
    '''    from . import (\n        complete_orchestrator_services,\n        forced_tool_execution_contract,\n        llama_completion_liveness_contract,\n        llama_decode_speed_contract,\n        llama_server_autotune,\n        llama_server_hardware_policy,\n        llama_server_runtime_tuning,\n        llama_stream_efficiency_contract,\n        model_context_budget,\n        model_registry,\n        model_router,\n    )\n''',
)
replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    from .llama_generation_budget import install as install_llama_generation_budget\n    from .llama_prefill_telemetry_contract import (\n        install as install_llama_prefill_telemetry,\n    )\n''',
    '''    from .llama_completion_liveness_contract import install as install_completion_liveness\n    from .llama_context_safety_contract import install as install_context_safety\n    from .llama_forced_tool_capability_contract import (\n        install as install_forced_tool_capability,\n    )\n    from .llama_generation_budget import install as install_llama_generation_budget\n    from .llama_kv_correctness_contract import install as install_kv_correctness\n    from .llama_prefill_telemetry_contract import (\n        install as install_llama_prefill_telemetry,\n    )\n    from .llama_sse_error_contract import install as install_sse_errors\n''',
)
replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    from .llama_stream_efficiency_contract import (\n        install as install_llama_stream_efficiency,\n    )\n    from .llama_tuning_pipeline import install_native_llama_tuning_pipeline\n''',
    '''    from .llama_stream_efficiency_contract import (\n        install as install_llama_stream_efficiency,\n    )\n    from .llama_tool_round_safety_contract import install as install_tool_round_safety\n    from .llama_tuning_pipeline import install_native_llama_tuning_pipeline\n''',
)
replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    install_llama_generation_budget(llama_server_hardware_policy)\n    install_llama_stream_efficiency(llama_server_hardware_policy)\n    install_llama_prefill_telemetry(llama_server_hardware_policy)\n    install_forced_tool_execution(\n''',
    '''    install_llama_generation_budget(llama_server_hardware_policy)\n    install_llama_stream_efficiency(llama_server_hardware_policy)\n    install_completion_liveness(llama_stream_efficiency_contract, llama_cpp_adapter)\n    install_sse_errors(\n        llama_completion_liveness_contract,\n        llama_stream_efficiency_contract,\n    )\n    install_kv_correctness(llama_decode_speed_contract)\n    install_context_safety(model_context_budget)\n    install_tool_round_safety(model_router)\n    install_forced_tool_capability(forced_tool_execution_contract)\n    install_llama_prefill_telemetry(llama_server_hardware_policy)\n    install_forced_tool_execution(\n''',
)

replace(
    "minecraft_mod_ai/platform_specialized_generator_contract.py",
    '''    from .minecraft_domain_correctness_contract import (\n        install as install_minecraft_domain_correctness,\n    )\n\n    # Legacy item/block/scaffold generators and specialized generators share the same\n    # authoritative platform stage. Install the legacy boundary here so package import\n    # does not create a second, untracked runtime patch chain after bootstrap integrity.\n    install_minecraft_domain_correctness()\n    _install_incremental_system_records(system_module)\n''',
    '''    _install_incremental_system_records(system_module)\n''',
)
replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    from .mod_scope_contract import install as install_mod_scope\n    from .platform_central_ai_contract import install as install_platform_central_ai\n''',
    '''    from .minecraft_domain_correctness_contract import (\n        install as install_minecraft_domain_correctness,\n    )\n    from .mod_scope_contract import install as install_mod_scope\n    from .platform_central_ai_contract import install as install_platform_central_ai\n''',
)
replace(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    install_platform_repair(repair_engine)\n    install_live_execution(complete_orchestrator)\n    install_specialized_generator_guards(\n''',
    '''    install_platform_repair(repair_engine)\n    install_live_execution(complete_orchestrator)\n    install_minecraft_domain_correctness()\n    install_specialized_generator_guards(\n''',
)

test_path = Path("tests/test_worker12_shared_core.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from minecraft_mod_ai import complete_orchestrator, validation_execution_contract\n",
    "from minecraft_mod_ai import (\n    complete_orchestrator,\n    validation_checkpoint_policy,\n    validation_diagnostic_contract,\n    validation_execution_contract,\n)\n",
    1,
)
text += '''\n\ndef test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:\n    modules = validation_checkpoint_policy._validation_modules("validate-jdt")\n    assert validation_diagnostic_contract in modules\n    assert all(module.__name__ != "minecraft_mod_ai.orchestrator_jdt_gate_contract" for module in modules)\n'''
test_path.write_text(text, encoding="utf-8")
