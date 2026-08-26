from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import llama_vram_parallel_policy as policy

_MIB = 1024 * 1024


def _runtime(*, original_feasible=False):
    def original(slots, config, model_path, resources):
        del slots, config, model_path, resources
        return original_feasible

    return SimpleNamespace(
        _parallel_resource_feasible=original,
        _selection_inputs=lambda config: {"model_id": config.model_id},
        _per_request_context=lambda config: config.max_context,
        _total_context=lambda context, slots: context * slots,
        _model_size=lambda model_path: 5 * 1024 * _MIB if model_path else 0,
        _kv_bytes_per_token=lambda: 24 * 1024,
    )


def _runtime_receipt(slots: int) -> str:
    return json.dumps(
        {"schema_version": "mmm/llama-runtime-receipt-v1", "slots": slots},
        sort_keys=True,
    )


def test_relaxed_admission_uses_incremental_host_ram(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_VRAM_PARALLEL", raising=False)
    runtime = _runtime()
    policy.install(runtime)
    config = SimpleNamespace(model_id="qwen", max_context=8192)
    resources = SimpleNamespace(
        gpu_free_bytes=9 * 1024 * _MIB,
        ram_available_bytes=2100 * _MIB,
    )

    assert runtime._parallel_resource_feasible(4, config, "/model.gguf", resources)


def test_relaxed_admission_still_rejects_insufficient_vram(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_VRAM_PARALLEL", raising=False)
    runtime = _runtime()
    policy.install(runtime)
    config = SimpleNamespace(model_id="qwen", max_context=32768)
    resources = SimpleNamespace(
        gpu_free_bytes=6 * 1024 * _MIB,
        ram_available_bytes=4 * 1024 * _MIB,
    )

    assert not runtime._parallel_resource_feasible(4, config, "/model.gguf", resources)


def test_original_feasible_decision_remains_authoritative():
    runtime = _runtime(original_feasible=True)
    policy.install(runtime)
    config = SimpleNamespace(model_id="qwen", max_context=32768)
    resources = SimpleNamespace(gpu_free_bytes=1, ram_available_bytes=1)

    assert runtime._parallel_resource_feasible(4, config, "/model.gguf", resources)


def test_validated_slots_require_matching_managed_receipt(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.delenv("MMM_LLAMA_RUNTIME_RECEIPT", raising=False)
    assert policy.validated_active_parallelism() == 1

    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _runtime_receipt(2))
    assert policy.validated_active_parallelism() == 1

    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _runtime_receipt(4))
    assert policy.validated_active_parallelism() == 4


def test_selection_version_forces_reconsideration_and_install_is_idempotent():
    runtime = _runtime()
    policy.install(runtime)
    first_resource = runtime._parallel_resource_feasible
    first_selection = runtime._selection_inputs

    assert runtime._selection_inputs(SimpleNamespace(model_id="qwen"))[
        "vram_parallel_policy_version"
    ] == 3

    policy.install(runtime)
    assert runtime._parallel_resource_feasible is first_resource
    assert runtime._selection_inputs is first_selection
