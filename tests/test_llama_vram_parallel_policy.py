from __future__ import annotations

import json
from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai import llama_parallel_runtime_contract as parallel
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


def _agentic(base_width=1, mode="auto"):
    return SimpleNamespace(
        _planner_candidate_count=lambda request, stage: base_width,
        _mode=lambda: mode,
    )


def _runtime_receipt(slots: int) -> str:
    return json.dumps(
        {"schema_version": "mmm/llama-runtime-receipt-v1", "slots": slots},
        sort_keys=True,
    )


def _local_router():
    config = SimpleNamespace(
        exclusive_gpu=True,
        provider="local",
        adapter="llama_cpp",
    )
    registry = SimpleNamespace(role=lambda profile, role: config)
    return SimpleNamespace(registry=registry, profile="default")


def test_relaxed_admission_uses_incremental_host_ram(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_VRAM_PARALLEL", raising=False)
    runtime = _runtime()
    policy.install(runtime, _agentic())
    config = SimpleNamespace(model_id="qwen", max_context=8192)
    resources = SimpleNamespace(
        gpu_free_bytes=9 * 1024 * _MIB,
        ram_available_bytes=2100 * _MIB,
    )

    assert runtime._parallel_resource_feasible(4, config, "/model.gguf", resources)


def test_relaxed_admission_still_rejects_insufficient_vram(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_VRAM_PARALLEL", raising=False)
    runtime = _runtime()
    policy.install(runtime, _agentic())
    config = SimpleNamespace(model_id="qwen", max_context=32768)
    resources = SimpleNamespace(
        gpu_free_bytes=6 * 1024 * _MIB,
        ram_available_bytes=4 * 1024 * _MIB,
    )

    assert not runtime._parallel_resource_feasible(4, config, "/model.gguf", resources)


def test_original_feasible_decision_remains_authoritative():
    runtime = _runtime(original_feasible=True)
    policy.install(runtime, _agentic())
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


def test_parallel_planner_fills_validated_local_llama_slots(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _runtime_receipt(4))
    monkeypatch.delenv("MMM_PLAN_SEARCH_WIDTH", raising=False)
    monkeypatch.delenv("MMM_PLAN_FILL_ACTIVE_LLAMA_SLOTS", raising=False)
    monkeypatch.delenv("MMM_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)

    assert parallel._planner_search_width(_local_router(), 1, _agentic()) == 4


def test_parallel_planner_never_promotes_fake_or_unvalidated_router(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _runtime_receipt(4))
    monkeypatch.delenv("MMM_PLAN_SEARCH_WIDTH", raising=False)
    monkeypatch.delenv("MMM_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)

    fake_router = SimpleNamespace()
    assert parallel._planner_search_width(fake_router, 1, _agentic()) == 1

    monkeypatch.delenv("MMM_LLAMA_RUNTIME_RECEIPT", raising=False)
    assert parallel._planner_search_width(_local_router(), 1, _agentic()) == 1


def test_parallel_planner_respects_operator_width_and_latency(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _runtime_receipt(4))
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "2")
    assert parallel._planner_search_width(_local_router(), 2, _agentic()) == 2

    monkeypatch.delenv("MMM_PLAN_SEARCH_WIDTH", raising=False)
    monkeypatch.setenv("MMM_PERFORMANCE_MODE", "latency")
    assert parallel._planner_search_width(_local_router(), 1, _agentic()) == 1


def test_policy_upgrade_removes_legacy_global_planner_wrapper():
    runtime = _runtime()

    def base_candidate_count(request, stage):
        del request, stage
        return 1

    @wraps(base_candidate_count)
    def legacy_candidate_count(request, stage):
        return max(2, base_candidate_count(request, stage))

    legacy_candidate_count._mmm_fill_active_llama_slots_v2 = True
    agentic = SimpleNamespace(
        _planner_candidate_count=legacy_candidate_count,
        _mode=lambda: "auto",
    )
    policy.install(runtime, agentic)

    assert agentic._planner_candidate_count is base_candidate_count


def test_selection_version_forces_reconsideration_and_install_is_idempotent():
    runtime = _runtime()
    agentic = _agentic()
    policy.install(runtime, agentic)
    first_resource = runtime._parallel_resource_feasible
    first_selection = runtime._selection_inputs
    first_planner = agentic._planner_candidate_count

    assert runtime._selection_inputs(SimpleNamespace(model_id="qwen"))[
        "vram_parallel_policy_version"
    ] == 3

    policy.install(runtime, agentic)
    assert runtime._parallel_resource_feasible is first_resource
    assert runtime._selection_inputs is first_selection
    assert agentic._planner_candidate_count is first_planner
