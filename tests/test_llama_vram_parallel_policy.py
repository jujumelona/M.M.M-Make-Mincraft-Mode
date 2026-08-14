from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_vram_parallel_policy as policy


_MIB = 1024 * 1024


def _runtime(*, original_feasible=False, performance_mode="auto"):
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
        _performance_mode=lambda: performance_mode,
    )


def _agentic(base_width=1, mode="auto"):
    return SimpleNamespace(
        _planner_candidate_count=lambda request, stage: base_width,
        _mode=lambda: mode,
    )


def test_relaxed_admission_uses_incremental_host_ram(monkeypatch):
    monkeypatch.delenv("MMM_LLAMA_VRAM_PARALLEL", raising=False)
    runtime = _runtime()
    agentic = _agentic()
    policy.install(runtime, agentic)
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


def test_planner_fills_validated_active_slots(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.delenv("MMM_PLAN_SEARCH_WIDTH", raising=False)
    monkeypatch.delenv("MMM_PLAN_FILL_ACTIVE_LLAMA_SLOTS", raising=False)
    runtime = _runtime()
    agentic = _agentic(base_width=1)
    policy.install(runtime, agentic)

    assert agentic._planner_candidate_count({}, "plan") == 4


def test_planner_respects_explicit_search_width_and_latency(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "2")
    runtime = _runtime()
    agentic = _agentic(base_width=2)
    policy.install(runtime, agentic)
    assert agentic._planner_candidate_count({}, "plan") == 2

    monkeypatch.delenv("MMM_PLAN_SEARCH_WIDTH", raising=False)
    latency_runtime = _runtime(performance_mode="latency")
    latency_agentic = _agentic(base_width=1)
    policy.install(latency_runtime, latency_agentic)
    assert latency_agentic._planner_candidate_count({}, "plan") == 1


def test_selection_version_forces_one_reconsideration_and_install_is_idempotent():
    runtime = _runtime()
    agentic = _agentic()
    policy.install(runtime, agentic)
    first_resource = runtime._parallel_resource_feasible
    first_selection = runtime._selection_inputs
    first_planner = agentic._planner_candidate_count

    assert runtime._selection_inputs(SimpleNamespace(model_id="qwen"))[
        "vram_parallel_policy_version"
    ] == 1

    policy.install(runtime, agentic)
    assert runtime._parallel_resource_feasible is first_resource
    assert runtime._selection_inputs is first_selection
    assert agentic._planner_candidate_count is first_planner
