from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
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


def test_fast_search_keeps_only_p1_and_highest_feasible_parallel_width():
    assert policy._shortlist_parallel(range(1, 9), "fast") == (1, 8)
    assert policy._shortlist_parallel((1, 2), "fast") == (1, 2)
    assert policy._shortlist_parallel((1,), "fast") == (1,)


def test_balanced_and_full_parallel_search_are_explicitly_wider():
    assert policy._shortlist_parallel(range(1, 9), "balanced") == (1, 7, 8)
    assert policy._shortlist_parallel(range(1, 5), "full") == (1, 2, 3, 4)


def test_fast_search_skips_restart_heavy_speculation_and_ubatch_sweeps():
    baseline = runtime_tuning.ServerVariant("baseline")
    mtp = runtime_tuning.ServerVariant("mtp-4", "draft-mtp", 4)
    ngram = runtime_tuning.ServerVariant("ngram-map-k", "ngram-map-k")

    assert policy._shortlist_variants((baseline, mtp, ngram), "fast") == (baseline,)
    assert policy._shortlist_ubatches((512, 1024, 2048), "fast") == (512,)


def test_balanced_search_samples_one_speculation_per_family_and_ubatch_edge():
    baseline = runtime_tuning.ServerVariant("baseline")
    mtp2 = runtime_tuning.ServerVariant("mtp-2", "draft-mtp", 2)
    mtp4 = runtime_tuning.ServerVariant("mtp-4", "draft-mtp", 4)
    ngram_simple = runtime_tuning.ServerVariant("ngram-simple", "ngram-simple")
    ngram_map = runtime_tuning.ServerVariant("ngram-map-k", "ngram-map-k")

    assert policy._shortlist_variants(
        (baseline, mtp2, mtp4, ngram_simple, ngram_map), "balanced"
    ) == (baseline, mtp4, ngram_map)
    assert policy._shortlist_ubatches((512, 1024, 2048), "balanced") == (
        512,
        2048,
    )


def test_fast_benchmark_defaults_are_short_and_operator_overrides_win(monkeypatch):
    for name in (
        "MMM_LLAMA_AUTOTUNE_SEARCH",
        "MMM_LLAMA_AUTOTUNE_TOKENS",
        "MMM_LLAMA_AUTOTUNE_MAX_SECONDS",
        "MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    policy._configure_benchmark_defaults()
    assert policy._search_mode() == "fast"
    assert policy.os.environ["MMM_LLAMA_AUTOTUNE_TOKENS"] == "16"
    assert policy.os.environ["MMM_LLAMA_AUTOTUNE_MAX_SECONDS"] == "45"
    assert policy.os.environ["MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS"] == "15"

    monkeypatch.setenv("MMM_LLAMA_AUTOTUNE_TOKENS", "24")
    policy._configure_benchmark_defaults()
    assert policy.os.environ["MMM_LLAMA_AUTOTUNE_TOKENS"] == "24"


def test_selection_version_forces_reconsideration_and_install_is_idempotent():
    runtime = _runtime()
    policy.install(runtime)
    first_resource = runtime._parallel_resource_feasible
    first_selection = runtime._selection_inputs

    selection = runtime._selection_inputs(SimpleNamespace(model_id="qwen"))
    assert selection["vram_parallel_policy_version"] == 5
    assert selection["autotune_search"] in {"fast", "balanced", "full"}

    policy.install(runtime)
    assert runtime._parallel_resource_feasible is first_resource
    assert runtime._selection_inputs is first_selection
