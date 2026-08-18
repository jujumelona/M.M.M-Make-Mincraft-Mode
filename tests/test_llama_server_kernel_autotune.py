from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_kernel_autotune as kernel


_ENV_NAMES = (
    "MMM_LLAMA_FLASH_ATTN",
    "MMM_LLAMA_FLASH_ATTN_CANDIDATES",
    "MMM_LLAMA_BATCH",
    "MMM_LLAMA_BATCH_CANDIDATES",
    "MMM_LLAMA_UBATCH",
    "MMM_LLAMA_UBATCH_CANDIDATES",
    "MMM_LLAMA_CACHE_TYPE_K",
    "MMM_LLAMA_CACHE_TYPE_V",
    "MMM_LLAMA_KV_PAIR_CANDIDATES",
    "MMM_KV_CACHE_QUANT",
    "MMM_LLAMA_ACTIVE_FLASH_ATTN",
    "MMM_LLAMA_ACTIVE_BATCH",
    "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
    "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
    "MMM_LLAMA_ACTIVE_KV_CACHE",
)


@pytest.fixture(autouse=True)
def clean_kernel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _probe(config: kernel.KernelConfig, *, tps: float, sha: str = "same", ok: bool = True) -> kernel.KernelProbe:
    return kernel.KernelProbe(
        config=config,
        ok=ok,
        output_sha256=sha,
        predicted_tps=tps,
        prompt_tps=tps * 2.0,
        elapsed_seconds=0.1,
        error="" if ok else "OOM",
    )


def test_t4_defaults_search_flash_batch_and_mixed_kv_pairs() -> None:
    base = kernel.KernelConfig("on", 2048, "q4_0", "q4_0")
    assert set(kernel._flash_candidates(base.flash_attn)) == {"auto", "on", "off"}
    assert {256, 512, 1024, 2048, 4096}.issubset(
        set(kernel._batch_candidates(base.batch, "NVIDIA Tesla T4"))
    )
    pairs = set(kernel._cache_candidates(base, "NVIDIA Tesla T4"))
    assert ("q4_0", "q8_0") in pairs
    assert ("q8_0", "f16") in pairs
    assert ("f16", "f16") in pairs


def test_explicit_operator_values_freeze_each_kernel_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_FLASH_ATTN", "auto")
    monkeypatch.setenv("MMM_LLAMA_BATCH", "1024")
    monkeypatch.setenv("MMM_LLAMA_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("MMM_LLAMA_CACHE_TYPE_V", "f16")
    base = kernel._baseline_config()
    assert kernel._flash_candidates(base.flash_attn) == ("auto",)
    assert kernel._batch_candidates(base.batch, "NVIDIA Tesla T4") == (1024,)
    assert kernel._cache_candidates(base, "NVIDIA Tesla T4") == (("q8_0", "f16"),)


def test_legacy_kv_override_freezes_both_cache_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q8_0")
    base = kernel._baseline_config()
    assert kernel._cache_candidates(base, "NVIDIA Tesla T4") == (("q8_0", "q8_0"),)


def test_stage_selector_requires_verified_output_and_minimum_gain() -> None:
    base_cfg = kernel.KernelConfig("on", 2048, "q4_0", "q4_0")
    fast_cfg = kernel.KernelConfig("auto", 2048, "q4_0", "q4_0")
    current = _probe(base_cfg, tps=20.0)
    assert kernel._select_stage(current, [_probe(fast_cfg, tps=30.0, sha="different")], minimum_gain=1.01) == current
    assert kernel._select_stage(current, [_probe(fast_cfg, tps=40.0, ok=False)], minimum_gain=1.01) == current
    assert kernel._select_stage(current, [_probe(fast_cfg, tps=20.1)], minimum_gain=1.01) == current
    assert kernel._select_stage(current, [_probe(fast_cfg, tps=24.0)], minimum_gain=1.01).config == fast_cfg


def test_benchmark_is_staged_not_cartesian(monkeypatch: pytest.MonkeyPatch) -> None:
    base = kernel.KernelConfig("on", 2048, "q4_0", "q4_0")
    calls: list[kernel.KernelConfig] = []
    monkeypatch.setattr(kernel, "_baseline_config", lambda: base)
    monkeypatch.setattr(kernel, "_flash_candidates", lambda _current: ("on", "auto"))
    monkeypatch.setattr(kernel, "_batch_candidates", lambda _current, _hardware: (2048, 1024))
    monkeypatch.setattr(kernel, "_cache_candidates", lambda _current, _hardware: (("q4_0", "q4_0"), ("q8_0", "f16")))

    def fake_run_probe(
        _autotune,
        _binary,
        _model_path,
        _config,
        _request,
        config,
        **_kwargs,
    ):
        calls.append(config)
        bonus = (10.0 if config.flash_attn == "auto" else 0.0)
        bonus += 10.0 if config.batch == 1024 else 0.0
        bonus += 10.0 if (config.cache_type_k, config.cache_type_v) == ("q8_0", "f16") else 0.0
        return _probe(config, tps=20.0 + bonus)

    monkeypatch.setattr(kernel, "_run_probe", fake_run_probe)
    fake_autotune = SimpleNamespace(_env_float=lambda _n, default: default)
    selected, probes = kernel._benchmark(fake_autotune, "llama-server", "/tmp/model.gguf", object(), object(), "NVIDIA Tesla T4")
    assert len(calls) == 4
    assert len(probes) == 4
    assert selected == kernel.KernelConfig("auto", 1024, "q8_0", "f16")


def test_install_applies_active_kernel_winner_and_clamps_ubatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def base_args(_binary, _model_path, _config, _port):
        return ["llama-server", "--flash-attn", "on", "--batch-size", "2048", "--ubatch-size", "2048", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]

    autotune = SimpleNamespace(
        _base_args=base_args,
        _fingerprint=lambda *_args: "base-fingerprint",
        ensure_tuned_server=lambda *_args: "http://127.0.0.1:8910/v1",
        _hardware_identity=lambda: "NVIDIA Tesla T4",
        _env_int=lambda name, default: int(__import__("os").environ.get(name, default)),
    )
    runtime = SimpleNamespace(_ubatch_candidates=lambda _autotune: (512,))
    kernel.install(autotune, runtime)
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_FLASH_ATTN", "auto")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_BATCH", "1024")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_CACHE_TYPE_V", "f16")
    args = autotune._base_args("llama-server", "/tmp/model.gguf", object(), 8910)

    def value(name: str) -> str:
        return args[args.index(name) + 1]

    assert value("--flash-attn") == "auto"
    assert value("--batch-size") == "1024"
    assert value("--cache-type-k") == "q8_0"
    assert value("--cache-type-v") == "f16"
    assert value("--ubatch-size") == "1024"


def test_t4_ubatch_search_is_expanded_and_never_exceeds_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    autotune = SimpleNamespace(
        _base_args=lambda *_args: [],
        _fingerprint=lambda *_args: "base",
        ensure_tuned_server=lambda *_args: "url",
        _hardware_identity=lambda: "NVIDIA Tesla T4",
        _env_int=lambda name, default: int(__import__("os").environ.get(name, default)),
    )
    runtime = SimpleNamespace(_ubatch_candidates=lambda _autotune: (512,))
    kernel.install(autotune, runtime)
    monkeypatch.setenv("MMM_LLAMA_BATCH", "1024")
    values = runtime._ubatch_candidates(autotune)
    assert {128, 256, 512, 1024}.issubset(set(values))
    assert all(value <= 1024 for value in values)


def test_pipeline_keeps_existing_runtime_layers_and_adds_kernel_outer_stage() -> None:
    from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline

    pipeline = NativeLlamaTuningPipeline(
        autotune=SimpleNamespace(),
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    assert tuple(stage.name for stage in pipeline.stages()) == (
        "hardware",
        "efficiency",
        "runtime",
        "cache-reuse",
        "decode-speed",
        "kernel-autotune",
        "multimodal",
    )
