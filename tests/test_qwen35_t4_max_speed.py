from types import SimpleNamespace

from minecraft_mod_ai.qwen35_t4_max_speed import (
    _context_buckets,
    _kv_candidates,
    _kv_mode,
    _p_min_candidates,
    _ubatches,
    _widths,
)


def test_max_speed_width_search_reaches_16(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_MAX_WIDTHS", raising=False)
    assert _widths() == (1, 2, 3, 4, 6, 8, 12, 16)


def test_max_speed_width_override_is_bounded_and_deduplicated(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_MAX_WIDTHS", "2,16,16,33,0,bad,8")
    assert _widths() == (2, 16, 8)


def test_max_speed_sweeps_ubatch_and_confidence(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_UBATCHES", raising=False)
    monkeypatch.delenv("MMM_QWEN35_T4_MAX_P_MIN", raising=False)
    assert _ubatches() == (512, 1024, 2048)
    assert _p_min_candidates() == (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_kv_auto_and_manual_modes(monkeypatch) -> None:
    config = SimpleNamespace(extra={"kv_cache_autotune": True, "kv_cache_quant": "q4_0"})
    monkeypatch.delenv("MMM_QWEN35_T4_KV_MODE", raising=False)
    assert _kv_mode(config) == "auto"
    config.extra["kv_cache_autotune"] = False
    assert _kv_mode(config) == "q4_0"
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "Q8")
    assert _kv_mode(config) == "q8_0"
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "native")
    assert _kv_mode(config) == "native-default"


def test_kv_candidates_keep_native_reference(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_MAX_KV", "q4_0,q8_0,q4_0")
    assert _kv_candidates() == ("native-default", "q4_0", "q8_0")


def test_context_buckets_preserve_full_32k_headroom(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_KV_CONTEXT_BUCKETS", raising=False)
    config = SimpleNamespace(max_context=32768)
    assert _context_buckets(config) == (2048, 8192, 16384, 28672)


def test_native_engine_is_latest_pinned_and_cub3dot2_enabled() -> None:
    from pathlib import Path

    latest = "0d0bfcd4fd8828e3e7906b6fc4561725b534511e"
    setup = Path("tools/colab_runtime_setup.py").read_text(encoding="utf-8")
    worker = Path(".github/workflows/build-native-llama-cuda.yml").read_text(encoding="utf-8")
    assert f'LLAMA_SERVER_SOURCE_REF = "{latest}"' in setup
    assert f"LLAMA_SOURCE_REF: {latest}" in worker
    assert '"-DGGML_CUDA_CUB_3DOT2=ON"' in setup
    assert "-DGGML_CUDA_CUB_3DOT2=ON" in worker
    assert "GGML_CUDA_CUB_3DOT2:BOOL=ON" in worker
    assert '"cuda_cub_3dot2": True' in worker
