from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import complete_orchestrator_services
from minecraft_mod_ai.image_runtime_residency import (
    _finish_image_shard,
    _full_gpu_threshold_mb,
    _is_cuda_memory_pressure,
)
from minecraft_mod_ai.llama_runtime_tuning import _tuned_constructor_kwargs
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.embedding import EmbeddingAdapter
from minecraft_mod_ai.model_adapters.image_diffusion import ImageDiffusionAdapter
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_adapters.reranker import RerankerAdapter


class _DummyPipeline:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str):
        self.moves.append(device)
        return self


def test_expensive_model_runtime_reuse_contract_is_installed() -> None:
    assert getattr(
        ImageDiffusionAdapter.generate_image,
        "_mmm_cached_image_pipeline",
        False,
    )
    assert getattr(
        ImageDiffusionAdapter.generate_image,
        "_mmm_adaptive_image_residency",
        False,
    )
    assert getattr(
        EmbeddingAdapter.embed,
        "_mmm_cached_embedding_model",
        False,
    )
    assert getattr(
        RerankerAdapter.score,
        "_mmm_cached_reranker_model",
        False,
    )
    assert getattr(
        complete_orchestrator_services.generate_assets,
        "_mmm_image_gpu_session",
        False,
    )
    assert getattr(
        complete_orchestrator_services.generate_assets,
        "_mmm_adaptive_image_gpu_session",
        False,
    )
    assert getattr(LlamaCppAdapter.generate, "_mmm_prefill_tuned", False)


def test_image_full_gpu_threshold_keeps_headroom_above_preflight(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MMM_IMAGE_FULL_GPU_MIN_FREE_MB", raising=False)
    assert _full_gpu_threshold_mb(SimpleNamespace(min_free_vram_mb=12_500)) == 14_000
    assert _full_gpu_threshold_mb(SimpleNamespace(min_free_vram_mb=16_000)) == 17_000


def test_image_full_gpu_threshold_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("MMM_IMAGE_FULL_GPU_MIN_FREE_MB", "24576")
    assert _full_gpu_threshold_mb(SimpleNamespace(min_free_vram_mb=12_500)) == 24_576


def test_image_memory_fallback_only_matches_allocation_pressure() -> None:
    assert _is_cuda_memory_pressure(RuntimeError("CUDA out of memory"))
    assert _is_cuda_memory_pressure(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not _is_cuda_memory_pressure(RuntimeError("CUDA driver is unavailable"))
    assert not _is_cuda_memory_pressure(ValueError("invalid image dimensions"))


def test_image_pipeline_cache_is_released_after_shard_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MMM_IMAGE_CACHE_ACROSS_SHARDS", raising=False)
    pipeline = _DummyPipeline()
    runtime = SimpleNamespace(
        _IMAGE_LOCK=threading.RLock(),
        _IMAGE_PIPELINE=pipeline,
        _IMAGE_PIPELINE_KEY=("model", "float16", "full_gpu"),
        _IMAGE_PIPELINE_ON_GPU=True,
    )
    released: list[bool] = []
    base = SimpleNamespace(_release_cuda=lambda: released.append(True))

    _finish_image_shard(runtime, base)

    assert runtime._IMAGE_PIPELINE is None
    assert runtime._IMAGE_PIPELINE_KEY is None
    assert runtime._IMAGE_PIPELINE_ON_GPU is False
    assert pipeline.moves == []
    assert released == [True]


def test_image_pipeline_can_be_parked_between_shards_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("MMM_IMAGE_CACHE_ACROSS_SHARDS", "1")
    pipeline = _DummyPipeline()
    runtime = SimpleNamespace(
        _IMAGE_LOCK=threading.RLock(),
        _IMAGE_PIPELINE=pipeline,
        _IMAGE_PIPELINE_KEY=("model", "float16", "full_gpu"),
        _IMAGE_PIPELINE_ON_GPU=True,
    )
    released: list[bool] = []
    base = SimpleNamespace(_release_cuda=lambda: released.append(True))

    _finish_image_shard(runtime, base)

    assert runtime._IMAGE_PIPELINE is pipeline
    assert runtime._IMAGE_PIPELINE_KEY[-1] == "full_gpu"
    assert runtime._IMAGE_PIPELINE_ON_GPU is False
    assert pipeline.moves == ["cpu"]
    assert released == [True]


def test_llama_prefill_tuning_increases_logical_not_physical_batch(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MMM_LLAMA_BATCH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_UBATCH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_VERBOSE", raising=False)
    tuned = _tuned_constructor_kwargs(
        {"n_batch": 512, "n_ubatch": 512, "verbose": True}
    )
    assert tuned["n_batch"] == 2048
    assert tuned["n_ubatch"] == 512
    assert tuned["verbose"] is False


def test_llama_prefill_tuning_preserves_larger_explicit_batch(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_BATCH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_UBATCH", raising=False)
    tuned = _tuned_constructor_kwargs({"n_batch": 4096, "n_ubatch": 256})
    assert tuned["n_batch"] == 4096
    assert tuned["n_ubatch"] == 256


def test_llama_prefill_tuning_respects_environment(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_BATCH", "1024")
    monkeypatch.setenv("MMM_LLAMA_UBATCH", "256")
    monkeypatch.setenv("MMM_LLAMA_VERBOSE", "1")
    tuned = _tuned_constructor_kwargs(
        {"n_batch": 512, "n_ubatch": 512, "verbose": False}
    )
    assert tuned["n_batch"] == 1024
    assert tuned["n_ubatch"] == 256
    assert tuned["verbose"] is False


def test_llama_generation_session_keeps_shared_gguf_runtime_resident() -> None:
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="local/test.gguf",
        )
    )
    assert getattr(
        LlamaCppAdapter.generation_session,
        "_mmm_resident_llama_session",
        False,
    )

    # Entering and leaving the bounded router session must not call close(), because
    # planner/researcher/coder commonly point at the same GGUF and can reuse it.
    called = False

    def fail_close() -> None:
        nonlocal called
        called = True
        raise AssertionError("resident generation session unexpectedly closed model")

    adapter.close = fail_close  # type: ignore[method-assign]
    with adapter.generation_session():
        pass
    assert called is False
