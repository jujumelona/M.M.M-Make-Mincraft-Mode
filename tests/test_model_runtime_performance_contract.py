from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import complete_orchestrator_services
from minecraft_mod_ai.image_runtime_residency import _full_gpu_threshold_mb
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.embedding import EmbeddingAdapter
from minecraft_mod_ai.model_adapters.image_diffusion import ImageDiffusionAdapter
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_adapters.reranker import RerankerAdapter


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


def test_image_full_gpu_threshold_keeps_headroom_above_preflight() -> None:
    assert _full_gpu_threshold_mb(SimpleNamespace(min_free_vram_mb=12_500)) == 14_000
    assert _full_gpu_threshold_mb(SimpleNamespace(min_free_vram_mb=16_000)) == 17_000


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
