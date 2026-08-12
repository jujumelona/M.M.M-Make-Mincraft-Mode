from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_server_runtime_tuning import (
    _candidate_variants_for_config,
    _model_supports_mtp,
)


def _config(model_id: str, **extra):
    return SimpleNamespace(model_id=model_id, extra=extra)


def test_qwen_mtp_models_keep_native_mtp_probes() -> None:
    config = _config(
        "unsloth/Qwen3.5-9B-MTP-GGUF",
        gguf_filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
    )
    assert _model_supports_mtp(config) is True
    variants = _candidate_variants_for_config(autotune, config)
    assert [value.draft_n_max for value in variants if value.spec_type == "draft-mtp"] == [1, 2, 3]


def test_gemma_never_pays_for_impossible_mtp_server_reloads() -> None:
    config = _config(
        "google/gemma-4-12B-it-qat-q4_0-gguf",
        gguf_filename="gemma-4-12b-it-qat-q4_0.gguf",
    )
    assert _model_supports_mtp(config) is False
    variants = _candidate_variants_for_config(autotune, config)
    assert variants[0].name == "baseline"
    assert all(value.spec_type != "draft-mtp" for value in variants)
    assert any(value.spec_type.startswith("ngram-") for value in variants)


def test_explicit_model_metadata_overrides_name_heuristic() -> None:
    disabled = _config("repo/model-MTP-GGUF", supports_mtp=False)
    enabled = _config("repo/plain-model", supports_mtp=True)
    assert _model_supports_mtp(disabled) is False
    assert _model_supports_mtp(enabled) is True
