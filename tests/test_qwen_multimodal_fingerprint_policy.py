from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_multimodal_contract as multimodal


def _qwen_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
            "runtime_contract": "qwen",
        },
    )


def test_multimodal_policy_invalidates_pre_safe_qwen_cache(monkeypatch) -> None:
    def fingerprint(_config, _binary, _model_path) -> str:
        return "legacy-fingerprint"

    fake = SimpleNamespace(_fingerprint=fingerprint)
    multimodal._install_fingerprint_policy(fake)

    monkeypatch.delenv(multimodal._ACTIVE_MEDIA_ENV, raising=False)
    text_value = fake._fingerprint(_qwen_config(), "server", "/tmp/model.gguf")
    monkeypatch.setenv(multimodal._ACTIVE_MEDIA_ENV, "1")
    media_value = fake._fingerprint(_qwen_config(), "server", "/tmp/model.gguf")

    assert text_value != "legacy-fingerprint"
    assert media_value == text_value
    assert getattr(fake._fingerprint, "_mmm_llama_multimodal_fingerprint_v1", False)


def test_multimodal_fingerprint_leaves_other_models_unchanged() -> None:
    def fingerprint(_config, _binary, _model_path) -> str:
        return "base"

    fake = SimpleNamespace(_fingerprint=fingerprint)
    multimodal._install_fingerprint_policy(fake)
    generic = SimpleNamespace(model_id="other/model", extra={})

    assert fake._fingerprint(generic, "server", "/tmp/model.gguf") == "base"
