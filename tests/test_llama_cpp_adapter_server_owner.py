from __future__ import annotations

from minecraft_mod_ai import llama_server_autotune
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def test_adapter_does_not_bypass_server_owner_when_env_url_exists(monkeypatch) -> None:
    config = AdapterConfig(
        role="visual_critic",
        adapter="llama_cpp",
        model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Inspect this image."},),
    )
    seen: list[tuple[AdapterConfig, GenerationRequest]] = []

    def ensure_tuned_server(
        seen_config: AdapterConfig,
        seen_request: GenerationRequest,
    ) -> str:
        seen.append((seen_config, seen_request))
        return "http://127.0.0.1:8920/v1"

    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8910/v1")
    monkeypatch.setattr(
        llama_server_autotune,
        "ensure_tuned_server",
        ensure_tuned_server,
    )

    adapter = LlamaCppAdapter(config)
    assert adapter._server_url(request) == "http://127.0.0.1:8920/v1"
    assert seen == [(config, request)]
