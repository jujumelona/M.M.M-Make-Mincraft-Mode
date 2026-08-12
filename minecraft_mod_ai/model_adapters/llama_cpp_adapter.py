"""Native llama.cpp server GGUF inference adapter.

Local GGUF inference is server-only.  The Python llama binding is intentionally not
an execution fallback: model selection, GPU residency, MTP benchmarking and metrics
all belong to the managed native llama-server process.
"""
from __future__ import annotations

import os
from typing import Any

from .base import AdapterConfig, GenerationRequest, ModelAdapter, ModelBackendError


class LlamaCppAdapter(ModelAdapter):
    """OpenAI-compatible client for the managed native llama-server."""

    _reported_server_url: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)

    def _server_url(self, request: GenerationRequest) -> str:
        explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
        if explicit:
            return explicit
        try:
            from .. import llama_server_autotune

            selected = llama_server_autotune.ensure_tuned_server(self.config, request)
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=RuntimeError(
                    "native llama-server could not be prepared; local GGUF inference "
                    "has no alternate in-process backend"
                ),
            ) from exc
        endpoint = (selected or "").strip().rstrip("/")
        if not endpoint:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=RuntimeError(
                    "native llama-server is required for local GGUF inference but no "
                    "server URL was produced"
                ),
            )
        return endpoint

    def generate(self, request: GenerationRequest) -> str:
        """Generate through native llama-server only.

        llama_server_hardware_policy normally wraps this method with the streaming
        client.  This direct implementation remains a correctness-safe fallback for
        imports/tests where that package-level contract has not yet been installed;
        it still talks to the same native server and never loads another model.
        """

        cfg = self.config
        server_url = self._server_url(request)
        try:
            import httpx

            health = httpx.get(f"{server_url}/models", timeout=2.0)
            health.raise_for_status()
            if LlamaCppAdapter._reported_server_url != server_url:
                print("llama server: connected", server_url, flush=True)
                LlamaCppAdapter._reported_server_url = server_url

            payload: dict[str, Any] = {
                "model": "local",
                "messages": [dict(message) for message in request.messages],
                "max_tokens": int(cfg.max_new_tokens),
                "temperature": 0.0,
            }
            if getattr(request, "response_format", None) == "json":
                payload["response_format"] = {"type": "json_object"}
                payload["reasoning_effort"] = "none"

            response = httpx.post(
                f"{server_url}/chat/completions",
                json=payload,
                timeout=None,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("native llama-server returned no completion choice")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("native llama-server returned no visible content")
            return content.strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    def close(self) -> None:
        # The managed native server owns model lifetime.  GPU handoff is performed by
        # llama_server_hardware_policy/llama_server_autotune, not by adapter teardown.
        return None
