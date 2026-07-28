from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Mapping

from .base import GenerationRequest, ModelAdapter, ModelBackendError, ModelConfigurationError


def _data_url(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelConfigurationError(f"Media file does not exist: {resolved}")
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleAdapter(ModelAdapter):
    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            import httpx

            if not cfg.base_url.startswith("https://"):
                raise ModelConfigurationError("Remote base_url must use HTTPS.")
            if not cfg.api_key:
                raise ModelConfigurationError(f"API key is missing for role {cfg.role!r}.")
            messages: list[dict[str, Any]] = [dict(item) for item in request.messages]
            if request.media_paths:
                if not messages or messages[-1].get("role") != "user":
                    raise ModelConfigurationError("Media requires a final user message.")
                text = messages[-1].get("content", "")
                if not isinstance(text, str):
                    raise ModelConfigurationError("Final user content must be text.")
                content: list[Mapping[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": _data_url(path)}}
                    for path in request.media_paths
                ]
                content.append({"type": "text", "text": text})
                messages[-1] = {"role": "user", "content": content}
            payload = {
                "model": cfg.model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": cfg.max_new_tokens,
            }
            with httpx.Client(timeout=120.0, follow_redirects=False) as client:
                response = client.post(
                    f"{cfg.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ModelConfigurationError("Remote model returned non-text content.")
            return content.strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
