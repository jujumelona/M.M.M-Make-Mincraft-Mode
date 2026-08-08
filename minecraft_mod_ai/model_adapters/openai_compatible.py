from __future__ import annotations

import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .base import GenerationRequest, ModelAdapter, ModelBackendError, ModelConfigurationError


_AUDIO_MIME_TYPES = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
_DEFAULT_MAX_AUDIO_BYTES = 256 * 1024 * 1024
_MAX_TRANSCRIPTION_RESPONSE_BYTES = 1024 * 1024


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
            if request.response_format == "json":
                # Standard OpenAI-compatible structured-output hint.  Keep it
                # scoped to text generation so image and speech endpoints retain
                # their own response contracts.
                payload["response_format"] = {"type": "json_object"}
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

    def generate_image(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        seed: int,
    ) -> Path:
        """Call the standard OpenAI-compatible image generation endpoint."""

        del seed  # The standard endpoint has no portable deterministic seed.
        cfg = self.config
        try:
            import httpx
            from PIL import Image

            if not cfg.base_url.startswith("https://"):
                raise ModelConfigurationError("Remote base_url must use HTTPS.")
            if not cfg.api_key:
                raise ModelConfigurationError(
                    f"API key is missing for role {cfg.role!r}."
                )
            if not prompt.strip() or width < 1 or height < 1:
                raise ModelConfigurationError(
                    "Remote image prompt and dimensions must be valid."
                )
            with httpx.Client(
                timeout=180.0,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = client.post(
                    f"{cfg.base_url.rstrip('/')}/images/generations",
                    headers={
                        "Authorization": f"Bearer {cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.model_id,
                        "prompt": prompt,
                        "size": f"{width}x{height}",
                        "response_format": "b64_json",
                        "n": 1,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                entries = payload.get("data")
                if not isinstance(entries, list) or len(entries) != 1:
                    raise ModelConfigurationError(
                        "Remote image endpoint returned invalid data."
                    )
                item = entries[0]
                if not isinstance(item, dict):
                    raise ModelConfigurationError(
                        "Remote image result must be an object."
                    )
                encoded = item.get("b64_json")
                if isinstance(encoded, str) and encoded:
                    image_bytes = base64.b64decode(encoded, validate=True)
                else:
                    raise ModelConfigurationError(
                        "Remote image result must contain inline b64_json bytes."
                    )
            if not image_bytes or len(image_bytes) > 128 * 1024 * 1024:
                raise ModelConfigurationError(
                    "Remote image response size is invalid."
                )
            target = output_path.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
            temporary.write_bytes(image_bytes)
            try:
                with Image.open(temporary) as image:
                    image.load()
                    if image.size != (width, height):
                        raise ModelConfigurationError(
                            "Remote image dimensions do not match the request."
                        )
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return target
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    def transcribe(self, audio_path: Path) -> str:
        """Call the OpenAI-compatible multipart transcription endpoint."""

        cfg = self.config
        try:
            import httpx

            parsed_base = urlparse(cfg.base_url)
            if (
                parsed_base.scheme != "https"
                or not parsed_base.hostname
                or parsed_base.username is not None
                or parsed_base.password is not None
                or parsed_base.query
                or parsed_base.fragment
            ):
                raise ModelConfigurationError(
                    "Remote base_url must be a plain HTTPS service URL."
                )
            if not cfg.api_key:
                raise ModelConfigurationError(
                    f"API key is missing for role {cfg.role!r}."
                )
            source = audio_path.expanduser().resolve()
            if not source.is_file():
                raise ModelConfigurationError(
                    f"Audio file does not exist: {source}"
                )
            mime_type = _AUDIO_MIME_TYPES.get(source.suffix.lower())
            if mime_type is None:
                raise ModelConfigurationError(
                    "Remote transcription requires FLAC, M4A, MP3, MP4, "
                    "MPEG, OGG, WAV or WebM audio."
                )
            configured_limit = cfg.extra.get(
                "max_audio_bytes",
                _DEFAULT_MAX_AUDIO_BYTES,
            )
            if (
                type(configured_limit) is not int
                or configured_limit < 1
            ):
                raise ModelConfigurationError(
                    "max_audio_bytes must be a positive integer."
                )
            source_size = source.stat().st_size
            if source_size < 1 or source_size > configured_limit:
                raise ModelConfigurationError(
                    "Audio file is empty or exceeds the configured per-request "
                    "upload limit."
                )
            with (
                source.open("rb") as audio_stream,
                httpx.Client(
                    timeout=180.0,
                    follow_redirects=False,
                    trust_env=False,
                ) as client,
            ):
                response = client.post(
                    f"{cfg.base_url.rstrip('/')}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                    data={"model": cfg.model_id},
                    files={
                        "file": (
                            source.name,
                            audio_stream,
                            mime_type,
                        )
                    },
                )
                response.raise_for_status()
                if (
                    len(response.content)
                    > _MAX_TRANSCRIPTION_RESPONSE_BYTES
                ):
                    raise ModelConfigurationError(
                        "Remote transcription response is too large."
                    )
                payload = response.json()
            if not isinstance(payload, dict):
                raise ModelConfigurationError(
                    "Remote transcription response must be an object."
                )
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ModelConfigurationError(
                    "Remote transcription response lacks text."
                )
            return text.strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc
