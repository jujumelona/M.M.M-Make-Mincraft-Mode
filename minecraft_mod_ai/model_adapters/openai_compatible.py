from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Mapping

from .base import (
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
    ModelBackendError,
    ModelConfigurationError,
    ToolCall,
)


def _data_url(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelConfigurationError(f"Media file does not exist: {resolved}")
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleAdapter(ModelAdapter):
    def generate(self, request: GenerationRequest) -> str:
        turn = self.generate_turn(request)
        if not turn.content and turn.tool_calls:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=(
                    "A tool-aware completion was requested through the text-only "
                    "generate() API. Use ModelRouter.generate_text() so tool calls "
                    "can be executed."
                ),
            )
        return turn.content

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        cfg = self.config
        try:
            import httpx

            if not cfg.base_url.startswith("https://"):
                raise ModelConfigurationError("Remote base_url must use HTTPS.")
            if not cfg.api_key:
                raise ModelConfigurationError(
                    f"API key is missing for role {cfg.role!r}."
                )
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

            payload: dict[str, Any] = {
                "model": cfg.model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": cfg.max_new_tokens,
            }
            if request.response_format == "json":
                if request.response_schema is not None:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "mmm_structured_response",
                            "strict": True,
                            "schema": dict(request.response_schema),
                        },
                    }
                else:
                    payload["response_format"] = {"type": "json_object"}
            if request.tools:
                payload["tools"] = [dict(tool) for tool in request.tools]
                payload["tool_choice"] = request.tool_choice or "auto"
                payload["parallel_tool_calls"] = bool(request.parallel_tool_calls)

            with httpx.Client(follow_redirects=False) as client:
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

            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                raise ModelConfigurationError(
                    "Remote model returned no completion choice."
                )
            message = (
                choices[0].get("message") if isinstance(choices[0], dict) else None
            )
            if not isinstance(message, Mapping):
                raise ModelConfigurationError(
                    "Remote model returned no assistant message."
                )
            content_value = message.get("content")
            visible = content_value if isinstance(content_value, str) else ""
            reasoning_value = message.get("reasoning_content")
            reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
            tool_calls = _parse_tool_calls(message.get("tool_calls"))
            if not visible.strip() and not tool_calls:
                raise ModelConfigurationError(
                    "Remote model returned neither text nor tool calls."
                )
            return GenerationResponse(
                content=visible.strip(),
                tool_calls=tool_calls,
                reasoning_content=reasoning.strip(),
            )
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
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
        """Call an OpenAI-compatible image endpoint; FLUX remains a separate role."""
        del seed
        cfg = self.config
        try:
            import httpx

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
            with httpx.Client(follow_redirects=False, trust_env=False) as client:
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
                if not isinstance(encoded, str) or not encoded:
                    raise ModelConfigurationError(
                        "Remote image result must contain inline b64_json bytes."
                    )
                image_bytes = base64.b64decode(encoded, validate=True)

            if not image_bytes or len(image_bytes) > 128 * 1024 * 1024:
                raise ModelConfigurationError(
                    "Remote image response size is invalid."
                )
            target = output_path.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
            temporary.write_bytes(image_bytes)
            try:
                from PIL import Image

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


def _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ModelConfigurationError("Remote tool_calls must be a list.")
    result: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ModelConfigurationError(
                "Remote model returned an invalid tool call."
            )
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise ModelConfigurationError("Remote tool call lacks function data.")
        name = str(function.get("name", "")).strip()
        if not name:
            raise ModelConfigurationError(
                "Remote tool call lacks a function name."
            )
        raw_value = function.get("arguments", "{}")
        if isinstance(raw_value, str):
            raw_arguments = raw_value.strip() or "{}"
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ModelConfigurationError(
                    f"Remote tool {name!r} returned invalid JSON arguments."
                ) from exc
        elif isinstance(raw_value, Mapping):
            parsed = dict(raw_value)
            raw_arguments = json.dumps(
                parsed,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            raise ModelConfigurationError(
                f"Remote tool {name!r} arguments must be a JSON object."
            )
        if not isinstance(parsed, Mapping):
            raise ModelConfigurationError(
                f"Remote tool {name!r} arguments must decode to an object."
            )
        result.append(
            ToolCall(
                id=str(item.get("id", "")).strip() or f"call_{index}",
                name=name,
                arguments=dict(parsed),
                raw_arguments=raw_arguments,
            )
        )
    return tuple(result)
