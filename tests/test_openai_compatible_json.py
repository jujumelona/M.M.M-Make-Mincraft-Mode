from __future__ import annotations

from typing import Any

import httpx
import pytest

import minecraft_mod_ai.model_adapters.openai_compatible as openai_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.openai_compatible import OpenAICompatibleAdapter


def _adapter() -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        AdapterConfig(
            role="planner",
            adapter="openai_compatible",
            provider="openai_compatible",
            model_id="remote-planner",
            base_url="https://models.example.test/v1",
            api_key="test-secret",
            max_new_tokens=321,
        )
    )


def _capture_payload(
    monkeypatch: pytest.MonkeyPatch,
    request: GenerationRequest,
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "{}"}}]}

    class _Client:
        def __init__(self, *, timeout: float, follow_redirects: bool) -> None:
            assert timeout == 120.0
            assert follow_redirects is False

        def post(self, url: str, **kwargs: Any) -> _Response:
            assert url == "https://models.example.test/v1/chat/completions"
            assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
            captured.append(kwargs["json"])
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    assert _adapter().generate(request) == "{}"
    assert len(captured) == 1
    return captured[0]


def test_openai_compatible_json_generation_requests_json_object_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _capture_payload(
        monkeypatch,
        GenerationRequest(
            messages=({"role": "user", "content": "Return a JSON plan."},),
            response_format="json",
        ),
    )

    assert payload["response_format"] == {"type": "json_object"}


def test_openai_compatible_text_generation_keeps_standard_text_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _capture_payload(
        monkeypatch,
        GenerationRequest(
            messages=({"role": "user", "content": "Explain the plan."},),
            response_format="text",
        ),
    )

    assert "response_format" not in payload


def test_openai_compatible_json_schema_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    payload = _capture_payload(
        monkeypatch,
        GenerationRequest(
            messages=({"role": "user", "content": "Return structured JSON."},),
            response_format="json",
            response_schema=schema,
        ),
    )

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_structured_response",
            "strict": True,
            "schema": schema,
        },
    }


def test_openai_compatible_reuses_one_completion_client_across_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = 0
    calls = 0

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def __init__(self, *, timeout: float, follow_redirects: bool) -> None:
            nonlocal created
            assert timeout == 120.0
            assert follow_redirects is False
            created += 1

        def post(self, url: str, **kwargs: Any) -> _Response:
            nonlocal calls
            assert url == "https://models.example.test/v1/chat/completions"
            calls += 1
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    request = GenerationRequest(
        messages=({"role": "user", "content": "reuse the transport"},),
    )

    assert _adapter().generate(request) == "ok"
    assert _adapter().generate(request) == "ok"

    assert created == 1
    assert calls == 2
    with openai_adapter._CLIENT_LOCK:
        keys = tuple(openai_adapter._CLIENTS)
    assert any(
        base_url == "https://models.example.test/v1" and purpose == "completion"
        for base_url, purpose, _factory in keys
    )
