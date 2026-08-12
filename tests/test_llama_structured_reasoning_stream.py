from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import (
    _server_payload,
    _stream_delta_parts,
)


def _adapter(max_new_tokens: int = 8192):
    return SimpleNamespace(config=SimpleNamespace(max_new_tokens=max_new_tokens))


def test_json_request_disables_reasoning_before_grammar_generation() -> None:
    request = SimpleNamespace(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Plan it."},
        ),
        response_format="json",
    )
    payload = _server_payload(_adapter(), request)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 8192


def test_text_request_does_not_force_reasoning_policy() -> None:
    request = SimpleNamespace(
        messages=({"role": "user", "content": "hello"},),
        response_format="text",
    )
    payload = _server_payload(_adapter(64), request)
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload


def test_reasoning_delta_is_progress_but_not_visible_content() -> None:
    reasoning, content = _stream_delta_parts(
        {"delta": {"reasoning_content": "hidden reasoning"}}
    )
    assert reasoning == "hidden reasoning"
    assert content == ""


def test_content_delta_is_visible_content() -> None:
    reasoning, content = _stream_delta_parts(
        {"delta": {"content": '{"ok":true}'}}
    )
    assert reasoning == ""
    assert content == '{"ok":true}'


def test_legacy_text_choice_is_still_supported() -> None:
    reasoning, content = _stream_delta_parts({"text": "legacy"})
    assert reasoning == ""
    assert content == "legacy"
