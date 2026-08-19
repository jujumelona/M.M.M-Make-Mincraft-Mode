from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import llama_cpp_adapter


def _response(finish_reason: str):
    return SimpleNamespace(
        status_code=200,
        json=lambda: {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": '{"ok":true}'},
                }
            ]
        },
    )


def test_completion_message_rejects_length_truncation(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_post_completion",
        lambda server_url, payload: _response("length"),
    )

    with pytest.raises(RuntimeError, match="context boundary before the assistant turn completed"):
        llama_cpp_adapter._completion_message("http://127.0.0.1:8910/v1", {})


def test_completion_message_accepts_normal_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_post_completion",
        lambda server_url, payload: _response("stop"),
    )

    message = llama_cpp_adapter._completion_message("http://127.0.0.1:8910/v1", {})
    assert message["content"] == '{"ok":true}'
