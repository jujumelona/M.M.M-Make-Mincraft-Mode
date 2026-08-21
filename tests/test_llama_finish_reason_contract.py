from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_finish_reason_contract import (
    _CONTEXT_ERROR,
    _OUTPUT_ERROR,
    install,
)


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _module(response_payload):
    return SimpleNamespace(
        _post_completion=lambda _url, _payload: _Response(response_payload),
        _bounded_response_body=lambda _response: "",
    )


def test_output_cap_is_not_misreported_as_context_pressure() -> None:
    module = _module(
        {
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
            "usage": {"prompt_tokens": 14000, "completion_tokens": 8192},
        }
    )
    install(module)

    with pytest.raises(RuntimeError, match="bounded output allowance") as captured:
        module._completion_message("http://localhost", {"max_tokens": 8192})

    assert _OUTPUT_ERROR in str(captured.value)
    assert _CONTEXT_ERROR not in str(captured.value)
    assert "completion_tokens=8192" in str(captured.value)


def test_context_pressure_remains_eligible_for_context_compaction() -> None:
    module = _module(
        {
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
            "usage": {"prompt_tokens": 31000, "completion_tokens": 1200},
        }
    )
    install(module)

    with pytest.raises(RuntimeError, match="context boundary") as captured:
        module._completion_message("http://localhost", {"max_tokens": 8192})

    assert _CONTEXT_ERROR in str(captured.value)
    assert "prompt_tokens=31000" in str(captured.value)


def test_successful_completion_is_returned_unchanged() -> None:
    message = {"role": "assistant", "content": "done"}
    module = _module(
        {
            "choices": [{"finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 4},
        }
    )
    install(module)

    assert module._completion_message("http://localhost", {"max_tokens": 8192}) == message
