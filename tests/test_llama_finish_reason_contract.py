from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_finish_reason_contract import (
    _CONTEXT_ERROR,
    _OUTPUT_ERROR,
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
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
    boundary = completion_boundary_error(captured.value)
    assert boundary is not None
    assert boundary.kind == OUTPUT_EXHAUSTED
    assert boundary.partial_message == {"content": "partial"}
    assert boundary.partial_bytes > 0
    assert len(boundary.partial_sha256) == 64
    assert "partial" not in str(boundary)


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
    boundary = completion_boundary_error(captured.value)
    assert boundary is not None
    assert boundary.kind == CONTEXT_PRESSURE
    assert boundary.prompt_tokens == 31000
    assert boundary.completion_tokens == 1200
    assert boundary.max_tokens == 8192


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


def test_unversioned_installer_does_not_accept_a_stale_versioned_hot_reload_marker() -> None:
    module = _module(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "new"}}],
        }
    )
    module._mmm_llama_finish_reason_classifier_v2 = True

    install(module)

    assert module._completion_message("http://localhost", {}) == {"content": "new"}
    assert module._mmm_llama_finish_reason_classifier is True


def test_pinned_llama_http_context_error_is_typed_without_echoing_body() -> None:
    body = (
        '{"error":{"message":"request (40001 tokens) exceeds the available '
        'context size (32768 tokens), try increasing it",'
        '"type":"exceed_context_size"}}'
    )
    response = SimpleNamespace(status_code=400, text=body)
    module = SimpleNamespace(
        _post_completion=lambda _url, _payload: response,
        _bounded_response_body=lambda returned: returned.text,
    )
    install(module)

    with pytest.raises(RuntimeError) as caught:
        module._completion_message("http://localhost", {"max_tokens": 8192})

    boundary = completion_boundary_error(caught.value)
    assert boundary is not None
    assert boundary.kind == CONTEXT_PRESSURE
    assert boundary.max_tokens == 8192
    assert "40001" not in str(boundary)


def test_unrelated_http_400_is_not_misclassified_as_context_pressure() -> None:
    response = SimpleNamespace(
        status_code=400,
        text='{"error":{"message":"invalid tool schema"}}',
    )
    module = SimpleNamespace(
        _post_completion=lambda _url, _payload: response,
        _bounded_response_body=lambda returned: returned.text,
    )
    install(module)

    with pytest.raises(RuntimeError, match="invalid tool schema") as caught:
        module._completion_message("http://localhost", {"max_tokens": 8192})

    assert completion_boundary_error(caught.value) is None
