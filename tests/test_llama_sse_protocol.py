from __future__ import annotations

from minecraft_mod_ai import llama_stream_efficiency_contract as stream
from minecraft_mod_ai.llama_sse_protocol import sse_error_from_line


def test_current_llama_sse_error_shape_is_normalized() -> None:
    parsed = sse_error_from_line(
        'data: {"error":{"code":400,"message":"context overflow","type":"invalid_request_error"}}'
    )
    assert parsed is not None
    status, error = parsed
    assert status == 400
    assert error["code"] == 400
    assert error["message"] == "context overflow"


def test_legacy_llama_sse_error_shape_is_preserved() -> None:
    parsed = sse_error_from_line('error: {"code":503,"message":"server busy"}')
    assert parsed == (503, {"code": 503, "message": "server busy"})


def test_legacy_text_error_is_normalized() -> None:
    parsed = sse_error_from_line("error: temporary backend failure")
    assert parsed == (
        500,
        {"code": 500, "message": "temporary backend failure", "type": "server_error"},
    )


def test_normal_content_is_not_misclassified_as_error() -> None:
    assert (
        sse_error_from_line('data: {"choices":[{"delta":{"content":"error: harmless"}}]}')
        is None
    )


def test_stream_aggregator_returns_http_error_without_runtime_monkeypatch() -> None:
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_lines(self):
            return iter(
                ['data: {"error":{"code":400,"message":"context overflow"}}']
            )

    class Client:
        def stream(self, *_args, **_kwargs):
            return Response()

        def post(self, *_args, **_kwargs):
            raise AssertionError("non-stream fallback must not be used")

    client = stream._StreamingCompletionClient(Client())
    response = client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"messages": [], "max_tokens": 16},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "context overflow"


def test_stream_module_has_no_native_slot_reporter_fallback() -> None:
    assert not hasattr(stream, "_native_tool_liveness_reporter")
    assert not hasattr(stream, "_probe_native_tool_progress")
    assert not hasattr(stream, "_needs_native_tool_liveness_reporter")
