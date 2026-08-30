from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_completion_liveness_contract as contract
from minecraft_mod_ai.model_adapters import llama_cpp_adapter


def test_current_slot_shape_reads_next_token_decode_progress() -> None:
    snapshot = contract._slot_progress_from_payload(
        [
            {
                "is_processing": True,
                "n_prompt_tokens": 8192,
                "n_prompt_tokens_processed": 6144,
                "next_token": {"n_decoded": 17},
            }
        ]
    )

    assert snapshot == {
        "processing_slots": 1,
        "decoded": 17,
        "prompt_processed": 6144,
    }


def test_legacy_slot_shape_remains_observable() -> None:
    snapshot = contract._slot_progress_from_payload(
        [
            {
                "is_processing": True,
                "n_prompt_tokens": 2048,
                "n_decoded": 9,
            }
        ]
    )

    assert snapshot == {
        "processing_slots": 1,
        "decoded": 9,
        "prompt_processed": 2048,
    }


def test_progress_payload_requests_prompt_events_and_bounded_ping() -> None:
    stream_module = SimpleNamespace(
        _tool_idle_timeout_seconds=lambda: 120.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )
    original = {"model": "local", "messages": [], "tools": [{"type": "function"}]}

    result = contract._progress_aware_payload(stream_module, original)

    assert result is not original
    assert "return_progress" not in original
    assert result["return_progress"] is True
    assert result["sse_ping_interval"] == 30


def test_install_wraps_nonstream_chat_completion_without_changing_timeout() -> None:
    calls: list[tuple[str, dict]] = []

    class FakeClient:
        def post(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return "ok"

    stream_module = SimpleNamespace(
        _StreamingCompletionClient=FakeClient,
        _tool_idle_timeout_seconds=lambda: 12.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
        _slot_progress_from_payload=lambda payload: None,
    )

    contract.install(stream_module)
    client = FakeClient()
    timeout = object()
    result = client.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={"messages": [], "tools": [{"type": "function"}]},
        timeout=timeout,
    )

    assert result == "ok"
    assert calls[0][1]["timeout"] is timeout
    assert calls[0][1]["json"]["return_progress"] is True
    assert calls[0][1]["json"]["sse_ping_interval"] == 4
    assert stream_module._slot_progress_from_payload is contract._slot_progress_from_payload


def test_semantic_progress_disables_native_slot_reporter() -> None:
    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract

    assert (
        stream_contract._needs_native_tool_liveness_reporter(
            {"tools": [{"type": "function"}], "return_progress": True}
        )
        is False
    )


def test_native_slot_reporter_remains_fallback_without_semantic_progress() -> None:
    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract

    assert (
        stream_contract._needs_native_tool_liveness_reporter(
            {"tools": [{"type": "function"}]}
        )
        is True
    )
    assert stream_contract._needs_native_tool_liveness_reporter({"messages": []}) is False


def test_liveness_install_has_no_reporter_monkeypatch_dependency() -> None:
    calls: list[tuple[str, dict]] = []

    class FakeClient:
        def __init__(self, _client=None):
            self._client = _client

        def post(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return "ok"

        def stream(self, method: str, url: str, **kwargs):
            return (method, url, kwargs)

    stream_module = SimpleNamespace(
        _StreamingCompletionClient=FakeClient,
        _CLIENTS={},
        _tool_idle_timeout_seconds=lambda: 12.0,
        _stream_idle_timeout_seconds=lambda: 120.0,
    )

    contract.install(stream_module)

    assert not hasattr(stream_module, "_native_tool_liveness_reporter")
    assert stream_module._slot_progress_from_payload is contract._slot_progress_from_payload


def test_runtime_completion_transport_has_one_progress_aware_owner() -> None:
    assert getattr(
        llama_cpp_adapter._post_completion,
        "_mmm_single_progress_aware_completion_owner_v1",
        False,
    )
