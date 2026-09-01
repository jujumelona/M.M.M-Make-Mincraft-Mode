from __future__ import annotations

import pytest

from minecraft_mod_ai import llama_exact_context
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
    completion_boundary_error,
)
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


class _Response:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _stub_exact_context_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda _url, payload: dict(payload),
    )


def _adapter(*, extra=None) -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="test/model",
            max_new_tokens=8192,
            extra=extra or {},
        )
    )


def _boundary(
    kind: str,
    content: str = "",
    *,
    tool_calls=None,
    prompt_tokens: int = 100,
):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return LlamaCompletionBoundaryError(
        "typed boundary",
        kind=kind,
        partial_message=message,
        prompt_tokens=prompt_tokens,
        completion_tokens=8192 if kind == OUTPUT_EXHAUSTED else 50,
        max_tokens=8192,
    )


def test_progressive_prefill_continues_beyond_two_exhausted_pages(monkeypatch) -> None:
    calls = []
    outcomes = [
        _boundary(OUTPUT_EXHAUSTED, "A"),
        _boundary(OUTPUT_EXHAUSTED, "B"),
        _boundary(OUTPUT_EXHAUSTED, "C"),
        {"role": "assistant", "content": "D"},
    ]

    def complete(_url, payload):
        calls.append(payload)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    result = adapter_module._completion_message_with_prefill(
        _adapter(),
        "http://local",
        {"messages": [{"role": "user", "content": "go"}], "max_tokens": 8192},
    )

    assert result["content"] == "ABCD"
    assert len(calls) == 4
    assert calls[1]["messages"][-1] == {"role": "assistant", "content": "A"}
    assert calls[2]["messages"][-1]["content"] == "AB"
    assert calls[3]["messages"][-1]["content"] == "ABC"


def test_prefill_concatenates_repeated_java_xml_boundary_bytes_losslessly(monkeypatch) -> None:
    outcomes = [
        _boundary(OUTPUT_EXHAUSTED, "value</parameter>"),
        {"role": "assistant", "content": "</parameter></function>"},
    ]

    def complete(_url, _payload):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    result = adapter_module._completion_message_with_prefill(
        _adapter(),
        "http://local",
        {"messages": [{"role": "user", "content": "go"}]},
    )

    assert result["content"] == "value</parameter></parameter></function>"


def test_two_full_attachment_pages_complete_one_tool_before_parse(monkeypatch) -> None:
    attachment_prompt = "attachment evidence\n" + ("p" * 39_500)
    first_value_page = "x" * 32_000
    second_value_page = "y" * 32_000
    injected_generation_prompt = "<think>\n\n</think>\n\n"
    request = GenerationRequest(
        messages=({"role": "user", "content": attachment_prompt},),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                        },
                        "required": ["path", "old", "new"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        parallel_tool_calls=False,
    )
    payloads = []
    outcomes = [
        _boundary(
            OUTPUT_EXHAUSTED,
            (
                "<tool_call>\n<function=apply_source_edit>\n"
                "<parameter=path>\nsrc/main/java/A.java\n</parameter>\n"
                "<parameter=old>\nold\n</parameter>\n"
                "<parameter=new>\n"
                + first_value_page
            ),
            prompt_tokens=12959,
        ),
        _boundary(
            OUTPUT_EXHAUSTED,
            injected_generation_prompt + second_value_page,
            prompt_tokens=21151,
        ),
        {
            "role": "assistant",
            "content": (
                injected_generation_prompt
                + "\n</parameter>\n</function>\n</tool_call>"
            ),
        },
    ]

    def complete(_url, payload):
        payloads.append(payload)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    parses = []
    original_parser = adapter_module._qwen_tool_generation_response

    def parse(message, parsed_request):
        parses.append(dict(message))
        return original_parser(message, parsed_request)

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        lambda _url, _payload: injected_generation_prompt,
    )
    monkeypatch.setattr(adapter_module, "_qwen_tool_generation_response", parse)
    from minecraft_mod_ai import llama_stream_efficiency_contract

    monkeypatch.setattr(
        llama_stream_efficiency_contract,
        "_report_server_connection",
        lambda _url: None,
    )

    qwen35_extra = {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
        "agent_thinking": True,
        "decode_hotpath": "t4_mtp",
        "request_policy": "task_aware_sampling",
        "sampling_profiles": {
            "non_thinking": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repeat_penalty": 1.0,
            }
        },
    }
    turn = adapter_module._tool_semantic_completion(
        _adapter(extra=qwen35_extra), "http://local", request
    )

    assert len(parses) == 1
    assert len(payloads) == 3
    assert len(payloads[0]["messages"][0]["content"]) > 39_500
    assert [len(payload["messages"]) for payload in payloads] == [1, 2, 2]
    wire_tools = payloads[0]["tools"]
    for payload in payloads:
        assert payload["messages"][0]["content"] == attachment_prompt
        assert payload["tools"] == wire_tools
        assert payload["tool_choice"] == "required"
        assert payload["parallel_tool_calls"] is False
        assert payload["max_tokens"] == 4096
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "reasoning_effort" not in payload
    assert payloads[1]["messages"][-1]["content"].endswith(first_value_page)
    assert payloads[2]["messages"][-1]["content"].endswith(
        first_value_page + second_value_page
    )
    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]
    assert turn.tool_calls[0].arguments == {
        "path": "src/main/java/A.java",
        "old": "old",
        "new": first_value_page + second_value_page,
    }


def test_live_zero_token_calibration_returns_only_template_owned_bytes(monkeypatch) -> None:
    sent = []
    generation_prompt = "<think>\n\n</think>\n\n"

    def post(_url, payload):
        sent.append(payload)
        return _Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": generation_prompt},
                    }
                ],
                "usage": {"prompt_tokens": 25, "completion_tokens": 0},
            }
        )

    monkeypatch.setattr(adapter_module, "_post_completion", post)
    original = {
        "model": "local",
        "messages": [{"role": "user", "content": "SECRET USER SOURCE"}],
        "max_tokens": 8192,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
        "tools": [{"type": "function", "function": {"name": "edit"}}],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }

    assert (
        adapter_module._calibrate_assistant_prefill_generation_prompt(
            "http://local", original
        )
        == generation_prompt
    )
    assert len(sent) == 1
    assert sent[0]["max_tokens"] == 0
    assert sent[0]["chat_template_kwargs"] == original["chat_template_kwargs"]
    assert sent[0]["tools"] == original["tools"]
    assert sent[0]["tool_choice"] == "required"
    assert sent[0]["messages"][-1] == {
        "role": "assistant",
        "content": adapter_module._PREFILL_CALIBRATION_SENTINEL,
    }
    assert "SECRET USER SOURCE" not in str(sent[0])


@pytest.mark.parametrize(
    "message,completion_tokens",
    [
        ({"role": "assistant", "content": ""}, 0),
        (
            {
                "role": "assistant",
                "content": adapter_module._PREFILL_CALIBRATION_SENTINEL,
            },
            0,
        ),
        ({"role": "assistant", "content": "prefix", "reasoning": "private"}, 0),
        ({"role": "assistant", "content": "prefix"}, 1),
    ],
)
def test_ambiguous_prefill_calibration_fails_closed(
    monkeypatch, message, completion_tokens
) -> None:
    monkeypatch.setattr(
        adapter_module,
        "_post_completion",
        lambda _url, _payload: _Response(
            {
                "choices": [{"message": message}],
                "usage": {"completion_tokens": completion_tokens},
            }
        ),
    )

    with pytest.raises(RuntimeError, match="assistant-prefill calibration"):
        adapter_module._calibrate_assistant_prefill_generation_prompt(
            "http://local",
            {
                "messages": [{"role": "user", "content": "go"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )


def test_calibrated_strip_removes_one_template_prefix_and_preserves_the_next() -> None:
    prefix = "<think>\n\n</think>\n\n"
    normalized = adapter_module._normalize_assistant_prefill_suffix(
        {"role": "assistant", "content": prefix + prefix + "tail"},
        continuation_page=True,
        template_prefix=prefix,
    )

    assert normalized["content"] == prefix + "tail"


def test_calibration_failure_hands_partial_to_scalar_split_without_second_decode(
    monkeypatch,
) -> None:
    calls = 0

    def complete(_url, _payload):
        nonlocal calls
        calls += 1
        raise _boundary(OUTPUT_EXHAUSTED, "partial-action")

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        lambda _url, _payload: (_ for _ in ()).throw(RuntimeError("ambiguous")),
    )
    extra = {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
    }

    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        adapter_module._completion_message_with_prefill(
            _adapter(extra=extra),
            "http://local",
            {
                "messages": [{"role": "user", "content": "go"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    assert calls == 1
    assert caught.value.kind == OUTPUT_EXHAUSTED
    assert caught.value.partial_message["content"] == "partial-action"
    assert "calibration was unavailable" in str(caught.value)


def test_plain_structured_output_is_joined_before_semantic_parse(monkeypatch) -> None:
    outcomes = [
        _boundary(OUTPUT_EXHAUSTED, '{"game_'),
        {"role": "assistant", "content": 'design":{"title":"x"}}'},
    ]

    def complete(_url, _payload):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    result = adapter_module._completion_message_with_prefill(
        _adapter(),
        "http://local",
        {"messages": [{"role": "user", "content": "json"}]},
    )

    assert result["content"] == '{"game_design":{"title":"x"}}'


def test_empty_exhausted_suffix_is_typed_no_progress_not_a_retry_loop(monkeypatch) -> None:
    calls = 0

    def complete(_url, _payload):
        nonlocal calls
        calls += 1
        raise _boundary(OUTPUT_EXHAUSTED, "")

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        adapter_module._completion_message_with_prefill(
            _adapter(),
            "http://local",
            {"messages": [{"role": "user", "content": "go"}]},
        )

    assert calls == 1
    assert caught.value.kind == OUTPUT_EXHAUSTED
    assert caught.value.partial_bytes == 0
    assert "partial_bytes=0" in str(caught.value)


def test_context_pressure_after_partial_preserves_cumulative_typed_state(monkeypatch) -> None:
    outcomes = [
        _boundary(OUTPUT_EXHAUSTED, "abc"),
        _boundary(CONTEXT_PRESSURE, "def"),
    ]

    def complete(_url, _payload):
        outcome = outcomes.pop(0)
        raise outcome

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        adapter_module._completion_message_with_prefill(
            _adapter(),
            "http://local",
            {"messages": [{"role": "user", "content": "go"}]},
        )

    boundary = completion_boundary_error(caught.value)
    assert boundary is not None
    assert boundary.kind == CONTEXT_PRESSURE
    assert boundary.partial_message["content"] == "abcdef"
    assert boundary.partial_bytes > 0
    assert len(boundary.partial_sha256) == 64


def test_calibrated_qwen_context_rejection_preserves_prior_partial(monkeypatch) -> None:
    prefix = "<think>\n\n</think>\n\n"
    outcomes = [
        _boundary(OUTPUT_EXHAUSTED, "verified-partial"),
        _boundary(CONTEXT_PRESSURE, "", prompt_tokens=32768),
    ]

    def complete(_url, _payload):
        raise outcomes.pop(0)

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    monkeypatch.setattr(
        adapter_module,
        "_calibrate_assistant_prefill_generation_prompt",
        lambda _url, _payload: prefix,
    )
    extra = {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
    }

    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        adapter_module._completion_message_with_prefill(
            _adapter(extra=extra),
            "http://local",
            {
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    assert caught.value.kind == CONTEXT_PRESSURE
    assert caught.value.partial_message["content"] == "verified-partial"
    assert caught.value.prompt_tokens == 32768


def test_server_parsed_partial_tool_call_is_rejected_before_continuation(monkeypatch) -> None:
    calls = 0

    def complete(_url, _payload):
        nonlocal calls
        calls += 1
        raise _boundary(
            OUTPUT_EXHAUSTED,
            "",
            tool_calls=[{"function": {"name": "write", "arguments": {}}}],
        )

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    with pytest.raises(RuntimeError, match="partial tool actions are never executable"):
        adapter_module._completion_message_with_prefill(
            _adapter(),
            "http://local",
            {"messages": [{"role": "user", "content": "go"}]},
        )

    assert calls == 1


def test_qwen_prefill_capability_mismatch_fails_before_decode(monkeypatch) -> None:
    calls = 0

    def complete(_url, _payload):
        nonlocal calls
        calls += 1
        return {"content": "unexpected"}

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    extra = {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.8",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": True,
        "qwen_reasoning_effort": True,
        "qwen_assistant_prefill": False,
    }

    with pytest.raises(ValueError, match="does not match supported qwen3.8"):
        adapter_module._completion_message_with_prefill(
            _adapter(extra=extra),
            "http://local",
            {"messages": [{"role": "user", "content": "go"}]},
        )

    assert calls == 0


def test_qwen38_thinking_page_never_sends_invalid_assistant_prefill(monkeypatch) -> None:
    payloads = []

    def complete(_url, payload):
        payloads.append(payload)
        raise _boundary(OUTPUT_EXHAUSTED, "<think>unfinished")

    monkeypatch.setattr(adapter_module, "_completion_message", complete)
    extra = {
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.8",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": True,
        "qwen_reasoning_effort": True,
        "qwen_assistant_prefill": True,
    }
    payload = {
        "messages": [{"role": "user", "content": "plan"}],
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
        "max_tokens": 8192,
    }

    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        adapter_module._completion_message_with_prefill(
            _adapter(extra=extra), "http://local", payload
        )

    assert caught.value.kind == OUTPUT_EXHAUSTED
    assert caught.value.partial_message["content"] == "<think>unfinished"
    assert len(payloads) == 1
    assert payloads[0]["messages"][-1]["role"] == "user"
