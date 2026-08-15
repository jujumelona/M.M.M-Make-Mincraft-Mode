from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters import GenerationRequest
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    generate_with_host_schema_repair,
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "keep": {"type": "integer"},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}


def _request(*, schema=_SCHEMA) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "ORIGINAL_TASK_SENTINEL"},),
        response_format="json",
        response_schema=schema,
    )


def test_llama_server_keeps_structured_validation_host_side() -> None:
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))
    payload = _server_payload(adapter, _request())

    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"


def test_valid_structured_output_is_returned_without_repair() -> None:
    calls: list[GenerationRequest] = []

    def generate(request: GenerationRequest) -> str:
        calls.append(request)
        return '{"name":"ok","count":1}'

    result = generate_with_host_schema_repair(_request(), generate)

    assert result == '{"name":"ok","count":1}'
    assert len(calls) == 1


def test_repair_edits_previous_invalid_output_with_exact_errors() -> None:
    outputs = iter(
        [
            '{"name":"ok","count":"wrong","keep":7}',
            '{"name":"ok","count":2,"keep":7}',
        ]
    )
    calls: list[GenerationRequest] = []

    def generate(request: GenerationRequest) -> str:
        calls.append(request)
        return next(outputs)

    result = generate_with_host_schema_repair(_request(), generate)

    assert result == '{"name":"ok","count":2,"keep":7}'
    assert len(calls) == 2
    repair = calls[1]
    repair_text = "\n".join(str(message.get("content", "")) for message in repair.messages)
    assert '"count":"wrong"' in repair_text
    assert "is not of type 'integer'" in repair_text
    assert "Previous invalid output" in repair_text
    assert "ORIGINAL_TASK_SENTINEL" not in repair_text
    assert repair.tools == ()
    assert repair.tool_choice is None


def test_each_repair_uses_latest_invalid_output_and_is_bounded() -> None:
    bad0 = '{"name":"ok","count":"zero"}'
    bad1 = '{"name":"ok","count":"one"}'
    bad2 = '{"name":"ok","count":"two"}'
    outputs = iter([bad0, bad1, bad2])
    calls: list[GenerationRequest] = []

    def generate(request: GenerationRequest) -> str:
        calls.append(request)
        return next(outputs)

    with pytest.raises(StructuredOutputValidationError) as raised:
        generate_with_host_schema_repair(_request(), generate)

    assert len(calls) == 3
    assert raised.value.repair_attempts == 2
    assert raised.value.output == bad2
    second_repair_text = "\n".join(
        str(message.get("content", "")) for message in calls[2].messages
    )
    assert bad1 in second_repair_text
    assert bad0 not in second_repair_text


def test_repair_transport_failure_is_not_retried_as_schema_failure() -> None:
    calls = 0

    def generate(request: GenerationRequest) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return '{"name":"ok","count":"wrong"}'
        raise RuntimeError("transport failed")

    with pytest.raises(RuntimeError, match="transport failed"):
        generate_with_host_schema_repair(_request(), generate)

    assert calls == 2


def test_non_schema_request_is_untouched() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "plain"},),
        response_format="text",
    )
    calls = 0

    def generate(value: GenerationRequest) -> str:
        nonlocal calls
        calls += 1
        assert value is request
        return "not json"

    assert generate_with_host_schema_repair(request, generate) == "not json"
    assert calls == 1


def test_invalid_schema_fails_before_model_generation() -> None:
    request = _request(schema={"type": "definitely-not-a-json-schema-type"})
    calls = 0

    def generate(value: GenerationRequest) -> str:
        nonlocal calls
        calls += 1
        return "{}"

    with pytest.raises(ValueError, match="invalid response_schema"):
        generate_with_host_schema_repair(request, generate)

    assert calls == 0


def test_runtime_does_not_install_blind_grammar_retry() -> None:
    from minecraft_mod_ai.model_adapters import llama_cpp_adapter

    assert not getattr(llama_cpp_adapter._post_completion, "_mmm_grammar_retry_v1", False)
