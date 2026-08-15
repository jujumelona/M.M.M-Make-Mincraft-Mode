from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.model_adapters import GenerationRequest, ModelBackendError
from minecraft_mod_ai.structured_output import generate_with_host_schema_repair


_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "steps": {"type": "array"},
    },
    "required": ["plan", "steps"],
    "additionalProperties": False,
}


def _decode_error(text: str) -> json.JSONDecodeError:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("test input must be malformed JSON")


def test_host_repair_recovers_json_rejected_by_early_stop_syntax_check() -> None:
    malformed = '{"plan":"build"\n"steps":[]}'
    calls: list[GenerationRequest] = []

    def generate(request: GenerationRequest) -> str:
        calls.append(request)
        if len(calls) == 1:
            raise ModelBackendError(
                role="planner",
                model_id="local-test",
                cause=_decode_error(malformed),
            )
        return '{"plan":"build","steps":[]}'

    request = GenerationRequest(
        messages=({"role": "user", "content": "ORIGINAL_TASK_SENTINEL"},),
        response_format="json",
        response_schema=_SCHEMA,
    )
    result = generate_with_host_schema_repair(request, generate)

    assert json.loads(result) == {"plan": "build", "steps": []}
    assert len(calls) == 2
    repair_text = "\n".join(
        str(message.get("content", "")) for message in calls[1].messages
    )
    assert malformed in repair_text
    assert "invalid JSON" in repair_text
    assert "ORIGINAL_TASK_SENTINEL" not in repair_text


def test_schema_less_json_requests_still_get_syntax_repair() -> None:
    outputs = iter(['{"plan":"build"\n"steps":[]}', '{"plan":"build","steps":[]}'])
    calls: list[GenerationRequest] = []

    def generate(request: GenerationRequest) -> str:
        calls.append(request)
        return next(outputs)

    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
    )
    result = generate_with_host_schema_repair(request, generate)

    assert json.loads(result) == {"plan": "build", "steps": []}
    assert len(calls) == 2
    assert calls[1].response_schema == {"type": "object"}


def test_transport_json_failure_is_not_misclassified_as_model_output() -> None:
    transport_failure = RuntimeError("llama server returned malformed SSE JSON")
    backend_failure = ModelBackendError(
        role="planner",
        model_id="local-test",
        cause=transport_failure,
    )
    calls = 0

    def generate(_request: GenerationRequest) -> str:
        nonlocal calls
        calls += 1
        raise backend_failure

    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
        response_schema=_SCHEMA,
    )

    with pytest.raises(ModelBackendError) as raised:
        generate_with_host_schema_repair(request, generate)

    assert raised.value is backend_failure
    assert calls == 1
