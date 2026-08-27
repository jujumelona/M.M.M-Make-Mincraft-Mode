from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters import GenerationRequest
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
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


def _validate(output: str, request: GenerationRequest | None = None) -> str:
    value = request or _request()
    return validate_structured_output(
        output,
        response_format=value.response_format,
        response_schema=value.response_schema,
    )


def test_llama_server_keeps_schema_validation_host_owned() -> None:
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))
    payload = _server_payload(adapter, _request())

    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    assert payload["reasoning_effort"] == "none"


def test_valid_structured_output_is_returned_unchanged() -> None:
    output = '{"name":"ok","count":1}'
    assert _validate(output) == output


def test_schema_violation_fails_without_repair_or_coercion() -> None:
    output = '{"name":"ok","count":"wrong","keep":7}'

    with pytest.raises(StructuredOutputValidationError) as raised:
        _validate(output)

    assert raised.value.output == output
    assert raised.value.repair_attempts == 0
    assert any("is not of type 'integer'" in error for error in raised.value.errors)


def test_invalid_json_reports_location_and_preserves_original_output() -> None:
    output = '{"name":"ok"\n"count":1}'

    with pytest.raises(StructuredOutputValidationError) as raised:
        _validate(output)

    assert raised.value.output == output
    assert raised.value.repair_attempts == 0
    assert raised.value.errors[0].startswith("$: invalid JSON at line ")


def test_json_without_schema_is_still_strictly_validated() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
    )

    assert _validate("[1,2,3]", request) == "[1,2,3]"
    with pytest.raises(StructuredOutputValidationError):
        _validate("not json", request)


def test_non_json_request_is_untouched_without_schema() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "plain"},),
        response_format="text",
    )
    output = "not json"

    assert _validate(output, request) == output


def test_invalid_schema_fails_at_router_validation_boundary() -> None:
    request = _request(schema={"type": "definitely-not-a-json-schema-type"})

    with pytest.raises(ValueError, match="invalid response_schema"):
        _validate("{}", request)


def test_runtime_does_not_install_blind_grammar_retry() -> None:
    from minecraft_mod_ai.model_adapters import llama_cpp_adapter

    assert not getattr(llama_cpp_adapter._post_completion, "_mmm_grammar_retry_v1", False)
