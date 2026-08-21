from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import GenerationRequest, ModelBackendError
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "steps": {"type": "array"},
    },
    "required": ["plan", "steps"],
    "additionalProperties": False,
}


def _validate(output: str, request: GenerationRequest) -> str:
    return validate_structured_output(
        output,
        response_format=request.response_format,
        response_schema=request.response_schema,
    )


def test_malformed_json_is_rejected_instead_of_regenerated() -> None:
    malformed = '{"plan":"build"\n"steps":[]}'
    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
        response_schema=_SCHEMA,
    )

    with pytest.raises(StructuredOutputValidationError) as raised:
        _validate(malformed, request)

    assert raised.value.output == malformed
    assert raised.value.repair_attempts == 0


def test_schema_less_json_is_validated_as_any_json_root() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
    )

    assert _validate('"native-only"', request) == '"native-only"'
    assert _validate("null", request) == "null"


def test_schema_less_invalid_json_is_not_silently_accepted() -> None:
    request = GenerationRequest(
        messages=({"role": "user", "content": "json only"},),
        response_format="json",
    )

    with pytest.raises(StructuredOutputValidationError):
        _validate("native-only", request)


def test_transport_failures_remain_backend_failures_before_validation() -> None:
    backend_failure = ModelBackendError(
        role="planner",
        model_id="local-test",
        cause=RuntimeError("llama server returned malformed SSE JSON"),
    )

    with pytest.raises(ModelBackendError) as raised:
        raise backend_failure

    assert raised.value is backend_failure
