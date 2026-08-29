from __future__ import annotations

import json
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

_RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "research_note": {
            "type": "object",
            "properties": {
                "domain_id": {"type": "string"},
                "claims": {"type": "array", "items": {}},
                "gaps": {"type": "array", "items": {}},
                "next_queries": {"type": "array", "items": {}},
                "sufficient": {"type": "boolean"},
                "procedures": {"type": "array", "items": {}},
            },
            "additionalProperties": True,
        }
    },
    "additionalProperties": True,
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


def test_invalid_json_console_log_contains_exact_raw_output_and_location(capsys) -> None:
    output = '{"name":"VISIBLE_RAW_FAILURE"\n"count":1}'

    with pytest.raises(StructuredOutputValidationError):
        _validate(output)

    logged = capsys.readouterr().out
    assert "MODEL STRUCTURED OUTPUT FAILURE:" in logged
    assert "VISIBLE_RAW_FAILURE" in logged
    assert '"output_chars"' in logged
    assert '"output_sha256"' in logged
    assert "invalid JSON at line" in logged
    assert '"response_schema"' in logged


def test_schema_failure_console_log_contains_exact_path_schema_and_raw_output(capsys) -> None:
    output = '{"name":"VISIBLE_SCHEMA_FAILURE","count":"wrong"}'

    with pytest.raises(StructuredOutputValidationError):
        _validate(output)

    logged = capsys.readouterr().out
    assert "MODEL STRUCTURED OUTPUT FAILURE:" in logged
    assert "VISIBLE_SCHEMA_FAILURE" in logged
    assert "$[\\\"count\\\"]" in logged
    assert "is not of type 'integer'" in logged
    assert '"required": ["name", "count"]' in logged


def test_parser_owned_research_schema_accepts_prose_wrapped_json_for_host_parser(capsys) -> None:
    output = (
        "연구 결과입니다.\n"
        '{"research_note":{"domain_id":"request","claims":["grounded"],'
        '"gaps":[],"next_queries":[],"sufficient":true}}\n끝'
    )
    request = _request(schema=_RESEARCH_SCHEMA)

    assert _validate(output, request) == output
    logged = capsys.readouterr().out
    assert "MODEL STRUCTURED OUTPUT RECOVERED:" in logged
    assert "parser_owned_embedded_json_recovery" in logged
    assert "grounded" in logged
    assert "MODEL STRUCTURED OUTPUT FAILURE:" not in logged


def test_parser_owned_research_rejects_sufficient_true_without_claims(capsys) -> None:
    output = (
        '{"research_note":{"domain_id":"request","claims":[],"gaps":[],'
        '"next_queries":[],"sufficient":true}}'
    )
    request = _request(schema=_RESEARCH_SCHEMA)

    with pytest.raises(StructuredOutputValidationError) as raised:
        _validate(output, request)

    assert any("sufficient=true" in error for error in raised.value.errors)
    logged = capsys.readouterr().out
    prefix = "MODEL STRUCTURED OUTPUT FAILURE: "
    assert logged.startswith(prefix)
    diagnostic = json.loads(logged[len(prefix) :].strip())
    assert diagnostic["event"] == "structured_output_validation_failure"
    assert "requires at least one non-empty claim" in " ".join(diagnostic["errors"])
    assert json.loads(diagnostic["output"])["research_note"]["claims"] == []


def test_parser_owned_research_allows_explicit_insufficient_gap_receipt() -> None:
    output = (
        '{"research_note":{"domain_id":"request","claims":[],'
        '"gaps":["missing evidence"],"next_queries":["more evidence"],'
        '"sufficient":false}}'
    )
    request = _request(schema=_RESEARCH_SCHEMA)

    assert _validate(output, request) == output


def test_strict_schema_still_rejects_prose_wrapped_json() -> None:
    output = 'prefix {"name":"ok","count":1} suffix'

    with pytest.raises(StructuredOutputValidationError):
        _validate(output)


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
