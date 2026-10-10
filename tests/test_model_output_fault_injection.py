from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "maxLength": 256}},
    "required": ["answer"],
    "additionalProperties": False,
}
MESSAGES = ({"role": "user", "content": "return an answer"},)


class ScriptedRouter:
    """Deterministic model transport used to drive the real fixed-template boundary."""

    def __init__(self, *events: object) -> None:
        self.events = list(events)
        self.calls = 0

    def generate_text(self, role, messages, **kwargs):
        del role, messages
        self.calls += 1
        assert kwargs["response_format"] == "json"
        assert kwargs["response_schema"] == SCHEMA
        if not self.events:
            raise AssertionError("scripted model output exhausted")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return str(event)


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        '{"answer":"truncated"',
        "",
        "null",
        "[]",
        '{"answer": 42}',
        '{}',
        '{"answer":"ok","hallucinated_field":true}',
    ],
    ids=[
        "malformed",
        "truncated",
        "empty",
        "null",
        "wrong-container",
        "wrong-type",
        "missing-required",
        "undeclared-field",
    ],
)
def test_hostile_text_model_outputs_are_rejected(raw: str) -> None:
    with pytest.raises(Exception):
        generate_fixed_template_value(
            ScriptedRouter(raw),
            "planner",
            MESSAGES,
            response_schema=SCHEMA,
            enable_tools=False,
        )


def test_valid_scripted_model_output_crosses_real_validation_boundary() -> None:
    router = ScriptedRouter('{"answer":"accepted"}')
    assert generate_fixed_template_value(
        router,
        "planner",
        MESSAGES,
        response_schema=SCHEMA,
        enable_tools=False,
    ) == {"answer": "accepted"}
    assert router.calls == 1


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("model timeout"), ConnectionError("backend disconnected"), RuntimeError("backend crashed")],
    ids=["timeout", "disconnect", "backend-error"],
)
def test_model_transport_failures_are_not_silently_converted_to_success(failure: Exception) -> None:
    with pytest.raises(type(failure), match=str(failure)):
        generate_fixed_template_value(
            ScriptedRouter(failure),
            "planner",
            MESSAGES,
            response_schema=SCHEMA,
            enable_tools=False,
        )


def test_undeclared_evidence_cannot_cross_fixed_host_schema() -> None:
    # The schema declares only 'answer'. Silently preserving evidence_refs
    # would grant an unvalidated model-owned extension host authority.
    router = ScriptedRouter(json.dumps({"answer": "ok", "evidence_refs": ["fixture:1"]}))
    with pytest.raises(Exception, match="evidence_refs"):
        generate_fixed_template_value(
            router,
            "planner",
            MESSAGES,
            response_schema=SCHEMA,
            enable_tools=False,
        )


def test_nested_undeclared_model_fields_are_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"answer": {"type": "string", "maxLength": 256}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }

    class Router:
        def generate_text(self, *_args, **_kwargs):
            return '{"payload":{"answer":"ok","injected":true}}'

    with pytest.raises(Exception, match="injected"):
        generate_fixed_template_value(
            Router(), "planner", MESSAGES,
            response_schema=schema, enable_tools=False,
        )


def test_wrapped_model_json_cannot_hide_extra_fields_during_recovery() -> None:
    # llama.cpp may wrap a valid JSON object with thinking prose. The recovery
    # parser may normalize its envelope, but cannot launder undeclared fields.
    router = ScriptedRouter(
        'analysis complete\\n{"answer":"ok","evidence_refs":["untrusted"]}'
    )
    with pytest.raises(Exception, match="evidence_refs"):
        generate_fixed_template_value(
            router, "planner", MESSAGES,
            response_schema=SCHEMA, enable_tools=False,
        )


def test_wrapped_valid_model_json_still_recovers() -> None:
    router = ScriptedRouter('preface\\n{"answer":"accepted"}')
    assert generate_fixed_template_value(
        router, "planner", MESSAGES,
        response_schema=SCHEMA, enable_tools=False,
    ) == {"answer": "accepted"}


def test_wrapped_valid_nested_json_is_checked_as_one_root() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"answer": {"type": "string", "maxLength": 64}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }
    class NestedRouter:
        def generate_text(self, *_args, **_kwargs):
            return 'preface\\n{"payload":{"answer":"valid"}}'

    assert generate_fixed_template_value(
        NestedRouter(), "planner", MESSAGES,
        response_schema=schema, enable_tools=False,
    ) == {"payload": {"answer": "valid"}}
