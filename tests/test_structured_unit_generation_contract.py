from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_structured_decode_policy as decode_policy
from minecraft_mod_ai import structured_unit_generation_contract as units
from minecraft_mod_ai.structured_output import StructuredOutputValidationError


class _FakeRouter:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format=None,
        response_schema=None,
        enable_tools=True,
        **kwargs,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": tuple(media_paths),
                "response_format": response_format,
                "response_schema": response_schema,
                "enable_tools": enable_tools,
            }
        )
        return self.outputs.pop(0)


def test_control_jsonpath_is_not_accepted_as_string_content(monkeypatch):
    monkeypatch.setenv("MMM_PLANNER_TRACE", "0")
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")
    monkeypatch.setenv("MMM_STRUCTURED_UNIT_ATTEMPTS", "3")

    router = _FakeRouter(
        [
            '{"section":{"status":"$.section.modules[0].status"}}',
            '{"section":{"status":"ready"}}',
        ]
    )
    result = units._generate_section_units(
        router,
        prompt="make the requested mod",
        section_id="module_status",
        fields=["status"],
        properties={"status": {"type": "string", "minLength": 1}},
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result == {"status": "ready"}
    assert len(router.calls) == 2
    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    for call in router.calls:
        schema = call["response_schema"]
        assert schema["properties"]["section"]["required"] == ["status"]


def test_section_is_generated_one_top_level_field_per_request(monkeypatch):
    monkeypatch.setenv("MMM_PLANNER_TRACE", "0")
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")

    router = _FakeRouter(
        [
            '{"section":{"title":"Alien Planet"}}',
            '{"section":{"progression":["scan","adapt","escape"]}}',
        ]
    )
    result = units._generate_section_units(
        router,
        prompt="alien planet interaction mod",
        section_id="core",
        fields=["title", "progression"],
        properties={
            "title": {"type": "string", "minLength": 1},
            "progression": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result == {
        "title": "Alien Planet",
        "progression": ["scan", "adapt", "escape"],
    }
    required = [
        call["response_schema"]["properties"]["section"]["required"]
        for call in router.calls
    ]
    assert required == [["title"], ["progression"]]


def test_structured_adapter_validation_does_not_launch_hidden_repair():
    class Adapter:
        calls = 0

        def generate(self, request):
            type(self).calls += 1
            return '{"section":{"status":[]}}'

    fake_module = SimpleNamespace(LlamaCppAdapter=Adapter)
    decode_policy._bind_structured_generation_retry(fake_module)

    request = SimpleNamespace(
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {
                "section": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                }
            },
            "required": ["section"],
            "additionalProperties": False,
        },
        tools=(),
    )

    with pytest.raises(StructuredOutputValidationError):
        Adapter().generate(request)
    assert Adapter.calls == 1


def test_llama_payload_receives_host_json_schema():
    schema = {
        "type": "object",
        "properties": {"section": {"type": "object"}},
        "required": ["section"],
    }
    request = SimpleNamespace(response_format="json", response_schema=schema)
    payload: dict[str, object] = {}

    decode_policy._apply_llama_json_schema(payload, request)

    assert payload["response_format"] == {"type": "json_object"}
    assert payload["json_schema"] == schema
    assert payload["json_schema"] is not schema
