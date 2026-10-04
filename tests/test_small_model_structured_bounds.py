from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.llama_server_hardware_policy import _enforce_required_tool_sampling
from minecraft_mod_ai.llama_stream_efficiency_contract import (
    _repetitive_tail,
    _required_tool_repetition_detected,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS, record_field_schema
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    worksheet_chunk_schema,
)


def test_authored_schema_has_host_fixed_cardinality_and_one_field_bounds() -> None:
    section = next(iter(DETAIL_RECORDS))
    page = pack_section_concerns(section)[0]
    concern = str(page[0])
    field = page.field_projection[concern][0]

    logical_field_schema = record_field_schema(section, concern, field)
    if logical_field_schema.get("type") == "string":
        assert logical_field_schema["maxLength"] <= 512

    chunk = worksheet_chunk_schema(
        section,
        page,
        record_counts={concern: 2},
    )
    assert set(chunk["properties"]) == {concern}
    assert chunk["required"] == [concern]
    assert chunk["properties"][concern]["minItems"] == 2
    assert chunk["properties"][concern]["maxItems"] == 2
    item = chunk["properties"][concern]["items"]
    assert tuple(item["properties"]) == (field,)
    assert item["required"] == [field]


def test_required_tool_sampling_keeps_native_repeat_penalty() -> None:
    payload = {
        "tool_choice": "required",
        "temperature": 0.7,
        "repeat_penalty": 1.0,
        "top_p": 0.8,
    }
    out = _enforce_required_tool_sampling(payload)
    assert out["temperature"] == 0.0
    assert out["repeat_penalty"] == 1.05
    assert "top_p" not in out


def test_exact_repetition_guard_detects_tool_argument_loop() -> None:
    unit = '{"state":"same","effect":"repeat"},'
    repeated = unit * 8
    assert _repetitive_tail(repeated)
    assert _required_tool_repetition_detected(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "author_probe",
                        "arguments": repeated,
                    }
                }
            ]
        }
    )


def test_exact_repetition_guard_does_not_flag_normal_structured_text() -> None:
    text = "".join(f'{{"index":{index},"value":"v{index}"}},' for index in range(40))
    assert not _repetitive_tail(text)


class _Registry:
    @staticmethod
    def role(profile: str, role: str):
        del profile, role
        return SimpleNamespace(adapter="llama_cpp")


class _PlannerJsonRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls = 0
        self.tool_calls = 0

    def generate_text(self, role, messages, **kwargs):
        del messages
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        assert kwargs["force_non_thinking"] is True
        self.text_calls += 1
        return json.dumps({"left": "value-left", "right": "value-right"})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner fixed templates must not use native tool transport")


def test_planner_structured_template_uses_schema_json_without_tool_repair() -> None:
    router = _PlannerJsonRouter()
    result = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "fill bounded record"},),
        response_schema={
            "type": "object",
            "properties": {
                "left": {"type": "string", "minLength": 1, "maxLength": 64},
                "right": {"type": "string", "minLength": 1, "maxLength": 64},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        enable_tools=False,
        tool_name="planner_bounded_probe",
    )

    assert result == {"left": "value-left", "right": "value-right"}
    assert router.text_calls == 1
    assert router.tool_calls == 0
