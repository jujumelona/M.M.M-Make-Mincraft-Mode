from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.llama_server_hardware_policy import _enforce_required_tool_sampling
from minecraft_mod_ai.llama_stream_efficiency_contract import (
    _repetitive_tail,
    _required_tool_repetition_detected,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS, record_field_schema
from minecraft_mod_ai.worksheet_atomic_chunker import worksheet_chunk_schema


def test_authored_schema_has_hard_cardinality_and_text_bounds() -> None:
    section = next(iter(DETAIL_RECORDS))
    concern = next(iter(DETAIL_RECORDS[section]))
    field = DETAIL_RECORDS[section][concern].split()[0]

    field_schema = record_field_schema(section, concern, field)
    if field_schema.get("type") == "string":
        assert field_schema["maxLength"] <= 512

    chunk = worksheet_chunk_schema(section, (concern,), include_evidence=True)
    assert chunk["properties"][concern]["maxItems"] == 4
    assert chunk["properties"]["inapplicable_concerns"]["maxItems"] == 1
    assert chunk["properties"]["constraint_evidence_refs"]["maxItems"] == 8


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


class _PlannerRepairRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.calls = 0

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
        output_token_ceiling=None,
        force_non_thinking=False,
    ):
        del role, messages, tool_name, description, output_token_ceiling, force_non_thinking
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated bounded tool repetition")
        field = next(iter(parameters["properties"]))
        return {field: f"value-{field}"}


def test_planner_structured_failure_recovers_by_top_level_field() -> None:
    router = _PlannerRepairRouter()
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
    assert router.calls == 3
