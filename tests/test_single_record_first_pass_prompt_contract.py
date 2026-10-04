from __future__ import annotations

import json
from typing import Any

from minecraft_mod_ai.single_record_template import run_single_record_template


_IDENTIFIER = "feature/algorithm/atomic_mutations"
_CONTEXT = {
    "requirement_id": "req_004",
    "requirement": (
        "Players can upgrade spacecraft performance and acquire new weapons "
        "or crew through purchase or trade."
    ),
    "criterion": (
        "Players can improve spacecraft stats or add new modules via "
        "in-game transactions."
    ),
    "evidence": [],
    "accepted_records": [],
    "record_index": 0,
    "record_ordinal": 1,
    "record_count": 1,
}
_EXPECTED = {
    "mutations": "Debit the purchase cost and apply the selected spacecraft upgrade atomically.",
    "commit": "Publish the upgrade only after the debit and state mutation both succeed.",
    "rollback": "Restore the original balance and spacecraft state if commit cannot complete.",
}
_FIELDS = tuple(_EXPECTED)


def _assert_field_contract(messages, schema, tool_name: str, index: int) -> str:
    expected_field = _FIELDS[index]
    assert tool_name == (
        "submit_one_feature_algorithm_atomic_mutations"
        f"_part_{index + 1}_of_{len(_FIELDS)}"
    )
    assert len(messages) == 2 + index
    context = json.loads(str(messages[1]["content"]))
    assert context["requirement_id"] == "req_004"
    assert all(
        "CURRENT_FIXED_OUTPUT_FIELDS" not in str(message.get("content", ""))
        for message in messages
    )
    assert tuple(schema["properties"]) == (expected_field,)
    assert schema["required"] == [expected_field]
    assert schema["additionalProperties"] is False
    if expected_field == "mutations":
        assert schema["properties"][expected_field]["description"].startswith(
            "Describe only the state change"
        )
    if index:
        accepted = str(messages[-1]["content"])
        for prior in _FIELDS[:index]:
            assert prior in accepted
    return expected_field


def test_single_record_is_host_paged_by_required_field() -> None:
    calls: list[dict[str, Any]] = []

    def generator(router, role, messages, **kwargs):
        del router
        index = len(calls)
        field = _assert_field_contract(
            messages,
            kwargs["response_schema"],
            kwargs["tool_name"],
            index,
        )
        assert role == "planner"
        assert int(kwargs["output_token_ceiling"]) > 0
        calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "response_schema": kwargs["response_schema"],
                "tool_name": kwargs["tool_name"],
                "output_token_ceiling": kwargs["output_token_ceiling"],
            }
        )
        return {field: _EXPECTED[field]}

    result = run_single_record_template(
        object(),
        _IDENTIFIER,
        context=dict(_CONTEXT),
        generator=generator,
    )

    assert len(calls) == len(_FIELDS)
    assert result == _EXPECTED


class _PlannerJsonRouter:
    def __init__(self) -> None:
        self.tool_calls = 0
        self.text_calls: list[dict[str, Any]] = []

    def generate_text(self, role, messages, **kwargs):
        index = len(self.text_calls)
        field = _FIELDS[index]
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        assert kwargs["force_non_thinking"] is True
        schema = kwargs["response_schema"]
        assert tuple(schema["properties"]) == (field,)
        assert int(kwargs["output_token_ceiling"]) > 0
        self.text_calls.append(
            {
                "messages": tuple(messages),
                "schema": schema,
                "output_token_ceiling": kwargs["output_token_ceiling"],
            }
        )
        return json.dumps({field: _EXPECTED[field]})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner single-record pages must never use native tools")


def test_atomic_mutations_uses_bounded_planner_json_pages_not_tools() -> None:
    router = _PlannerJsonRouter()

    result = run_single_record_template(
        router,
        _IDENTIFIER,
        context=dict(_CONTEXT),
    )

    assert result == _EXPECTED
    assert len(router.text_calls) == len(_FIELDS)
    assert router.tool_calls == 0
