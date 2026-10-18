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


def test_single_record_parts_cover_all_required_fields_without_model_loops() -> None:
    calls: list[dict[str, Any]] = []

    def generator(router, role, messages, **kwargs):
        del router
        schema = kwargs["response_schema"]
        assert role == "planner"
        assert kwargs["tool_name"].startswith("submit_one_feature_algorithm_atomic_mutations")
        fields = tuple(schema["required"])
        assert set(schema["properties"]) == set(fields)
        assert set(fields) <= set(_FIELDS)
        assert schema["additionalProperties"] is False
        assert int(kwargs["output_token_ceiling"]) > 0
        assert "CURRENT_FIXED_OUTPUT_FIELDS" not in str(messages)
        assert json.loads(messages[1]["content"])["requirement_id"] == "req_004"
        calls.append(dict(kwargs))
        return {field: _EXPECTED[field] for field in fields}

    result = run_single_record_template(
        object(), _IDENTIFIER, context=dict(_CONTEXT), generator=generator,
    )
    assert result == _EXPECTED
    assert len(calls) >= 1
    emitted = [field for call in calls for field in call["response_schema"]["required"]]
    assert sorted(emitted) == sorted(_FIELDS)


class _PlannerJsonRouter:
    def __init__(self) -> None:
        self.tool_calls = 0
        self.text_calls: list[dict[str, Any]] = []

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        assert kwargs["force_non_thinking"] is True
        schema = kwargs["response_schema"]
        assert set(schema["properties"]) <= set(_FIELDS)
        assert int(kwargs["output_token_ceiling"]) > 0
        self.text_calls.append({"messages": tuple(messages), "schema": schema})
        return json.dumps({field: _EXPECTED[field] for field in schema["required"]})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner bounded record pages must never use native tools")


def test_batched_record_uses_bounded_planner_json_not_tools() -> None:
    router = _PlannerJsonRouter()
    result = run_single_record_template(
        router, _IDENTIFIER, context=dict(_CONTEXT),
    )
    assert result == _EXPECTED
    assert router.text_calls
    assert sorted(field for call in router.text_calls
                  for field in call["schema"]["required"]) == sorted(_FIELDS)
    assert router.tool_calls == 0
