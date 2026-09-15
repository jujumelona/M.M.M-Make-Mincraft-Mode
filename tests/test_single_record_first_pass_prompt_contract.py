from __future__ import annotations

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


def _assert_native_contract(messages, schema, tool_name: str) -> None:
    assert tool_name == "submit_one_feature_algorithm_atomic_mutations"
    assert len(messages) == 2
    assert str(messages[1]["content"]).startswith("READ_ONLY_INPUT_CONTEXT:\n")
    assert "req_004" in str(messages[1]["content"])
    assert all(
        "CURRENT_FIXED_OUTPUT_FIELDS" not in str(message.get("content", ""))
        for message in messages
    )
    assert tuple(schema["properties"]) == ("mutations", "commit", "rollback")
    assert schema["properties"]["mutations"]["description"].startswith(
        "Describe only the state change"
    )
    assert schema["additionalProperties"] is False


def test_single_record_prompt_does_not_duplicate_native_tool_schema() -> None:
    calls: list[dict[str, Any]] = []

    def generator(router, role, messages, **kwargs):
        del router
        calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "response_schema": kwargs["response_schema"],
                "tool_name": kwargs["tool_name"],
            }
        )
        assert role == "planner"
        _assert_native_contract(
            messages,
            kwargs["response_schema"],
            kwargs["tool_name"],
        )
        return dict(_EXPECTED)

    result = run_single_record_template(
        object(),
        _IDENTIFIER,
        context=dict(_CONTEXT),
        generator=generator,
    )

    assert len(calls) == 1
    assert result == _EXPECTED


class _NativeToolRouter:
    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []
        self.text_calls = 0

    def generate_text(self, *args, **kwargs):
        import json

        self.text_calls += 1
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        return json.dumps(_EXPECTED)

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, description):
        self.tool_calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        _assert_native_contract(messages, parameters, tool_name)
        return dict(_EXPECTED)


def test_atomic_mutations_first_pass_writes_design_without_a_tool_call() -> None:
    router = _NativeToolRouter()

    result = run_single_record_template(
        router,
        _IDENTIFIER,
        context=dict(_CONTEXT),
    )

    assert result == _EXPECTED
    assert router.text_calls == 1
    assert router.tool_calls == []
