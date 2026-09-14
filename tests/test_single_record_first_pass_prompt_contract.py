from __future__ import annotations

from typing import Any

from minecraft_mod_ai.single_record_template import run_single_record_template


_IDENTIFIER = "feature/algorithm/atomic_mutations"


def test_single_record_first_pass_prompt_is_derived_from_canonical_schema() -> None:
    context = {
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
    expected = {
        "mutations": "Debit the purchase cost and apply the selected spacecraft upgrade atomically.",
        "commit": "Publish the upgrade only after the debit and state mutation both succeed.",
        "rollback": "Restore the original balance and spacecraft state if commit cannot complete.",
    }
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
        assert kwargs["tool_name"] == "submit_one_feature_algorithm_atomic_mutations"
        assert str(messages[1]["content"]).startswith("READ_ONLY_INPUT_CONTEXT:\n")
        assert "req_004" in str(messages[1]["content"])

        contract = str(messages[2]["content"])
        assert "CURRENT_FIXED_OUTPUT_FIELDS" in contract
        assert "READ_ONLY_INPUT_CONTEXT, not an output example" in contract
        assert "- mutations (string): Describe only the state change" in contract
        assert "- commit (string): Describe only the commit boundary" in contract
        assert "- rollback (string): Describe only the rollback action" in contract
        assert "Do not substitute the input object" in contract

        schema = kwargs["response_schema"]
        assert tuple(schema["properties"]) == ("mutations", "commit", "rollback")
        assert schema["properties"]["mutations"]["description"].startswith(
            "Describe only the state change"
        )
        assert schema["additionalProperties"] is False
        return dict(expected)

    result = run_single_record_template(
        object(),
        _IDENTIFIER,
        context=context,
        generator=generator,
    )

    assert len(calls) == 1
    assert result == expected
