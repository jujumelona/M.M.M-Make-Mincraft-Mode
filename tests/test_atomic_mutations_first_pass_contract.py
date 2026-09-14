from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse, ToolCall
from minecraft_mod_ai.native_atomic_argument_recovery import host_selected_argument_turn
from minecraft_mod_ai.task_template_catalog import load_record_template


_IDENTIFIER = "feature/algorithm/atomic_mutations"
_TOOL_NAME = "submit_one_feature_algorithm_atomic_mutations"


def _tool(schema: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Fill the canonical atomic-mutation record exactly once.",
            "parameters": schema,
        },
    }


def test_atomic_mutations_schema_rejects_logged_context_echo_shape() -> None:
    schema = load_record_template(_IDENTIFIER)["record_schema"]
    echoed_context = json.dumps(
        {
            "requirement_id": "req_004",
            "requirement": (
                "Players can upgrade spacecraft performance and acquire new weapons "
                "or crew through purchase or trade."
            ),
            "criterion": (
                "Players can improve spacecraft stats or add new modules via "
                "in-game transactions."
            ),
        },
        ensure_ascii=False,
    )

    for mutations in (echoed_context, "Context copy: " + echoed_context):
        candidate = {
            "mutations": mutations,
            "commit": "Apply the validated purchase atomically after payment succeeds.",
            "rollback": "Restore the pre-transaction spacecraft state if commit fails.",
        }
        errors = list(Draft202012Validator(schema).iter_errors(candidate))
        assert errors
        assert tuple(errors[0].absolute_path) == ("mutations",)


def test_atomic_mutations_first_valid_generation_needs_no_repair_turn() -> None:
    template = load_record_template(_IDENTIFIER)
    schema = template["record_schema"]
    tool = _tool(schema)
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
    request = GenerationRequest(
        messages=(
            {
                "role": "system",
                "content": template["task"] + "\n" + "\n".join(template["rules"]),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        parallel_tool_calls=False,
    )
    expected = {
        "mutations": "Debit the purchase cost and apply the selected spacecraft upgrade in one transaction.",
        "commit": "Publish the upgraded spacecraft state only after both debit and upgrade succeed.",
        "rollback": "Restore the original balance and spacecraft state if either mutation fails.",
    }
    calls: list[GenerationRequest] = []

    def current(adapter, page_request: GenerationRequest) -> GenerationResponse:
        del adapter
        calls.append(page_request)
        parameters = page_request.tools[0]["function"]["parameters"]
        assert parameters["properties"]["mutations"]["description"].startswith(
            "Describe only the state change"
        )
        assert parameters["properties"]["mutations"]["pattern"] == r"^(?!.*[{}\[\]]).+$"
        assert "Never copy, serialize, quote, or embed that context" in str(
            page_request.messages[0]["content"]
        )
        raw = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_atomic_mutations_first_pass",
                    name=_TOOL_NAME,
                    arguments=dict(expected),
                    raw_arguments=raw,
                ),
            )
        )

    turn = host_selected_argument_turn(
        current,
        object(),
        request,
        _TOOL_NAME,
        prefix="test",
    )

    assert len(calls) == 1
    assert "previous forced-call arguments were invalid" not in str(
        calls[0].messages[-1]["content"]
    ).lower()
    assert turn.tool_calls[0].arguments == expected
