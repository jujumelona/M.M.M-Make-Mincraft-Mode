from __future__ import annotations

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value


class _SchemaRepairRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_tool_decision(
        self,
        role: str,
        messages,
        *,
        tool_name: str,
        parameters,
        description: str,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        if len(self.calls) == 1:
            raise ValueError("Invalid tool arguments: required property statement is missing")
        return {"statement": "validated plan detail"}


class _ReturnedInvalidArgumentsRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_tool_decision(
        self,
        role: str,
        messages,
        *,
        tool_name: str,
        parameters,
        description: str,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        if len(self.calls) == 1:
            # Simulate an adapter that returns a mapping without enforcing the native
            # function schema itself. Host validation must still keep this inside repair.
            return {"payload": {}}
        return {"payload": {"statement": "validated nested plan detail"}}


def test_fixed_template_regenerates_after_schema_rejection() -> None:
    router = _SchemaRepairRouter()
    schema = {
        "type": "object",
        "properties": {"statement": {"type": "string"}},
        "required": ["statement"],
        "additionalProperties": False,
    }

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "fill the planning detail"},),
        response_schema=schema,
        tool_name="submit_planning_detail",
        enable_tools=False,
    )

    assert value == {"statement": "validated plan detail"}
    assert len(router.calls) == 2
    assert router.calls[0]["tool_name"] == "submit_planning_detail"
    assert router.calls[1]["tool_name"] == "submit_planning_detail_repair_1"
    repair_messages = router.calls[1]["messages"]
    assert "Host validation error" in repair_messages[-1]["content"]
    assert "required property statement is missing" in repair_messages[-1]["content"]


def test_fixed_template_repairs_mapping_rejected_by_host_nested_schema() -> None:
    router = _ReturnedInvalidArgumentsRouter()
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"statement": {"type": "string", "minLength": 1}},
                "required": ["statement"],
                "additionalProperties": False,
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "fill nested planning detail"},),
        response_schema=schema,
        tool_name="submit_nested_planning_detail",
        enable_tools=False,
    )

    assert value == {"payload": {"statement": "validated nested plan detail"}}
    assert len(router.calls) == 2
    assert router.calls[1]["tool_name"] == "submit_nested_planning_detail_repair_1"
    repair_messages = router.calls[1]["messages"]
    assert "Host validation error" in repair_messages[-1]["content"]
    assert "$.payload.statement" in repair_messages[-1]["content"]
