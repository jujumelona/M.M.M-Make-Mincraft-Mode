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
