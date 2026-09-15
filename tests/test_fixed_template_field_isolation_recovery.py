from __future__ import annotations

from minecraft_mod_ai.fixed_template_generation import _generate_native_template_arguments


class _FailWholeObjectRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description,
    ):
        del role, messages, tool_name, description
        fields = tuple(parameters["properties"])
        self.calls.append(fields)

        if len(fields) > 1:
            raise RuntimeError("schema-invalid arguments at mappings")

        field = fields[0]
        values = {
            "mappings": "official named mappings",
            "constraint": "compatible with the selected loader and version",
        }
        return {field: values[field]}


def test_failed_multi_field_template_is_recovered_one_field_at_a_time() -> None:
    router = _FailWholeObjectRouter()
    schema = {
        "type": "object",
        "properties": {
            "mappings": {"type": "string", "minLength": 1, "maxLength": 256},
            "constraint": {"type": "string", "minLength": 1, "maxLength": 256},
        },
        "required": ["mappings", "constraint"],
        "additionalProperties": False,
    }

    value = _generate_native_template_arguments(
        router,
        "coder",
        ({"role": "user", "content": "compatibility fixture"},),
        tool_name="submit_one_feature_reuse_assessment_compatibility_part_2_of_2",
        parameters=schema,
        description="Fill the compatibility record slice.",
    )

    assert value == {
        "mappings": "official named mappings",
        "constraint": "compatible with the selected loader and version",
    }
    assert router.calls == [
        ("mappings", "constraint"),
        ("mappings",),
        ("constraint",),
    ]
