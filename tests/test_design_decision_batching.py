"""Regression: large independent design slot sets must not cause per-slot LLM calls."""
from minecraft_mod_ai import design_record_runtime as runtime


def test_decision_value_pages_preserve_every_slot_and_reduce_calls(monkeypatch):
    selected = [f"mechanic_{index:02d}" for index in range(24)]
    calls = []
    saved = {}

    monkeypatch.setattr(
        runtime, "load_record_template",
        lambda _name: {
            "id": "design/decision",
            "record_schema": {
                "type": "object",
                "properties": {"slot_id": {"type": "string"},
                               "value": {"type": "string", "minLength": 1, "maxLength": 256}},
                "required": ["slot_id", "value"], "additionalProperties": False,
            },
        },
    )
    monkeypatch.setattr(runtime, "task_context", lambda _template, context: dict(context))
    monkeypatch.setattr(
        runtime, "task_binding",
        lambda _template, context, _refs: ",".join(context["allowed_slots"]),
    )
    monkeypatch.setattr(runtime, "serialized_callback", lambda callback: callback)

    def fake_generate(_router, _role, _messages, *, response_schema, **_kw):
        names = list(response_schema["required"])
        calls.append(names)
        if response_schema["properties"][names[0]]["type"] == "boolean":
            return {name: True for name in names}
        return {name: "implementation:" + name for name in names}

    monkeypatch.setattr(runtime, "generate_fixed_template_value", fake_generate)
    result = runtime.run_record_template(
        object(), "design/decision",
        context={"requirement_id": "req", "requirement": "All mechanics matter",
                 "allowed_slots": selected},
        checkpoint=lambda key, value: saved.setdefault(key, value),
    )
    assert [row["slot_id"] for row in result["records"]] == selected
    assert [row["value"] for row in result["records"]] == [
        "implementation:" + name for name in selected
    ]
    assert [len(page) for page in calls] == [24, 8, 8, 8]
    assert len(saved) == 4
