"""Regression: large independent design slot sets must not cause per-slot LLM calls."""
import pytest

from minecraft_mod_ai import design_record_runtime as runtime


@pytest.mark.parametrize("slot_count", [8, 24])
def test_decision_value_pages_preserve_every_slot_and_reduce_calls(monkeypatch, slot_count):
    selected = [f"mechanic_{index:02d}" for index in range(slot_count)]
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
    assert len(calls[0]) == slot_count  # applicability, never skip a requirement
    assert all(1 <= len(page) <= 8 for page in calls[1:])
    assert sum(len(page) for page in calls[1:]) == slot_count
    assert len(calls) < slot_count + 1  # fewer calls than the per-slot route
    assert len(saved) == len(calls)


def test_property_pages_keep_exact_schema_and_resume(monkeypatch):
    from minecraft_mod_ai import design_property_batching as props
    from minecraft_mod_ai.task_template_catalog import load_record_template

    template = load_record_template("design/content_property")
    requested = ["display_name", "stack_limit", "material", "main_color", "surface", "category"]
    schema_calls = []
    saved = {}

    def generate(_router, _role, _messages, *, response_schema, **_kwargs):
        schema_calls.append(response_schema)
        values = {}
        for name, rule in response_schema["properties"].items():
            values[name] = (rule["enum"][0] if "enum" in rule
                            else "1" if rule.get("pattern") else "stone")
        return values

    monkeypatch.setattr(props, "generate_fixed_template_value", generate)
    context = {"requirement": "A specific target-compatible content block",
               "allowed_properties": requested, "required_properties": requested}
    results = props.batched_property_records(
        object(), template, context, requested, saved,
        lambda key, value: saved.setdefault(key, value))
    assert [row["property"] for row in results] == requested
    assert len(schema_calls) < len(requested)
    assert all("property" not in schema["properties"] for schema in schema_calls)
    assert props.batched_property_records(
        object(), template, context, requested, saved, None) == results
    assert len(schema_calls) < len(requested)
