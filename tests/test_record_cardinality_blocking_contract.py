from minecraft_mod_ai.bounded_record_template import record_cardinality_response_schema
from minecraft_mod_ai.task_template_catalog import load_record_template


def test_prompt_parse_cardinality_cannot_be_semantically_blocked():
    template = load_record_template("prompt/parse")

    assert template["cardinality_blocking"] is False
    schema = record_cardinality_response_schema(template)

    assert schema["required"] == ["count"]
    assert "blocked_reason" not in schema["properties"]


def test_record_cardinality_blocking_remains_default_for_other_templates():
    schema = record_cardinality_response_schema({})

    assert schema["required"] == ["count", "blocked_reason"]
    assert "blocked_reason" in schema["properties"]
