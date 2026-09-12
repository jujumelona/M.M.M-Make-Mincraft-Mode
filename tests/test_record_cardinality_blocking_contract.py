from minecraft_mod_ai.bounded_record_template import record_cardinality_response_schema
from minecraft_mod_ai.task_template_catalog import load_record_template, load_template


def test_prompt_record_cardinality_cannot_be_semantically_blocked():
    workflow = load_template("prompt/workflow")
    record_identifiers = [
        identifier
        for identifier in workflow["steps"]
        if load_template(identifier)["execution"] == "records"
    ]

    assert record_identifiers
    for identifier in record_identifiers:
        template = load_record_template(identifier)
        assert template.get("cardinality_blocking") is False, identifier
        schema = record_cardinality_response_schema(template)
        assert schema["required"] == ["count"], identifier
        assert "blocked_reason" not in schema["properties"], identifier


def test_record_cardinality_blocking_remains_default_for_other_templates():
    schema = record_cardinality_response_schema({})

    assert schema["required"] == ["count", "blocked_reason"]
    assert "blocked_reason" in schema["properties"]
