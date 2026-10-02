from minecraft_mod_ai.bounded_record_template import record_cardinality_response_schema
from minecraft_mod_ai.task_template_catalog import load_record_template, load_template


def test_prompt_record_progress_never_asks_model_for_total_count():
    workflow = load_template("prompt/workflow")
    record_identifiers = [
        identifier
        for identifier in workflow["steps"]
        if load_template(identifier)["execution"] == "records"
    ]

    assert record_identifiers
    for identifier in record_identifiers:
        template = load_record_template(identifier)
        schema = record_cardinality_response_schema(template)
        assert schema["required"] == ["record"], identifier
        assert "count" not in schema["properties"], identifier
        assert "blocked_reason" not in schema["properties"], identifier


def test_next_record_contract_has_no_cardinality_or_loop_control_fields():
    schema = record_cardinality_response_schema({})
    assert schema["required"] == ["record"]
    assert set(schema["properties"]) == {"record"}
    assert schema["additionalProperties"] is False
