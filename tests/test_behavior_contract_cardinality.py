from pathlib import Path

import yaml

from minecraft_mod_ai.bounded_record_template import record_cardinality_response_schema


BEHAVIOR_CONTRACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "minecraft_mod_ai"
    / "templates"
    / "feature"
    / "behavior_contract"
)


def _behavior_contract_templates():
    paths = sorted(BEHAVIOR_CONTRACT_ROOT.glob("*.yaml"))
    assert paths, "behavior-contract templates must exist"
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in paths]


def test_behavior_contract_generation_never_requires_a_model_owned_count():
    for path, template in _behavior_contract_templates():
        schema = record_cardinality_response_schema(template)
        assert schema["required"] == ["record"], path.name
        assert "count" not in schema["properties"], path.name
        assert "blocked_reason" not in schema["properties"], path.name


def test_behavior_contract_records_do_not_delegate_loop_protocol_to_model():
    forbidden = ("return done", "return not_applicable", "return blocked")
    for path, template in _behavior_contract_templates():
        rules = "\n".join(str(rule).lower() for rule in template.get("rules", ()))
        for phrase in forbidden:
            assert phrase not in rules, f"{path.name}: {phrase}"


def test_next_record_contract_is_closed_without_cardinality_metadata():
    schema = record_cardinality_response_schema({})
    assert schema["required"] == ["record"]
    assert set(schema["properties"]) == {"record"}
    assert schema["additionalProperties"] is False
