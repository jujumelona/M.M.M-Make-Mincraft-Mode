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


def test_behavior_contract_generation_uses_only_bounded_count_decision():
    for path, template in _behavior_contract_templates():
        schema = record_cardinality_response_schema(template)
        assert schema["required"] == ["count"], path.name
        assert set(schema["properties"]) == {"count"}, path.name
        count = schema["properties"]["count"]
        assert count["enum"] == list(range(17)), path.name
        assert count["minimum"] == 0, path.name
        assert count["maximum"] == 16, path.name
        assert "blocked_reason" not in schema["properties"], path.name


def test_behavior_contract_records_do_not_delegate_loop_protocol_to_model():
    forbidden = ("return done", "return not_applicable", "return blocked")
    for path, template in _behavior_contract_templates():
        rules = "\n".join(str(rule).lower() for rule in template.get("rules", ()))
        for phrase in forbidden:
            assert phrase not in rules, f"{path.name}: {phrase}"


def test_cardinality_contract_is_closed_and_contains_no_loop_protocol():
    schema = record_cardinality_response_schema({})
    assert schema["required"] == ["count"]
    assert set(schema["properties"]) == {"count"}
    assert schema["additionalProperties"] is False
