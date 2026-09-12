from pathlib import Path

import yaml

from minecraft_mod_ai.bounded_record_template import (
    _cardinality_blocking_enabled,
    record_cardinality_response_schema,
)


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


def test_behavior_contract_cardinality_never_blocks_on_missing_external_candidates():
    for path, template in _behavior_contract_templates():
        assert template["cardinality_blocking"] is False, path.name
        schema = record_cardinality_response_schema(template)
        assert schema["required"] == ["count"], path.name
        assert "blocked_reason" not in schema["properties"], path.name


def test_behavior_contract_records_do_not_delegate_host_control_to_model():
    forbidden = ("return done", "return not_applicable", "return blocked")
    for path, template in _behavior_contract_templates():
        rules = "\n".join(str(rule).lower() for rule in template.get("rules", ()))
        for phrase in forbidden:
            assert phrase not in rules, f"{path.name}: {phrase}"
        assert "host owns cardinality and iteration" in rules, path.name


def test_cardinality_blocking_remains_the_default_for_other_record_templates():
    assert _cardinality_blocking_enabled({}) is True
    schema = record_cardinality_response_schema({})
    assert schema["required"] == ["count", "blocked_reason"]
    assert "blocked_reason" in schema["properties"]
