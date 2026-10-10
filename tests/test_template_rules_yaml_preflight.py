"""Regression tests for YAML mappings accidentally parsed out of prompt rules."""

from __future__ import annotations

import pytest

from minecraft_mod_ai.single_record_template import run_single_record_template
from minecraft_mod_ai.task_template_catalog import (
    RUNTIME_TEMPLATE_ROOT,
    load_record_template,
)
from minecraft_mod_ai.template_contract_validation import (
    validate_catalog_rule_types,
    validate_template_contract,
)


def test_content_entity_rules_are_strings_and_single_record_can_generate() -> None:
    template = load_record_template("design/content_entity")
    assert all(isinstance(rule, str) and rule.strip() for rule in template["rules"])
    assert "travels, or explores: identify" in template["rules"][9]

    accepted = {
        "role": "Collect ore as an inventory item.",
        "kind": "item",
        "entity_id": "ore_token",
    }
    seen = []

    def generator(_router, role, messages, **kwargs):
        assert role == "planner"
        assert isinstance(messages[0]["content"], str)
        field = next(iter(kwargs["response_schema"]["properties"]))
        seen.append(field)
        return {field: accepted[field]}

    record = run_single_record_template(
        object(),
        "design/content_entity",
        context={
            "requirement": "A player collects an ore item.",
            "allowed_content_kinds": ["item"],
        },
        generator=generator,
    )
    assert record == accepted
    assert seen == ["role", "kind", "entity_id"]


def test_catalog_rule_preflight_accepts_packaged_rules() -> None:
    assert validate_catalog_rule_types(RUNTIME_TEMPLATE_ROOT) > 0


@pytest.mark.parametrize("rule", [{"bad": "mapping"}, None, 42, "", "   "])
def test_template_rejects_nonstring_or_empty_rule(rule) -> None:
    with pytest.raises(ValueError, match="TEMPLATE_RULE_TYPE.*rules\\[0\\]"):
        validate_template_contract({
            "id": "design/example",
            "rules": [rule],
        })


def test_yaml_colon_mapping_fails_preflight_with_named_record(tmp_path) -> None:
    broken = tmp_path / "design" / "content_entity.yaml"
    broken.parent.mkdir(parents=True)
    broken.write_text(
        "id: design/content_entity\nrules:\n"
        "  - A gameplay mechanic: identify the concrete item\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"TEMPLATE_RULE_TYPE: design/content_entity rules\[0\]"):
        validate_catalog_rule_types(tmp_path)


def test_api_preflight_rejects_bad_rule_before_planning(monkeypatch) -> None:
    from minecraft_mod_ai import api, internal_package_preflight, template_contract_validation

    monkeypatch.setattr(
        internal_package_preflight,
        "validate_internal_package_integrity",
        lambda: {"status": "PASS"},
    )

    def bad_rules(_root):
        raise ValueError("TEMPLATE_RULE_TYPE: design/content_entity rules[9]")

    monkeypatch.setattr(
        template_contract_validation,
        "validate_catalog_rule_types",
        bad_rules,
    )
    with pytest.raises(api.SpecValidationError, match="TEMPLATE_RULE_TYPE"):
        api._validate_internal_engine_preflight()
