"""Regression coverage for memory-bounded content relation prompting."""

import pytest

from minecraft_mod_ai.design_record_runtime import _pair_relation_prompt_context
from minecraft_mod_ai.template_errors import TemplateBlocked


def _fixture():
    entities = {
        "a": {"requirement_refs": ["req_a"], "entity_id": "a", "kind": "item"},
        "b": {"requirement_refs": ["req_b"], "entity_id": "b", "kind": "item"},
        "c": {"requirement_refs": ["req_c"], "entity_id": "c", "kind": "item"},
    }
    normalized = {
        "requirements": [
            {"requirement_id": "req_a", "requirement": "A behavior"},
            {"requirement_id": "req_b", "requirement": "B behavior"},
            {"requirement_id": "req_c", "requirement": "Unrelated C behavior"},
            {"requirement_id": "req_global", "requirement": "Global gameplay rule"},
        ],
        "design_contexts": [
            {"context": "A"}, {"context": "B"},
            {"context": "C"}, {"context": "global"},
        ],
        "requirement_design_contexts": [
            {"requirement_id": "req_a", "design_context": {"context": "A"}},
            {"requirement_id": "req_b", "design_context": {"context": "B"}},
            {"requirement_id": "req_c", "design_context": {"context": "C"}},
            {"requirement_id": "req_global", "design_context": {"context": "global"}},
        ],
    }
    return normalized, entities


def test_pair_prompt_excludes_unrelated_owned_context_but_keeps_global_rules():
    normalized, entities = _fixture()
    result = _pair_relation_prompt_context(
        normalized, entities["a"], entities["b"], entities
    )
    assert [row["requirement_id"] for row in result["requirements"]] == [
        "req_a", "req_b", "req_global"
    ]
    assert result["design_contexts"] == [
        {"context": "A"}, {"context": "B"}, {"context": "global"}
    ]
    assert normalized["requirements"][2]["requirement_id"] == "req_c"


def test_missing_legacy_ownership_does_not_drop_any_requirement():
    normalized, entities = _fixture()
    entities["c"].pop("requirement_refs")
    result = _pair_relation_prompt_context(
        normalized, entities["a"], entities["b"], entities
    )
    assert result["requirements"] == normalized["requirements"]
    assert result["design_contexts"] == normalized["design_contexts"]


def test_unmapped_design_contexts_remain_visible_in_legacy_callers():
    normalized, entities = _fixture()
    del normalized["requirement_design_contexts"]
    result = _pair_relation_prompt_context(
        normalized, entities["a"], entities["b"], entities
    )
    assert [row["requirement_id"] for row in result["requirements"]] == [
        "req_a", "req_b", "req_global"
    ]
    assert result["design_contexts"] == normalized["design_contexts"]


def test_unknown_requirement_owner_fails_closed_not_silently_pruned():
    normalized, entities = _fixture()
    entities["a"]["requirement_refs"].append("missing_requirement")
    with pytest.raises(TemplateBlocked, match="TEMPLATE_RELATION_REQUIREMENT_REF_UNKNOWN"):
        _pair_relation_prompt_context(normalized, entities["a"], entities["b"], entities)


def test_pair_scope_is_independent_of_entity_dictionary_order():
    normalized, entities = _fixture()
    reverse_entities = dict(reversed(tuple(entities.items())))
    assert _pair_relation_prompt_context(
        normalized, entities["a"], entities["b"], entities
    ) == _pair_relation_prompt_context(
        normalized, entities["a"], entities["b"], reverse_entities
    )
