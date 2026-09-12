from pathlib import Path

from minecraft_mod_ai.atomic_slot_executor import SlotDefinition
from minecraft_mod_ai.fact_source_reuse import FactReuseClassifier, ReuseMode
from minecraft_mod_ai.implementation_fact import FactType, ImplementationFact
from minecraft_mod_ai.task_template_catalog import load_template


def test_graph_closure_is_host_owned_for_entity_property_relation():
    source = Path("minecraft_mod_ai/design_record_runtime.py").read_text(encoding="utf-8")
    assert '"design/content_entity_count"' in source
    assert 'target_count = int(cardinality["count"])' in source
    assert '"design/content_relation_count"' in source
    assert '"design/relation_set"' not in source
    assert '"design/continue_record"' not in source
    assert "requested_property" in source
    assert "for source_id in entity_ids:" in source
    assert "for target_id in entity_ids:" in source


def test_relation_atomic_schemas_are_bounded():
    for identifier in ("design/content_relation", "design/content_relation_count"):
        template = load_template(identifier)
        SlotDefinition(identifier, template["record_schema"]).validate_schema()


def test_passive_entity_does_not_require_positive_attack_damage_in_graph():
    source = Path("minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    numeric = source[source.index("for numeric_key in ("):source.index('if "attack_damage" in props:')]
    assert '"attack_damage"' not in numeric
    assert 'float(props.get("attack_damage", "0"))' in source
    assert 'behavior_value in {"hostile_melee", "neutral_melee"}' in source


def test_geckolib_allows_zero_damage_only_for_noncombat_profiles():
    source = Path("minecraft_mod_ai/geckolib_generator.py").read_text(encoding="utf-8")
    assert "attack_damage must be a non-negative finite number" in source
    assert "combat entity attack_damage must be positive" in source


def test_weak_substring_reuse_is_rejected():
    classifier = FactReuseClassifier(project_index={"symbols": {"ore": {"file": "src/main/java/items/Ore.java"}}})
    fact = ImplementationFact(fact_id="core.exists", fact_type=FactType.ITEM_EXISTS, subject="core")
    assert classifier.classify(fact).mode is ReuseMode.NEW


def test_relation_semantics_reach_executable_generation():
    graph = Path("minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    orchestrator = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    assert 'module.config["requires_custom_generation"] = True' in graph
    assert 'module.config.setdefault("executable_relations", [])' in graph
    assert 'module.config.get("requires_custom_generation")' in orchestrator
    assert 'receipts.append(generate_custom(module))' in orchestrator
