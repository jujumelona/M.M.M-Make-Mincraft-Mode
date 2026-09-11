from pathlib import Path

from minecraft_mod_ai.fact_source_reuse import FactReuseClassifier, ReuseMode
from minecraft_mod_ai.implementation_fact import FactType, ImplementationFact
from minecraft_mod_ai.atomic_slot_executor import SlotDefinition
from minecraft_mod_ai.task_template_catalog import load_template


def test_graph_cardinality_surfaces_do_not_use_model_count_or_continue():
    source = Path("minecraft_mod_ai/task_template_runner.py").read_text(encoding="utf-8")
    graph_section = source[source.index('if identifier == "design/content_property"'):source.index('# Research facts and design decisions')]
    assert '"design/record_count"' not in source
    assert '"design/continue_record"' not in graph_section
    assert '"design/relation_set"' in graph_section
    assert "entity_count" in graph_section


def test_relation_templates_are_bounded_atomic_schemas():
    for identifier in ("design/content_relation", "design/relation_set"):
        template = load_template(identifier)
        SlotDefinition(identifier, template["record_schema"]).validate_schema()


def test_content_graph_declares_required_property_closure_and_conditional_attack():
    source = Path("minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    assert '"required_properties": sorted(required_properties)' in source
    assert 'behavior_value in {"hostile_melee", "neutral_melee"}' in source
    assert 'float(props.get("attack_damage", "0"))' in source


def test_weak_substring_is_not_adapted():
    classifier = FactReuseClassifier(project_index={"symbols": {"ore": {"file": "src/items/Ore.java"}}})
    fact = ImplementationFact(fact_id="core.exists", fact_type=FactType.ITEM_EXISTS, subject="core")
    assert classifier.classify(fact).mode is ReuseMode.NEW


def test_relation_lowering_requires_executable_augmentation():
    graph = Path("minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    orchestrator = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    assert 'module.config["requires_custom_generation"] = True' in graph
    assert 'module.config.get("requires_custom_generation")' in orchestrator
    assert 'receipts.append(generate_custom(module))' in orchestrator
