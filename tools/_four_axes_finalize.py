from pathlib import Path

p = Path("minecraft_mod_ai/task_template_runner.py")
text = p.read_text(encoding="utf-8")
loop_anchor = text.index("    records: list[dict[str, Any]] = []\n    seen: set[str] = set()\n")
start = text.index('    if identifier == "design/content_entity":\n', loop_anchor)
end = text.index('\n    if identifier == "design/content_relation":', start)
entity_block = '''    if identifier == "design/content_entity":
        target_count = 1
        for index in range(target_count):
            record = run_single_record_template(
                router,
                identifier,
                context={
                    **normalized_context,
                    "entity_ordinal": index + 1,
                    "entity_count": target_count,
                    "accepted_records": deepcopy(records),
                },
                progress=progress,
                checkpoint=checkpoint,
            )
            key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if key in seen:
                raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
            seen.add(key)
            records.append(record)
        return {"records": records, "reason": "", "evidence_refs": refs}
'''
text = text[:start] + entity_block + text[end:]
p.write_text(text, encoding="utf-8")

p = Path("minecraft_mod_ai/content_design_graph.py")
text = p.read_text(encoding="utf-8")
old = '''            for numeric_key in (
                "health",
                "attack_damage",
                "speed",
                "tracking_range",
                "width",
                "height",
            ):
'''
new = '''            for numeric_key in (
                "health",
                "speed",
                "tracking_range",
                "width",
                "height",
            ):
'''
if old not in text:
    raise SystemExit("entity numeric validation anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

Path("tests/test_four_axes_final_contract.py").write_text('''from pathlib import Path

from minecraft_mod_ai.atomic_slot_executor import SlotDefinition
from minecraft_mod_ai.fact_source_reuse import FactReuseClassifier, ReuseMode
from minecraft_mod_ai.implementation_fact import FactType, ImplementationFact
from minecraft_mod_ai.task_template_catalog import load_template


def test_graph_closure_is_host_owned_for_entity_property_relation():
    source = Path("minecraft_mod_ai/task_template_runner.py").read_text(encoding="utf-8")
    loop_anchor = source.index("    records: list[dict[str, Any]] = []\\n    seen: set[str] = set()\\n")
    prop_start = source.index('if identifier == "design/content_property"', loop_anchor)
    entity_start = source.index('if identifier == "design/content_entity"', prop_start)
    relation_start = source.index('if identifier == "design/content_relation"', entity_start)
    generic_start = source.index('# Research facts and design decisions', relation_start)
    prop = source[prop_start:entity_start]
    entity = source[entity_start:relation_start]
    relation = source[relation_start:generic_start]
    assert "target_count = 1" in entity
    assert '"design/continue_record"' not in entity + prop + relation
    assert '"design/relation_set"' in relation
    assert "requested_property" in prop


def test_relation_atomic_schemas_are_bounded():
    for identifier in ("design/content_relation", "design/relation_set"):
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
''', encoding="utf-8")
