from pathlib import Path

# This script runs after _four_axes_surgery.py in the same working tree and
# replaces its one-time model cardinality decision with deterministic host closure
# for entity/property/relation records.
p = Path("minecraft_mod_ai/task_template_runner.py")
text = p.read_text(encoding="utf-8")
start = text.index("    records: list[dict[str, Any]] = []\n    seen: set[str] = set()\n")
end = text.index("\n\ndef record_response_schema(template):", start)
new_block = '''    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    if identifier == "design/content_property":
        allowed = normalized_context.get("allowed_properties")
        required = normalized_context.get("required_properties")
        allowed_set = {str(name) for name in allowed} if isinstance(allowed, list) else set()
        required_properties = (
            list(dict.fromkeys(str(name) for name in required))
            if isinstance(required, list)
            else []
        )
        unknown = [name for name in required_properties if name not in allowed_set]
        if unknown:
            raise TemplateBlocked(f"TEMPLATE_REQUIRED_PROPERTY_UNKNOWN: {unknown}")
        for index, requested_property in enumerate(required_properties):
            record = run_single_record_template(
                router,
                identifier,
                context={
                    **normalized_context,
                    "allowed_properties": [requested_property],
                    "requested_property": requested_property,
                    "record_index": index,
                    "record_count": len(required_properties),
                    "accepted_records": deepcopy(records),
                },
                progress=progress,
                checkpoint=checkpoint,
            )
            if record.get("property") != requested_property:
                raise TemplateBlocked(
                    f"TEMPLATE_REQUIRED_PROPERTY_MISMATCH: expected {requested_property}, "
                    f"received {record.get('property')}"
                )
            records.append(record)
        return {"records": records, "reason": "", "evidence_refs": refs}

    if identifier == "design/content_entity":
        requirement = str(normalized_context.get("requirement") or "").lower()
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3,
            "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7,
            "여덟": 8, "아홉": 9, "열": 10,
        }
        explicit_counts = [int(value) for value in re.findall(r"(?<![A-Za-z0-9_])(\\d{1,2})(?![A-Za-z0-9_])", requirement)]
        tokens = re.findall(r"[a-z]+|[가-힣]+", requirement)
        explicit_counts.extend(number_words[token] for token in tokens if token in number_words)
        target_count = max(explicit_counts, default=1)
        if not 1 <= target_count <= 64:
            raise TemplateBlocked(f"TEMPLATE_ENTITY_CARDINALITY_UNSUPPORTED: {target_count}")
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

    if identifier == "design/content_relation":
        entity_ids = normalized_context.get("entity_ids")
        if not isinstance(entity_ids, list):
            raise TemplateBlocked("TEMPLATE_RELATION_ENTITY_IDS_REQUIRED")
        relation_types = (
            "consumes", "produces", "contains", "drops", "requires", "unlocks",
            "upgrades", "opens", "controls", "spawns", "transports_to", "displays",
            "synchronizes",
        )
        for source_id in entity_ids:
            for target_id in entity_ids:
                if source_id == target_id:
                    continue
                decision = run_single_record_template(
                    router,
                    "design/relation_set",
                    context={
                        **normalized_context,
                        "source_id": source_id,
                        "target_id": target_id,
                        "allowed_relation_types": list(relation_types),
                    },
                    progress=progress,
                    checkpoint=checkpoint,
                )
                if decision.get("overflow"):
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_CARDINALITY_EXCEEDED: {source_id}->{target_id}"
                    )
                selected = decision.get("relations", [])
                if not isinstance(selected, list) or any(item not in relation_types for item in selected):
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_UNSUPPORTED: {source_id}->{target_id}: {selected}"
                    )
                if len(selected) != len(set(selected)):
                    raise TemplateBlocked(
                        f"TEMPLATE_RELATION_DUPLICATE: {source_id}->{target_id}: {selected}"
                    )
                for relation_type in selected:
                    records.append(
                        {
                            "relation_type": relation_type,
                            "source_id": str(source_id),
                            "target_id": str(target_id),
                        }
                    )
                key_code = int(decision.get("key_code", 0))
                if key_code:
                    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    if not 1 <= key_code <= len(alphabet):
                        raise TemplateBlocked(
                            f"TEMPLATE_RELATION_KEY_INVALID: {source_id}->{target_id}: {key_code}"
                        )
                    records.append(
                        {
                            "relation_type": "key_" + alphabet[key_code - 1],
                            "source_id": str(source_id),
                            "target_id": str(target_id),
                        }
                    )
        return {"records": records, "reason": "", "evidence_refs": refs}

    # Research facts and design decisions still use the generic host-owned loop;
    # they are not graph-cardinality closure surfaces.
    while True:
        continuation = run_single_record_template(
            router,
            "design/continue_record",
            context={
                "target_template": identifier,
                "target_task": template["task"],
                "target_rules": list(template.get("rules", ())),
                "active_context": normalized_context,
                "accepted_record_ids": _accepted_record_index(identifier, records),
            },
            progress=progress,
            checkpoint=checkpoint,
        )
        if not bool(continuation["required"]):
            return {"records": records, "reason": "", "evidence_refs": refs}
        record = run_single_record_template(
            router,
            identifier,
            context={**normalized_context, "accepted_records": deepcopy(records)},
            progress=progress,
            checkpoint=checkpoint,
        )
        key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if key in seen:
            raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
        seen.add(key)
        records.append(record)
'''
p.write_text(text[:start] + new_block + text[end:], encoding="utf-8")

# record_count is obsolete; restore generic continuation only for research/decision.
Path("minecraft_mod_ai/templates/design/record_count.yaml").unlink(missing_ok=True)
Path("minecraft_mod_ai/templates/design/continue_record.yaml").write_text(
    '''id: design/continue_record
task: Decide whether exactly one additional record is still required for the active atomic concern.
record_schema:
  type: object
  properties:
    required:
      type: boolean
  required:
    - required
  additionalProperties: false
rules:
  - Return required=true only when the target task still has one distinct unresolved concern that is not represented by accepted_record_ids.
  - Return required=false when the target task is complete, not applicable, or another record would duplicate accepted_record_ids.
  - Do not invent a new requirement merely to continue iteration.
  - The host owns repetition and termination; this record only answers the current one-step continuation question.
''',
    encoding="utf-8",
)
Path("minecraft_mod_ai/templates/design/relation_set.yaml").write_text(
    '''id: design/relation_set
task: Select all explicit relationships from the fixed host-supplied source entity to the fixed host-supplied target entity.
record_schema:
  type: object
  properties:
    relations:
      type: array
      maxItems: 4
      uniqueItems: true
      items:
        type: string
        enum: [consumes, produces, contains, drops, requires, unlocks, upgrades, opens, controls, spawns, transports_to, displays, synchronizes]
    key_code:
      type: integer
      minimum: 0
      maximum: 36
    overflow:
      type: boolean
  required: [relations, key_code, overflow]
  additionalProperties: false
rules:
  - Select only relationships explicitly required by the active requirement; an empty relations array means none.
  - The source_id and target_id are fixed by the host and must not be reinterpreted.
  - Set overflow=true rather than silently omitting a fifth or later ordinary relation for this exact ordered pair.
  - key_code is 0 when there is no crafting-key relation; otherwise 1-26 encode A-Z and 27-36 encode 0-9.
''',
    encoding="utf-8",
)

# Make the pre-existing relation schema itself comply with the global atomicity bound.
p = Path("minecraft_mod_ai/templates/design/content_relation.yaml")
text = p.read_text(encoding="utf-8")
needle = "    relation_type:\n      type: string\n      pattern:"
if needle in text and "relation_type:\n      type: string\n      maxLength:" not in text:
    text = text.replace(
        "    relation_type:\n      type: string\n      pattern:",
        "    relation_type:\n      type: string\n      maxLength: 32\n      pattern:",
        1,
    )
p.write_text(text, encoding="utf-8")

# Required-property calls are exact host requests; make that binding normative.
p = Path("minecraft_mod_ai/templates/design/content_property.yaml")
text = p.read_text(encoding="utf-8")
rule = "  - When requested_property is supplied by the host, return exactly that property and no other property.\n"
if rule not in text:
    text += rule
p.write_text(text, encoding="utf-8")

# Replace the regression contract created by the first surgery with the deterministic contract.
Path("tests/test_four_axes_contract.py").write_text(
    '''from pathlib import Path

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
''',
    encoding="utf-8",
)
