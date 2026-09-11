from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Host-owned bounded cardinality: one count decision, then exact host iteration.
p = Path("minecraft_mod_ai/task_template_runner.py")
text = p.read_text(encoding="utf-8")
if "import re\n" not in text:
    text = text.replace("import json\n", "import json\nimport re\n", 1)
start = text.index("    records: list[dict[str, Any]] = []\n    seen: set[str] = set()\n    while True:\n")
end = text.index("\n\ndef record_response_schema(template):", start)
new_block = '''    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    required_properties: list[str] = []
    if identifier == "design/content_property":
        allowed = normalized_context.get("allowed_properties")
        requested = normalized_context.get("required_properties", ())
        if isinstance(allowed, list):
            allowed_set = {str(name) for name in allowed}
            if isinstance(requested, list):
                required_properties = list(
                    dict.fromkeys(str(name) for name in requested if str(name) in allowed_set)
                )
            maximum_records = len(allowed)
        else:
            maximum_records = 0
        minimum_records = len(required_properties)
    elif identifier == "design/content_entity":
        requirement = str(normalized_context.get("requirement") or "")
        lexical_tokens = re.findall(r"[A-Za-z0-9_]+|[가-힣]+", requirement)
        minimum_records = 1
        maximum_records = max(1, len(lexical_tokens))
    elif identifier == "design/content_relation":
        entity_ids = normalized_context.get("entity_ids")
        n = len(entity_ids) if isinstance(entity_ids, list) else 0
        minimum_records = 0
        maximum_records = n * max(0, n - 1) * 14
    elif identifier == "design/decision":
        allowed_slots = normalized_context.get("allowed_slots")
        minimum_records = 0
        maximum_records = len(allowed_slots) if isinstance(allowed_slots, list) else 0
    else:
        def structural_size(value):
            if isinstance(value, dict):
                return 1 + sum(structural_size(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return 1 + sum(structural_size(item) for item in value)
            return 1

        minimum_records = 0
        maximum_records = structural_size(normalized_context.get("evidence_shard", {}))

    if maximum_records < minimum_records:
        raise TemplateBlocked(
            f"TEMPLATE_CARDINALITY: {identifier} structural maximum is below required minimum"
        )
    if maximum_records == 0:
        return {"records": [], "reason": "", "evidence_refs": refs}

    count_record = run_single_record_template(
        router,
        "design/record_count",
        context={
            "target_template": identifier,
            "target_task": template["task"],
            "target_rules": list(template.get("rules", ())),
            "active_context": normalized_context,
            "minimum_records": minimum_records,
            "maximum_records": maximum_records,
        },
        progress=progress,
        checkpoint=checkpoint,
    )
    target_count = int(count_record["count"])
    if not minimum_records <= target_count <= maximum_records:
        raise TemplateBlocked(
            f"TEMPLATE_CARDINALITY: {identifier} returned {target_count}, expected "
            f"{minimum_records}..{maximum_records}"
        )

    for index in range(target_count):
        record_context = {
            **normalized_context,
            "accepted_records": deepcopy(records),
            "record_index": index,
            "record_count": target_count,
        }
        if identifier == "design/content_property":
            allowed_properties = normalized_context.get("allowed_properties")
            if isinstance(allowed_properties, list):
                used = {str(row.get("property", "")) for row in records}
                remaining = [name for name in allowed_properties if name not in used]
                if not remaining:
                    raise TemplateBlocked(
                        f"TEMPLATE_CARDINALITY: {identifier} exhausted properties before count"
                    )
                record_context["allowed_properties"] = remaining
                if index < len(required_properties):
                    requested_property = required_properties[index]
                    if requested_property in used:
                        raise TemplateBlocked(
                            f"TEMPLATE_CARDINALITY: duplicate required property {requested_property}"
                        )
                    record_context["requested_property"] = requested_property

        record = run_single_record_template(
            router,
            identifier,
            context=record_context,
            progress=progress,
            checkpoint=checkpoint,
        )
        if _contains_blank_string(record, template["record_schema"]):
            raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
        key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if key in seen:
            raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
        seen.add(key)
        records.append(record)

    if identifier == "design/content_property":
        produced = {str(row.get("property", "")) for row in records}
        missing = [name for name in required_properties if name not in produced]
        if missing:
            raise TemplateBlocked(
                f"TEMPLATE_REQUIRED_RECORD_MISSING: {identifier}: {missing}"
            )
    return {"records": records, "reason": "", "evidence_refs": refs}
'''
p.write_text(text[:start] + new_block + text[end:], encoding="utf-8")

Path("minecraft_mod_ai/templates/design/record_count.yaml").write_text(
    '''id: design/record_count
task: Count how many distinct records are explicitly required for the target atomic task.
record_schema:
  type: object
  properties:
    count:
      type: integer
      minimum: 0
  required:
    - count
  additionalProperties: false
rules:
  - Count distinct explicit concerns only; do not invent records to increase the count.
  - count must be at least minimum_records and at most maximum_records supplied by the host.
  - This call decides cardinality once. It does not generate content and cannot extend the sequence later.
  - The host owns repetition and termination after this bounded cardinality decision.
''',
    encoding="utf-8",
)
Path("minecraft_mod_ai/templates/design/continue_record.yaml").unlink(missing_ok=True)

# 2) Required property closure and behavior-dependent entity attack semantics.
p = Path("minecraft_mod_ai/content_design_graph.py")
text = p.read_text(encoding="utf-8")
needle = '''        }[fact_type]
        for prop in records(
            "design/content_property",
            {**context, "allowed_properties": sorted(allowed_properties)},
        ):
'''
replacement = '''        }[fact_type]
        required_properties = {
            FactType.ITEM_EXISTS: {"display_name"},
            FactType.BLOCK_EXISTS: {"display_name"},
            FactType.ENTITY_EXISTS: {
                "display_name", "category", "health", "speed", "tracking_range",
                "width", "height", "archetype", "behavior", "main_color",
            },
            FactType.GUI_EXISTS: {"display_name", "screen_type"},
            FactType.NETWORK_PACKET: {"display_name", "packet_name", "channel", "direction"},
            FactType.BLOCK_ENTITY_EXISTS: {"display_name", "sync_type", "container_size"},
            FactType.DATA_COMPONENT: {"display_name", "component_name", "value_type", "codec"},
            FactType.WORLDGEN_FEATURE: {"display_name", "feature_type", "step", "biomes"},
            FactType.DIMENSION: {"display_name", "dimension_type", "ambient_light", "coordinate_scale"},
            FactType.BIOME: {"display_name", "temperature", "downfall", "precipitation"},
            FactType.STATUS_EFFECT: {"display_name", "category", "color", "beneficial"},
            FactType.SOUND_EVENT: {"display_name", "sound_id", "category"},
            FactType.PARTICLE_TYPE: {"display_name", "particle_name", "override_limiter"},
            FactType.ENTITY_LOOT: {"display_name", "loot_table_id", "type"},
            FactType.ADVANCEMENT: {"display_name", "frame_type"},
            FactType.EQUIPMENT_ARMOR: {"display_name", "slot", "defense", "toughness"},
            FactType.CUSTOM_ITEM_BEHAVIOR: {"display_name", "action", "cooldown"},
            FactType.CUSTOM_BLOCK_BEHAVIOR: {"display_name", "trigger", "interaction"},
            FactType.CRAFTING_RECIPE: {"recipe_kind", "count"},
            FactType.SMELTING_RECIPE: {"cooking_type", "experience", "cookingtime"},
            FactType.REGISTRY_TAG: {"registry_kind"},
        }[fact_type]
        for prop in records(
            "design/content_property",
            {
                **context,
                "allowed_properties": sorted(allowed_properties),
                "required_properties": sorted(required_properties),
            },
        ):
'''
if needle not in text:
    raise SystemExit("property closure insertion point not found")
text = text.replace(needle, replacement, 1)

needle = '''        if fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
'''
insertion = '''        if fact_type == FactType.ENTITY_EXISTS:
            behavior_value = str(props.get("behavior", "")).strip().lower()
            if behavior_value in {"hostile_melee", "neutral_melee"} and "attack_damage" not in props:
                combat_rows = records(
                    "design/content_property",
                    {
                        **context,
                        "allowed_properties": ["attack_damage"],
                        "required_properties": ["attack_damage"],
                    },
                )
                if len(combat_rows) != 1 or combat_rows[0]["property"] != "attack_damage":
                    raise SlotFillError(f"CONTENT_PROPERTY_UNRESOLVED: {eid}.attack_damage")
                props["attack_damage"] = combat_rows[0]["value"]

        if fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
'''
if needle not in text:
    raise SystemExit("entity conditional insertion point not found")
text = text.replace(needle, insertion, 1)
text = text.replace(
    '''                "category",
                "health",
                "attack_damage",
                "speed",
''',
    '''                "category",
                "health",
                "speed",
''',
    1,
)
text = text.replace(
    '''                "health",
                "attack_damage",
                "speed",
''',
    '''                "health",
                "speed",
''',
    1,
)
marker = '''            if props["category"].strip().lower() not in {
'''
attack_validation = '''            if "attack_damage" in props:
                try:
                    attack_damage = float(props["attack_damage"])
                except (TypeError, ValueError) as exc:
                    raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.attack_damage") from exc
                if attack_damage < 0 or (
                    props["behavior"].strip().lower() in {"hostile_melee", "neutral_melee"}
                    and attack_damage <= 0
                ):
                    raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.attack_damage")
            elif props["behavior"].strip().lower() in {"hostile_melee", "neutral_melee"}:
                raise SlotFillError(f"CONTENT_PROPERTY_UNRESOLVED: {eid}.attack_damage")

            if props["category"].strip().lower() not in {
'''
if marker not in text:
    raise SystemExit("attack validation insertion point not found")
text = text.replace(marker, attack_validation, 1)
text = text.replace(
    '                    "attack_damage": float(props["attack_damage"]),\n',
    '                    "attack_damage": float(props.get("attack_damage", "0")),\n',
    1,
)

# 4) Relation lowering must request executable custom augmentation.
marker = '''    module_by_id = {m.module_id: m for m in modules}
    module_deps = {m.module_id: list(m.depends_on) for m in modules}

'''
helper = '''    module_by_id = {m.module_id: m for m in modules}
    module_deps = {m.module_id: list(m.depends_on) for m in modules}

    def require_relation_codegen(module_id, relation_type, target_id):
        module = module_by_id.get(module_id)
        if module is None:
            return
        module.config["requires_custom_generation"] = True
        binding = {"relation": relation_type, "target": target_id}
        bindings = module.config.setdefault("executable_relations", [])
        if binding not in bindings:
            bindings.append(binding)

'''
if marker not in text:
    raise SystemExit("relation helper insertion point not found")
text = text.replace(marker, helper, 1)
relation_replacements = {
    'module_by_id[source].config.setdefault(rel_type, []).append(target)': 'module_by_id[source].config.setdefault(rel_type, []).append(target)\n                require_relation_codegen(source, rel_type, target)',
    'module_by_id[target].config.setdefault("unlocked_by", []).append(source)': 'module_by_id[target].config.setdefault("unlocked_by", []).append(source)\n                require_relation_codegen(target, "unlocked_by", source)',
    'module_by_id[source].config["opens_gui"] = target': 'module_by_id[source].config["opens_gui"] = target\n                require_relation_codegen(source, "opens", target)',
    'module_by_id[source].config["controls"] = target': 'module_by_id[source].config["controls"] = target\n                require_relation_codegen(source, "controls", target)',
    'module_by_id[source].config["spawns"] = target': 'module_by_id[source].config["spawns"] = target\n                require_relation_codegen(source, "spawns", target)',
    'module_by_id[source].config["transports_to"] = target': 'module_by_id[source].config["transports_to"] = target\n                require_relation_codegen(source, "transports_to", target)',
    'module_by_id[source].config.setdefault("displays", []).append(target)': 'module_by_id[source].config.setdefault("displays", []).append(target)\n                require_relation_codegen(source, "displays", target)',
    'module_by_id[source].config["sync_packet"] = target': 'module_by_id[source].config["sync_packet"] = target\n                require_relation_codegen(source, "synchronizes", target)',
}
for old, new in relation_replacements.items():
    if old not in text:
        raise SystemExit(f"relation lowering point missing: {old}")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Entity generator accepts zero damage only for non-combat behavior.
p = Path("minecraft_mod_ai/geckolib_generator.py")
text = p.read_text(encoding="utf-8")
old = '''    for name, value in {
        "max_health": max_health,
        "attack_damage": attack_damage,
        "movement_speed": movement_speed,
        "entity_width": entity_width,
        "entity_height": entity_height,
        "follow_range": follow_range,
    }.items():
'''
new = '''    for name, value in {
        "max_health": max_health,
        "movement_speed": movement_speed,
        "entity_width": entity_width,
        "entity_height": entity_height,
        "follow_range": follow_range,
    }.items():
'''
if old not in text:
    raise SystemExit("geckolib positive numeric block not found")
text = text.replace(old, new, 1)
marker = '''    if archetype not in _ARCHETYPES or (archetype == "custom" and not custom_bones):
'''
validation = '''    if (
        isinstance(attack_damage, bool)
        or not isinstance(attack_damage, (int, float))
        or not math.isfinite(float(attack_damage))
        or float(attack_damage) < 0
    ):
        raise GeckoLibGenerationError("attack_damage must be a non-negative finite number.")
    if behavior in {"hostile_melee", "neutral_melee"} and float(attack_damage) <= 0:
        raise GeckoLibGenerationError("combat entity attack_damage must be positive.")
    if archetype not in _ARCHETYPES or (archetype == "custom" and not custom_bones):
'''
if marker not in text:
    raise SystemExit("geckolib attack insertion point not found")
text = text.replace(marker, validation, 1)
p.write_text(text, encoding="utf-8")

# Add custom relation augmentation after base/extended generation and after GeckoLib entity generation.
p = Path("minecraft_mod_ai/complete_orchestrator.py")
text = p.read_text(encoding="utf-8")
old = 'receipts.extend(generate_custom(module) for module in members if module.kind not in extended_kinds and module not in sidecars and (module not in research_shards) and (module not in artifact_handled_members))'
new = 'receipts.extend(generate_custom(module) for module in members if (module.kind not in extended_kinds or module.config.get("requires_custom_generation")) and module not in sidecars and (module not in research_shards) and (module not in artifact_handled_members))'
if old not in text:
    raise SystemExit("custom augmentation selector not found")
text = text.replace(old, new, 1)
marker = '''                            policy=self.policy,
                        )
                    )
            elif stage == 'custom':
'''
replacement = '''                            policy=self.policy,
                        )
                    )
                    if config.get("requires_custom_generation"):
                        receipts.append(generate_custom(module))
            elif stage == 'custom':
'''
if marker not in text:
    raise SystemExit("entity custom augmentation insertion point not found")
text = text.replace(marker, replacement, 1)
p.write_text(text, encoding="utf-8")

Path("tests/test_four_axes_contract.py").write_text(
    '''from pathlib import Path

from minecraft_mod_ai.fact_source_reuse import FactReuseClassifier, ReuseMode
from minecraft_mod_ai.implementation_fact import FactType, ImplementationFact


def test_host_sequence_uses_one_bounded_count_not_continue_boolean():
    source = Path("minecraft_mod_ai/task_template_runner.py").read_text(encoding="utf-8")
    assert '"design/record_count"' in source
    assert '"design/continue_record"' not in source
    assert "maximum_records" in source and "minimum_records" in source
    assert not Path("minecraft_mod_ai/templates/design/continue_record.yaml").exists()


def test_content_graph_declares_required_property_closure_and_conditional_attack():
    source = Path("minecraft_mod_ai/content_design_graph.py").read_text(encoding="utf-8")
    assert '"required_properties": sorted(required_properties)' in source
    assert 'behavior_value in {"hostile_melee", "neutral_melee"}' in source
    assert 'float(props.get("attack_damage", "0"))' in source


def test_weak_substring_is_not_adapted():
    classifier = FactReuseClassifier(project_index={"symbols": {"ore": {"file": "src/Ore.java"}}})
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
