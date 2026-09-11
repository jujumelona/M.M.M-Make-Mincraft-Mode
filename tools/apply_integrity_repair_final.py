from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    value = read(path)
    old_count = value.count(old)
    if old_count == 0 and value.count(new) >= 1:
        return
    if old_count != 1:
        raise RuntimeError(f"{path}: expected exactly one patch target, found {old_count}")
    write(path, value.replace(old, new, 1))


path = "minecraft_mod_ai/content_design_graph.py"
value = read(path)
replacements = {
    '            FactType.ITEM_EXISTS: {"display_name"},\n':
        '            FactType.ITEM_EXISTS: {"display_name", "main_color"},\n',
    '            FactType.BLOCK_EXISTS: {"display_name"},\n':
        '            FactType.BLOCK_EXISTS: {"display_name", "main_color"},\n',
    '            FactType.GUI_EXISTS: {"display_name", "screen_type"},\n':
        '            FactType.GUI_EXISTS: {"display_name", "screen_type", "main_color"},\n',
    '            FactType.EQUIPMENT_ARMOR: {"display_name", "slot", "defense", "toughness"},\n':
        '            FactType.EQUIPMENT_ARMOR: {"display_name", "slot", "defense", "toughness", "main_color"},\n',
}
for old, new in replacements.items():
    if value.count(old) == 0 and value.count(new) == 1:
        continue
    if value.count(old) != 1:
        raise RuntimeError(f"{path}: visual-authority patch target missing: {old.strip()}")
    value = value.replace(old, new, 1)
write(path, value)

replace_once(
    "minecraft_mod_ai/populate_version_artifact_rules.py",
    '        "executor_type": "deterministic_renderer" if template_id else "python_generator",\n',
    '        "executor_type": executor_type,\n',
)

DEFAULT_PROPERTY_BLOCK = '''                defaults = {
                    "main_color": "#808080",
                    "hardness": "1.5",
                    "attack_damage": "4",
                    "attack_speed": "1.0",
                    "hunger": "4",
                    "saturation": "0.4",
                    "seed_color": "#70A050",
                    "category": "creature",
                    "health": "20",
                    "speed": "0.25",
                    "tracking_range": "32",
                    "width": "0.6",
                    "height": "1.8",
                    "archetype": "biped",
                    "behavior": "passive",
                    "screen_type": "container_9x3",
                    "slot_count": "27",
                    "packet_name": f"{eid}_packet",
                    "channel": f"test:{eid}",
                    "direction": "s2c",
                    "sync_type": "ticking_block_entity",
                    "container_size": "9",
                    "component_name": eid,
                    "value_type": "string",
                    "codec": "Codec.STRING",
                    "feature_type": "ore",
                    "step": "underground_ores",
                    "biomes": "minecraft:plains",
                    "dimension_type": "minecraft:overworld",
                    "ambient_light": "0.0",
                    "coordinate_scale": "1.0",
                    "temperature": "0.8",
                    "downfall": "0.4",
                    "precipitation": "rain",
                    "color": "#808080",
                    "beneficial": "true",
                    "sound_id": f"test:{eid}",
                    "particle_name": eid,
                    "override_limiter": "false",
                    "loot_table_id": f"test:entities/{eid}",
                    "type": "entity",
                    "frame_type": "task",
                    "slot": "chestplate",
                    "defense": "4",
                    "toughness": "1.0",
                    "action": "use",
                    "cooldown": "20",
                    "trigger": "use",
                    "interaction": "activate",
                    "input_item": "minecraft:stone",
                    "output_item": "minecraft:cobblestone",
                    "output_count": "1",
                    "processing_ticks": "20",
                    "max_level": "1",
                    "literal": "test",
                    "message": "test",
                    "permission_level": "0",
                }
                if requested not in defaults:
                    raise AssertionError(f"missing requested property {requested}")
                return {"property": requested, "value": defaults[requested]}
'''

for path in ("tests/test_atomic_design_pipeline.py", "tests/test_content_design_graph_expansion.py"):
    value = read(path)
    eid_line = '            eid = context["entity"]["entity_id"]\n'
    fallback = '''            entity = context.get("entity")
            eid = entity["entity_id"] if isinstance(entity, dict) else str(context.get("module_id") or "")
            if not eid:
                raise AssertionError("content-property fake requires entity_id or module_id")
'''
    value = value.replace(eid_line, fallback)

    old_missing = '''                matching = [row for row in rows if row.get("property") == requested]
                if len(matching) != 1:
                    raise AssertionError(f"missing requested property {requested}")
                return deepcopy(matching[0])
'''
    if value.count(old_missing) != 1:
        raise RuntimeError(f"{path}: requested-property fake target missing")
    new_missing = '''                matching = [row for row in rows if row.get("property") == requested]
                if len(matching) == 1:
                    return deepcopy(matching[0])
''' + DEFAULT_PROPERTY_BLOCK
    value = value.replace(old_missing, new_missing, 1)
    write(path, value)

path = "tests/test_atomic_design_pipeline.py"
value = read(path)
old = '    "capability", ["MAGIC_SPELL", "DIMENSION_EXISTS", "SCREEN_EXISTS", "UNSUPPORTED"]\n'
new = '    "capability", ["UNSUPPORTED"]\n'
if value.count(old) != 1:
    raise RuntimeError("unsupported capability parametrization target missing")
write(path, value.replace(old, new, 1))

path = "tests/test_content_design_graph_expansion.py"
value = read(path)
if value.count('{"property": "main_color", "value": "void purple"}') != 1:
    raise RuntimeError("entity color fixture target missing")
value = value.replace(
    '{"property": "main_color", "value": "void purple"}',
    '{"property": "main_color", "value": "#5B2A86"}',
    1,
)
value = value.replace('assert "void purple" in asset.prompt', 'assert "#5B2A86" in asset.prompt', 1)
write(path, value)

path = "tests/test_deterministic_minecraft_content_contract.py"
value = read(path)
pattern = re.compile(
    r'(?P<indent>\s+)"id": "copper_hammer",\n'
    r'(?P=indent)"kind": "item",\n'
    r'(?P=indent)"depends_on": \["copper_hammer"\],'
)
match = pattern.search(value)
if match is None:
    already = re.search(
        r'"id": "copper_hammer",\n\s+"kind": "item",\n\s+"config": \{"display_name": "Copper Hammer", "main_color": "#B87333"\},\n\s+"depends_on": \["copper_hammer"\],',
        value,
    )
    if already is None:
        raise RuntimeError("self-dependency fixture target missing")
else:
    indent = match.group("indent")
    replacement = (
        f'{indent}"id": "copper_hammer",\n'
        f'{indent}"kind": "item",\n'
        f'{indent}"config": {{"display_name": "Copper Hammer", "main_color": "#B87333"}},\n'
        f'{indent}"depends_on": ["copper_hammer"],'
    )
    value = value[:match.start()] + replacement + value[match.end():]
    write(path, value)

print("final repair applied")
