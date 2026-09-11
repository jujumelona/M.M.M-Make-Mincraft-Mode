from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    value = read(path)
    if value.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one patch target")
    write(path, value.replace(old, new, 1))


# Test routers emulate the current host-owned one-record protocol exactly. They return
# the record itself for submit_one_* calls and implement relation_set as the fixed
# ordered-pair decision that production now uses instead of free relation streaming.
for path in ("tests/test_atomic_design_pipeline.py", "tests/test_content_design_graph_expansion.py"):
    value = read(path)
    old = '''        if tool_name == "submit_one_design_continue_record":\n            target = str(context.get("target_template") or "")\n            return {"required": target == "design/research_fact" and not context.get("accepted_record_ids")}\n        accepted = context["accepted_records"]\n'''
    new = '''        if tool_name == "submit_one_design_continue_record":\n            target = str(context.get("target_template") or "")\n            return {"required": target == "design/research_fact" and not context.get("accepted_record_ids")}\n        if tool_name == "submit_one_design_relation_set":\n            selected = [\n                edge["relation_type"]\n                for edge in self.edges\n                if edge["source_id"] == context["source_id"] and edge["target_id"] == context["target_id"]\n            ]\n            return {"relations": selected[:4], "key_code": 0, "overflow": len(selected) > 4}\n        single_record = tool_name.startswith("submit_one_")\n        normalized_tool = tool_name.replace("submit_one_", "submit_", 1) if single_record else tool_name\n        accepted = context.get("accepted_records", [])\n'''
    if value.count(old) != 1:
        raise RuntimeError(f"{path}: single-record prelude target not found")
    value = value.replace(old, new, 1)
    value = value.replace('if tool_name == "submit_design_content_entity":', 'if normalized_tool == "submit_design_content_entity":', 1)
    value = value.replace('elif tool_name == "submit_design_content_relation":', 'elif normalized_tool == "submit_design_content_relation":', 1)
    value = value.replace('elif tool_name == "submit_design_content_capability":', 'elif normalized_tool == "submit_design_content_capability":', 1)
    value = value.replace('elif tool_name == "submit_design_content_property":', 'elif normalized_tool == "submit_design_content_property":', 1)
    value = value.replace('elif tool_name == "submit_design_decision":', 'elif normalized_tool == "submit_design_decision":', 1)
    marker = '''        if len(accepted) < len(rows):\n            return {\n                "status": "record",\n                "record": deepcopy(rows[len(accepted)]),\n                "reason": "",\n            }\n'''
    replacement = '''        if single_record:\n            if normalized_tool == "submit_design_content_entity":\n                index = int(context.get("entity_ordinal", len(accepted) + 1)) - 1\n            elif normalized_tool == "submit_design_content_property" and context.get("requested_property"):\n                requested = context["requested_property"]\n                matching = [row for row in rows if row.get("property") == requested]\n                if len(matching) != 1:\n                    raise AssertionError(f"missing requested property {requested}")\n                return deepcopy(matching[0])\n            else:\n                index = len(accepted)\n            if not 0 <= index < len(rows):\n                raise AssertionError(f"single record index out of range for {normalized_tool}: {index}")\n            return deepcopy(rows[index])\n        if len(accepted) < len(rows):\n            return {\n                "status": "record",\n                "record": deepcopy(rows[len(accepted)]),\n                "reason": "",\n            }\n'''
    if value.count(marker) != 1:
        raise RuntimeError(f"{path}: record envelope target not found")
    write(path, value.replace(marker, replacement, 1))

# The specialized research fake must follow the same one-record return shape.
path = "tests/test_atomic_design_pipeline.py"
value = read(path)
old = '''            if tool_name == "submit_design_research_fact":\n                context = json.loads(messages[-1]["content"])\n                assert context["source_ref"] == "source_b"\n                if not context["accepted_records"]:\n                    return {\n                        "status": "record",\n                        "record": {"fact": "Material is brittle"},\n                        "reason": "",\n                    }\n                return {"status": "done", "record": None, "reason": ""}\n'''
new = '''            if tool_name in {"submit_design_research_fact", "submit_one_design_research_fact"}:\n                context = json.loads(messages[-1]["content"])\n                assert context["source_ref"] == "source_b"\n                if tool_name == "submit_one_design_research_fact":\n                    return {"fact": "Material is brittle"}\n                if not context["accepted_records"]:\n                    return {\n                        "status": "record",\n                        "record": {"fact": "Material is brittle"},\n                        "reason": "",\n                    }\n                return {"status": "done", "record": None, "reason": ""}\n'''
if value.count(old) != 1:
    raise RuntimeError("research fake target not found")
value = value.replace(old, new, 1)
# Entity cardinality is one host-owned count call followed by exactly one model call
# per entity; assert the current tool names rather than the removed stream protocol.
old_assert = '''    assert router.calls.count("submit_design_content_entity") == 2\n    assert router.calls.count("submit_one_design_content_entity_count") == 1\n'''
new_assert = '''    assert router.calls.count("submit_one_design_content_entity") == 2\n    assert router.calls.count("submit_one_design_content_entity_count") == 1\n'''
if value.count(old_assert) != 1:
    raise RuntimeError("entity call-count assertion target not found")
value = value.replace(old_assert, new_assert, 1)

# RecipeRouter overrides must use the same one-record protocol for its custom process
# entity instead of assuming the old accepted_records stream envelope.
old_recipe = '''                if context.get("entity", {}).get("entity_id") == "conversion":\n                    rows = (\n                        [{"fact_type": "CRAFTING_RECIPE"}]\n                        if tool_name == "submit_design_content_capability"\n                        else [\n                            {"property": "recipe_kind", "value": "shapeless"},\n                            {"property": "count", "value": "1"},\n                        ]\n                    )\n                    index = len(context["accepted_records"])\n                    return (\n                        {"status": "record", "record": rows[index], "reason": ""}\n                        if index < len(rows)\n                        else {"status": "done", "record": None, "reason": ""}\n                    )\n'''
new_recipe = '''                if context.get("entity", {}).get("entity_id") == "conversion":\n                    if tool_name == "submit_one_design_content_capability":\n                        return {"fact_type": "CRAFTING_RECIPE"}\n                    if tool_name == "submit_one_design_content_property":\n                        rows = [\n                            {"property": "recipe_kind", "value": "shapeless"},\n                            {"property": "count", "value": "1"},\n                        ]\n                        requested = context.get("requested_property")\n                        return next(row for row in rows if row["property"] == requested)\n                    rows = (\n                        [{"fact_type": "CRAFTING_RECIPE"}]\n                        if tool_name == "submit_design_content_capability"\n                        else [\n                            {"property": "recipe_kind", "value": "shapeless"},\n                            {"property": "count", "value": "1"},\n                        ]\n                    )\n                    index = len(context.get("accepted_records", []))\n                    return (\n                        {"status": "record", "record": rows[index], "reason": ""}\n                        if index < len(rows)\n                        else {"status": "done", "record": None, "reason": ""}\n                    )\n'''
if value.count(old_recipe) != 1:
    raise RuntimeError("recipe router target not found")
write(path, value.replace(old_recipe, new_recipe, 1))

# Test-only deterministic generation must keep the same PlatformLock object visible to
# the caller; otherwise generated metadata and the proposal lock disagree. Production
# code remains fail-closed because this is isolated to tests/conftest.py.
replace_once(
    "tests/conftest.py",
    '''    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved() or not spec.platform.deterministic_module_kinds:\n            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n''',
    '''    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved() or not spec.platform.deterministic_module_kinds:\n            object.__setattr__(spec, "platform", _platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n''',
)

# Self-dependency structural rejection must not be masked by missing strict semantic
# generation fields in the fixture.
path = "tests/test_deterministic_minecraft_content_contract.py"
value = read(path)
old = '''                        {\n                            "id": "copper_hammer",\n                            "kind": "item",\n                            "depends_on": ["copper_hammer"],\n                        }\n'''
new = '''                        {\n                            "id": "copper_hammer",\n                            "kind": "item",\n                            "config": {"display_name": "Copper Hammer", "main_color": "#B87333"},\n                            "depends_on": ["copper_hammer"],\n                        }\n'''
if value.count(old) != 1:
    raise RuntimeError("self-dependency fixture target not found")
write(path, value.replace(old, new, 1))

print("follow-up repair applied")
