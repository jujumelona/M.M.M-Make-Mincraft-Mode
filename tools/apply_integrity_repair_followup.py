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


# Test routers emulate the current host-owned one-record protocol exactly.  They return
# the record itself for submit_one_* calls rather than the removed record/done envelope.
for path in ("tests/test_atomic_design_pipeline.py", "tests/test_content_design_graph_expansion.py"):
    value = read(path)
    old = '''        if tool_name == "submit_one_design_continue_record":\n            target = str(context.get("target_template") or "")\n            return {"required": target == "design/research_fact" and not context.get("accepted_record_ids")}\n        accepted = context["accepted_records"]\n'''
    new = '''        if tool_name == "submit_one_design_continue_record":\n            target = str(context.get("target_template") or "")\n            return {"required": target == "design/research_fact" and not context.get("accepted_record_ids")}\n        single_record = tool_name.startswith("submit_one_")\n        normalized_tool = tool_name.replace("submit_one_", "submit_", 1) if single_record else tool_name\n        accepted = context["accepted_records"]\n'''
    if value.count(old) != 1:
        raise RuntimeError(f"{path}: single-record prelude target not found")
    value = value.replace(old, new, 1)
    value = value.replace('if tool_name == "submit_design_content_entity":', 'if normalized_tool == "submit_design_content_entity":', 1)
    value = value.replace('elif tool_name == "submit_design_content_relation":', 'elif normalized_tool == "submit_design_content_relation":', 1)
    value = value.replace('elif tool_name == "submit_design_content_capability":', 'elif normalized_tool == "submit_design_content_capability":', 1)
    value = value.replace('elif tool_name == "submit_design_content_property":', 'elif normalized_tool == "submit_design_content_property":', 1)
    value = value.replace('elif tool_name == "submit_design_decision":', 'elif normalized_tool == "submit_design_decision":', 1)
    marker = '''        if len(accepted) < len(rows):\n            return {\n                "status": "record",\n                "record": deepcopy(rows[len(accepted)]),\n                "reason": "",\n            }\n'''
    replacement = '''        if single_record:\n            index = int(context.get("entity_ordinal", len(accepted) + 1)) - 1 if normalized_tool == "submit_design_content_entity" else len(accepted)\n            if not 0 <= index < len(rows):\n                raise AssertionError(f"single record index out of range for {normalized_tool}: {index}")\n            return deepcopy(rows[index])\n        if len(accepted) < len(rows):\n            return {\n                "status": "record",\n                "record": deepcopy(rows[len(accepted)]),\n                "reason": "",\n            }\n'''
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
write(path, value.replace(old, new, 1))

# Test-only deterministic generation must keep the same PlatformLock object visible to
# the caller; otherwise generated metadata and the proposal lock disagree.  Production
# code remains fail-closed because this is isolated to tests/conftest.py.
replace_once(
    "tests/conftest.py",
    '''    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved() or not spec.platform.deterministic_module_kinds:\n            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n''',
    '''    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved() or not spec.platform.deterministic_module_kinds:\n            object.__setattr__(spec, "platform", _platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n''',
)

print("follow-up repair applied")
