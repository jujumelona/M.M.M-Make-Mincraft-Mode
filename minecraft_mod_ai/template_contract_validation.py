"""Shared load-time and CI checks for the package-owned template catalog."""

from collections.abc import Mapping
from pathlib import Path
import re

from jsonschema import Draft202012Validator
import yaml


PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def placeholders(value):
    if isinstance(value, str):
        return set(PLACEHOLDER.findall(value))
    if isinstance(value, Mapping):
        return set().union(*(placeholders(k) | placeholders(v) for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(placeholders(item) for item in value))
    return set()


def validate_template_contract(template):
    identifier = template.get("id", "<unknown>")
    execution = template.get("execution")
    if execution == "sequence":
        steps = template.get("steps")
        if not isinstance(steps, list) or not steps or any(not isinstance(s, str) for s in steps):
            raise ValueError(f"TEMPLATE_SEQUENCE: {identifier} needs explicit steps")
        if len(steps) != len(set(steps)):
            raise ValueError(f"TEMPLATE_SEQUENCE_DUPLICATE: {identifier}")
    if execution == "value" and "output_schema" not in template:
        raise ValueError(f"TEMPLATE_OUTPUT_SCHEMA: {identifier}")
    schemas = []
    if "record_schema" in template:
        schemas.append(template["record_schema"])
    if execution == "value":
        schemas.append(template["output_schema"])
    for slot in template.get("ai_slots", ()):
        if not isinstance(slot, dict) or "schema" not in slot:
            raise ValueError(f"TEMPLATE_SLOT_SCHEMA: {identifier}")
        if "default" in slot:
            raise ValueError(f"TEMPLATE_SLOT_DEFAULT: {identifier}")
        schemas.append(slot["schema"])
    if schemas:
        from .model_output_atomicity_contract import _assert_closed_object_schemas

        for schema in schemas:
            Draft202012Validator.check_schema(schema)
            _assert_closed_object_schemas(schema)
    if "render" in template and "inputs" in template:
        declared = template["inputs"]
        if not isinstance(declared, dict) or placeholders(template["render"]) != set(declared):
            raise ValueError(f"TEMPLATE_PLACEHOLDERS: {identifier} inputs must exactly match render fields")
    if "render" in template and "requires" in template:
        declared = set(template["requires"])
        declared.update(slot.get("id", slot.get("slot_id")) for slot in template.get("ai_slots", ()))
        missing = placeholders(template["render"]) - declared
        if missing:
            raise ValueError(f"TEMPLATE_PLACEHOLDERS: {identifier} undeclared render fields {sorted(missing)}")


def validate_catalog(root: Path, *, consumer_roots=None):
    """Audit every file and every manifest edge, including unreachable broken leaves."""
    templates = {}
    for path in sorted(root.rglob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        identifier = path.relative_to(root).with_suffix("").as_posix()
        if not isinstance(value, dict) or value.get("id") != identifier:
            raise ValueError(f"TEMPLATE_ID: {identifier}")
        if value["id"] in templates:
            raise ValueError(f"TEMPLATE_ID_DUPLICATE: {identifier}")
        validate_template_contract(value)
        templates[identifier] = value
    if not templates:
        raise ValueError("TEMPLATE_CATALOG_EMPTY")
    active, visited = set(), set()

    def visit(identifier):
        if identifier not in templates:
            raise ValueError(f"TEMPLATE_MISSING: {identifier}")
        if identifier in active:
            raise ValueError(f"TEMPLATE_SEQUENCE_CYCLE: {identifier}")
        if identifier in visited:
            return
        active.add(identifier)
        for child in templates[identifier].get("steps", ()):
            visit(child)
        active.remove(identifier)
        visited.add(identifier)

    for identifier in templates:
        visit(identifier)
    if consumer_roots is not None:
        consumed = set()

        def consume(identifier):
            if identifier not in templates:
                raise ValueError(f"TEMPLATE_CONSUMER_MISSING: {identifier}")
            if identifier in consumed:
                return
            consumed.add(identifier)
            for child in templates[identifier].get("steps", ()):
                consume(child)

        for identifier in consumer_roots:
            consume(identifier)
        orphaned = sorted(identifier for identifier, template in templates.items()
                          if identifier not in consumed and template.get("standalone") is not True)
        if orphaned:
            raise ValueError(f"TEMPLATE_UNCONSUMED: {orphaned}")
    return templates


def runtime_consumer_roots():
    """Bind CI to the host's actual dispatch tables, not directory discovery."""
    from .artifact_expansion import FACT_EXPANSIONS
    from .atomic_design_pipeline import ALL_DESIGN_SLOTS
    from .feature_template_pipeline import FEATURE_DETAIL_STEPS
    from .minecraft_template_catalog import CANONICAL_ARTIFACT_KINDS
    from .research_template_pipeline import RESEARCH_SEQUENCE
    from .reuse_template_pipeline import REUSE_SEQUENCE
    from .stage_template_pipeline import KNOWN_STAGES
    from .task_template_catalog import CRITERION_SECTIONS
    from .translation_runtime import TRANSLATION_SEQUENCE

    roots = set(ALL_DESIGN_SLOTS) | set(RESEARCH_SEQUENCE) | set(REUSE_SEQUENCE) | set(TRANSLATION_SEQUENCE)
    roots.update(identifier for expansion in FACT_EXPANSIONS.values() for identifier in expansion)
    roots.update(f"feature/{step}" for step in FEATURE_DETAIL_STEPS)
    roots.update(f"minecraft/{kind}" for kind in CANONICAL_ARTIFACT_KINDS)
    roots.update(f"criterion/{section}" for section in CRITERION_SECTIONS)
    roots.update(f"{stage}/workflow" for stage in KNOWN_STAGES)
    # Direct call sites in content_design_graph, task_template_runner and the
    # prompt/feature pipelines. New leaves need a consumer or standalone status.
    roots.update({
        "prompt/workflow", "prompt/capture", "minecraft/generation_contract",
        "feature/discover", "feature/decompose", "feature/atomic_check",
        "design/content_capability", "design/content_entity", "design/content_entity_count",
        "design/content_relation", "design/content_property", "design/decision",
        "design/research_fact", "design/continue_record", "design/relation_set",
        "asset/block_tile", "asset/entity_texture", "asset/gui_panel", "asset/item_sprite",
    })
    return frozenset(roots)
