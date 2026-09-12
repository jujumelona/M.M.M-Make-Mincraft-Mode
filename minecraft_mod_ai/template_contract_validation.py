"""Shared load-time, startup and CI checks for the package-owned template catalog."""

from collections.abc import Mapping
from pathlib import Path
from string import Formatter
import json
import re

from jsonschema import Draft202012Validator
import yaml

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_HOST_WORKFLOWS = (
    "code/workflow",
    "asset/workflow",
    "integration/workflow",
    "validation/workflow",
    "research/workflow",
    "reuse/workflow",
)
_SYSTEM_SOURCE_FIELDS = {
    "persistent_store": {"package_name", "mod_id"},
    "config_loader": {"package_name", "mod_id"},
    "class_skill": {"package_name", "class_name", "resource"},
    "economy": {"package_name", "class_name", "resource"},
    "party_guild": {"package_name", "class_name", "resource"},
    "quest": {"package_name", "class_name", "resource"},
    "gui_networking": {"package_name", "mod_id", "class_name", "resource"},
}


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


def _validate_response_contracts(root: Path) -> None:
    path = root / "response" / "contracts.json"
    if not path.is_file():
        raise ValueError("RESPONSE_TEMPLATE: missing response/contracts.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("RESPONSE_TEMPLATE: contracts.json must contain named schemas")
    for name, schema in value.items():
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise ValueError("RESPONSE_TEMPLATE: invalid named response contract")
        Draft202012Validator.check_schema(schema)


def _validate_system_source_templates(root: Path) -> None:
    source_root = root / "implementation" / "system_pack"
    actual = {path.stem.removesuffix(".java") for path in source_root.glob("*.java.fmt")}
    expected = set(_SYSTEM_SOURCE_FIELDS)
    if actual != expected:
        raise ValueError(
            f"SYSTEM_SOURCE_TEMPLATE: expected {sorted(expected)}, found {sorted(actual)}"
        )
    for name, expected_fields in _SYSTEM_SOURCE_FIELDS.items():
        raw = (source_root / f"{name}.java.fmt").read_text(encoding="utf-8")
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(raw)
            if field_name is not None
        }
        if fields != expected_fields:
            raise ValueError(
                f"SYSTEM_SOURCE_TEMPLATE: {name} expects fields {sorted(expected_fields)}, "
                f"found {sorted(fields)}"
            )


def _validate_host_workflows(templates: Mapping[str, Mapping]) -> None:
    """Host-owned workflow children are deterministic manifests, never hidden prompts."""
    for workflow_id in _HOST_WORKFLOWS:
        workflow = templates.get(workflow_id)
        if workflow is None:
            continue
        if workflow.get("execution") != "sequence":
            raise ValueError(f"HOST_WORKFLOW: {workflow_id} must be a sequence")
        for identifier in workflow.get("steps", ()):
            template = templates.get(identifier)
            if template is None:
                raise ValueError(f"HOST_WORKFLOW_MISSING: {identifier}")
            if template.get("execution") != "host":
                raise ValueError(f"HOST_TEMPLATE_EXECUTION: {identifier} must use execution: host")
            dead_prompt_fields = sorted(key for key in ("task", "rules") if key in template)
            if dead_prompt_fields:
                raise ValueError(
                    f"HOST_TEMPLATE_PROMPT: {identifier} contains dead prompt fields {dead_prompt_fields}"
                )
            if not isinstance(template.get("input"), Mapping):
                raise ValueError(f"HOST_TEMPLATE_INPUT: {identifier} must declare input")
            if not isinstance(template.get("output"), Mapping):
                raise ValueError(f"HOST_TEMPLATE_OUTPUT: {identifier} must declare output")
            proof = template.get("proof")
            if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
                raise ValueError(f"HOST_TEMPLATE_PROOF: {identifier} must declare a proof predicate")


def validate_catalog(root: Path, *, consumer_roots=None):
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
    _validate_host_workflows(templates)
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
        orphaned = sorted(
            identifier for identifier, template in templates.items()
            if identifier not in consumed and template.get("standalone") is not True
        )
        if orphaned:
            raise ValueError(f"TEMPLATE_UNCONSUMED: {orphaned}")
    if (root / "response").exists():
        _validate_response_contracts(root)
    if (root / "implementation" / "system_pack").exists():
        _validate_system_source_templates(root)
    return templates


def runtime_consumer_roots():
    """Bind startup/CI to actual dispatch roots; sequence ownership stays in YAML."""
    from .artifact_expansion import FACT_EXPANSIONS
    from .atomic_design_pipeline import ALL_DESIGN_SLOTS
    from .feature_template_pipeline import FEATURE_DETAIL_STEPS
    from .minecraft_template_catalog import CANONICAL_ARTIFACT_KINDS
    from .populate_version_artifact_rules import HOST_TEMPLATE_CANDIDATES
    from .stage_template_pipeline import KNOWN_STAGES
    from .task_template_catalog import CRITERION_SECTIONS
    from .translation_runtime import TRANSLATION_SEQUENCE

    roots = set(ALL_DESIGN_SLOTS) | set(TRANSLATION_SEQUENCE)
    roots.update(identifier for expansion in FACT_EXPANSIONS.values() for identifier in expansion)
    roots.update(f"feature/{step}" for step in FEATURE_DETAIL_STEPS)
    roots.update(f"minecraft/{kind}" for kind in CANONICAL_ARTIFACT_KINDS)
    roots.update(f"criterion/{section}" for section in CRITERION_SECTIONS)
    roots.update(f"{stage}/workflow" for stage in KNOWN_STAGES)
    roots.update(HOST_TEMPLATE_CANDIDATES)
    roots.update({
        "prompt/workflow", "research/workflow", "reuse/workflow",
        "minecraft/resource/workflow",
        "minecraft/generation_contract", "feature/discover", "feature/decompose",
        "feature/atomic_check", "design/content_capability", "design/content_entity",
        "design/content_entity_count", "design/content_relation",
        "design/content_relation_count", "design/content_property",
        "design/decision", "design/research_fact",
        "asset/item_sprite",
        "asset/visual_spec",
    })
    return frozenset(roots)
