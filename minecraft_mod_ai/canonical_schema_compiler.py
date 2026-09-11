"""Compile typed leaf ports from the canonical responsibility manifests."""
from __future__ import annotations

from copy import deepcopy
from jsonschema import Draft202012Validator

from .task_template_catalog import load_template


HASH = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
TEXT = {"type": "string", "minLength": 1}


def canonical_leaves() -> tuple[str, ...]:
    from .structural_routing_contract import CANONICAL_ARTIFACT_KINDS
    from .minecraft_template_steps import responsibility_ids_for_artifact
    leaves = tuple(leaf for kind in CANONICAL_ARTIFACT_KINDS for leaf in responsibility_ids_for_artifact(kind))
    if len(leaves) != len(set(leaves)):
        raise ValueError("DUPLICATE_CANONICAL_LEAF")
    return leaves


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def compile_leaf_schemas(leaf: str) -> tuple[dict, dict]:
    manifest = load_template(leaf)
    if manifest["id"] != leaf:
        raise ValueError("CANONICAL_MANIFEST_ID_MISMATCH")
    specification = object_schema({
        "leaf_id": {"const": leaf}, "context_id": TEXT,
        "requirement": TEXT, "target_path": {"type": "string", "pattern": r"^(?![/\\])(?!.*(?:^|[/\\])\.\.(?:[/\\]|$))(?!.*:).+"},
        "language": {"enum": ["java", "json", "text"]},
        "operation": {"enum": ["CREATE_FILE", "REPLACE_FILE", "JAVA_PATCH", "JSON_OBJECT_MERGE"]},
        "side": {"enum": [s.upper() for s in manifest["side"]]},
        "bindings": {"type": "object", "minProperties": 1, "additionalProperties": TEXT},
        "output_schema": {"type": "object", "minProperties": 1},
        "render_mold": TEXT,
        "slots": {"type": "array", "maxItems": 128, "items": object_schema({
            "name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
            "description": TEXT, "schema": {"type": "object", "minProperties": 1}})},
        "expected_sha256": HASH, "anchor": TEXT,
        "java_filename": {"type": "string", "pattern": r"^[A-Za-z_$][\w$]*\.java$"},
        "java_prefix": {"type": "string"}, "java_suffix": {"type": "string"},
    }, ["leaf_id", "context_id", "requirement", "target_path", "language", "operation", "side", "bindings", "output_schema", "render_mold", "slots"])
    specification["allOf"] = [
        {"if": {"properties": {"operation": {"const": "JAVA_PATCH"}}}, "then": {"required": ["anchor", "java_prefix", "java_suffix", "java_filename"]}},
        {"if": {"properties": {"operation": {"const": "REPLACE_FILE"}}}, "then": {"required": ["expected_sha256"]}},
    ]
    receipt = object_schema({"leaf_id": {"const": leaf}, "context_id": TEXT,
                             "content_sha256": HASH, "contract_sha256": HASH})
    types = {
        "identifier": {"type": "string", "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$"},
        "specification": specification, "receipt": receipt,
        "code_fragment": {"type": "string", "minLength": 1, "maxLength": 262144},
    }
    result = []
    for direction in ("inputs", "outputs"):
        ports = manifest[direction]
        properties, required = {}, []
        for port in ports:
            name = port["name"]
            if name in properties or port["type"] not in types:
                raise ValueError(f"INVALID_CANONICAL_PORT: {leaf}: {port}")
            properties[name] = deepcopy(types[port["type"]])
            if port.get("required", True):
                required.append(name)
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
                  **object_schema(properties, required)}
        Draft202012Validator.check_schema(schema)
        result.append(schema)
    return tuple(result)


def compile_all_schemas() -> dict[str, dict]:
    return {f"{leaf}:{direction}": schema for leaf in canonical_leaves()
            for direction, schema in zip(("input", "output"), compile_leaf_schemas(leaf), strict=True)}
