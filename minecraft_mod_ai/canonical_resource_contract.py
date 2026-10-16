"""Manifest-owned data resources bound to the resolved target's real registries."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from zipfile import ZipFile

from packaging.version import Version

from .implementation_template_renderer import render_template
from .integrity_validators import validate_json_schema

_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")


class _UnavailableResourceRegistry(ValueError):
    pass


def resource_candidate_inputs(manifest, fact, *, mod_id, subject,
                              minecraft_version, context_id,
                              source_requirement="", parent_requirement=""):
    """Persist JSON authoring semantics before Java construction can be selected."""
    contract = manifest["resource_contract"]
    minimum = contract.get("min_minecraft")
    if minimum and Version(minecraft_version) < Version(str(minimum)):
        raise ValueError("CANONICAL_RESOURCE_SCHEMA_VERSION_UNSUPPORTED: " + minecraft_version)
    requirement = (source_requirement or str(getattr(fact, "source_clause", "") or "").strip()
                   or str(getattr(fact, "display_name", "") or "").strip()
                   or f"Implement {subject} through {manifest['id']}.")
    authored = getattr(fact, "value", None)
    bindings = {
        "mod_id": mod_id, "registry_path": subject,
        "minecraft_version": minecraft_version,
        "fact_type": fact.fact_type.value if fact is not None else "REQUEST_FEATURE",
        "source_requirement": requirement[:12000],
        "authored_resource_properties_json": json.dumps(authored, ensure_ascii=False, sort_keys=True),
    }
    parent = parent_requirement or str(getattr(fact, "parent_requirement", "") or "").strip()
    if parent:
        bindings["parent_requirement"] = parent
    schema = deepcopy(contract["model_schema"])
    supplied = isinstance(authored, Mapping) and {"type", "generator"} <= set(authored)
    if supplied:
        validate_json_schema(authored, schema)
        bindings["resource_definition_json"] = json.dumps(authored, ensure_ascii=False, sort_keys=True)
    spec = {
        "leaf_id": manifest["id"], "context_id": context_id,
        "requirement": requirement[:12000],
        "target_path": render_template({"render": contract["target_path"]}, bindings),
        "language": "json", "operation": "CREATE_FILE",
        "side": manifest["side"][0].upper(), "bindings": bindings,
        "output_schema": deepcopy(contract["output_schema"]),
        "render_mold": "{{resource_definition}}",
        "slots": [] if supplied else [{
            "name": "resource_definition", "description": contract["description"],
            "schema": schema,
        }],
    }
    _authored_type_overrides(manifest, spec)
    inputs = {}
    for port in manifest["inputs"]:
        if port["type"] == "identifier":
            inputs[port["name"]] = f"{mod_id}:{subject}"
        elif port["type"] == "specification":
            inputs[port["name"]] = spec
        else:
            raise ValueError("CANONICAL_RESOURCE_INPUT_UNSUPPORTED: " + str(port["type"]))
    return inputs


def _registry_names(schema):
    if isinstance(schema, Mapping):
        name = schema.get("x-mmm-resource-registry")
        if name:
            yield name
        for value in schema.values():
            yield from _registry_names(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _registry_names(value)


def prepare_resource_context(manifest, spec, *, java_classpath, project_root):
    """Read datapack JSON only; never run Minecraft, datagen, or a Java compiler."""
    registries = {name: {} for name in _registry_names(manifest["resource_contract"]["model_schema"])}
    expected_version = spec["bindings"]["minecraft_version"]
    if not java_classpath:
        raise ValueError("CANONICAL_RESOURCE_TARGET_CLASSPATH_REQUIRED")

    def wanted(name):
        parts = name.split("/", 2)
        return (len(parts) == 3 and parts[0] == "data" and name.endswith(".json")
                and any(parts[2].startswith(registry + "/") for registry in registries))

    def add(name, raw, source, *, project=False):
        parts = name.split("/")
        if len(parts) < 4 or parts[0] != "data" or not name.endswith(".json"):
            return
        for registry, entries in registries.items():
            prefix = f"data/{parts[1]}/{registry}/"
            if not name.startswith(prefix):
                continue
            identifier = parts[1] + ":" + name[len(prefix):-5]
            if not _RESOURCE_ID.fullmatch(identifier):
                raise ValueError("CANONICAL_RESOURCE_IDENTIFIER_INVALID: " + identifier)
            value = json.loads(raw)
            row = {"value": value, "source_path": source,
                   "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
            prior = entries.get(identifier)
            if prior and prior["value"] != value and not project:
                raise ValueError("CANONICAL_RESOURCE_REGISTRY_AMBIGUOUS: " + identifier)
            entries[identifier] = row

    for raw_path in dict.fromkeys(java_classpath):
        path = Path(raw_path).resolve()
        if path.is_dir():
            for resource in sorted((path / "data").rglob("*.json")):
                name = resource.relative_to(path).as_posix()
                if wanted(name):
                    add(name, resource.read_bytes(), str(resource))
        elif path.suffix == ".jar":
            with ZipFile(path) as archive:
                names = archive.namelist()
                if "version.json" in names:
                    version = json.loads(archive.read("version.json"))
                    if "world_version" in version and version.get("id") != expected_version:
                        raise ValueError("CANONICAL_RESOURCE_TARGET_VERSION_MISMATCH: " + str(path))
                for name in sorted(names):
                    if wanted(name):
                        add(name, archive.read(name), str(path) + "!/" + name)
        elif not path.exists():
            raise ValueError("CANONICAL_RESOURCE_CLASSPATH_PATH_MISSING: " + str(path))
    if project_root is not None:
        root = Path(project_root).resolve()
        for relative in ("src/main/resources", "src/main/generated"):
            folder = root / relative
            for resource in sorted((folder / "data").rglob("*.json")):
                if not resource.resolve().is_relative_to(root):
                    raise ValueError("CANONICAL_RESOURCE_PROJECT_PATH_ESCAPE")
                name = resource.relative_to(folder).as_posix()
                if wanted(name):
                    add(name, resource.read_bytes(), resource.relative_to(root).as_posix(), project=True)
    authored = json.loads(spec["bindings"].get("authored_resource_properties_json", "null"))
    requested_type = authored.get("dimension_type", "") if isinstance(authored, dict) else ""
    if isinstance(requested_type, str) and requested_type:
        available_types = registries.get("dimension_type", {})
        if ":" not in requested_type and "minecraft:" + requested_type in available_types:
            requested_type = "minecraft:" + requested_type
        if _RESOURCE_ID.fullmatch(requested_type) and requested_type not in available_types:
            raise ValueError("CANONICAL_RESOURCE_AUTHORED_TYPE_MISSING: " + requested_type)
        if requested_type not in available_types:
            requested_type = ""
    else:
        requested_type = ""
    return {"registries": registries, "minecraft_version": expected_version,
            "authored_type_ref": requested_type,
            "authored_type_overrides": _authored_type_overrides(manifest, spec)}


def bind_resource_schema(schema, resource_context):
    """Constrain model choices to IDs actually present in the locked target/project."""
    if resource_context is None:
        raise ValueError("CANONICAL_RESOURCE_CONTEXT_REQUIRED")
    result = deepcopy(schema)

    def bind(node):
        if not isinstance(node, dict):
            return
        registry = node.pop("x-mmm-resource-registry", None)
        if registry:
            identifiers = sorted(resource_context["registries"][registry])
            if not identifiers:
                raise _UnavailableResourceRegistry("CANONICAL_RESOURCE_REGISTRY_EMPTY: " + registry)
            if registry == "dimension_type" and resource_context.get("authored_type_ref"):
                identifiers = [resource_context["authored_type_ref"]]
            node["enum"] = identifiers
        for key, child in node.items():
            if isinstance(child, dict):
                bind(child)
            elif key in {"anyOf", "oneOf"} and isinstance(child, list):
                available = []
                unavailable = None
                for item in child:
                    try:
                        bind(item)
                    except _UnavailableResourceRegistry as exc:
                        unavailable = exc
                    else:
                        available.append(item)
                if not available:
                    raise unavailable or _UnavailableResourceRegistry("CANONICAL_RESOURCE_ALTERNATIVES_EMPTY")
                node[key] = available
            elif isinstance(child, list):
                for item in child:
                    bind(item)

    bind(result)
    overrides = resource_context.get('authored_type_overrides', {})
    override_schema = result.get('properties', {}).get('dimension_type_overrides')
    if overrides and isinstance(override_schema, dict):
        for name, value in overrides.items():
            override_schema['properties'][name]['const'] = value
        override_schema['required'] = sorted(overrides)
        result['required'] = list(dict.fromkeys([*result.get('required', ()), 'dimension_type_overrides']))
    return result


def _authored_type_overrides(manifest, spec):
    authored = json.loads(spec["bindings"].get("authored_resource_properties_json", "null"))
    if not isinstance(authored, dict) or "generator" in authored:
        return {}
    overrides = {}
    for name, schema in manifest["resource_contract"].get("authored_type_fields", {}).items():
        if name not in authored:
            continue
        value = authored[name]
        if isinstance(value, str):
            # Legacy design properties were untyped strings, sometimes prose
            # or IDs. Only exact numeric syntax can become a host constraint;
            # the complete original properties remain in the model context.
            decimal = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
            text = value.strip()
            if re.fullmatch(decimal, text):
                value = float(text)
            elif name == 'coordinate_scale' and re.fullmatch(decimal + r'\s*:\s*' + decimal, text):
                left, right = text.split(':')
                try:
                    value = float(Fraction(left.strip()) / Fraction(right.strip()))
                except ZeroDivisionError as exc:
                    raise ValueError("CANONICAL_RESOURCE_AUTHORED_NUMBER_INVALID: " + name) from exc
            else:
                continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("CANONICAL_RESOURCE_AUTHORED_NUMBER_INVALID: " + name)
        validate_json_schema(value, schema)
        overrides[name] = value
    return overrides


def render_resource_artifact(manifest, spec, values, resource_context):
    """Serialize a structured definition, retaining supplied environmental properties."""
    if manifest["resource_contract"]["renderer"] != "dimension":
        raise ValueError("CANONICAL_RESOURCE_RENDERER_UNSUPPORTED")
    definition = deepcopy(values.get("resource_definition"))
    if definition is None:
        definition = json.loads(values["resource_definition_json"])
    validate_json_schema(definition, bind_resource_schema(
        manifest["resource_contract"]["model_schema"], resource_context))
    overrides = definition.pop('dimension_type_overrides', {})
    overrides.update(_authored_type_overrides(manifest, spec))
    if overrides:
        resource = resource_context["registries"]["dimension_type"][definition["type"]]
        dimension_type = deepcopy(resource["value"])
        if not isinstance(dimension_type, dict) or not set(overrides) <= set(dimension_type):
            raise ValueError("CANONICAL_RESOURCE_TARGET_TYPE_FIELDS_UNSUPPORTED")
        dimension_type.update(overrides)
        definition["type"] = dimension_type
    source = json.dumps(definition, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    validate_resource_artifact(manifest, spec, source, resource_context)
    return source


def validate_resource_artifact(manifest, spec, source, resource_context):
    """Recheck resource references on generation, installation and checkpoint reuse."""
    definition = json.loads(source)
    validate_json_schema(definition, spec["output_schema"])
    references = []

    def reference(registry, identifier):
        row = resource_context["registries"][registry].get(identifier)
        if row is None:
            raise ValueError("CANONICAL_RESOURCE_REFERENCE_MISSING: " + registry + ":" + str(identifier))
        references.append({"registry": registry, "id": identifier,
                           "source_path": row["source_path"], "sha256": row["sha256"]})

    dimension_type = definition["type"]
    overrides = _authored_type_overrides(manifest, spec)
    requested_type = resource_context.get("authored_type_ref")
    if isinstance(dimension_type, str):
        if overrides or (requested_type and dimension_type != requested_type):
            raise ValueError("CANONICAL_RESOURCE_AUTHORED_TYPE_MISMATCH")
        reference("dimension_type", dimension_type)
    else:
        matches = []
        fields = manifest['resource_contract'].get('authored_type_fields', {})
        for name, schema in fields.items():
            if name in dimension_type:
                validate_json_schema(dimension_type[name], schema)
        for identifier, row in resource_context["registries"]["dimension_type"].items():
            if requested_type and identifier != requested_type:
                continue
            expected = deepcopy(row["value"])
            if isinstance(expected, dict) and set(overrides) <= set(expected):
                expected.update({name: dimension_type[name] for name in fields
                                 if name in expected and name in dimension_type})
                expected.update(overrides)
                if expected == dimension_type:
                    matches.append(identifier)
        if not matches:
            raise ValueError("CANONICAL_RESOURCE_INLINE_TYPE_NOT_TARGET_BOUND")
        reference("dimension_type", matches[0])
    generator = definition["generator"]
    reference("worldgen/noise_settings", generator["settings"])
    biome_source = generator["biome_source"]
    if biome_source["type"] == "minecraft:fixed":
        reference("worldgen/biome", biome_source["biome"])
    elif biome_source["type"] == "minecraft:multi_noise":
        reference("worldgen/multi_noise_biome_source_parameter_list", biome_source["preset"])
    return {"validator": "resource_references", "status": "PASS",
            "minecraft_version": resource_context["minecraft_version"],
            "references": references}
